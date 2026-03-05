import hydra
import os
import sys
import warnings
warnings.filterwarnings('ignore')


os.environ['MESA_GL_VERSION_OVERRIDE']='4.5'
os.environ['MESA_GLSL_VERSION_OVERRIDE']='450'
os.environ['MESA_GLES_VERSION_OVERRIDE']='4.5'

import gym
import pathlib
import time
import collections
import numpy as np
import torch
import torch.nn as nn
import pickle
from utils import set_seed, parse_cfg, Logger
from torch.optim.lr_scheduler import CosineAnnealingLR
import multiprocessing
from collections import defaultdict
import dill

from cleandiffuser.env.robomimic.momart_image_wrapper import MomartImageWrapper
from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
from cleandiffuser.env.async_vector_env import AsyncVectorEnv
from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils.utils import report_parameters, dict_apply 
from cleandiffuser.utils.mlp_correlation import batch_mlp_corr
from cleandiffuser.dataset.dataset_utils import MinMaxNormalizer, ImageNormalizer
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

class DictToObj: 
    def __init__(self, dict_): 
        for k, v in dict_.items(): 
            if isinstance(v, dict): 
                v = DictToObj(v) 
            setattr(self, k, v)

def make_async_envs(args):
    
    print(f"Starting to create {args.num_envs} asynchronous Robomimic environments...")

    def create_robomimic_env(env_meta, shape_meta, enable_render=True):
        modality_mapping = collections.defaultdict(list)
        for key, attr in shape_meta['obs'].items():
            modality_mapping[attr.get('type', 'low_dim')].append(key)
        ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False, 
            render_offscreen=enable_render,
            use_image_obs=enable_render, 
        )
        return env
    
    dataset_path = os.path.expanduser(args.dataset_path)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    # disable object state observation
    # env_meta['env_kwargs']['use_object_obs'] = False
    # abs_action = args.abs_action  
    # if abs_action:
    #     env_meta['env_kwargs']['controller_configs']['control_delta'] = False

    def env_fn():
        env = create_robomimic_env(
            env_meta=env_meta, 
            shape_meta=args.shape_meta
        )
        # Robosuite's hard reset causes excessive memory consumption.
        # Disabled to run more envs.
        # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
        env.env.hard_reset = False
        return MultiStepWrapper(
            VideoRecordingWrapper(
                MomartImageWrapper(
                    env=env,
                    shape_meta=args.shape_meta,
                    init_state=None,
                    render_obs_key=args.render_obs_key
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=20,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=22,
                    thread_type='FRAME',
                    thread_count=1
                ),
                file_path=None,
                steps_per_render=5
            ),
            n_obs_steps=args.obs_steps,
            n_action_steps=args.action_steps,
            max_episode_steps=args.max_episode_steps
        )
    
    # See https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env_runner/robomimic_image_runner.py
    # For each process the OpenGL context can only be initialized once
    # Since AsyncVectorEnv uses fork to create worker process,
    # a separate env_fn that does not create OpenGL context (enable_render=False)
    # is needed to initialize spaces.
    def dummy_env_fn():
        env = create_robomimic_env(
                env_meta=env_meta, 
                shape_meta=args.shape_meta,
                enable_render=False
            )
        return MultiStepWrapper(
            VideoRecordingWrapper(
                MomartImageWrapper(
                    env=env,
                    shape_meta=args.shape_meta,
                    init_state=None,
                    render_obs_key=args.render_obs_key
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=20,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=22,
                    thread_type='FRAME',
                    thread_count=1
                ),
                file_path=None,
                steps_per_render=5
            ),
            n_obs_steps=args.obs_steps,
            n_action_steps=args.action_steps,
            max_episode_steps=args.max_episode_steps
        )
    
    env_fns = [env_fn] * args.num_envs
    env_init_fn_dills = list()
    for j in range(args.eval_episodes):
        enable_render = True
        def init_fn(env, enable_render=enable_render):
            # setup rendering
            # video_wrapper
            import time
            import pathlib
            from cleandiffuser.env.wrapper import VideoRecordingWrapper
            assert isinstance(env.env, VideoRecordingWrapper)
            env.env.video_recoder.stop()
            env.env.file_path = None
            if enable_render:
                filename = pathlib.Path(args.work_dir).joinpath(
                            "videos", f"{time.time()}_{j}.mp4")
                filename.parent.mkdir(parents=False, exist_ok=True)
                env.env.file_path = filename
        env_init_fn_dills.append(dill.dumps(init_fn))
    # env_fn() and dummy_env_fn() should be function!
    envs = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
    envs.seed(args.seed)
    return envs, env_init_fn_dills


def inference(args, envs, dataset, agent, logger, env_init_fn_dills):
    """Evaluate a trained agent and optionally save a video."""
    # ---------------- Start Rollout ----------------
    episode_rewards = []
    episode_steps = []
    episode_success = []
    predictability_ratio = []
    avg_infer_times = []
    all_all_actions = []
    
    if args.diffusion == "ddpm":
        solver = None
    elif args.diffusion == "ddim":
        solver = "ddim"
    elif args.diffusion == "dpm":
        solver = "ode_dpmpp_2"
    elif args.diffusion == "edm":
        solver = "euler"
        
    for i in range(args.eval_episodes // args.num_envs): 
        if args.save_video:
            this_global_slice = slice(i * args.num_envs, (i + 1) * args.num_envs)
            this_init_fns = env_init_fn_dills[this_global_slice]
            n_diff = args.num_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == args.num_envs
            envs.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])
        # initialize video stream
        # if args.save_video:
        #     logger.video_init(envs.env, enable=True, video_id=str(i))  # save videos
        ep_reward = [0.0] * args.num_envs
        obs, t = envs.reset(), 0
        all_actions = []
        all_done = False
        success = [False] * args.num_envs

        while t < args.max_episode_steps:
            
            t0 = time.time()
            obs_dict = {}
            for k in obs.keys():
                obs_seq = obs[k].astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                nobs = dataset.normalizer['obs'][k].normalize(obs_seq)
                # nobs = obs_seq
                obs_dict[k] = nobs = torch.tensor(nobs, device=args.device, dtype=torch.float32)  # (num_envs, obs_steps, obs_dim)
            with torch.no_grad():
                condition = obs_dict
                # run sampling (num_envs, horizon, action_dim)
                prior = torch.zeros((args.num_envs, args.horizon, args.action_dim), device=args.device)
                naction, _ = agent.sample(prior=prior, n_samples=args.num_envs, sample_steps=args.sample_steps, solver=solver, condition_cfg=condition, w_cfg=1.0, temperature=args.temperature, use_ema=True)
            
            # unnormalize prediction
            naction = naction.detach().to('cpu').numpy()  # (num_envs, horizon, action_dim)
            # action_pred = naction
            action_pred = dataset.normalizer['action'].unnormalize(naction)  
            
            # get action
            # start = 0
            start = args.obs_steps - 1
            end = start + args.action_steps
            action = action_pred[:, start:end, :]
            all_actions.append(action)

            t1 = time.time()
            avg_infer_times.append(t1 - t0)
            
            # if args.abs_action:
            #     action = dataset.undo_transform_action(action)
            obs, reward, done, info = envs.step(action)
            done = [True if j['done'][-1] else False for j in info]
            all_done = all(done)
            success = [success[j] or info[j]['sc'][-1]['task'] for j in range(args.num_envs)]
            
            # print(info)
            ep_reward += reward
            t += args.action_steps
        
        if args.save_video:
            _ = envs.render()
        
        print(f"[Episode {1+i*(args.num_envs)}-{(i+1)*(args.num_envs)}] reward: {np.around(ep_reward, 2)} success:{success}")
        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)

        all_actions = np.transpose(all_actions, (1,0,2,3))
        B, D1, D2, C = all_actions.shape  # Get dimensions dynamically
        all_actions = all_actions.reshape(B, D1 * D2, C)
        all_all_actions.append(all_actions)
    
    all_all_actions = np.concatenate(all_all_actions, axis=0)
    predictability_ratio.append(batch_mlp_corr(all_all_actions))

    log_dict = {'mean_step': np.nanmean(episode_steps), 
                'mean_reward': np.nanmean(episode_rewards), 
                'mean_success': np.nanmean(episode_success),
                "mlp_corr_pred_actions_full_traj_online_fixed": np.nanmean(predictability_ratio),
                "avg_infer_time": np.mean(avg_infer_times),
                }
    print(log_dict, flush=True)
    return log_dict


@hydra.main(config_path="../configs/dp/momart/chi_transformer_ptp", config_name="momart")
def pipeline(args):
    # ---------------- Create Logger ----------------
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)

    # ---------------- Create Environment ----------------
    if multiprocessing.get_start_method(allow_none=True) != "spawn":  
        multiprocessing.set_start_method("spawn", force=True)
    
    envs, env_init_fn_dills = make_async_envs(args)
    # modality_mapping = collections.defaultdict(list)
    # for key, attr in args.shape_meta['obs'].items():
    #     modality_mapping[attr.get('type', 'low_dim')].append(key)
    # ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
        
    # ---------------- Create Dataset ----------------
    # dataset_path = os.path.expanduser(args.dataset_path)
    # dataset = MomartImageDataset(dataset_path, horizon=args.horizon, shape_meta=args.shape_meta,
    #                                 n_obs_steps=args.obs_steps, pad_before=args.obs_steps-1,
    #                                 pad_after=args.action_steps-1, abs_action=args.abs_action)
    # print(dataset)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     num_workers=8,
    #     shuffle=True,
    #     pin_memory=True,
    #     persistent_workers=True
    # )

    config_dict = {
        "train": {
            "data": args.dataset_path,
            "num_data_workers": 8,
            "hdf5_cache_mode": "low_dim",
            "hdf5_use_swmr": True,
            "hdf5_load_next_obs": False,
            "hdf5_normalize_obs": False,
            "hdf5_filter_key": "train",
            "hdf5_validation_filter_key": "valid",
            "seq_length": args.horizon,
            "pad_seq_length": True,
            "frame_stack": 1,
            "pad_frame_stack": True,
            "dataset_keys": [
                "actions",
                "rewards",
                "dones"
            ],
            "goal_mode": None,
            "cuda": True,
            "batch_size": args.batch_size,
        },
        "experiment": {
            "validate": False
        }
    }
    config = DictToObj(config_dict)
    
    dataset, _ = TrainUtils.load_data_for_training(config, obs_keys=args.shape_meta["obs"])
    train_sampler = dataset.get_dataset_sampler()
    normalizer = defaultdict(dict)
    rgb_normalizer = ImageNormalizer()
    depth_normalizer = ImageNormalizer()
    proprio_max, proprio_min = -np.inf*np.ones(args.shape_meta["obs"]["proprio"].shape[0]), np.inf*np.ones(args.shape_meta["obs"]["proprio"].shape[0])
    # proprio_nav_max, proprio_nav_min = -np.inf*np.ones(args.shape_meta["obs"]["proprio_nav"].shape[0]), np.inf*np.ones(args.shape_meta["obs"]["proprio_nav"].shape[0])
    # scan_max, scan_min = -np.inf, np.inf
    action_max, action_min = -np.inf*np.ones(args.shape_meta["action"].shape[0]), np.inf*np.ones(args.shape_meta["action"].shape[0])
    for demo_idx, demo in dataset.hdf5_cache.items():
        proprio_max = np.maximum(np.max(demo['obs']['proprio'], axis=0), proprio_max)
        proprio_min = np.minimum(np.min(demo['obs']['proprio'], axis=0), proprio_min)
        # proprio_nav_max = np.maximum(np.max(demo['obs']['proprio_nav'], axis=0), proprio_nav_max)
        # proprio_nav_min = np.minimum(np.min(demo['obs']['proprio_nav'], axis=0), proprio_nav_min)
        action_max = np.maximum(np.max(demo['actions'], axis=0), action_max)
        action_min = np.minimum(np.min(demo['actions'], axis=0), action_min)
    proprio_normalizer = MinMaxNormalizer(np.array([proprio_min, proprio_max]))
    # proprio_nav_normalizer = MinMaxNormalizer(np.array([proprio_nav_min, proprio_nav_max]))
    scan_normalizer = ImageNormalizer()
    action_normalizer = MinMaxNormalizer(np.array([action_min, action_max]))
    normalizer['obs']['rgb'] = rgb_normalizer
    normalizer['obs']['rgb_wrist'] = rgb_normalizer
    normalizer['obs']['depth'] = depth_normalizer
    normalizer['obs']['depth_wrist'] = depth_normalizer
    normalizer['obs']['proprio'] = proprio_normalizer
    # normalizer['obs']['proprio_nav'] = proprio_nav_normalizer
    normalizer['obs']['scan'] = scan_normalizer
    normalizer['action'] = action_normalizer
    setattr(dataset, 'normalizer', normalizer)
    print("\n============= Training Dataset =============")
    print(dataset)
    print("")

    # initialize data loaders
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=True,
    )
    
    # --------------- Create Diffusion Model -----------------
    # if args.nn == "dit":
    #     from cleandiffuser.nn_condition import MultiImageObsCondition
    #     from cleandiffuser.nn_diffusion import DiT1d
        
    #     nn_condition = MultiImageObsCondition(
    #         shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
    #         crop_shape=args.crop_shape, random_crop=args.random_crop, 
    #         use_group_norm=args.use_group_norm, use_seq=args.use_seq).to(args.device)
    #     nn_diffusion = DiT1d(
    #         args.action_dim, emb_dim=256*args.obs_steps, d_model=320, n_heads=10, depth=2, timestep_emb_type="fourier").to(args.device)

    # elif args.nn == "chi_unet":
    #     from cleandiffuser.nn_condition import MultiImageObsCondition
    #     from cleandiffuser.nn_diffusion import ChiUNet1d

    #     nn_condition = MultiImageObsCondition(
    #         shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
    #         crop_shape=args.crop_shape, random_crop=args.random_crop, 
    #         use_group_norm=args.use_group_norm, use_seq=args.use_seq).to(args.device)
    #     nn_diffusion = ChiUNet1d(
    #         args.action_dim, 256, args.obs_steps, model_dim=256, emb_dim=256, dim_mult=[1, 2, 2],
    #         obs_as_global_cond=True, timestep_emb_type="positional").to(args.device)
        
    if args.nn == "chi_transformer":
        from cleandiffuser.nn_condition.multi_image_condition import MultiImageObsConditionMomart
        from cleandiffuser.nn_diffusion.chitransformerptp import ChiTransformerPTP
        
        nn_condition = MultiImageObsConditionMomart(
            shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop, 
            use_group_norm=args.use_group_norm, use_seq=args.use_seq, keep_horizon_dims=True).to(args.device)
        nn_diffusion = ChiTransformerPTP(
            args.action_dim, 256, args.horizon, args.obs_steps, d_model=256, nhead=4, num_layers=4,
            timestep_emb_type="positional").to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}")
    
    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================")
    print(f"======================= Parameter Report of Condition Model =======================")
    report_parameters(nn_condition)
    print(f"===================================================================================")

    if args.diffusion == "ddpm":
        from cleandiffuser.diffusion.ddpm import DDPM
        x_max = torch.ones((1, args.horizon, args.action_dim), device=args.device) * +1.0
        x_min = torch.ones((1, args.horizon, args.action_dim), device=args.device) * -1.0
        agent = DDPM(
            nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=args.device,
            diffusion_steps=args.sample_steps, x_max=x_max, x_min=x_min,
            optim_params={"lr": args.lr})
    elif args.diffusion == "edm":
        from cleandiffuser.diffusion.edm import EDM
        agent = EDM(nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=args.device,
                    optim_params={"lr": args.lr})
    else:
        raise NotImplementedError
    
   
    condition_model_path = f"PATH_TO_SHORT_CONTEXT_POLICY_CKPT_MODEL"
    condition_ckpt = torch.load(condition_model_path)
    keys_to_remove = [k for k in condition_ckpt['model'].keys() if k.startswith("diffusion")]
    for k in keys_to_remove:
        del condition_ckpt['model'][k]
        del condition_ckpt['model_ema'][k]
    msg = agent.model.load_state_dict(condition_ckpt["model"], strict=False)
    print(f"Condition Model loaded with message: {msg}")
    msg = agent.model_ema.load_state_dict(condition_ckpt["model_ema"], strict=False)
    print(f"Condition Model EMA loaded with message: {msg}")

    ckpt = 600000
    print(ckpt)
    diffusion_model_path = os.path.join(args.work_dir, f"models/model_{ckpt}.pt")
    diffusion_ckpt = torch.load(diffusion_model_path)
    msg = agent.model.load_state_dict(diffusion_ckpt["model"], strict=False)
    print(f"Diffusion Model loaded with message: {msg}")
    msg = agent.model_ema.load_state_dict(diffusion_ckpt["model_ema"], strict=False)
    print(f"Diffusion Model EMA loaded with message: {msg}")
    
    agent.model.eval()
    agent.model_ema.eval()

    result = inference(args, envs, dataset, agent, logger, env_init_fn_dills)
    result_path = os.path.join(args.work_dir, f"result_{ckpt}.npy")
    np.save(result_path, result)


if __name__ == "__main__":
    pipeline()









    

