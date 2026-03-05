import hydra
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import dill

import gym
import pathlib
import time
import collections
import numpy as np
import torch
import torch.nn as nn
from utils import set_seed, parse_cfg, Logger
from torch.optim.lr_scheduler import CosineAnnealingLR

from cleandiffuser.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
from cleandiffuser.env.async_vector_env import AsyncVectorEnv
from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.dataset.robomimic_dataset import RobomimicImageDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from cleandiffuser.utils.mlp_correlation import batch_mlp_corr

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
    env_meta['env_kwargs']['use_object_obs'] = False
    abs_action = args.abs_action  
    if abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False

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
                RobomimicImageWrapper(
                    env=env,
                    shape_meta=args.shape_meta,
                    init_state=None,
                    render_obs_key=args.render_obs_key
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=10,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=22,
                    thread_type='FRAME',
                    thread_count=1
                ),
                file_path=None,
                steps_per_render=2
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
                RobomimicImageWrapper(
                    env=env,
                    shape_meta=args.shape_meta,
                    init_state=None,
                    render_obs_key=args.render_obs_key
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=10,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=22,
                    thread_type='FRAME',
                    thread_count=1
                ),
                file_path=None,
                steps_per_render=2
            ),
            n_obs_steps=args.obs_steps,
            n_action_steps=args.action_steps,
            max_episode_steps=args.max_episode_steps
        )
    
    env_fns = [env_fn] * args.num_envs
    env_init_fn_dills = list()
    for i in range(args.num_envs):
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
                            "videos", f"{time.time()}_{i}.mp4")
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

        ep_reward = [0.0] * args.num_envs
        obs, t = envs.reset(), 0
        all_actions = []
        all_done = False
        # initialize video stream
        # if args.save_video:
        #     logger.video_init(envs.envs[0], enable=True, video_id=str(i))  # save videos

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
                prior = torch.zeros((args.num_envs, args.action_steps, args.action_dim), device=args.device)
                naction, _ = agent.sample(prior=prior, n_samples=args.num_envs, sample_steps=args.sample_steps, 
                                          solver=solver, condition_cfg=condition, w_cfg=1.0, temperature=args.temperature, use_ema=True)
                
            # unnormalize prediction
            naction = naction.detach().to('cpu').numpy()  # (num_envs, horizon, action_dim)
            # action_pred = naction
            action_pred = dataset.normalizer['action'].unnormalize(naction)
            
            # get action
            start = 0
            # start = args.obs_steps - 1
            end = start + args.action_steps
            action = action_pred[:, start:end, :]
            all_actions.append(action)

            t1 = time.time()
            avg_infer_times.append(t1 - t0)
            
            if args.abs_action:
                action = dataset.undo_transform_action(action)
            obs, reward, done, info = envs.step(action)
            ep_reward += reward
            t += args.action_steps
        
        success = [1.0 if s > 0 else 0.0 for s in ep_reward]
        print(f"[Episode {1+i*(args.num_envs)}-{(i+1)*(args.num_envs)}] reward: {np.around(ep_reward, 2)} success:{success}")
        
        all_actions = np.transpose(all_actions, (1,0,2,3))
        B, D1, D2, C = all_actions.shape  # Get dimensions dynamically
        all_actions = all_actions.reshape(B, D1 * D2, C)
        all_all_actions.append(all_actions)
        
        episode_rewards.extend(ep_reward)
        episode_success.extend(success)
        episode_steps.extend([t] * args.num_envs)
    
    all_all_actions = np.concatenate(all_all_actions, axis=0)
    predictability_ratio.append(batch_mlp_corr(all_all_actions))

    log_dict = {'mean_step': np.nanmean(episode_steps), 
                'mean_reward': np.nanmean(episode_rewards), 
                'mean_success': np.nanmean(episode_success),
                "mlp_corr_pred_actions_full_traj_online_fixed": np.nanmean(predictability_ratio),
                "avg_infer_time": np.nanmean(avg_infer_times),
                }
    print(log_dict, flush=True)
    return log_dict


@hydra.main(config_path="../configs/dp/robomimic_multi_modal/chi_transformer", config_name="square_abs")
def pipeline(args):
    # ---------------- Create Logger ----------------
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)

    # ---------------- Create Environment ----------------
    envs, env_init_fn_dills = make_async_envs(args)
        
    # ---------------- Create Dataset ----------------
    dataset_path = os.path.expanduser(args.dataset_path)
    dataset = RobomimicImageDataset(dataset_path, horizon=args.horizon, shape_meta=args.shape_meta,
                                    n_obs_steps=args.obs_steps, pad_before=args.obs_steps-1,
                                    pad_after=args.action_steps-1, abs_action=args.abs_action)
    print(dataset)
    # assert args.horizon == args.obs_steps + args.action_steps - 1
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     num_workers=8,
    #     shuffle=True,
    #     pin_memory=True,
    #     persistent_workers=True
    # )
    
    if args.nn == "chi_transformer":
        from cleandiffuser.nn_condition import MultiImageObsConditionRobomimic
        from cleandiffuser.nn_diffusion.chitransformer import ChiTransformer
        
        nn_condition = MultiImageObsConditionRobomimic(
            shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop, 
            use_group_norm=args.use_group_norm, use_seq=args.use_seq, keep_horizon_dims=True).to(args.device)
        nn_diffusion = ChiTransformer(
            args.action_dim, 256, args.action_steps, args.obs_steps, d_model=256, nhead=4, num_layers=4,
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
        x_max = torch.ones((1, args.action_steps, args.action_dim), device=args.device) * +1.0
        x_min = torch.ones((1, args.action_steps, args.action_dim), device=args.device) * -1.0
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

    ckpt = 100000
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
    # save results
    result_path = os.path.join(args.work_dir, f"result_{ckpt}.npy")
    np.save(result_path, result)


if __name__ == "__main__":
    pipeline()
