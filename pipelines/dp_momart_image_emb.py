import hydra
import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ['HF_ENDPOINT']="https://hf-mirror.com"
# os.environ['MESA_GL_VERSION_OVERRIDE']='4.5'
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


@hydra.main(config_path="../configs/dp/momart/chi_transformer", config_name="momart_emb")
def pipeline(args):
    # ---------------- Create Logger ----------------
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)
    assert args.horizon == args.obs_steps + args.action_steps - 1

    modality_mapping = collections.defaultdict(list)
    for key, attr in args.shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
        
    # ---------------- Create Dataset ----------------

    dataset_config_dict = {
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
    dataset_config = DictToObj(dataset_config_dict)
    
    dataset, _ = TrainUtils.load_data_for_training(dataset_config, obs_keys=args.shape_meta["obs"])
    train_sampler = dataset.get_dataset_sampler()
    normalizer = defaultdict(dict)
    # rgb_normalizer = ImageNormalizer()
    # depth_normalizer = ImageNormalizer()
    # proprio_max, proprio_min = -np.inf*np.ones(args.shape_meta["obs"]["proprio"].shape[0]), np.inf*np.ones(args.shape_meta["obs"]["proprio"].shape[0])
    # proprio_nav_max, proprio_nav_min = -np.inf*np.ones(args.shape_meta["obs"]["proprio_nav"].shape[0]), np.inf*np.ones(args.shape_meta["obs"]["proprio_nav"].shape[0])
    # scan_max, scan_min = -np.inf, np.inf
    action_max, action_min = -np.inf*np.ones(args.shape_meta["action"].shape[0]), np.inf*np.ones(args.shape_meta["action"].shape[0])
    for demo_idx, demo in dataset.hdf5_cache.items():
        # proprio_max = np.maximum(np.max(demo['obs']['proprio'], axis=0), proprio_max)
        # proprio_min = np.minimum(np.min(demo['obs']['proprio'], axis=0), proprio_min)
        # proprio_nav_max = np.maximum(np.max(demo['obs']['proprio_nav'], axis=0), proprio_nav_max)
        # proprio_nav_min = np.minimum(np.min(demo['obs']['proprio_nav'], axis=0), proprio_nav_min)
        action_max = np.maximum(np.max(demo['actions'], axis=0), action_max)
        action_min = np.minimum(np.min(demo['actions'], axis=0), action_min)
    # proprio_normalizer = MinMaxNormalizer(np.array([proprio_min, proprio_max]))
    # proprio_nav_normalizer = MinMaxNormalizer(np.array([proprio_nav_min, proprio_nav_max]))
    # scan_normalizer = ImageNormalizer()
    action_normalizer = MinMaxNormalizer(np.array([action_min, action_max]))
    # normalizer['obs']['rgb'] = rgb_normalizer
    # normalizer['obs']['rgb_wrist'] = rgb_normalizer
    # normalizer['obs']['depth'] = depth_normalizer
    # normalizer['obs']['depth_wrist'] = depth_normalizer
    # normalizer['obs']['proprio'] = proprio_normalizer
    # normalizer['obs']['proprio_nav'] = proprio_nav_normalizer
    # normalizer['obs']['scan'] = scan_normalizer
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
        shuffle=True,
        num_workers=8,
        drop_last=True
    )

    if args.nn == "chi_transformer":
        from cleandiffuser.nn_condition.multi_image_condition import MultiImageObsConditionMomart
        from cleandiffuser.nn_diffusion.chitransformer import ChiTransformer
        
        # nn_condition = MultiImageObsConditionMomart(
        #     shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
        #     crop_shape=args.crop_shape, random_crop=args.random_crop, 
        #     use_group_norm=args.use_group_norm, use_seq=args.use_seq, keep_horizon_dims=True).to(args.device)
        nn_diffusion = ChiTransformer(
            args.action_dim, 256, args.action_steps, args.obs_steps, d_model=256, nhead=4, num_layers=4,
            timestep_emb_type="positional").to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}")
    
    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================")
    # print(f"======================= Parameter Report of Condition Model =======================")
    # report_parameters(nn_condition)
    # print(f"===================================================================================")

    if args.diffusion == "ddpm":
        from cleandiffuser.diffusion.ddpm import DDPM
        x_max = torch.ones((1, args.action_steps, args.action_dim), device=args.device) * +1.0
        x_min = torch.ones((1, args.action_steps, args.action_dim), device=args.device) * -1.0
        agent = DDPM(
            nn_diffusion=nn_diffusion, nn_condition=None, device=args.device,
            diffusion_steps=args.sample_steps, x_max=x_max, x_min=x_min,
            optim_params={"lr": args.lr})
    # elif args.diffusion == "edm":
    #     from cleandiffuser.diffusion.edm import EDM
    #     agent = EDM(nn_diffusion=nn_diffusion, nn_condition=None, device=args.device,
    #                 optim_params={"lr": args.lr})
    # else:
    #     raise NotImplementedError
    
    print("No previous model found, starting from scratch.")
    n_gradient_step = 0
    lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps, last_epoch=-1)

    if args.mode == "train":
        # ----------------- Training ----------------------
        diffusion_loss_list = []
        start_time = time.time()
        for batch in loop_dataloader(dataloader):
            # get condition
            nobs = batch['obs']
            # condition = {}
            # for k in nobs.keys():
            #     con = nobs[k][:, :args.obs_steps, :].numpy().astype(np.float32)#.to(args.device).to(torch.float32)  # (B, To, H, W, C)
            #     # if k.startswith('rgb'):
            #     #     con /= 255.
            #     #     con = np.transpose(con, (0, 1, 4, 2, 3))  # (B, T, C, H, W)
            #     # elif k.startswith('depth'):
            #     #     con = np.transpose(con, (0, 1, 4, 2, 3))  # (B, T, C, H, W)
            #     con = dataset.normalizer['obs'][k].normalize(con)
            #     con = torch.tensor(con, device=args.device, dtype=torch.float32)  # (B, T, C, H, W)
            #     condition[k] = con
            condition = nobs['dp_emb'][:, :args.obs_steps, :].to(args.device)
            naction = batch['actions'][:, -args.action_steps:, :].numpy()#.to(args.device)  # (B, Ta, D)
            naction = torch.tensor(dataset.normalizer['action'].normalize(naction), device=args.device, dtype=torch.float32)

            # update diffusion
            diffusion_loss = agent.update(naction, condition)['loss']
            lr_scheduler.step()
            diffusion_loss_list.append(diffusion_loss)

            if n_gradient_step % args.log_freq == 0:
                metrics = {
                    'step': n_gradient_step,
                    'total_time': time.time() - start_time,
                    'avg_diffusion_loss': np.mean(diffusion_loss_list)
                }
                logger.log(metrics, category='train')
                diffusion_loss_list = []
            
            if n_gradient_step % args.save_freq == 0 and n_gradient_step > 0:
                logger.save_agent(agent=agent, identifier=n_gradient_step)
                torch.save(lr_scheduler.state_dict(), args.work_dir + f'/models/scheduler_{str(n_gradient_step)}.pt')
            
            n_gradient_step += 1
            if n_gradient_step > args.gradient_steps:
                # finish
                logger.finish(agent)
                break
        
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    pipeline()









    

