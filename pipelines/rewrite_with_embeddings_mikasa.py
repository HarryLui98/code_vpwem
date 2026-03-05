import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import hydra
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import gym
import pathlib
import time
import collections
import pickle
import numpy as np
import torch
import torch.nn as nn
from utils import set_seed, parse_cfg, Logger
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict

# from cleandiffuser.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
# from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
# from cleandiffuser.env.async_vector_env import AsyncVectorEnv
# from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.dataset.robomimic_dataset import RobomimicImageDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from cleandiffuser.dataset.dataset_utils import MinMaxNormalizer, ImageNormalizer
from cleandiffuser.dataset.mikasa_dataset import MikasaDataset
from cleandiffuser.utils import GaussianNormalizer, dict_apply


@hydra.main(config_path="../configs/dp/mikasa/chi_transformer_ptp", config_name="shell_game_touch")
def main(args):
    # ---------------- Create Logger ----------------
    set_seed(args.seed)
    # logger = Logger(pathlib.Path(args.work_dir), args)
    assert args.horizon == args.obs_steps + args.action_steps - 1
        
    # ---------------- Create Dataset ----------------
    dataset_path = os.path.expanduser(args.dataset_path)
    # dataset = MikasaDataset(dataset_path, horizon=args.horizon, shape_meta=args.shape_meta,
    #                                 n_obs_steps=args.obs_steps, pad_before=args.obs_steps-1,
    #                                 pad_after=args.action_steps-1)
    # print(dataset)
    normalizer_path = os.path.join(args.dataset_path, "normalizer.pkl")
    # with open(normalizer_path, 'wb') as f:
    #     pickle.dump(normalizer, f)
    with open(normalizer_path, 'rb') as f:
        normalizer_loaded = pickle.load(f)
    
    if args.nn == "chi_transformer":
        from cleandiffuser.nn_condition.multi_image_condition import MultiImageObsConditionRobomimic
        from cleandiffuser.nn_diffusion.chitransformerptp import ChiTransformerPTP
        
        nn_condition = MultiImageObsConditionRobomimic(
            shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop, 
            use_group_norm=args.use_group_norm, use_seq=args.use_seq, keep_horizon_dims=True).to(args.device)
        nn_diffusion = ChiTransformerPTP(
            args.action_dim, 256, args.horizon, args.obs_steps, d_model=256, nhead=4, num_layers=4,
            timestep_emb_type="positional").to(args.device)

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
    
    last_saved_model_path = "PATH_TO_CKPT_MODEL"
    if os.path.exists(last_saved_model_path):
        agent.load(last_saved_model_path)
    
    # _convert_h5_to_embeddings(dataset, agent.model_ema.condition, args.shape_meta.obs, dataset.normalizer, batch_size=args.batch_size, device="cuda:0")
    # print("Done converting!")
    for i in tqdm(range(1000)):
        demo = np.load(f'{dataset_path}/train_data_{i}.npz')
        rgb = demo['rgb'][:,:,:,:3]
        rgb_wrist = demo['rgb'][:,:,:,3:]
        joints = demo['joints']
        sample = {
            'rgb': rgb,
            'rgb_wrist': rgb_wrist,
            'joints': joints,
        }
        obs_dict = dict()
        for key in ['rgb', 'rgb_wrist']:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = sample[key].astype(np.float32).transpose(0, 3, 1, 2)[None, :] / 255.
            # T,C,H,W
            obs_dict[key] = normalizer_loaded['obs'][key].normalize(obs_dict[key])

        for key in ['joints']:
            obs_dict[key] = sample[key].astype(np.float32)[None, :]
            obs_dict[key] = normalizer_loaded['obs'][key].normalize(obs_dict[key])
        
        nobs = dict_apply(obs_dict, torch.tensor)
        condition = {}
        for k in nobs.keys():
            condition[k] = nobs[k].to(args.device)
        with torch.no_grad():
            embeddings = agent.model_ema.condition(condition).cpu().numpy()
        np.savez(f'{dataset_path}/dp_emb_{i}.npz', dp_emb=embeddings)
    

if __name__ == "__main__":
    main()
