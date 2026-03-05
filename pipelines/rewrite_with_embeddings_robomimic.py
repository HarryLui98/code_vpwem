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
import numpy as np
import torch
import torch.nn as nn
from utils import set_seed, parse_cfg, Logger
from torch.optim.lr_scheduler import CosineAnnealingLR

# from cleandiffuser.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
# from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
# from cleandiffuser.env.async_vector_env import AsyncVectorEnv
# from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.dataset.robomimic_dataset import RobomimicImageDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters
# import robomimic.utils.train_utils as TrainUtils
# import robomimic.utils.file_utils as FileUtils
# import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

def _convert_h5_to_embeddings(dataset_path, policy, shape_dict, normalizer, embedding_key='dp_emb', batch_size=32, device='cpu'):
    # Open the HDF5 dataset
    with h5py.File(dataset_path, 'r+') as h5_file:
        demos = list(h5_file['data'].keys())
        demo_keys = [int(key.split('_')[1]) for key in demos]

        # ** Delete existing embeddings if present (using safe deletion) **
        for demo_idx in demo_keys:
            demo_group = h5_file[f'data/demo_{demo_idx}']
            if embedding_key in demo_group['obs']:
                try:
                    print(f"Deleting existing embeddings in demo_{demo_idx}")
                    demo_group['obs'].pop(embedding_key)  # Safe deletion
                except KeyError as e:
                    print(f"Failed to delete existing embedding in demo_{demo_idx}: {e}")
                except Exception as e:
                    print(f"Unexpected error while deleting embedding in demo_{demo_idx}: {e}")
                    
                    
        # Prepare PyTorch dataset and dataloader
        dataset = HDF5Dataset(h5_file, demo_keys, shape_dict)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        obs_keys = list(shape_dict.keys())
        policy.eval()  # Set the policy to evaluation mode

        # Process each batch and save embeddings
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating embeddings")):
                # Move batch data to the device
                # breakpoint()
                # [print(f"{key}: {batch[key].shape}") for key in obs_keys]

                # Generate embeddings
                obs_dict = {}
                for key in obs_keys:
                    obs_seq = batch[key].cpu().numpy().astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                    nobs = normalizer['obs'][key].normalize(obs_seq)
                    obs_dict[key] = nobs = torch.tensor(nobs, device=device, dtype=torch.float32)  # (num_envs, obs_steps, obs_dim)

                embeddings = policy(obs_dict).cpu().numpy().squeeze()

                # Save embeddings back into the HDF5 file
                batch_start = batch_idx * batch_size
                batch_end = batch_start + len(batch[obs_keys[0]])

                for i, (demo_idx, timestep) in enumerate(dataset.indices[batch_start:batch_end]):
                    # breakpoint()
                    demo_group = h5_file[f'data/demo_{demo_idx}']
                    if embedding_key not in demo_group['obs']:
                        shape = (demo_group['obs'][obs_keys[0]].shape[0],) + embeddings.shape[1:]
                        print(f"demo shape: {shape}")
                        demo_group['obs'].create_dataset(embedding_key, shape=shape, dtype=embeddings.dtype)
                    demo_group['obs'][embedding_key][timestep] = embeddings[i]
                    
    print("Embeddings generated and saved successfully!")
    return

class HDF5Dataset(Dataset):
    def __init__(self, h5_file, demo_keys, obs_dict):
        self.h5_file = h5_file
        self.demo_keys = demo_keys
        self.obs_dict = obs_dict
        self.obs_keys = list(obs_dict.keys())
        self.indices = self._create_indices()

    def _create_indices(self):
        indices = []
        for demo_idx in self.demo_keys:
            demo = self.h5_file[f'data/demo_{demo_idx}']
            n_timesteps = demo['obs'][self.obs_keys[0]].shape[0]
            indices.extend([(demo_idx, t) for t in range(n_timesteps)])
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        demo_idx, timestep = self.indices[idx]
        demo = self.h5_file[f'data/demo_{demo_idx}']
        obs = {key: np.expand_dims(demo['obs'][key][timestep], axis=0) for key in self.obs_keys}
        
        # Normalize RGB keys
        for key in self.obs_keys:
            if 'type' in self.obs_dict[key] and self.obs_dict[key]['type'] == 'rgb':
                obs[key] = np.moveaxis(obs[key],-1,1).astype(np.float32) / 255.

        return obs

@hydra.main(config_path="../configs/dp/robomimic_multi_modal/chi_transformer_ptp", config_name="square_abs")
def main(args):
    # ---------------- Create Logger ----------------
    set_seed(args.seed)
    # logger = Logger(pathlib.Path(args.work_dir), args)

    # ---------------- Create Environment ----------------
    # envs = make_async_envs(args)

    modality_mapping = collections.defaultdict(list)
    for key, attr in args.shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
        
    # ---------------- Create Dataset ----------------
    dataset_path = os.path.expanduser(args.dataset_path)
    dataset = RobomimicImageDataset(dataset_path, horizon=args.horizon-1, shape_meta=args.shape_meta,
                                    n_obs_steps=1, pad_before=0,
                                    pad_after=args.action_steps-1, abs_action=args.abs_action)
    print(dataset)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     num_workers=8,
    #     shuffle=False,
    #     pin_memory=True,
    #     persistent_workers=True
    # )
    
    if args.nn == "chi_transformer":
        from cleandiffuser.nn_condition import MultiImageObsConditionRobomimic
        from cleandiffuser.nn_diffusion import ChiTransformer
        
        nn_condition = MultiImageObsConditionRobomimic(
            shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop, 
            use_group_norm=args.use_group_norm, use_seq=args.use_seq, keep_horizon_dims=True).to(args.device)
        nn_diffusion = ChiTransformer(
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
    
    condition_model_path = "PATH_TO_CKPT_MODEL"
    if os.path.exists(condition_model_path):
        agent.load(condition_model_path)
    
    _convert_h5_to_embeddings(os.path.expanduser(args.dataset_path), agent.model_ema.condition, args.shape_meta.obs, dataset.normalizer, batch_size=args.batch_size, device="cuda:3")
    print("Done converting!")

if __name__ == "__main__":
    main()
