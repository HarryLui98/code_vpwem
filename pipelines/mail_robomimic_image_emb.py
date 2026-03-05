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

from cleandiffuser.dataset.robomimic_dataset import RobomimicImageDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

@hydra.main(config_path="../configs/dp/robomimic_multi_modal/mail", config_name="square_abs")
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
    dataset_path = os.path.expanduser(args.dataset_path)
    dataset = RobomimicImageDataset(dataset_path, horizon=args.horizon, shape_meta=args.shape_meta,
                                    n_obs_steps=args.obs_steps, pad_before=args.obs_steps-1,
                                    pad_after=args.action_steps-1, abs_action=args.abs_action)
    print(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
        
    from cleandiffuser.nn_condition import MultiImageObsConditionRobomimic
    from cleandiffuser.nn_diffusion.mail import MaIL
        
    # nn_condition = MultiImageObsConditionRobomimic(
    #         shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
    #         crop_shape=args.crop_shape, random_crop=args.random_crop, 
    #         use_group_norm=args.use_group_norm, use_seq=args.use_seq, keep_horizon_dims=True).to(args.device)
    nn_diffusion = MaIL(args.mamba, 256, args.action_dim, 256, args.obs_steps, args.action_steps).to(args.device)
    
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
            #     condition[k] = nobs[k][:, :args.obs_steps, :].to(args.device)
            condition = nobs['mail_emb'][:, :args.obs_steps, :].to(args.device)
            naction = batch['action'][:, -args.action_steps:, :].to(args.device)

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
                logger.finish()
                break
        
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    pipeline()
