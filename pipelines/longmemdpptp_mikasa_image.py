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

# from cleandiffuser.dataset.robomimic_dataset import RobomimicImageDataset
from cleandiffuser.dataset.mikasa_dataset_longmem import LongMemMikasaDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader, dynamic_batch_collate_fn
from cleandiffuser.utils import report_parameters
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

@hydra.main(config_path="../configs/dp/mikasa/chi_transformer_ptp_longmem", config_name="shell_game_touch")
def pipeline(args):
    # ---------------- Create Logger ----------------
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)

    assert args.horizon == args.obs_steps + args.action_steps - 1

    # modality_mapping = collections.defaultdict(list)
    # for key, attr in args.shape_meta['obs'].items():
    #     modality_mapping[attr.get('type', 'low_dim')].append(key)
    # ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
        
    # ---------------- Create Dataset ----------------
    dataset_path = os.path.expanduser(args.dataset_path)
    
    dataset = LongMemMikasaDataset(
        dataset_dir=dataset_path,
        shape_meta=args.shape_meta,
        n_obs_steps=args.obs_steps,
        horizon=args.horizon,
        pad_before=args.obs_steps-1,
        pad_after=args.action_steps-1,
        dataset_memory_type=args.dataset_memory_type,
        subsample_ratio=args.dataset_memory_subsample_ratio,
        device=args.device,
    )
    print(dataset)
    
    # Get dataset sampler for dynamic batching
    train_sampler = dataset.get_dataset_sampler()
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=train_sampler is None,  # Only shuffle if no custom sampler
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=dynamic_batch_collate_fn,
    )
        
    if args.nn == "chi_transformer":
        from cleandiffuser.nn_condition import MultiImageObsConditionRobomimic
        from cleandiffuser.nn_diffusion.longmem_chitransformerptp import LongMemChiTransformerPTP
        
        nn_condition = MultiImageObsConditionRobomimic(
            shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop, 
            use_group_norm=args.use_group_norm, use_seq=args.use_seq, keep_horizon_dims=True).to(args.device)
        nn_diffusion = LongMemChiTransformerPTP(
            args.action_dim, 256, args.horizon, args.obs_steps, 
            args.mem_compress_length,
            d_model=256, nhead=4, num_layers=4,
            p_drop_emb=args.p_drop_emb,
            p_drop_attn=args.p_drop_attn,
            timestep_emb_type="positional").to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}")
    
    from cleandiffuser.nn_memory.random_memory import RandomMemory
    from cleandiffuser.nn_memory.kmeans_memory import KMeansMemory
    from cleandiffuser.nn_memory.adjsim_memory import AdjSimMemory
    from cleandiffuser.nn_memory.qformer_memory import QFormerMemory
    if args.mem_compress_method == "random":
        nn_memory = RandomMemory(mem_compress_length=args.mem_compress_length).to(args.device)
    elif args.mem_compress_method == "kmeans":
        nn_memory = KMeansMemory(mem_compress_length=args.mem_compress_length).to(args.device)
    elif args.mem_compress_method == "adjsim":
        nn_memory = AdjSimMemory(mem_compress_length=args.mem_compress_length).to(args.device)
    elif args.mem_compress_method == "qformer":
        nn_memory = QFormerMemory(mem_compress_length=args.mem_compress_length,
                                        emb_dim=args.qformer_emb_dim,
                                        num_layers=args.qformer_num_layers,
                                        cache_compress=args.qformer_cache_compress,
                                        cache_max_length=args.qformer_cache_max_length,
                                        use_pos_emb=args.memory_pos_emb,
                                        subsample_ratio=args.dataset_memory_subsample_ratio,
                                        ).to(args.device)
    else:
        raise ValueError(f"Invalid memory method {args.mem_compress_method}")
    
    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================")
    print(f"======================= Parameter Report of Memory Model =======================")
    report_parameters(nn_memory)
    print(f"===================================================================================")
    print(f"======================= Parameter Report of Condition Model =======================")
    report_parameters(nn_condition)
    print(f"===================================================================================")

    x_max = torch.ones((1, args.horizon, args.action_dim), device=args.device) * +1.0
    x_min = torch.ones((1, args.horizon, args.action_dim), device=args.device) * -1.0
    # Get normalizer parameters (already on device)
    action_range = dataset.normalizer['action'].range.to(args.device)
    action_min = dataset.normalizer['action'].min.to(args.device)
    if args.diffusion == "ddpm":
        from cleandiffuser.diffusion.longmem_ddpm import LongMemDDPM
        agent = LongMemDDPM(
            nn_diffusion=nn_diffusion, nn_condition=nn_condition, nn_memory=nn_memory,
            device=args.device,
            diffusion_steps=args.sample_steps, x_max=x_max, x_min=x_min,
            args_dict=args,
            action_range=action_range,
            action_min=action_min,
            short_cond_dropout=args.short_cond_dropout,
            long_cond_dropout=args.long_cond_dropout,
            optim_params={"lr": args.lr})
    else:
        raise NotImplementedError(f"Diffusion method {args.diffusion} not supported for LongMem")
    
    
    n_gradient_step = 0
    lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps, last_epoch=-1)
    
    if args.mode == "train":
        # ----------------- Training ----------------------
        diffusion_loss_list = []
        start_time = time.time()
        for batch in loop_dataloader(dataloader):
            # get condition
            nobs = batch['obs']
            condition = {}
            memory = {}
            for k in nobs.keys():
                if k.endswith('_mem'):
                    # For memory keys, use the base key (without _mem) to normalize
                    base_key = k[:-4]  # Remove '_mem' suffix
                    con = nobs[k].to(args.device)  # Memory data (full sequence)
                    # Normalize using the base key's normalizer
                    con = dataset.normalizer['obs'][base_key].normalize(con)
                    memory[base_key] = con  # Store in memory dict with base key name
                elif k != 'ep_step' and k != 'actual_num_frames':
                    # For condition keys (non-memory, non-ep_step), normalize normally
                    con = nobs[k][:, :args.obs_steps, :].to(args.device)  # (B, To, ...)
                    con = dataset.normalizer['obs'][k].normalize(con)
                    condition[k] = con
            # Convert actions to torch tensor and normalize (normalizer handles device)
            naction = batch['action']  # Already torch tensor from dataloader
            if isinstance(naction, np.ndarray):
                naction = torch.from_numpy(naction).to(args.device)
            else:
                naction = naction.to(args.device)
            naction = dataset.normalizer['action'].normalize(naction)  # Returns torch tensor on device
            
            mem_steps = nobs['ep_step'].int().to(args.device)

            # update diffusion
            diffusion_loss = agent.update(naction, condition, memory, mem_steps)['loss']
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
                # logger.finish(agent)
                break
        
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    pipeline()
