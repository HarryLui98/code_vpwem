#!/bin/bash
time=$(date +%Y-%m-%d_%H-%M-%S)

mem_frame_type='random_from_full'
cache_compress='fifo'
mem_token_len=2
qformer_lyr=2
cache_len=16
dropout=0.3
mem_pos_emb='sin_pos'
env="longsquare"
type="ph"
subsample=20

for seed in 100 200 300;
do
    export CUDA_VISIBLE_DEVICES=0 && \
    export MUJOCO_EGL_DEVICE_ID=0 && \
    python pipelines/longmemdpptp_robomimic_image.py >> /PATH_TO_SAVE_MODEL/robomimic/${env}_${type}/obs_1_act_8/qformer_${mem_token_len}_${mem_frame_type}_${mem_frame_len}_${mem_frame_len_randomization}_${mem_pos_emb}_${qformer_lyr}_${cache_compress}_${cache_len}_${dropout}_seed_${seed}_log_${time}.out \
        seed=${seed} env_name=${env} env_type=${type} \
        obs_steps=2 action_steps=8 horizon=9 \
        mem_compress_method='qformer' \
        mem_compress_length=${mem_token_len} \
        dataset_memory_type=${mem_frame_type} \
        dataset_memory_num_frames=${mem_frame_len} \
        dataset_memory_num_frames_randomization=${mem_frame_len_randomization} \
        memory_pos_emb=${mem_pos_emb} \
        qformer_num_layers=${qformer_lyr} \
        qformer_cache_compress=${cache_compress} \
        qformer_cache_max_length=${cache_len} \
        short_cond_dropout=${dropout} \
        long_cond_dropout=${dropout} \
        dataset_memory_subsample_ratio=${subsample}
    export CUDA_VISIBLE_DEVICES=0 && \
    export MUJOCO_EGL_DEVICE_ID=0 && \
    python pipelines/longmemdpptp_robomimic_lhsq_image_eval.py >> /PATH_TO_SAVE_MODEL/robomimic/${env}_${type}/obs_1_act_8/qformer_${mem_token_len}_${mem_frame_type}_${mem_frame_len}_${mem_frame_len_randomization}_${mem_pos_emb}_${qformer_lyr}_${cache_compress}_${cache_len}_${dropout}_seed_${seed}_log_${time}_eval.out \
        seed=${seed} env_name=${env} env_type=${type} \
        obs_steps=2 action_steps=8 horizon=9 \
        mem_compress_method='qformer' \
        mem_compress_length=${mem_token_len} \
        dataset_memory_type=${mem_frame_type} \
        dataset_memory_num_frames=${mem_frame_len} \
        dataset_memory_num_frames_randomization=${mem_frame_len_randomization} \
        memory_pos_emb=${mem_pos_emb} \
        qformer_num_layers=${qformer_lyr} \
        qformer_cache_compress=${cache_compress} \
        qformer_cache_max_length=${cache_len} \
        short_cond_dropout=${dropout} \
        long_cond_dropout=${dropout} \
        dataset_memory_subsample_ratio=${subsample}
done
wait
