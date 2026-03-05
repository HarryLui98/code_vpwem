#!/bin/bash
time=$(date +%Y-%m-%d_%H-%M-%S)

mem_frame_type='random_from_full'
cache_compress='fifo'
mem_token_len=2
qformer_lyr=2
cache_len=16
dropout=0.3
mem_pos_emb='sin_pos'
env="unload_dishwasher_to_dresser"
type="suboptimal"
subsample=20

for seed in 100 200 300;
do
    export CUDA_VISIBLE_DEVICES=0 && \
    export GIBSON_DEVICE_ID=0 && \
    python pipelines/longmemdpptp_momart_image_emb.py >> PATH_TO_SAVE_MODEL/momart/${env}_${type}/obs_2_act_8/qformer_${mem_token_len}_${mem_frame_type}_${mem_pos_emb}_${qformer_lyr}_${cache_compress}_${cache_len}_${dropout}_${subsample}_seed_${seed}_log_${time}.out \
        seed=${seed} env_name=${env} env_type=${type} \
        obs_steps=2 action_steps=8 horizon=9 \
        mem_compress_method='qformer' \
        mem_compress_length=${mem_token_len} \
        dataset_memory_type=${mem_frame_type} \
        memory_pos_emb=${mem_pos_emb} \
        qformer_num_layers=${qformer_lyr} \
        qformer_cache_compress=${cache_compress} \
        qformer_cache_max_length=${cache_len} \
        short_cond_dropout=${dropout} \
        long_cond_dropout=${dropout} \
        dataset_memory_subsample_ratio=${subsample}
    export CUDA_VISIBLE_DEVICES=0 && \
    export GIBSON_DEVICE_ID=0 && \
    python pipelines/longmemdpptp_momart_image_emb_eval.py >> PATH_TO_SAVE_MODEL/momart/${env}_${type}/obs_2_act_8/qformer_${mem_token_len}_${mem_frame_type}_${mem_pos_emb}_${qformer_lyr}_${cache_compress}_${cache_len}_${dropout}_${subsample}_seed_${seed}_log_${time}_eval.out \
        seed=${seed} env_name=${env} env_type=${type} \
        obs_steps=2 action_steps=8 horizon=9\
        mem_compress_method='qformer' \
        mem_compress_length=${mem_token_len} \
        dataset_memory_type=${mem_frame_type} \
        memory_pos_emb=${mem_pos_emb} \
        qformer_num_layers=${qformer_lyr} \
        qformer_cache_compress=${cache_compress} \
        qformer_cache_max_length=${cache_len} \
        short_cond_dropout=${dropout} \
        long_cond_dropout=${dropout} \
        dataset_memory_subsample_ratio=${subsample}
done
wait
