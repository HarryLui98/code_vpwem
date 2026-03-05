#!/bin/bash

time=$(date +%Y-%m-%d_%H-%M-%S)
for seed in 100 200 300;
do
    export CUDA_VISIBLE_DEVICES=1 && \
    export MUJOCO_EGL_DEVICE_ID=1 && \
    python pipelines/mail_mikasa_image.py >> PATH_TO_SAVE_MODEL/mikasa/mail/ShellGameTouch-v0/obs_2_act_8/${seed}_log_${time}.out \
            --config-name=shell_game_touch \
            seed=${seed} \
            obs_steps=2 action_steps=8 horizon=9
    export CUDA_VISIBLE_DEVICES=1 && \
    export MUJOCO_EGL_DEVICE_ID=1 && \
    python pipelines/mail_mikasa_image_eval.py >> PATH_TO_SAVE_MODEL/mikasa/mail/ShellGameTouch-v0/obs_2_act_8/${seed}_log_${time}_eval.out \
            --config-name=shell_game_touch \
            seed=${seed} \
            obs_steps=2 action_steps=8 horizon=9
done
wait