#!/bin/bash

CUDA_VISIBLE_DEVICES=2  python jukebox/train.py --hps=small_vqvae --name=small_vqvae --sample_length=262144 --bs=4 --nworkers=20 --audio_files_dir=/data/audio --labels=False --train --aug_shift --aug_blend 

CUDA_VISIBLE_DEVICES=2  python jukebox/train.py --hps=small_vqvae,small_prior,all_fp16,cpu_ema --name=small_prior --sample_length=2097152 --bs=4 --nworkers=10 --audio_files_dir=/data/audio --labels=False --train --test --aug_shift --aug_blend --restore_vqvae=logs/small_vqvae/checkpoint_step_1000001.pth.tar --prior --levels=2 --level=1 --weight_decay=0.01 --save_iters=1000

CUDA_VISIBLE_DEVICES=2  python jukebox/train.py --hps=small_vqvae,small_upsampler,all_fp16,cpu_ema --name=small_upsampler --sample_length 262144 --bs 4 --nworkers 4 --audio_files_dir=/data/audio--labels False --train --test --aug_shift --aug_blend --restore_vqvae logs/small_vqvae/checkpoint_step_1000001.pth.tar --prior --levels 2 --level 0 --weight_decay 0.01 --save_iters 1000



#python jukebox/train.py --hps=vqvae,small_prior,all_fp16,cpu_ema --name=pretrained_vqvae_small_prior --sample_length=1048576 --bs=4 --aug_shift --aug_blend --audio_files_dir=/data/audio/ --labels=False --train --test --prior --levels=3 --level=2 --weight_decay=0.01 --save_iters=1000
