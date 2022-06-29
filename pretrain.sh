#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python jukebox/train.py --hps=vqvae,small_prior,all_fp16,cpu_ema --name=pretrained_vqvae_small_prior --sample_length=1048576 --bs=4 --aug_shift --aug_blend --audio_files_dir=/data/audio/ --labels=False --train --test --restore_prior=logs/pretrained_vqvae_small_prior/checkpoint_latest.pth.tar --prior --levels=3 --level=2 --weight_decay=0.01 --save_iters=1000
