#!/bin/bash

python jukebox/sample.py --model=5b_lyrics --name=pretrained_vqvae_small_prior --restore_vqvae=logs/pretrained_vqvae_small_prior/checkpoint_latest.pth.tar --restore_prior=logs/pretrained_vqvae_small_prior/checkpoint_latest.pth.tar --levels=3 --sample_length_in_seconds=20 --total_sample_length_in_seconds=480 --sr=44100 --n_samples=6 --hop_fraction=0.5,0.5,0.125
