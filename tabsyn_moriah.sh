#!/bin/bash
#SBATCH --job-name=tabsynfork_gpu_v1
#SBATCH --time=14:30:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40s:2
#SBATCH --output=/sci/labs/yuvalb/lee.carlin/output/%x_%j.out


source  ~/.zshrc
micromamba activate tabsyn

python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/main.py --dataname adult --method vae --mode train
python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/main.py --dataname adult --method tabsyn --mode train