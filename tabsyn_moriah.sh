#!/bin/bash
#SBATCH --job-name=tabsynfork_gpu_v1
#SBATCH --time=14:30:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/sci/labs/yuvalb/lee.carlin/output/%x_%j.out


source  ~/.zshrc
micromamba activate tabsyn



python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/main.py --dataname petfinder_tab --method great --mode train
python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/main.py --dataname petfinder_tab --method great --mode sample --sample_size 2249   

# python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/main.py --dataname petfinder_tab --method vae --mode train
# python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/main.py --dataname petfinder_tab --method tabsyn --mode train
# python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/main.py --dataname petfinder_tab --method tabsyn --mode sample --sample_size 2249