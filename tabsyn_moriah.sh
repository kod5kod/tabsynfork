#!/bin/bash
#SBATCH --job-name=tabsynfork_gpu_v2
#SBATCH --time=16:30:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/sci/labs/yuvalb/lee.carlin/output/%x_%j.out
#SBATCH --killable 

jobid=%j
echo "Job $jobid Started"
source /etc/profile.d/huji-lmod.sh
echo "Loading nvidia module"
module load nvidia

source  ~/.zshrc
echo "Activating tabsyn environment"
micromamba activate tabsyn

echo "Running tabsyn_executer.py"
python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/tabsyn_executer.py 

echo "Job Ended"
# python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/main.py --dataname petfinder_tab --method tabddpm --mode train
# python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/main.py --dataname petfinder_tab --method tabddpm --mode sample --sample_size 2249   

# python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/main.py --dataname petfinder_tab --method vae --mode train
# python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/main.py --dataname petfinder_tab --method tabsyn --mode train
# python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/main.py --dataname petfinder_tab --method tabsyn --mode sample --sample_size 2249