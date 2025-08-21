#!/bin/bash
#SBATCH --job-name=baseline_seqcnn
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=cimes3    
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
# I had to comment this #SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00

# --- load CUDA module first ---
module purge
module load cudatoolkit/12.6
module load anaconda3/2024.10

# --- activate conda env ---
conda activate bgc-ocean-ml

# --- get into root dir ---
cd /scratch/gpfs/GEOCLIM/LRGROUP/maximek/INMOS/bgc-ocean-ml
export PYTHONPATH=$(pwd):$PYTHONPATH


# --- run training ---
torchrun \
  --nproc_per_node=1 \
  scripts/train_multitarget.py \
  --paths_config config/paths_config.json \
  --experiment config/experiments/baseline_seqcnn_multitarget.yaml
