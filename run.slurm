#!/bin/bash
#SBATCH -A nexus_1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH -t 2-20:00:00
#SBATCH -J nexus_1
#SBATCH -o nexus_1-%A.out
#SBATCH -e nexus_1-%A.err


### change here

module purge
module load miniforge
conda activate nexus
nvidia-smi

python -m experiments.evaluate --alg_name=HYPE --model_name=EleutherAI/gpt-j-6b --hparams_fname=cf/gpt-j-6b.json --ds_name=cf --num_edits=1
