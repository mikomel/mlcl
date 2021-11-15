#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --time=0-16:00:00
#SBATCH --export=ALL,HYDRA_FULL_ERROR=1

enroot start \
  --rw \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  -m ~/projects/mlcl/config:/app/config \
  -m ~/projects/mlcl/mlcl:/app/mlcl \
  -m ~/datasets:/app/datasets \
  mikomel-mlcl-latest \
  mlcl/raven/train.py --config-name "${1}" "${@:2}"
