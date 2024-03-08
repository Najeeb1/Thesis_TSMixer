#!/usr/bin/env bash
#SBATCH --job-name=mmnist_tsmixer
#SBATCH --output=runs/TSMIXER_MNINST_STUD%j.log
#SBATCH --error=runs/TSMIXER_MNIST_STUD%j.err
#SBATCH --mail-user=hussainin@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

srun python3 -u main.py \
  --epochs 150