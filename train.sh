#!/bin/bash



source /home/sngwon/.bashrc

cd /home/sngwon/workspace/XR/switch_detect

conda activate XR

python train.py


#sbatch -p suma_rtx4090 -q big_qos --gres=gpu:1 train.sh
