#!/bin/bash
#SBATCH --cpus-per-task 16
#SBATCH --mem=16G
#SBATCH --gres=gpu:2,tmpdisk:10240
#SBATCH --time=7-00:00:00
#SBATCH --partition=compute_dense

echo "Loading modules"

source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load cuda/12.2
module load cudnn
module load nccl

echo "Begin training"

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so mpirun -np 2 --oversubscribe ./train_gpt2cu \
	-i "dev/data/edu_fineweb100B/edu_fineweb_train_*.bin" \
	-j "dev/data/edu_fineweb100B/edu_fineweb_val_*.bin" \
	-o "log_gpt2_1558M_slurm_${SLURM_JOB_ID}" \
	-v 250 -s 300000 -g 384 \
	-h 1 \
	-b 4 -t 1024 \
	-d 1048576 \
	-r 1 \
	-z 1 \
	-c 0.1 \
	-k "cosine" \
	-l 0.0006 \
	-q 0.1 \
	-u 700 \
	-n 2000 \
	-x 32000 \
	-ge 1 \
	-y 1 \
	-e "d48"

echo "Done"