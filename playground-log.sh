# Playground log
# https://github.com/karpathy/llm.c/discussions/677

# MARK: setup
./dev/download_starter_pack.sh
(cd dev/data && ./edu_fineweb.sh)

ln -s /mnt/wato-drive2/ben-shared/gpt2-llm.c/data/* dev/data/
ln -s /mnt/wato-drive2/ben-shared/gpt2-llm.c/*.bin .

source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load cuda/12.2
module load cudnn
module load nccl

make train_gpt2cu USE_CUDNN=1

# MARK: EXP 2-GPU training (2x RTX 4090)
NCCL_DEBUG=INFO #optional
# OOM
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so mpirun -np 2 --oversubscribe ./train_gpt2cu \
	-i "dev/data/edu_fineweb100B/edu_fineweb_train_*.bin" \
	-j "dev/data/edu_fineweb100B/edu_fineweb_val_*.bin" \
	-o "log_gpt2_1558M" \
	-v 250 -s 300000 -g 384 \
	-h 1 \
	-b 16 -t 1024 \
	-d 1048576 \
	-r 0 \
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

# OOM
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so mpirun -np 2 --oversubscribe ./train_gpt2cu \
	-i "dev/data/edu_fineweb100B/edu_fineweb_train_*.bin" \
	-j "dev/data/edu_fineweb100B/edu_fineweb_val_*.bin" \
	-o "log_gpt2_1558M" \
	-v 250 -s 300000 -g 384 \
	-h 1 \
	-b 8 -t 1024 \
	-d 1048576 \
	-r 0 \
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

# OOM
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so mpirun -np 2 --oversubscribe ./train_gpt2cu \
	-i "dev/data/edu_fineweb100B/edu_fineweb_train_*.bin" \
	-j "dev/data/edu_fineweb100B/edu_fineweb_val_*.bin" \
	-o "log_gpt2_1558M" \
	-v 250 -s 300000 -g 384 \
	-h 1 \
	-b 4 -t 1024 \
	-d 1048576 \
	-r 0 \
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


# HellaSwag EvalLoader: batch size 2 is < 4
# ---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so mpirun -np 2 --oversubscribe ./train_gpt2cu \
	-i "dev/data/edu_fineweb100B/edu_fineweb_train_*.bin" \
	-j "dev/data/edu_fineweb100B/edu_fineweb_val_*.bin" \
	-o "log_gpt2_1558M" \
	-v 250 -s 300000 -g 384 \
	-h 1 \
	-b 2 -t 1024 \
	-d 1048576 \
	-r 0 \
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

# Works
# +---------------------------------------------------------------------------------------+
# | NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |
# |-----------------------------------------+----------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
# |                                         |                      |               MIG M. |
# |=========================================+======================+======================|
# |   0  NVIDIA GeForce RTX 4090        Off | 00000000:01:00.0 Off |                  Off |
# | 30%   59C    P2             435W / 450W |  23540MiB / 24564MiB |     90%      Default |
# |                                         |                      |                  N/A |
# +-----------------------------------------+----------------------+----------------------+
# |   1  NVIDIA GeForce RTX 4090        Off | 00000000:02:00.0 Off |                  Off |
# | 33%   67C    P2             440W / 450W |  23540MiB / 24564MiB |    100%      Default |
# |                                         |                      |                  N/A |
# +-----------------------------------------+----------------------+----------------------+

# +---------------------------------------------------------------------------------------+
# | Processes:                                                                            |
# |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
# |        ID   ID                                                             Usage      |
# |=======================================================================================|
# |    0   N/A  N/A   1094073      C   ./train_gpt2cu                            23406MiB |
# |    1   N/A  N/A   1094074      C   ./train_gpt2cu                            23406MiB |
# +---------------------------------------------------------------------------------------+

# step    1/32000 | loss 11.132117 (+nanz)| norm 54.4216 (+nanz)| lr 8.57e-07 | 41503.39 ms | 75.1% bf16 MFU | 25265 tok/s
# step    2/32000 | loss 10.541794 (+nanz)| norm 43.3089 (+nanz)| lr 1.71e-06 | 42348.47 ms | 73.6% bf16 MFU | 24761 tok/s
# step    3/32000 | loss 9.879589 (+nanz)| norm 23.6661 (+nanz)| lr 2.57e-06 | 42234.13 ms | 73.8% bf16 MFU | 24795 tok/s
# step    4/32000 | loss 9.588540 (+nanz)| norm 33.3721 (+nanz)| lr 3.43e-06 | 42021.53 ms | 74.1% bf16 MFU | 24851 tok/s
# step    5/32000 | loss 9.483225 (+nanz)| norm 24.4922 (+nanz)| lr 4.29e-06 | 41932.03 ms | 74.3% bf16 MFU | 24893 tok/s
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so mpirun -np 2 --oversubscribe ./train_gpt2cu \
	-i "dev/data/edu_fineweb100B/edu_fineweb_train_*.bin" \
	-j "dev/data/edu_fineweb100B/edu_fineweb_val_*.bin" \
	-o "log_gpt2_1558M" \
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

# MARK: EXP Single-GPU training (RTX 4090)
# OOM
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so mpirun -np 1 --oversubscribe ./train_gpt2cu \
	-i "dev/data/edu_fineweb100B/edu_fineweb_train_*.bin" \
	-j "dev/data/edu_fineweb100B/edu_fineweb_val_*.bin" \
	-o "log_gpt2_1558M" \
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

# MARK: EXP 2-GPU training (2x RTX 3090)
# Tue Jul 16 17:10:57 2024
# +---------------------------------------------------------------------------------------+
# | NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |
# |-----------------------------------------+----------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
# |                                         |                      |               MIG M. |
# |=========================================+======================+======================|
# |   0  NVIDIA GeForce RTX 3090        Off | 00000000:01:00.0 Off |                  N/A |
# | 60%   71C    P2             347W / 350W |  23382MiB / 24576MiB |    100%      Default |
# |                                         |                      |                  N/A |
# +-----------------------------------------+----------------------+----------------------+
# |   1  NVIDIA GeForce RTX 3090        Off | 00000000:02:00.0 Off |                  N/A |
# | 46%   68C    P2             347W / 350W |  23382MiB / 24576MiB |    100%      Default |
# |                                         |                      |                  N/A |
# +-----------------------------------------+----------------------+----------------------+

# +---------------------------------------------------------------------------------------+
# | Processes:                                                                            |
# |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
# |        ID   ID                                                             Usage      |
# |=======================================================================================|
# |    0   N/A  N/A   2009312      C   ./train_gpt2cu                            23248MiB |
# |    1   N/A  N/A   2009313      C   ./train_gpt2cu                            23248MiB |
# +---------------------------------------------------------------------------------------+
# step    1/32000 | loss 11.132113 (+nanz)| norm 54.1606 (+nanz)| lr 8.57e-07 | 89431.73 ms | 80.9% bf16 MFU | 11725 tok/s
# step    2/32000 | loss 10.541922 (+nanz)| norm 43.1549 (+nanz)| lr 1.71e-06 | 90014.20 ms | 80.4% bf16 MFU | 11649 tok/s
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so mpirun -np 2 --oversubscribe ./train_gpt2cu \
	-i "dev/data/edu_fineweb100B/edu_fineweb_train_*.bin" \
	-j "dev/data/edu_fineweb100B/edu_fineweb_val_*.bin" \
	-o "log_gpt2_1558M" \
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

# MARK: EXP 4-GPU training (4x RTX 3090)

# About as fast as 2-GPU 4090 training. Though uses less memory per GPU.
# Tue Jul 16 17:17:45 2024
# +---------------------------------------------------------------------------------------+
# | NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |
# |-----------------------------------------+----------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
# |                                         |                      |               MIG M. |
# |=========================================+======================+======================|
# |   0  NVIDIA GeForce RTX 3090        Off | 00000000:01:00.0 Off |                  N/A |
# | 53%   70C    P2             346W / 350W |  18924MiB / 24576MiB |     99%      Default |
# |                                         |                      |                  N/A |
# +-----------------------------------------+----------------------+----------------------+
# |   1  NVIDIA GeForce RTX 3090        Off | 00000000:02:00.0 Off |                  N/A |
# | 30%   59C    P2             346W / 350W |  18924MiB / 24576MiB |     95%      Default |
# |                                         |                      |                  N/A |
# +-----------------------------------------+----------------------+----------------------+
# |   2  NVIDIA GeForce RTX 3090        Off | 00000000:03:00.0 Off |                  N/A |
# | 30%   54C    P2             346W / 350W |  18924MiB / 24576MiB |     99%      Default |
# |                                         |                      |                  N/A |
# +-----------------------------------------+----------------------+----------------------+
# |   3  NVIDIA GeForce RTX 3090        Off | 00000000:04:00.0 Off |                  N/A |
# | 30%   51C    P2             345W / 350W |  18924MiB / 24576MiB |     90%      Default |
# |                                         |                      |                  N/A |
# +-----------------------------------------+----------------------+----------------------+

# +---------------------------------------------------------------------------------------+
# | Processes:                                                                            |
# |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
# |        ID   ID                                                             Usage      |
# |=======================================================================================|
# |    0   N/A  N/A   2013361      C   ./train_gpt2cu                            18790MiB |
# |    1   N/A  N/A   2013362      C   ./train_gpt2cu                            18790MiB |
# |    2   N/A  N/A   2013363      C   ./train_gpt2cu                            18790MiB |
# |    3   N/A  N/A   2013364      C   ./train_gpt2cu                            18790MiB |
# +---------------------------------------------------------------------------------------+
# step    1/32000 | loss 11.133206 (+nanz)| norm 53.7835 (+nanz)| lr 8.57e-07 | 46083.57 ms | 78.5% bf16 MFU | 22754 tok/s
# step    2/32000 | loss 10.543051 (+nanz)| norm 43.3902 (+nanz)| lr 1.71e-06 | 45733.08 ms | 79.1% bf16 MFU | 22928 tok/s
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so mpirun -np 4 --oversubscribe ./train_gpt2cu \
	-i "dev/data/edu_fineweb100B/edu_fineweb_train_*.bin" \
	-j "dev/data/edu_fineweb100B/edu_fineweb_val_*.bin" \
	-o "log_gpt2_1558M" \
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

# Can we bump up the micro batch size?
# OOM
# allocating 2971 MiB for parameter gradients
# allocating 1485 MiB for AdamW optimizer state m
# allocating 1485 MiB for AdamW optimizer state v
# [CUDA ERROR] at file train_gpt2.cu:1031:
# out of memory
# [CUDA ERROR] at file train_gpt2.cu:1031:
# out of memory
# [CUDA ERROR] at file train_gpt2.cu:1031:
# out of memory
# [CUDA ERROR] at file train_gpt2.cu:1031:
# out of memory
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so mpirun -np 4 --oversubscribe ./train_gpt2cu \
	-i "dev/data/edu_fineweb100B/edu_fineweb_train_*.bin" \
	-j "dev/data/edu_fineweb100B/edu_fineweb_val_*.bin" \
	-o "log_gpt2_1558M" \
	-v 250 -s 300000 -g 384 \
	-h 1 \
	-b 8 -t 1024 \
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

# Can we avoid recalculating?
# Works, slightly faster and uses more memory
# Tue Jul 16 17:23:57 2024
# +---------------------------------------------------------------------------------------+
# | NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |
# |-----------------------------------------+----------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
# |                                         |                      |               MIG M. |
# |=========================================+======================+======================|
# |   0  NVIDIA GeForce RTX 3090        Off | 00000000:01:00.0 Off |                  N/A |
# | 58%   72C    P2             346W / 350W |  21274MiB / 24576MiB |    100%      Default |
# |                                         |                      |                  N/A |
# +-----------------------------------------+----------------------+----------------------+
# |   1  NVIDIA GeForce RTX 3090        Off | 00000000:02:00.0 Off |                  N/A |
# | 39%   67C    P2             348W / 350W |  21274MiB / 24576MiB |    100%      Default |
# |                                         |                      |                  N/A |
# +-----------------------------------------+----------------------+----------------------+
# |   2  NVIDIA GeForce RTX 3090        Off | 00000000:03:00.0 Off |                  N/A |
# | 30%   60C    P2             347W / 350W |  21274MiB / 24576MiB |    100%      Default |
# |                                         |                      |                  N/A |
# +-----------------------------------------+----------------------+----------------------+
# |   3  NVIDIA GeForce RTX 3090        Off | 00000000:04:00.0 Off |                  N/A |
# | 33%   64C    P2             348W / 350W |  21274MiB / 24576MiB |    100%      Default |
# |                                         |                      |                  N/A |
# +-----------------------------------------+----------------------+----------------------+

# +---------------------------------------------------------------------------------------+
# | Processes:                                                                            |
# |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
# |        ID   ID                                                             Usage      |
# |=======================================================================================|
# |    0   N/A  N/A   2017228      C   ./train_gpt2cu                            21140MiB |
# |    1   N/A  N/A   2017229      C   ./train_gpt2cu                            21140MiB |
# |    2   N/A  N/A   2017230      C   ./train_gpt2cu                            21140MiB |
# |    3   N/A  N/A   2017231      C   ./train_gpt2cu                            21140MiB |
# +---------------------------------------------------------------------------------------+
# val loss 11.133013
# allocating 2971 MiB for parameter gradients
# allocating 1485 MiB for AdamW optimizer state m
# allocating 1485 MiB for AdamW optimizer state v
# allocating 1485 MiB for master copy of params
# step    1/32000 | loss 11.133206 (+nanz)| norm 53.7835 (+nanz)| lr 8.57e-07 | 45297.97 ms | 79.8% bf16 MFU | 23148 tok/s
# step    2/32000 | loss 10.543036 (+nanz)| norm 43.3904 (+nanz)| lr 1.71e-06 | 45195.48 ms | 80.0% bf16 MFU | 23201 tok/s
# step    3/32000 | loss 9.879162 (+nanz)| norm 23.3768 (+nanz)| lr 2.57e-06 | 45277.27 ms | 79.9% bf16 MFU | 23179 tok/s
# step    4/32000 | loss 9.576423 (+nanz)| norm 30.8756 (+nanz)| lr 3.43e-06 | 45690.91 ms | 79.1% bf16 MFU | 23099 tok/s
# step    5/32000 | loss 9.448011 (+nanz)| norm 24.1391 (+nanz)| lr 4.29e-06 | 45958.85 ms | 78.7% bf16 MFU | 23022 tok/s
# step    6/32000 | loss 9.323767 (+nanz)| norm 16.1434 (+nanz)| lr 5.14e-06 | 45854.38 ms | 78.9% bf16 MFU | 22988 tok/s
# step    7/32000 | loss 9.196299 (+nanz)| norm 15.2757 (+nanz)| lr 6.00e-06 | 45946.05 ms | 78.7% bf16 MFU | 22957 tok/s
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so mpirun -np 4 --oversubscribe ./train_gpt2cu \
	-i "dev/data/edu_fineweb100B/edu_fineweb_train_*.bin" \
	-j "dev/data/edu_fineweb100B/edu_fineweb_val_*.bin" \
	-o "log_gpt2_1558M" \
	-v 250 -s 300000 -g 384 \
	-h 1 \
	-b 4 -t 1024 \
	-d 1048576 \
	-r 0 \
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

# MARK: 2-GPU training (2x RTX 4090)
# > sbatch --nodelist trpro-slurm2 ./slurm_2_gpu.sh
# sbatch: INFO: You are submitting to the 'compute_dense' partition. This is a special partition for jobs that use all reserved resources effectively. Please ensure that your job satisfies this requirement. You can test your job using the default 'compute' partition before submitting to 'compute_dense'.
# Submitted batch job 6292
# Note: our MFU is much higher (74% vs 50% in Karpathy's post). Perhaps it's because our GPU doesn't have too many FLOPS in the first place?
# EDIT: yep, looks like it. We get around 80% MFU on the 3090 2-GPU training.
