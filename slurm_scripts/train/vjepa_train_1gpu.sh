#!/bin/bash
#SBATCH --job-name=vjepa
#SBATCH --partition=medai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=auto
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=30G
#SBATCH --exclude=SH-IDC1-10-140-0-[136,137,168-175,177,224,228,207,217],SH-IDC1-10-140-0-[222-230],SH-IDC1-10-140-1-[3,33,79,148,156-160,162,163,164,165,168,177]
#SBATCH --chdir=/mnt/petrelfs/linweixiong/jepa
#SBATCH --output=/mnt/petrelfs/linweixiong/jepa/logs/%x-%j.out
#SBATCH --error=/mnt/petrelfs/linweixiong/jepa/logs/%x-%j.error
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2

export NCCL_DEBUG=INFO
export NCCL_IBEXT_DISABLE=1
export NCCL_SOCKET_IFNAME=eth2
# export NCCL_IB_HCA=mlx5_0
# export NCCL_SOCKET_IFNAME=eth0
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
MASTER_PORT=$((RANDOM % 101 + 20000))
echo "MASTER_ADDR="$MASTER_ADDR

export WANDB_API_KEY=25c5d7a01737cb2a8e2b8357f008d4e518a54390

python -m app.main \
  --fname configs/pretrain/vitl16.yaml \
  --devices cuda:0 \
  --name vjepa-train \
  --use_wandb

