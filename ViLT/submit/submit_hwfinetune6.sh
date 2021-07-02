N_GPU_PER_WORKER=$1
N_WORKER=$2
NODE_RANK=$3
MASTER_ADDR=$4
MASTER_PORT=$5

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NODE_RANK=$NODE_RANK
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=Ring
export NCCL_DEBUG=INFO

DATA_ROOT=/cache/VilT_dataset
LOG_DIR=/cache/checkpoint
RESUME_DIR=/cache/checkpoint/resume_dir/last.ckpt
mkdir -p LOG_DIR

echo address,$MASTER_ADDR
echo port,$MASTER_PORT
echo rank,$NODE_RANK
echo gpu_per_node,$N_GPU_PER_WORKER
echo nodes,$N_WORKER

EXP_NAME=pretrain_indomain24GPU_h512_without_pretrain_ft
#RESUME_DIR=${LOG_DIR}/${EXP_NAME}/version_0/checkpoints/last.ckpt
CKPT_DIR=s3://bucket-7001/luoxu/dataset/MMT/vilt_checkpoint/${EXP_NAME}

python run.py with data_root=$DATA_ROOT log_dir=$LOG_DIR \
exp_name=$EXP_NAME num_gpus=$N_GPU_PER_WORKER \
num_nodes=${N_WORKER} \
task_finetune_middle_vqa_randaug \
huawei_target_dir=$CKPT_DIR \
huawei_flag=True \
huawei_root_path=/cache/VilT_dataset \
load_path=$RESUME_DIR \
per_gpu_batchsize=64