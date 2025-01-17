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
mkdir -p LOG_DIR

echo address,$MASTER_ADDR
echo port,$MASTER_PORT
echo rank,$NODE_RANK
echo gpu_per_node,$N_GPU_PER_WORKER
echo nodes,$N_WORKER

EXP_NAME=pretrain_indomain24GPU_debug
#RESUME_DIR=${LOG_DIR}/${EXP_NAME}/version_0/checkpoints/last.ckpt
BATCH_SIZE=64
CKPT_DIR=s3://bucket-7001/luoxu/dataset/MMT/vilt_checkpoint/${EXP_NAME}

python run.py with data_root=$DATA_ROOT log_dir=$LOG_DIR \
exp_name=$EXP_NAME num_gpus=$N_GPU_PER_WORKER \
num_nodes=${N_WORKER} task_mlm_itm_indomain whole_word_masking=True \
proxy_dataset_debug=True \
huawei_target_dir=$CKPT_DIR \
huawei_flag=True \
huawei_root_path=/cache/VilT_dataset \
step100k per_gpu_batchsize=$BATCH_SIZE

#python run.py with data_root=$DATA_ROOT log_dir=$LOG_DIR \
#exp_name=$EXP_NAME num_gpus=$N_GPU_PER_WORKER \
#num_nodes=${N_WORKER} task_mlm_itm_indomain whole_word_masking=True \
#resume_from=$RESUME_DIR \
#step100k batch_size=$TOTAL_BATCH_SIZE per_gpu_batchsize=$BATCH_SIZE