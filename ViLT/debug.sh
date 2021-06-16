#MASTER_PORT=54321       # Port of master server
#N_GPU_LOCAL=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
#N_GPU_PER_WORKER=${DLTS_NUM_GPU_PER_WORKER:-${N_GPU_LOCAL}}
#N_WORKER=${DLTS_NUM_WORKER:-1}
#NODE_RANK=${DLTS_ROLE_IDX:-0}
#MASTER_ADDR=${MASTER_IP:-127.0.0.1}
#
#export NCCL_TREE_THRESHOLD=0
#export NCCL_ALGO=Ring
#export NCCL_DEBUG=INFO
#
#
#dist_config="-m torch.distributed.launch --nproc_per_node=${N_GPU_PER_WORKER} --nnodes=${N_WORKER} --node_rank=${NODE_RANK} \
#    --master_addr="${MASTER_ADDR}" --master_port=${MASTER_PORT}"

#CUDA_VISIBLE_DEVICES=0
MASTER_PORT=54321       # Port of master server
N_GPU_LOCAL=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
N_GPU_PER_WORKER=${DLTS_NUM_GPU_PER_WORKER:-${N_GPU_LOCAL}}
N_WORKER=${DLTS_NUM_WORKER:-1}
NODE_RANK=${DLTS_ROLE_IDX:-0}
MASTER_ADDR=${MASTER_IP:-127.0.0.1}
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NODE_RANK=$NODE_RANK
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=Ring
export NCCL_DEBUG=INFO

exp_name=debug_pretrain
data_root=/blob/v-jinx/data/VilT_dataset
log_dir=/blob/v-jinx/checkpoint_vilt/pre_train
python run_pdb.py with data_root=${data_root} log_dir=${log_dir} exp_name=${exp_name} num_gpus=1 num_nodes=1 task_mlm_itm whole_word_masking=True step100k per_gpu_batchsize=32