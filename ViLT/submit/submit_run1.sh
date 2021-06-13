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



export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK

exp_name=pretrain_vilt
data_root=/blob/v-jinx/data/VilT_dataset
log_dir=/blob/v-jinx/checkpoint_vilt/pre_train

python run.py with data_root=${data_root} log_dir=${log_dir} exp_name=${exp_name} num_gpus=1 num_nodes=1 task_mlm_itm whole_word_masking=True step200k per_gpu_batchsize=64