export LANG=en_US.UTF-8
export LANGUAGE=
export LC_CTYPE="en_US.UTF-8"
export LC_NUMERIC=zh_CN.UTF-8
export LC_TIME=zh_CN.UTF-8
export LC_COLLATE="en_US.UTF-8"
export LC_MONETARY=zh_CN.UTF-8
export LC_MESSAGES="en_US.UTF-8"
export LC_PAPER=zh_CN.UTF-8
export LC_NAME=zh_CN.UTF-8
export LC_ADDRESS=zh_CN.UTF-8
export LC_TELEPHONE=zh_CN.UTF-8
export LC_MEASUREMENT=zh_CN.UTF-8
export LC_IDENTIFICATION=zh_CN.UTF-8
export LC_ALL=

sudo apt-get update
sudo apt-get install -y locales
sudo locale-gen en_US.UTF-8
sudo locale-gen zh_CN.UTF-8

sudo apt-get update
sudo apt-get install -y locales

PHILLY_USER=xuta
version=1.2.0
path=/opt/conda/envs/pytorch-py3.6/bin:/opt/conda/bin:

sudo rm /etc/sudoers.d/${PHILLY_USER}
sudo touch /etc/sudoers.d/${PHILLY_USER}
sudo chmod 777 /etc/sudoers.d/${PHILLY_USER}
sudo echo "Defaults        secure_path=\"$path:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"" > /etc/sudoers.d/${PHILLY_USER}
sudo chmod 0440 /etc/sudoers.d/${PHILLY_USER}


cd /blob/v-jinx/virtex/ViLT_convstem_conmask
sudo apt-get install -y git
sudo pip --no-cache-dir install -r requirements.txt &>/dev/null
sudo pip --no-cache-dir install -e . &>/dev/null


#sudo pip --no-cache-dir install torch==$version &>/dev/null

# 4卡即可 V100 32GB Wu2
# 8ka V100 32GB SC
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

EXP_NAME=pretrain_indomain24GPU_pretrain_conv_bottom_conmask_vqa_best_ft
DATA_ROOT=/blob/v-jinx/data/VilT_dataset
LOG_DIR=/blob/v-jinx/checkpoint_vilt/finetune_val_best
RESUME_DIR=/blob/v-jinx/checkpoint_vilt/pre_train/pretrain_indomain24GPU_h512_without_pretrain_conv_bottom_conmask/version_4/checkpoints/epoch=25-step=45603.ckpt
SAVE_DIR=$LOG_DIR/$EXP_NAME

mkdir -p $SAVE_DIR

python run.py with data_root=$DATA_ROOT log_dir=$LOG_DIR \
exp_name=$EXP_NAME num_gpus=$N_GPU_PER_WORKER num_nodes=${N_WORKER} \
vit=vit_middle_conv_patch32_384 \
task_finetune_middle_vqa_randaug \
per_gpu_batchsize=32 load_path=$RESUME_DIR | tee -a $SAVE_DIR/log.txt