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




cd /blob/v-jinx/DistilNAS/FairSeq-DistilNAS-EXPLatSoTa
sudo pip --no-cache-dir install torch==$version &>/dev/null
sudo pip --no-cache-dir install setuptools==39.1.0 &>/dev/null
sudo pip --no-cache-dir install --editable . &>/dev/null
sudo pip --no-cache-dir install pandas &>/dev/null

MASTER_PORT=54321       # Port of master server
N_GPU_LOCAL=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
N_GPU_PER_WORKER=${DLTS_NUM_GPU_PER_WORKER:-${N_GPU_LOCAL}}
N_WORKER=${DLTS_NUM_WORKER:-1}
NODE_RANK=${DLTS_ROLE_IDX:-0}
MASTER_ADDR=${MASTER_IP:-127.0.0.1}

export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=Ring
export NCCL_DEBUG=INFO


dist_config="-m torch.distributed.launch --nproc_per_node=${N_GPU_PER_WORKER} --nnodes=${N_WORKER} --node_rank=${NODE_RANK} \
    --master_addr="${MASTER_ADDR}" --master_port=${MASTER_PORT}"

TOTAL_UPDATES=150000    # Total number of training steps
WARMUP_UPDATES=16000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x
ARCH=nas_electras_base

DATA_DIR=/blob/v-kasong/MPNet/wikibooks/bert/
SAVE_DIR=/blob/v-jinx/checkpoint_distilnas/fairsep/nas_electras_110M_6L_generator_125k_sc

mkdir -p $SAVE_DIR

python $dist_config train_pretrain.py --fp16 $DATA_DIR \
        --save-dir $SAVE_DIR \
    --task masked_lm --criterion electra_loss \
    --arch $ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ --num-workers 8 \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 50 --ddp-backend=no_c10d \
    --nas_encoder_layers 13 \
    --ft_arch '2, 2, 0, 0, 0, 2, 3, 0, 1, 3, 1, 0, 3, 1, 3, 0, 3, 0, 1, 0, 3, 1, 1, 3, 0, 1' \
    --distributed-backend 'nccl' \
    --ddp-backend "no_c10d" \
    --generator-layers 6 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn 2>&1 | tee -a ${SAVE_DIR}/train.log