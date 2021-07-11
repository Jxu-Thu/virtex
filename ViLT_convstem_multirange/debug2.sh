DATA_ROOT=/blob/v-jinx/data/VilT_dataset
LOG_DIR=/blob/v-jinx/checkpoint_vilt/pre_train



EXP_NAME=pretrain_indomain24GPU_h512_without_pretrain_conv_bottom_conmask_debug
#RESUME_DIR=${LOG_DIR}/${EXP_NAME}/version_0/checkpoints/last.ckpt
BATCH_SIZE=32
TOTAL_BATCH_SIZE=64
# accum three


python run_pdb.py with data_root=$DATA_ROOT log_dir=$LOG_DIR \
exp_name=$EXP_NAME num_gpus=1 \
num_nodes=1 task_mlm_itm_indomain_nopretrain_middle whole_word_masking=True \
vit=vit_middle_conv_patch32_384_c \
convc=True \
max_steps=100000 \
max_patch_len=15 \
step100k batch_size=$TOTAL_BATCH_SIZE per_gpu_batchsize=$BATCH_SIZE