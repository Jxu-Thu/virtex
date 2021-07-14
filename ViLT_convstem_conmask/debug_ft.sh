EXP_NAME=pretrain_indomain24GPU_pretrain_conv_bottom_conmask_vqa_ft
DATA_ROOT=/blob/v-jinx/data/VilT_dataset
LOG_DIR=/blob/v-jinx/checkpoint_vilt/finetune_val_best
RESUME_DIR=/blob/v-jinx/checkpoint_vilt/pre_train/pretrain_indomain24GPU_h512_without_pretrain_conv_bottom_conmask/version_4/checkpoints/last.ckpt
SAVE_DIR=$LOG_DIR/$EXP_NAME

python run.py with data_root=$DATA_ROOT log_dir=$LOG_DIR \
exp_name=$EXP_NAME num_gpus=1 num_nodes=1 \
vit=vit_middle_conv_patch32_384 \
task_finetune_vqa_randaug \
per_gpu_batchsize=32 load_path=$RESUME_DIR | tee -a $SAVE_DIR/log.txt