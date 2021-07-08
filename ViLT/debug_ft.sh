EXP_NAME=pretrain_indomain24GPU_vqa__nopretrain_ft_debug
DATA_ROOT=/blob/v-jinx/data/VilT_dataset
LOG_DIR=/blob/v-jinx/checkpoint_vilt/finetune_val_best
RESUME_DIR=/blob/v-jinx/checkpoint_vilt/pre_train/pretrain_indomain24GPU_nopretrain/version_2/checkpoints/epoch=79-step=99839.ckpt

python run.py with data_root=$DATA_ROOT log_dir=$LOG_DIR \
exp_name=$EXP_NAME num_gpus=1 num_nodes=1 \
task_finetune_vqa_randaug \
per_gpu_batchsize=32 load_path=$RESUME_DIR