e
EXP_NAME=pretrain_indomain24GPU_h512_without_pretrain_ft
DATA_ROOT=/blob/v-jinx/data/VilT_dataset
LOG_DIR=/blob/v-jinx/checkpoint_vilt/finetune
RESUME_DIR=/blob/v-jinx/checkpoint_vilt/pre_train/pretrain_indomain24GPU/version_2/checkpoints/last.ckpt

python run.py with data_root=$DATA_ROOT log_dir=$LOG_DIR \
exp_name=$EXP_NAME num_gpus=$N_GPU_PER_WORKER num_nodes=${N_WORKER} \
task_finetune_vqa_randaug \
per_gpu_batchsize=64 load_path=$RESUME_DIR
