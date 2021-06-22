DATA_ROOT=/blob/v-jinx/data/VilT_dataset
EXP_NAME=pretrain_indomain24GPU_vqa_ft_debug
LOG_DIR=/blob/v-jinx/checkpoint_vilt/finetune
RESUME_DIR=${LOG_DIR}/pretrain_indomain24GPU/version_2/checkpoints/last.ckpt

python run_pdb.py with data_root=$DATA_ROOT log_dir=$LOG_DIR \
exp_name=$EXP_NAME num_gpus=1 num_nodes=1 \
task_finetune_vqa_randaug \
per_gpu_batchsize=32 load_path=$RESUME_DIR
