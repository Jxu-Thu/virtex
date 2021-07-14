EXP_NAME=pretrain_vqa_ft_full_debug
DATA_ROOT=/blob/v-jinx/data/VilT_dataset
LOG_DIR=/blob/v-jinx/checkpoint_vilt/finetune_full_last
RESUME_DIR=/blob/v-jinx/checkpoint_vilt/pre_train/pretrain_full24GPU/version_2/checkpoints/last.ckpt

python run_pdb.py with data_root=$DATA_ROOT log_dir=$LOG_DIR \
exp_name=$EXP_NAME num_gpus=1 num_nodes=1 \
task_finetune_irtr_f30k_randaug \
per_gpu_batchsize=32 load_path=$RESUME_DIR

