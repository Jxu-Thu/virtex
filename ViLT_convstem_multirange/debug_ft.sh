EXP_NAME=pretrain_indomain24GPU_h512_without_pretrain_conv_multirange_flick30k_ft
DATA_ROOT=/blob/v-jinx/data/VilT_dataset
LOG_DIR=/blob/v-jinx/checkpoint_vilt/finetune

python run.py with data_root=$DATA_ROOT log_dir=$LOG_DIR \
exp_name=$EXP_NAME num_gpus=1 num_nodes=1 \
task_finetune_irtr_f30k_randaug \
middle_size \
per_gpu_batchsize=4



