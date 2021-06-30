DATA_ROOT=/blob/v-jinx/data/VilT_dataset
LOG_DIR=/blob/v-jinx/checkpoint_vilt/pre_train



EXP_NAME=pretrain_indomain24GPU_nopretrain
RESUME_DIR=${LOG_DIR}/${EXP_NAME}/version_0/checkpoints/last.ckpt
BATCH_SIZE=32
TOTAL_BATCH_SIZE=64


python run.py with data_root=$DATA_ROOT log_dir=$LOG_DIR \
exp_name=$EXP_NAME num_gpus=1 \
num_nodes=1 task_mlm_itm_indomain_nopretrain_middle_no_itm whole_word_masking=True \
resume_from=$RESUME_DIR \
step100k batch_size=$TOTAL_BATCH_SIZE per_gpu_batchsize=$BATCH_SIZE