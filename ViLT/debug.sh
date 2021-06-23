
DATA_ROOT=/blob/v-jinx/data/VilT_dataset
LOG_DIR=/blob/v-jinx/checkpoint_vilt/pre_train



EXP_NAME=pretrain_indomain24GPU_nopretrain_debug
RESUME_DIR=${LOG_DIR}/${EXP_NAME}/version_0/checkpoints/last.ckpt
BATCH_SIZE=60
TOTAL_BATCH_SIZE=4320


python run_pdb.py with data_root=$DATA_ROOT log_dir=$LOG_DIR \
exp_name=$EXP_NAME num_gpus=$N_GPU_PER_WORKER \
num_nodes=${N_WORKER} task_mlm_itm_indomain_nopretrain whole_word_masking=True \
resume_from=$RESUME_DIR \
step100k batch_size=$TOTAL_BATCH_SIZE per_gpu_batchsize=$BATCH_SIZE