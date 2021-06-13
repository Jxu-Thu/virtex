export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
exp_name=debug_pretrain
data_root=/blob/v-jinx/data/VilT_dataset
log_dir=/blob/v-jinx/checkpoint_vilt/pre_train
python run.py with data_root=${data_root} log_dir=${log_dir} exp_name=${exp_name} num_gpus=1 num_nodes=1 task_mlm_itm whole_word_masking=True step200k per_gpu_batchsize=32