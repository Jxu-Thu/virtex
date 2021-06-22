DATA_ROOT=/blob/v-jinx/data/VilT_dataset

python -m pdb run_pdb.py with data_root=$DATA_ROOT \
num_gpus=1 num_nodes=1 \
task_finetune_vqa_randaug \
per_gpu_batchsize=32 load_path=$RESUME_DIR