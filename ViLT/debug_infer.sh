python run_pdb.py with data_root=/blob/v-jinx/data/VilT_dataset \
num_gpus=1 num_nodes=1 \
per_gpu_batchsize=32 task_finetune_vqa_randaug test_only=True \
load_path="/blob/v-jinx/checkpoint_vilt/finetune_full_last/pretrain_indomain24GPU_vqa_ft/version_1/checkpoints/last.ckpt"