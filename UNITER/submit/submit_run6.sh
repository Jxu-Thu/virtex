cd /blob/v-jinx/virtex/UNITER
horovodrun -np 1 python train_vqa.py --config config/train-vqa-base-1gpu-re1.json --output_dir /blob/v-jinx/checkpoint_uniter/pre_train/pretrain/vqa/default_batch10240