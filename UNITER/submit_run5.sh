cd /blob/v-jinx/virtex/UNITER
horovodrun -np 8 python train_vqa.py --config config/train-vqa-large-8gpu.json --output_dir /blob/v-jinx/checkpoint_uniter/pre_train/pretrain/vqa/default_large