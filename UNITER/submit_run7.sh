cd /blob/v-jinx/virtex/UNITER
horovodrun -np 4 python train_vqa.py --config config/train-vqa-base-4gpu-re2.json --output_dir /blob/v-jinx/checkpoint_uniter/pre_train/pretrain/vqa/default_batch40920