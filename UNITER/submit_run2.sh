cd /blob/v-jinx/virtex/UNITER
horovodrun -np 8 python pretrain.py --config config/pretrain-indomain-base-8gpu.json \
    --output_dir --output_dir /blob/v-jinx/checkpoint_uniter/pre_train/pretrain/indomain_submit