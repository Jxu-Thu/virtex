#python train_nlvr2.py --config config/train-nlvr2-base-1gpu.json --output_dir /blob/v-jinx/checkpoint_uniter/pre_train/finetune/nlvr2/debug --debug

python pretrain_debug.py --config config/pretrain-indomain-base-8gpu.json \
    --output_dir /blob/v-jinx/checkpoint_uniter/pre_train/pretrain/indomain_debug --debug