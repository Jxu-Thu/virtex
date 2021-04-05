cd /blob/v-jinx/virtex/UNITER
python train_nlvr2.py --config config/train-nlvr2-base-1gpu.json --output_dir /blob/v-jinx/checkpoint_uniter/pre_train/finetune/nlvr2/submit


python inf_nlvr2.py --txt_db /blob/v-jinx/checkpoint_uniter/pre_train/txt_db/nlvr2_test1.db --img_db /blob/v-jinx/checkpoint_uniter/pre_train/img_db/nlvr2_test/ \
    --train_dir /blob/v-jinx/checkpoint_uniter/pre_train/finetune/nlvr2/submit --ckpt 6500 --output_dir /blob/v-jinx/checkpoint_uniter/pre_train/finetune/nlvr2/submit --fp16