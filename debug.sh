CUDA_VISIBLE_DEVICES=0 python scripts/pretrain_virtex.py \
    --config configs/_base_bicaptioning_R_50_L1_H1024.yaml \
    --num-gpus-per-machine 1 \
    --debug True \
    --cpu-workers 4 \
    --serialization-dir /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024