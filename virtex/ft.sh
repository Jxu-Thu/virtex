# CoCo detection /
python scripts/eval_detectron2.py \
    --config /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024/pretrain_config.yaml \
    --d2-config configs/detectron2/coco_segm_default_init_2x.yaml \
    --checkpoint-path /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024/checkpoint_500000.pth \
    --weight-init virtex \
    --num-gpus-per-machine 8 \
    --cpu-workers 2 \
    --serialization-dir /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024/coco_segm_500000 \
    --checkpoint-every 5000