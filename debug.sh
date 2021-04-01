#CUDA_VISIBLE_DEVICES=0 python scripts/pretrain_virtex.py \
#    --config configs/_base_bicaptioning_R_50_L1_H1024.yaml \
#    --num-gpus-per-machine 1 \
#    --debug True \
#    --cpu-workers 1 \
#    --serialization-dir /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024_debug


#python scripts/preprocess/build_vocabulary.py \
#    --captions /blob/v-jinx/data/COCO/annotations/captions_train2017.json \
#    --vocab-size 10000 \
#    --output-prefix /blob/v-jinx/data/COCO/virtex/vocab/coco_10k_debug \
#    --do-lower-case


#python scripts/preprocess/preprocess_coco.py \
#    --data-root /blob/v-jinx/data/COCO/ \
#    --split val \
#    --output /blob/v-jinx/data/COCO/virtex/serialized_val.lmdb.debug


CUDA_VISIBLE_DEVICES=0 python scripts/eval_detectron2.py \
    --config /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024/pretrain_config.yaml \
    --d2-config configs/detectron2/coco_segm_default_init_2x.yaml \
    --checkpoint-path /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024/checkpoint_500000.pth \
    --weight-init virtex \
    --num-gpus-per-machine 1 \
    --cpu-workers 2 \
    --serialization-dir /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024/coco_segm_500000 \
    --checkpoint-every 5000