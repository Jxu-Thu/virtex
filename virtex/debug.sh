#CUDA_VISIBLE_DEVICES=0 python scripts/pretrain_virtex.py \
#    --config configs/_base_bicaptioning_R_50_L1_H1024.yaml \
#    --num-gpus-per-machine 1 \
#    --debug True \
#    --cpu-workers 1 \
#    --serialization-dir /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024_debug


#python scripts/preprocess/build_vocabulary.py \
#    --captions /blob/v-jinx/data/coco/annotations/captions_train2017.json \
#    --vocab-size 10000 \
#    --output-prefix /blob/v-jinx/data/coco/virtex/vocab/coco_10k_debug \
#    --do-lower-case


#python scripts/preprocess/preprocess_coco.py \
#    --data-root /blob/v-jinx/data/coco/ \
#    --split val \
#    --output /blob/v-jinx/data/coco/virtex/serialized_val.lmdb.debug


#export DETECTRON2_DATASETS=/blob/v-jinx/data
#CUDA_VISIBLE_DEVICES=0 python scripts/eval_detectron2.py \
#    --config /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024/pretrain_config.yaml \
#    --d2-config configs/detectron2/coco_segm_default_init_2x.yaml \
#    --checkpoint-path /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024/checkpoint_500000.pth \
#    --weight-init virtex \
#    --num-gpus-per-machine 1 \
#    --cpu-workers 2 \
#    --serialization-dir /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024/coco_segm_500000 \
#    --checkpoint-every 5000


export DETECTRON2_DATASETS=/blob/v-jinx/data
CUDA_VISIBLE_DEVICES=0 python scripts/eval_detectron2.py \
    --config /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024/pretrain_config.yaml \
    --d2-config configs/detectron2/coco_segm_default_init_2x.yaml \
    --resume \
    --checkpoint-path /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024/coco_segm_500000/model_0179999.pth \
    --eval-only \
    --weight-init virtex \
    --num-gpus-per-machine 1 \
    --cpu-workers 2 \
    --serialization-dir /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024/coco_segm_500000 \
    --checkpoint-every 5000


#CUDA_VISIBLE_DEVICES=0  python scripts/eval_captioning.py \
#    --config /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024/pretrain_config.yaml \
#    --checkpoint-path  /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024/checkpoint_500000.pth \
#    --calc-metrics \
#    --num-gpus-per-machine 1 \
#    --cpu-workers 4