python scripts/preprocess/build_vocabulary.py \
    --captions /blob/v-jinx/data/COCO/annotations/captions_train2017.json \
    --vocab-size 10000 \
    --output-prefix /blob/v-jinx/data/COCO/virtex/vocab/coco_10k \
    --do-lower-case


python scripts/preprocess/preprocess_coco.py \
    --data-root /blob/v-jinx/data/COCO \
    --split train \
    --output /blob/v-jinx/data/COCO/virtex/serialized_train.lmdb


python scripts/preprocess/preprocess_coco.py \
    --data-root /blob/v-jinx/data/COCO/ \
    --split val \
    --output /blob/v-jinx/data/COCO/virtex/serialized_val.lmdb



# for caption generation
# follow https://blog.csdn.net/kdongyi/article/details/107002068
# Download Jave Pacakge for captioning tasks


cd utils/assets
bash download_spice.sh
unzip stanford-corenlp-full-2014-08-27.zip
#
# aria2c -c https://nlp.stanford.edu/software/stanford-corenlp-3.7.0.zip
# unzip stanford-corenlp-3.7.0.zip
