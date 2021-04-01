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

# aria2c -c https://nlp.stanford.edu/software/stanford-corenlp-3.7.0.zip
# export CLASSPATH=/path/to/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar
