# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# bash scripts/create_txtdb.sh $PATH_TO_STORAGE/txt_db $PATH_TO_STORAGE/ann
OUT_DIR=/blob/v-jinx/checkpoint_uniter/processed_data/nlvr2
ANN_DIR=/blob/v-jinx/data/nlvr2

set -e

URL='https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data'
if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi
if [ ! -d $ANN_DIR ]; then
    mkdir -p $ANN_DIR
fi

BLOB='https://convaisharables.blob.core.windows.net/uniter'
MISSING=$BLOB/ann/missing_nlvr2_imgs.json
if [ ! -f $ANN_DIR/missing_nlvr2_imgs.json ]; then
    wget $MISSING -O $ANN_DIR/missing_nlvr2_imgs.json
fi

for SPLIT in 'train' 'dev' 'test1'; do
    if [ ! -f $ANN_DIR/$SPLIT.json ]; then
        echo "downloading ${SPLIT} annotations..."
        wget $URL/$SPLIT.json -O $ANN_DIR/$SPLIT.json
    fi

    echo "preprocessing ${SPLIT} annotations..."

#    python prepro.py --annotation $ANN_DIR/$SPLIT.json \
#                     --missing_imgs $ANN_DIR/missing_nlvr2_imgs.json \
#                     --output $OUT_DIR/nlvr2_${SPLIT}.db --task nlvr2
done

echo "done"