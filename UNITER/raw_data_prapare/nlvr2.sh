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

    python prepro.py --annotation $ANN_DIR/$SPLIT.json \
                     --missing_imgs $ANN_DIR/missing_nlvr2_imgs.json \
                     --output $OUT_DIR/nlvr2_${SPLIT}.db --task nlvr
done

echo "done"

#{
# "validation": {"61": "False"},
# "sentence": "The right image shows a curving walkway of dark glass circles embedded in dirt and flanked by foliage.",
# "left_url": "https://i.kinja-img.com/gawker-media/image/upload/s--UyhVSznS--/18iy0hwo5wdrpjpg.jpg",
# "writer": "61",
# "label": "False",
# "right_url": "https://cdn.pixabay.com/photo/2015/09/21/12/57/beer-bottles-949793_960_720.jpg",
# "synset": "beer bottle",
# "query": "some beer bottles38",
# "identifier": "dev-850-0-0", # 唯一标识符 (key)
# "extra_validations": {"154": "False", "139": "False", "149": "False", "62": "False"}
# }


# image feature extract
# first follow the https://github.com/lil-lab/nlvr to download the images (fuck) [need crawl images by ourself]
# switch to another docker chenrocks/butd-caffe:nlvr2
pip install imagehash --user
pip install progressbar --user

IMG_DIR=/blob/v-jinx/data/nlvr2/img
OUT_DIR=/blob/v-jinx/checkpoint_uniter/processed_data/nlvr2

set -e

echo "extracting image features..."
if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi
docker run --gpus all --ipc=host --rm \
    --mount src=$IMG_DIR,dst=/img,type=bind,readonly \
    --mount src=$OUT_DIR,dst=/output,type=bind \
    -w /src chenrocks/butd-caffe:nlvr2 \
    bash -c "python tools/generate_npz.py --gpu 0"

echo "done"