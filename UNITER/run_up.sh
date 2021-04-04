bash scripts/download_nlvr2.sh /blob/v-jinx/checkpoint_uniter/pre_train

source launch_container.sh /blob/v-jinx/checkpoint_uniter/pre_train/txt_db /blob/v-jinx/checkpoint_uniter/pre_train/img_db \
    /blob/v-jinx/checkpoint_uniter/pre_train/finetune /blob/v-jinx/checkpoint_uniter/pre_train/pretrained

# Jump the docker setting, directly use docker image provide by docker.io/chenrocks/uniter:latest



TXT_DB=/blob/v-jinx/checkpoint_uniter/pre_train/txt_db
IMG_DIR=/blob/v-jinx/checkpoint_uniter/pre_train/img_db
OUTPUT=/blob/v-jinx/checkpoint_uniter/pre_train/finetune
PRETRAIN_DIR=/blob/v-jinx/checkpoint_uniter/pre_train/pretrained

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi


docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$OUTPUT,dst=/storage,type=bind \
    --mount src=$PRETRAIN_DIR,dst=/pretrain,type=bind,readonly \
    --mount src=$TXT_DB,dst=/txt,type=bind,readonly \
    --mount src=$IMG_DIR,dst=/img,type=bind,readonly \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /src chenrocks/uniter
