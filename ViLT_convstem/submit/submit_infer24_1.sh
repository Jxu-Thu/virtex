export LANG=en_US.UTF-8
export LANGUAGE=
export LC_CTYPE="en_US.UTF-8"
export LC_NUMERIC=zh_CN.UTF-8
export LC_TIME=zh_CN.UTF-8
export LC_COLLATE="en_US.UTF-8"
export LC_MONETARY=zh_CN.UTF-8
export LC_MESSAGES="en_US.UTF-8"
export LC_PAPER=zh_CN.UTF-8
export LC_NAME=zh_CN.UTF-8
export LC_ADDRESS=zh_CN.UTF-8
export LC_TELEPHONE=zh_CN.UTF-8
export LC_MEASUREMENT=zh_CN.UTF-8
export LC_IDENTIFICATION=zh_CN.UTF-8
export LC_ALL=

sudo apt-get update
sudo apt-get install -y locales
sudo locale-gen en_US.UTF-8
sudo locale-gen zh_CN.UTF-8

sudo apt-get update
sudo apt-get install -y locales

PHILLY_USER=xuta
version=1.2.0
path=/opt/conda/envs/pytorch-py3.6/bin:/opt/conda/bin:

sudo rm /etc/sudoers.d/${PHILLY_USER}
sudo touch /etc/sudoers.d/${PHILLY_USER}
sudo chmod 777 /etc/sudoers.d/${PHILLY_USER}
sudo echo "Defaults        secure_path=\"$path:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"" > /etc/sudoers.d/${PHILLY_USER}
sudo chmod 0440 /etc/sudoers.d/${PHILLY_USER}


cd /blob/v-jinx/virtex/ViLT_convstem
sudo apt-get install -y git
sudo pip --no-cache-dir install -r requirements.txt &>/dev/null
sudo pip --no-cache-dir install -e . &>/dev/null



mkdir -p $SAVE_DIR

TASK=nlvr2
run_model=task_finetune_nlvr2_randaug
pre_trained_model="/blob/v-jinx/checkpoint_vilt/finetune_val_best/pretrain_indomain24GPU_pretrain_conv_bottom_conmask_nlvr_ft/version_1/checkpoints/last.ckpt"

if [ "$TASK" == "nlvr2" ]; then
	num_gpus=4
	per_gpu_batchsize=32
elif [ "$TASK" == "flick30k" ]; then
	num_gpus=4
	per_gpu_batchsize=4
elif [ "$TASK" == "vqa2" ]; then
	num_gpus=4
	per_gpu_batchsize=32
fi

python run.py with data_root=/blob/v-jinx/data/VilT_dataset \
 num_gpus=${num_gpus} num_nodes=1 per_gpu_batchsize=${per_gpu_batchsize} \
 middle_size \
 ${run_model} test_only=True load_path=${pre_trained_model}
