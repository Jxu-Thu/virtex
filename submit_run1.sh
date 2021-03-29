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


cd /blob/v-jinx/virtex
pip install tensorflow==2.4.1 --user
pip install -r requirements.txt &>/dev/null
python setup.py develop --user


python scripts/pretrain_virtex.py \
    --config configs/_base_bicaptioning_R_50_L1_H1024.yaml \
    --num-gpus-per-machine 8 \
    --cpu-workers 4 \
    --serialization-dir /blob/v-jinx/checkpoint_virtex/VIRTEX_R_50_L1_H1024

#python scripts/pretrain_virtex.py \
#    --config configs/_base_bicaptioning_R_50_L1_H1024.yaml \
#    --num-gpus-per-machine 2 \
#    --cpu-workers 4 \
#    --serialization-dir /tmp/VIRTEX_R_50_L1_H1024