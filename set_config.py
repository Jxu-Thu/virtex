from sacred import Experiment
import yaml
import time 
import os
ex = Experiment("MMC", save_git_info=False)

def _loss_names(d={}):
    ret = {
        "IC": 1,
        "TC": 1,
        "I2TC": 1,
        "T2IC": 1,
    }
    ret.update(d)
    return ret

@ex.config
def config():
    #trainer
    accelerator = "ddp"
    default_root_dir  = f"result/{str(time.time())}"
    overfit_batches = 10
    fast_dev_run = False
    val_check_interval = 1.0
    log_dir = "result/"
    exp_name = "MMCresults"
    logger = {"class_path":"pytorch_lightning.loggers.TensorBoardLogger",
              "init_args": {"save_dir": log_dir,"name": exp_name}
            }
    num_gpus = [0,1]
    num_nodes = 1
    precision = 16
    callbacks = [{"class_path": "pytorch_lightning.callbacks.LearningRateMonitor", 
                  "init_args": {"logging_interval": "step"}
                  },
                {"class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
                  "init_args":{"verbose": True, "save_last": True}
                }
    ]
    max_epochs = 1000

    # datamodule
    data_root = "../data/coco2017"
    num_workers = 4
    per_gpu_batchsize = 32
    train_transform_keys = ["Moco_transform","Moco_transform"]
    val_transform_keys = ["Moco_transform","Moco_transform"]
    image_size = 224
    image_only = False
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    whole_word_masking = True
    huawei_flag = False

    # model
    alpha = 0.2
    loss_names = _loss_names()
    lambda_ = {"IC_loss":1.0,"TC_loss":1.0,"I2TC_loss":0.0001,"T2IC_loss":0.0001}
    intra_dim = 128
    inter_dim = 1024
    hidden_dim = 2048
    queue_size = 32768
    lr_image = 0.03
    lr_text = 4e-5
    weight_decay = 1e-4
    optim_type = "sgd"
    decay_power = "cosine"
    warmup_steps = 0

@ex.automain
def main(_config):
    config_ = {}
    config_["trainer"] = {}
    # print(_config["default_root_dir"])
    os.makedirs(_config["default_root_dir"])
    config_["trainer"]["overfit_batches"] = _config["overfit_batches"]
    config_["trainer"]["accelerator"] = _config["accelerator"]
    config_["trainer"]["default_root_dir"] = _config["default_root_dir"]
    config_["trainer"]["fast_dev_run"] = _config["fast_dev_run"]
    config_["trainer"]["val_check_interval"] = _config["val_check_interval"]
    config_["trainer"]["logger"] = _config["logger"]
    config_["trainer"]["gpus"] = _config["num_gpus"]
    config_["trainer"]["num_nodes"] = _config["num_nodes"]
    config_["trainer"]["precision"] = _config["precision"]
    config_["trainer"]["callbacks"] = _config["callbacks"]
    config_["trainer"]["max_epochs"] = _config["max_epochs"]
    config_["data"] = {}
    config_["data"]["data_root"] = _config["data_root"]
    config_["data"]["num_workers"] = _config["num_workers"]
    config_["data"]["per_gpu_batchsize"] = _config["per_gpu_batchsize"]
    config_["data"]["train_transform_keys"] = _config["train_transform_keys"]
    config_["data"]["val_transform_keys"] = _config["val_transform_keys"]
    config_["data"]["image_size"] = _config["image_size"]
    config_["data"]["image_only"] = _config["image_only"]
    config_["data"]["max_text_len"] = _config["max_text_len"]
    config_["data"]["tokenizer"] = _config["tokenizer"]
    config_["data"]["whole_word_masking"] = _config["whole_word_masking"]
    config_["data"]["huawei_flag"] = _config["huawei_flag"]
    config_["model"] = {}
    config_["model"]["alpha"] = _config["alpha"]
    config_["model"]["loss_names"] = _config["loss_names"]
    config_["model"]["lambda_"] = _config["lambda_"]
    config_["model"]["intra_dim"] = _config["intra_dim"]
    config_["model"]["inter_dim"] = _config["inter_dim"]
    config_["model"]["hidden_dim"] = _config["hidden_dim"]
    config_["model"]["queue_size"] = _config["queue_size"]
    config_["model"]["lr_image"] = _config["lr_image"]
    config_["model"]["lr_text"] = _config["lr_text"]
    config_["model"]["weight_decay"] = _config["weight_decay"]
    config_["model"]["optim_type"] = _config["optim_type"]
    config_["model"]["decay_power"] = _config["decay_power"]
    config_["model"]["warmup_steps"] = _config["warmup_steps"]
    file_ = 'config.yaml'
    stream = open(file_, 'w')
    yaml.safe_dump(config_, stream=stream,default_flow_style=False)
    
