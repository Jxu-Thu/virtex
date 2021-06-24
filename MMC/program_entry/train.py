



import torch
import numpy as np
import random
import sys 
sys.path.append("..")
os.environ['MASTER_PORT'] = '6008'
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_arguments():
    """ 
    Parse input arguments. 
    return:
        args_opt:argument from keyboard
        exp_directory: derectory for experimental results 
        config: a dictinary of all configurations
    """
    parser = argparse.ArgumentParser(
        description="Code for Multimodal Contrastive Training for Visual Representation Learning."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="",
        help="config file with parameters of the experiment.",
    )
    parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=0,
        help="checkpoint (epoch id) that will be loaded. If a negative value "
        "is given then the latest existing checkpoint is loaded.",
    )
    parser.add_argument(
        "--time", type = str, required=True, help="name of the experiment"
    )
    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
    args_opt = parser.parse_args()
    exp_base_directory = os.path.join(project_root, "results")#会自动在中间加/符号
    exp_directory = os.path.join(exp_base_directory, args_opt.config) + "/" + args_opt.time

    # Load the configuration params of the experiment
    #configuration来自config文件夹，录入exp_directory中
    exp_config_file = "config." + args_opt.config.replace("/", ".")#字符串自带替换函数，将/全部替换为.
    print(f"Loading experiment {args_opt.config}")#f表示格式化输出
    #得到目标config文件的config字典
    config = __import__(exp_config_file, fromlist=[""]).config#__import__() 函数用于动态加载类和函数  fromlist (Optional): 被导入的 submodule 名称 
    config["exp_dir"] = exp_directory  # where logs, models, etc will be stored.
    print(f"Generated logs and/or snapshots will be stored on {exp_directory}")
    return args_opt, exp_directory, config

def main():
    args_opt, exp_directory, config = get_arguments()

    if config["is_distributed"] is not None:
        main_worker(args_opt, exp_directory, config,  gpu_id = args_opt.local_rank)
    else:
        main_worker(args_opt, exp_directory, config)


def main_worker(args_opt, exp_directory, config, gpu_id = None):
    """
    The entry program to process the whole code.
    input:
        gpu_id: gpu id of current process
    """
    if config["is_distributed"] is not None:
        #set up gpu id of current process
        config["gpu_id"] = gpu_id
        config["networks"]["encoder"]["opt"]["gpu_id"] = gpu_id
        print("Use GPU: {} for training".format(config["gpu_id"]))
        #set up distribution
        dist.init_process_group(backend=args_opt.dist_backend)
        setup_seed(50 * config["gpu_id"])
    else:
        print("Use one GPU")
        setup_seed(10)

    if args_opt.checkpoint != 0:
        algorithm.load_checkpoint(
            epoch=args_opt.checkpoint if (args_opt.checkpoint > 0) else "*", train=True
        )

    