"""
============================================================================================================

============================================================================================================
"""

"""Define a generic class for training and testing learning algorithms."""
import json
import datetime
import glob
import logging
import os
import os.path
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from tqdm import tqdm
import architectures
import utils
import math



class Algorithm:
    def __init__(self, opt, _run=None, _log=None):
        self.set_experiment_dir(pathlib.Path(opt["exp_dir"]))
        self.exp_name = self.exp_dir.name
        self.set_log_file_handler(_log)
        self.global_steps = 0
        self.logger.info(f"Algorithm options {opt}")
        self.opt = opt
        self._run = _run
        self.init_all_networks()
        self.init_all_criterions()
        self.allocate_tensors()
        self.curr_epoch = 0
        self.optimizers = {}
        self.is_train = True

        self.keep_best_model_metric_name = (
            opt["best_metric"] if ("best_metric" in opt) else None
        )
        

    def set_experiment_dir(self, directory_path):
        self.exp_dir = directory_path
        os.makedirs(self.exp_dir, exist_ok=True)

        self.vis_dir = directory_path / "visuals"
        os.makedirs(self.vis_dir, exist_ok=True)

        self.preds_dir = directory_path / "preds"
        os.makedirs(self.preds_dir, exist_ok=True)

    def set_log_file_handler(self, _log=None):
        if _log is not None:
            self.logger = _log
        else:
            self.logger = logging.getLogger(__name__)

            str_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s"
            )
            str_handler.setFormatter(formatter)
            self.logger.addHandler(str_handler)
            self.logger.setLevel(logging.INFO)

        log_dir = self.exp_dir / "logs"
        os.makedirs(log_dir, exist_ok=True)

        now_str = datetime.datetime.now().__str__().replace(" ", "_")
        now_str = now_str.replace(" ", "_").replace("-", "").replace(":", "")

        self.log_file = log_dir / f"LOG_INFO_{now_str}.txt"
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.logger.addHandler(self.log_fileHandler)



    def init_all_networks(self):
        networks_defs = self.opt["networks"]
        self.networks = {}
        self.optim_params = {}

        for key, val in networks_defs.items():
            self.logger.info(f"Set network {key}")
            def_file = val["def_file"]
            net_opt = val["opt"]
            self.optim_params[key] = val["optim_params"] if ("optim_params" in val) else None
            pretrained_path = val["pretrained"] if ("pretrained" in val) else None
            force = val["force"] if ("force" in val) else False
            self.networks[key] = self.init_network(
                def_file, net_opt, pretrained_path, key, force=force
            )

    def init_network(self, net_def_file, net_opt, pretrained_path, key, force=False):
        self.logger.info(
            f"==> Initiliaze network {key} from " f"file {net_def_file} with opts: {net_opt}"
        )
        architecture_name = os.path.basename(net_def_file).split(".py")[0]
        network = architectures.factory(architecture_name=architecture_name, opt=net_opt)
        self.logger.info("==> Network architecture:")
        self.logger.info(network)
        if pretrained_path is not None:
            self.load_pretrained(network, pretrained_path, force=force)

        return network

    def init_all_optimizers(self):
        self.optimizers = {}
        for key, oparams in self.optim_params.items():
            self.optimizers[key] = None
            if oparams is not None:
                self.optimizers[key] = self.init_optimizer(self.networks[key], oparams, key)

    def init_optimizer(self, net, optim_opts, key):
        optim_type = optim_opts["optim_type"]
        learning_rate = optim_opts["lr"]
        optimizer = None
        self.logger.info(
            f"Initialize optimizer: {optim_type} "
            f"with params: {optim_opts} "
            f"for netwotk: {key}"
        )
        if optim_type == "RMSprop":
            optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, momentum=0.5)
        elif optim_type == "adam":
            weight_decay = (
                optim_opts["weight_decay"] if ("weight_decay" in optim_opts) else 0.0
            )
            optimizer = torch.optim.Adam(
                parameters,
                lr=learning_rate,
                betas=optim_opts["beta"],
                weight_decay=weight_decay,
            )

            if 'beta' in optim_opts:
                weight_decay = (
                   optim_opts['weight_decay']
                   if ('weight_decay' in optim_opts) else 0.0)
                optimizer = torch.optim.Adam(parameters, lr=learning_rate,
                   betas=optim_opts['beta'], weight_decay=weight_decay)
            elif 'weight_decay' in optim_opts:
                optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                   weight_decay=optim_opts['weight_decay'])
            else:
                optimizer = torch.optim.Adam(net.parameters())
        elif optim_type == "sgd":
            optimizer = torch.optim.SGD(
                net.parameters(),
                lr=learning_rate,
                momentum=optim_opts["momentum"],
                nesterov=optim_opts["nesterov"] if ("nesterov" in optim_opts) else False,
                weight_decay=optim_opts["weight_decay"],
            )
        else:
            raise ValueError("Not supported or recognized optim_type", optim_type)

        return optimizer

    def init_all_criterions(self):
        self.criterions = {}
        criterions_defs = self.opt.get("criterions")
        if criterions_defs is not None:
            for key, val in criterions_defs.items():
                crit_type = val["ctype"]
                crit_opt = val["opt"] if ("opt" in val) else None
                self.logger.info(
                    f"Initialize criterion[{key}]: {crit_type} " f"with options: {crit_opt}"
                )
                self.criterions[key] = self.init_criterion(crit_type, crit_opt)

    def init_criterion(self, ctype, copt):
        if copt is None:
            return getattr(nn, ctype)()
        else:
            return getattr(nn, ctype)(copt)

    def load_to_gpu(self, num_gpu):
        if self.opt["is_distributed"]:
            torch.cuda.set_device(self.opt["gpu_id"])
            self.logger.info(f"==> Loading networks to device: {self.opt['gpu_id']}")
            for key, net in self.networks.items():
                if key == "encoder":
                    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
                net.cuda(self.opt["gpu_id"])
                if key == "classifier":
                    self.networks[key] = self.networks[key].cuda(self.opt["gpu_id"])
                else:
                    self.networks[key] = torch.nn.parallel.DistributedDataParallel(self.networks[key], device_ids=[self.opt["gpu_id"]])
            self.logger.info(f"==> Loading criterions to device: {self.opt['gpu_id']}")
            for key, criterion in self.criterions.items():
                self.criterions[key] = criterion.cuda(self.opt["gpu_id"])
            self.logger.info(f"==> Loading tensors to device: {self.opt['gpu_id']}")
            for key, tensor in self.tensors.items():
                self.tensors[key] = tensor.cuda(self.opt["gpu_id"])
        else:
            self.device = torch.device("cuda")
            self.logger.info(f"==> Loading criterions to device: {self.device}")
            for key, criterion in self.criterions.items():
                self.criterions[key] = criterion.to(self.device)

            self.logger.info(f"==> Loading tensors to device: {self.device}")
            for key, tensor in self.tensors.items():
                self.tensors[key] = tensor.to(self.device)

            self.logger.info(f"==> Loading networks to device: {self.device}")
            for key, net in self.networks.items():
                self.networks[key] = net.to(self.device)

    def save_checkpoint(self, epoch, suffix="", metric=None):
        if self.opt["is_distributed"]:
            if self.opt["gpu_id"] == 0:
                for key, net in self.networks.items():
                    if self.optimizers[key] is None:
                        continue
                    self.save_network(key, epoch, suffix=suffix, metric=metric)
                    self.save_optimizer(key, epoch, suffix=suffix)
        else:
            for key, net in self.networks.items():
                if self.optimizers[key] is None:
                    continue
                self.save_network(key, epoch, suffix=suffix, metric=metric)
                self.save_optimizer(key, epoch, suffix=suffix)
                
    def find_most_recent_epoch(self, key, suffix, other_wd = None):
        search_patern = self._get_net_checkpoint_filename(key, "*", other_wd) + suffix
        all_files = glob.glob(search_patern)
        if len(all_files) == 0:
            raise ValueError(f"{search_patern}: no such file.")

        substrings = search_patern.split("*")
        assert len(substrings) == 2
        start, end = substrings
        all_epochs = [fname.replace(start, "").replace(end, "") for fname in all_files]
        all_epochs = [int(epoch) for epoch in all_epochs if epoch.isdigit()]
        assert len(all_epochs) > 0
        all_epochs = sorted(all_epochs)
        most_recent_epoch = int(all_epochs[-1])
        self.logger.info("Load checkpoint of most recent epoch %s" % str(most_recent_epoch))
        return most_recent_epoch

    def load_checkpoint(self, epoch, train=True, suffix="", other_wd = None, no_load = ""):
        self.logger.info(f"Load checkpoint of epoch {epoch}")
        if self.opt["is_distributed"] is None:
            for key, net in self.networks.items():  # Load networks
                if key == no_load:
                    continue
                if epoch == "*":
                    epoch = self.find_most_recent_epoch(key, suffix, other_wd)

                self.load_network(key, epoch, suffix, other_wd =other_wd)

            if train:  # initialize and load optimizers
                self.init_all_optimizers()
                for key, net in self.networks.items():
                    if self.optim_params[key] is None:
                        continue
                    self.load_optimizer(key, epoch, suffix, other_wd =other_wd)
        else:

            loc = 'cuda:{}'.format(self.opt["gpu_id"])
            for key, net in self.networks.items():  # Load networks
                if self.optim_params[key] is None:
                    continue
                if epoch == "*":
                    epoch = self.find_most_recent_epoch(key, suffix)
                self.load_network(key, epoch,suffix, loc)

            if train:  # initialize and load optimizers
                self.init_all_optimizers()
                for key, net in self.networks.items():
                    if self.optim_params[key] is None:
                        continue
                    self.load_optimizer(key, epoch,suffix, loc)
            loc = 'cuda:{}'.format(self.opt["gpu_id"])

        self.curr_epoch = epoch
        return epoch

    def delete_checkpoint(self, epoch, suffix=""):
        if self.opt["is_distributed"]:
            if self.opt["gpu_id"] == 0:
                for key, net in self.networks.items():
                    if self.optimizers[key] is None:
                        continue

                    filename_net = pathlib.Path(self._get_net_checkpoint_filename(key, epoch) + suffix)
                    if filename_net.is_file():
                        os.remove(filename_net)

                    filename_optim = pathlib.Path(
                        self._get_optim_checkpoint_filename(key, epoch) + suffix
                    )
                    if filename_optim.is_file():
                        os.remove(filename_optim)
        else:
            for key, net in self.networks.items():
                if self.optimizers[key] is None:
                    continue

                filename_net = pathlib.Path(self._get_net_checkpoint_filename(key, epoch) + suffix)
                if filename_net.is_file():
                    os.remove(filename_net)

                filename_optim = pathlib.Path(
                    self._get_optim_checkpoint_filename(key, epoch) + suffix
                )
                if filename_optim.is_file():
                    os.remove(filename_optim)

    def save_network(self, net_key, epoch, suffix="", metric=None):
        assert net_key in self.networks
        filename = self._get_net_checkpoint_filename(net_key, epoch) + suffix
        state = {
            "epoch": epoch,
            "network": self.networks[net_key].state_dict(),
            "metric": metric,
        }
        torch.save(state, filename)

    def save_optimizer(self, net_key, epoch, suffix=""):
        assert net_key in self.optimizers
        filename = self._get_optim_checkpoint_filename(net_key, epoch) + suffix
        state = {"epoch": epoch, "optimizer": self.optimizers[net_key].state_dict()}
        torch.save(state, filename)

    def load_network(self, net_key, epoch, suffix="", loc=None, other_wd = None):
        assert net_key in self.networks
        print("original "+ net_key)
        filename = pathlib.Path(self._get_net_checkpoint_filename(net_key, epoch, other_wd) + suffix)
        self.logger.info(f"Loading {filename} for network {net_key}")
        assert filename.is_file()
        if loc:
            checkpoint = torch.load(filename, map_location=loc)
        else:
            checkpoint = torch.load(filename)
        if "network" in checkpoint:
            net = checkpoint["network"]
        elif "state_dict" in checkpoint:
            net = checkpoint["state_dict"]
        elif "model" in checkpoint:
            net = checkpoint["model"]
        else:
            net = checkpoint
        net_keys = list(net.keys())
        if self.opt["is_distributed"] and not self.opt["load_from_multi"]:
            for i, key in enumerate(net_keys):
                newkey = "module." + key 
                net[newkey] = net.pop(key)
        net_keys = list(net.keys())
        if self.opt["load_from_multi"] and self.opt["is_distributed"] is None:
            for i, key in enumerate(net_keys):
                newkey = key.replace("module.", "")
                net[newkey] = net.pop(key)
        net_keys = list(net.keys())
        if self.opt["load_from_contrastive"]:
            for i, key in enumerate(net_keys):
                if "fc" in key:
                    net.pop(key)
                elif "encoder_q" in key:
                    newkey = key.replace("encoder_q.","")
                    net[newkey] = net.pop(key)
                else:
                    net.pop(key)
        
        self.networks[net_key].load_state_dict(net)

    def load_optimizer(self, net_key, epoch, suffix="", loc=None, other_wd = None):
        assert net_key in self.optimizers
        filename = pathlib.Path(self._get_optim_checkpoint_filename(net_key, epoch, other_wd) + suffix)
        self.logger.info(f"Loading {filename} for network {net_key}")
        assert filename.is_file()
        if loc:
            checkpoint = torch.load(filename, map_location=loc)
        else:
            checkpoint = torch.load(filename)
        self.optimizers[net_key].load_state_dict(checkpoint["optimizer"])

    def _get_net_checkpoint_filename(self, net_key, epoch, other_wd = None):
        if other_wd is not None:
            return str(other_wd / f"{net_key}_net_epoch{epoch}")
        return str(self.exp_dir / f"{net_key}_net_epoch{epoch}")
    
    def _get_optim_checkpoint_filename(self, net_key, epoch, other_wd = None):
        if other_wd is not None:
            return str(other_wd / f"{net_key}_optim_epoch{epoch}")
        else:
            return str(self.exp_dir / f"{net_key}_optim_epoch{epoch}")


    def solve(self, data_loader_train, train_sampler, data_loader_test=None):
                self.data_loader_test = data_loader_test
        training_loss = []
        testing_loss = []
        # if os.path.exists(self.opt["exp_dir"] +'/logs/train_loss.json'):
        #     # print("hhhhhhhhhhhhhhhh")
        #     with open(self.opt["exp_dir"] +'/logs/train_loss.json', 'r') as f:
        #         training_loss = json.load(f)
        #     with open(self.opt["exp_dir"] +'/logs/test_loss.json', 'r') as f:
        #         testing_loss = json.load(f)
        self.max_num_epochs = self.opt["max_num_epochs"]
            
        start_epoch = self.curr_epoch
        if len(self.optimizers) == 0:
            self.init_all_optimizers()
            
        self.init_record_of_best_model()#self.max_metric_val = None,self.best_stats = None,self.best_epoch = None
        for self.curr_epoch in range(start_epoch, self.max_num_epochs):
            #Shuffle the dataset when multi-gpu mode is on
            if train_sampler is not None:
                train_sampler.set_epoch(self.curr_epoch)
            self.logger.info(
                "Training epoch [%3d / %3d]" % (self.curr_epoch + 1, self.max_num_epochs)
            )
            if not self.opt["cosine_lr"]:
                self.adjust_learning_rates(self.curr_epoch, cosine = False)
            #epoch loop
            train_stats, train_loss = self.run_train_epoch(data_loader_train, self.curr_epoch)
            training_loss.append(train_loss)
                
            print(f"trianloss:{train_loss}")
                
            self.logger.info("==> Training stats: %s" % train_stats)

            self.save_checkpoint(
                self.curr_epoch + 1
            )  # create a checkpoint in the current epoch
                
            if start_epoch != self.curr_epoch and self.curr_epoch < self.opt["save_model_epoch"]:  # delete the checkpoint of the previous epoch
                self.delete_checkpoint(self.curr_epoch)
            #val_time
            if self.curr_epoch % self.opt["val_interval"] == self.opt["val_interval"]-1 and data_loader_test is not None:# and self.curr_epoch % 4 == 3:
                #val
                eval_stats, test_loss = self.evaluate(data_loader_test, self.curr_epoch)
                # eval_stats = self.evaluate(data_loader_test)
                self.logger.info("==> Evaluation stats: %s" % eval_stats)
                self.keep_record_of_best_model(eval_stats, self.curr_epoch)
                testing_loss.append(test_loss)
                print(f"trianloss:{test_loss}")
                with open(self.opt["exp_dir"] +'/logs/test_loss.json', 'w') as f:
                    json.dump(testing_loss, f)
                self.print_eval_stats_of_best_model()
            with open(self.opt["exp_dir"] +'/logs/train_loss.json', 'w') as f:
                json.dump(training_loss, f)
            # self.plot_loss(training_loss, testing_loss)
                
    def run_train_epoch(self, data_loader, epoch):
        #epoch loop
        self.epoch = epoch
        self.logger.info(f"Training: {self.exp_name}")
        self.dloader = data_loader

        for key, network in self.networks.items():
            if self.optimizers[key] is None:
                network.eval()
            else:
                network.train()

        dname = ""

        if "name" in self.dloader.dataset.__dict__.keys():
            dname = self.dloader.dataset.name

        self.logger.info(f"==> Dataset: {dname} [{len(self.dloader)} batches]")

        disp_step = self.opt["display_step"] if ("display_step" in self.opt) else 50

        train_stats = utils.DAverageMeter("train", self._run)
        self.bnumber = len(data_loader)
        train_loss = 0.
        #batch loop
        for idx, batch in enumerate(tqdm(data_loader)):
            print(f"global_steps/total_steps: {self.global_steps}/{self.total_steps}")
            self.adjust_learning_rates()
            self.biter = idx  # batch iteration.
            train_stats_this = self.train_step(batch)

            train_loss = train_loss + train_stats_this["accuracy"]
            
            train_stats.update(train_stats_this)
            with torch.no_grad():
                if (idx + 1) % disp_step == 0:
                    self.logger.info(
                        "==> Iteration [%3d][%4d / %4d]: %s"
                        % (epoch + 1, idx + 1, len(data_loader), train_stats)
                    )
            self.global_steps = self.global_steps + 1
            
        train_loss = train_loss / len(data_loader)
        if self.opt["is_distributed"]:
            train_loss = concat_all_gather(torch.tensor(train_loss).cuda(self.opt["gpu_id"])).item()
        train_stats.log()

        return train_stats, train_loss

    def evaluate(self, dloader, idx=None):
        self.logger.info(f"Evaluating: {self.exp_name}")

        self.dloader = dloader
        self.logger.info("==> Dataset: %s [%d batches]" % (self.opt["data_test_opt"]["dataset_name"], len(dloader)))
        for key, network in self.networks.items():
            network.eval()

        if idx is not None:
            meter_name = f"eval_{idx}"
        else:
            meter_name = "eval"
        eval_stats = utils.DAverageMeter(meter_name, self._run)
        self.bnumber = len(dloader)
        test_loss = 0.
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dloader)):
                self.biter = idx
                eval_stats_this = self.evaluation_step(batch)
                test_loss = test_loss + eval_stats_this["accuracy"]
                eval_stats.update(eval_stats_this)
        test_loss = test_loss / len(dloader)
        if self.opt["is_distributed"]:
            test_loss = concat_all_gather(torch.tensor(test_loss).cuda(self.opt["gpu_id"])).item()

        self.logger.info("==> Results: %s" % eval_stats)
        eval_stats.log()

        return eval_stats, test_loss


    def adjust_learning_rates(self):
        optim_params_filtered = {
                k: v for k, v in self.optim_params.items() if (v is not None)
            }
        for key, oparams in optim_params_filtered.items():
            if self.opt["warm_up"] and self.global_steps <= self.warm_up_steps:
                lr = self.opt["initial_lr"]*(self.global_steps/self.warm_up_steps)
            elif self.opt["warm_up"]:
                lr = 0.5 * self.opt["initial_lr"]* (1. + math.cos(math.pi * (self.global_steps-self.warm_up_steps) / (self.total_steps-self.warm_up_steps)))
            else:
                lr = 0.5 * self.opt["initial_lr"]* (1. + math.cos(math.pi * self.global_steps / self.total_steps))
            for param_group in self.optimizers[key].param_groups:
                if key == "pretrain_classifier":
                    param_group["lr"] = self.opt["lr_num"]*lr
                else:
                    param_group["lr"] = lr
        print(f"lr adjusts to:{lr}")

    def init_record_of_best_model(self):
        self.max_metric_val = None
        self.best_stats = None
        self.best_epoch = None

    def keep_record_of_best_model(self, eval_stats, current_epoch):
        if self.keep_best_model_metric_name is not None:
            metric_name = self.keep_best_model_metric_name
            if metric_name not in eval_stats.values:
                raise ValueError(
                    f"The provided metric {metric_name} for keeping the best "
                    "model is not computed by the evaluation routine."
                )
            metric_val = eval_stats.average()[metric_name]
            if self.opt["is_distributed"]:
                a = torch.tensor(metric_val).cuda(self.opt["gpu_id"])
                metric_val = concat_all_gather(a).item()
            if self.max_metric_val is None or metric_val > self.max_metric_val:
                self.max_metric_val = metric_val
                self.best_stats = eval_stats
                self.save_checkpoint(
                    self.curr_epoch + 1, suffix=".best", metric=self.max_metric_val
                )
                if self.best_epoch is not None:
                    self.delete_checkpoint(self.best_epoch + 1, suffix=".best")
                self.best_epoch = current_epoch

    def print_eval_stats_of_best_model(self):
        if self.best_stats is not None:
            metric_name = self.keep_best_model_metric_name
            self.logger.info(
                f"==> Best results w.r.t. {metric_name} "
                f"metric: epoch: {self.best_epoch+1} - {self.best_stats}"
                
            )
            self.logger.info(
            f"{self.max_metric_val}"
            )
    def test_(self, dloader_test):
        #for test mode
        eval_stats, _ = self.evaluate(dloader_test[0])
        metric = eval_stats.average()[self.keep_best_model_metric_name]
        if self.opt["is_distributed"]:
            a = torch.tensor( metric).cuda(self.opt["gpu_id"])
            metric_test=concat_all_gather(a).item()
        else:
            metric_test=metric
        self.logger.info(
            f"mean: {metric_test}"
        )
    # FROM HERE ON ARE ABSTRACT FUNCTIONS THAT MUST BE IMPLEMENTED BY THE CLASS
    # THAT INHERITS THE Algorithms CLASS
    def train_step(self, batch, aux_iter = None):
        """Implements a training step that includes:
            * Forward a batch through the network(s)
            * Compute loss(es)
            * Backward propagation through the networks
            * Apply optimization step(s)
            * Return a dictionary with the computed losses and any other desired
                stats. The key names on the dictionary can be arbitrary.
        """
        pass

    def evaluation_step(self, batch):
        """Implements an evaluation step that includes:
            * Forward a batch through the network(s)
            * Compute loss(es) or any other evaluation metrics.
            * Return a dictionary with the computed losses the evaluation
                metrics for that batch. The key names on the dictionary can be
                arbitrary.
        """
        pass

    def allocate_tensors(self):
        """(Optional) allocate torch tensors that could potentially be used in
            in the train_step() or evaluation_step() functions. If the
            load_to_gpu() function is called then those tensors will be moved to
            the gpu device.
        """
        self.tensors = {}

@torch.no_grad()
def concat_all_gather(tensor):


    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensor.unsqueeze_(0)
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.mean(torch.cat(tensors_gather))
    return output


