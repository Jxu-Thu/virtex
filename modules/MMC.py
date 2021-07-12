import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Dict
# import sys 
# sys.path.append("..")
from modules import utils, objectives
from modules.resnet50 import resnet50
from modules.Moco import MoCo
from transformers import BertModel, BertConfig

class Mixed_Moco(pl.LightningModule):
    def __init__(
        self,
        alpha: float = 0.2,
        loss_names: Dict = {"IC":1,"TC":1,"I2TC":1,"T2IC":1},
        lambda_: Dict = {"IC_loss":1.0,"TC_loss":1.0,"I2TC_loss":0.0001,"T2IC_loss":0.0001},
        intra_dim: int = 128,
        inter_dim: int = 1024,
        hidden_dim: int = 2048,
        queue_size: int = 32768,
        lr_image: float = 0.03,
        lr_text: float = 4e-5,
        weight_decay: float = 1e-4,
        optim_type: str = "sgd",
        decay_power: str = "cosine",
        warmup_steps: int = 0,       
    ) -> None:
        """Multi-Modal Contrastive module
        Args:
            alpha: the margin of inter contrastive learning losses
            loss_names: loss names for training
            lambda_: weights of different losses
            intra_dim: The feature dim of intra contrastive learning
            inter_dim: The feature dim of inter contrastive learning
            hidden_dim: The dim of hidden layer of MLPs
            queue_size: The maximum size of each queue
            lr_image: learning rate for image feature extractor
            lr_text: learning rate for text feature extractor
            weight_decay: the coffecient of weight decay
            optim_type: optimizer category
            decay_power: the way of decaying
            warmup_steps: num of steps for warmup
        """
        super().__init__()
        self.save_hyperparameters()
        bert_config = BertConfig()
        self.Text_Moco = MoCo(BertModel, intra_dim, inter_dim, hidden_dim,queue_size,config = bert_config)
        self.Image_Moco = MoCo(resnet50, intra_dim, inter_dim, hidden_dim,queue_size)
        utils.set_metrics(self)
        self.current_tasks = list()


    def forward(self, batch):
        ret = dict()
        #image
        if "IC" in self.current_tasks:
            ret.update(objectives.image_contrastive(self, batch))

        if "TC" in self.current_tasks:
            ret.update(objectives.text_contrastive(self, batch)) 

        if "I2TC" in self.current_tasks and "T2IC" in self.current_tasks:
            ret.update(objectives.mixed_contrastive(self, batch))
        return ret

    def training_step(self, batch, batch_idx):
        utils.set_task(self)
        output = self(batch)
        total_loss = sum([self.hparams.lambda_[k]*v for k, v in output.items() if "loss" in k])
        return total_loss

    def training_epoch_end(self, outs):
        utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return utils.set_schedule(self)


if __name__ == '__main__':
    model = Mixed_Moco()
    # print(model.Text_Moco.named_parameters(recurse=True))
    for n, p in model.named_parameters():
        if "norm" in n:
            print(n)