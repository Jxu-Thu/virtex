import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Dict
# import typing
# import sys 
# sys.path.append("..")
from modules import utils, objectives
from modules.resnet50 import resnet50
from modules.Moco import MoCo
from transformers import BertModel

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
        is_pretrained: bool = True,
        pretrained_or_config_path: str = "../pretrained_model/"
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
            is_pretrained: whether use pretrained BERT
            pretrained_or_config_path: the path for pretraied model or config file of BERT
        """
        super().__init__()
        self.save_hyperparameters()
        self.Text_Moco = MoCo(BertModel,
                             intra_dim,
                             inter_dim, 
                             hidden_dim,queue_size, 
                             is_pretrained = is_pretrained, 
                             path = pretrained_or_config_path
        )
        self.Image_Moco = MoCo(resnet50, intra_dim, inter_dim, hidden_dim,queue_size)
        utils.set_metrics(self)
        self.current_tasks = list()


    def forward(self, batch):
        # import pdb
        # pdb.set_trace()
        # print_image(batch["image"][0][0],"./hh_0.jpg")
        # print_image(batch["image"][1][0],"./hh_1.jpg")
        # import pdb
        # pdb.set_trace()
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

    # def validation_step(self, batch, batch_idx):
    #     utils.set_task(self)
    #     output = self(batch)

    # def validation_epoch_end(self, outs):
    #     utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return utils.set_schedule(self)

def print_image(x, root):
    #optional
    import torch
    import torchvision
    x = x.cpu()
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    t_mean = torch.FloatTensor(mean).view(3,1,1).expand(3, x.size(1), x.size(2))
    t_std = torch.FloatTensor(std).view(3,1,1).expand(3, x.size(1), x.size(2))
    img_GT = x * t_std +t_mean
    img = torchvision.transforms.ToPILImage()(img_GT).convert('RGB')
    img.save(root)

if __name__ == '__main__':
    model = Mixed_Moco()
    # print(model.Text_Moco.named_parameters(recurse=True))
    for n, p in model.named_parameters():
        if "norm" in n:
            print(n)