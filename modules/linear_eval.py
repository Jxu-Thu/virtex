import pytorch_lightning as pl
from modules.resnet50 import resnet50
import torch.nn as nn
import torch
from modules.utils import epoch_wrapup_eval_linear, set_schedule_eval_linear
import pytorch_lightning.callbacks.finetuning
from torchmetrics import Accuracy, AverageMeter
import torch.nn.functional as F
class Linear_Eval_Module(pl.LightningModule):
    def __init__(
        self,
        model_path: str = "../results/MMCresult/version_13/checkpoints/epoch=199-step=46199.ckpt",
        num_classes: int = 1000,
        lr: float = 30,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.backbone = resnet50(pretrained_path=model_path, remove_keys="Image_Moco.encoder_q")
        self.linear_classifier = nn.Linear(self.backbone.block.expansion*512, num_classes)
        nn.init.normal_(self.linear_classifier.weight.data, mean=0.0, std=0.01)
        nn.init.constant_(self.linear_classifier.bias.data, 0.0)

    # self.train_loss = AverageMeter()
    # self.test_loss = AverageMeter()
    # self.train_loss = Accuracy()
    # self.test_loss = Accuracy()
        for split in ["train", "val"]:
            setattr(self, f"{split}_loss", AverageMeter())
            setattr(self, f"{split}_acc", Accuracy())
    
    def forward(self, batch):
        x, labels = batch
        feature = self.backbone(x)
        res = self.linear_classifier(feature)
        return res, labels
        
    def training_step(self, batch, batch_idx):
        logit, labels = self(batch)
        loss = F.cross_entropy(logit, labels)
        loss_ = self.train_loss(loss)
        acc = self.train_acc(logit, labels)
        self.log("train/loss", loss_)
        self.log("tran/acc", acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logit, labels = self(batch)
        loss = F.cross_entropy(logit, labels)
        loss_ = self.val_loss(loss)
        acc = self.val_acc(logit, labels)
        self.log("val/loss", loss_)
        self.log("val/acc", acc)
        return loss
    
    def training_epoch_end(self, outs):
        epoch_wrapup_eval_linear(self)
        
    def validation_epoch_end(self, outs):
        epoch_wrapup_eval_linear(self)
    
    def configure_optimizers(self):
        return set_schedule_eval_linear(self)

if __name__ == '__main__':
    model = Linear_Eval_Module()
    # print(model.Text_Moco.named_parameters(recurse=True))
    for n, p in model.named_parameters():
        # if "norm" in n:
        print(n)

