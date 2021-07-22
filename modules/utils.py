from torchmetrics import Accuracy, AverageMeter
import torch
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from torch.optim.lr_scheduler import MultiStepLR
def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.loss_names.items():
            if v < 1:
                continue
            setattr(pl_module, f"{split}_{k}_loss", AverageMeter())
            if k == "IC" or k == "TC":
                setattr(pl_module, f"{split}_{k}_acc1", Accuracy())
                setattr(pl_module, f"{split}_{k}_acc5", Accuracy(top_k=5))
                

def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.loss_names.items() if v >= 1
    ]

def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    the_metric = 0
    for loss_name, v in pl_module.hparams.loss_names.items():
        if v < 1:
            continue
        value = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
        pl_module.log(f"{loss_name}/{phase}/loss_epoch", value)
        getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        the_metric += value
        if loss_name == "IC" or loss_name == "TC":
            value = getattr(pl_module, f"{phase}_{loss_name}_acc1").compute()
            pl_module.log(f"{loss_name}/{phase}/acc1_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_acc1").reset()
            value = getattr(pl_module, f"{phase}_{loss_name}_acc5").compute()
            pl_module.log(f"{loss_name}/{phase}/acc5_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_acc5").reset()
    pl_module.log(f"{phase}/the_metric", the_metric)

def epoch_wrapup_eval_linear(pl_module):
    phase = "train" if pl_module.training else "val"
    value = getattr(pl_module, f"{phase}_loss").compute()
    pl_module.log(f"{phase}/loss_epoch", value)
    getattr(pl_module, f"{phase}_loss").reset()
    value = getattr(pl_module, f"{phase}_acc").compute()
    pl_module.log(f"{phase}/acc_epoch", value)
    getattr(pl_module, f"{phase}_acc").reset()

def set_schedule_eval_linear(pl_module):
    lr = pl_module.hparams.lr
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, pl_module.parameters()),lr=lr, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[30,40,50], gamma=0.2)
    
    return [optimizer], [scheduler]


def set_schedule(pl_module):
    lr_image = pl_module.hparams.lr_image
    lr_text = pl_module.hparams.lr_text
    wd = pl_module.hparams.weight_decay
    decay_power = pl_module.hparams.decay_power
    no_decay = [
        "LayerNorm.bias",
        "LayerNorm.weight",
        "bn.bias",
        "bn.weight",
    ]
    optim_type = pl_module.hparams.optim_type
    names = [n for n, p in pl_module.named_parameters()]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and "Image" in n
            ],
            "weight_decay": wd,
            "lr": lr_image,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and "Image" in n
            ],
            "weight_decay": 0.0,
            "lr": lr_image,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and "Text" in n
            ],
            "weight_decay": wd,
            "lr": lr_text,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and "Text" in n
            ],
            "weight_decay": 0.0,
            "lr": lr_text,
        }]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, momentum=0.9)
    gpu_num = len(pl_module.trainer.gpus) if isinstance(pl_module.trainer.gpus,list) else pl_module.trainer.gpus
    if pl_module.trainer.max_steps is None:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // (pl_module.trainer.accumulate_grad_batches*gpu_num)
        )
    else:
        max_steps = pl_module.trainer.max_steps
    print(max_steps)
    warmup_steps = pl_module.hparams.warmup_steps
    if isinstance(pl_module.hparams.warmup_steps, float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )
    sched = {"scheduler": scheduler, "interval": "step"}
    return (
        [optimizer],
        [sched],
    )
