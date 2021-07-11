import os
import copy
import sys
import pytorch_lightning as pl
from vilt.config import ex
from vilt.modules import ViLTransformerSS, MoveMosCKPT
from vilt.datamodules.multitask_datamodule import MTDataModule
# https://tensorboard.dev/experiment/mNHxDM08R6eHKeU0JHn5vg/#scalars&_smoothingWeight=0.75
# https://github.com/PyTorchLightning/pytorch-lightning/issues/2534


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)
    print(_config)

    model = ViLTransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'
    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=exp_name,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    if _config['huawei_target_dir'] is not None:
        moveckpt_callback = MoveMosCKPT(_config["huawei_flag"], _config["huawei_target_dir"])
        callbacks = [checkpoint_callback, lr_callback, moveckpt_callback]
    else:
        callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )
    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    print(f'N_GPUS: {num_gpus}, grad_steps: {grad_steps}')

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    if _config["proxy_dataset_debug"] is True:
        trainer = pl.Trainer(
            gpus=_config["num_gpus"],
            num_nodes=_config["num_nodes"],
            precision=_config["precision"],
            accelerator="ddp",
            benchmark=True,
            deterministic=True,
            max_epochs=_config["max_epoch"] if max_steps is None else 1000,
            max_steps=max_steps,
            callbacks=callbacks,
            logger=logger,
            prepare_data_per_node=False,
            replace_sampler_ddp=False,
            accumulate_grad_batches=grad_steps,
            log_every_n_steps=10,
            flush_logs_every_n_steps=10,
            progress_bar_refresh_rate=5,
            resume_from_checkpoint=_config["resume_from"],
            weights_summary="full",
            fast_dev_run=_config["fast_dev_run"],
            val_check_interval=_config["val_check_interval"],
            limit_train_batches=1,
            limit_val_batches=1
        )
    else:
        trainer = pl.Trainer(
            gpus=_config["num_gpus"],
            num_nodes=_config["num_nodes"],
            precision=_config["precision"],
            accelerator="ddp",
            benchmark=True,
            deterministic=True,
            max_epochs=_config["max_epoch"] if max_steps is None else 1000,
            max_steps=max_steps,
            callbacks=callbacks,
            logger=logger,
            prepare_data_per_node=False,
            replace_sampler_ddp=False,
            accumulate_grad_batches=grad_steps,
            log_every_n_steps=10,
            flush_logs_every_n_steps=10,
            progress_bar_refresh_rate=50,
            resume_from_checkpoint=_config["resume_from"],
            weights_summary="full",
            fast_dev_run=_config["fast_dev_run"],
            val_check_interval=_config["val_check_interval"],
        )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
