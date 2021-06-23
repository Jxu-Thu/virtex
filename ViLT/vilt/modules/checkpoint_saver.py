import pytorch_lightning as pl
import time
import os
import shutil
class MoveMosCKPT(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(self, huawei_flag, target_dir):
        self.huawei_flag = huawei_flag
        self.target_dir = target_dir


    def on_epoch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        import pdb
        pdb.set_trace()
        source_dir = trainer.logger.log_dir
        strings = time.strftime("%Y,%m,%d,%H")
        t = strings.split(',')
        current_path = '-'.join([str(x) for x in t])
        target_dir = os.path.join(self.target_dir, current_path)
        if self.huawei_flag:
            import moxing as mox
            mox.file.copy_parallel(source_dir, target_dir)
        else:
            shutil.copytree(source_dir, target_dir)
