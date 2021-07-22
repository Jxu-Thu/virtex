from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transforms import keys_to_transforms
from typing import List
import os
class ImageNetDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str = "../../public_data/compression/data",
        num_workers: int = 8,
        per_gpu_batchsize: int = 32,
        image_size: int = 224,
        train_transform_keys: List = ["Linear_eval_train"],
        val_transform_keys: List = ["Linear_eval_val"],
        ):
        super().__init__()
        self.train_data_dir = os.path.join(data_root, 'train')
        # print(os.path.isdir(self.train_data_dir))
        self.val_data_dir = os.path.join(data_root, 'val')
        self.train_transform = keys_to_transforms(train_transform_keys, size=image_size)[0]
        self.val_transform = keys_to_transforms(val_transform_keys, size=image_size)[0]
        self.batch_size = per_gpu_batchsize
        self.num_workers = num_workers

    def set_train_dataset(self):
        self.train_dataset = ImageFolder(
            self.train_data_dir,
            self.train_transform
        )
    def set_val_dataset(self):
        self.val_dataset = ImageFolder(
            self.val_data_dir,
            self.val_transform
        )

    def setup(self, stage):
        self.set_val_dataset()
        self.set_train_dataset()

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader
    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader
    # @property
    # def dataset_cls_no_false(self):
    #     return coco2017_caption_dataset

    @property
    def dataset_name(self):
        return "ImageNet"

if __name__ == "__main__":
    data = ImageNetDataModule()
    data.set_train_dataset()
    print(len(data.train_dataset.imgs))
    import pdb
    pdb.set_trace()