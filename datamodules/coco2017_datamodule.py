from datasets.coco2017_caption_dataset import Coco2017CaptionDataset
from .datamodule_base import BaseDataModule


class CocoCaptionKarpathyDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return Coco2017CaptionDataset

    # @property
    # def dataset_cls_no_false(self):
    #     return coco2017_caption_dataset

    @property
    def dataset_name(self):
        return "ImageNet"
