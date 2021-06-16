from glob import glob
from .base_dataset import BaseDataset


class SBUCaptionDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        if split == "test":
            split = "val"

        if split == "train":
            names = [f"sbu_{i}" for i in range(9)]
        elif split == "val":
            names = []

        import pdb
        pdb.set_trace()
        super().__init__(*args, **kwargs, remove_duplicate=False, names=names, text_column_name="caption")

    def __getitem__(self, index):
        return self.get_suite(index)
