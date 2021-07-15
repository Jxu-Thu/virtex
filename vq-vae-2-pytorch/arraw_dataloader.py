
import torch
import io
import pyarrow as pa

import random
import torch
import io
import pyarrow as pa
import os
import numpy as np
from PIL import Image
from vilt.transforms import keys_to_transforms

from PIL import Image
from torchvision import transforms


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        image_size: int,
        transform
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        super().__init__()
        self.transforms = transform
        self.data_dir = data_dir


        # cc3m = [f"conceptual_caption_train_{i}" for i in range(30)] + ["conceptual_caption_val_0"]
        # vg = ["vg"]
        # coco = ["coco_caption_karpathy_train", "coco_caption_karpathy_restval", "coco_caption_karpathy_test"]
        flicks30 = ["f30k_caption_karpathy_train", "f30k_caption_karpathy_val"]
        nlvr2 = ["nlvr2_train", "nlvr2_dev"]
        # sbu = [f"sbu_{i}" for i in range(9)]
        # vqav2 = ["vqav2_train", "vqav2_trainable_val", 'vqav2_rest_val']
        # names = cc3m + vg + coco + flicks30 + nlvr2 + sbu + vqav2

        self.names = flicks30 + nlvr2

        data_lens = []
        tables = []
        for name in self.names:
            pyarrow_table = pa.ipc.RecordBatchFileReader(
                pa.memory_map(f"{data_dir}/{name}.arrow", "r")
            ).read_all()
            tables.append(pyarrow_table)
            data_lens.append(len(pyarrow_table))
        self.tables = tables
        self.data_lens = np.cumsum(data_lens)
        print(f'read the dataset from {self.names}')


    def __len__(self):
        return self.data_lens[-1]

    def get_raw_image(self, dataset_index, data_index, image_key="image"):
        image_bytes = io.BytesIO(self.tables[dataset_index][image_key][data_index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def find_index(self, index):
        dataset_index = np.digitize(index, self.data_lens)
        if dataset_index == 0:
            data_index = index
        else:
            data_index = index - self.data_lens[dataset_index - 1]
        return dataset_index, data_index, self.names[dataset_index]

    def get_image(self, index, image_key="image"):
        dataset_index, data_index, dataset_name = self.find_index(index)
        if 'nlvr' in dataset_name:
            image_key = np.random.choice(['image_0', 'image_1'])
        else:
            image_key = "image"
        image = self.get_raw_image(dataset_index, data_index, image_key=image_key)
        image_tensor = self.transforms(image)
        return image_tensor

    def __getitem__(self, idx):
        image_tensor = self.get_image(idx)
        return image_tensor


