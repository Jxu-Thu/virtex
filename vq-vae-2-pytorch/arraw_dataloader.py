
import torch
import io
import pyarrow as pa
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

        names = flicks30 + nlvr2


        tables = []
        for name in names:
            pyarrow_table = pa.ipc.RecordBatchFileReader(
                pa.memory_map(f"{data_dir}/{name}.arrow", "r")
            ).read_all()
            tables.append(pyarrow_table)


        print(f'read the dataset from {names}')
        import pdb
        pdb.set_trace()
        self.table = pa.concat_tables(tables, promote=True)


    def __len__(self):
        return len(self.table)

    def get_raw_image(self, index, image_key="image"):
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)
        image_tensor = self.transforms(image)
        return image_tensor

    def __getitem__(self, idx):
        import pdb
        pdb.set_trace()
        image_tensor = self.get_image(idx)
        return image_tensor


