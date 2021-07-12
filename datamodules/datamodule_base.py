import torch
import functools
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)
from typing import List


def get_pretrained_tokenizer(from_pretrained, huawei_flag=False):
    if not huawei_flag:
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                BertTokenizer.from_pretrained(
                    from_pretrained, cache_dir='../data/totenizer', do_lower_case="uncased" in from_pretrained
                )
            torch.distributed.barrier()
        return BertTokenizer.from_pretrained(
            from_pretrained, cache_dir='../data/totenizer', do_lower_case="uncased" in from_pretrained
        )
    else:
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                BertTokenizer.from_pretrained(
                    from_pretrained, cache_dir='/cache/VilT_dataset/bert-base-uncased', do_lower_case=True
                )
            torch.distributed.barrier()
        return BertTokenizer.from_pretrained(
            from_pretrained, cache_dir='/cache/VilT_dataset/bert-base-uncased', do_lower_case=True
        )



class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str = "../../data/coco2017",
        num_workers: int = 8,
        per_gpu_batchsize: int = 32,
        image_size: int = 224,
        max_text_len: int = 40,
        image_only: bool = False,
        train_transform_keys: List = [],
        val_transform_keys: List = [],
        tokenizer: str = "bert-base-uncased",
        huawei_flag: bool = False,
        whole_word_masking: bool = True,
        
    ):
        super().__init__()

        self.data_dir = data_root

        self.num_workers = num_workers
        self.batch_size = per_gpu_batchsize
        self.eval_batch_size = self.batch_size

        self.image_size = image_size
        self.max_text_len = max_text_len
        self.image_only = image_only

        self.train_transform_keys = (
            ["default_train"]
            if len(train_transform_keys) == 0
            else train_transform_keys
        )

        self.val_transform_keys = (
            ["default_val"]
            if len(val_transform_keys) == 0
            else val_transform_keys
        )


        self.tokenizer = get_pretrained_tokenizer(tokenizer, huawei_flag)
        self.vocab_size = self.tokenizer.vocab_size

        collator = (
            DataCollatorForWholeWordMask
            if whole_word_masking
            else DataCollatorForLanguageModeling
        )

        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=True#, mlm_probability=mlm_prob
        )
        self.setup_flag = False

    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            self.train_transform_keys,
            split="train",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            # draw_false_image=self.draw_false_image,
            # draw_false_text=self.draw_false_text,
            image_only=self.image_only,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            # draw_false_image=self.draw_false_image,
            # draw_false_text=self.draw_false_text,
            image_only=self.image_only,
        )

        # if hasattr(self, "dataset_cls_no_false"):
        #     self.val_dataset_no_false = self.dataset_cls_no_false(
        #         self.data_dir,
        #         self.val_transform_keys,
        #         split="val",
        #         image_size=self.image_size,
        #         max_text_len=self.max_text_len,
        #         # draw_false_image=0,
        #         # draw_false_text=0,
        #         image_only=self.image_only,
        #     )

    # def make_no_false_val_dset(self, image_only=False):
    #     return self.dataset_cls_no_false(
    #         self.data_dir,
    #         self.val_transform_keys,
    #         split="val",
    #         image_size=self.image_size,
    #         max_text_len=self.max_text_len,
    #         draw_false_image=0,
    #         draw_false_text=0,
    #         image_only=image_only,
    #     )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            # draw_false_image=self.draw_false_image,
            # draw_false_text=self.draw_false_text,
            image_only=self.image_only,
        )

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.train_dataset.tokenizer = self.tokenizer
            self.val_dataset.tokenizer = self.tokenizer
            # self.test_dataset.tokenizer = self.tokenizer
            self.collate = functools.partial(
                self.train_dataset.collate, mlm_collator=self.mlm_collator,
            )
            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate,
        )
        return loader
