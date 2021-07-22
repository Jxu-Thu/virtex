import random
import torch
import io
import pyarrow as pa
import os
import numpy as np
from PIL import Image
from transforms import keys_to_transforms
import json

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transform_keys: list,
        image_size: int,
        names: list,
        text_column_name: str = "",
        remove_duplicate=True,
        max_text_len=40,
        # draw_false_image=0,
        # draw_false_text=0,
        image_only=False,
        image_per_unit = True
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        assert len(transform_keys) >= 1
        super().__init__()
        
        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        # self.draw_false_image = draw_false_image
        # self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.image_per_unit = image_per_unit


        with open(f"{data_dir}/annotations/captions_train2017_ch_jp_fra.json", "r") as fp:
            self.dataset = json.load(fp)
        
        if image_per_unit:
            self.length = len(self.dataset)




        # # 就是把所有arror读进来concat
        # if len(names) != 0:

        #     tables = [
        #         pa.ipc.RecordBatchFileReader(
        #             pa.memory_map(f"{data_dir}/{name}.arrow", "r")
        #         ).read_all()
        #         for name in names
        #         if os.path.isfile(f"{data_dir}/{name}.arrow")
        #     ]

        #     print(f'read the dataset from {names}')
        #     self.table_names = list()
            
        #     # print(len(tables))
        #     # import pdb
        #     # pdb.set_trace()
        #     for i, name in enumerate(names):
        #         self.table_names += [name] * len(tables[i])

        #     self.table = pa.concat_tables(tables, promote=True)
        #     self.all_texts = []
    
        #     if len(text_column_name):
        #         self.text_column_name = text_column_name
        #         for name in text_column_name:
        #             self.all_texts_ = self.table[name].to_pandas().tolist()
        #             if (not isinstance(self.all_texts_[0], list)) and (not isinstance(self.all_texts_[0], np.ndarray)):
        #                 self.all_texts_ = [[texts] for texts in self.all_texts_]
        #             # import pdb
        #             # pdb.set_trace()
        #             # self.all_texts_ = (
        #             #     [list(set(texts)) for texts in self.all_texts_]
        #             #     if remove_duplicate
        #             #     else self.all_texts_
        #             # )
        #             self.all_texts.append(self.all_texts_)
        #     else:
        #         self.all_texts = list()
        # else:
        #     self.all_texts = list()
        # # pdb.set_trace()
        # # for i in range(len(self.all_texts[0])):
        # #     if len(self.all_texts[0][i])!=len(self.all_texts[1][i]):
        # #         print(i)
        # #         print(self.all_texts[0][i])
        # #         print(self.all_texts[1][i])
        # #         print(len(self.all_texts[0][i]))
        # #         print(len(self.all_texts[1][i]))
        # #self.all_texts[0]每一个元素是一张图片的几个caption
        # # import pdb
        # # pdb.set_trace()
        # self.index_mapper = dict()

        # # index: img_index, all_texts_index
        # if len(text_column_name) and not self.image_only:
        #     j = 0
        #     for i, texts in enumerate(self.all_texts[0]):
        #         for _j in range(len(texts)):
        #             self.index_mapper[j] = (i, _j)
        #             j += 1
        # else:
        #     for i in range(len(self.table)):
        #         self.index_mapper[i] = (i, None)
        # # pdb.set_trace()
    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return self.length#50025 for 10000

    def get_raw_image(self, index, image_key="image"):
        if self.image_per_unit:
            return Image.open(os.path.join(self.data_dir, 'train2017',self.dataset[index]['image_path'])).convert("RGB")
        else:
            index, caption_index = self.index_mapper[index]
            image_bytes = io.BytesIO(self.table[image_key][index].as_py())
            image_bytes.seek(0)
            return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {
            "image": image_tensor,
            # "img_index": self.index_mapper[index][0],
            # "cap_index": self.index_mapper[index][1],
            # "raw_index": index,
        }

    def get_text(self, raw_index):
        pairs = []
        all_texts = []
        if self.image_per_unit:
            data = self.dataset[raw_index]
            random_text_id = random.randint(0, len(data['caption'])-1)
            caption = data['caption'][random_text_id]
            all_texts.append(caption)
            back_captions = data['back_captions'][random_text_id]
            random_BT_id = random.randint(0, len(back_captions)-1)
            all_texts.append(back_captions[random_BT_id])
        

        for text in all_texts:
            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_special_tokens_mask=True,
            )
            pairs.append((text, encoding))
        return {
            "text": pairs,
            # "img_index": index,
            # "cap_index": caption_index,
            # "raw_index": raw_index,
        }

    def get_suite(self, index):
        # print("get_one_item")
        result = None
        while result is None:
            # try:
            ret = dict()
            ret.update(self.get_image(index))
            if not self.image_only:
                txt = self.get_text(index)
                # ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update(txt)
            result = True
            # except Exception as e:
            #     # debug : not printing anymore
            #     # print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
            #     index = random.randint(0, len(self.index_mapper) - 1)
        return ret

    def collate(self, batch, mlm_collator):
        #batch[0]:{"image":[view1,view2],'img_index': 975, 'cap_index': 2,
        #           'raw_index': 4882, 'replica': True, 'text': [(content_1,encoding_1),(content_2,encoding_2)]}
        # print("collate")
        # print(len(batch))
        # print(batch[0]["image"][0].shape)
        # print(batch[0]["text"][0][0])
        
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        # Reshape key: batch * data
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        # ['image', 'false_image_0']
        img_sizes = list()
        # import pdb
        # pdb.set_trace()
        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i]

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]
            # view_size=1
            view_size = len(img[0])

            new_images = [
                torch.zeros(batch_size, 3, max_height, max_width)
                for _ in range(view_size)
            ]

            for bi in range(batch_size):
                orig_batch = img[bi]
                for vi in range(view_size):
                    if orig_batch is None:
                        new_images[vi][bi] = None
                    else:
                        orig = img[bi][vi]
                        new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

            dict_batch[img_key] = new_images

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[[d[view][0] for d in dict_batch[txt_key]] for view in range(2)] for txt_key in txt_keys]
            #[1,view,bs]
            encodings = [[[d[view][1] for d in dict_batch[txt_key]] for view in range(2)] for txt_key in txt_keys]
            # encodings[0][0][0] =
            # {'input_ids': [101, 1037, 2931, 5093, 2447, 11820, 2014, 14513, 3388, 1998, 2770, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # 'special_tokens_mask': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            # 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
            draw_text_len = len(encodings)#1
            flatten_encodings = [e for encoding in encodings for e in encoding]#[view,bs]
            flatten_mlms = [mlm_collator(view_encoding) for view_encoding in flatten_encodings]
            # ('inputs_id', 'labels)
            # inputs_id : batch size (32) * 40 tokens
            # labels : batch size * 40 tokens

            for i, txt_key in enumerate(txt_keys):
                #[view,bs]
                texts, encodings = (
                    [[d[view][0] for d in dict_batch[txt_key]] for view in range(2)],
                    [[d[view][1] for d in dict_batch[txt_key]] for view in range(2)]
                )

                # no thin
                mlm_ids, mlm_labels = (
                    [flatten_mlm["input_ids"][batch_size * (i) : batch_size * (i + 1)] for flatten_mlm in flatten_mlms],
                    [flatten_mlm["labels"][batch_size * (i) : batch_size * (i + 1)] for flatten_mlm in flatten_mlms],
                )
                #[2,bs,40]
                input_ids = [torch.zeros_like(mlm_id) for mlm_id in mlm_ids]
                attention_mask = [torch.zeros_like(mlm_id) for mlm_id in mlm_ids]
                for _x, view in enumerate(encodings):
                    for _i, encoding in enumerate(view):
                        _input_ids, _attention_mask = (
                            torch.tensor(encoding["input_ids"]),
                            torch.tensor(encoding["attention_mask"]),
                        )
                        # print('hh')
                        # print(_x)
                        # print(_i)
                        # print(len(_input_ids))
                        # print(input_ids)
                        input_ids[_x][_i, :len(_input_ids)] = _input_ids
                        attention_mask[_x][_i, :len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                # dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                # dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                # dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

        return dict_batch
