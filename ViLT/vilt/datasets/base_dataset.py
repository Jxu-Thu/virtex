import random
import torch
import io
import pyarrow as pa
import os
import numpy as np
from PIL import Image
from vilt.transforms import keys_to_transforms


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
        draw_false_image=0,
        draw_false_text=0,
        image_only=False,
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
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir

        # 就是把所有arror读进来concat
        if len(names) != 0:
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")
                ).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]

            print(f'read the dataset from {names}')
            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])

            self.table = pa.concat_tables(tables, promote=True)
            if text_column_name != "":
                self.text_column_name = text_column_name
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                if (not isinstance(self.all_texts[0], list)) and (not isinstance(self.all_texts[0], np.ndarray)):
                    self.all_texts = [[texts] for texts in self.all_texts]

                self.all_texts = (
                    [list(set(texts)) for texts in self.all_texts]
                    if remove_duplicate
                    else self.all_texts
                )

            else:
                self.all_texts = list()
        else:
            self.all_texts = list()

        self.index_mapper = dict()

        # index: img_index, all_texts_index
        if text_column_name != "" and not self.image_only:
            j = 0
            for i, texts in enumerate(self.all_texts):
                for _j in range(len(texts)):
                    self.index_mapper[j] = (i, _j)
                    j += 1
        else:
            for i in range(len(self.table)):
                self.index_mapper[i] = (i, None)

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_mapper)

    def get_raw_image(self, index, image_key="image"):
        index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {
            "image": image_tensor,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }

    def get_false_image(self, rep, image_key="image"):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        image = self.get_raw_image(random_index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {f"false_image_{rep}": image_tensor}

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]

        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text": (text, encoding),
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.index_mapper) - 1)

        index, caption_index = self.index_mapper[random_index]
        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def get_suite(self, index):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(index))
                if not self.image_only:
                    txt = self.get_text(index)
                    ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    ret.update(txt)

                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
                result = True
            except Exception as e:
                # debug : not printing anymore
                # print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)
        return ret

    def get_word_boundary(self, examples, max_len=40):
        word_boundary_index_array = []
        ref_texts = []
        for e in examples:
            ref_tokens = []
            for id in e["input_ids"]:
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            cand_indexes = []
            ref_text = []
            id=0
            for (i, token) in enumerate(ref_tokens):
                if not token.startswith("##"):
                    id=id+1
                cand_indexes.append(id)
                ref_text.append(token)
            if len(cand_indexes) < max_len:
                cand_indexes += [cand_indexes[-1]]*(max_len-len(cand_indexes))
            word_boundary_index_array.append(cand_indexes)
            ref_texts.append(ref_text)
        return np.array(word_boundary_index_array), ref_texts

    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        # Reshape key: batch * data
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        # ['image', 'false_image_0']
        img_sizes = list()

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
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            # encodings[0][0] =
            # {'input_ids': [101, 1037, 2931, 5093, 2447, 11820, 2014, 14513, 3388, 1998, 2770, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # 'special_tokens_mask': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            # 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            import pdb
            pdb.set_trace()
            flatten_mlms = mlm_collator(flatten_encodings)
            # ('inputs_id', 'labels)
            # inputs_id : batch size (32) * 40 tokens
            # labels : batch size * 40 tokens

            for i, txt_key in enumerate(txt_keys):
                import pdb
                pdb.set_trace()
                word_boundary_index, ref_texts = self.get_word_boundary(encodings[i], self.max_text_len)
                word_boundary_index = torch.tensor(word_boundary_index)
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                # no thin
                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask
                dict_batch[f"{txt_key}_word_boundary"] = word_boundary_index

        return dict_batch
