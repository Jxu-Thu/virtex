import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict


def path2rest(path, iid2captions,iid2BTcaptions):
    name = path.split("/")[-1]
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    bt_captions = iid2BTcaptions[name]
    return [binary, captions, bt_captions, name]

# def path2rest(path, path2labels):
#     name = path.split("/")[-1]
#     with open(path, "rb") as fp:
#         binary = fp.read()
#     label = iid2labels[name]
#     return [binary, label]


def make_arrow(root, dataset_root):
    with open(f"{root}/annotations/captions_train2017_new_new.json", "r") as fp:
        captions = json.load(fp)

    # captions = captions["images"]

    iid2captions = defaultdict(list)
    iid2BTcaptions = defaultdict(list)
    for cap in tqdm(captions):
        # print(type(cap["image_id"]))
        filename = str(cap["image_id"])+".jpg"
        filename = (16-len(filename))*'0' + filename
        iid2captions[filename].append(cap['caption'])
        iid2BTcaptions[filename].append(cap['back_caption'])
    # for key, value in iid2captions.items():
    #     assert len(value)==len(iid2BTcaptions[key])
    
    paths = list(glob(f"{root}/train2017/*.jpg"))
    random.shuffle(paths)
    caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths), len(caption_paths), len(iid2captions),
    )

    bs = [path2rest(path, iid2captions, iid2BTcaptions) for path in tqdm(caption_paths)]
    # bs = [path2rest(path, iid2labels) for path in tqdm(paths)]
    # for i in bs:
    #     assert len(i[1])==len(i[2])
    # print('hh')
    # return
    for split in ["train"]:
        # batches = [b for b in bs if b[-1] == split]
        batches = bs[:10000]
        # batches = bs
        dataframe = pd.DataFrame(
            batches, columns=["image", "caption", "BT_caption", "image_id"],
        )
        # dataframe = pandas.DataFrame(
        #     bs, columns=["image", "label"],
        # )

        table = pa.Table.from_pandas(dataframe)

        # table = pyarrow.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)

        with pa.OSFile(
            f"{dataset_root}/coco2017_caption_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

        # with pyarrow.OSFile(
        #     f"./MNist_train.arrow", "wb"
        # ) as sink:
        #     with pyarrow.RecordBatchFileWriter(sink, table.schema) as writer:
        #         writer.write_table(table)

if __name__ == '__main__':
    make_arrow('../../data/coco2017', '../../data/coco2017')