from .pixelbert import (
    pixelbert_transform,
    pixelbert_transform_randaug,
)
from .contrastive_tr import (
    BYOL_transform,
    Moco_transform,
)

_transforms = {
    "pixelbert": pixelbert_transform,
    "pixelbert_randaug": pixelbert_transform_randaug,
    "Moco_transform": Moco_transform,
    "BYOL_transform": BYOL_transform
}


def keys_to_transforms(keys: list, size=224):
    # print(keys)
    # import pdb
    # pdb.set_trace()
    return [_transforms[key](size=size) for key in keys]
