import os

import numpy as np
import pandas as pd

import torch
from torchvision.datasets.folder import pil_loader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from frozendict import frozendict

from cxai import utils as putils

current_path = os.path.relpath(__file__)
root_module_dir = "/".join(current_path.split("/")[:-2])

# todo: perhaps, this module should be in cxai/data/imagenet

df_label_mapping = pd.read_csv(f"{root_module_dir}/config/imagenet-label-mapping.csv")

# Taken from https://github.com/lightly-ai/lightly/blob/master/lightly/data/collate.py#L17
statistics = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def get_index_from_imagenet_id(imagenet_id):
    row = df_label_mapping[df_label_mapping["imagenet-id"] == imagenet_id]

    assert len(row) == 1, f"we should have only one row {row}"

    return int(row.index[0])


def get_desc_from_imagenet_id(imagenet_id):
    row = df_label_mapping[df_label_mapping["imagenet-id"] == imagenet_id]

    return row.desc.values[0]


def get_desc_from_label_id(ix: int):
    nsid = get_imagenet_id_from_ix(ix)

    return get_desc_from_imagenet_id(nsid)


def get_all_imagenet_ids():
    return df_label_mapping["imagenet-id"].values


def get_imagenet_id_from_ix(ix: int) -> str:
    return df_label_mapping[df_label_mapping.index == ix]["imagenet-id"].values[0]


imgclasses = dict(zip(df_label_mapping.index.values, df_label_mapping.desc.values))

ix_to_classname = frozendict(imgclasses)
classname_to_ix = frozendict(
    dict(zip(ix_to_classname.values(), ix_to_classname.keys()))
)


class ImageNetSubset(Dataset):
    """ImageNetSubset takes a list of imagenet files and allow to load with DataLoader

    Arguments:
        flist (str): path to list of imagenet files
        transform (T.Compose): transform (model dependent)
        data_dir (str): path to the directory of these image files
    """

    def __init__(
        self,
        flist: str,
        transform: T.Compose,
        data_dir: str = os.path.expanduser("~/datasets/imagenet/train"),
    ):

        self.flist = flist
        self.data_dir = data_dir

        with open(self.flist, "r") as fh:
            filenames = []
            for l in fh:
                filenames.append(l.strip())
            self.filenames = filenames

        self.transform = transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        img = putils.image.load_image(f"{self.data_dir}/{filename}")

        img = self.transform(img)

        nsid = filename.split("/")[0]
        y = putils.imagenet.get_index_from_imagenet_id(nsid)

        return img, y, filename

    def __len__(self):
        return len(self.filenames)
