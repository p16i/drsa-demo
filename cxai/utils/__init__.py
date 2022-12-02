from typing import Tuple, Union, List, Dict

import os
from pathlib import Path

import json

from PIL import Image


import torch

from .image import *
from . import imagenet
from . import viz


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def save_pretty_json(dest: str, data: dict):
    with open(dest, "w") as fh:
        json.dump(
            data,
            fh,
            indent=4,
            sort_keys=True,
        )


def clean_ext(filename: str):
    return Path(filename).stem


def parent_dir(path: str) -> Path:
    return Path(path).parent


def string_to_tuple_of_numbers(t: str, sep=",", dtype=int) -> Tuple[Union[int, float]]:
    entries = t.split(sep)

    entries = list(filter(lambda e: e, entries))

    entries = np.array(entries)

    return entries.astype(dtype)
