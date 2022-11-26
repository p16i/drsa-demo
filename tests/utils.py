import pytest

import numpy as np
import torch

from cxai import utils


@pytest.mark.parametrize(
    "nsid,label,name",
    [
        ("n13044778", 995, "earthstar"),
        ("n02391049", 340, "zebra"),
        ("n07920052", 967, "espresso"),
        ("n04591157", 906, "Windsor_tie"),
        ("n02980441", 483, "castle"),
    ],
)
def test_imagenet_utils(nsid, label, name):
    assert utils.imagenet.get_index_from_imagenet_id(nsid) == label
    assert utils.imagenet.ix_to_classname[label] == name


@pytest.mark.parametrize(
    "data,expected,dtype",
    [
        ("232,322,55", [232, 322, 55], int),
        ("0.0,0.25,0.5", [0.0, 0.25, 0.5], float),
    ],
)
def test_string_to_tuple(data, expected, dtype):
    actual = utils.string_to_tuple_of_numbers(data, dtype=dtype)

    np.testing.assert_almost_equal(actual, expected)
