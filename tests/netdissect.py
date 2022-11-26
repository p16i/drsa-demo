import pytest
import numpy as np

from cxai import netdissect


@pytest.mark.parametrize(
    "layer,expected",
    [
        ["conv5_3", 24],
        ["conv4_3", 3],
    ],
)
def test_sanity_check_statistics(layer, expected):
    arch = "netdissect-vgg16-imagenet"

    (
        concepts,
        W,
        filters_per_concept,
    ) = netdissect.get_netdissect_concepts_and_basis_matrix(arch, layer)

    ix = np.argwhere(np.array(concepts) == "object:cat")

    assert filters_per_concept[ix] == expected
