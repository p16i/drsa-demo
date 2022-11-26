import pytest

import torch
import numpy as np

from cxai import factory, inspector
from cxai import utils as putils

torch.manual_seed(1)


@pytest.mark.parametrize(
    "d,ns,ss,w",
    [
        (6, 2, 2, 4),
        (18, 6, 3, 4),
        (18, 6, 3, 8),
    ],
)
def test_group_basis_inspector(d, ns, ss, w):

    weights = np.zeros((d, ns, ss))

    eyes = np.eye(d)

    for gix in range(ns):
        weights[:, gix, :] = eyes[:, ss * gix : ss * (gix + 1)]

    insp = inspector.GroupBasisInspector("layer", weights)

    activation = torch.randn((1, d, w, w))
    context = torch.randn((1, d, w, w))

    arr_encoded = []
    for t in [activation, context]:

        encoded_t = insp.encode_activation(t)

        assert encoded_t.shape == (1, ns * ss, *activation.shape[2:])

        np.testing.assert_allclose(encoded_t, t[:, : ns * ss, :, :])

        arr_encoded.append(encoded_t)

    encoded_act, encoded_ctx = arr_encoded

    rel_subspaces = insp.compute_subspace_relevance(encoded_act, encoded_ctx)

    assert rel_subspaces.shape == (1, ns, *activation.shape[2:])

    np.testing.assert_allclose(
        rel_subspaces.sum(), (activation * context)[:, : ns * ss].sum(), atol=1e-5
    )


@pytest.mark.parametrize(
    "d,ns,ss,w,k",
    [
        (6, 3, 1, 1, 3),
        (20, 5, 3, 1, 5),
        (18, 6, 3, 4, 2),
        (18, 6, 3, 8, 3),
    ],
)
@pytest.mark.parametrize("with_coefficient", [False, True])
def test_ranking(d, ns, ss, w, k, with_coefficient):
    weights = np.zeros((d, ns, ss))

    eyes = np.eye(d)

    for gix in range(ns):
        weights[:, gix, :] = eyes[:, ss * gix : ss * (gix + 1)]

    if with_coefficient:
        subspace_coefficients = np.power(10, np.arange(ns)[::-1])
        expected_order = np.argsort(-subspace_coefficients)[:k]
        data = torch.ones(ns)
    else:
        subspace_coefficients = None
        expected_order = np.arange(ns)[::-1][:k]
        data = torch.arange(1, ns + 1)

    insp = inspector.GroupBasisInspector(
        "layer", weights, subspace_coefficients=subspace_coefficients
    )

    activation = data.view((1, ns, 1, 1)).repeat((1, 1, w, w))
    context = data.view((1, ns, 1, 1)).repeat((1, 1, w, w))

    rel_subspaces = (activation * context).flatten(start_dim=2).sum(dim=2).squeeze(0)

    sorted_indices = insp.rank_subspaces(rel_subspaces, top_k=k)

    np.testing.assert_allclose(sorted_indices, expected_order)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "name,expected",
    [
        ("random1-ns1-ss5", (1, 5)),
        ("random1-ns5-ss10", (5, 10)),
        ("netdissect", (None, None)),
        ("learn--something", (None, None)),
    ],
)
def test_parse_parameters(name, expected):

    nsss = inspector._parse_subspace_parameters(name)

    assert nsss == expected


@pytest.mark.parametrize("attr_method", ["lrp"])
@pytest.mark.parametrize("layer", ["conv4_3", "conv5_3"])
def test_netdissect_complete(attr_method, layer):
    if "svs" in attr_method:
        pytest.skip(
            f"we skip {attr_method} for now because 1) we plan not to use it and 2) it is very slow!"
        )
    arch = "torchvision-vgg16-imagenet"

    device = putils.get_device()

    model, (_, input_transform) = factory.make_model(arch)
    model.to(device)

    explainer = factory.make_explainer(attr_method, model)

    img = putils.image.load_image("./tests/data/castle.jpg")
    tsimg = input_transform(img).to(device)
    label = 483
    top_k = 3

    insp_netdissect = inspector.get_inspector_for_basis(
        arch, layer, base_dir=None, basis_name="netdissect"
    ).to(device)
    _, hm_ori_standard, info_standard = explainer.explain_with_inspector(
        tsimg, label, inspector=insp_netdissect, top_k=top_k
    )
    del insp_netdissect

    insp_netdissect_complete = inspector.get_inspector_for_basis(
        arch,
        layer,
        base_dir=None,
        basis_name="netdissect-complete",
    ).to(device)

    _, hm_ori_complete, info_complete = explainer.explain_with_inspector(
        tsimg, label, inspector=insp_netdissect_complete, top_k=top_k
    )
    del insp_netdissect_complete

    np.testing.assert_allclose(
        info_complete.top_k_sources[:-1],
        info_standard.top_k_sources[:-1],
    )

    np.testing.assert_allclose(hm_ori_complete, hm_ori_standard, atol=1e-6)

    assert (
        info_complete.top_k_sources[-1] is None
    ), "the last subspace should be -1, which is None!"

    np.testing.assert_allclose(
        info_complete.input_subspace_residue_heatmap,
        np.zeros_like(info_complete.input_subspace_residue_heatmap),
        err_msg="residue subspace should be zero",
        atol=1e-6,
    )

    np.testing.assert_allclose(
        info_complete.input_top_k_source_heatmaps[
            :-1,
        ],
        info_standard.input_top_k_source_heatmaps[
            :-1,
        ],
        err_msg="all first k-1 subspaces are the same.",
        atol=1e-6,
    )

    np.testing.assert_allclose(
        info_complete.input_top_k_source_heatmaps[
            -1,
        ],
        info_standard.input_top_k_source_heatmaps[
            -1,
        ]
        + info_standard.input_subspace_residue_heatmap,
        err_msg="the last subspace is the sum of the original residue and last subspace",
        atol=1e-6,
    )
