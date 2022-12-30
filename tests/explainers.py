import numpy as np
import pytest

from cxai import factory, constants, inspector
from . import _generate_input, device

IMAGE_LABEL_PAIRS = [
    ("castle.jpg", 483),
    ("noise20", 20),
]


def _test_producing_standard_explanation(attr_method, arch, imgname, label):
    model, (_, input_transform) = factory.make_model(arch)
    model.to(device)

    input_size = constants.INPUT_SHAPE[arch]

    inp = _generate_input(imgname, input_size, input_transform).to(device)

    explainer = factory.make_explainer(attr_method, model)

    actual_logits, heatmap = explainer.explain(inp, label)
    expected_logits = model(inp.unsqueeze(0)).detach().cpu().numpy().squeeze()

    assert not np.isnan(heatmap).any()


@pytest.mark.parametrize("arch", ["torchvision-vgg16-imagenet"])
@pytest.mark.parametrize("imgname,label", IMAGE_LABEL_PAIRS)
@pytest.mark.parametrize("attr_method", ["lrp"])
def test_standard_explanation_vgg16(attr_method, arch, imgname, label):
    _test_producing_standard_explanation(attr_method, arch, imgname, label)


@pytest.mark.parametrize("arch", ["dm_nfnet_f0"])
@pytest.mark.parametrize("imgname,label", IMAGE_LABEL_PAIRS)
@pytest.mark.parametrize("attr_method", ["lrp0.1"])
def test_standard_explanation_nfnet(attr_method, arch, imgname, label):
    _test_producing_standard_explanation(attr_method, arch, imgname, label)


def _test_explanation_subspaces_with_random_basis(
    arch, layer, attr_method, imgname, label, ns, ss, top_k, atol=1e-5
):
    model, (_, input_transform) = factory.make_model(arch)
    model.to(device)

    input_size = constants.INPUT_SHAPE[arch]

    inp = _generate_input(imgname, input_size, input_transform).to(device)

    explainer = factory.make_explainer(attr_method, model)

    random_inspector = inspector.get_inspector_for_basis(
        arch, layer, f"random1-ns{ns}-ss{ss}", base_dir=None
    ).to(device)

    expected_logits, expected_ori_heatmap = explainer.explain(inp, label)

    logits, heatmap, info = explainer.explain_with_inspector(
        inp, label, random_inspector, top_k=top_k
    )

    # these asserts just check that those two explain_(...) methods have the same original heatmap
    np.testing.assert_allclose(logits, expected_logits, atol=atol)
    if "svs" in attr_method or "ig" in attr_method:
        np.testing.assert_allclose(
            heatmap,
            info.input_top_k_source_heatmaps.sum(axis=0)
            + info.input_subspace_residue_heatmap,
            atol=atol,
        )
    else:
        np.testing.assert_allclose(heatmap, expected_ori_heatmap, atol=atol)

    assert info.input_top_k_source_heatmaps.shape == (top_k, *input_size)
    assert info.rel_per_source.shape == (1, top_k)

    info.input_top_k_source_heatmaps.shape == (top_k, *input_size)

    if not ("svs" in attr_method or "ig" in attr_method):
        # remark: we don't produce the all subspace heatmap for SVS and IG
        # due to the computational reason.
        np.testing.assert_allclose(
            info.input_top_k_source_heatmaps.sum(axis=0),
            info.input_all_subspaces_heatmap,
            atol=1e-5,
        )

    if ss == constants.get_arch_layer_dimensions(arch, layer):
        np.testing.assert_allclose(
            info.input_subspace_residue_heatmap.sum(), 0, atol=atol
        )


@pytest.mark.parametrize("arch", ["torchvision-vgg16-imagenet"])
@pytest.mark.parametrize("imgname,label", IMAGE_LABEL_PAIRS)
@pytest.mark.parametrize("attr_method", ["lrp"])
@pytest.mark.parametrize("layer", ["conv4_3"])
@pytest.mark.parametrize(
    "ns,ss,top_k",
    [
        (1, 512, 1),
        (5, 10, 3),
        (1, 20, 1),
    ],
)
def test_producing_subspace_explanations_vgg16(
    arch, layer, attr_method, imgname, label, ns, ss, top_k
):
    _test_explanation_subspaces_with_random_basis(
        arch, layer, attr_method, imgname, label, ns, ss, top_k
    )


@pytest.mark.slow
@pytest.mark.parametrize("arch", ["dm_nfnet_f0"])
@pytest.mark.parametrize("imgname,label", IMAGE_LABEL_PAIRS)
@pytest.mark.parametrize("attr_method", ["lrp0.1"])
@pytest.mark.parametrize("layer", ["stage2"])
@pytest.mark.parametrize(
    "ns,ss,top_k",
    [
        (1, 1536, 1),
        (5, 10, 3),
        (1, 20, 1),
    ],
)
def test_producing_subspace_explanations_nfnet(
    arch, layer, attr_method, imgname, label, ns, ss, top_k
):
    _test_explanation_subspaces_with_random_basis(
        arch, layer, attr_method, imgname, label, ns, ss, top_k
    )
