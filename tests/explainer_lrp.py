import pytest

import torch
import numpy as np

from cxai import factory, constants
from cxai.explainers.lrp.base import lrp_rule_ratio

from . import _generate_input, device, _find_biases_and_set_to_zero


def _test_logit_not_affected_by_forward_hooks(slug, attr_method):
    model, (_, input_transform) = factory.make_model(slug)

    input_size = constants.INPUT_SHAPE[slug]

    explainer = factory.make_explainer(attr_method, model)
    model_clean, _ = factory.make_model(slug)

    for mod in [model, model_clean]:
        mod.to(device)

    for filename, label in [
        ("noise0", 20),
        ("noise1", 500),
        ("noise2", 22),
        ("castle.jpg", 483),
        ("viaduct.jpg", 888),
    ]:

        inp = _generate_input(filename, input_size, input_transform)

        actual_logits, heatmap = explainer.explain(inp, label)

        expected_logits = model_clean(inp.unsqueeze(0)).detach().cpu().numpy().squeeze()

        np.testing.assert_allclose(actual_logits, expected_logits, atol=1e-6)
        assert not np.isnan(heatmap).any()


@pytest.mark.slow
@pytest.mark.parametrize(
    "slug,attr_method",
    [
        ("torchvision-vgg16-imagenet", "lrp"),
        ("netdissect-vgg16-imagenet", "lrp"),
    ],
)
def test_vgg16_logit_not_affected_by_forward_hooks(slug, attr_method):
    _test_logit_not_affected_by_forward_hooks(slug, attr_method)


@pytest.mark.slow
@pytest.mark.parametrize(
    "arch,attr_method",
    [
        ("torchvision-vgg16-imagenet", "lrp"),
        ("netdissect-vgg16-imagenet", "lrp"),
    ],
)
@pytest.mark.parametrize(
    "filename,label",
    [
        ("castle.jpg", 483),
        ("viaduct.jpg", 888),
        ("volcano.jpg", 980),
        ("zebra.jpg", 340),
    ],
)
def test_conservation_no_bias(arch, attr_method, filename, label):
    atol = 1e-3 if arch == "dm_nfnet_f0" else 1e-5

    input_size = constants.INPUT_SHAPE[arch]

    model, (_, input_transform) = factory.make_model(arch)
    _find_biases_and_set_to_zero(model)

    model.to(device)

    img = _generate_input(filename, input_size, input_transform)

    explainer = factory.make_explainer(attr_method, model)

    logits, heatmaps = explainer.explain(img, label)

    np.testing.assert_allclose(heatmaps.sum(), logits[label], atol=atol)


def test_lrp_div_standard_case_with_gamma_varying():
    arr_pos_neg_relevance_ratios = []

    for gamma in [0, 0.1, 0.2, 0.5]:
        ai = torch.tensor([1, 2, 3]).reshape((1, -1)).float()
        ai.requires_grad_(True)

        # This simulate a linear layer w/o activations.
        w = torch.tensor([1, -1, 1]).float()
        w_mod = w + gamma * w.clamp(min=0)

        aj = ai @ w

        nom = ai @ w_mod

        overriden_aj = lrp_rule_ratio(input=nom, output=aj, eps=0)

        overriden_aj.backward()

        relevance = ai * ai.grad

        assert torch.allclose(overriden_aj, aj)
        assert torch.allclose(relevance.sum(), aj)

        pos_part = relevance.clamp(min=0).sum()
        neg_part = relevance.clamp(max=0).abs().sum()

        ratio = neg_part / pos_part
        ratio = ratio.detach().numpy()
        print(f"gamma={gamma}: relevance={relevance}")

        arr_pos_neg_relevance_ratios.append(ratio)

    arr_pos_neg_relevance_ratios = np.array(arr_pos_neg_relevance_ratios)

    np.testing.assert_equal(
        arr_pos_neg_relevance_ratios[:-1] > arr_pos_neg_relevance_ratios[1:],
        True,
        err_msg="""
        Increasing `gamma` should decreasing the ratio abs(neg_part)/pos_part!
        This means that the negative part proportionally gets smaller.
        """,
    )


def test_lrp_div_with_eps_stabilizer():
    eps = 0.1
    ai = torch.tensor([1, 2, 3]).reshape((1, -1)).float()
    ai.requires_grad_(True)

    gamma = 0.1

    # This simulate a linear layer w/o activations.
    # todo: add a case where denom > eps
    w = (eps**2) * torch.tensor([1, 1, 1]).float()
    w_mod = w + gamma * w.clamp(min=0)

    aj = ai @ w
    print("aj", aj)

    nom = ai @ w_mod
    print("nom", nom)
    print("denom > eps", nom.abs() > eps)

    overriden_aj = lrp_rule_ratio(input=nom, output=aj, eps=eps)

    assert torch.allclose(overriden_aj, aj)
