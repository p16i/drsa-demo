import torch
import pytest
import numpy as np

from cxai import models, factory, constants


from . import _generate_input, device


import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def prediction_from_timm(arch, x):
    # Reference: https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/docs/models/.templates/code_snippets.md#how-do-i-use-this-model-on-an-image

    model = timm.create_model(arch, pretrained=True).to(device).eval()

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    logits = model(x)

    return logits


@torch.no_grad()
@pytest.mark.slow
@pytest.mark.parametrize("layer", ["stage2", "stage3"])
@pytest.mark.parametrize("imgname", ["castle.jpg", "noise2", "noise20"])
@pytest.mark.parametrize(
    "slug",
    ["dm_nfnet_f0"],
)
def test_same_activation_at_layer(slug, layer, imgname):
    input_size = constants.INPUT_SHAPE[slug]

    model, (rc_transform, input_transform) = factory.make_model(slug)
    inp = _generate_input(imgname, input_size, input_transform)
    inp = inp.unsqueeze(0)

    model.to(device)

    feature_extractor, _ = models.split_model_at_layer(model, layer)
    actual_activation = feature_extractor(inp).detach().cpu().numpy()

    clean_model, _ = factory.make_model(slug)
    if layer == "stage2":
        spliting_ix = 2
    elif layer == "stage3":
        spliting_ix = 3
    clean_feature_extractor = torch.nn.Sequential(
        clean_model.stem, clean_model.stages[: spliting_ix + 1]
    )

    clean_feature_extractor.to(device)
    expected_activation = clean_feature_extractor(inp).detach().cpu().numpy()

    np.testing.assert_allclose(
        actual_activation,
        expected_activation,
        err_msg=f"spliting `{slug}` `{layer}` returns an incorrect feature map!",
    )


@torch.no_grad()
@pytest.mark.slow
@pytest.mark.parametrize("layer", ["stage2", "stage3"])
@pytest.mark.parametrize("filename", ["noise0", "noise1", "noise0-big", "castle.jpg"])
@pytest.mark.parametrize(
    "slug",
    ["dm_nfnet_f0"],
)
def test_composing_spliting_components_same_prediction(layer, filename, slug):

    input_size = constants.INPUT_SHAPE[slug]

    model, (rc_transform, input_transform) = factory.make_model(slug)

    feature_extractor, classification_head = models.split_model_at_layer(model, layer)

    for mod in (feature_extractor, classification_head):
        mod.to(device)

    inp = _generate_input(filename, input_size, input_transform).unsqueeze(0)

    actual_logits = classification_head(feature_extractor(inp))

    expected_logits = prediction_from_timm("dm_nfnet_f0", inp)

    np.testing.assert_allclose(
        actual_logits.detach().cpu().numpy(), expected_logits.detach().cpu().numpy()
    )
