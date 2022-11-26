import torch
import pytest
import numpy as np

from cxai import models, factory, constants


from . import _generate_input, device

ACTIVATION_ATTRIBUTE = "__act"

# The hook is for getting expected activation.
def _fh_save_activation(module, inp, out):
    (inp,) = inp

    setattr(module, ACTIVATION_ATTRIBUTE, inp)


@torch.no_grad()
@pytest.mark.slow
@pytest.mark.parametrize("layer", ["conv5_3", "conv4_3"])
@pytest.mark.parametrize("imgname", ["castle.jpg", "noise2", "noise20"])
@pytest.mark.parametrize(
    "slug",
    [
        "torchvision-vgg16-imagenet",
        "netdissect-vgg16-imagenet",
    ],
)
def test_same_activation_at_layer(slug, layer, imgname):
    input_size = constants.INPUT_SHAPE[slug]

    model, (rc_transform, input_transform) = factory.make_model(slug)
    inp = _generate_input(imgname, input_size, input_transform)
    inp = inp.unsqueeze(0)

    model.to(device)

    feature_extractor, _ = models.split_model_at_layer(model, layer)
    actual_activation = feature_extractor(inp).detach().cpu().numpy()

    # This try-catch is for getting expected_activation.
    # remark: this forward-hook approach is more convenient when using with layer names.
    try:
        if layer == "conv5_3":
            layer_with_act_at = model.features.pool5
        elif layer == "conv4_3":
            layer_with_act_at = model.features.pool4

        fh = layer_with_act_at.register_forward_hook(_fh_save_activation)
        _ = model(inp)
    finally:
        fh.remove()

    expected_activation = (
        getattr(layer_with_act_at, ACTIVATION_ATTRIBUTE).detach().cpu().numpy()
    )

    np.testing.assert_allclose(
        actual_activation,
        expected_activation,
        err_msg=f"spliting `{slug}` `{layer}` returns an incorrect feature map!",
    )


@torch.no_grad()
@pytest.mark.slow
@pytest.mark.parametrize("layer", ["conv5_3", "conv4_3"])
@pytest.mark.parametrize("filename", ["noise0", "noise1", "noise0-big", "castle.jpg"])
@pytest.mark.parametrize(
    "slug",
    [
        "torchvision-vgg16-imagenet",
        "netdissect-vgg16-imagenet",
    ],
)
def test_composing_spliting_components_same_prediction(layer, filename, slug):

    input_size = constants.INPUT_SHAPE[slug]

    model, (rc_transform, input_transform) = factory.make_model(slug)

    feature_extractor, classification_head = models.split_model_at_layer(model, layer)

    model_clean, _ = models.vgg16.get_model(slug)

    for mod in (feature_extractor, classification_head, model_clean):
        mod.to(device)

    inp = _generate_input(filename, input_size, input_transform)

    actual = classification_head(feature_extractor(inp.unsqueeze(0)))

    expected = model_clean(inp.unsqueeze(0))

    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), expected.detach().cpu().numpy()
    )
