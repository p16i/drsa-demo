import torch
import pytest
import numpy as np

from cxai import models, factory, constants

from . import _generate_input, device

# todo: add more images
# remark: surely, using more images would slow down the test suite.
IMAGE_LABEL_PAIRS = [("castle.jpg", 483)]


def _test_extracting_activation_with_lrp_and_inspection_context(
    arch, attr_method, imgname, label, layer
):
    input_size = constants.INPUT_SHAPE[arch]
    model, (_, input_transform) = factory.make_model(arch)
    model.to(device)

    explainer = factory.make_explainer(attr_method, model)

    feature_extractor, _ = models.split_model_at_layer(model, layer)

    x = _generate_input(imgname, input_size, input_transform)
    actual_activation, context = explainer.get_intermediate_activation_and_context(
        layer, x, label
    )

    expected_activation = feature_extractor(x.unsqueeze(0))

    np.testing.assert_allclose(
        actual_activation.detach().cpu().numpy(),
        expected_activation.detach().cpu().numpy(),
    )
    assert not torch.isnan(context).any()


@pytest.mark.slow
@pytest.mark.parametrize("layer", ["conv4_3", "conv5_3"])
@pytest.mark.parametrize("arch", ["torchvision-vgg16-imagenet"])
@pytest.mark.parametrize("imgname,label", IMAGE_LABEL_PAIRS)
def test_vgg16_lrp_extracting_activation_and_context(arch, layer, imgname, label):
    _test_extracting_activation_with_lrp_and_inspection_context(
        arch, "lrp", imgname, label, layer
    )


@pytest.mark.parametrize("layer", ["conv3_3", "conv4_3", "conv5_3"])
def test_lrp_layer_context_and_split_model(layer):
    arch = "torchvision-vgg16-imagenet"
    attr_method = "lrp"

    model, _ = factory.make_model(arch)
    explainer = factory.make_explainer(attr_method, model)

    _, classifier = models.vgg16.split_model_at_layer(model, layer)

    layer_ctx = explainer.inspect_layer(layer)

    assert (
        layer_ctx.module == classifier[0][0]
    ), """
    The location employs input forward hook for LPR
    and the first module of the classifier head for IG
    are the same.
    """
