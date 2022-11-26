import collections

import numpy as np
import torch
import torchvision

from torchvision import transforms as T
from cxai import constants
from cxai.explainers.base import Explainer

from . import ATTRIBUTE_TRANSFORMATION

# This function is adapted from https://github.com/davidbau/dissect/blob/9421eaa8672fd051088de6c0225a385064070935/experiment/oldvgg16.py#L4
# The goal here is to strict with the layer names defined by the Bau et al.'s convention.
def get_model(slug: str):
    if slug == "torchvision-vgg16-imagenet":
        # This is of course trained on imagenet.
        model = torchvision.models.vgg16(pretrained=True)

        mean = constants.IMAGENET_MEAN
        std = constants.IMAGENET_STD
    elif slug == "netdissect-vgg16-imagenet":
        model = torchvision.models.vgg16(pretrained=False)
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                "https://tubcloud.tu-berlin.de/s/i3rN9NHey8FeaAf/download/netdissect-vgg16_imagenet-2b51436b.pth"
            )
        )
        mean = constants.NETDISSECT_RGB_MEAN
        std = constants.NETDISSECT_RGB_STD
    elif slug == "netdissect-vgg16-places365":
        model = torchvision.models.vgg16(pretrained=False)

        model.classifier[-1] = torch.nn.Linear(4096, 365)

        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                "https://tubcloud.tu-berlin.de/s/DWzkicoGAgkMJTF/download/netdissect-vgg16_places365-dab93d8c.pth"
            )
        )
        mean = constants.NETDISSECT_RGB_MEAN
        std = constants.NETDISSECT_RGB_STD
    else:
        raise ValueError(f"No model {slug} available.")

    model = model.eval()

    # Generally, our purpose doesn't need grad of weights
    # Ref: https://github.com/chr5tphr/zennit/blob/648b24aefa2292064d871cf41482901d10115513/share/example/feed_forward.py#L139
    for param in model.parameters():
        param.requires_grad = False

    # Ref: we use this signature frrom https://github.com/davidbau/dissect/blob/9421eaa8672fd051088de6c0225a385064070935/experiment/oldvgg16.py#L4
    model.features = torch.nn.Sequential(
        collections.OrderedDict(
            zip(
                [
                    "conv1_1",
                    "relu1_1",
                    "conv1_2",
                    "relu1_2",
                    "pool1",
                    "conv2_1",
                    "relu2_1",
                    "conv2_2",
                    "relu2_2",
                    "pool2",
                    "conv3_1",
                    "relu3_1",
                    "conv3_2",
                    "relu3_2",
                    "conv3_3",
                    "relu3_3",
                    "pool3",
                    "conv4_1",
                    "relu4_1",
                    "conv4_2",
                    "relu4_2",
                    "conv4_3",
                    "relu4_3",
                    "pool4",
                    "conv5_1",
                    "relu5_1",
                    "conv5_2",
                    "relu5_2",
                    "conv5_3",
                    "relu5_3",
                    "pool5",
                ],
                model.features,
            )
        )
    )

    model.classifier = torch.nn.Sequential(
        collections.OrderedDict(
            zip(
                ["fc6", "relu6", "drop6", "fc7", "relu7", "drop7", "fc8a"],
                model.classifier,
            )
        )
    )

    # These reize and crop are mainly for visualization
    rc_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
        ]
    )

    input_transform = T.Compose(
        [
            rc_transform,
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    )

    return model, (rc_transform, input_transform)


def split_model_at_layer(model: torchvision.models.VGG, layer: str):
    # this part of code split model at a given layer.
    # The computational flow must match https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py#L65.

    assert layer in ["conv3_3", "conv4_3", "conv5_3"]

    # We have + 1 for the layer because the actual activation from NetDissect is taken from there
    # due to the inplace operator of the next ReLU layer.
    if layer == "conv3_3":
        layer_ix = 14 + 1
    elif layer == "conv4_3":
        layer_ix = 21 + 1
    elif layer == "conv5_3":
        layer_ix = 28 + 1

    # we have another +1 here because the index is NOT inclusive.
    feature_extractor = model.features[: layer_ix + 1]

    _modules = list(feature_extractor.named_modules())

    assert _modules[-2][0] == layer and _modules[-1][0] == layer.replace(
        "conv", "relu"
    ), "sanity check: the last module is at the one we want"

    classification_head = torch.nn.Sequential(
        model.features[layer_ix + 1 :],
        model.avgpool,
        torch.nn.Flatten(start_dim=1),
        model.classifier,
    )

    return feature_extractor, classification_head
