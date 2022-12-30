from typing import Tuple

import torch
import timm

from torchvision import transforms

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class Output4D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert len(x.shape) == 2, "Our input has two dimensions"

        return x.reshape((*x.shape, 1, 1))


def get_transformation(model):
    """
    input: nfnet model object (not string)
    output:
    - transform for resizing and center cropping
    - transform containg above and to tensor and normalization
    """
    # Reference: https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/docs/models/.templates/code_snippets.md#how-do-i-use-this-model-on-an-image
    config = resolve_data_config({}, model=model)
    nfnet_transform = create_transform(**config)

    resize_and_crop = nfnet_transform.transforms[:2]

    # this is only for sanity check, i.e. when timm changes the way they construction transformations
    for t in resize_and_crop:
        if not type(t) in [transforms.CenterCrop, transforms.Resize]:
            raise ValueError("{t} is not a valid type of transform we want")

    rc_transform = transforms.Compose(resize_and_crop)
    return (rc_transform, nfnet_transform)


def get_model(arch: str):
    nfnet = timm.create_model(arch, pretrained=True)
    nfnet = nfnet.eval()

    # Generally, our purpose doesn't need grad of weights
    # Ref: https://github.com/chr5tphr/zennit/blob/648b24aefa2292064d871cf41482901d10115513/share/example/feed_forward.py#L139
    for param in nfnet.parameters():
        param.requires_grad = False

    # Important remark: there several places that we have activation(inplace=True)
    # This can affect the implementation of some forward hook functions.
    # However, for GeLU, the inplace option has NO influence.
    # See
    # - https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/activations.py#L138
    # - https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html#torch.nn.functional.gelu

    return nfnet


def get_nfnet_at_stage(arch, layer):
    raise NotImplementedError("obsolete")
    model = get_model(arch)

    input_transform = get_transformation(model)

    print(f"Loading NFNet variant: {arch}-{layer}")
    if layer == "logit":
        # here, we make sure that the output of the logit layer has a 4d tensor.
        # this eases downstream analysis
        submodel = torch.nn.Sequential(model, Output4D())
    elif layer == "stem":
        submodel = model.stem
    elif "stage" in layer[:5]:
        # we expect layer to be something like `stage3`
        stage_ix = int(layer.split("stage")[1])

        # nfnet has only 4 stages
        assert stage_ix in list(range(4))

        # here, we take the model inclusively upto the stage
        submodel = torch.nn.Sequential(model.stem, *model.stages[: stage_ix + 1])
    else:
        raise ValueError(f"Hey!, {arch} doesn't have `{layer}`.")

    return submodel, input_transform


def split_model_at_layer(
    model: torch.nn.Module, layer: str
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    assert isinstance(model, timm.models.nfnet.NormFreeNet)

    # sanity checks
    assert "stage" in layer

    # we expect layer to be something like `stage3`
    stage_ix = int(layer.split("stage")[1])
    # nfnet has only 4 stages
    assert stage_ix in list(range(4))

    # The flow below must match what implemented at
    # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/nfnet.py#L569

    # here, we take the model inclusively upto the stage
    feature_extractor = torch.nn.Sequential(model.stem, *model.stages[: stage_ix + 1])

    classification_head = torch.nn.Sequential(
        *model.stages[stage_ix + 1 :],
        model.final_conv,
        model.final_act,
        model.head,
    )

    return feature_extractor, classification_head
