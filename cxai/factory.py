import torch
import torchvision

from cxai import models, explainers
from cxai import utils as putils

AVAILABLE_ARCHITECTURES = [
    "torchvision-vgg16-imagenet",
    "netdissect-vgg16-imagenet",
]


def make_model(arch: str):
    """
    Args:
        arch (str):

    Returns:
        torch.nn.Sequential: _description_
        Tuple(transform, transform): 1) resize and cropping; 2) resize, cropping and normalization
    """

    assert arch in AVAILABLE_ARCHITECTURES

    if "dm_nfnet" in arch:
        model = models.nfnet.get_model(arch)
        (rc_transform, input_transform) = models.nfnet.get_transformation(model)

    elif "vgg16" in arch:
        model, (rc_transform, input_transform) = models.vgg16.get_model(arch)

    else:
        raise ValueError(f"{model} not available.")

    setattr(model, models.ATTRIBUTE_TRANSFORMATION, (rc_transform, input_transform))

    return model, (rc_transform, input_transform)


def make_explainer(name: str, model: torch.nn.Sequential):
    if "lrp" in name:
        if name == "lrp" and isinstance(model, torchvision.models.VGG):
            return explainers.lrp.vgg16.VGGLRPExplainer(model)
            # return explainers.VGGLRPExplainer(model)
        else:
            raise ValueError("{name} for LRP is not available!")

    raise ValueError(f"Combination of attr-method={name} and {model} is not available!")


def make_label_desc(dataset: str):
    if dataset == "imagenet":
        return lambda label: putils.imagenet.ix_to_classname[label]
