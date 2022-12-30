from typing import Callable, Tuple
import torch

from timm.models.nfnet import NormFreeNet
from torchvision.models import VGG

ATTRIBUTE_TRANSFORMATION = "__transformation"


from . import nfnet
from . import vgg16


def split_model_at_layer(
    model: torch.nn.Module, layer: str
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """The function split a model into two components, namely
    feature extractor and classification head.

    Remark: The prediction of the composition of these two components must be the same
    as the prediction of the original model.

    Args:
        model (torch.nn.Module): model instance we want to split it in two components.
        layer (str): at which layer we want to split the model.
            Remark: the layer will be included in the first component.

    Returns:
        Tuple[torch.nn.Module, torch.nn.Module]: feature_extractor and classification head
    """

    if isinstance(model, VGG):
        return vgg16.split_model_at_layer(model, layer)
    elif isinstance(model, NormFreeNet):
        return nfnet.split_model_at_layer(model, layer)
    else:
        raise ValueError(f"We can't split {model} at {layer}.")
