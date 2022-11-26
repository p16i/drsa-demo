from typing import Callable

import copy
import torch
from torch import nn

from . import LRPExplainerWithInspector
from ..base import lrp_rule_ratio
from ..layer_inspection_context import LayerInspectionContext

EPS = 0


# Ref: https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/main/utils.py#L64
def make_new_layer(oldlayer: nn.Module, g: Callable) -> nn.Module:
    """This is a utility function to create a shadow module
    whose parameters (weights and biases) are modified with g().

    Args:
        oldlayer (torch.nn.Module): module to be copied
        g (Callable): function to modify weights and

    Returns:
        torch.nn.Module
    """

    layer = copy.deepcopy(oldlayer)

    # remove forward hooks
    layer._forward_hooks = None

    try:
        layer.weight = torch.nn.Parameter(g(layer.weight))
    except AttributeError:
        pass

    try:
        layer.bias = torch.nn.Parameter(g(layer.bias))
    except AttributeError:
        pass

    return layer


def case_pooling(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
    # This is the hook for simply getting the intermediate activation.
    (aj,) = input

    if hasattr(module, LayerInspectionContext.IS_INSPECTED) and getattr(
        module, LayerInspectionContext.IS_INSPECTED
    ):
        aj.retain_grad()
        setattr(module, LayerInspectionContext.IS_INSPECTED_ACTIVATION, aj)


def case_conv_or_fc(
    module: nn.Module,
    input: torch.Tensor,
    output: torch.Tensor,
    rho: Callable,
    incr: Callable,
):
    (aj,) = input
    ak = output

    shadow_module = make_new_layer(module, rho)

    zk = incr(shadow_module(aj))

    overridden_ak = lrp_rule_ratio(zk, ak, eps=EPS)

    assert torch.allclose(
        overridden_ak, ak
    ), "overriden_ak is the same as the original ak"

    return overridden_ak


def case_conv_input(
    module: nn.Module,
    input: torch.Tensor,
    output: torch.Tensor,
    rho: Callable,
    lb: torch.Tensor,
    hb: torch.Tensor,
    eps=1e-9,
):
    lb_layer = make_new_layer(module, lambda p: p.clamp(min=0))
    hb_layer = make_new_layer(module, lambda p: p.clamp(max=0))

    shadow_module = make_new_layer(module, rho)

    (aj,) = input
    ak = output

    lb = lb.reshape((1, 3, 1, 1)).to(aj.device)
    hb = hb.reshape((1, 3, 1, 1)).to(aj.device)

    lbv = (aj.data * 0 + lb).requires_grad_(True)
    hbv = (aj.data * 0 + hb).requires_grad_(True)

    z = shadow_module(aj) + eps
    z -= lb_layer(lbv)
    z -= hb_layer(hbv)

    # this is variable will be used to get final explanation
    setattr(module, LRPExplainerWithInspector.LRP_BETA_ATTRIBUTE, (aj, lbv, hbv))

    overridden_ak = lrp_rule_ratio(z, ak, eps=EPS)

    assert torch.allclose(
        overridden_ak, ak
    ), "overridden_ak is the same as the original ak"

    return overridden_ak
