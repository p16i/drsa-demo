# This package is largely based on https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/main/utils.py#L64.
# The significant difference between the original and this code base is that
# here we directly utilizes Pytorch's forward hooks.
from functools import partial

import numpy as np
import torch
import torchvision

from cxai import models

from cxai.inspector import Inspector, InspectionRelevanceInfo

from ..base import LRPExplainerWithInspector
from ..layer_inspection_context import LayerInspectionContext

from . import forwardhooks as fh

FORWARDHOOK_COLLECTION = "__forward_hooks"


def make_rho(gamma: float):
    def rho(p: torch.tensor) -> torch.tensor:
        return p + gamma * p.clamp(min=0)

    return rho


def make_incr(eps: float):
    def incr(z: torch.tensor) -> torch.tensor:
        return z + eps

    return incr


class VGGLRPExplainer(LRPExplainerWithInspector):
    def __init__(self, model: torch.nn.Sequential, verbose=False):
        # todo: perhaps, we might also be better to assert that it is VGG16 only.
        assert isinstance(model, torchvision.models.VGG)

        rc_transform, input_transform = getattr(model, models.ATTRIBUTE_TRANSFORMATION)

        super().__init__(
            model=model,
            first_layer=model.features.conv1_1,
            rc_transform=rc_transform,
            input_transform=input_transform,
            verbose=verbose,
        )

    def inspect_layer(self, layer):
        # we take the pooling layers because
        # 1) NetDissect-Lite attaches a forward hook to the conv module and store the ouput
        # 2) and the next ReLU is in place
        # Therefore, this output is equivalent to the input of the next after which is the pooling layer.
        # Ref: https://github.com/CSAILVision/NetDissect-Lite/blob/2163454ebeb5b15aac64e5cbd4ed8876c5c200df/feature_operation.py#L16

        # So, we attach at the next pooling layer because we want the output
        # from conv, which is the input of the pooling layer.
        if layer == "conv5_3":
            return LayerInspectionContext(self, self.model.features.pool5)
        elif layer == "conv4_3":
            return LayerInspectionContext(self, self.model.features.pool4)
        elif layer == "conv3_3":
            return LayerInspectionContext(self, self.model.features.pool3)
        else:
            raise ValueError(f"no inspection context for `{layer}`")

    def __enter__(self):
        hooks = VGGLRPExplainer.setup_forward_hooks(
            self.model, self.mean, self.std, self.verbose
        )

        setattr(self, FORWARDHOOK_COLLECTION, hooks)

    def __exit__(self, *args):
        hooks = getattr(self, FORWARDHOOK_COLLECTION)

        for hook in hooks:
            hook.remove()

        # make sure that we clear up temporary variables and all the forward hooks
        if hasattr(self.first_layer, LRPExplainerWithInspector.LRP_BETA_ATTRIBUTE):
            delattr(self.first_layer, LRPExplainerWithInspector.LRP_BETA_ATTRIBUTE)

        # as well as its collection!
        delattr(self, FORWARDHOOK_COLLECTION)

    @staticmethod
    def setup_forward_hooks(
        model: torch.nn.Sequential, mean: torch.tensor, std: torch.tensor, verbose: bool
    ):
        hooks = []

        # This is basically based on Cell 15 of https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/tree/main.
        all_layers = list(model.features) + list(model.classifier)

        # Remark: the entry computational graph is model.features -> avgpool -> classifier
        # and here we do NOT attach any forward hook to the avgpooling because
        # its standard gradient is already what we want.
        for ix, layer in enumerate(all_layers):
            hook_func = None
            if isinstance(layer, torch.nn.MaxPool2d):
                hook_func = partial(fh.case_pooling)
            elif isinstance(layer, torch.nn.Conv2d):
                if layer == model.features.conv1_1:
                    hook_func = partial(
                        fh.case_conv_input,
                        rho=make_rho(0.0),
                        # Ref: Cell 18 in https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/main/tutorial.ipynb.
                        lb=(0 - mean) / std,
                        hb=(1 - mean) / std,
                    )
                else:
                    # The heuristics is similar to https://github.com/oeberle/BiLRP_explain_similarity/blob/master/model/bilrp.py#L10.
                    # BUT, we correct the layer indices.
                    # See https://colab.research.google.com/drive/1wcbbMB3mR_T-889JcXqxMbU7OImoDmM8#scrollTo=lP-3ayktTets
                    if 0 <= ix <= 9:
                        gamma = 0.5
                    elif 10 <= ix <= 16:
                        gamma = 0.25
                    elif 17 <= ix <= 23:
                        gamma = 0.1
                    elif ix >= 24:
                        gamma = 0.0

                    hook_func = partial(
                        fh.case_conv_or_fc, rho=make_rho(gamma), incr=make_incr(1e-9)
                    )
            elif isinstance(layer, torch.nn.Linear):
                hook_func = partial(
                    fh.case_conv_or_fc, rho=make_rho(0.0), incr=make_incr(1e-9)
                )
            else:
                # sanity checks
                assert isinstance(layer, torch.nn.ReLU) or isinstance(
                    layer, torch.nn.Dropout
                )

                if verbose:
                    print(f"We do not attach any forward hook to: [ix={ix}] {layer}")

            if hook_func is not None:
                hook = layer.register_forward_hook(hook_func)
                hooks.append(hook)

        if verbose:
            print(f"We have attached in total `{len(hooks)}` forward hooks.")

        return hooks
