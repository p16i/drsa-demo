from functools import partial

from collections import namedtuple

import torch
import copy

import types
import weakref

LRP_BETA_ATTRIBUTE = "__act_for_lrp_beta"

# This cell allows use to create symbolic links between cases and their named used in the code.
# It makes sure there will be only allowed case named used.
NFNET_LRP_CASE_NAMES = ("orphan_conv", "orphan_act")

NFNET_LRP_CASES = namedtuple("LRP_NFNet_Cases", NFNET_LRP_CASE_NAMES)(
    *(map(lambda c: f"__case_{c}", NFNET_LRP_CASE_NAMES))
)

# This module trivially represents the last operator in NFNets
class Summation(torch.nn.Module):
    def forward(self, a, s):
        return a + s


# This function is almost the same as the original NormFreeBlock's `.forward()`,
# except the last step that is replaced with Summation().
# See: https://github.com/rwightman/pytorch-image-models/blob/v0.4.9/timm/models/nfnet.py#L357
# Remark: The exact version of `timm` is important here.
def overrided_normfreeblock_forward(self, x):
    out = self.act1(x) * self.beta

    # shortcut branch
    shortcut = x
    if self.downsample is not None:
        shortcut = self.downsample(out)

    # residual branch
    out = self.conv1(out)
    out = self.conv2(self.act2(out))
    if self.conv2b is not None:
        out = self.conv2b(self.act2b(out))
    if self.attn is not None:
        out = self.attn_gain * self.attn(out)
    out = self.conv3(self.act3(out))
    if self.attn_last is not None:
        out = self.attn_gain * self.attn_last(out)
    out = self.drop_path(out)

    if self.skipinit_gain is not None:
        out.mul_(self.skipinit_gain)  # this slows things down more than expected, TBD

    # this is the only line that is different from the original function.
    out = self.shortcut_summation(out * self.alpha, shortcut)

    return out


def get_stage_conv_and_gamma_layers(model, stage_ix: int, verbose=False):

    results = dict(conv=[], act=[], attn_last=[], shortcut=[], pooling=[])
    for nfblock in model.stages[stage_ix]:
        if nfblock.downsample is not None:
            setattr(nfblock.downsample.conv, NFNET_LRP_CASES.orphan_conv, True)

            results["conv"].append(nfblock.downsample.conv)

            if type(nfblock.downsample.pool) != torch.nn.Identity:
                results["pooling"].append(nfblock.downsample.pool)

        setattr(nfblock.act1, NFNET_LRP_CASES.orphan_act, True)
        setattr(nfblock.conv3, NFNET_LRP_CASES.orphan_conv, True)

        # Having this module allows us to implement LRP for the the shortcut connection.
        nfblock.shortcut_summation = Summation()
        results["shortcut"].append(nfblock.shortcut_summation)

        # this line below overrides the original `.forward()` with
        # the one that last step uses `Summation()`.
        # todo: we need to check this line again; maybe there is a better way.
        nfblock.forward = types.MethodType(
            overrided_normfreeblock_forward, weakref.proxy(nfblock)
        )

        results["act"].extend([nfblock.act1, nfblock.act2, nfblock.act2b, nfblock.act3])

        results["conv"].extend(
            [nfblock.conv1, nfblock.conv2, nfblock.conv2b, nfblock.conv3]
        )
        results["attn_last"].append(nfblock.attn_last)

    if verbose:
        print(f"[Stage {stage_ix}]")
        for k, v in results.items():
            print(f">  {k:10s} : {len(v)}")

    return results


def get_stem_conv_and_gamma_layers(model, verbose=False):
    setattr(model.stem.conv4, NFNET_LRP_CASES.orphan_conv, True)

    results = dict(
        conv=[
            model.stem.conv2,
            model.stem.conv3,
            model.stem.conv4,
        ],
        act=[
            model.stem.act2,
            model.stem.act3,
            model.stem.act4,
        ],
    )

    if verbose:
        print(f"[Stem]")
        for k, v in results.items():
            print(f">  {k:10s} : {len(v)}")

    return results


# This function create a new layer with modified weights.
# We use it from computing the value associated with w_{jk}^+ and w_{jk}^-
# It is adapted from https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/main/utils.py#L64
def make_new_conv_layer(oldlayer, g, without_bias=False):
    layer = copy.deepcopy(oldlayer)

    # remove hooks
    layer._forward_hooks = None

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * F.layer_norm(
                self.weight, self.weight.shape[1:], eps=self.eps
            )
        else:
            std, mean = torch.std_mean(
                self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False
            )
            weight = self.scale * (self.weight - mean) / (std + self.eps)
        return self.gain * g(weight)

    if without_bias:
        layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))
        layer.bias.requires_grad_(False)

    # overrided the scaled implementation
    layer.get_weight = lambda: get_weight(layer)

    return layer


def make_new_fc(module, g, with_bias=True):
    copied_module = copy.deepcopy(module)
    # because the copied module comes with forwardhook, we have to remove it
    # to prevent recuively call the hook.
    copied_module._forward_hooks = None

    copied_module.weight = torch.nn.Parameter(g(copied_module.weight))
    copied_module.weight.requires_grad_(False)

    if not with_bias:
        copied_module.bias = torch.nn.Parameter(torch.zeros_like(copied_module.bias))
        copied_module.bias.requires_grad_(False)

    return copied_module


def get_modified_conv_layer(conv_layer, gamma):
    # We extract the positive and negative weights to the convolutional layers
    def _rho_p(g, w):
        return w + g * w.clamp(min=0)

    def _rho_n(g, w):
        return w + g * w.clamp(max=0)

    rho_p = partial(_rho_p, gamma)
    rho_n = partial(_rho_n, gamma)

    return (
        make_new_conv_layer(conv_layer, rho_p),
        make_new_conv_layer(conv_layer, rho_p, without_bias=True),
        make_new_conv_layer(conv_layer, rho_n),
        make_new_conv_layer(conv_layer, rho_n, without_bias=True),
    )


def get_modified_conv_for_lrp_beta(conv, mean, std):
    lb_layer = make_new_conv_layer(conv, lambda p: p.clamp(min=0))
    hb_layer = make_new_conv_layer(conv, lambda p: p.clamp(max=0))

    # Ref: Cell 18 in https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/main/tutorial.ipynb
    lb = (0 - mean) / std
    hb = (1 - mean) / std

    return lb_layer, hb_layer, lb, hb
