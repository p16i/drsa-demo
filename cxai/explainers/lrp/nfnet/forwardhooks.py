from functools import partial
import copy

import torch

from ..layer_inspection_context import LayerInspectionContext
from ..base import LRPExplainerWithInspector, lrp_rule_ratio

from . import utils as nfnetlrp_utils

EPS = 1e-3


def make_forward_hook(fh_hook, **kwargs):
    return partial(fh_hook, **kwargs)


# Case: `Conv` (LRP-$\gamma$) and Orphan `Conv` (LRP-$\gamma_z$)
# It contains 2 subcases, namely
# 1. Regular: these are convs that are followed by act;
# 2. Orphan Conv: these are convs that are NOT followed by act.
# The difference between the two cases is which context variable we use.
# The former use ak while the latter uses zk.

# It also contains 1 crucial condition, namely
# 1. if the module is inspected, we store its input for further usage in the inspection.
def case_conv_lrp_gamma(module, input, output, gamma_act, gamma):
    (aj,) = input
    # the is the output of Conv before GammaAct
    zk = output

    # prepare for positive and negative weights
    pconv, pconvnb, nconv, nconvnb = nfnetlrp_utils.get_modified_conv_layer(
        module, gamma
    )

    # if we intercept this layer, we will also get explanation for this layer
    # More precisely, this is for stage 3 inspection
    if hasattr(module, LayerInspectionContext.IS_INSPECTED) and getattr(
        module, LayerInspectionContext.IS_INSPECTED
    ):
        aj.retain_grad()
        setattr(module, LayerInspectionContext.IS_INSPECTED_ACTIVATION, aj)

    assert not torch.isnan(zk).any(), "zk is nan!"

    def get_pos_neg_part(aj):
        aj_p = aj.clamp(min=0)
        aj_n = aj.clamp(max=0)

        # Positive case (z_k > 0)
        # This is the cases that the signs of a_j and w_{jk} align.
        zk_pp = pconv(aj_p)
        # We exluce bias here because we wil bias to be a neuron with weight 1.
        # Effectively, it means that  aj^- = 0 => aj^- x b = 0.
        zk_nn = nconvnb(aj_n)

        # Negative case (z_k < 0)
        zk_np = nconv(aj_p)
        zk_pn = pconvnb(aj_n)

        pos_part = zk_pp + zk_nn
        assert not torch.isnan(pos_part).any(), "pos_part is nan!"

        neg_part = zk_np + zk_pn
        assert not torch.isnan(neg_part).any(), "neg_part is nan!"

        return pos_part, neg_part

    pos_part, neg_part = get_pos_neg_part(aj)

    if hasattr(module, nfnetlrp_utils.NFNET_LRP_CASES.orphan_conv):
        output = zk
    else:
        ak = gamma_act(zk)
        assert not torch.isnan(ak).any(), "ak is nan!"

        output = ak

    overriden_ak = lrp_rule_ratio(
        pos_part, output.clamp(min=0), eps=EPS
    ) + lrp_rule_ratio(neg_part, output.clamp(max=0), eps=EPS)

    assert torch.allclose(
        overriden_ak, output
    ), "Overriden output is the same as original."

    return overriden_ak


# Case: Orphan  `GammaAct` and `GammaAct` of LRP-$\gamma$
def case_gamma_act(module, input, output):
    zk = input[0]
    ak = output[0]

    # if we intercept this layer, we will also get explanation for this layer
    # More precisely, this is for `stage 2` inspection
    if hasattr(module, LayerInspectionContext.IS_INSPECTED) and getattr(
        module, LayerInspectionContext.IS_INSPECTED
    ):
        zk.retain_grad()
        setattr(module, LayerInspectionContext.IS_INSPECTED_ACTIVATION, zk)

    if hasattr(module, nfnetlrp_utils.NFNET_LRP_CASES.orphan_act):
        # This is the case when GammaAct is on it owns.
        # e.g., normfreeblock.act1
        overriden_ak = lrp_rule_ratio(zk, ak, eps=EPS)

        assert torch.allclose(
            overriden_ak, ak
        ), "Overriden output is the same as original."

        return overriden_ak
    else:
        # This is the case when GammaAct follows a Conv; this is common cases.
        # In this case, we override the module to be an identity map.
        return zk


# Case: `Attn_Last` (also known as `Self-Exciation` Block)
def case_attn_last(module, input, output):
    (aj,) = input
    ak = output[0]

    # For some reason, aj has batch axis and ak doesn't.
    assert (
        aj.shape[0] == 1 and aj.squeeze().shape == ak.shape
    ), f"sanity check aj.shape={aj.shape}, ak.shape={ak.shape}"

    overriden_ak = lrp_rule_ratio(aj, ak, eps=EPS)

    assert torch.allclose(overriden_ak, ak), "overriden output is the same as original"

    return overriden_ak


# Case: ShortCut Connection
def case_shortcut_connection(module, input, output, lrp_gamma):
    aj, sj = input
    zk = output

    ajp = aj.clamp(min=0)
    sjp = sj.clamp(min=0)

    asp_0 = ajp + sjp
    asp_gamma = asp_0 * (1 + lrp_gamma)

    ajn = aj.clamp(max=0)
    sjn = sj.clamp(max=0)

    asn_0 = ajn + sjn
    asn_gamma = asn_0 * (1 + lrp_gamma)

    pos_part = asp_gamma + asn_0
    neg_part = asn_gamma + asp_0

    overriden_zk = lrp_rule_ratio(pos_part, zk.clamp(min=0), eps=EPS) + lrp_rule_ratio(
        neg_part, zk.clamp(max=0), eps=EPS
    )

    assert torch.allclose(overriden_zk, zk), "Overriden zk is the same as the original."

    return overriden_zk


# Case: Pooling Layer
def case_pooling_layer(module, input, output, lrp_gamma):
    (aj,) = input
    zk = output

    shadow_module = copy.deepcopy(module).eval()
    # because the copied module comes with forwardhook, we have to remove it
    # to prevent recuively call the hook.
    shadow_module._forward_hooks = None

    ajp = aj.clamp(min=0)
    ajn = aj.clamp(max=0)

    z_ajp_0 = shadow_module(ajp)
    z_ajn_0 = shadow_module(ajn)

    # essentially, we view pooling as convolution with all weights equal to 1/S
    # where S is the size of the receptive field
    z_ajp_gamma = z_ajp_0 * (1 + lrp_gamma)
    z_ajn_gamma = z_ajn_0 * (1 + lrp_gamma)

    pos_part = z_ajp_gamma + z_ajn_0
    neg_part = z_ajn_gamma + z_ajp_0

    overriden_zk = lrp_rule_ratio(pos_part, zk.clamp(min=0), eps=EPS) + lrp_rule_ratio(
        neg_part, zk.clamp(max=0), eps=EPS
    )

    assert torch.allclose(
        overriden_zk,
        zk,
    ), "Overriden zk is the same as the original."

    return overriden_zk


def case_fc(module, input, output, lrp_gamma):
    (aj,) = input
    zk = output

    pfc = nfnetlrp_utils.make_new_fc(module, lambda w: w + lrp_gamma * w.clamp(min=0))
    pfc_nb = nfnetlrp_utils.make_new_fc(
        module, lambda w: w + lrp_gamma * w.clamp(min=0), with_bias=False
    )

    nfc = nfnetlrp_utils.make_new_fc(module, lambda w: w + lrp_gamma * w.clamp(max=0))
    nfc_nb = nfnetlrp_utils.make_new_fc(
        module, lambda w: w + lrp_gamma * w.clamp(max=0), with_bias=False
    )

    aj_p = aj.clamp(min=0)
    aj_n = aj.clamp(max=0)

    zk_pp = pfc(aj_p)
    zk_np = nfc(aj_p)

    # here, we view biases as neurons whose values are 1.
    # That means ajn should be compute without biases.
    zk_pn = pfc_nb(aj_n)
    zk_nn = nfc_nb(aj_n)

    # If output is posive, we consider contributions from (input, weight) that align, e.g., (+,+) or (-,-).
    pos_part = zk_pp + zk_nn
    assert not torch.isnan(pos_part).any(), "pos_part is nan!"

    # If output is negative, we consider contributions from (input, weight) that do NOT align (+,-) or (-,+).
    neg_part = zk_np + zk_pn
    assert not torch.isnan(neg_part).any(), "neg_part is nan!"

    overriden_zk = lrp_rule_ratio(pos_part, zk.clamp(min=0), eps=EPS) + lrp_rule_ratio(
        neg_part, zk.clamp(max=0), eps=EPS
    )

    assert torch.allclose(overriden_zk, zk), "Overriden zk is the same as original zk."

    return overriden_zk


# Case: LRP-$\mathcal B$ (for input layer)
def case_conv_input(module, input, output, gamma_act, mean, std):
    (aj,) = input
    (zk,) = output

    ak = gamma_act(zk)

    lb_layer, hb_layer, lb, hb = nfnetlrp_utils.get_modified_conv_for_lrp_beta(
        module, mean, std
    )

    lbv = (aj.data * 0 + lb.reshape((1, 3, 1, 1)).to(aj.device)).requires_grad_(True)
    hbv = (aj.data * 0 + hb.reshape((1, 3, 1, 1)).to(aj.device)).requires_grad_(True)

    zk_lb = lb_layer(lbv)
    zk_hb = hb_layer(hbv)

    z = zk - (zk_lb + zk_hb)

    assert not torch.isnan(zk).any(), "output contain nan!"

    overriden_ak = lrp_rule_ratio(z, ak, eps=EPS)

    assert torch.allclose(overriden_ak, ak), "Overriden zk is the same as the original."

    # this is variable will be used to get final explanation
    setattr(module, LRPExplainerWithInspector.LRP_BETA_ATTRIBUTE, (aj, lbv, hbv))

    return overriden_ak
