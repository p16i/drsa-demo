import numpy as np

import timm

from timm.models.nfnet import (
    GammaAct,
)

from cxai.models import nfnet

from ..base import LRPExplainerWithInspector
from ..layer_inspection_context import LayerInspectionContext


from . import utils as nfnetlrp_utils
from . import forwardhooks as fh


FORWARDHOOK_COLLECTION = "__forward_hooks"


class Explainer(LRPExplainerWithInspector):
    def __init__(self, model, lrp_gamma: float, verbose=False) -> None:

        assert type(model) == timm.models.nfnet.NormFreeNet

        rc_transform, input_transform = nfnet.get_transformation(model)

        super().__init__(
            model,
            first_layer=model.stem.conv1,
            rc_transform=rc_transform,
            input_transform=input_transform,
            verbose=verbose,
        )

        self.lrp_gamma = lrp_gamma

        # get modules that we will apply LRP
        # this part is quite crucial
        # TODO [#40]: think about how to verify that we have all the modules
        relevant_modules_from_stem = nfnetlrp_utils.get_stem_conv_and_gamma_layers(
            self.model, self.verbose
        )

        num_stages = len(self.model.stages)
        assert num_stages == 4, "sanity check!"

        relevant_modules_from_stages = list(
            map(
                lambda stage_ix: nfnetlrp_utils.get_stage_conv_and_gamma_layers(
                    self.model, stage_ix, self.verbose
                ),
                range(num_stages),
            )
        )

        self.layers_scaled_conv = [
            *relevant_modules_from_stem["conv"],
            *relevant_modules_from_stages[0]["conv"],
            *relevant_modules_from_stages[1]["conv"],
            *relevant_modules_from_stages[2]["conv"],
            *relevant_modules_from_stages[3]["conv"],
            self.model.final_conv,
        ]

        self.layers_gamma_act = [
            *relevant_modules_from_stem["act"],
            *relevant_modules_from_stages[0]["act"],
            *relevant_modules_from_stages[1]["act"],
            *relevant_modules_from_stages[2]["act"],
            *relevant_modules_from_stages[3]["act"],
            self.model.final_act,
        ]

        self.layers_attn_last = [
            *relevant_modules_from_stages[0]["attn_last"],
            *relevant_modules_from_stages[1]["attn_last"],
            *relevant_modules_from_stages[2]["attn_last"],
            *relevant_modules_from_stages[3]["attn_last"],
        ]

        self.layers_shortcut = [
            *relevant_modules_from_stages[0]["shortcut"],
            *relevant_modules_from_stages[1]["shortcut"],
            *relevant_modules_from_stages[2]["shortcut"],
            *relevant_modules_from_stages[3]["shortcut"],
        ]

        self.layers_pooling = [
            *relevant_modules_from_stages[0]["pooling"],
            *relevant_modules_from_stages[1]["pooling"],
            *relevant_modules_from_stages[2]["pooling"],
            *relevant_modules_from_stages[3]["pooling"],
            self.model.head.global_pool,
        ]

        if verbose:
            cases = (
                ("scaled_conv", self.layers_scaled_conv),
                ("gamma_act", self.layers_gamma_act),
                ("attn_last", self.layers_attn_last),
                ("shortcut", self.layers_shortcut),
                ("pooling", self.layers_pooling),
            )

            print("Number of modules found for these cases")
            for name, modules in cases:
                print(f"> case={name:15s}: {len(modules)}")

        self.gamma_act = GammaAct(
            act_type="gelu", gamma=self.layers_gamma_act[0].gamma, inplace=False
        )

    def __enter__(self):
        # setup hooks
        all_hooks = [
            # LRP-Beta (input layer)
            self.model.stem.conv1.register_forward_hook(
                fh.make_forward_hook(
                    fh.case_conv_input,
                    gamma_act=self.gamma_act,
                    mean=self.mean,
                    std=self.std,
                )
            ),
            *list(
                map(
                    lambda cl: cl.register_forward_hook(
                        fh.make_forward_hook(
                            fh.case_conv_lrp_gamma,
                            gamma_act=self.gamma_act,
                            gamma=self.lrp_gamma,
                        )
                    ),
                    self.layers_scaled_conv,
                )
            ),
            *list(
                map(
                    lambda cl: cl.register_forward_hook(fh.case_gamma_act),
                    self.layers_gamma_act,
                )
            ),
            *list(
                map(
                    lambda cl: cl.register_forward_hook(fh.case_attn_last),
                    self.layers_attn_last,
                )
            ),
            *list(
                map(
                    lambda cl: cl.register_forward_hook(
                        fh.make_forward_hook(
                            fh.case_shortcut_connection, lrp_gamma=self.lrp_gamma
                        )
                    ),
                    self.layers_shortcut,
                )
            ),
            *list(
                map(
                    lambda cl: cl.register_forward_hook(
                        fh.make_forward_hook(
                            fh.case_pooling_layer, lrp_gamma=self.lrp_gamma
                        )
                    ),
                    self.layers_pooling,
                )
            ),
            self.model.head.fc.register_forward_hook(
                fh.make_forward_hook(fh.case_fc, lrp_gamma=self.lrp_gamma)
            ),
        ]

        setattr(self, FORWARDHOOK_COLLECTION, all_hooks)

    def __exit__(self, *args):
        all_hooks = getattr(self, FORWARDHOOK_COLLECTION)
        for hook in all_hooks:
            hook.remove()

        # make sure that we clear up temporary variables and all the forward hooks
        if hasattr(self.model.stem.conv1, nfnetlrp_utils.LRP_BETA_ATTRIBUTE):
            delattr(self.model.stem.conv1, nfnetlrp_utils.LRP_BETA_ATTRIBUTE)

        # as well as its collection!
        delattr(self, FORWARDHOOK_COLLECTION)

    def inspect_layer(self, layer):
        if layer == "stage3":
            # This is the layer that the activation we want is fed as `input`
            # Ref: https://github.com/rwightman/pytorch-image-models/blob/v0.4.9/timm/models/nfnet.py#L565
            return LayerInspectionContext(self, self.model.final_conv)
        elif layer == "stage2":
            # Ref: https://github.com/rwightman/pytorch-image-models/blob/v0.4.9/timm/models/nfnet.py#L358
            return LayerInspectionContext(self, self.model.stages[3][0].act1)
        elif layer == "stage1":
            return LayerInspectionContext(self, self.model.stages[2][0].act1)
        elif layer == "stage0":
            return LayerInspectionContext(self, self.model.stages[1][0].act1)
        elif layer == "stem":
            return LayerInspectionContext(self, self.model.stages[0][0].act1)
        else:
            raise ValueError(f"Inspecting of NFNet's Layer `{layer}` is NOT valid")
