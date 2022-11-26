from typing import Tuple, List
from nptyping import NDArray
import torch

import numpy as np

from cxai.inspector import Inspector, InspectionRelevanceInfo
from cxai.explainers.base import Explainer

from .layer_inspection_context import LayerInspectionContext


def zero_grad(arr: Tuple[torch.Tensor]):

    for t in arr:
        t.grad.zero_()


def lrp_rule_ratio(input: torch.Tensor, output: torch.Tensor, eps=0) -> torch.Tensor:
    """This functions provides the calculation of the form

        aj * (c/aj).detach(),

    where aj = \sum_i ai w_ij. This pattern is common in implementing LRP
    using the forward hook tick.

    Args:
        input (torch.Tensor): This is `aj`.
        output (torch.Tensor): This is `c`.
        eps (int, optional): if aj is smaller than eps, then we discard the term. Defaults to 0.

    Returns:
        torch.Tensor: _description_
    """

    # This function does nom * (context / nom).detach().
    output = output.clone().detach()
    denom = input.detach()

    # Remark: for some reason, torch automatically remove the batch axis of context
    # could this be PyTorch's bug?
    if input.shape[0] == 1 and len(input.shape) == 4 and len(output.shape) == 3:
        output = output.unsqueeze(0)

    # this trick combats getting nan from backprop of x/0.
    # see https://github.com/pytorch/pytorch/issues/4132
    nonzero_ix = denom.abs() > eps

    new_output = torch.zeros_like(output)

    new_output[nonzero_ix] = (input[nonzero_ix] / denom[nonzero_ix]) * output[
        nonzero_ix
    ]

    # Remark: we set the value of these entries to the value of the `detached` output.
    # Therefore, we can assure that all entries in `output` and `new_output` are the same,
    # despite the fact that the backward of those discared entries are now disconnected.
    # This means that their contributions are ignored (because of its contribution is small anyway, < eps)
    new_output[~nonzero_ix] = output[~nonzero_ix]

    return new_output


class LRPExplainerWithInspector(Explainer):
    LRP_BETA_ATTRIBUTE = "__act_for_lrp_beta"

    def __init__(
        self,
        model: torch.nn.Sequential,
        # This is for LRP-beta
        first_layer: torch.nn.Module,
        rc_transform,
        input_transform,
        verbose=False,
    ):
        self.model = model
        self.verbose = verbose
        self.first_layer = first_layer

        self.rc_transform = rc_transform
        self.input_transform = input_transform

        # these are for LRP-beta.
        self.mean = self.input_transform.transforms[-1].mean
        self.std = self.input_transform.transforms[-1].std

    def inspect_layer(self, layer: str) -> LayerInspectionContext:
        raise NotImplementedError()

    def __enter__(self):
        # This is for setup forward hooks.
        raise NotImplementedError()

    def __exit__(self, *args):
        # This is for removing hooks and other auxilarity variables.
        raise NotImplementedError()

    @staticmethod
    def _lrp_beta(x: torch.Tensor, lb: torch.Tensor, hb: torch.Tensor) -> NDArray:
        """This function computes the final relvance from those input tensors

        In particular, we compute

            rel = x * x.grad + lb * lb.grad + hb * hb.grad

        We assume those tensors have `.grad`.

        Args:
            inp (torch.Tensor): input
            lb (torch.Tensor): lower bound
            hb (torch.Tensor): upper bound

        Returns:
            relevance (torch.Tensor): relevance input heatmap
        """

        # Remark:
        #   We have + (R_lb + R_hb) here because the minus sign is already
        # incorporated when we setup the forward hook.
        heatmap = x * x.grad + lb * lb.grad + hb * hb.grad
        return heatmap.detach().cpu().numpy()

    def explain(self, x: torch.Tensor, label: int) -> Tuple[NDArray, NDArray]:

        input_shape = x.shape
        assert len(input_shape) == 3

        with self:
            x = x.unsqueeze(0)
            x = x.detach().requires_grad_(True)

            x.grad = None

            logits = self.model(x)
            logits[:, label].backward()

            logits = logits.squeeze().detach().cpu().numpy()

            aj, lb, hb = getattr(
                self.first_layer, LRPExplainerWithInspector.LRP_BETA_ATTRIBUTE
            )

            assert torch.allclose(x, aj)

            heatmap = LRPExplainerWithInspector._lrp_beta(aj, lb, hb)
            heatmap = heatmap.squeeze()

            assert len(logits.shape) == 1
            assert heatmap.shape == input_shape

        return logits, heatmap

    def explain_with_inspector(
        self, x: torch.Tensor, target_label: int, inspector: Inspector, top_k=5
    ) -> Tuple[NDArray, NDArray, InspectionRelevanceInfo]:
        input_shape = x.shape
        assert len(input_shape) == 3

        # setup forward hooks
        with self:
            x = x.clone()
            x = x.unsqueeze(0).requires_grad_(True)
            x.grad = None

            # setup inspection layer
            with self.inspect_layer(inspector.layer):
                logits = self.model(x)
                assert len(logits.shape) == 2 and logits.shape[0] == 1

                logits = logits.squeeze()

                logits[target_label].backward(retain_graph=True)

                # LRP-beta
                aj, lb, hb = getattr(
                    self.first_layer, LRPExplainerWithInspector.LRP_BETA_ATTRIBUTE
                )

                heatmap_standard = LRPExplainerWithInspector._lrp_beta(aj, lb, hb)
                heatmap_standard = heatmap_standard.squeeze()

                # shape: [b, d, h, w]
                (
                    activation,
                    context,
                ) = self.get_intermediate_activation_and_context_without_forward()

                # shape: [b, d, h, w]
                rel_layer = activation * context
                rel_sum_total_layer = rel_layer.detach().sum().cpu().numpy()

                (
                    top_k_subspaces,
                    relevance_top_k_subspaces,
                ) = inspector.get_top_k_subspaces(activation, context, top_k)

                relevance_all_subspaces = relevance_top_k_subspaces.sum(
                    dim=1, keepdim=True
                )

                relevance_residue = (
                    rel_layer.flatten(start_dim=1).sum(dim=1, keepdim=True)
                    - relevance_all_subspaces
                )

                relevance_top_k_subspaces_with_residue_and_all_subspaces = torch.hstack(
                    [
                        relevance_top_k_subspaces,
                        relevance_residue,
                        relevance_all_subspaces,
                    ]
                )

                arr_heatmaps = np.zeros((top_k + 2, *input_shape))

                for ix in range(arr_heatmaps.shape[0]):
                    zero_grad((aj, lb, hb))

                    is_last_step = ix == arr_heatmaps.shape[0] - 1

                    relevance_top_k_subspaces_with_residue_and_all_subspaces[
                        :, ix
                    ].sum().backward(retain_graph=not is_last_step)

                    heatmap_subspace = LRPExplainerWithInspector._lrp_beta(aj, lb, hb)
                    heatmap_subspace = heatmap_subspace.squeeze()

                    assert heatmap_subspace.shape == input_shape
                    with torch.no_grad():
                        np.testing.assert_allclose(
                            rel_sum_total_layer,
                            (activation * context).sum().detach().cpu().numpy(),
                            err_msg="We incidentally accumulate the relevance of the intermediate activation",
                        )

                    arr_heatmaps[ix, :, :, :] = heatmap_subspace

                arr_subspace_heatmaps = arr_heatmaps[:top_k, :, :, :]

                # reminder: the index of the last entry of subspace heatmaps is top_k - 1.
                # Therefore, the residue and all subspace heatmaps are at top_k and top_k + 1.
                rel_input_subspace_residue = arr_heatmaps[top_k, :, :, :]
                rel_input_all_subspaces = arr_heatmaps[top_k + 1, :, :, :]

                return (
                    logits.detach().cpu().numpy(),
                    heatmap_standard,
                    InspectionRelevanceInfo(
                        rel_per_source=relevance_top_k_subspaces.detach().cpu().numpy(),
                        rel_sum_total_layer=rel_layer.sum().detach().cpu().numpy(),
                        top_k_sources=top_k_subspaces,
                        input_top_k_source_heatmaps=arr_subspace_heatmaps,
                        input_subspace_residue_heatmap=rel_input_subspace_residue,
                        input_all_subspaces_heatmap=rel_input_all_subspaces,
                    ),
                )

    def get_intermediate_activation_and_context(
        self, layer: str, x: torch.Tensor, label: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        with self.inspect_layer(layer) as insp:
            self.explain(x, label)

            # Remark: please be careful that after explain(...), forward hooks for LPR are detached
            # so, the intermediate act and ctx (LRP context) are correct.models
            # Hoewver, if we call backward again,  `input.grad` wouldn't be LRP context
            if not hasattr(self, LayerInspectionContext.LAYER_INSPECTED):
                raise RuntimeError("No intermediate heatmap layer specified")

            module = getattr(self, LayerInspectionContext.LAYER_INSPECTED)
            act = getattr(module, LayerInspectionContext.IS_INSPECTED_ACTIVATION)

        ctx = act.grad

        return act, ctx

    def get_intermediate_activation_and_context_without_forward(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, LayerInspectionContext.LAYER_INSPECTED):
            raise RuntimeError("No intermediate heatmap layer specified")

        module = getattr(self, LayerInspectionContext.LAYER_INSPECTED)
        activation = getattr(module, LayerInspectionContext.IS_INSPECTED_ACTIVATION)

        return activation, activation.grad

    def total_input_relevance(
        self, x: torch.Tensor, label: int, insp: Inspector, top_k: int
    ):

        logits, heatmap, info = self.explain_with_inspector(x, label, insp, top_k=top_k)

        # aka. logit
        total_relevance = logits[label]

        total_input_relevance_model = heatmap.sum()

        topk_subspaces = info.top_k_sources
        total_input_subspaces_relevance = info.input_top_k_source_heatmaps.reshape(
            (top_k, -1)
        ).sum(axis=1)

        total_input_residue_subspace_relevance = (
            info.input_subspace_residue_heatmap.sum()
        )

        total_a_x_relevance = info.rel_sum_total_layer
        total_a_xp_relevance = 0

        return (
            total_relevance,
            total_input_relevance_model,
            topk_subspaces,
            total_input_subspaces_relevance,
            total_input_residue_subspace_relevance,
            total_a_x_relevance,
            total_a_xp_relevance,
        )
