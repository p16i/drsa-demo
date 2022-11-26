from typing import Tuple
from nptyping import NDArray
import torch
import torchvision
import numpy as np

from cxai.inspector import Inspector, InspectionRelevanceInfo
from cxai import models


class WithSplitModelMixin:
    """The class provides a cache mechanism to cache parts of models used
    in producing attribution for activation vectors.
    """

    CACHE_STORE_ATTR = "split_model_cache_store"

    def _get_split_model(self, layer: str) -> Tuple[torch.nn.Module, torch.nn.Module]:
        # todo: we might need to find a way to purge this
        # otherwise, this could lead to memory leak
        # remark: if we generalize to other architectures, e.g., nfnet, then,
        # we have to make sure that the cache key is parameterized by that.
        # actually, we dont' need to parameterized by that if we assume that
        # each explainer has one model.
        cache_key = layer

        if not hasattr(self, WithSplitModelMixin.CACHE_STORE_ATTR):
            setattr(self, WithSplitModelMixin.CACHE_STORE_ATTR, dict())

        cache_store = getattr(self, WithSplitModelMixin.CACHE_STORE_ATTR)

        try:
            obj = cache_store[cache_key]
            return obj
        except KeyError:
            cache_store[cache_key] = (
                feature_extractor,
                classification_head,
            ) = models.split_model_at_layer(self.model, layer)

            return cache_store[cache_key]


class Explainer(object):
    def explain(self, x: torch.tensor, label: int) -> Tuple[NDArray, NDArray]:
        """_summary_

        Args:
            x (torch.tensor): _description_
            label (int): _description_

        Returns:
            Tuple[NDArray, NDArray]: logits, heatmap
        """
        raise NotImplementedError()

    def explain_with_inspector(
        self, x: torch.tensor, target_label: int, inspector: Inspector, top_k: int
    ) -> Tuple[NDArray, NDArray, InspectionRelevanceInfo]:
        """_summary_

        Args:
            x (torch.tensor): _description_
            target_label (int): _description_
            inspector (Inspector): _description_
            top_k (int): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Tuple[NDArray, NDArray, InspectionRelevanceInfo]: logits, original heatmaps, inspection information
        """
        raise NotImplementedError()

    def get_intermediate_activation_and_context(
        self,
        layer: str,
        x: torch.tensor,
        label: int,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """_summary_

        Args:
            layer (str): _description_
            x (torch.tensor): _description_
            label (int): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Tuple[torch.tensor, torch.tensor]: activation and context
        """
        raise NotImplementedError()


class WithImplicitTotalInputRelevanceCalculation:
    def total_input_relevance(
        self, x: torch.Tensor, label: int, insp: Inspector, top_k: int
    ):

        feature_extractor, classification_head = self._get_split_model(insp.layer)

        xp = torch.ones_like(x).unsqueeze(0).to(x.device) * self.baseline

        f_x = classification_head(feature_extractor(x.unsqueeze(0)))[:, label]
        f_xp = classification_head(feature_extractor(xp))[:, label]

        act, ctx = self.get_intermediate_activation_and_context(insp.layer, x, label)
        act_xp = feature_extractor(xp)

        ap = torch.ones_like(act).to(x.device) * self.baseline

        y_ap = classification_head(ap)[:, label]

        relevance_model_xp = (act_xp * ctx).sum()

        total_relevance = (f_x - f_xp).squeeze()
        total_input_relevance_model = (f_x - y_ap - relevance_model_xp).squeeze(0)

        topk_subspaces, relevance_a_x_subspaces = insp.get_top_k_subspaces(
            act, ctx, top_k
        )

        relevance_a_xp_subspaces = (
            insp.compute_subspace_relevance(
                insp.encode_activation(act_xp), insp.encode_context(ctx)
            )
            .flatten(start_dim=2)
            .sum(dim=2)
        )
        relevance_a_xp_subspaces = relevance_a_xp_subspaces[:, topk_subspaces]

        total_input_subspaces_relevance = (
            relevance_a_x_subspaces - relevance_a_xp_subspaces
        )

        assert total_input_subspaces_relevance.shape == (1, top_k)
        total_input_subspaces_relevance = total_input_subspaces_relevance.squeeze(0)

        act_x_residue = act - insp.decode_activation(insp.encode_activation(act))
        act_xp_residue = act_xp - insp.decode_activation(insp.encode_activation(act_xp))

        relevance_a_x_residue_subspace = (act_x_residue * ctx).sum()

        relevance_a_xp_residue_subspace = (act_xp_residue * ctx).sum()

        total_input_residue_subspace_relevance = (
            relevance_a_x_residue_subspace - relevance_a_xp_residue_subspace
        )

        total_a_x_relevance = (
            relevance_a_x_subspaces.sum() + relevance_a_x_residue_subspace
        )
        total_a_xp_relevance = (
            relevance_a_xp_subspaces.sum() + relevance_a_xp_residue_subspace
        )

        return (
            total_relevance.detach().cpu().numpy(),
            total_input_relevance_model.detach().cpu().numpy(),
            topk_subspaces,
            total_input_subspaces_relevance.detach().cpu().numpy(),
            total_input_residue_subspace_relevance.detach().cpu().numpy(),
            total_a_x_relevance.detach().cpu().numpy(),
            total_a_xp_relevance.detach().cpu().numpy(),
        )
