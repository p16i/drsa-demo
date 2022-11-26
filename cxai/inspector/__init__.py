from typing import List, Tuple
from nptyping import NDArray

import re
from dataclasses import dataclass

import numpy as np
import torch


from cxai import netdissect, constants

from .base import *

from .bases.random import RandomBasisInspector
from .bases.group import GroupBasisInspector

from . import construction


def _parse_subspace_parameters(txt: str) -> Tuple[int, int]:
    """This function is a utility function for parsing
    ns5-ss10 into (5, 10)

    Args:
        text (str): text to be parsed
    Returns:
        tuple(int, int): these values correspond to
            - number of subspaces (ns)
            - size of each subspace (ss)
    """
    result = re.search("ns(\d+)-ss(\d+)", txt)

    if result is not None:
        return int(result.group(1)), int(result.group(2))
    else:
        return None, None


@dataclass
class InspectionRelevanceInfo:
    # shape=(K,)
    rel_per_source: List[float]

    rel_sum_total_layer: float
    # shape=(top_k, )
    top_k_sources: List[int]

    # Heatmaps of each top_k subspace (A_k)
    # shape=(K, *input_size)
    input_top_k_source_heatmaps: NDArray[float]

    # Heatmap of the residue subspace (I - A)
    # shape=(K, *input_size)
    input_subspace_residue_heatmap: NDArray[float]

    # Heatmap of all top_k subpsaces (A = \sum_k A_k)
    # shape=(K, *input_size)
    input_all_subspaces_heatmap: NDArray[float]


def get_inspector_for_basis(
    arch: str, layer: str, basis_name: str, **kwargs
) -> Inspector:

    dims = constants.get_arch_layer_dimensions(arch, layer)

    num_subspaces, subspace_size = _parse_subspace_parameters(basis_name)

    constructor: construction.BasisProducer = None

    if basis_name[:6] == "random":
        constructor = construction.RandomBasisProducer
    elif "max-act" in basis_name:
        constructor = construction.MaxActivationFilterProducer
    elif "max-rel" in basis_name:
        constructor = construction.MaxRelevanceFilterProducer
    elif basis_name[:3] == "pca":
        constructor = construction.PCABasisProducer
    elif basis_name[:4] == "prca":
        constructor = construction.PRCABasisProducer
    elif basis_name == "netdissect" or basis_name == "netdissect-complete":
        constructor = construction.NetDissectProducer
    elif "learnt" in basis_name:
        constructor = construction.LearntBasisProducer
    else:
        raise ValueError(f"No inspector for basis `{basis_name}`!")

    base_dir = kwargs["base_dir"]
    del kwargs["base_dir"]

    return constructor()(
        basis_name=basis_name,
        arch=arch,
        layer=layer,
        layer_dimensions=dims,
        num_subspaces=num_subspaces,
        subspace_size=subspace_size,
        base_dir=base_dir,
        **kwargs,
    )


class SubspaceModule(torch.nn.Module):
    """This module is a helper for attribution methods such as Int. Grad and
    Shapley Vaule Sampling. In these cases, we have to first approximate the context
    vector and then perform another rounds of attribution using this context vector.

    This module therefore helps build the attribution between the input and latent features.

    Args:
        torch (_type_): _description_
    """

    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        inspector: Inspector,
        context: torch.Tensor,
        selected_subspaces: List[int],
    ):
        super().__init__()
        self.inspector = inspector
        self.context = context
        self.feature_extractor = feature_extractor

        self.containing_complement = None in selected_subspaces

        # The selected_subspaces and valid_selected_subspaces can be different
        # in the case of netdissect-complete where the last subspace is manually
        # to be residue.
        self.selected_subspaces = selected_subspaces

        self.valid_selected_subspaces = list(
            filter(lambda i: i is not None, selected_subspaces)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        activation = self.feature_extractor(x)
        # remark: this is because we use parallel_run_per_eval = constants.SHAPLEY_VALUE_SAMPLING_BATCH
        batch_size = activation.shape[0]

        subspace_act = self.inspector.encode_activation(activation)
        subspace_ctx = self.inspector.encode_context(self.context)

        # shape: [b, num_subspaces, h, w]
        # remark: num_subspaces can be larger than `top_k`
        # this is for example the case of NetDissect.
        rel_subspaces = self.inspector.compute_subspace_relevance(
            subspace_act, subspace_ctx
        )

        np.testing.assert_allclose(
            rel_subspaces.shape,
            (
                batch_size,
                self.inspector.num_subspaces,
                *activation.shape[2:],
            ),
        )

        # shape: [b, 1]
        total_relevance = (
            (activation * self.context).flatten(start_dim=1).sum(dim=1, keepdim=True)
        )

        # shape: [b, K]
        output = (
            rel_subspaces[:, self.valid_selected_subspaces]
            .flatten(start_dim=2)
            .sum(dim=2)
        )

        if self.containing_complement:
            rel_complement = total_relevance - output.sum(dim=1, keepdim=True)
            output = torch.hstack([output, rel_complement])

        # shape: [b, 1]
        # todo: here, we must sum only the ones that are in top_k
        residue = total_relevance - output.sum(dim=1, keepdim=True)
        assert residue.shape == (batch_size, 1)

        if self.containing_complement:
            assert torch.allclose(residue, torch.zeros_like(residue), atol=1e-4)

        output = torch.hstack((output, residue))

        assert output.shape == (batch_size, len(self.selected_subspaces) + 1)

        return output
