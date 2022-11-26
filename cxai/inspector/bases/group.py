import torch
import numpy as np
from nptyping import NDArray

from .. import Inspector


class GroupBasisInspector(Inspector):
    # todo: rename tihs to subspace
    def __init__(self, layer: str, weights: NDArray, subspace_coefficients=None):

        # dims, number of groups, number of dimensions per group
        self.nd, self.num_subspaces, self.subspace_size = weights.shape

        weights_flatten = weights.reshape(
            (self.nd, self.num_subspaces * self.subspace_size)
        )

        if subspace_coefficients is not None:
            assert len(subspace_coefficients.shape) == 1
            assert subspace_coefficients.shape == (self.num_subspaces,)
            self.subspace_coefficients = torch.from_numpy(subspace_coefficients)

        super().__init__(
            layer=layer,
            mean=np.zeros(self.nd),
            mat_encoding=weights_flatten,
            mat_decoding=weights_flatten,
        )

    def to(self, device):
        super().to(device)

        if isinstance(self.subspace_coefficients, torch.Tensor):
            self.subspace_coefficients = self.subspace_coefficients.to(device)

        return self


class NetDissectInspector(GroupBasisInspector):
    # this is just a dummary
    pass


class NetDissectCompleteInspector(GroupBasisInspector):
    pass

    def get_top_k_subspaces(
        self, activation: torch.Tensor, context: torch.Tensor, top_k: int
    ):

        top_k_subspaces, relevance_top_subspaces = super().get_top_k_subspaces(
            activation, context, top_k
        )

        selected_top_k = top_k_subspaces[:-1] + [None]

        # remark: the first dimension is batch dimension.
        selected_top_subspaces = relevance_top_subspaces[:, :-1]

        residue = (activation * context).flatten(start_dim=1).sum(
            dim=1, keepdim=True
        ) - selected_top_subspaces.sum(dim=1, keepdim=True)
        selected_top_subspaces = torch.hstack([selected_top_subspaces, residue])

        return (
            selected_top_k,
            selected_top_subspaces,
        )
