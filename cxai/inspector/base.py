from typing import List, Tuple

import numpy as np
from scipy.stats import ortho_group

import torch
from torch.nn import functional as F

from cxai import utils as putils


class Inspector:
    subspace_coefficients = 1

    def __init__(
        self, layer: str, mean: np.array, mat_encoding: np.array, mat_decoding: np.array
    ):
        # we need this for setting up inspection context
        self.layer = layer

        assert len(mean.shape) == 1

        # todo: remove mean.
        self.mean = torch.as_tensor(mean.reshape((1, -1, 1, 1)), dtype=torch.float)

        # shape: [D, K]
        mat_act_encoding = torch.as_tensor(mat_encoding, dtype=torch.float)
        print("mat_act_encoding", mat_act_encoding.shape)

        # shape: [K, D]
        mat_act_decoding = torch.as_tensor(mat_decoding, dtype=torch.float)
        print("mat_act_decoding", mat_act_encoding.shape)

        # shape: [K, D]
        # remark: conv2d requires (out_channel (K), in_channel (D))
        # see https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
        # here, we go from D -> K
        self.mat_act_encoding = mat_act_encoding.T.unsqueeze(2).unsqueeze(3)

        # here, we go from K -> D
        self.mat_act_decoding = mat_act_decoding.unsqueeze(2).unsqueeze(3)

        # here, we got from D -> K
        self.mat_ctx_encoding = mat_act_decoding.T.unsqueeze(2).unsqueeze(3)
        print("self.mat_ctx_encoding.shape", self.mat_ctx_encoding.shape)

    def encode_activation(self, act: torch.Tensor) -> torch.Tensor:
        """Encode activation to K-subspace

        Args:
            act (Torch.Tensor): (b, d, h, w)

        Returns:
            src (Torch.Tensor): (b, K, h, w)
        """

        centered_act = act - self.mean
        return F.conv2d(centered_act, self.mat_act_encoding)

    def decode_activation(self, src: torch.Tensor) -> torch.Tensor:
        """Decode activation from K-subspace

        Args:
            src (torch.Tensor): (b, K, h, w)

        Returns:
            recon_act (torch.Tensor): (b, d, h, w)
        """
        recon_act = F.conv2d(src, self.mat_act_decoding)
        return recon_act + self.mean

    def encode_context(self, ctx: torch.Tensor) -> torch.Tensor:
        """Encode LRP context to K-subspace

        Remark: This encoding uses the transpose of the decoding matrix.

        Args:
            ctx (torch.Tensor): (b, d, h, w)

        Returns:
            encoded_ctx (torch.Tensor): (b, K, h, w)
        """
        return F.conv2d(ctx, self.mat_ctx_encoding)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.mat_act_encoding = self.mat_act_encoding.to(device)
        self.mat_act_decoding = self.mat_act_decoding.to(device)
        self.mat_ctx_encoding = self.mat_ctx_encoding.to(device)

        return self

    def rank_subspaces(self, rel_subspaces: torch.tensor, top_k: int) -> List[int]:
        assert len(rel_subspaces.shape) == 1

        # In some case, e.g., using NetDissect basis,
        # each concept has different numbers of members of filters.
        # Therefore, we take into account each concept's size.
        weighted_subspaces = rel_subspaces * self.subspace_coefficients

        top_k_subspaces = (
            torch.argsort(-weighted_subspaces)[:top_k].detach().cpu().numpy()
        )
        return top_k_subspaces.tolist()

    def compute_subspace_relevance(
        self, encoded_activation: torch.Tensor, encoded_context: torch.Tensor
    ) -> torch.Tensor:
        b, nh, h, w = encoded_activation.shape

        assert nh == self.num_subspaces * self.subspace_size

        rel_src = encoded_activation * encoded_context

        rel_subspace = rel_src.view(
            b, self.num_subspaces, self.subspace_size, h, w
        ).sum(dim=2)

        assert rel_subspace.shape == (b, self.num_subspaces, h, w)

        return rel_subspace

    def get_top_k_subspaces(
        self, activation: torch.Tensor, context: torch.Tensor, top_k: int
    ) -> Tuple[List[int], torch.Tensor]:
        """This function takes raw activation and context and return the top-k subspaces
        based on the ranking.

        Args:
            activation (torch.Tensor): raw activation vector; shape: [b, d, h, w]
            context (torch.Tensor): raw context vector; shape: [b, d, hw]
            top_k (int):

        Returns:
            Tuple[List[int], torch.Tensor]: top_k_subspaces, relevance_subspaces
        """

        # assert shape

        assert len(activation.shape) == len(context.shape)

        encoded_activation = self.encode_activation(activation)
        encoded_context = self.encode_context(context)

        relevance_subspaces = (
            self.compute_subspace_relevance(encoded_activation, encoded_context)
            .flatten(start_dim=2)
            .sum(dim=2)
        )

        top_k_subspaces = self.rank_subspaces(relevance_subspaces.squeeze(0), top_k)

        relevance_top_k_subspaces = relevance_subspaces[:, top_k_subspaces]

        assert relevance_top_k_subspaces.shape == (1, top_k)

        assert relevance_top_k_subspaces.shape == (1, top_k)
        assert len(top_k_subspaces) == top_k

        return (top_k_subspaces, relevance_top_k_subspaces)
