import numpy as np
from scipy.stats import ortho_group

from .. import Inspector


class RandomBasisInspector(Inspector):
    def __init__(
        self, num_subspaces: int, subspace_size: int, layer: str, dims: int, seed=1
    ):

        np.random.seed(seed=seed)
        m = ortho_group.rvs(dim=dims)

        # dims, number of groups, number of dimensions per group
        self.nd = dims

        # number of subspaces
        self.num_subspaces = num_subspaces

        # size of each subspace
        self.subspace_size = subspace_size

        nh = self.num_subspaces * self.subspace_size

        mat_encoding = m[:, :nh]

        # ok, we will do the transpose in the inspector operator
        mat_decoding = m[:, :nh]

        np.testing.assert_allclose(mat_encoding.T @ mat_decoding, np.eye(nh), atol=1e-6)

        super().__init__(
            layer=layer,
            mean=np.zeros(dims),
            mat_encoding=mat_encoding,
            mat_decoding=mat_decoding,
        )
