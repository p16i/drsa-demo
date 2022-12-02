import os

from nptyping import NDArray


import numpy as np

from cxai import netdissect, utils as putils

from .base import Inspector
from .bases.random import RandomBasisInspector
from .bases.group import (
    GroupBasisInspector,
    NetDissectInspector,
    NetDissectCompleteInspector,
)


class BasisProducer:
    def __call__(
        self,
        basis_name: str,
        arch: str,
        layer: str,
        layer_dimensions: int,
        num_subspaces: int,
        subspace_size: int,
        base_dir: str,
        **kwargs,
    ) -> Inspector:
        raise NotImplementedError("")


class RandomBasisProducer(BasisProducer):
    def __call__(
        self,
        basis_name: str,
        arch: str,
        layer: str,
        layer_dimensions: int,
        num_subspaces: int,
        subspace_size: int,
        base_dir: str,
        **kwargs,
    ):
        seed = int(basis_name[6])

        return RandomBasisInspector(
            num_subspaces=num_subspaces,
            subspace_size=subspace_size,
            layer=layer,
            dims=layer_dimensions,
            seed=seed,
        )


class MaxRelevanceFilterProducer(BasisProducer):
    def __call__(
        self,
        basis_name: str,
        arch: str,
        layer: str,
        layer_dimensions: int,
        num_subspaces: int,
        subspace_size: int,
        base_dir: str,
        **kwargs,
    ):

        assert num_subspaces == 1, "this basis only works with num-subspaces=1"

        act = np.load(
            os.path.join(
                putils.parent_dir(base_dir),
                "act.npy",
            )
        )
        ctx = np.load(os.path.join(base_dir, "ctx.npy"))

        rel = act * ctx

        score = np.mean(rel, axis=0)
        assert score.shape == (layer_dimensions,)
        sorted_indices = np.argsort(score)[::-1][:subspace_size]

        weights = np.zeros((layer_dimensions, subspace_size))

        weights[sorted_indices, list(range(subspace_size))] = 1

        assert np.argwhere(weights[:, 0]).reshape(-1) == sorted_indices[:1]

        return GroupBasisInspector(layer=layer, weights=weights[:, None, :].copy())


class EigenBasisProducer(BasisProducer):
    def _construct_matrix(self, base_dir: str) -> NDArray:
        raise NotImplementedError()

    def __call__(
        self,
        basis_name: str,
        arch: str,
        layer: str,
        layer_dimensions: int,
        num_subspaces: int,
        subspace_size: int,
        base_dir: str,
        **kwargs,
    ):

        nh = num_subspaces * subspace_size
        assert nh <= layer_dimensions

        print(f"Constructing eigvecs basis from {base_dir}")

        matrix = self._construct_matrix(base_dir)

        assert matrix.shape == (layer_dimensions, layer_dimensions)

        _, eigvecs = np.linalg.eigh(matrix)

        # by convention, numpy returns eigvectors in the ascending order of eigvalues
        # so, we sort them in descending order instead.
        eigvecs = eigvecs[:, ::-1].copy()
        selected_eigvecs = eigvecs[:, :nh]

        assert selected_eigvecs.shape[0] == layer_dimensions

        weights = np.reshape(
            selected_eigvecs, (layer_dimensions, num_subspaces, subspace_size)
        )

        # sanity checks
        for six in range(num_subspaces):
            jix = np.random.randint(low=0, high=subspace_size)
            np.testing.assert_allclose(
                weights[:, six, jix], eigvecs[:, subspace_size * six + jix]
            )

        return GroupBasisInspector(layer=layer, weights=weights)


class PCABasisProducer(EigenBasisProducer):
    def _construct_matrix(self, base_dir):
        print("Constructing PCA basis from `uncentered covariance` matrix")
        # This basis is constructed from eigenvectors of E[aa^t],
        # i.e., outer-product of activation vectors.
        # Therefore, this is not exactly PCA per se because of non-centering.
        act = np.load(os.path.join(putils.parent_dir(base_dir), "act.npy"))

        uncentered_cov = act.T @ act / act.shape[0]

        return uncentered_cov


class PRCABasisProducer(EigenBasisProducer):
    def _construct_matrix(self, base_dir):
        print("Constructing PRCA basis from `A.T @ C + C.T @ A` matrix")

        # This basis is formed from eigenvectors (u's) that maximize E[(u^T a)(u^T c)].
        act = np.load(os.path.join(putils.parent_dir(base_dir), "act.npy"))
        ctx = np.load(os.path.join(base_dir, "ctx.npy"))

        uncentered_cov = (act.T @ ctx + ctx.T @ act) / act.shape[0]

        return uncentered_cov


class NetDissectProducer(BasisProducer):
    def __call__(
        self,
        basis_name: str,
        arch: str,
        layer: str,
        layer_dimensions: int,
        num_subspaces: int,
        subspace_size: int,
        base_dir: str,
        **kwargs,
    ):
        print(f"Using NetDissect Basis for {arch}-{layer}")

        (
            concepts,
            mat_basis,
            filters_per_concept,
        ) = netdissect.get_netdissect_concepts_and_basis_matrix(arch, layer)

        print(f"> we have {len(concepts)} unique concepts")
        print(f"Remark: we don't any any thresholding as in the original work")

        kwargs = dict(
            layer=layer,
            weights=mat_basis,
            subspace_coefficients=1 / filters_per_concept,
        )

        if "complete" in basis_name:
            print(
                "[attention]: we use the complete version of NetDissect basis"
                + "where the last subspace is manualy set to the complement of the first top_k - 1 subspaces."
                + "Remark: we also manually set its index to be None."
            )
            return NetDissectCompleteInspector(**kwargs)
        else:
            return NetDissectInspector(**kwargs)


class LearntBasisProducer(BasisProducer):
    def __call__(
        self,
        basis_name: str,
        arch: str,
        layer: str,
        layer_dimensions: int,
        num_subspaces: int,
        subspace_size: int,
        base_dir: str,
        **kwargs,
    ):
        basis_name = basis_name.split("--")[-1]

        weights_path = os.path.join(
            base_dir,
            "bases",
            basis_name,
            "weights.npy",
        )

        weights = np.load(weights_path)
        print(f"Using learnt basis from {weights_path}")
        print("> shape", weights.shape)

        return GroupBasisInspector(layer=layer, weights=weights)
