from typing import Callable, Tuple, List
from nptyping import NDArray

import torch
import numpy as np

from tqdm import tqdm


from torch.nn import functional as F

from scipy.stats import ortho_group


class Model:
    def __init__(
        self,
        obj_func: Callable,
        nd: int,
        ns: int,
        ss: int,
        init_weight: torch.Tensor,
        device="cpu",
    ):

        self.obj_func = obj_func

        # original dimensions
        self.nd = nd

        self.ns = ns
        self.ss = ss
        # latent dimensions
        self.nh = ns * ss

        self.device = device

        self.U = torch.nn.Parameter(init_weight)

    def fit(self, A: torch.Tensor, C: torch.Tensor, epochs=100):
        arr_objval = []

        tbar = tqdm(range(epochs), mininterval=1)

        for epoch in tbar:
            self.U.requires_grad_(True)
            self.U.grad = None
            self.U.retain_grad()

            J = Model._compute_objval(A, C, self.U, self.obj_func, self.ns, self.ss)
            J.backward()

            # gradients update!
            self.U = decorrelate(self.U + self.U.grad)

            objval = J.detach().cpu().numpy()
            arr_objval.append(objval)

            if epoch % 100 == 0:
                tbar.set_description(f"Epoch {epoch:3d}: J(U)={objval:.4f}")

        # remark: this final loss is NOT equal to arr_loss[-1] because of
        # the last update!
        final_objval = Model._compute_objval(
            A, C, self.U, self.obj_func, self.ns, self.ss
        )
        final_objval = final_objval.detach().cpu().numpy()

        return float(final_objval), arr_objval

    @staticmethod
    def _compute_objval(
        A: torch.Tensor,
        C: torch.Tensor,
        U: torch.Tensor,
        loss_func: Callable,
        ns: int,
        ss: int,
    ) -> torch.Tensor:
        A_K = A.matmul(U).view(-1, ns, ss)
        C_K = C.matmul(U).view(-1, ns, ss)

        return loss_func(A_K, C_K)


def softmax_p(x: torch.Tensor, ord: int, dim=0) -> torch.Tensor:
    """Perform softmax order `ord` along `dim`

    Returns:
        torch.Tensor:
    """

    assert ord >= 1

    return (x**ord).mean(dim=dim) ** (1 / ord)


def softmin_p(x: torch.Tensor, ord: int, dim=0) -> torch.Tensor:
    """Perform softmin order `ord` along `dim`

    Returns:
        torch.Tensor:
    """

    assert ord >= 1

    return ((x ** (1 / ord)).mean(dim=dim)) ** (ord)


@torch.no_grad()
def decorrelate(A: torch.Tensor) -> torch.Tensor:
    # U_ = U @ (U.T @ U)^{-1/2}
    # ref: Hyvärinen et al. (2003), Independent Component Analysis, eq 6.37

    S = A.T @ A
    # somehow, for some classes (e.g. basketball at nfnet-f0's stage1),
    # the eigenvalue decomposition fails if using float.
    # Using double makes it more stable but slightly decreases the speed.
    # See: https://gist.github.com/p16i/4a37e10230c016fcde6c0e571c9ae010
    D, E = torch.linalg.eigh(S.double())
    D = D.float()
    E = E.float()

    inv = E @ torch.diag(1 / (torch.pow(D, 0.5))) @ E.T

    return A @ inv


def obj_dsa(
    activation: torch.Tensor, context: torch.Tensor, softmin_ord: int
) -> torch.Tensor:
    """_summary_

    remark: This loss is similar to Q. Le's paper.
    remark: Check this again.

    Args:
        activation (torch.Tensor): _description_
        context (torch.Tensor): we do NOT use context here.
            We keep it to make the interface compatible with the DRSA objective.

    Returns:
        torch.Tensor: _description_
    """

    assert len(activation.shape) == 3
    # shape: [n, ng]
    n, ns, _ = activation.shape

    # here, we use only the activation,
    magnitude = activation * activation

    component_magnitude = magnitude.sum(dim=2)
    assert component_magnitude.shape == (n, ns)

    # Step 1: Perform soft-max pooling over data (n)
    # e.g., smax_n
    component_magnitude = softmax_p(component_magnitude, ord=2, dim=0)

    assert component_magnitude.shape == (ns,)

    # Step 2: Perform soft-min pooling over K subspaces
    # e.g., smin_K
    obj = softmin_p(component_magnitude, ord=softmin_ord, dim=0)

    assert len(obj.shape) == 0

    return obj


def obj_drsa(
    activation: torch.Tensor, context: torch.Tensor, softmin_ord: int
) -> torch.Tensor:
    """_summary_

    This implements the DRSA objective.

    Args:
        activation (torch.Tensor): _description_
        context (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """

    assert (
        len(activation.shape) == len(context.shape) == 3
        and activation.shape == context.shape
    )

    relevance = activation * context

    # shape: [n, ng]
    n, ns, _ = relevance.shape

    component_relevance = relevance.sum(dim=2)

    component_relevance = F.relu(component_relevance)

    assert component_relevance.shape == (n, ns)

    # Step 1: Perform soft-max pooling over data (n)
    # e.g., smax_n
    component_relevance = softmax_p(component_relevance, ord=2, dim=0)

    assert component_relevance.shape == (ns,)

    # Step 2: Perform soft-min pooling over K subspaces
    # e.g., smin_K
    obj = softmin_p(component_relevance, ord=softmin_ord, dim=0)

    assert len(obj.shape) == 0

    return obj


def train_model_with_inner_model_selection(
    obj_func: Callable,
    act: torch.Tensor,
    ctx: torch.Tensor,
    seed: int,
    ns: int,
    ss: int,
    epochs: int,
    device="cpu",
    total_trials=3,
) -> Tuple[float, torch.Tensor, List[float]]:
    np.random.seed(seed)

    nd = act.shape[1]
    nh = ns * ss

    best_obj = -1
    best_weights = None
    best_obj_values = None

    Q = ortho_group.rvs(nd)

    for _ in tqdm(
        list(range(total_trials)), desc=f"Training with {total_trials} Trials"
    ):
        selected = np.random.permutation(nd)[:nh]

        U = torch.as_tensor(
            Q[:, selected],
            dtype=act.dtype,
            device=device,
        )

        model = Model(
            obj_func=obj_func,
            nd=nd,
            ns=ns,
            ss=ss,
            init_weight=U,
            device=device,
        )

        _curr_obj, _curr_obj_values = model.fit(act, ctx, epochs=epochs)

        if _curr_obj > best_obj:
            best_obj = _curr_obj
            best_obj_values = _curr_obj_values
            best_weights = model.U.detach()

    return best_obj, best_weights, best_obj_values
