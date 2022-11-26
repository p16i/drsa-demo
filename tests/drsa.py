import torch
import numpy as np

import pytest
from functools import partial

from cxai import drsa


def is_smaller(a, b):
    return a < b


def is_larger(a, b):
    return a > b


@pytest.mark.parametrize(
    "n,d",
    [
        (4, 3),
        (10, 8),
    ],
)
@pytest.mark.parametrize(
    "actual_func,cond,bound_func",
    [
        (
            drsa.softmax_p,
            is_smaller,
            np.max,
        ),
        (
            drsa.softmin_p,
            is_larger,
            np.min,
        ),
    ],
)
def test_softmax_p(n, d, actual_func, cond, bound_func):
    data = np.random.rand(n, d)
    tsdata = torch.from_numpy(data)

    arr_actuals = []
    for ord in range(1, 7):
        arr_actuals.append(actual_func(tsdata, ord=ord, dim=0).numpy())

    bound = bound_func(data, axis=0, keepdims=True)

    arr_actuals = np.array(arr_actuals)

    np.testing.assert_allclose(arr_actuals[0, :], np.mean(data, axis=0))

    np.testing.assert_equal(cond(arr_actuals, bound), True)
    np.testing.assert_equal(cond(arr_actuals[:-1], arr_actuals[1:]), True)


def test_loss():
    # This test just makes sure that these losses are callable.
    act = torch.randn(5, 3, 2)
    ctx = torch.randn(5, 3, 2)

    for p in [1, 2]:
        _ = drsa.obj_drsa(act, ctx, softmin_ord=p)
        _ = drsa.obj_dsa(act, act, softmin_ord=p)

    assert True


@pytest.mark.parametrize("sm_ord", [1, 2])
@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("mode", ["ica", "irca"])
def test_train_with_model_selection(sm_ord, seed, mode):
    ns, ss = 4, 2
    d = 8
    n = 10

    act = torch.rand(n, d).float()
    ctx = torch.rand(n, d).float()

    obj_func = drsa.obj_drsa if mode == "irca" else drsa.obj_dsa
    obj_func = partial(obj_func, softmin_ord=sm_ord)

    kwargs = dict(
        obj_func=obj_func, act=act, ctx=ctx, ns=ns, ss=ss, seed=seed, epochs=1
    )

    best_from_1, _, _ = drsa.train_model_with_inner_model_selection(
        **kwargs,
        total_trials=1,
    )

    best_from_5, _, _ = drsa.train_model_with_inner_model_selection(
        **kwargs,
        total_trials=5,
    )

    assert (
        best_from_5 >= best_from_1
    ), "Selecting from 5 random initialization trials should always better than just using one trial."
