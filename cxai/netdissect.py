import os

from pathlib import Path

from typing import Union, Tuple

import pandas as pd
import numpy as np

from cxai import constants

NETDISSECT_CONFIG_DIR = constants.PACKAGE_DIR / "config" / "netdissect"


def _build_filter_with_category(df: pd.DataFrame) -> pd.DataFrame:
    """Here, for each filter, we create a slug based on the assigned category
    and its sublabel provided NetDissect

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """

    rows = df.to_dict("records")

    arr_rows = []
    for row in rows:
        # they start index from 1.
        unit = row["unit"] - 1

        category = row["category"]
        label = row[f"{category}-label"]

        slug = f"{category}:{label}"

        arr_rows.append(dict(ix=unit, slug=slug))

    _df = pd.DataFrame(arr_rows)

    return _df


def _build_basis_matrix(df_group: pd.DataFrame) -> Tuple[Union[list, np.array]]:

    concepts = sorted(df_group.slug.unique().tolist())
    # number of channels
    dims = df_group.shape[0]

    # shape: (dims, num groups, group size)
    # here, we just set group size = dims for simplicity
    W = np.zeros((dims, len(concepts), dims))

    for cix, concept in enumerate(concepts):
        filter_indices = df_group[df_group.slug == concept].ix.values.reshape(-1)
        W[filter_indices, cix, filter_indices] = 1

    filters_per_concept = (W == 1).sum(axis=2).sum(axis=0)

    assert len(filters_per_concept) == len(concepts)

    return concepts, W, filters_per_concept


def get_netdissect_concepts_and_basis_matrix(
    arch: str, layer: str, verbose=False
) -> Tuple[Union[list, np.array]]:

    df_tally = pd.read_csv(
        NETDISSECT_CONFIG_DIR / "results" / arch / f"tally-{layer}.csv"
    )

    if verbose:
        print(f"NetDissect of `{arch}--{layer}`: num filters={df_tally.shape[0]}")

    df_filter_assigned_to_cat_group = _build_filter_with_category(df_tally)

    concepts, W, filters_per_concept = _build_basis_matrix(
        df_filter_assigned_to_cat_group
    )

    assert len(concepts) == W.shape[1]
    assert (
        W.sum(axis=2).sum(axis=1) == 1
    ).all(), "each filter should be assigned to only one concept."

    if verbose:
        print(f"We have unique {len(concepts)} concepts for {W.shape[0]} filters.")

    return concepts, W, filters_per_concept
