# sctools/utils.py

"""
Miscellaneous utilities: AnnData subsetting, cell counting, and aggfunc helpers.

These are small, self-contained helpers that don't belong to any single
analysis module but are called across several of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "subset_and_filter",
    "count_cells_per_sample",
    "agg_name",
]


# ── Private helpers ────────────────────────────────────────────────────────────

def _validate_obs_columns(adata, *columns: str) -> None:
    """Raise a clear error if any of *columns* are absent from adata.obs."""
    missing = [c for c in columns if c not in adata.obs.columns]
    if missing:
        raise ValueError(
            f"adata.obs is missing column(s): {missing}. "
            f"Available columns: {list(adata.obs.columns)}."
        )


def _validate_min_cells(min_cells: int) -> None:
    if min_cells < 0:
        raise ValueError(f"min_cells must be >= 0; got {min_cells}.")


def _filter_by_min_cells(
    adata,
    sample_col: str,
    min_cells: int,
) -> object:
    """
    Remove samples (barcodes) from *adata* that have fewer than *min_cells*
    cells after all other subsetting has been applied.

    Parameters
    ----------
    adata      : AnnData already subsetted to the cell/isotype of interest.
    sample_col : Column in adata.obs identifying biological replicates.
    min_cells  : Minimum cell count per sample; samples below this are dropped.

    Returns
    -------
    AnnData
        A copy with low-count samples removed.
    """
    counts      = adata.obs[sample_col].value_counts()
    keep        = counts[counts >= min_cells].index
    n_dropped   = adata.obs[sample_col].nunique() - len(keep)

    if n_dropped > 0:
        print(
            f"Dropping {n_dropped} sample(s) with fewer than "
            f"{min_cells} cells after subsetting."
        )

    return adata[adata.obs[sample_col].isin(keep)].copy()


# ── Public API ────────────────────────────────────────────────────────────────

def subset_and_filter(
    adata,
    celltype: str,
    celltype_col: str = "CellType",
    isotype_col: str = "c_call_VDJ_main",
    isotype: str = "IGHA",
    sample_col: str = "sample",
    min_cells: int = 20,
) -> object:
    """
    Subset an AnnData to one cell type and isotype, then drop sparse samples.

    A common pattern before repertoire-level analysis: isolate the population
    of interest and remove any sample that has too few cells to yield a
    reliable per-sample summary statistic.

    Parameters
    ----------
    adata : AnnData
        Full annotated data object.
    celltype : str
        Value in *celltype_col* to retain (e.g. ``"Plasmablast"``).
    celltype_col : str
        Column in ``adata.obs`` containing cell-type labels.
    isotype_col : str
        Column in ``adata.obs`` containing isotype calls.
    isotype : str
        Isotype to retain (e.g. ``"IGHA"``, ``"IGHG"``).
    sample_col : str
        Column in ``adata.obs`` identifying biological replicates.
    min_cells : int
        Samples with fewer than *min_cells* cells after subsetting are removed.

    Returns
    -------
    AnnData
        A copy containing only the requested cell type / isotype, with
        low-count samples removed.

    Raises
    ------
    ValueError
        If any of *celltype_col*, *isotype_col*, or *sample_col* are absent
        from ``adata.obs``.

    Examples
    --------
    >>> ad = subset_and_filter(
    ...     adata,
    ...     celltype="Plasmablast",
    ...     isotype="IGHA",
    ...     min_cells=20,
    ... )
    >>> ad.n_obs
    1234
    """
    _validate_obs_columns(adata, celltype_col, isotype_col, sample_col)
    _validate_min_cells(min_cells)

    n_start = adata.n_obs

    ad = adata[adata.obs[celltype_col] == celltype].copy()
    print(f"Retained {ad.n_obs:,} / {n_start:,} cells for celltype='{celltype}'.")

    n_after_celltype = ad.n_obs
    ad = ad[ad.obs[isotype_col] == isotype].copy()
    print(
        f"Retained {ad.n_obs:,} / {n_after_celltype:,} cells "
        f"for isotype='{isotype}'."
    )

    if ad.n_obs == 0:
        print("Warning: no cells remain after subsetting. Returning empty AnnData.")
        return ad

    ad = _filter_by_min_cells(ad, sample_col=sample_col, min_cells=min_cells)
    print(
        f"Final object: {ad.n_obs:,} cells across "
        f"{ad.obs[sample_col].nunique()} sample(s)."
    )
    return ad


def count_cells_per_sample(
    df: pd.DataFrame,
    y: str = "mu_freq_IGH",
    group: str = "sample",
) -> pd.Series:
    """
    Count non-missing observations of *y* per sample.

    Useful as a quick sanity check before running per-sample aggregations,
    or for constructing WLS weights in :func:`~sctools.stats.anova_pairwise`.

    Parameters
    ----------
    df : DataFrame
        Cell-level data.
    y : str
        Column whose non-missing values are counted.
    group : str
        Column identifying biological replicates.

    Returns
    -------
    pd.Series
        Index = sample labels, values = cell counts, name = ``"n_cells"``.

    Examples
    --------
    >>> counts = count_cells_per_sample(df, y="mu_freq_IGH")
    >>> counts[counts < 20]   # samples too sparse for reliable estimates
    """
    if group not in df.columns:
        raise ValueError(f"Column '{group}' not found in DataFrame.")
    if y not in df.columns:
        raise ValueError(f"Column '{y}' not found in DataFrame.")

    return (
        df[[y, group]]
        .dropna(subset=[y])
        .groupby(group, observed=True)
        .size()
        .rename("n_cells")
    )


def agg_name(aggfunc) -> str:
    """
    Return a readable string label for an aggregation function.

    Handles string names, named callables, and anonymous lambdas.
    Used internally for labelling output columns when the aggregation
    function is passed as an argument.

    Parameters
    ----------
    aggfunc : str or callable
        An aggregation function or its name, e.g. ``"median"``, ``np.mean``.

    Returns
    -------
    str

    Examples
    --------
    >>> agg_name("median")
    'median'
    >>> agg_name(np.mean)
    'mean'
    >>> agg_name(lambda x: x.sum())
    'agg'
    """
    if isinstance(aggfunc, str):
        return aggfunc
    if callable(aggfunc):
        return getattr(aggfunc, "__name__", "agg")
    return str(aggfunc)