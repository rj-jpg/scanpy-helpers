# sctools/qc.py

"""
QC metric calculation, cell filtering, and doublet-rate estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LinearRegression

__all__ = ["calc_qc_metrics", "flag_cells", "expected_doublet_rate"]


# ── Private helpers ────────────────────────────────────────────────────────────

# 10X Genomics published multiplet rate table (cells recovered → expected rate)
_MULTIPLET_RATE_TABLE = pd.DataFrame(
    {
        "cells_recovered": [500, 1000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
        "rate":            [0.004, 0.008, 0.031, 0.039, 0.046, 0.054, 0.061, 0.069, 0.076],
    }
)


def _fit_doublet_rate_model() -> LinearRegression:
    """
    Fit a linear model mapping cells recovered → expected doublet rate,
    using the 10X Genomics published multiplet rate table.

    Fitted once at import time and reused for all calls to
    `expected_doublet_rate`.
    """
    X = _MULTIPLET_RATE_TABLE[["cells_recovered"]].values
    y = _MULTIPLET_RATE_TABLE["rate"].values
    return LinearRegression().fit(X, y)


# Fit once when the module is imported — no repeated work per sample
_DOUBLET_RATE_MODEL = _fit_doublet_rate_model()


def _validate_flag_inputs(adata, pct_mito_cutoff: float, min_counts_cutoff: int) -> None:
    """
    Raise informative errors if required columns are missing or cutoffs are
    clearly wrong, before any mutations are made to adata.
    """
    required = {"pct_counts_mt", "total_counts"}
    missing = required - set(adata.obs.columns)
    if missing:
        raise ValueError(
            f"adata.obs is missing columns: {missing}. "
            "Run calc_qc_metrics() before flag_cells()."
        )
    if not (0 <= pct_mito_cutoff <= 100):
        raise ValueError(f"pct_mito_cutoff must be in [0, 100]; got {pct_mito_cutoff}.")
    if min_counts_cutoff < 0:
        raise ValueError(f"min_counts_cutoff must be >= 0; got {min_counts_cutoff}.")


def _build_flag_mask(adata, pct_mito_cutoff: float, min_counts_cutoff: int) -> pd.Series:
    """
    Return a boolean Series that is True for any cell that fails at least
    one QC criterion:
      - predicted doublet (Scrublet)
      - mitochondrial fraction above cutoff
      - total counts below cutoff
    """
    is_doublet = (
        adata.obs["predicted_doublet"].fillna(False).astype(bool)
        if "predicted_doublet" in adata.obs
        else pd.Series(False, index=adata.obs_names)
    )
    high_mito   = adata.obs["pct_counts_mt"] > pct_mito_cutoff
    low_counts  = adata.obs["total_counts"]  < min_counts_cutoff

    return is_doublet | high_mito | low_counts


def _summarise_flags(adata, column: str) -> None:
    """Print a short breakdown of flagged vs passing cells."""
    counts = adata.obs[column].value_counts()
    n_flagged = counts.get(True, 0)
    n_total   = adata.n_obs
    pct       = 100 * n_flagged / n_total if n_total > 0 else 0.0
    print(
        f"Flagged {n_flagged:,} / {n_total:,} cells "
        f"({pct:.1f}%) → stored in obs['{column}']."
    )


# ── Public API ────────────────────────────────────────────────────────────────

def calc_qc_metrics(adata):
    """
    Annotate gene-level boolean flags and compute standard per-cell QC metrics.

    Flags three gene sets in ``adata.var``:
      - ``mt``   — mitochondrial genes (prefix ``MT-``)
      - ``ribo`` — ribosomal genes (prefix ``RPS`` or ``RPL``)
      - ``hb``   — haemoglobin genes (pattern ``HB[^P]``)

    Then calls :func:`scanpy.pp.calculate_qc_metrics` to add per-cell columns
    such as ``total_counts``, ``n_genes_by_counts``, and ``pct_counts_mt``.

    Parameters
    ----------
    adata : AnnData
        Object with raw counts in ``adata.X``.

    Returns
    -------
    AnnData
        The same object, modified in place and returned for chaining.

    Notes
    -----
    Gene-set patterns are specific to human gene symbols.  Mouse symbols use
    a lower-case prefix (``mt:``) and would need a separate implementation.
    """
    adata.var["mt"]   = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"]   = adata.var_names.str.contains(r"^HB[^P]", regex=True)

    print("Calculating QC metrics...")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        log1p=True,
    )
    return adata


def flag_cells(
    adata,
    pct_mito_cutoff: float = 5.0,
    min_counts_cutoff: int = 1000,
    column_suffix: str = "",
) -> object:
    """
    Flag low-quality cells by adding a boolean column to ``adata.obs``.

    A cell is flagged (``True``) if **any** of the following apply:

    - Predicted doublet by Scrublet (``predicted_doublet == True``)
    - Mitochondrial read fraction exceeds *pct_mito_cutoff*
    - Total UMI count is below *min_counts_cutoff*

    Parameters
    ----------
    adata : AnnData
        Object that has already been processed by :func:`calc_qc_metrics`.
    pct_mito_cutoff : float
        Upper bound on mitochondrial read percentage (default 5).
    min_counts_cutoff : int
        Lower bound on total UMI counts (default 1 000).
    column_suffix : str
        Appended to the output column name, e.g. ``"_round2"`` → ``"flagged_round2"``.
        Useful when re-running with different thresholds.

    Returns
    -------
    AnnData
        The same object, modified in place and returned for chaining.
    """
    _validate_flag_inputs(adata, pct_mito_cutoff, min_counts_cutoff)

    column = f"flagged{column_suffix}"
    flag_mask = _build_flag_mask(adata, pct_mito_cutoff, min_counts_cutoff)
    adata.obs[column] = flag_mask.astype(bool)

    _summarise_flags(adata, column)
    return adata


def expected_doublet_rate(n_cells: int) -> float:
    """
    Predict the expected Scrublet doublet rate for a given cell recovery count.

    Uses a linear model fitted to the 10X Genomics published multiplet rate
    table.  Predictions are clipped to ``[0, 1]``.

    Parameters
    ----------
    n_cells : int
        Number of cells recovered in the library.

    Returns
    -------
    float
        Predicted doublet rate, e.g. ``0.046`` for ~6 000 cells.

    Examples
    --------
    >>> expected_doublet_rate(6000)
    0.046...
    """
    if n_cells <= 0:
        raise ValueError(f"n_cells must be a positive integer; got {n_cells}.")

    rate = _DOUBLET_RATE_MODEL.predict(np.array([[n_cells]]))[0]
    return float(np.clip(rate, 0.0, 1.0))