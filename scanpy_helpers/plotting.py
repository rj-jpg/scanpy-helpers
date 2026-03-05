# sctools/plotting.py

"""
Plotting utilities for single-cell QC, UMAP visualisation, and group comparisons.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from adjustText import adjust_text
from statannotations.Annotator import Annotator

__all__ = [
    "plot_with_repel",
    "boxplot_sample_means",
    "boxplot_sample_means_v2",
    "boxplot_cells_with_sample_means",
]


# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_FIGSIZE      = (14, 10)
_DEFAULT_LABEL_SIZE   = 6
_DEFAULT_DPI          = 300
_JITTER_WIDTH         = 0.2   # fraction of box width for strip jitter


# ── Private helpers ────────────────────────────────────────────────────────────

def _condition_order(series: pd.Series, order: list | None) -> list:
    """
    Return the display order for a condition column.

    Respects an existing categorical order if present; otherwise sorts
    alphabetically.  Filters *order* to only levels that actually exist.
    """
    if pd.api.types.is_categorical_dtype(series):
        levels = list(series.cat.categories)
    else:
        levels = sorted(series.unique())
    if order is None:
        return levels
    return [lvl for lvl in order if lvl in levels]


def _per_sample_means(
    df: pd.DataFrame,
    y: str,
    group: str,
    cond: str,
) -> pd.DataFrame:
    """
    Collapse cell-level data to one mean per (condition, sample) pair.

    Parameters
    ----------
    df    : cell-level DataFrame containing at least y, group, cond columns.
    y     : numeric column to summarise.
    group : column identifying biological replicates (e.g. "sample").
    cond  : column identifying experimental conditions.

    Returns
    -------
    DataFrame with columns [cond, group, "sample_mean"].
    """
    d = df[[y, group, cond]].dropna()
    if d.empty:
        raise ValueError(
            f"No rows remain after dropping NaNs from columns "
            f"'{y}', '{group}', '{cond}'."
        )
    return (
        d.groupby([cond, group], observed=True)[y]
         .mean()
         .reset_index(name="sample_mean")
    )


def _get_or_create_ax(
    ax: plt.Axes | None,
    figsize: tuple[float, float],
) -> tuple[plt.Figure, plt.Axes]:
    """
    Return (fig, ax).  Creates a new figure only when ax is not supplied.
    Allows callers to embed plots in an existing subplot grid.
    """
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _save_figure(fig: plt.Figure, save: str | Path | None, dpi: int) -> None:
    """Save *fig* to *save* at *dpi* resolution if a path was provided."""
    if save is not None:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")


def _umap_centroids(adata, color: str) -> dict[str, np.ndarray]:
    """
    Compute the 2-D centroid of each group in ``adata.obs[color]``
    using the UMAP embedding.
    """
    coords     = adata.obsm["X_umap"]
    labels     = adata.obs[color]
    centroids  = {}
    for group in labels.unique():
        mask = (labels == group).values
        centroids[group] = coords[mask].mean(axis=0)
    return centroids


def _add_repelled_labels(
    ax: plt.Axes,
    centroids: dict[str, np.ndarray],
    fontsize: int,
) -> None:
    """
    Place one text label per group at its centroid, then use adjustText to
    push overlapping labels apart with arrows.
    """
    texts = [
        ax.text(
            xy[0], xy[1], label,
            fontsize=fontsize,
            fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="none", alpha=0.7),
        )
        for label, xy in centroids.items()
    ]
    adjust_text(
        texts,
        ax=ax,
        only_move={"points": "y", "text": "y"},
        lim=15,
        arrowprops=None,
    )


def _annotate_pairs(
    ax: plt.Axes,
    per: pd.DataFrame,
    cond: str,
    order: list,
    test: str,
    text_format: str,
) -> None:
    """
    Add pairwise statistical annotations to an existing seaborn boxplot axis.
    Uses Benjamini-Hochberg FDR correction across all pairs.
    Skips annotation silently if fewer than two conditions are present.
    """
    if len(order) < 2:
        return
    pairs = list(combinations(order, 2))
    annotator = Annotator(
        ax, pairs, data=per, x=cond, y="sample_mean", order=order
    )
    annotator.configure(
        test=test,
        text_format=text_format,
        loc="inside",
        comparisons_correction="fdr_bh",
    )
    annotator.apply_and_annotate()


def _strip_jitter(
    ax: plt.Axes,
    data_by_group: list[np.ndarray],
    jitter_width: float = _JITTER_WIDTH,
) -> None:
    """
    Overlay jittered scatter points on a matplotlib (non-seaborn) boxplot.
    *data_by_group* is a list of 1-D arrays, one per box position (1-indexed).
    """
    for i, ys in enumerate(data_by_group, start=1):
        if ys.size == 0:
            continue
        xs = i + (np.random.default_rng().uniform(-0.5, 0.5, size=len(ys)) * jitter_width)
        ax.scatter(xs, ys, alpha=0.7, s=18)


# ── Public API ────────────────────────────────────────────────────────────────

def plot_with_repel(
    adata,
    color: str,
    label_fontsize: int = _DEFAULT_LABEL_SIZE,
    legend_fontsize: int | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    legend_loc: str = "right margin",
    save: str | Path | None = None,
    title: str | None = None,
    size: float | None = None,
    dpi: int = _DEFAULT_DPI,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    UMAP coloured by *color* with cluster labels placed directly on the plot.

    Labels are positioned at each cluster's centroid and then pushed apart
    with ``adjustText`` to minimise overlap.  A legend is also drawn on the
    right margin for reference.

    Parameters
    ----------
    adata : AnnData
        Must have ``adata.obsm["X_umap"]`` and ``adata.obs[color]``.
    color : str
        Column in ``adata.obs`` to colour by (e.g. ``"CellType"``).
    label_fontsize : int
        Font size for the on-plot cluster labels.
    legend_fontsize : int, optional
        Font size for the right-margin legend.  Defaults to scanpy's setting.
    figsize : tuple of float
        Figure size in inches ``(width, height)``.
    legend_loc : str
        Passed to ``sc.pl.umap``; use ``"right margin"`` or ``"on data"``.
    save : path-like, optional
        If provided, the figure is saved to this path.
    title : str, optional
        Plot title.
    size : float, optional
        Point size forwarded to ``sc.pl.umap``.
    dpi : int
        Resolution for saved figures.
    ax : Axes, optional
        Embed into an existing subplot.  A new figure is created when *None*.

    Returns
    -------
    matplotlib.axes.Axes
    """
    # Reset any cached per-category colours so scanpy uses its defaults
    color_key = f"{color}_colors"
    if color_key in adata.uns:
        del adata.uns[color_key]

    _, ax = _get_or_create_ax(ax, figsize)

    sc.pl.umap(
        adata,
        color=color,
        legend_loc=legend_loc,
        legend_fontsize=legend_fontsize,
        title=title,
        size=size,
        ax=ax,
        show=False,
    )

    centroids = _umap_centroids(adata, color)
    _add_repelled_labels(ax, centroids, fontsize=label_fontsize)

    _save_figure(ax.figure, save, dpi)
    return ax


def boxplot_sample_means(
    df: pd.DataFrame,
    y: str = "mu_freq_IGH",
    group: str = "sample",
    cond: str = "condition",
    order: list | None = None,
    ax: plt.Axes | None = None,
    jitter: bool = True,
    figsize: tuple[float, float] = (6, 4),
    yscale: str = "linear",
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Boxplot of per-sample means grouped by condition (matplotlib style).

    Collapses cell-level observations to one mean per biological replicate,
    then draws a plain matplotlib boxplot with optional jittered dots.
    Use :func:`boxplot_sample_means_v2` if you want seaborn styling and
    automatic statistical annotations.

    Parameters
    ----------
    df : DataFrame
        Cell-level data with at least columns *y*, *group*, *cond*.
    y : str
        Numeric column to plot.
    group : str
        Column identifying biological replicates (e.g. ``"sample"``).
    cond : str
        Column identifying experimental conditions.
    order : list, optional
        Display order of conditions.  Defaults to alphabetical / categorical.
    ax : Axes, optional
        Embed into an existing subplot.
    jitter : bool
        Overlay jittered per-sample dots on the boxes.
    figsize : tuple of float
        Figure size when creating a new figure.
    yscale : str
        Y-axis scale passed to ``ax.set_yscale`` (e.g. ``"log"``).
    title : str, optional
        Plot title.

    Returns
    -------
    fig : Figure
    ax : Axes
    per : DataFrame
        Per-sample mean values used to draw the plot (columns: cond, group,
        ``"sample_mean"``).
    """
    per   = _per_sample_means(df, y, group, cond)
    order = _condition_order(per[cond], order)

    box_data = [per.loc[per[cond] == lvl, "sample_mean"].values for lvl in order]

    fig, ax = _get_or_create_ax(ax, figsize)
    ax.boxplot(box_data, labels=order, showfliers=False)

    if jitter:
        _strip_jitter(ax, box_data)

    ax.set_xlabel(cond)
    ax.set_ylabel(f"Per-sample mean of {y}")
    ax.set_yscale(yscale)
    ax.margins(x=0.05)
    if title:
        ax.set_title(title)

    return fig, ax, per


def boxplot_sample_means_v2(
    df: pd.DataFrame,
    y: str = "mu_freq_IGH",
    group: str = "sample",
    cond: str = "condition",
    order: list | None = None,
    figsize: tuple[float, float] = (6, 4),
    yscale: str = "linear",
    title: str | None = None,
    test: str = "Mann-Whitney",
    text_format: str = "star",
    palette=None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Seaborn boxplot of per-sample means with automatic statistical annotations.

    Like :func:`boxplot_sample_means` but uses seaborn for styling and
    ``statannotations`` to overlay pairwise test results with FDR correction.

    Parameters
    ----------
    df : DataFrame
        Cell-level data with at least columns *y*, *group*, *cond*.
    y : str
        Numeric column to plot.
    group : str
        Column identifying biological replicates (e.g. ``"sample"``).
    cond : str
        Column identifying experimental conditions.
    order : list, optional
        Display order of conditions.
    figsize : tuple of float
        Figure size.
    yscale : str
        Y-axis scale.
    title : str, optional
        Plot title.
    test : str
        Statistical test passed to ``statannotations``
        (e.g. ``"Mann-Whitney"``, ``"t-test_ind"``).
    text_format : str
        Annotation format: ``"star"`` or ``"simple"``
        (p-value as ``*``/``**``/``***`` or numeric).
    palette : optional
        Seaborn palette.  Defaults to ``"Set2"``.

    Returns
    -------
    fig : Figure
    ax : Axes
    per : DataFrame
        Per-sample mean values used to draw the plot.
    """
    per   = _per_sample_means(df, y, group, cond)
    order = _condition_order(per[cond], order)

    if palette is None:
        palette = sns.color_palette("Set2", n_colors=len(order))

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=per, x=cond, y="sample_mean",
        order=order, palette=palette, ax=ax,
    )
    sns.stripplot(
        data=per, x=cond, y="sample_mean",
        order=order, palette=palette,
        alpha=0.7, size=5, edgecolor="black", linewidth=0.5, ax=ax,
    )

    _annotate_pairs(ax, per, cond, order, test=test, text_format=text_format)

    ax.set_ylabel(f"Per-sample mean of {y}")
    ax.set_yscale(yscale)
    if title:
        ax.set_title(title)

    return fig, ax, per


def boxplot_cells_with_sample_means(
    df: pd.DataFrame,
    y: str = "mu_freq_IGH",
    group: str = "sample",
    cond: str = "condition",
    order: list | None = None,
    ax: plt.Axes | None = None,
    jitter: bool = True,
    figsize: tuple[float, float] = (6, 4),
    yscale: str = "linear",
    title: str | None = None,
    jitter_width: float = _JITTER_WIDTH,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Boxplot of cell-level values with per-sample means overlaid as dots.

    The boxes summarise the full distribution of individual cells within each
    condition.  Jittered dots show the per-sample mean, making both the
    within-sample spread and the between-sample variance visible simultaneously.

    Parameters
    ----------
    df : DataFrame
        Cell-level data with at least columns *y*, *group*, *cond*.
    y : str
        Numeric column to plot.
    group : str
        Column identifying biological replicates.
    cond : str
        Column identifying experimental conditions.
    order : list, optional
        Display order of conditions.
    ax : Axes, optional
        Embed into an existing subplot.
    jitter : bool
        Overlay jittered per-sample mean dots.
    figsize : tuple of float
        Figure size when creating a new figure.
    yscale : str
        Y-axis scale.
    title : str, optional
        Plot title.
    jitter_width : float
        Horizontal spread of the jittered dots (fraction of box width).

    Returns
    -------
    fig : Figure
    ax : Axes
    per : DataFrame
        Per-sample mean values overlaid as dots.
    """
    d = df[[y, group, cond]].dropna()
    if d.empty:
        raise ValueError(
            f"No rows remain after dropping NaNs from '{y}', '{group}', '{cond}'."
        )

    order = _condition_order(d[cond], order)
    per   = _per_sample_means(d, y, group, cond)

    # Boxes show all cells; dots show per-sample means
    box_data = [d.loc[d[cond] == lvl, y].values for lvl in order]

    fig, ax = _get_or_create_ax(ax, figsize)
    ax.boxplot(box_data, labels=order, showfliers=False)

    if jitter:
        dot_data = [per.loc[per[cond] == lvl, "sample_mean"].values for lvl in order]
        _strip_jitter(ax, dot_data, jitter_width=jitter_width)

    ax.set_xlabel(cond)
    ax.set_ylabel(f"{y}  (boxes: all cells  •  dots: sample means)")
    ax.set_yscale(yscale)
    ax.margins(x=0.05)
    if title:
        ax.set_title(title)

    return fig, ax, per