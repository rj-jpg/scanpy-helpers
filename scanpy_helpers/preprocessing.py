# sctools/preprocessing.py

"""
Normalisation, dimensionality reduction, clustering, and gene-set filtering.
"""

from __future__ import annotations

import scanpy as sc

__all__ = ["normalize_and_pca", "remove_ig_hvg"]


# ── Constants ─────────────────────────────────────────────────────────────────

# Gene prefixes treated as immunoglobulin genes.
# Extend this tuple if you work with non-human species or want to include TCR genes.
_IG_PREFIXES = ("IGH", "IGK", "IGL")

# Default Leiden resolutions to compute.  Fine-grained (0.02) → coarse (2.0).
_DEFAULT_RESOLUTIONS = [0.02, 0.2, 0.5, 1.0, 2.0]

# Resolutions pulled out for the summary UMAP plot (a subset of the above).
_UMAP_PLOT_RESOLUTIONS = [0.02, 0.5, 2.0]


# ── Private helpers ────────────────────────────────────────────────────────────

def _save_path(qc_output_dir, sample: str | None) -> tuple[bool, str | None]:
    """
    Return (should_save, figdir_str).
    Figures are only saved when both qc_output_dir and sample are provided.
    """
    if qc_output_dir is not None and sample is not None:
        return True, str(qc_output_dir)
    return False, None


def _set_figdir(figdir: str | None) -> None:
    """Point scanpy's figure output to figdir, if provided."""
    if figdir is not None:
        sc.settings.figdir = figdir


def _run_normalisation(adata) -> object:
    """
    Library-size normalise then log1p-transform counts in place.

    Stores the log-normalised matrix in ``adata.X``.  Raw counts should
    already be in ``adata.layers["counts"]`` before this step.
    """
    print("Normalising...")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


def _select_hvgs(
    adata,
    n_top_genes: int,
    remove_ig: bool,
    figdir: str | None,
    sample: str | None,
) -> object:
    """
    Identify highly variable genes, optionally remove immunoglobulin genes
    from the HVG set, and save a dispersion plot.
    """
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

    if remove_ig:
        adata = remove_ig_hvg(adata)

    _set_figdir(figdir)
    sc.pl.highly_variable_genes(
        adata,
        save=f"_{sample}_hvg" if sample else None,
        show=False,
    )
    return adata


def _run_pca(
    adata,
    figdir: str | None,
    sample: str | None,
) -> object:
    """
    Run PCA and save a variance-ratio elbow plot.
    """
    print("Running PCA...")
    sc.tl.pca(adata)

    _set_figdir(figdir)
    sc.pl.pca_variance_ratio(
        adata,
        n_pcs=50,
        log=True,
        save=f"_{sample}_pca_variance" if sample else None,
        show=False,
    )
    return adata


def _run_neighbors_and_umap(adata) -> object:
    """
    Build a kNN graph and embed in two dimensions with UMAP.
    """
    print("Running UMAP...")
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    return adata


def _run_leiden(adata, resolutions: list[float]) -> object:
    """
    Run Leiden clustering at each resolution in *resolutions*.

    Results are stored in ``adata.obs[f"leiden_res_{r:.2f}"]`` for each *r*.
    An initial run at the default resolution is also stored under
    ``adata.obs["leiden"]`` for convenience.
    """
    print("Finding Leiden clusters...")

    # Baseline call — populates adata.obs["leiden"] for quick access
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)

    for res in resolutions:
        sc.tl.leiden(
            adata,
            key_added=f"leiden_res_{res:4.2f}",
            resolution=res,
            flavor="igraph",
            n_iterations=2,
        )
    return adata


def _plot_umap_leiden(
    adata,
    resolutions: list[float],
    figdir: str | None,
    sample: str | None,
) -> None:
    """
    Save a multi-panel UMAP coloured by a subset of Leiden resolutions.
    Only resolutions that were actually computed are included.
    """
    res_cols = [
        f"leiden_res_{r:4.2f}"
        for r in resolutions
        if f"leiden_res_{r:4.2f}" in adata.obs
    ]
    if not res_cols:
        return

    _set_figdir(figdir)
    sc.pl.umap(
        adata,
        color=res_cols,
        legend_loc="on data",
        save=f"_{sample}_umap_leiden" if sample else None,
        show=False,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def normalize_and_pca(
    adata,
    n_top_genes: int = 2000,
    resolutions: list[float] | None = None,
    qc_output_dir=None,
    sample: str | None = None,
    remove_ig: bool = True,
) -> object:
    """
    Run the standard preprocessing cascade on a filtered AnnData object.

    Steps
    -----
    1. Library-size normalisation + log1p transform.
    2. Highly variable gene (HVG) selection.
    3. (Optional) Remove immunoglobulin genes from the HVG set.
    4. PCA.
    5. kNN graph + UMAP embedding.
    6. Leiden clustering at multiple resolutions.

    Parameters
    ----------
    adata : AnnData
        Filtered object with raw counts in ``adata.X`` or ``adata.layers["counts"]``.
        Should already have had :func:`~sctools.qc.calc_qc_metrics` applied.
    n_top_genes : int
        Number of highly variable genes to select (default 2 000).
    resolutions : list of float, optional
        Leiden resolutions to compute.  Defaults to ``[0.02, 0.2, 0.5, 1.0, 2.0]``.
    qc_output_dir : path-like, optional
        Directory for QC plots.  No plots are saved when *None*.
    sample : str, optional
        Sample name used as a filename suffix for saved plots.
    remove_ig : bool
        If ``True`` (default), immunoglobulin genes are excluded from the
        HVG set before PCA to prevent antibody-driven clustering.

    Returns
    -------
    AnnData
        The same object, modified in place and returned for chaining.

    Notes
    -----
    Normalisation overwrites ``adata.X``.  If you need the raw counts later,
    make sure they are stored in ``adata.layers["counts"]`` before calling
    this function (``io.process_library`` does this automatically).
    """
    if resolutions is None:
        resolutions = _DEFAULT_RESOLUTIONS

    should_save, figdir = _save_path(qc_output_dir, sample)

    adata = _run_normalisation(adata)
    adata = _select_hvgs(adata, n_top_genes=n_top_genes, remove_ig=remove_ig,
                         figdir=figdir if should_save else None, sample=sample)
    adata = _run_pca(adata, figdir=figdir if should_save else None, sample=sample)
    adata = _run_neighbors_and_umap(adata)
    adata = _run_leiden(adata, resolutions)

    _plot_umap_leiden(
        adata,
        resolutions=_UMAP_PLOT_RESOLUTIONS,
        figdir=figdir if should_save else None,
        sample=sample,
    )

    return adata


def remove_ig_hvg(adata) -> object:
    """
    Remove immunoglobulin genes from the highly variable gene set.

    Does **not** remove the genes from the object — it only sets
    ``adata.var["highly_variable"]`` to ``False`` for matching genes.
    This prevents antibody constant-region expression from dominating
    PCA in B-cell-rich datasets.

    Parameters
    ----------
    adata : AnnData
        Object on which :func:`scanpy.pp.highly_variable_genes` has already
        been called.

    Returns
    -------
    AnnData
        The same object, modified in place and returned for chaining.

    Notes
    -----
    Ig gene prefixes are defined in the module-level constant ``_IG_PREFIXES``
    (default: ``IGH``, ``IGK``, ``IGL``).  TCR genes (``TRA``, ``TRB``) are
    not excluded by default but can be added there if needed.
    """
    if "highly_variable" not in adata.var.columns:
        raise ValueError(
            "adata.var has no 'highly_variable' column. "
            "Run sc.pp.highly_variable_genes() before remove_ig_hvg()."
        )

    ig_mask = adata.var_names.str.startswith(_IG_PREFIXES)
    n_removed = int((adata.var["highly_variable"] & ig_mask).sum())
    print(f"Removing {n_removed} Ig gene(s) from HVG set (prefixes: {_IG_PREFIXES}).")

    adata.var["highly_variable"] = adata.var["highly_variable"] & ~ig_mask
    return adata