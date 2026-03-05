# sctools/io.py

"""
I/O utilities for loading and saving 10X single-cell libraries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


from .qc import calc_qc_metrics, expected_doublet_rate, flag_cells
from .preprocessing import normalize_and_pca

__all__ = ["process_library"]


# ── Private helpers ────────────────────────────────────────────────────────────

def _load_adata(
    raw_h5: Path | None,
    cellbender_h5: Path | None,
    matrix_folder: Path | None,
):
    if cellbender_h5 is not None and raw_h5 is not None:
        try:
            from cellbender.remove_background.downstream import (
                load_anndata_from_input_and_output,
            )
        except ImportError as e:
            raise ImportError(
                "CellBender is required when cellbender_h5 is provided. "
                "Install it following the instructions at "
                "https://cellbender.readthedocs.io"
            ) from e
        adata = load_anndata_from_input_and_output(
            input_file=str(raw_h5),
            output_file=str(cellbender_h5),
            input_layer_key="raw",
        )
        return adata, True

    if raw_h5 is not None:
        adata = sc.read_10x_h5(str(raw_h5))
        return adata, False

    if matrix_folder is not None:
        adata = sc.read_10x_mtx(str(matrix_folder))
        return adata, False

    raise ValueError(
        "Provide at least one of: raw_h5, cellbender_h5 + raw_h5, or matrix_folder."
    )

def _assign_sample_labels(adata, sample: str, multiplex_csv: Path | None, cmo_dict: dict | None):
    """
    Assign per-cell sample labels, either from a multiplexing CSV or as a
    single constant label for the whole library.
    """
    if multiplex_csv is not None:
        if cmo_dict is None:
            raise ValueError("cmo_dict is required when multiplex_csv is provided.")

        multiplex_df = pd.read_csv(multiplex_csv)
        multiplex_df["sample"] = multiplex_df["feature_call"].map(cmo_dict)

        adata.obs = pd.merge(
            left=adata.obs,
            right=multiplex_df,
            how="left",
            left_on="barcode",
            right_on="cell_barcode",
        )
        adata.obs["sample"] = (
            adata.obs["sample"]
            .astype("category")
            .cat.add_categories("Missing")
            .fillna("Missing")
        )
    else:
        adata.obs["sample"] = sample

    return adata


def _prefix_barcodes(adata):
    """
    Prepend the sample label to every barcode so barcodes stay unique
    after merging multiple libraries.
    e.g. 'ACGT-1'  →  'SampleA_ACGT-1'
    """
    adata.obs_names = [
        f"{s}_{bc}"
        for s, bc in zip(adata.obs["sample"].astype(str), adata.obs_names)
    ]
    adata.obs_names_make_unique()
    return adata


def _resolve_output_path(save_obj: str | Path | None, sample: str) -> Path:
    """
    Return the .h5ad output path, creating parent directories as needed.
    Falls back to <cwd>/data/tempobjs/<sample>_raw.h5ad if save_obj is None.
    """
    if save_obj is None:
        outpath = Path.cwd() / "data" / "tempobjs" / f"{sample}_raw.h5ad"
    else:
        outpath = Path(save_obj)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    return outpath


def _set_counts_layer(adata, cellbender: bool):
    """
    Standardise the counts layer so downstream code always reads from
    adata.layers["counts"], regardless of input type.
    """
    if cellbender and "cellbender" in adata.layers:
        adata.layers["counts"] = adata.layers["cellbender"]
        adata.X = adata.layers["cellbender"]
    else:
        # Raw counts are already in adata.X; just copy them
        adata.layers["counts"] = adata.X.copy()
    return adata


def _coerce_scrublet_dtypes(adata):
    """
    Scrublet can leave nullable/object dtypes that break h5ad serialisation.
    Cast to plain Python bool / float.
    """
    if "predicted_doublet" in adata.obs:
        adata.obs["predicted_doublet"] = (
            adata.obs["predicted_doublet"].fillna(False).astype(bool)
        )
    if "doublet_score" in adata.obs:
        adata.obs["doublet_score"] = adata.obs["doublet_score"].astype(float)
    return adata


# ── Public API ────────────────────────────────────────────────────────────────

def process_library(
    sample: str,
    raw_h5: str | Path | None = None,
    matrix_folder: str | Path | None = None,
    cellbender_h5: str | Path | None = None,
    multiplex_csv: str | Path | None = None,
    cmo_dict: dict | None = None,
    species: str = "human",
    save_obj: str | Path | None = None,
    qc_output_dir: str | Path | None = None,
    pct_mito_cutoff: float = 5.0,
    min_counts_cutoff: int = 1000,
):
    """
    Load a 10X library and run initial QC, doublet detection, and clustering.

    Steps
    -----
    1. Load data from .h5 / CellBender .h5 / MEX matrix folder.
    2. (Optional) Assign per-cell sample labels from a multiplexing CSV.
    3. Calculate QC metrics and save violin plots.
    4. Predict doublets with Scrublet.
    5. Normalise, run PCA/UMAP, and find Leiden clusters.
    6. Flag low-quality cells (high mito %, low counts, predicted doublets).
    7. Write the object to disk and return it.

    Parameters
    ----------
    sample
        Sample name used for labelling barcodes and output filenames.
    raw_h5
        Path to the raw CellRanger .h5 file.
    matrix_folder
        Path to a CellRanger MEX matrix folder (alternative to .h5).
    cellbender_h5
        Path to a CellBender output .h5 (requires raw_h5 too).
    multiplex_csv
        CSV with columns ``cell_barcode`` and ``feature_call`` for CMO demultiplexing.
    cmo_dict
        Mapping from CMO feature call → sample name, e.g. ``{"CMO301": "SampleA"}``.
    species
        Only ``"human"`` is supported right now.
    save_obj
        Full path for the output .h5ad file.  Defaults to
        ``<cwd>/data/tempobjs/<sample>_raw.h5ad``.
    qc_output_dir
        Directory for QC plots.  Defaults to ``<cwd>/data/QC/<sample>``.
    pct_mito_cutoff
        Cells with mitochondrial read fraction above this are flagged.
    min_counts_cutoff
        Cells with total counts below this are flagged.

    Returns
    -------
    AnnData
        Annotated data object with QC metrics, doublet scores, leiden clusters,
        and a boolean ``flagged`` column in ``.obs``.
    """
    # ── 1. Load ───────────────────────────────────────────────────────────────
    adata, cellbender = _load_adata(
        raw_h5=Path(raw_h5) if raw_h5 else None,
        cellbender_h5=Path(cellbender_h5) if cellbender_h5 else None,
        matrix_folder=Path(matrix_folder) if matrix_folder else None,
    )
    adata.var_names_make_unique()
    print(f"Loaded data for sample '{sample}'...")

    # ── 2. Sample labels ──────────────────────────────────────────────────────
    adata = _assign_sample_labels(adata, sample, multiplex_csv, cmo_dict)

    # ── 3. QC metrics & plots ─────────────────────────────────────────────────
    if species.lower() != "human":
        raise NotImplementedError(
            f"QC metric calculations are not implemented for species={species!r}. "
            "Only 'human' is supported."
        )

    adata = calc_qc_metrics(adata)

    qc_output_dir = (
        Path(qc_output_dir) if qc_output_dir else Path.cwd() / "data" / "QC" / sample
    )
    qc_output_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = str(qc_output_dir)

    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
        save=f"_{sample}_qc_violin",
        show=False,
    )

    # ── 4. Doublet detection ──────────────────────────────────────────────────
    doublet_rate = expected_doublet_rate(adata.n_obs)
    adata = _prefix_barcodes(adata)
    print("Predicting doublets...")
    sc.pp.scrublet(adata, expected_doublet_rate=doublet_rate)

    # Basic pre-filter before clustering
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)

    # ── 5. Normalise / PCA / UMAP / Leiden ───────────────────────────────────
    adata = normalize_and_pca(adata, qc_output_dir=qc_output_dir, sample=sample)

    # ── 6. Flag low-quality cells ─────────────────────────────────────────────
    adata = flag_cells(
        adata,
        pct_mito_cutoff=pct_mito_cutoff,
        min_counts_cutoff=min_counts_cutoff,
    )

    sc.pl.umap(
        adata,
        color=["flagged", "predicted_doublet", "pct_counts_mt", "total_counts"],
        ncols=2,
        save=f"_{sample}_flagged_cells",
        show=False,
    )

    # ── 7. Finalise & write ───────────────────────────────────────────────────
    adata = _set_counts_layer(adata, cellbender)
    adata = _coerce_scrublet_dtypes(adata)

    # Keep barcode as an explicit column (useful after concat)
    adata.obs["barcode"] = adata.obs_names
    adata.obs["sample"] = sample          # ← overwrite in case multiplex relabelled it

    outpath = _resolve_output_path(save_obj, sample)
    adata.write(outpath)
    print(f"Saved to {outpath}")

    return adata