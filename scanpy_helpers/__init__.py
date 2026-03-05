# sctools/__init__.py

from .io import process_library

from .qc import calc_qc_metrics, flag_cells, expected_doublet_rate

from .preprocessing import normalize_and_pca, remove_ig_hvg

from .plotting import (
    plot_with_repel,
    boxplot_sample_means,
    boxplot_sample_means_v2,
    boxplot_cells_with_sample_means,   # ← was missing
)

from .stats import (
    aggregate_by_sample,               # ← was missing
    sample_means,
    drop_conds_with_few_samples,
    has_enough_replicates,             # ← was missing
    anova_pairwise,
    get_variance_components,           # ← was missing (was private _get_variance_components)
)

from .utils import (                   # ← entire module was missing
    subset_and_filter,
    count_cells_per_sample,
    agg_name,
)