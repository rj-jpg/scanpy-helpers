# scanpy-helpers

Helper functions for processing 10X single-cell libraries with scanpy.

## Installation
```bash
pip install git+https://github.com/YOURNAME/scanpy-helpers.git
```

For RNA velocity support:
```bash
pip install "scanpy-helpers[velocity] @ git+https://github.com/YOURNAME/scanpy-helpers.git"
```

CellBender users: install CellBender separately following their
[official instructions](https://cellbender.readthedocs.io) before calling
`process_library` with `cellbender_h5=`.

## Quick start
```python
import scanpy_helpers as sh

# Load and QC a library
adata = sh.process_library(
    sample="donor1",
    raw_h5="path/to/raw_feature_bc_matrix.h5",
)

# Subset to plasmablasts expressing IgA
ad = sh.subset_and_filter(adata, celltype="Plasmablast", isotype="IGHA")

# Per-sample means and ANOVA
per = sh.sample_means(ad.obs, y="mu_freq_IGH")
per = sh.drop_conds_with_few_samples(per)
if sh.has_enough_replicates(per):
    results, fit = sh.anova_pairwise(per)
```