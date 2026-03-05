# sctools/stats.py

"""
Sample-level aggregation, ANOVA with pairwise contrasts, and model diagnostics.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.formula.api as smf

__all__ = [
    "aggregate_by_sample",
    "sample_means",
    "drop_conds_with_few_samples",
    "has_enough_replicates",
    "anova_pairwise",
]


# ── Private helpers: data wrangling ───────────────────────────────────────────

def _as_fit(obj):
    """
    Accept either a bare statsmodels result or a (table, fit, ...) tuple;
    always return the fit object.
    """
    if isinstance(obj, (tuple, list)) and len(obj) >= 2:
        return obj[1]
    return obj


def _is_sm_result(x) -> bool:
    """
    Duck-type check for a statsmodels result object.
    Any result that exposes ``.params`` and ``.cov_params()`` qualifies.
    """
    return (
        x is not None
        and hasattr(x, "params")
        and hasattr(x, "cov_params")
    )


def _condition_levels(series: pd.Series) -> list:
    """
    Return the unique levels of a condition column in a stable order.
    Respects an existing categorical order; falls back to alphabetical.
    """
    if pd.api.types.is_categorical_dtype(series):
        return list(series.cat.categories)
    return sorted(series.unique())


def _validate_sample_means_inputs(
    df: pd.DataFrame,
    y: str,
    group: str,
    cond: str,
) -> None:
    """Raise an informative error if required columns are missing."""
    missing = {y, group, cond} - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}.")


# ── Private helpers: ANOVA machinery ──────────────────────────────────────────

def _treatment_contrast_vector(
    level: str,
    base: str,
    pnames: list[str],
    cond: str,
) -> np.ndarray:
    """
    Build the linear contrast vector L such that L @ params = mean(level).

    Under treatment (dummy) coding the intercept encodes the baseline mean,
    and each ``C(cond)[T.level]`` coefficient encodes the *difference* from
    baseline.  For a pairwise contrast ``mean(a) - mean(b)`` we compute
    ``L_a - L_b``.

    Parameters
    ----------
    level  : Condition level whose mean we want to recover.
    base   : Baseline level (treatment-coding reference).
    pnames : Parameter names from the fitted model (``fit.params.index``).
    cond   : Name of the condition column used in the formula.
    """
    v = np.zeros(len(pnames))
    if level == base:
        # Baseline mean = intercept alone → coefficient already 1
        v[pnames.index("Intercept")] = 1.0
    else:
        nm = f"C({cond})[T.{level}]"
        if nm in pnames:
            v[pnames.index("Intercept")] = 1.0   # intercept is always part of the mean
            v[pnames.index(nm)]          = 1.0
    return v


def _pairwise_contrasts(
    fit,
    levels: list[str],
    cond: str,
    alpha: float,
) -> pd.DataFrame:
    """
    Compute all pairwise contrasts from a fitted OLS/WLS model.

    Each contrast ``a - b`` yields an estimate, robust SE, z-score, and
    two-sided p-value.  FDR correction (Benjamini–Hochberg) is applied
    across all pairs.

    Parameters
    ----------
    fit    : Fitted statsmodels result (OLS or WLS with robust covariance).
    levels : Ordered list of condition levels.
    cond   : Name of the condition column used in the formula.
    alpha  : Significance level used to construct confidence intervals.

    Returns
    -------
    DataFrame with columns: contrast, estimate, se, z, p_two_sided,
    ci{100*(1-alpha)}_low, ci{100*(1-alpha)}_high, p_fdr,
    significant_fdr_0_05.
    """
    params = fit.params
    cov    = fit.cov_params()
    pnames = list(params.index)
    base   = levels[0]
    zcrit  = stats.norm.ppf(1 - alpha / 2)
    ci_tag = f"ci{int((1 - alpha) * 100)}"

    rows = []
    for a, b in combinations(levels, 2):
        La = _treatment_contrast_vector(a, base, pnames, cond)
        Lb = _treatment_contrast_vector(b, base, pnames, cond)
        L  = La - Lb

        est = float(L @ params.values)
        se2 = float(L @ cov.values @ L)
        se  = np.sqrt(se2) if se2 > 0 else np.nan

        if np.isfinite(se) and se > 0:
            z        = est / se
            p        = float(2 * stats.norm.sf(abs(z)))
            lo, hi   = est - zcrit * se, est + zcrit * se
        else:
            z = p = lo = hi = np.nan

        rows.append({
            "contrast":         f"{a} - {b}",
            "estimate":         est,
            "se":               se,
            "z":                z,
            "p_two_sided":      p,
            f"{ci_tag}_low":    lo,
            f"{ci_tag}_high":   hi,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        _, out["p_fdr"] = fdrcorrection(out["p_two_sided"].fillna(1).values, alpha=0.05)
        out["significant_fdr_0_05"] = out["p_fdr"] < 0.05
        out = out.sort_values(
            ["p_fdr", "p_two_sided"], na_position="last"
        ).reset_index(drop=True)
    return out


# ── Private helpers: variance components ──────────────────────────────────────

def _extract_tau2(fit) -> float:
    """
    Extract the random-intercept variance (tau²) from a MixedLM result.
    Returns 0.0 for fixed-effects models or on failure.
    """
    if not hasattr(fit, "cov_re"):
        return 0.0
    try:
        cov_re = np.asarray(fit.cov_re)
        return float(cov_re[0, 0]) if cov_re.size > 0 else 0.0
    except Exception:
        return 0.0


def _extract_sigma2(fit) -> float:
    """
    Extract the residual variance (sigma²) from a statsmodels result.

    Tries ``scale`` first (GLM/MixedLM), then ``mse_resid`` (OLS), then
    computes SSR / df_resid as a last resort.  Returns 1.0 if none succeed
    so that ICC = 0 rather than NaN.
    """
    for attr in ("scale", "mse_resid"):
        if hasattr(fit, attr):
            try:
                val = float(getattr(fit, attr))
                if np.isfinite(val):
                    return val
            except Exception:
                pass

    if hasattr(fit, "ssr") and hasattr(fit, "df_resid") and fit.df_resid > 0:
        return float(fit.ssr / fit.df_resid)

    return 1.0   # harmless fallback → ICC = 0


# ── Public API ────────────────────────────────────────────────────────────────

def aggregate_by_sample(
    adata,
    value_col: str = "mu_freq_IGH",
    aggfunc: str | callable = "median",
) -> pd.DataFrame:
    """
    Collapse an AnnData's ``.obs`` to one row per sample.

    Parameters
    ----------
    adata : AnnData
        Must have ``sample``, ``condition``, and *value_col* in ``.obs``.
    value_col : str
        Numeric column to aggregate.
    aggfunc : str or callable
        Aggregation function accepted by ``groupby.agg`` (e.g. ``"median"``,
        ``"mean"``, ``np.std``).

    Returns
    -------
    DataFrame with columns: condition, sample, *value_col*.
    """
    df = adata.obs[["sample", "condition", value_col]].dropna()
    return (
        df.groupby(["condition", "sample"])[value_col]
          .agg(aggfunc)
          .reset_index()
          .dropna(subset=[value_col])
    )


def sample_means(
    df: pd.DataFrame,
    y: str = "mu_freq_IGH",
    group: str = "sample",
    cond: str = "condition",
    min_cells: int = 3,
    stat: str = "mean",
) -> pd.DataFrame:
    """
    Collapse cell-level data to one summary value per biological replicate.

    Parameters
    ----------
    df : DataFrame
        Cell-level data.
    y : str
        Numeric column to summarise.
    group : str
        Column identifying biological replicates (e.g. ``"sample"``).
    cond : str
        Column identifying experimental conditions.
    min_cells : int
        Samples with fewer than *min_cells* non-missing observations of *y*
        are dropped.  Set to ``0`` or ``None`` to keep all samples.
    stat : str
        Summary statistic passed to ``groupby.agg`` (default ``"mean"``).

    Returns
    -------
    DataFrame with columns: group, cond, ``"sample_mean"``, ``"n_cells"``.
    An empty DataFrame with those columns is returned when no data survive
    filtering.
    """
    _validate_sample_means_inputs(df, y, group, cond)

    _empty = pd.DataFrame(columns=[group, cond, "sample_mean", "n_cells"])

    d = df[[y, group, cond]].dropna()
    if d.empty:
        return _empty

    cell_counts = d.groupby([group, cond], observed=True).size().rename("n_cells")

    if min_cells:
        keep = cell_counts[cell_counts >= int(min_cells)].index
        if len(keep) == 0:
            return _empty
        d = d.set_index([group, cond]).loc[keep].reset_index()

    agg = (
        d.groupby([group, cond], observed=True)[y]
         .agg(stat)
         .rename("sample_mean")
         .reset_index()
    )
    return agg.merge(cell_counts.reset_index(), on=[group, cond], how="left")


def drop_conds_with_few_samples(
    per: pd.DataFrame | None,
    cond: str = "condition",
    min_per_cond: int = 2,
) -> pd.DataFrame:
    """
    Remove conditions that have fewer than *min_per_cond* samples.

    Useful as a pre-flight check before ``anova_pairwise``: the ANOVA needs
    at least two replicates per group and at least two groups.

    Parameters
    ----------
    per : DataFrame or None
        Per-sample summary table (output of :func:`sample_means`).
    cond : str
        Condition column name.
    min_per_cond : int
        Minimum number of samples required to keep a condition.

    Returns
    -------
    Filtered DataFrame.  An empty DataFrame is returned when fewer than two
    conditions survive the filter.
    """
    if per is None or per.empty:
        return per

    counts = per.groupby(cond, observed=True).size()
    keep   = counts[counts >= int(min_per_cond)].index
    per2   = per[per[cond].isin(keep)].copy()

    if per2[cond].nunique() < 2:
        return per2.iloc[0:0]   # empty but correctly typed
    return per2


def has_enough_replicates(
    per: pd.DataFrame,
    cond: str = "condition",
    min_per_cond: int = 2,
) -> bool:
    """
    Return ``True`` if every condition in *per* has at least *min_per_cond* samples.

    Parameters
    ----------
    per : DataFrame
        Per-sample summary table.
    cond : str
        Condition column name.
    min_per_cond : int
        Minimum replicate count required.
    """
    if per is None or per.empty:
        return False
    counts = per.groupby(cond, observed=True).size()
    return bool((counts.min() >= min_per_cond) and counts.size >= 2)


def anova_pairwise(
    per: pd.DataFrame,
    y: str = "sample_mean",
    cond: str = "condition",
    alpha: float = 0.05,
    weight: str | None = None,
    cov_type: str = "HC3",
) -> tuple[pd.DataFrame, object]:
    """
    Fit a sample-level ANOVA and return all pairwise condition contrasts.

    The model is ``y ~ C(cond)`` fit by OLS (or WLS when *weight* is given),
    with heteroscedasticity-robust standard errors (HC3 by default).
    Pairwise contrasts are derived analytically from the model coefficients
    rather than by re-fitting, so the FDR correction is applied across all
    pairs simultaneously.

    Parameters
    ----------
    per : DataFrame
        Per-sample summary table; typically the output of :func:`sample_means`.
        Must contain columns *y* and *cond*.
    y : str
        Column to use as the response variable (default ``"sample_mean"``).
    cond : str
        Column identifying experimental conditions.
    alpha : float
        Significance level for confidence intervals (default 0.05 → 95 % CI).
    weight : str, optional
        Column of non-negative weights for WLS (e.g. ``"n_cells"``).
        When *None*, ordinary least squares is used.
    cov_type : str
        Covariance estimator for robust SEs.  ``"HC3"`` is recommended for
        small samples; ``"HC1"`` matches Stata's default.

    Returns
    -------
    results : DataFrame
        One row per pair with columns: contrast, estimate, se, z,
        p_two_sided, ci95_low, ci95_high, p_fdr, significant_fdr_0_05.
        Empty DataFrame when there are fewer than two conditions.
    fit : statsmodels result or None
        The fitted model object; useful for residual diagnostics.

    Notes
    -----
    The z-distribution (rather than t) is used for p-values because robust
    SE estimators do not have a clean t-distribution theory in small samples.
    For very small n (< 10 total samples) interpret p-values cautiously.

    Examples
    --------
    >>> per = sample_means(df, y="mu_freq_IGH")
    >>> per = drop_conds_with_few_samples(per)
    >>> if has_enough_replicates(per):
    ...     results, fit = anova_pairwise(per)
    """
    if per is None or per.empty:
        return pd.DataFrame(), None

    levels = _condition_levels(per[cond])
    if len(levels) < 2:
        return pd.DataFrame(), None

    formula = f"{y} ~ C({cond})"
    if weight is None:
        fit = smf.ols(formula, data=per).fit(cov_type=cov_type)
    else:
        fit = smf.wls(formula, data=per, weights=per[weight]).fit(cov_type=cov_type)

    results = _pairwise_contrasts(fit, levels, cond, alpha)
    return results, fit


def get_variance_components(fit_or_tuple) -> tuple[float, float]:
    """
    Extract (tau², sigma²) for intraclass correlation (ICC) calculation.

    tau²   — between-sample (random-intercept) variance; 0 for fixed-effects models.
    sigma² — within-sample (residual) variance.

    ICC = tau² / (tau² + sigma²).  Returns ``(0.0, 1.0)`` (ICC = 0) when the
    fit object is unusable.

    Parameters
    ----------
    fit_or_tuple : statsmodels result or (table, fit, ...) tuple
        Accepts the direct output of :func:`anova_pairwise` or a bare fit.

    Returns
    -------
    tau2   : float
    sigma2 : float
    """
    fit = _as_fit(fit_or_tuple)
    if not _is_sm_result(fit):
        return 0.0, 1.0

    tau2   = _extract_tau2(fit)
    sigma2 = _extract_sigma2(fit)
    return tau2, sigma2