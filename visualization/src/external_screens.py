"""External CRISPR screen results (casTLE) as comparison axes for the current OPS screen.

Two prior screens are supported, both keyed on the 'Symbol' column:

* ADCP combo (rep 1+2) -- ``ADCP_2208_combo_rep12.csv``, a 2208-gene sublibrary
  screen. Its 'casTLE p-value' column is 'N/A' for every row, so this screen has
  **no significance measure**; 'Combo casTLE Score' (a likelihood-ratio-like
  statistic, higher = stronger) is the only hit-strength signal available.
* Beads genome-wide -- ``beads_genome_wide.tsv``, exported from
  ``beads_genome_wide.xlsx`` by ``external_data/export_screens.py`` (the
  visualization env has pandas but not openpyxl). This one does carry a real
  'casTLE p-value', a Hochberg-adjusted p-value, and boolean 1%/5%/10% FDR flags.

Gene symbols are matched case-insensitively (both screens use all-caps mouse
symbols, e.g. MAGT1, while the aggregate tables use Magt1). Duplicate symbols --
104 in the genome-wide screen, 1 in ADCP -- are resolved by keeping the row with
the lowest casTLE p-value, breaking ties on the largest absolute combo effect.
"""

import logging
import os

import numpy as np
import pandas as pd
import streamlit as st

from src.config import ADCP_COMBO_PATH, BEADS_GENOME_WIDE_PATH

logger = logging.getLogger(__name__)

GENE_COL = "Symbol"

# Screen key -> display/loading spec.
SCREEN_SPECS = {
    "ADCP": {
        "label": "ADCP combo (rep 1+2)",
        "path": ADCP_COMBO_PATH,
        "sep": ",",
        # 'casTLE p-value' is N/A for every row in this file.
        "fdr_col": None,
    },
    "Beads": {
        "label": "Beads genome-wide",
        "path": BEADS_GENOME_WIDE_PATH,
        "sep": "\t",
        "fdr_col": "Hochberg",
    },
}

# Columns offered on the external axis, in dropdown order.
METRIC_COLS = [
    "Combo casTLE Effect",
    "Combo casTLE Score",
    "casTLE Effect 1",
    "casTLE Effect 2",
]

# Additional numeric columns kept for hover text, error bars and filtering.
EXTRA_NUMERIC_COLS = [
    "casTLE Score 1",
    "casTLE Score 2",
    "# elements 1",
    "# elements 2",
    "casTLE p-value",
    "Hochberg",
    "Minimum Effect Estimate",
    "Maximum Effect Estimate",
    "1% FDR",
    "5% FDR",
    "10% FDR",
]

# Text columns kept for the gene detail panel.
TEXT_COLS = ["#GeneID", "GeneInfo"]

# Metrics that are unsigned magnitudes rather than signed effects; a zero line
# and symmetric colour scales make no sense for them.
UNSIGNED_METRICS = {"Combo casTLE Score", "casTLE Score 1", "casTLE Score 2"}


@st.cache_data
def _load_screen(path, sep, screen_key):
    """Return a DataFrame indexed by uppercased gene symbol, one row per symbol."""
    if not path:
        return pd.DataFrame()
    if not os.path.exists(path):
        logger.warning(f"External screen {screen_key} not found at: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, sep=sep)
    if GENE_COL not in df.columns:
        logger.warning(f"External screen {path} has no {GENE_COL!r} column")
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    for col in METRIC_COLS + EXTRA_NUMERIC_COLS:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    for col in TEXT_COLS:
        if col in df.columns:
            out[col] = df[col].astype(str)

    present_metrics = [c for c in METRIC_COLS if c in out.columns]
    if not present_metrics:
        logger.warning(f"External screen {path} has none of {METRIC_COLS}")
        return pd.DataFrame()

    out["_gene_key"] = df[GENE_COL].astype(str).str.strip().str.upper()
    out = out[(out["_gene_key"] != "") & (out["_gene_key"].str.lower() != "nan")]
    # Drop rows with no usable measurement at all.
    out = out.dropna(subset=present_metrics, how="all")

    # Resolve duplicate symbols: most significant first, then largest effect.
    effect = out.get("Combo casTLE Effect", pd.Series(np.nan, index=out.index))
    out["_abs_effect"] = effect.abs()
    sort_cols, ascending = [], []
    if "casTLE p-value" in out.columns:
        sort_cols.append("casTLE p-value")
        ascending.append(True)
    sort_cols.append("_abs_effect")
    ascending.append(False)
    out = out.sort_values(sort_cols, ascending=ascending, na_position="last")
    out = out.drop_duplicates(subset="_gene_key", keep="first")
    return out.drop(columns="_abs_effect").set_index("_gene_key")


def load_screen(screen_key):
    """Return the screen table for ``screen_key``, or an empty DataFrame."""
    spec = SCREEN_SPECS.get(screen_key)
    if spec is None:
        return pd.DataFrame()
    return _load_screen(spec["path"], spec["sep"], screen_key)


def available_screens():
    """Screen keys whose backing file is present and loadable."""
    return [k for k in SCREEN_SPECS if not load_screen(k).empty]


def screen_label(screen_key):
    return SCREEN_SPECS.get(screen_key, {}).get("label", screen_key)


def screen_fdr_col(screen_key):
    """Adjusted-p column for this screen, or None when it has no significance data."""
    spec = SCREEN_SPECS.get(screen_key, {})
    fdr_col = spec.get("fdr_col")
    if not fdr_col:
        return None
    table = load_screen(screen_key)
    if fdr_col not in table.columns or table[fdr_col].notna().sum() == 0:
        return None
    return fdr_col


def screen_metrics(screen_key):
    """Metric columns actually populated in this screen, in dropdown order."""
    table = load_screen(screen_key)
    if table.empty:
        return []
    return [c for c in METRIC_COLS if c in table.columns and table[c].notna().any()]


def attach_screen(df, screen_key, gene_col=None, columns=None):
    """Add external screen columns to ``df``, matched on uppercased gene symbol.

    Added columns are prefixed with ``'<screen_key>: '`` so two screens can be
    attached to the same frame without collisions.

    :param df: DataFrame of gene-level data
    :param screen_key: key into :data:`SCREEN_SPECS`
    :param gene_col: column holding gene symbols; if None, the index is used
    :param columns: subset of screen columns to attach; defaults to all
    :returns: a copy of ``df`` with the added column(s)
    """
    out = df.copy()
    table = load_screen(screen_key)
    if table.empty:
        return out
    genes = (out.index if gene_col is None else out[gene_col]).astype(str).str.upper()
    for col in columns or list(table.columns):
        if col in table.columns:
            out[f"{screen_key}: {col}"] = genes.map(table[col]).values
    return out


def matched_genes(gene_symbols, screen_key):
    """Return (matched, unmatched) uppercased symbols for ``screen_key``."""
    table = load_screen(screen_key)
    keys = pd.Index(pd.Series(gene_symbols, dtype=object).astype(str).str.upper())
    if table.empty:
        return pd.Index([]), keys
    hit = keys.isin(table.index)
    return keys[hit], keys[~hit]
