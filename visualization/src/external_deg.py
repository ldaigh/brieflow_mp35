"""External bulk RNA-seq DEG data as plottable features.

Source: TSV exports of two workbooks (CZ DEG ADCP 6 12 24 hr.xlsx and CZ DEG
ADCP effero super_effero.xlsx), produced by external_data/export_deg_sheets.py
(the visualization env has pandas but not openpyxl, so the workbooks are not
read directly).

Each feature behaves like a CellProfiler feature: a value column plus a
matching '<feature>_fdr' column (from the workbook's padj), so genes can be
ranked/filtered/colored the same way as bootstrap-significant features.
Duplicate gene-symbol rows (a handful of genes with multiple ensembl IDs) are
resolved by keeping the row with the lowest padj.
"""

import logging
import os

import pandas as pd
import streamlit as st

from src.config import (
    ADCP_6HR_DEG_PATH,
    ADCP_12HR_DEG_PATH,
    ADCP_24HR_DEG_PATH,
    ADCP_EFFERO_DEG_PATH,
    EFFERO_SUPEREFFERO_DEG_PATH,
)

logger = logging.getLogger(__name__)

# Feature name -> (tsv path, gene col, value col, padj col)
_DEG_SPECS = {
    "ADCP_6hr_log2FC": (ADCP_6HR_DEG_PATH, "mgi_symbol", "log2FoldChange", "padj"),
    "ADCP_12hr_log2FC": (ADCP_12HR_DEG_PATH, "mgi_symbol", "log2FoldChange", "padj"),
    "ADCP_24hr_log2FC": (ADCP_24HR_DEG_PATH, "mgi_symbol", "log2FoldChange", "padj"),
    "Effero_baseMean": (
        ADCP_EFFERO_DEG_PATH,
        "Effero_mgi_symbol",
        "Effero_baseMean",
        "Effero_padj",
    ),
    "Effero_log2FC": (
        ADCP_EFFERO_DEG_PATH,
        "Effero_mgi_symbol",
        "Effero_log2FoldChange",
        "Effero_padj",
    ),
    "SuperEffero_log2FC": (
        EFFERO_SUPEREFFERO_DEG_PATH,
        "SuperEffero_mgi_symbol",
        "SuperEffero_log2FoldChange",
        "SuperEffero_padj",
    ),
}

EXTERNAL_DEG_FEATURES = list(_DEG_SPECS.keys())


@st.cache_data
def _load_deg_table(path, gene_col, value_col, fdr_col, feature_name):
    """Return DataFrame indexed by uppercased gene symbol with value + fdr columns."""
    if not path or not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, sep="\t")
    missing = [c for c in (gene_col, value_col, fdr_col) if c not in df.columns]
    if missing:
        logger.warning(f"DEG TSV {path} missing columns: {missing}")
        return pd.DataFrame()

    fdr_feature = f"{feature_name}_fdr"
    out = pd.DataFrame(
        {
            feature_name: pd.to_numeric(df[value_col], errors="coerce"),
            fdr_feature: pd.to_numeric(df[fdr_col], errors="coerce"),
        }
    )
    out["_gene_key"] = df[gene_col].astype(str).str.strip().str.upper()
    out = out.dropna(subset=[feature_name])
    out = out[out["_gene_key"] != ""]
    # Keep the most significant (lowest padj) row per duplicated gene symbol.
    out = out.sort_values(fdr_feature, na_position="last")
    out = out.drop_duplicates(subset="_gene_key", keep="first")
    return out.set_index("_gene_key")


def load_deg_table(feature_name):
    spec = _DEG_SPECS.get(feature_name)
    if spec is None:
        return pd.DataFrame()
    return _load_deg_table(*spec, feature_name)


def available_external_deg_features():
    """Feature names whose backing TSV is present and loadable."""
    return [f for f in EXTERNAL_DEG_FEATURES if not load_deg_table(f).empty]


def attach_external_deg(df, gene_col=None, features=None, add_fdr=True):
    """Add value (and fdr) columns for the given/available DEG features, matched on gene symbol.

    :param df: DataFrame of gene-level data
    :param gene_col: column holding gene symbols; if None, the index is used
    :param features: subset of EXTERNAL_DEG_FEATURES to attach; defaults to all
    :param add_fdr: also add the '<feature>_fdr' column(s)
    :returns: a copy of ``df`` with the added column(s)
    """
    out = df.copy()
    genes = (out.index if gene_col is None else out[gene_col]).astype(str).str.upper()
    for feature_name in features or EXTERNAL_DEG_FEATURES:
        table = load_deg_table(feature_name)
        if table.empty:
            continue
        out[feature_name] = genes.map(table[feature_name]).values
        if add_fdr:
            out[f"{feature_name}_fdr"] = genes.map(table[f"{feature_name}_fdr"]).values
    return out
