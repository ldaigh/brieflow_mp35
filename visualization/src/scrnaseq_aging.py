"""External scRNA-seq aging data (macrophage age-coefficients) as a plottable feature.

Source: the 'Ranked_Summary' sheet of the aging myeloid scRNA-seq workbook,
exported to TSV by scRNA-seq/export_ranked_summary.py (the visualization env has
pandas but not openpyxl, so the workbook is not read directly).

Values come from 'Macrophage: mean age-coef'; significance from
'Macrophage: min FDR'. Both are exposed under the feature name
'scRNAseq_aging_data' so the column behaves like any CellProfiler feature
(value column + matching '<feature>_fdr' column).
"""

import logging
import os

import pandas as pd

from src.config import SCRNASEQ_AGING_PATH

logger = logging.getLogger(__name__)

# Feature name shown in the dropdowns
SCRNASEQ_AGING_FEATURE = "scRNAseq_aging_data"
SCRNASEQ_AGING_FDR = f"{SCRNASEQ_AGING_FEATURE}_fdr"

GENE_COL = "Gene (mouse)"
VALUE_COL = "Macrophage: mean age-coef"
FDR_COL = "Macrophage: min FDR"


def load_scrnaseq_aging_table():
    """Return DataFrame indexed by uppercased gene symbol with value + FDR columns.

    Returns an empty DataFrame when the TSV is missing or malformed so callers
    can simply omit the option from their dropdowns.
    """
    if not SCRNASEQ_AGING_PATH:
        return pd.DataFrame()
    if not os.path.exists(SCRNASEQ_AGING_PATH):
        logger.warning(f"scRNA-seq aging TSV not found at: {SCRNASEQ_AGING_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(SCRNASEQ_AGING_PATH, sep="\t")
    missing = [c for c in (GENE_COL, VALUE_COL, FDR_COL) if c not in df.columns]
    if missing:
        logger.warning(
            f"scRNA-seq aging TSV {SCRNASEQ_AGING_PATH} missing columns: {missing}"
        )
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            SCRNASEQ_AGING_FEATURE: pd.to_numeric(df[VALUE_COL], errors="coerce"),
            SCRNASEQ_AGING_FDR: pd.to_numeric(df[FDR_COL], errors="coerce"),
        }
    )
    out["_gene_key"] = df[GENE_COL].astype(str).str.strip().str.upper()
    # Drop genes with no macrophage measurement, and any duplicate gene rows
    out = out.dropna(subset=[SCRNASEQ_AGING_FEATURE])
    out = out[out["_gene_key"] != ""]
    out = out.drop_duplicates(subset="_gene_key", keep="first")
    return out.set_index("_gene_key")


def has_scrnaseq_aging_data():
    """True when the aging feature can be offered as a plotting option."""
    return not load_scrnaseq_aging_table().empty


def attach_scrnaseq_aging(df, gene_col=None, add_fdr=True):
    """Add the aging value (and FDR) columns to ``df``, matched on gene symbol.

    :param df: DataFrame of gene-level data
    :param gene_col: column holding gene symbols; if None, the index is used
    :param add_fdr: also add the '<feature>_fdr' column
    :returns: a copy of ``df`` with the added column(s); unchanged copy if no data
    """
    table = load_scrnaseq_aging_table()
    if table.empty:
        return df.copy()

    out = df.copy()
    genes = (out.index if gene_col is None else out[gene_col]).astype(str).str.upper()
    out[SCRNASEQ_AGING_FEATURE] = genes.map(table[SCRNASEQ_AGING_FEATURE]).values
    if add_fdr:
        out[SCRNASEQ_AGING_FDR] = genes.map(table[SCRNASEQ_AGING_FDR]).values
    return out
