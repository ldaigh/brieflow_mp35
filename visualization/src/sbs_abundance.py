"""Gene-level abundance QC for the SBS screen, read from the per-well cell tables.

Two figures, both served from ``sbs/parquets/*__cells.parquet``:

* the pipeline's gene-symbol count histogram, re-rendered live so it can be
  exported as a vector (the pipeline only writes a PNG under ``sbs/eval/``);
* a scatter of per-gene abundance against an external gDNA count library.

The external file (``GDNA_COUNTS_PATH``, e.g.
``LHD_MCB716_MP31_gDNA_counts.csv``) is a headerless two-column CSV of
``<guide name>,<read count>`` from amplicon sequencing of extracted gDNA. Guide
names are ``<ensembl id>_<symbol>_<sublibrary>_<oligo id>``, e.g.
``ENSMUSG00000026878_Rab14_SPA_50369.1``; safe-harbour controls are
``0Safe_safe_<sublibrary>_<oligo id>``. Counts are summed over every guide of a
gene, so both axes are gene-level.

The current-screen side is the number of mapped cells per gene, read from the
SBS ``*__cells.parquet`` tables (``gene_id_0``/``gene_symbol_0``, the same
column the pipeline's gene-symbol histogram uses). Cells with no mapped barcode
are dropped.

Genes are matched on ensembl id (``gene_id_0`` vs the guide name's first field),
which is exact for this pair of libraries -- all 398 targeted genes plus the
pooled ``0Safe`` set are present on both sides, and the symbols agree. Controls
pool into a single point (441 external guides, tens of thousands of cells) that
would dominate any fit, so they are excluded from the fit and hidden by default.
"""

import glob
import io
import logging
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from src.config import BRIEFLOW_OUTPUT_PATH, GDNA_COUNTS_PATH, load_config
from src.labels import HAS_ADJUST_TEXT, add_point_labels

if HAS_ADJUST_TEXT:
    from adjustText import adjust_text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from workflow.lib.sbs.eval_mapping import (  # noqa: E402
    plot_gene_symbol_histogram,
)

logger = logging.getLogger(__name__)

# Vector formats keep text as text (editable in Illustrator/Inkscape) rather
# than outlines, which is what makes these exports usable in figures.
_VECTOR_RC = {"svg.fonttype": "none", "pdf.fonttype": 42}

# `P-1_W-A1__cells.parquet` -> plate 1, well A1.
_CELLS_RE = re.compile(r"P-(?P<plate>[^_]+)_W-(?P<well>[^_]+)__cells\.parquet$")

DEFAULT_CONTROL_KEY = "0Safe"

NORMALIZE_MODES = ["Raw counts", "Counts per million"]

_EXTERNAL_COLS = [
    "guide",
    "external_count",
    "gene_id",
    "gene_symbol",
    "sublibrary",
]


def has_gdna_data() -> bool:
    """True when the external gDNA count file is configured and readable."""
    return bool(GDNA_COUNTS_PATH) and os.path.exists(GDNA_COUNTS_PATH)


@st.cache_data
def control_key() -> str:
    """Gene-symbol/id prefix marking control perturbations (from the pipeline config)."""
    try:
        config = load_config()
    except Exception as exc:  # config is optional for this figure
        logger.warning(f"could not read config for control_key: {exc}")
        return DEFAULT_CONTROL_KEY
    key = (config.get("aggregate") or {}).get("control_key")
    return key or DEFAULT_CONTROL_KEY


@st.cache_data
def load_external_guide_counts(path: str | None = GDNA_COUNTS_PATH) -> pd.DataFrame:
    """Per-guide external gDNA counts with the guide name parsed into fields."""
    if not path or not os.path.exists(path):
        logger.warning(f"external gDNA counts not found at {path}")
        return pd.DataFrame(columns=_EXTERNAL_COLS)

    df = pd.read_csv(
        path, header=None, names=["guide", "external_count"], dtype={"guide": str}
    )
    # The file is headerless, but tolerate one if it ever gains a header row.
    if pd.to_numeric(df["external_count"].head(1), errors="coerce").isna().all():
        df = df.iloc[1:]
    df["external_count"] = pd.to_numeric(df["external_count"], errors="coerce")
    df = df.dropna(subset=["guide", "external_count"])

    parts = df["guide"].str.split("_")
    df["gene_id"] = parts.str[0]
    df["gene_symbol"] = parts.str[1]
    df["sublibrary"] = parts.str[2]
    return df[_EXTERNAL_COLS].reset_index(drop=True)


@st.cache_data
def external_gene_counts(path: str | None = GDNA_COUNTS_PATH) -> pd.DataFrame:
    """External counts summed over the guides of each gene."""
    guides = load_external_guide_counts(path)
    if guides.empty:
        return pd.DataFrame(
            columns=["gene_id", "external_count", "external_guides", "external_symbol"]
        )

    return guides.groupby("gene_id", as_index=False).agg(
        external_count=("external_count", "sum"),
        external_guides=("guide", "size"),
        external_symbol=("gene_symbol", "first"),
    )


@st.cache_data
def list_sbs_cell_tables(root: str = BRIEFLOW_OUTPUT_PATH) -> pd.DataFrame:
    """The SBS per-well cell tables available under the brieflow output root."""
    paths = sorted(glob.glob(os.path.join(root, "sbs", "parquets", "*__cells.parquet")))
    records = []
    for path in paths:
        match = _CELLS_RE.search(os.path.basename(path))
        records.append(
            {
                "path": path,
                "plate": match.group("plate") if match else "?",
                "well": match.group("well") if match else "?",
            }
        )
    return pd.DataFrame(records, columns=["path", "plate", "well"])


@st.cache_data
def screen_gene_counts(
    plates: tuple[str, ...] | None = None,
    wells: tuple[str, ...] | None = None,
    root: str = BRIEFLOW_OUTPUT_PATH,
) -> pd.DataFrame:
    """Mapped cells per gene in the current SBS screen, for the selected wells."""
    tables = list_sbs_cell_tables(root)
    if plates:
        tables = tables[tables["plate"].isin(plates)]
    if wells:
        tables = tables[tables["well"].isin(wells)]
    if tables.empty:
        return pd.DataFrame(columns=["gene_id", "screen_count", "screen_symbol"])

    cells = pd.concat(
        [
            pd.read_parquet(path, columns=["gene_symbol_0", "gene_id_0"])
            for path in tables["path"]
        ],
        ignore_index=True,
    )
    cells = cells.dropna(subset=["gene_id_0"])
    counts = cells.groupby("gene_id_0", as_index=False).agg(
        screen_count=("gene_id_0", "size"),
        screen_symbol=("gene_symbol_0", "first"),
    )
    return counts.rename(columns={"gene_id_0": "gene_id"})


@st.cache_data
def load_sbs_gene_symbols(
    plates: tuple[str, ...] | None = None,
    wells: tuple[str, ...] | None = None,
    root: str = BRIEFLOW_OUTPUT_PATH,
) -> pd.DataFrame:
    """Mapped gene symbols (one row per cell) for the selected wells."""
    tables = list_sbs_cell_tables(root)
    if plates:
        tables = tables[tables["plate"].isin(plates)]
    if wells:
        tables = tables[tables["well"].isin(wells)]
    if tables.empty:
        return pd.DataFrame(columns=["gene_symbol_0"])

    return pd.concat(
        [pd.read_parquet(path, columns=["gene_symbol_0"]) for path in tables["path"]],
        ignore_index=True,
    ).dropna(subset=["gene_symbol_0"])


def build_gene_symbol_histogram(
    cells: pd.DataFrame,
    exclude_controls: bool = True,
    x_cutoff: float | None = None,
):
    """Re-render the pipeline's gene-symbol count histogram as a live figure.

    Calls the same `lib.sbs.eval_mapping.plot_gene_symbol_histogram` the pipeline
    uses for `sbs/eval/mapping/P-*__gene_symbol_histogram*.png`, so the figure
    matches the stored PNG; rendering it here is what makes a vector export
    possible.

    Returns:
        (outliers Series above the x cutoff, matplotlib Figure).
    """
    prefix = control_key() if exclude_controls else None
    with plt.rc_context(_VECTOR_RC):
        return plot_gene_symbol_histogram(
            cells, x_cutoff=x_cutoff, control_prefix=prefix
        )


def figure_to_vector(fig, fmt: str = "svg"):
    """Serialize a matplotlib figure to vector SVG text or PDF bytes."""
    with plt.rc_context(_VECTOR_RC):
        buf = io.StringIO() if fmt == "svg" else io.BytesIO()
        fig.savefig(buf, format=fmt, bbox_inches="tight")
        return buf.getvalue()


@st.cache_data
def match_gene_counts(
    plates: tuple[str, ...] | None = None,
    wells: tuple[str, ...] | None = None,
    path: str | None = GDNA_COUNTS_PATH,
    root: str = BRIEFLOW_OUTPUT_PATH,
) -> tuple[pd.DataFrame, dict]:
    """Join gene-level external gDNA counts to current-screen mapped-cell counts.

    Returns:
        (matched, info) where `matched` has one row per gene present in both
        libraries (columns gene_id, gene_symbol, screen_count, external_count,
        external_guides, is_control) and `info` carries library totals and the
        genes that failed to match.
    """
    screen = screen_gene_counts(plates, wells, root)
    external = external_gene_counts(path)
    prefix = control_key()

    info = {
        "screen_total": int(screen["screen_count"].sum()) if not screen.empty else 0,
        "external_total": (
            int(external["external_count"].sum()) if not external.empty else 0
        ),
        "screen_genes": 0,
        "external_genes": 0,
        "missing_in_external": [],
        "missing_in_screen": [],
    }
    if screen.empty or external.empty:
        return pd.DataFrame(), info

    merged = screen.merge(external, on="gene_id", how="outer", indicator=True)
    is_control = merged["gene_id"].astype(str).str.startswith(prefix)

    info["screen_genes"] = int((~is_control & merged["screen_count"].notna()).sum())
    info["external_genes"] = int((~is_control & merged["external_count"].notna()).sum())
    info["missing_in_external"] = sorted(
        merged.loc[
            (merged["_merge"] == "left_only") & ~is_control, "screen_symbol"
        ].astype(str)
    )
    info["missing_in_screen"] = sorted(
        merged.loc[
            (merged["_merge"] == "right_only") & ~is_control, "external_symbol"
        ].astype(str)
    )

    matched = merged[merged["_merge"] == "both"].copy()
    matched["is_control"] = matched["gene_id"].astype(str).str.startswith(prefix)
    # Control rows carry a per-barcode symbol (`0Safe_AAATGTTAG`) on the screen
    # side; the pooled point is clearer labelled by the control key itself.
    matched["gene_symbol"] = np.where(
        matched["is_control"],
        prefix,
        matched["screen_symbol"].fillna(matched["external_symbol"]),
    )
    matched["external_count"] = matched["external_count"].astype(float)
    matched["screen_count"] = matched["screen_count"].astype(float)

    cols = [
        "gene_id",
        "gene_symbol",
        "screen_count",
        "external_count",
        "external_guides",
        "is_control",
    ]
    matched = matched[cols].sort_values("gene_symbol").reset_index(drop=True)
    return matched, info


def prepare_axes(
    matched: pd.DataFrame, info: dict, normalize: str = "Raw counts"
) -> tuple[pd.DataFrame, str, str]:
    """Add plotting columns `_x`/`_y` and return them with their axis titles.

    Normalisation is against each library's full total (controls included), so
    the values are a genuine share of the sequenced/imaged library.
    """
    df = matched.copy()
    if normalize == "Counts per million":
        ext_total = info.get("external_total") or 1
        screen_total = info.get("screen_total") or 1
        df["_x"] = df["external_count"] / ext_total * 1e6
        df["_y"] = df["screen_count"] / screen_total * 1e6
        x_title = "External gDNA counts per million (summed over guides)"
        y_title = "Current screen mapped cells per million"
    else:
        df["_x"] = df["external_count"]
        df["_y"] = df["screen_count"]
        x_title = "External gDNA read count (summed over guides)"
        y_title = "Current screen mapped cells"
    return df, x_title, y_title


def fit_stats(x, y, log_space: bool = False) -> dict:
    """OLS fit plus Pearson/Spearman statistics over the pairwise-complete values.

    With `log_space=True` the fit is on log10 of both axes (a power law,
    y = 10**intercept * x**slope); non-positive values are dropped.
    """
    x = pd.Series(x, dtype=float).reset_index(drop=True)
    y = pd.Series(y, dtype=float).reset_index(drop=True)
    keep = x.notna() & y.notna()
    if log_space:
        keep &= (x > 0) & (y > 0)
    x, y = x[keep], y[keep]

    out = {
        "n": len(x),
        "log_space": log_space,
        "slope": np.nan,
        "intercept": np.nan,
        "r": np.nan,
        "r_p": np.nan,
        "r2": np.nan,
        "rho": np.nan,
        "rho_p": np.nan,
    }
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return out

    fx, fy = (np.log10(x), np.log10(y)) if log_space else (x, y)
    slope, intercept = np.polyfit(fx, fy, 1)
    r, r_p = stats.pearsonr(fx, fy)
    rho, rho_p = stats.spearmanr(x, y)
    out.update(
        slope=float(slope),
        intercept=float(intercept),
        r=float(r),
        r_p=float(r_p),
        r2=float(r**2),
        rho=float(rho),
        rho_p=float(rho_p),
    )
    return out


def _fit_line(fit: dict, x_lo: float, x_hi: float):
    """Points tracing the fitted line across [x_lo, x_hi]."""
    if not np.isfinite(fit.get("slope", np.nan)):
        return None, None
    if fit["log_space"]:
        if x_lo <= 0:
            x_lo = x_hi / 1e4 if x_hi > 0 else 1.0
        line_x = np.logspace(np.log10(x_lo), np.log10(x_hi), 100)
        line_y = 10 ** (fit["intercept"] + fit["slope"] * np.log10(line_x))
    else:
        line_x = np.linspace(x_lo, x_hi, 100)
        line_y = fit["intercept"] + fit["slope"] * line_x
    return line_x, line_y


def _fit_label(fit: dict) -> str:
    if not np.isfinite(fit.get("slope", np.nan)):
        return "OLS fit"
    if fit["log_space"]:
        return f"OLS fit (log-log slope {fit['slope']:.3g}, R² = {fit['r2']:.3f})"
    return f"OLS fit (slope {fit['slope']:.3g}, R² = {fit['r2']:.3f})"


def _limits(values: np.ndarray, log_scale: bool, pad: float = 0.05):
    """Axis limits with a 5% pad, in log10 units when the axis is logarithmic."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if log_scale:
        values = values[values > 0]
    if values.size == 0:
        return (0.0, 1.0)
    lo, hi = float(values.min()), float(values.max())
    if log_scale:
        lo, hi = np.log10(lo), np.log10(hi)
    span = (hi - lo) or (abs(hi) or 1.0)
    return (lo - pad * span, hi + pad * span)


def outlier_genes(df: pd.DataFrame, fit: dict, max_labels: int = 25) -> pd.DataFrame:
    """Genes furthest from the fitted line (residual magnitude in fit space)."""
    if df.empty or not np.isfinite(fit.get("slope", np.nan)):
        return df.head(0)
    x, y = df["_x"].astype(float), df["_y"].astype(float)
    if fit["log_space"]:
        ok = (x > 0) & (y > 0)
        x, y = np.log10(x.where(ok)), np.log10(y.where(ok))
    residual = (y - (fit["intercept"] + fit["slope"] * x)).abs()
    return df.assign(_residual=residual).nlargest(
        min(max_labels, int(residual.notna().sum())), "_residual"
    )


def build_gdna_scatter(
    df: pd.DataFrame,
    info: dict,
    x_title: str,
    y_title: str,
    fit: dict,
    log_scale: bool = False,
    show_fit: bool = True,
    show_identity: bool = False,
    show_labels: bool = False,
    max_labels: int = 25,
) -> go.Figure:
    """Interactive gene-level abundance scatter (external gDNA vs current screen)."""
    genes = df[~df["is_control"]]
    controls = df[df["is_control"]]

    hover_cols = [
        "gene_symbol",
        "gene_id",
        "_x",
        "_y",
        "external_guides",
    ]
    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        "%{customdata[1]}<br>"
        "external: %{customdata[2]:,.4g}<br>"
        "current screen: %{customdata[3]:,.4g}<br>"
        "external guides: %{customdata[4]}<extra></extra>"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=genes["_x"],
            y=genes["_y"],
            mode="markers",
            name=f"Genes (n = {len(genes)})",
            marker=dict(color="#4575b4", size=7, opacity=0.75, line=dict(width=0)),
            customdata=genes[hover_cols].values,
            hovertemplate=hovertemplate,
        )
    )
    if not controls.empty:
        fig.add_trace(
            go.Scatter(
                x=controls["_x"],
                y=controls["_y"],
                mode="markers",
                name="Pooled controls (not in fit)",
                marker=dict(
                    color="#d62728",
                    size=13,
                    symbol="diamond",
                    line=dict(width=0.5, color="#5c1213"),
                ),
                customdata=controls[hover_cols].values,
                hovertemplate=hovertemplate,
            )
        )

    xlim = _limits(genes["_x"].values, log_scale)
    ylim = _limits(genes["_y"].values, log_scale)
    x_lo, x_hi = (10 ** xlim[0], 10 ** xlim[1]) if log_scale else xlim

    if show_fit:
        line_x, line_y = _fit_line(fit, x_lo, x_hi)
        if line_x is not None:
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    name=_fit_label(fit),
                    line=dict(color="#222222", width=1.5, dash="dash"),
                    hoverinfo="skip",
                )
            )

    if show_identity:
        lo = max(min(x_lo, ylim[0] if not log_scale else 10 ** ylim[0]), 1e-12)
        hi = max(x_hi, ylim[1] if not log_scale else 10 ** ylim[1])
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="y = x",
                line=dict(color="#999999", width=1, dash="dot"),
                hoverinfo="skip",
            )
        )

    if show_labels and not genes.empty:
        labelled = outlier_genes(genes, fit, max_labels)
        if not labelled.empty:
            to_plot = np.log10 if log_scale else (lambda v: v)
            add_point_labels(
                fig,
                to_plot(labelled["_x"].values),
                to_plot(labelled["_y"].values),
                labelled["gene_symbol"].astype(str).values,
                to_plot(genes["_x"].values),
                to_plot(genes["_y"].values),
                xlim,
                ylim,
            )

    fig.update_layout(
        title="Gene abundance: current SBS screen vs external gDNA counts",
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis=dict(
            type="log" if log_scale else "linear",
            range=list(xlim),
        ),
        yaxis=dict(
            type="log" if log_scale else "linear",
            range=list(ylim),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=90, b=50, l=60, r=20),
        height=620,
    )
    return fig


def build_gdna_scatter_mpl(
    df: pd.DataFrame,
    x_title: str,
    y_title: str,
    fit: dict,
    log_scale: bool = False,
    show_fit: bool = True,
    show_identity: bool = False,
    show_labels: bool = False,
    max_labels: int = 25,
):
    """Matplotlib twin of `build_gdna_scatter`, for true vector (SVG/PDF) export.

    Plotly's server-side image export needs kaleido, which is not installed in
    the visualization env, so publication figures are rendered here instead. The
    browser modebar's "download as svg" still works client-side.
    """
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Nimbus Sans",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            **_VECTOR_RC,
        }
    ):
        genes = df[~df["is_control"]]
        controls = df[df["is_control"]]

        fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=150)
        ax.scatter(
            genes["_x"],
            genes["_y"],
            s=18,
            c="#4575b4",
            alpha=0.75,
            linewidths=0,
            label=f"Genes (n = {len(genes)})",
        )
        if not controls.empty:
            ax.scatter(
                controls["_x"],
                controls["_y"],
                s=70,
                c="#d62728",
                marker="D",
                edgecolors="#5c1213",
                linewidths=0.5,
                label="Pooled controls (not in fit)",
            )

        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")

        x_lo, x_hi = ax.get_xlim()
        if show_fit:
            line_x, line_y = _fit_line(fit, x_lo, x_hi)
            if line_x is not None:
                ax.plot(
                    line_x,
                    line_y,
                    color="#222222",
                    lw=1.5,
                    ls="--",
                    label=_fit_label(fit),
                )
        if show_identity:
            y_lo, y_hi = ax.get_ylim()
            lo, hi = max(min(x_lo, y_lo), 1e-12), max(x_hi, y_hi)
            ax.plot([lo, hi], [lo, hi], color="#999999", lw=1, ls=":", label="y = x")

        if show_labels and not genes.empty:
            labelled = outlier_genes(genes, fit, max_labels)
            texts = [
                ax.text(
                    row["_x"],
                    row["_y"],
                    str(row["gene_symbol"]),
                    fontsize=7,
                    color="#222222",
                )
                for _, row in labelled.iterrows()
            ]
            if texts and HAS_ADJUST_TEXT:
                adjust_text(
                    texts,
                    ax=ax,
                    time_lim=3,
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5),
                )

        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        ax.set_title("Gene abundance: current SBS screen vs external gDNA counts")
        if np.isfinite(fit.get("r", np.nan)):
            ax.text(
                0.02,
                0.98,
                f"n = {fit['n']}\nPearson r = {fit['r']:.3f}\n"
                f"Spearman ρ = {fit['rho']:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
    return fig
