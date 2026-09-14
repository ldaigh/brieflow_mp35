"""casTLE per-feature gene effects: loaders and figure builders for page 14.

Backs ``pages/14_CasTLE_Effects.py``. Reads the outputs of the sibling
``castle_analysis`` project via ``$CASTLE_OUTPUT_PATH``; every loader returns an
empty frame or None when a file is missing, so the page can warn and stop rather
than raise.

Factored into ``src/`` rather than kept page-local, against the repo's
copy-paste-the-page convention, for one specific reason:
``display_sgrna_representativeness`` already exists in four verbatim copies
(``Cluster_Analysis.py``, ``6_Feature_Explorer.py``, ``7_Feature_Scatter.py``,
``12_External_Screen_Comparison.py``). A fifth casTLE-flavoured copy is where that
tips over. Gating follows the pattern of ``src/scrnaseq_aging.py`` and
``src/external_deg.py``: an optional env var, a ``has_*_data()`` predicate, and
loaders that fail soft.

The figure builders deliberately mirror the existing CellProfiler volcano
(``6_Feature_Explorer.py::build_volcano_plot``) and scatter
(``7_Feature_Scatter.py::build_feature_scatter``) down to the palettes and marker
sizes, so the two can be compared by eye across pages. They are separate objects,
not a mode switch on those pages: the axes, the feature menu and the significance
column all differ, and putting both behind one control would invite misreading.
"""

import math
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.labels import add_point_labels

CASTLE_OUTPUT_PATH = os.environ.get("CASTLE_OUTPUT_PATH", None)

GENE_COL = "gene_symbol_0"

# Signed quantities need a diverging scale centred at zero, or a zero-effect gene
# reads as "low" instead of "neutral". Matches how Cluster_Analysis.py treats the
# signed scRNA-seq aging coefficient.
SIGNED_COLORSCALE = "RdBu_r"
UNSIGNED_COLORSCALE = "Viridis"

# Colour metrics offered on the landscape, and whether each is signed.
COLOR_METRICS = {
    "casTLE effect (shrunk)": ("castle_effect_shrunk", True),
    "casTLE effect (MLE)": ("castle_effect", True),
    "casTLE score": ("castle_score", False),
    "-log10(FDR)": ("_neg_log10_fdr", False),
    "P(no effect)": ("castle_posterior_null", False),
}

# Volcano y-axis options. -log10(FDR) first so the plot reads like page 6's.
VOLCANO_Y = {
    "-log10(FDR)": ("_neg_log10_fdr", "-log10(FDR)"),
    "casTLE score": ("castle_score", "casTLE score"),
}

FDR_THRESHOLD = 0.05

# Page 6 / page 7 palettes, reused verbatim for cross-page comparability.
BACKGROUND_COLOR = "#aaaaaa"
SIGNIFICANT_COLOR = "#756bb1"
X_ONLY_COLOR = "#4575b4"
Y_ONLY_COLOR = "#d73027"
BOTH_COLOR = "#f1c40f"


def has_castle_data(cell_class: str = "all", channel_combo: str = "DAPI_YFP_RFP_Cy5") -> bool:
    """True when the long-form casTLE table is present and readable."""
    return CASTLE_OUTPUT_PATH is not None and os.path.exists(
        _path(f"CeCl-{cell_class}_ChCo-{channel_combo}__castle_gene_feature.parquet")
    )


def _path(name: str) -> str:
    return os.path.join(CASTLE_OUTPUT_PATH or "", name)


def _stem(cell_class: str, channel_combo: str) -> str:
    return f"CeCl-{cell_class}_ChCo-{channel_combo}"


# =====================
# LOADERS


@st.cache_data
def load_castle_long(cell_class: str, channel_combo: str) -> pd.DataFrame:
    """The long-form gene x feature table, with derived plotting columns.

    ``_neg_log10_fdr`` is added here rather than in the pipeline so the ceiling
    used for FDR = 0 stays a presentation choice.
    """
    path = _path(f"{_stem(cell_class, channel_combo)}__castle_gene_feature.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)

    with np.errstate(divide="ignore"):
        df["_neg_log10_fdr"] = -np.log10(df["castle_fdr"].to_numpy(dtype=float))
    # An FDR of exactly 0 cannot happen given the tail fit, but guard anyway so a
    # single infinity cannot blow up an axis range.
    finite_max = df.loc[np.isfinite(df["_neg_log10_fdr"]), "_neg_log10_fdr"]
    ceiling = float(finite_max.max()) if len(finite_max) else 10.0
    df["_neg_log10_fdr"] = df["_neg_log10_fdr"].replace([np.inf], ceiling)
    df["significant"] = df["castle_fdr"] < FDR_THRESHOLD
    return df


@st.cache_data
def load_castle_rhos(cell_class: str, channel_combo: str) -> pd.DataFrame:
    """Standardized per-guide rho values, for the guide-evidence plot."""
    path = _path(f"{_stem(cell_class, channel_combo)}__castle_rho_constructs.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_castle_background(cell_class: str, channel_combo: str) -> pd.DataFrame:
    """Quantile summary of the cell-count-matched background, per (feature, stratum)."""
    path = _path(f"{_stem(cell_class, channel_combo)}__castle_background.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_feature_ranking(cell_class: str, channel_combo: str) -> pd.DataFrame:
    """Per-feature discriminability statistics and the meaningful flag."""
    path = _path(f"{_stem(cell_class, channel_combo)}__castle_feature_ranking.tsv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


@st.cache_data
def load_embedding(cell_class: str, channel_combo: str, feature_set: str) -> pd.DataFrame:
    """The PHATE + Leiden embedding for one feature set."""
    path = _path(f"castle_gene_phate_leiden_{feature_set}.tsv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


@st.cache_data
def load_run_metadata(cell_class: str, channel_combo: str) -> pd.DataFrame:
    """Run settings, so the page can state which background produced the numbers."""
    path = _path(f"{_stem(cell_class, channel_combo)}__castle_run_metadata.tsv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def meaningful_features(ranking: pd.DataFrame, top_n: int | None = None) -> list[str]:
    """Ranked meaningful features, most discriminating first.

    Ranked by discriminability rather than sorted alphabetically like
    ``load_significant_features`` on pages 6/7: with ~1509 features an alphabetical
    menu buries the useful ones.
    """
    if ranking.empty or "is_meaningful" not in ranking.columns:
        return []
    rows = ranking.loc[ranking["is_meaningful"]].sort_values("rank")
    features = rows["feature"].tolist()
    return features[:top_n] if top_n else features


def significant_features(long: pd.DataFrame) -> list[str]:
    """Every feature with at least one gene below the FDR threshold."""
    if long.empty:
        return []
    real = long.loc[~long["is_control"]]
    return sorted(real.loc[real["significant"], "feature"].unique().tolist())


def all_scored_features(long: pd.DataFrame, ranking: pd.DataFrame) -> list[str]:
    """Every feature casTLE scored: curated representatives first, then the rest.

    The menus need to be able to reach *any* scored feature. A feature can drop out
    of the meaningful set for two quite different reasons -- too few significant
    genes (``cell_DAPI_int``), or being redundant with a sibling that scored higher
    in its |r| > 0.9 group -- and neither is a reason you should be unable to look
    at that specific feature for a specific gene.

    Ordering is deliberate: the meaningful representatives come first in
    discriminability rank, so the default selection of every dropdown and the
    cluster heatmap's top-40 are unchanged by widening the menu. The remainder
    follows by discriminability, which beats alphabetical when scanning and is no
    worse when typing to search.
    """
    if long.empty:
        return []
    scored = set(long["feature"].astype(str).unique().tolist())
    ordered = [f for f in meaningful_features(ranking) if f in scored]
    seen = set(ordered)

    if not ranking.empty and {"feature", "discriminability"} <= set(ranking.columns):
        rest = ranking.loc[
            ranking["feature"].isin(scored - seen)
        ].sort_values("discriminability", ascending=False)["feature"].tolist()
        ordered.extend(rest)
        seen.update(rest)

    # Anything scored but absent from the ranking table (or with no ranking table at
    # all) still has to appear, or the menu silently hides it.
    ordered.extend(sorted(scored - seen))
    return ordered


# =====================
# PER-GENE VIEWS


def top_features_for_gene(long: pd.DataFrame, gene: str, top_n: int = 20) -> pd.DataFrame:
    """A gene's features ranked by casTLE score, formatted for display."""
    rows = long.loc[long[GENE_COL] == gene].nlargest(top_n, "castle_score")
    if rows.empty:
        return rows
    out = rows[
        [
            "feature",
            "castle_effect",
            "effect_ci_low",
            "effect_ci_high",
            "castle_score",
            "castle_fdr",
            "hit_rate_map",
            "n_guides_used",
            "castle_pval_extrapolated",
        ]
    ].copy()
    out["effect ± 95% CI"] = [
        f"{e:+.2f}  [{lo:+.2f}, {hi:+.2f}]"
        for e, lo, hi in zip(
            out["castle_effect"], out["effect_ci_low"], out["effect_ci_high"]
        )
    ]
    return out


def build_guide_evidence(
    gene: str,
    feature: str,
    rhos: pd.DataFrame,
    background: pd.DataFrame,
    fit_row: pd.Series,
) -> go.Figure:
    """Each sgRNA's rho against the cell-count-matched control-median background.

    The direct upgrade of ``cluster_eval.plot_feature_vs_control``. Two deliberate
    differences from that plot:

    * the background is the distribution of **control guide medians** resampled at
      matched cell count, not the single-cell control distribution. The latter is
      far wider and is the wrong comparator for a guide median -- it makes every
      gene look unremarkable;
    * a fitted effect with its 95% credible interval is drawn, instead of leaving
      the reader to judge the guide spread by eye.

    Each guide's band is drawn from its own stratum, so a guide with few cells
    visibly sits against a wider background.
    """
    fig = go.Figure()

    gene_guides = rhos.loc[rhos[GENE_COL] == gene]
    bg_feature = background.loc[background["feature"] == feature] if not background.empty else pd.DataFrame()

    # Background: one shaded interquartile band per stratum actually used by this
    # gene's guides, widest first so narrower ones stay visible on top.
    used_strata = sorted(gene_guides["stratum"].unique(), reverse=True) if "stratum" in gene_guides else []
    for stratum in used_strata:
        row = bg_feature.loc[bg_feature["stratum"] == stratum]
        if row.empty:
            continue
        row = row.iloc[0]
        for lo_q, hi_q, alpha in (("q0.025", "q0.975", 0.10), ("q0.25", "q0.75", 0.18)):
            if lo_q not in row or hi_q not in row:
                continue
            fig.add_vrect(
                x0=float(row[lo_q]),
                x1=float(row[hi_q]),
                fillcolor="gray",
                opacity=alpha,
                line_width=0,
                layer="below",
            )
        fig.add_trace(
            go.Scatter(
                x=[float(row["q0.5"])],
                y=[0],
                mode="markers",
                marker=dict(color="gray", size=1),
                name=f"control n≈{int(row['stratum_size'])} cells (SD {row['mad_sd']:.2f})",
                hoverinfo="skip",
            )
        )

    # The fitted effect and its credible interval.
    if fit_row is not None and len(fit_row):
        lo, hi = float(fit_row["effect_ci_low"]), float(fit_row["effect_ci_high"])
        fig.add_vrect(
            x0=lo, x1=hi, fillcolor="#756bb1", opacity=0.18, line_width=0, layer="below"
        )
        fig.add_vline(
            x=float(fit_row["castle_effect"]),
            line_dash="dash",
            line_color="#756bb1",
            line_width=2.5,
            annotation_text=(
                f"casTLE effect {float(fit_row['castle_effect']):+.2f} "
                f"[{lo:+.2f}, {hi:+.2f}]"
            ),
            annotation_position="top right",
        )

    # One marker per sgRNA, staggered so overlapping guides stay distinguishable --
    # the same trick plot_feature_vs_control uses.
    n = len(gene_guides)
    for i, (_, guide) in enumerate(gene_guides.iterrows()):
        y = 0.95 - 0.7 * (i / max(n - 1, 1))
        fig.add_trace(
            go.Scatter(
                x=[float(guide[feature])],
                y=[y],
                mode="markers",
                marker=dict(size=11, symbol="line-ns", line=dict(width=2.5)),
                name=f"{guide['cell_barcode_0']} (n={int(guide['cell_count'])})",
                hovertemplate=(
                    f"<b>{guide['cell_barcode_0']}</b><br>"
                    f"rho=%{{x:.3f}}<br>"
                    f"cells={int(guide['cell_count'])}<br>"
                    f"stratum={int(guide['stratum'])}<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(
        title=f"{gene} — {feature}: sgRNA evidence vs matched control background",
        xaxis_title="rho (safe-guide SD units)",
        yaxis=dict(range=[0, 1.05], showticklabels=False, title=None),
        height=420,
        margin=dict(t=60, b=40, l=40, r=20),
        legend=dict(font=dict(size=10)),
    )
    return fig


# =====================
# VOLCANO AND SCATTER


def _axis_limits(values: pd.Series, pad: float = 0.05):
    """5%-padded range, matching pages 6 and 7 so label placement lines up."""
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return (-1.0, 1.0)
    span = (clean.max() - clean.min()) or 1.0
    return (clean.min() - pad * span, clean.max() + pad * span)


def build_castle_volcano(
    long: pd.DataFrame,
    feature: str,
    y_metric: str = "-log10(FDR)",
    x_column: str = "castle_effect",
    show_labels: bool = False,
    label_fdr: float = FDR_THRESHOLD,
    max_labels: int = 50,
    show_error_bars: bool = False,
    hide_controls: bool = True,
) -> go.Figure | None:
    """casTLE effect versus significance for one feature, across genes.

    The casTLE analogue of ``6_Feature_Explorer.py::build_volcano_plot``, with the
    same visual grammar. ``show_error_bars`` draws the 95% credible interval --
    something the bootstrap volcano has nothing to show.
    """
    y_column, y_label = VOLCANO_Y[y_metric]
    df = long.loc[long["feature"] == feature].copy()
    if hide_controls:
        df = df.loc[~df["is_control"]]
    df = df.dropna(subset=[x_column, y_column])
    if df.empty:
        return None

    background = df.loc[~df["significant"]]
    significant = df.loc[df["significant"]]

    def make_trace(subset, name, color, size):
        error_x = None
        if show_error_bars and not subset.empty:
            error_x = dict(
                type="data",
                symmetric=False,
                array=(subset["effect_ci_high"] - subset[x_column]).clip(lower=0),
                arrayminus=(subset[x_column] - subset["effect_ci_low"]).clip(lower=0),
                thickness=0.8,
                width=0,
                color="rgba(120,120,120,0.5)",
            )
        return go.Scatter(
            x=subset[x_column],
            y=subset[y_column],
            mode="markers",
            name=name,
            marker=dict(color=color, size=size, opacity=0.6),
            error_x=error_x,
            customdata=subset[[GENE_COL, "castle_score", "castle_fdr", "n_guides_used"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "effect=%{x:.3f}<br>"
                f"{y_label}=%{{y:.3f}}<br>"
                "score=%{customdata[1]:.1f}<br>"
                "FDR=%{customdata[2]:.2e}<br>"
                "guides=%{customdata[3]}<extra></extra>"
            ),
        )

    fig = go.Figure()
    fig.add_trace(make_trace(background, "Background", BACKGROUND_COLOR, 7))
    fig.add_trace(
        make_trace(significant, f"Significant (FDR<{FDR_THRESHOLD})", SIGNIFICANT_COLOR, 10)
    )

    xlim = _axis_limits(df[x_column])
    ylim = _axis_limits(df[y_column])

    if y_column == "_neg_log10_fdr":
        fig.add_hline(
            y=-math.log10(FDR_THRESHOLD),
            line_dash="dash",
            line_color="#888888",
            line_width=1,
        )
    fig.add_vline(x=0, line_dash="dash", line_color="#888888", line_width=1)

    if show_labels:
        labelled = df.loc[df["castle_fdr"] <= label_fdr]
        if len(labelled) > max_labels:
            labelled = labelled.nlargest(max_labels, y_column)
        if not labelled.empty:
            add_point_labels(
                fig,
                labelled[x_column].values,
                labelled[y_column].values,
                labelled[GENE_COL].astype(str).values,
                df[x_column].values,
                df[y_column].values,
                xlim,
                ylim,
            )

    fig.update_layout(
        title=f"casTLE volcano: {feature}",
        xaxis_title="casTLE effect (safe-guide SD units)",
        yaxis_title=y_label,
        xaxis=dict(range=list(xlim)),
        yaxis=dict(range=list(ylim)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=40, r=20),
        height=500,
    )
    return fig


def build_castle_scatter(
    long: pd.DataFrame,
    feature_x: str,
    feature_y: str,
    effect_column: str = "castle_effect",
    show_labels: bool = False,
    label_fdr: float = FDR_THRESHOLD,
    max_labels: int = 50,
    hide_controls: bool = True,
) -> go.Figure | None:
    """casTLE effect for one feature against another, across genes.

    The casTLE analogue of ``7_Feature_Scatter.py::build_feature_scatter``, reusing
    its four significance categories and exact palette so the two can be compared
    side by side.
    """
    wide = long.pivot_table(
        index=[GENE_COL, "is_control"],
        columns="feature",
        values=[effect_column, "castle_fdr"],
        aggfunc="first",
    )
    for feature in (feature_x, feature_y):
        if (effect_column, feature) not in wide.columns:
            return None

    df = pd.DataFrame(
        {
            "x": wide[(effect_column, feature_x)],
            "y": wide[(effect_column, feature_y)],
            "fdr_x": wide[("castle_fdr", feature_x)],
            "fdr_y": wide[("castle_fdr", feature_y)],
        }
    ).reset_index()

    if hide_controls:
        df = df.loc[~df["is_control"]]
    df = df.dropna(subset=["x", "y"])
    if df.empty:
        return None

    df["sig_x"] = df["fdr_x"] < FDR_THRESHOLD
    df["sig_y"] = df["fdr_y"] < FDR_THRESHOLD

    def make_trace(subset, name, color, size, opacity=0.6):
        return go.Scatter(
            x=subset["x"],
            y=subset["y"],
            mode="markers",
            name=name,
            marker=dict(color=color, size=size, opacity=opacity),
            customdata=subset[[GENE_COL, "fdr_x", "fdr_y"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"{feature_x}=%{{x:.3f}} (FDR %{{customdata[1]:.2e}})<br>"
                f"{feature_y}=%{{y:.3f}} (FDR %{{customdata[2]:.2e}})"
                "<extra></extra>"
            ),
        )

    fig = go.Figure()
    fig.add_trace(
        make_trace(
            df[~df["sig_x"] & ~df["sig_y"]], "Neither significant", BACKGROUND_COLOR, 7
        )
    )
    fig.add_trace(
        make_trace(df[df["sig_x"] & ~df["sig_y"]], "Sig in X only", X_ONLY_COLOR, 9)
    )
    fig.add_trace(
        make_trace(df[~df["sig_x"] & df["sig_y"]], "Sig in Y only", Y_ONLY_COLOR, 9)
    )
    fig.add_trace(
        make_trace(df[df["sig_x"] & df["sig_y"]], "Sig in both", BOTH_COLOR, 11, 0.85)
    )

    xlim = _axis_limits(df["x"])
    ylim = _axis_limits(df["y"])
    fig.add_hline(y=0, line_dash="dash", line_color="#888888", line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color="#888888", line_width=1)

    if show_labels:
        labelled = df.assign(_fdr_min=df[["fdr_x", "fdr_y"]].min(axis=1))
        labelled = labelled.loc[labelled["_fdr_min"] <= label_fdr]
        if len(labelled) > max_labels:
            labelled = labelled.nsmallest(max_labels, "_fdr_min")
        if not labelled.empty:
            add_point_labels(
                fig,
                labelled["x"].values,
                labelled["y"].values,
                labelled[GENE_COL].astype(str).values,
                df["x"].values,
                df["y"].values,
                xlim,
                ylim,
            )

    fig.update_layout(
        title=f"casTLE effect: {feature_x}  vs  {feature_y}",
        xaxis_title=feature_x,
        yaxis_title=feature_y,
        xaxis=dict(range=list(xlim)),
        yaxis=dict(range=list(ylim)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40, l=40, r=20),
        height=600,
    )
    return fig


def effect_lookup(long: pd.DataFrame, feature: str, column: str) -> pd.Series:
    """``{gene: value}`` for one feature, for colouring the embedding."""
    rows = long.loc[long["feature"] == feature]
    return rows.set_index(GENE_COL)[column]
