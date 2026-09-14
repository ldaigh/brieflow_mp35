"""Compare prior casTLE CRISPR screens against the current OPS screen.

The external axis holds a casTLE metric from one of the prior screens (ADCP
combo, or the genome-wide bead screen); the current axis holds any gene-level
parameter from this screen. Only genes present in both are plotted.

Direction of correlation is feature-dependent: a negative casTLE effect means
knockout depletes the gene from the sorted-high bin, so whether a positive or
negative rho is the "agreeing" one depends on which current feature is chosen.
No sign flipping is applied anywhere on this page.
"""

import glob
import io
import os
import sys
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(
    page_title="External Screen Comparison - Brieflow Analysis", layout="wide"
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import BRIEFLOW_OUTPUT_PATH, STATIC_ASSET_PATH, STATIC_ASSET_URL_ROOT
from src.config import load_config
from src.filesystem import FileSystem
from src.labels import add_point_labels
from src.rendering import render_composite_montage
from src.external_screens import (
    UNSIGNED_METRICS,
    available_screens,
    load_screen,
    screen_fdr_col,
    screen_label,
    screen_metrics,
)
from workflow.lib.cluster.cluster_eval import (
    get_significant_features_for_gene,
    load_control_feature_values,
    plot_feature_vs_control,
    rank_constructs_for_gene,
)

CELL_CLASS = "all"
CHANNEL_COMBO = "DAPI_YFP_RFP_Cy5"
CLUSTER_ROOT = os.path.join(BRIEFLOW_OUTPUT_PATH, "cluster")

# Gene-level summary metrics from the PHATE/Leiden table, offered alongside the
# aggregate feature columns because they summarise overall phenotype strength.
CLUSTER_METRICS = [
    "perturbation_auc",
    "mean_potential_to_nontargeting",
    "normalized_potential_to_nontargeting",
]

COLOR_NONE = "None"
COLOR_CURRENT_FDR = "Current-feature significance"
COLOR_EXTERNAL_FDR = "Old-screen significance"
COLOR_QUADRANT = "Quadrant agreement"
COLOR_CLUSTER = "Leiden cluster"

# =====================
# DATA LOADERS


@st.cache_data
def load_bootstrap_results(cell_class, channel_combo):
    path = os.path.join(
        BRIEFLOW_OUTPUT_PATH,
        "aggregate",
        "bootstrap",
        f"CeCl-{cell_class}_ChCo-{channel_combo}__all_gene_bootstrap_results.tsv",
    )
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep="\t")


@st.cache_data
def load_gene_table(cell_class, channel_combo):
    path = os.path.join(
        BRIEFLOW_OUTPUT_PATH,
        "aggregate",
        "tsvs",
        f"CeCl-{cell_class}_ChCo-{channel_combo}__features_genes.tsv",
    )
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep="\t")


@st.cache_data
def load_construct_table(cell_class, channel_combo):
    path = os.path.join(
        BRIEFLOW_OUTPUT_PATH,
        "aggregate",
        "tsvs",
        f"CeCl-{cell_class}_ChCo-{channel_combo}__features_constructs.tsv",
    )
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep="\t")


@st.cache_data
def load_cached_control_feature_values(
    cell_class, channel_combo, feature, perturbation_name_col, control_key
):
    parquet_path = os.path.join(
        BRIEFLOW_OUTPUT_PATH,
        "aggregate",
        "parquets",
        f"CeCl-{cell_class}_ChCo-{channel_combo}__features_singlecell.parquet",
    )
    if not os.path.exists(parquet_path):
        return None
    return load_control_feature_values(
        parquet_path, feature, perturbation_name_col, control_key
    )


@st.cache_data
def load_montage_data(root_dir, gene_name):
    files = FileSystem.find_files(
        root_dir + "/" + gene_name, include_all=["montages"], extensions=["png"]
    )
    filtered_df = FileSystem.extract_features(root_dir, files)
    filtered_df["gene"] = filtered_df["file_path"].apply(lambda x: x.split("/")[-3])
    filtered_df["guide"] = filtered_df["file_path"].apply(lambda x: x.split("/")[-2])
    filtered_df["channel"] = filtered_df["file_path"].apply(
        lambda x: x.split("/")[-1].split("__")[0]
    )
    return filtered_df


@st.cache_data
def load_cluster_table(leiden_resolution):
    """PHATE/Leiden table for the given resolution, falling back to any available."""
    tsv_path = os.path.join(
        CLUSTER_ROOT, CHANNEL_COMBO, CELL_CLASS, str(leiden_resolution),
        "phate_leiden_clustering.tsv",
    )
    if not os.path.exists(tsv_path):
        pattern = os.path.join(
            CLUSTER_ROOT, CHANNEL_COMBO, CELL_CLASS, "*", "phate_leiden_clustering.tsv"
        )
        matches = sorted(glob.glob(pattern))
        if not matches:
            return None
        tsv_path = matches[0]
    return pd.read_csv(tsv_path, sep="\t")


def load_uniprot_for_gene(gene, leiden_resolution):
    df = load_cluster_table(leiden_resolution)
    if df is None or "gene_symbol_0" not in df.columns:
        return None
    rows = df[df["gene_symbol_0"] == gene]
    return rows.iloc[0] if not rows.empty else None


@st.cache_data
def list_current_features(cell_class, channel_combo, bootstrap_only=True):
    """Current-screen parameters offered on the Y axis."""
    gene_df = load_gene_table(cell_class, channel_combo)
    if gene_df is None:
        return []
    numeric = gene_df.select_dtypes(include="number").columns.tolist()
    if bootstrap_only:
        bootstrap_df = load_bootstrap_results(cell_class, channel_combo)
        if bootstrap_df is not None:
            fdr_backed = {c[:-4] for c in bootstrap_df.columns if c.endswith("_fdr")}
            numeric = [c for c in numeric if c in fdr_backed]
    cluster_df = load_cluster_table(st.session_state.get("leiden_resolution", "15"))
    extras = []
    if cluster_df is not None:
        extras = [
            c for c in CLUSTER_METRICS if c in cluster_df.columns and c not in numeric
        ]
    return sorted(numeric) + extras


# =====================
# STATISTICS


def benjamini_hochberg(pvals):
    """BH-adjusted p-values; NaN inputs stay NaN and are excluded from the count."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    n = ok.sum()
    if n == 0:
        return out
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    restored = np.empty(n)
    restored[order] = adj
    out[ok] = restored
    return out


def correlation_stats(x, y):
    """Spearman and Pearson statistics over the pairwise-complete values."""
    x = pd.Series(x, dtype=float).reset_index(drop=True)
    y = pd.Series(y, dtype=float).reset_index(drop=True)
    keep = x.notna() & y.notna()
    x, y = x[keep], y[keep]
    n = len(x)
    empty = {"n": n, "rho": np.nan, "rho_p": np.nan, "r": np.nan, "r_p": np.nan}
    if n < 3 or x.nunique() < 2 or y.nunique() < 2:
        return empty
    rho, rho_p = stats.spearmanr(x, y)
    r, r_p = stats.pearsonr(x, y)
    return {"n": n, "rho": rho, "rho_p": rho_p, "r": r, "r_p": r_p}


@st.cache_data
def rank_features_by_correlation(
    screen_key, metric, cell_class, channel_combo, bootstrap_only, min_score
):
    """Spearman correlation of every current feature against the external metric."""
    table = load_screen(screen_key)
    gene_df = load_gene_table(cell_class, channel_combo)
    if table.empty or gene_df is None:
        return pd.DataFrame()

    keys = gene_df["gene_symbol_0"].astype(str).str.upper()
    ext = keys.map(table[metric])
    if min_score and "Combo casTLE Score" in table.columns:
        score = keys.map(table["Combo casTLE Score"])
        ext = ext.where(score >= min_score)
    matched = ext.notna()
    if matched.sum() < 3:
        return pd.DataFrame()

    ext = ext[matched]
    features = list_current_features(cell_class, channel_combo, bootstrap_only)
    cluster_df = load_cluster_table(st.session_state.get("leiden_resolution", "15"))

    records = []
    for feature in features:
        if feature in gene_df.columns:
            values = gene_df.loc[matched, feature]
        elif cluster_df is not None and feature in cluster_df.columns:
            lookup = cluster_df.set_index(
                cluster_df["gene_symbol_0"].astype(str).str.upper()
            )[feature]
            lookup = lookup[~lookup.index.duplicated()]
            values = keys[matched].map(lookup)
        else:
            continue
        s = correlation_stats(ext.values, values.values)
        if not np.isfinite(s["rho"]):
            continue
        records.append(
            {
                "feature": feature,
                "spearman_rho": s["rho"],
                "abs_rho": abs(s["rho"]),
                "spearman_p": s["rho_p"],
                "pearson_r": s["r"],
                "n_genes": s["n"],
            }
        )

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["spearman_fdr"] = benjamini_hochberg(out["spearman_p"].values)
    return out.sort_values("abs_rho", ascending=False).reset_index(drop=True)


# =====================
# MATCHED TABLE


def build_matched_df(screen_key, metric, feature):
    """Genes present in both screens, with the external metric and current feature."""
    gene_df = load_gene_table(CELL_CLASS, CHANNEL_COMBO)
    table = load_screen(screen_key)
    if gene_df is None or table.empty:
        return None, None

    cluster_df = load_cluster_table(st.session_state.get("leiden_resolution", "15"))
    cluster_lookup = None
    if cluster_df is not None and "gene_symbol_0" in cluster_df.columns:
        cluster_lookup = cluster_df.set_index(
            cluster_df["gene_symbol_0"].astype(str).str.upper()
        )
        cluster_lookup = cluster_lookup[~cluster_lookup.index.duplicated()]

    if feature in gene_df.columns:
        base = gene_df[["gene_symbol_0", feature]].copy()
    elif cluster_lookup is not None and feature in cluster_lookup.columns:
        base = gene_df[["gene_symbol_0"]].copy()
        base[feature] = (
            base["gene_symbol_0"].astype(str).str.upper().map(cluster_lookup[feature])
        )
    else:
        return None, None

    keys = base["gene_symbol_0"].astype(str).str.upper()
    base["_gene_key"] = keys.values

    # Current-screen significance, when the feature has a bootstrap FDR column.
    bootstrap_df = load_bootstrap_results(CELL_CLASS, CHANNEL_COMBO)
    fdr_col = f"{feature}_fdr"
    if bootstrap_df is not None and fdr_col in bootstrap_df.columns:
        fdr_lookup = bootstrap_df.set_index(
            bootstrap_df["gene"].astype(str).str.upper()
        )[fdr_col]
        fdr_lookup = fdr_lookup[~fdr_lookup.index.duplicated()]
        base["current_fdr"] = keys.map(fdr_lookup).values
    else:
        base["current_fdr"] = np.nan

    # Leiden cluster for the colour-by option.
    if cluster_lookup is not None and "cluster" in cluster_lookup.columns:
        base["cluster"] = keys.map(cluster_lookup["cluster"]).values
    else:
        base["cluster"] = np.nan

    for col in table.columns:
        base[col] = keys.map(table[col]).values

    unmatched = base.loc[base[metric].isna(), "gene_symbol_0"]
    matched = base.dropna(subset=[metric, feature]).reset_index(drop=True)
    return matched, unmatched


def external_significance_labels(df, screen_key):
    """Discrete significance tier per gene for the external screen, or None.

    Prefers the workbook's own boolean 1%/5%/10% FDR flags; falls back to
    thresholding the screen's adjusted-p column. Returns None for screens with
    no significance data at all (ADCP, whose casTLE p-value is N/A throughout).
    """
    flags = ["1% FDR", "5% FDR", "10% FDR"]
    tier = pd.Series("Not significant", index=df.index)
    if all(f in df.columns for f in flags):
        tier[df["10% FDR"].fillna(0) > 0] = "FDR < 10%"
        tier[df["5% FDR"].fillna(0) > 0] = "FDR < 5%"
        tier[df["1% FDR"].fillna(0) > 0] = "FDR < 1%"
        return tier

    fdr_col = screen_fdr_col(screen_key)
    if fdr_col is None or fdr_col not in df.columns:
        return None
    adjusted = df[fdr_col]
    tier[adjusted < 0.10] = "FDR < 10%"
    tier[adjusted < 0.05] = "FDR < 5%"
    tier[adjusted < 0.01] = "FDR < 1%"
    return tier


# =====================
# PLOT BUILDER

GROUP_COLORS = {
    "FDR < 1%": "#a50f15",
    "FDR < 5%": "#ef6548",
    "FDR < 10%": "#fdbb84",
    "Not significant": "#bbbbbb",
    "Significant (FDR < 0.05)": "#756bb1",
    "Not significant (current)": "#bbbbbb",
    "No FDR available": "#7fb3d5",
    "Concordant (same sign)": "#2c7fb8",
    "Discordant (opposite sign)": "#d95f02",
    "At zero": "#cccccc",
    "All matched genes": "#4575b4",
}


def assign_color_groups(df, mode, screen_key, metric, feature):
    """Return (group Series, ordered group names, colour map, note)."""
    note = None
    if mode == COLOR_CURRENT_FDR:
        if df["current_fdr"].notna().any():
            groups = np.where(
                df["current_fdr"] < 0.05,
                "Significant (FDR < 0.05)",
                "Not significant (current)",
            )
            order = ["Not significant (current)", "Significant (FDR < 0.05)"]
            return pd.Series(groups, index=df.index), order, GROUP_COLORS, note
        note = f"'{feature}' has no bootstrap FDR column; colouring by nothing."
        mode = COLOR_NONE

    elif mode == COLOR_EXTERNAL_FDR:
        tier = external_significance_labels(df, screen_key)
        if tier is not None:
            order = ["Not significant", "FDR < 10%", "FDR < 5%", "FDR < 1%"]
            return tier, order, GROUP_COLORS, note
        note = (
            f"{screen_label(screen_key)} carries no significance columns "
            "(its casTLE p-value is N/A for every gene) — use the casTLE score "
            "threshold below instead."
        )
        mode = COLOR_NONE

    elif mode == COLOR_QUADRANT:
        if metric in UNSIGNED_METRICS:
            note = f"'{metric}' is an unsigned magnitude, so quadrants are undefined."
            mode = COLOR_NONE
        else:
            sx = np.sign(df[metric].values)
            sy = np.sign(df[feature].values)
            groups = np.where(
                (sx == 0) | (sy == 0),
                "At zero",
                np.where(sx == sy, "Concordant (same sign)", "Discordant (opposite sign)"),
            )
            order = ["At zero", "Discordant (opposite sign)", "Concordant (same sign)"]
            return pd.Series(groups, index=df.index), order, GROUP_COLORS, note

    elif mode == COLOR_CLUSTER:
        if df["cluster"].notna().any():
            groups = df["cluster"].apply(
                lambda c: "No cluster" if pd.isna(c) else f"Cluster {int(c)}"
            )
            order = sorted(
                groups.unique(),
                key=lambda s: (s == "No cluster", int(s.split()[-1]) if s != "No cluster" else 0),
            )
            palette = {
                name: ("#cccccc" if name == "No cluster" else _cluster_color(i))
                for i, name in enumerate(order)
            }
            return groups, order, palette, note
        note = "No Leiden cluster assignments found for these genes."
        mode = COLOR_NONE

    groups = pd.Series("All matched genes", index=df.index)
    return groups, ["All matched genes"], GROUP_COLORS, note


def _cluster_color(i):
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]
    return palette[i % len(palette)]


def build_comparison_scatter(
    df,
    screen_key,
    metric,
    feature,
    color_mode,
    min_score,
    use_ranks=False,
    show_fit=True,
    show_error_bars=False,
    show_labels=False,
    label_by="Strongest old-screen effect",
    max_labels=40,
):
    x_raw = df[metric].astype(float)
    y_raw = df[feature].astype(float)
    if use_ranks:
        x = x_raw.rank(method="average")
        y = y_raw.rank(method="average")
        x_title = f"{screen_label(screen_key)} — {metric} (rank)"
        y_title = f"{feature} (rank)"
    else:
        x, y = x_raw, y_raw
        x_title = f"{screen_label(screen_key)} — {metric}"
        y_title = feature

    plot_df = df.assign(_x=x.values, _y=y.values)

    # Marker emphasis is driven by old-screen hit strength, independently of colour.
    if min_score and "Combo casTLE Score" in plot_df.columns:
        strong = plot_df["Combo casTLE Score"].fillna(-np.inf) >= min_score
    else:
        strong = pd.Series(True, index=plot_df.index)
    sizes = np.where(strong, 10.0, 6.0)
    opacities = np.where(strong, 0.85, 0.30)

    groups, order, palette, note = assign_color_groups(
        plot_df, color_mode, screen_key, metric, feature
    )
    plot_df = plot_df.assign(_group=groups.values, _size=sizes, _opacity=opacities)

    hover_cols = ["gene_symbol_0", metric, feature]
    for col in ("Combo casTLE Score", "current_fdr", "Hochberg"):
        if col in plot_df.columns and col not in hover_cols:
            hover_cols.append(col)
    hover_lines = ["<b>%{customdata[0]}</b>"]
    for i, col in enumerate(hover_cols[1:], start=1):
        label = {"current_fdr": f"{feature} FDR", "Hochberg": "old-screen Hochberg p"}.get(
            col, col
        )
        hover_lines.append(f"{label}: %{{customdata[{i}]:.4g}}")
    hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

    fig = go.Figure()
    for name in order:
        subset = plot_df[plot_df["_group"] == name]
        if subset.empty:
            continue
        marker = dict(
            color=palette.get(name, "#4575b4"),
            size=subset["_size"].tolist(),
            opacity=subset["_opacity"].tolist(),
            line=dict(width=0),
        )
        error_x = None
        if (
            show_error_bars
            and not use_ranks
            and metric == "Combo casTLE Effect"
            and {"Minimum Effect Estimate", "Maximum Effect Estimate"} <= set(subset.columns)
        ):
            error_x = dict(
                type="data",
                symmetric=False,
                array=(subset["Maximum Effect Estimate"] - subset[metric]).clip(lower=0),
                arrayminus=(subset[metric] - subset["Minimum Effect Estimate"]).clip(lower=0),
                thickness=0.7,
                width=0,
                color="rgba(120,120,120,0.35)",
            )
        fig.add_trace(
            go.Scatter(
                x=subset["_x"],
                y=subset["_y"],
                mode="markers",
                name=name,
                marker=marker,
                error_x=error_x,
                customdata=subset[hover_cols].values,
                hovertemplate=hovertemplate,
            )
        )

    xspan = (x.max() - x.min()) or 1
    yspan = (y.max() - y.min()) or 1
    xlim = (x.min() - 0.05 * xspan, x.max() + 0.05 * xspan)
    ylim = (y.min() - 0.05 * yspan, y.max() + 0.05 * yspan)

    # Zero reference lines only where zero is a meaningful origin.
    if not use_ranks:
        if metric not in UNSIGNED_METRICS and xlim[0] < 0 < xlim[1]:
            fig.add_vline(x=0, line_dash="dot", line_color="#999999", line_width=1)
        if ylim[0] < 0 < ylim[1]:
            fig.add_hline(y=0, line_dash="dot", line_color="#999999", line_width=1)

    if show_fit and len(x.dropna()) >= 3:
        keep = x.notna() & y.notna()
        if keep.sum() >= 3 and x[keep].nunique() > 1:
            slope, intercept = np.polyfit(x[keep], y[keep], 1)
            fit_x = np.array(xlim)
            fig.add_trace(
                go.Scatter(
                    x=fit_x,
                    y=slope * fit_x + intercept,
                    mode="lines",
                    name=f"OLS fit (slope {slope:.3g})",
                    line=dict(color="#222222", width=1.5, dash="dash"),
                    hoverinfo="skip",
                )
            )

    if show_labels and not plot_df.empty:
        lab = _select_labels(plot_df, metric, feature, label_by, max_labels)
        if not lab.empty:
            add_point_labels(
                fig,
                lab["_x"].values,
                lab["_y"].values,
                lab["gene_symbol_0"].astype(str).values,
                plot_df["_x"].values,
                plot_df["_y"].values,
                xlim,
                ylim,
            )

    fig.update_layout(
        title=f"{y_title}  vs  {x_title}",
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis=dict(range=list(xlim)),
        yaxis=dict(range=list(ylim)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=90, b=40, l=40, r=20),
        height=620,
    )
    return fig, note


def _select_labels(plot_df, metric, feature, label_by, max_labels):
    """Pick which genes get text labels."""
    df = plot_df.copy()
    if label_by == "Strongest old-screen effect":
        df["_rank_key"] = df[metric].abs()
        return df.nlargest(min(max_labels, len(df)), "_rank_key")
    if label_by == "Highest old-screen casTLE score" and "Combo casTLE Score" in df:
        return df.nlargest(min(max_labels, len(df)), "Combo casTLE Score")
    if label_by == "Most significant current feature" and df["current_fdr"].notna().any():
        return df.nsmallest(min(max_labels, len(df)), "current_fdr")
    if label_by == "Largest disagreement":
        # Distance from the OLS fit, in standardised units.
        x = df[metric].rank(method="average")
        y = df[feature].rank(method="average")
        if x.nunique() < 2 or y.nunique() < 2:
            return df.head(0)
        slope, intercept = np.polyfit(x, y, 1)
        df["_rank_key"] = (y - (slope * x + intercept)).abs()
        return df.nlargest(min(max_labels, len(df)), "_rank_key")
    return df.head(0)


# =====================
# GENE INFO DISPLAYS


def display_uniprot_info(gene):
    leiden_resolution = st.session_state.get("leiden_resolution", "15")
    row = load_uniprot_for_gene(gene, str(leiden_resolution))
    if row is None:
        st.info("UniProt info not found for this gene.")
        return
    entry = row.get("uniprot_entry", "")
    link = row.get("uniprot_link", "")
    function_text = row.get("uniprot_function", "")
    if entry and link:
        st.write(f"Uniprot Entry: [{entry}]({link})")
    if isinstance(function_text, str) and function_text.strip():
        st.markdown(f"Uniprot Function:\n>{function_text}")
    else:
        st.write("Uniprot Function: Not available")


def display_external_record(gene):
    """Side-by-side casTLE record for this gene in every available screen."""
    key = str(gene).upper()
    rows = {}
    for screen_key in available_screens():
        table = load_screen(screen_key)
        if key in table.index:
            rows[screen_label(screen_key)] = table.loc[key]
    if not rows:
        st.info("This gene is not present in any of the external screens.")
        return
    record = pd.DataFrame(rows)
    numeric_rows = [i for i in record.index if i not in ("#GeneID", "GeneInfo")]
    st.dataframe(record.loc[numeric_rows], use_container_width=True)
    info = record.loc["GeneInfo"].dropna().unique() if "GeneInfo" in record.index else []
    for text in info:
        if isinstance(text, str) and text.strip() and text.strip().lower() != "nan":
            st.caption(text)
            break


def display_gene_montages(gene_montages_root, gene):
    gene_dir = os.path.join(gene_montages_root, gene)
    if not os.path.exists(gene_dir):
        st.warning(f"No montage directory found for gene {gene}")
        return

    montage_data = load_montage_data(gene_montages_root, gene)
    if montage_data.empty:
        st.write(f"No montage data found for gene {gene}")
        return

    available_guides = sorted(montage_data["guide"].unique())

    if f"esc_guide_{gene}" not in st.session_state:
        st.session_state[f"esc_guide_{gene}"] = None

    def on_guide_select():
        st.session_state[f"esc_guide_{gene}"] = st.session_state[
            f"esc_guide_dropdown_{gene}"
        ]

    selected_guide = st.session_state.get(f"esc_guide_{gene}", None)
    selected_index = 0
    if selected_guide in available_guides:
        selected_index = available_guides.index(selected_guide)
    elif available_guides:
        selected_guide = available_guides[0]
        st.session_state[f"esc_guide_{gene}"] = selected_guide

    selected_guide = st.selectbox(
        "Select Guide",
        available_guides,
        index=selected_index,
        key=f"esc_guide_dropdown_{gene}",
        on_change=on_guide_select,
    )

    filtered_montage_data = montage_data[montage_data["guide"] == selected_guide]
    if filtered_montage_data.empty:
        st.warning(f"No image found for {gene} - {selected_guide}")
        return

    for _, row in filtered_montage_data.iterrows():
        image_path = os.path.join(gene_montages_root, row["file_path"])
        channel_name = row["channel"].replace("CH-", "")
        try:
            if os.path.exists(image_path):
                st.image(image_path, caption=f"Channel: {channel_name}")
            else:
                st.error(f"Image file not found: {image_path}")
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")

    overlay_tiff_path = os.path.join(
        gene_montages_root, gene, selected_guide, "overlay_montage.tiff"
    )
    if os.path.exists(overlay_tiff_path):
        render_composite_montage(
            overlay_tiff_path, key_prefix=f"esc_{gene}_{selected_guide}"
        )
        if STATIC_ASSET_URL_ROOT and STATIC_ASSET_PATH:
            relative_path = overlay_tiff_path.replace(STATIC_ASSET_PATH, "")
            st.markdown(f"[Download Overlay TIFF]({STATIC_ASSET_URL_ROOT}{relative_path})")
        else:
            with open(overlay_tiff_path, "rb") as f:
                st.download_button(
                    label="Download Overlay TIFF",
                    data=f,
                    file_name=f"{gene}_{selected_guide}_overlay.tiff",
                    key=f"esc_download_{gene}_{selected_guide}_{uuid.uuid4()}",
                )
    else:
        st.warning(f"No overlay tiff found: {overlay_tiff_path}")


def display_sgrna_representativeness(gene, cell_class, channel_combo):
    st.markdown("#### sgRNA Representativeness vs. Control")

    aggregate_cfg = load_config().get("aggregate", {})
    perturbation_name_col = aggregate_cfg.get("perturbation_name_col", "gene_symbol_0")
    perturbation_id_col = aggregate_cfg.get("perturbation_id_col", "cell_barcode_0")
    control_key = aggregate_cfg.get("control_key", "0Safe")

    bootstrap_df = load_bootstrap_results(cell_class, channel_combo)
    construct_df = load_construct_table(cell_class, channel_combo)
    gene_df = load_gene_table(cell_class, channel_combo)

    if bootstrap_df is None:
        st.info("No gene bootstrap results found for this cell class/channel combo.")
        return
    if construct_df is None or gene_df is None:
        st.info("Gene/construct feature tables not found for this cell class/channel combo.")
        return

    sig_features = get_significant_features_for_gene(bootstrap_df, gene, gene_col="gene")
    if not sig_features:
        st.write(f"No bootstrap-significant features found for {gene} (FDR < 0.05).")
        return

    ranking_df = rank_constructs_for_gene(
        gene,
        construct_df,
        gene_df,
        sig_features,
        perturbation_name_col,
        perturbation_id_col,
        min_cell_count=50,
    )
    st.write(
        f"{len(sig_features)} significant feature(s); sgRNAs ranked by distance "
        "to gene median (most representative first):"
    )
    st.dataframe(ranking_df)

    feature_options = [f for f, _ in sig_features]
    selected_feature = st.selectbox(
        "Feature to plot vs. control",
        options=feature_options,
        key=f"esc_rep_feature_{gene}_{cell_class}_{channel_combo}",
    )

    control_values = load_cached_control_feature_values(
        cell_class, channel_combo, selected_feature, perturbation_name_col, control_key
    )
    if control_values is None:
        st.warning("Single-cell parquet not found for this cell class/channel combo.")
        return

    fig, ax = plot_feature_vs_control(
        selected_feature,
        gene,
        construct_df,
        gene_df,
        control_values,
        perturbation_name_col,
        perturbation_id_col,
        control_key,
    )
    st.pyplot(fig)
    svg_buf = io.StringIO()
    fig.savefig(svg_buf, format="svg", bbox_inches="tight")
    st.download_button(
        label="Download plot as SVG (vector, editable in Illustrator)",
        data=svg_buf.getvalue(),
        file_name=f"{gene}_{selected_feature}.svg",
        mime="image/svg+xml",
        key=f"esc_svg_dl_{uuid.uuid4()}",
    )


# =====================
# SESSION STATE INIT


def init_state():
    st.session_state.setdefault("esc_selected_gene", None)
    st.session_state.setdefault("esc_feature", None)


init_state()

# =====================
# PAGE HEADER

title_col, reset_col = st.columns([4, 1])
with title_col:
    st.title("External Screen Comparison")
with reset_col:
    st.write("")
    if st.button("Reset All", type="secondary"):
        st.session_state.esc_selected_gene = None
        st.session_state.esc_feature = None
        st.rerun()

screens = available_screens()
if not screens:
    st.warning(
        "No external screen data found. Set ADCP_COMBO_PATH and/or "
        "BEADS_GENOME_WIDE_PATH (see analysis/14.run_visualization.sh) and run "
        "`python external_data/export_screens.py` to build the bead screen TSV."
    )
    st.stop()

gene_df_check = load_gene_table(CELL_CLASS, CHANNEL_COMBO)
if gene_df_check is None:
    st.warning(
        f"No aggregate gene table found for {CELL_CLASS} / {CHANNEL_COMBO}."
    )
    st.stop()

# =====================
# SELECTORS

st.sidebar.title("Options")
show_all_features = st.sidebar.toggle(
    "Show all gene-table features",
    value=False,
    key="esc_show_all_features",
    help=(
        "Off: only features with a bootstrap FDR column (significance colouring "
        "available). On: every numeric column in the aggregate gene table."
    ),
)

sel_screen, sel_metric, sel_feature = st.columns([1, 1, 2])

with sel_screen:
    screen_key = st.selectbox(
        "Old screen",
        options=screens,
        format_func=screen_label,
        key="esc_screen",
    )

metrics = screen_metrics(screen_key)
if not metrics:
    st.warning(f"{screen_label(screen_key)} has no usable casTLE metric columns.")
    st.stop()

with sel_metric:
    metric = st.selectbox("Old-screen metric (X)", options=metrics, key="esc_metric")

feature_options = list_current_features(CELL_CLASS, CHANNEL_COMBO, not show_all_features)
if not feature_options:
    st.warning("No current-screen parameters available.")
    st.stop()

with sel_feature:
    pending = st.session_state.get("esc_feature")
    default_index = feature_options.index(pending) if pending in feature_options else 0
    feature = st.selectbox(
        f"Current parameter (Y) — {CELL_CLASS} / {CHANNEL_COMBO}",
        options=feature_options,
        index=default_index,
        key="esc_feature_select",
    )
st.session_state.esc_feature = feature

matched, unmatched = build_matched_df(screen_key, metric, feature)
if matched is None or matched.empty:
    st.warning(
        f"No genes matched between {screen_label(screen_key)} and the current "
        f"screen for '{feature}'."
    )
    st.stop()

# =====================
# TABS

scatter_tab, ranking_tab = st.tabs(["Scatter", "Feature correlation ranking"])

with scatter_tab:
    ctrl_a, ctrl_b, ctrl_c = st.columns([2, 2, 2])
    with ctrl_a:
        color_mode = st.selectbox(
            "Colour points by",
            options=[
                COLOR_CURRENT_FDR,
                COLOR_EXTERNAL_FDR,
                COLOR_QUADRANT,
                COLOR_CLUSTER,
                COLOR_NONE,
            ],
            key="esc_color_mode",
        )
    score_max = float(np.nanmax(matched.get("Combo casTLE Score", pd.Series([0.0]))))
    with ctrl_b:
        min_score = st.slider(
            "Emphasise old-screen casTLE score ≥",
            min_value=0.0,
            max_value=max(round(score_max, 1), 1.0),
            value=0.0,
            step=max(round(score_max / 100, 2), 0.01),
            key="esc_min_score",
            help=(
                "casTLE score is a likelihood-ratio-like hit strength, not a "
                "p-value. Genes below the threshold are drawn small and faded."
            ),
        )
    with ctrl_c:
        restrict = st.checkbox(
            "Drop genes below the score threshold",
            value=False,
            key="esc_restrict",
            help="Also excludes them from the correlation statistics.",
        )

    plot_df = matched
    if restrict and min_score and "Combo casTLE Score" in plot_df.columns:
        plot_df = plot_df[plot_df["Combo casTLE Score"].fillna(-np.inf) >= min_score]
    if plot_df.empty:
        st.warning("No genes remain above the casTLE score threshold.")
        st.stop()

    opt_a, opt_b, opt_c, opt_d = st.columns([1, 1, 1, 1])
    with opt_a:
        use_ranks = st.toggle(
            "Rank–rank",
            value=False,
            key="esc_use_ranks",
            help=(
                "Plot ranks instead of raw values. casTLE effects are rounded to "
                "0.1, so raw scatters are heavily gridded; ranks are immune to that."
            ),
        )
    with opt_b:
        show_fit = st.toggle("OLS fit line", value=True, key="esc_show_fit")
    with opt_c:
        show_error_bars = st.toggle(
            "casTLE effect CI",
            value=False,
            key="esc_error_bars",
            help="Min/Max Effect Estimate as x error bars (Combo casTLE Effect only).",
        )
    with opt_d:
        show_labels = st.toggle("Label genes", value=False, key="esc_show_labels")

    label_by = "Strongest old-screen effect"
    max_labels = 40
    if show_labels:
        lab_a, lab_b = st.columns([2, 1])
        with lab_a:
            label_by = st.selectbox(
                "Label which genes",
                options=[
                    "Strongest old-screen effect",
                    "Highest old-screen casTLE score",
                    "Most significant current feature",
                    "Largest disagreement",
                ],
                key="esc_label_by",
            )
        with lab_b:
            max_labels = st.slider(
                "Max labels", min_value=5, max_value=100, value=40, step=5,
                key="esc_max_labels",
            )

    s = correlation_stats(plot_df[metric].values, plot_df[feature].values)
    n_total_current = len(gene_df_check)
    n_controls = int(
        gene_df_check["gene_symbol_0"].astype(str).str.startswith("0Safe").sum()
    )
    st.caption(
        f"**{s['n']} genes matched** (of {n_total_current - n_controls} real genes in "
        f"the current screen; {n_controls} control rows never match) · "
        f"Spearman ρ = {s['rho']:.3f} (p = {s['rho_p']:.3g}) · "
        f"Pearson r = {s['r']:.3f} (p = {s['r_p']:.3g}) · Click a point to select a gene"
    )

    fig, note = build_comparison_scatter(
        plot_df,
        screen_key,
        metric,
        feature,
        color_mode,
        min_score,
        use_ranks=use_ranks,
        show_fit=show_fit,
        show_error_bars=show_error_bars,
        show_labels=show_labels,
        label_by=label_by,
        max_labels=max_labels,
    )
    if note:
        st.info(note)

    scatter_event = st.plotly_chart(
        fig,
        on_select="rerun",
        key="esc_scatter",
        use_container_width=True,
        config={"toImageButtonOptions": {"format": "svg"}},
    )
    if (
        scatter_event
        and hasattr(scatter_event, "selection")
        and scatter_event.selection
        and scatter_event.selection.points
    ):
        clicked = scatter_event.selection.points[0].get("customdata")
        if clicked and clicked[0] != st.session_state.esc_selected_gene:
            st.session_state.esc_selected_gene = clicked[0]
            st.rerun()

    dl_col, unmatched_col = st.columns([1, 3])
    with dl_col:
        export_cols = [
            c for c in ["gene_symbol_0", metric, feature, "current_fdr", "cluster"]
            if c in plot_df.columns
        ]
        export_cols += [
            c for c in plot_df.columns
            if c not in export_cols and c not in ("_gene_key", "#GeneID", "GeneInfo")
        ]
        st.download_button(
            "Download matched table (CSV)",
            data=plot_df[export_cols].to_csv(index=False),
            file_name=(
                f"{screen_key}_{metric}__vs__{feature}".replace(" ", "_").replace("/", "-")
                + ".csv"
            ),
            mime="text/csv",
            key="esc_download_matched",
        )

    with st.expander(
        f"Genes in the current screen with no {screen_label(screen_key)} record "
        f"({len(unmatched)})"
    ):
        real_unmatched = sorted(
            g for g in unmatched.astype(str) if not g.startswith("0Safe")
        )
        st.caption(
            f"{len(real_unmatched)} non-control genes unmatched "
            f"(the other {len(unmatched) - len(real_unmatched)} are 0Safe control rows). "
            "Check these for symbol aliases — e.g. ADCP uses the retired name IKBKAP "
            "for Elp1."
        )
        st.write(", ".join(real_unmatched) if real_unmatched else "None.")

with ranking_tab:
    st.markdown(
        f"Spearman correlation of **every current parameter** against "
        f"**{screen_label(screen_key)} — {metric}**, ranked by |ρ|. "
        "Select a row to load that parameter into the scatter."
    )
    ranking = rank_features_by_correlation(
        screen_key,
        metric,
        CELL_CLASS,
        CHANNEL_COMBO,
        not show_all_features,
        min_score if st.session_state.get("esc_restrict") else 0.0,
    )
    if ranking.empty:
        st.info("Not enough matched genes to rank features.")
    else:
        n_sig = int((ranking["spearman_fdr"] < 0.05).sum())
        st.caption(
            f"{len(ranking)} parameters tested · {n_sig} with BH-adjusted p < 0.05 · "
            f"strongest |ρ| = {ranking['abs_rho'].max():.3f}"
        )
        top_n = st.slider(
            "Show top N", min_value=10, max_value=min(200, len(ranking)),
            value=min(30, len(ranking)), step=5, key="esc_rank_top_n",
        )
        shown = ranking.head(top_n)
        table_event = st.dataframe(
            shown[
                ["feature", "spearman_rho", "spearman_p", "spearman_fdr",
                 "pearson_r", "n_genes"]
            ].style.format(
                {
                    "spearman_rho": "{:.3f}",
                    "spearman_p": "{:.3g}",
                    "spearman_fdr": "{:.3g}",
                    "pearson_r": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="esc_ranking_table",
        )
        selected_rows = []
        if table_event and hasattr(table_event, "selection"):
            selected_rows = table_event.selection.get("rows", []) or []
        if selected_rows:
            picked = shown.iloc[selected_rows[0]]["feature"]
            if picked != st.session_state.get("esc_feature"):
                st.session_state.esc_feature = picked
                st.rerun()

        bar = shown.head(25).iloc[::-1]
        bar_fig = go.Figure(
            go.Bar(
                x=bar["spearman_rho"],
                y=bar["feature"],
                orientation="h",
                marker=dict(
                    color=np.where(bar["spearman_rho"] >= 0, "#2c7fb8", "#d95f02")
                ),
                hovertemplate="<b>%{y}</b><br>ρ = %{x:.3f}<extra></extra>",
            )
        )
        bar_fig.update_layout(
            title=f"Top {len(bar)} parameters by |Spearman ρ| vs {metric}",
            xaxis_title="Spearman ρ",
            margin=dict(t=60, b=40, l=10, r=20),
            height=max(320, 22 * len(bar)),
        )
        st.plotly_chart(
            bar_fig,
            use_container_width=True,
            key="esc_ranking_bar",
            config={"toImageButtonOptions": {"format": "svg"}},
        )

        st.download_button(
            "Download full ranking (CSV)",
            data=ranking.to_csv(index=False),
            file_name=f"{screen_key}_{metric}__feature_correlations.csv".replace(" ", "_"),
            mime="text/csv",
            key="esc_download_ranking",
        )

# =====================
# GENE INFO PANEL

selected_gene = st.session_state.esc_selected_gene
if not selected_gene:
    st.stop()

st.divider()

gene_title_col, clear_col = st.columns([4, 1])
with gene_title_col:
    st.markdown(f"## Gene: {selected_gene}")
with clear_col:
    st.write("")
    if st.button("Clear Gene", type="secondary"):
        st.session_state.esc_selected_gene = None
        st.rerun()

st.markdown("#### Old-screen casTLE records")
display_external_record(selected_gene)

st.markdown("#### UniProt Summary")
display_uniprot_info(selected_gene)

gene_montages_root = os.path.join(
    BRIEFLOW_OUTPUT_PATH, "aggregate", "montages", f"{CELL_CLASS}__montages"
)
if os.path.exists(gene_montages_root):
    st.markdown("#### Gene Montages")
    display_gene_montages(gene_montages_root, selected_gene)

display_sgrna_representativeness(selected_gene, CELL_CLASS, CHANNEL_COMBO)
