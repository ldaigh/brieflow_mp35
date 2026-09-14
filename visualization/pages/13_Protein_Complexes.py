"""Protein-complex level view of the screen.

Aggregates gene effects into complex effects the same way the pipeline
aggregates sgRNA effects into gene effects, so a complex can be a hit even when
no single member clears threshold on its own.

Reads the precomputed tables written by `scripts/complex_analysis/`. Nothing is
recomputed here: complex significance requires stacking ~74 MB bootstrap null
arrays per member, which is a batch job, not an interactive one.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Protein Complexes - Brieflow Analysis", layout="wide")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import BRIEFLOW_OUTPUT_PATH, COMPLEX_OUTPUT_PATH
from src.labels import add_point_labels
from workflow.lib.cluster.cluster_analysis import cluster_heatmap

CELL_CLASS = "all"
CHANNEL_COMBO = "DAPI_YFP_RFP_Cy5"
GROUP_COL = "group"
GENE_COL = "gene_symbol_0"
FDR_THRESHOLD = 0.05

# Sources whose members form physical complexes. GO_BP / GO_MF are broader
# functional groupings -- available, but off by default so they don't swamp the
# complex-level panels.
COMPLEX_LIKE_SOURCES = ["curated", "CORUM", "GO_CC"]


# =====================
# DATA LOADERS


def _complex_fp(kind, channel_combo=CHANNEL_COMBO, cell_class=CELL_CLASS):
    return os.path.join(
        COMPLEX_OUTPUT_PATH, f"CeCl-{cell_class}_ChCo-{channel_combo}__{kind}.tsv"
    )


@st.cache_data
def load_tsv(path):
    if not path or not os.path.exists(path):
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
    return load_tsv(path)


@st.cache_data
def load_gene_bootstrap(cell_class, channel_combo):
    path = os.path.join(
        BRIEFLOW_OUTPUT_PATH,
        "aggregate",
        "bootstrap",
        f"CeCl-{cell_class}_ChCo-{channel_combo}__all_gene_bootstrap_results.tsv",
    )
    return load_tsv(path)


@st.cache_data
def load_clustering(channel_combo, cell_class, resolution):
    path = os.path.join(
        BRIEFLOW_OUTPUT_PATH,
        "cluster",
        channel_combo,
        cell_class,
        str(resolution),
        "phate_leiden_clustering.tsv",
    )
    return load_tsv(path)


@st.cache_data
def available_resolutions(channel_combo, cell_class):
    base = os.path.join(BRIEFLOW_OUTPUT_PATH, "cluster", channel_combo, cell_class)
    if not os.path.isdir(base):
        return []
    return sorted(
        int(d)
        for d in os.listdir(base)
        if d.isdigit()
        and os.path.exists(os.path.join(base, d, "phate_leiden_clustering.tsv"))
    )


@st.cache_data
def significant_features(fdr_threshold=FDR_THRESHOLD):
    """Feature names significant for at least one complex."""
    bootstrap = load_tsv(_complex_fp("complex_bootstrap_results"))
    if bootstrap is None:
        return []
    fdr_cols = [c for c in bootstrap.columns if c.endswith("_fdr")]
    return sorted(c[:-4] for c in fdr_cols if (bootstrap[c] < fdr_threshold).any())


# =====================
# HELPERS


def merge_effect_and_stats(complex_features, complex_bootstrap, feature):
    """Join a feature's complex-level effect size to its p-value and FDR."""
    if complex_features is None or complex_bootstrap is None:
        return None
    fdr_col, log10_col = f"{feature}_fdr", f"{feature}_log10"
    if feature not in complex_features.columns:
        return None
    if fdr_col not in complex_bootstrap.columns:
        return None

    merged = complex_features[[GROUP_COL, "n_members", feature]].merge(
        complex_bootstrap[[GROUP_COL, log10_col, fdr_col]], on=GROUP_COL, how="inner"
    )
    merged["significant"] = merged[fdr_col] < FDR_THRESHOLD
    return merged.dropna(subset=[feature, log10_col])


def build_volcano(merged, feature, show_labels, label_fdr, max_labels):
    """Complex-level volcano: effect size vs -log10(FDR), one point per complex."""
    fdr_col, log10_col = f"{feature}_fdr", f"{feature}_log10"

    def trace(df, name, color, size):
        return go.Scatter(
            x=df[feature],
            y=df[log10_col],
            mode="markers",
            name=name,
            marker=dict(color=color, size=size, opacity=0.7),
            customdata=df[[GROUP_COL, "n_members"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "members: %{customdata[1]}<br>"
                "effect (z): %{x:.3f}<br>"
                "-log10(FDR): %{y:.3f}<extra></extra>"
            ),
        )

    fig = go.Figure()
    fig.add_trace(trace(merged[~merged["significant"]], "Background", "#aaaaaa", 8))
    fig.add_trace(
        trace(merged[merged["significant"]], "Significant (FDR<0.05)", "#756bb1", 12)
    )

    x_vals, y_vals = merged[feature], merged[log10_col]
    xspan = (x_vals.max() - x_vals.min()) or 1
    yspan = (y_vals.max() - y_vals.min()) or 1
    xlim = (x_vals.min() - 0.05 * xspan, x_vals.max() + 0.05 * xspan)
    ylim = (y_vals.min() - 0.05 * yspan, y_vals.max() + 0.05 * yspan)

    fig.add_hline(
        y=-math.log10(FDR_THRESHOLD),
        line_dash="dash",
        line_color="#888888",
        line_width=1,
    )

    if show_labels:
        lab = merged[merged[fdr_col] <= label_fdr]
        if len(lab) > max_labels:
            lab = lab.nlargest(max_labels, log10_col)
        if not lab.empty:
            add_point_labels(
                fig,
                lab[feature].values,
                lab[log10_col].values,
                lab[GROUP_COL].astype(str).values,
                x_vals.values,
                y_vals.values,
                xlim,
                ylim,
            )

    fig.update_layout(
        title=f"Complex volcano: {feature}",
        xaxis_title="complex effect (median of members, z)",
        yaxis_title="-log10(FDR)",
        xaxis=dict(range=list(xlim)),
        yaxis=dict(range=list(ylim)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=40, r=20),
        height=520,
    )
    return fig


def build_scatter(complex_features, gene_table, gene_sets, x, y, show_members):
    """Two-feature complex scatter, optionally tethering members to centroids.

    The member overlay is what makes fragmentation visible: a coherent complex
    shows members tightly around their centroid, a fragmented one scatters.
    """
    fig = go.Figure()

    if show_members:
        # Dedupe: the two axis dropdowns share one option list, so x == y is a
        # legal selection (useful for checking a feature against itself), and
        # `gene_table[[GENE_COL, x, x]]` would carry a duplicate column name all
        # the way into Plotly, which rejects it outright.
        member_cols = list(dict.fromkeys([GENE_COL, x, y]))
        members = gene_sets[["gene_name", GROUP_COL]].merge(
            gene_table[member_cols],
            left_on="gene_name",
            right_on=GENE_COL,
            how="inner",
        )
        centroids = complex_features.set_index(GROUP_COL)
        members = members[members[GROUP_COL].isin(centroids.index)]

        # Tether each member to its centroid: a None between segments breaks
        # the line so Plotly draws one spoke per member rather than a polyline.
        gap = np.full(len(members), None)
        seg_x = np.column_stack(
            [members[x].values, centroids.loc[members[GROUP_COL], x].values, gap]
        ).ravel()
        seg_y = np.column_stack(
            [members[y].values, centroids.loc[members[GROUP_COL], y].values, gap]
        ).ravel()

        fig.add_trace(
            go.Scatter(
                x=seg_x,
                y=seg_y,
                mode="lines",
                line=dict(color="#cccccc", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=members[x],
                y=members[y],
                mode="markers",
                name="Member genes",
                marker=dict(color="#bbbbbb", size=5, opacity=0.7),
                customdata=members[["gene_name", GROUP_COL]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=complex_features[x],
            y=complex_features[y],
            mode="markers",
            name="Complexes",
            marker=dict(color="#756bb1", size=12, opacity=0.85),
            customdata=complex_features[[GROUP_COL, "n_members"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>members: %{customdata[1]}<br>"
                f"{x}: %{{x:.3f}}<br>{y}: %{{y:.3f}}<extra></extra>"
            ),
        )
    )

    fig.add_hline(y=0, line_dash="dot", line_color="#dddddd", line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color="#dddddd", line_width=1)
    fig.update_layout(
        title=f"{y} vs {x}",
        xaxis_title=x,
        yaxis_title=y,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=40, r=20),
        height=560,
    )
    return fig


def build_coherence_scatter(coherence, complex_bootstrap, features):
    """Coherence vs strongest effect: separates coherent+strong from fragmented."""
    fdr_cols = [f"{f}_fdr" for f in features if f"{f}_fdr" in complex_bootstrap.columns]
    n_sig = (complex_bootstrap.set_index(GROUP_COL)[fdr_cols] < FDR_THRESHOLD).sum(
        axis=1
    )

    df = coherence.set_index(GROUP_COL).join(n_sig.rename("n_sig_features")).reset_index()
    df = df.dropna(subset=["coherence_z", "n_sig_features"])

    fig = go.Figure(
        go.Scatter(
            x=df["coherence_z"],
            y=df["n_sig_features"],
            mode="markers",
            marker=dict(
                size=np.clip(df["n_members"] * 1.6, 6, 26),
                color=df["mean_pairwise_r"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="mean<br>pairwise r"),
                opacity=0.8,
                line=dict(width=0.5, color="#555555"),
            ),
            customdata=df[[GROUP_COL, "n_members", "coherence_p", "source"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>members: %{customdata[1]}<br>"
                "source: %{customdata[3]}<br>"
                "coherence z: %{x:.2f} (p=%{customdata[2]:.4f})<br>"
                "significant features: %{y}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Coherence vs. phenotypic strength",
        xaxis_title="coherence z (vs random same-size gene sets)",
        yaxis_title=f"# features at FDR < {FDR_THRESHOLD}",
        margin=dict(t=60, b=40, l=40, r=20),
        height=520,
    )
    return fig


# =====================
# PAGE

st.title("Protein Complex Analysis")

gene_sets_all = load_tsv(os.path.join(COMPLEX_OUTPUT_PATH, "gene_sets.tsv"))

if gene_sets_all is None:
    st.error(
        f"No gene-set table found at `{COMPLEX_OUTPUT_PATH}/gene_sets.tsv`.\n\n"
        "Build the complex-level outputs first:\n"
        "```\n"
        "bash scripts/complex_analysis/fetch_go_annotations.sh\n"
        "python -m scripts.complex_analysis.build_gene_sets\n"
        "sbatch scripts/complex_analysis/aggregate_complexes.sbatch\n"
        "python -m scripts.complex_analysis.complex_cluster_enrichment\n"
        "```"
    )
    st.stop()

complex_features_all = load_tsv(_complex_fp("complex_features"))
complex_bootstrap_all = load_tsv(_complex_fp("complex_bootstrap_results"))
coherence_all = load_tsv(_complex_fp("complex_coherence"))
representativeness = load_tsv(_complex_fp("member_representativeness"))
power_summary = load_tsv(_complex_fp("power_summary"))
overlap_all = load_tsv(os.path.join(COMPLEX_OUTPUT_PATH, "gene_sets_overlap.tsv"))
gene_table = load_gene_table(CELL_CLASS, CHANNEL_COMBO)

st.caption(
    f"Channel combo `{CHANNEL_COMBO}`, cell class `{CELL_CLASS}`. "
    "Complex effects are the median across member genes of the control-centered "
    "gene medians; complex p-values come from stacking the members' bootstrap "
    "null distributions, mirroring how the pipeline builds gene stats from "
    "constructs."
)

# --- filters ---
with st.sidebar:
    st.header("Gene set filters")
    all_sources = sorted(gene_sets_all["source"].unique())
    default_sources = [s for s in COMPLEX_LIKE_SOURCES if s in all_sources]
    sources = st.multiselect(
        "Sources",
        all_sources,
        default=default_sources or all_sources,
        help=(
            "curated / CORUM / GO_CC are physical complexes. GO_BP and GO_MF are "
            "broader functional groupings and add hundreds of gene sets."
        ),
    )

    size_lo, size_hi = int(gene_sets_all["n_in_screen"].min()), int(
        gene_sets_all["n_in_screen"].max()
    )
    member_range = st.slider(
        "Members in screen", size_lo, size_hi, (size_lo, size_hi)
    )

    require_coherent = st.checkbox(
        "Only coherent gene sets (p < 0.05)",
        value=False,
        help="Members phenocopy each other more than a random same-size gene set.",
    )

gene_sets = gene_sets_all[
    gene_sets_all["source"].isin(sources)
    & gene_sets_all["n_in_screen"].between(*member_range)
]

if coherence_all is not None and require_coherent:
    coherent_groups = set(
        coherence_all.loc[coherence_all["coherence_p"] < 0.05, GROUP_COL]
    )
    gene_sets = gene_sets[gene_sets[GROUP_COL].isin(coherent_groups)]

selected_groups = sorted(gene_sets[GROUP_COL].unique())

if not selected_groups:
    st.warning("No gene sets match the current filters.")
    st.stop()


def restrict(df):
    """Restrict a complex-level table to the currently filtered groups."""
    if df is None:
        return None
    return df[df[GROUP_COL].isin(selected_groups)]


complex_features = restrict(complex_features_all)
complex_bootstrap = restrict(complex_bootstrap_all)
coherence = restrict(coherence_all)

col1, col2, col3 = st.columns(3)
col1.metric("Gene sets", len(selected_groups))
col2.metric("Genes covered", gene_sets["gene_name"].nunique())
if coherence is not None:
    col3.metric(
        "Coherent (p<0.05)", int((coherence["coherence_p"] < 0.05).sum())
    )

if complex_features is None:
    st.warning(
        "Complex effect sizes not found. Run "
        "`sbatch scripts/complex_analysis/aggregate_complexes.sbatch`."
    )
    st.stop()

feature_cols = [
    c
    for c in complex_features.columns
    if c not in [GROUP_COL, "n_members", "cell_count"]
]
sig_features = significant_features()

tabs = st.tabs(
    [
        "Overview",
        "Volcano",
        "Scatter",
        "Heatmap",
        "Coherence",
        "Cluster enrichment",
        "Overlap",
    ]
)

# --- Overview -----------------------------------------------------------
with tabs[0]:
    st.subheader("Gene sets")

    table = complex_features[[GROUP_COL, "n_members", "cell_count"]].copy()
    table = table.merge(
        gene_sets.drop_duplicates(GROUP_COL)[[GROUP_COL, "source", "sources"]],
        on=GROUP_COL,
        how="left",
    )
    if coherence is not None:
        table = table.merge(
            coherence[[GROUP_COL, "mean_pairwise_r", "coherence_z", "coherence_p"]],
            on=GROUP_COL,
            how="left",
        )
    if complex_bootstrap is not None:
        fdr_cols = [c for c in complex_bootstrap.columns if c.endswith("_fdr")]
        stats = complex_bootstrap.set_index(GROUP_COL)[fdr_cols]
        table = table.merge(
            pd.DataFrame(
                {
                    GROUP_COL: stats.index,
                    "n_sig_features": (stats < FDR_THRESHOLD).sum(axis=1).values,
                    "best_feature": [c[:-4] for c in stats.idxmin(axis=1).values],
                    "best_fdr": stats.min(axis=1).values,
                }
            ),
            on=GROUP_COL,
            how="left",
        )
    table = table.merge(
        gene_sets.groupby(GROUP_COL)["gene_name"]
        .apply(lambda s: ", ".join(sorted(s)))
        .rename("members")
        .reset_index(),
        on=GROUP_COL,
        how="left",
    )

    sort_col = "n_sig_features" if "n_sig_features" in table else "n_members"
    st.dataframe(
        table.sort_values(sort_col, ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    if power_summary is not None:
        st.subheader("Does aggregating buy power?")
        st.caption(
            "Features significant for the complex vs. for its single best member. "
            "Positive `power_gain` means the complex clears FDR on more features "
            "than any one of its genes does alone."
        )
        power = restrict(power_summary).sort_values("power_gain", ascending=False)
        gained = int((power["power_gain"] > 0).sum())
        st.metric(
            "Gene sets with a net gain", f"{gained} / {len(power)}"
        )
        st.dataframe(power, use_container_width=True, hide_index=True)

# --- Volcano ------------------------------------------------------------
with tabs[1]:
    if complex_bootstrap is None:
        st.warning(
            "No complex bootstrap results. Only "
            f"`{CHANNEL_COMBO}` has stored gene nulls, and the aggregation job "
            "must finish first."
        )
    else:
        options = sig_features or feature_cols
        feature = st.selectbox("Feature", options, key="volcano_feature")
        c1, c2, c3 = st.columns([1, 1, 1])
        show_labels = c1.checkbox("Label complexes", value=True)
        label_fdr = c2.number_input(
            "Label FDR ≤", 0.0, 1.0, 0.05, step=0.01, format="%.3f"
        )
        max_labels = int(c3.number_input("Max labels", 1, 200, 30))

        merged = merge_effect_and_stats(complex_features, complex_bootstrap, feature)
        if merged is None or merged.empty:
            st.info("No data for this feature.")
        else:
            st.plotly_chart(
                build_volcano(merged, feature, show_labels, label_fdr, max_labels),
                use_container_width=True,
            )
            st.dataframe(
                merged.sort_values(f"{feature}_fdr"),
                use_container_width=True,
                hide_index=True,
            )

# --- Scatter ------------------------------------------------------------
with tabs[2]:
    options = sig_features or feature_cols
    c1, c2 = st.columns(2)
    x = c1.selectbox("X feature", options, index=0, key="scatter_x")
    y = c2.selectbox(
        "Y feature", options, index=min(1, len(options) - 1), key="scatter_y"
    )
    show_members = st.checkbox(
        "Overlay member genes",
        value=True,
        help="Each member is tethered to its complex centroid.",
    )

    if gene_table is None:
        st.warning("Gene-level feature table not found; member overlay unavailable.")
    else:
        st.plotly_chart(
            build_scatter(
                complex_features, gene_table, gene_sets, x, y, show_members
            ),
            use_container_width=True,
        )

# --- Heatmap ------------------------------------------------------------
with tabs[3]:
    st.caption(
        "Member genes grouped by complex, against the phenotypic features. "
        "Rendered with the pipeline's own `cluster_heatmap`, with complex "
        "membership standing in for cluster assignment."
    )
    if gene_table is None:
        st.warning("Gene-level feature table not found.")
    else:
        chosen = st.multiselect(
            "Complexes",
            selected_groups,
            default=selected_groups[: min(6, len(selected_groups))],
        )
        default_features = (sig_features or feature_cols)[:30]
        heat_features = st.multiselect(
            "Features", sig_features or feature_cols, default=default_features
        )
        group_by = st.checkbox(
            "Group genes by complex (no gene clustering)", value=True
        )

        if chosen and heat_features:
            # cluster_heatmap keys off a "cluster" column; complex name stands in.
            membership = (
                gene_sets[gene_sets[GROUP_COL].isin(chosen)][["gene_name", GROUP_COL]]
                .drop_duplicates()
                .rename(columns={"gene_name": GENE_COL, GROUP_COL: "cluster"})
            )
            result = cluster_heatmap(
                gene_table,
                membership,
                chosen,
                heat_features,
                perturbation_name_col=GENE_COL,
                group_by_cluster=group_by,
                z_score="global",
                figsize=(16, 9),
            )
            # cluster_heatmap returns a seaborn ClusterGrid (or None); its
            # docstring's "(clustermap_obj, heatmap_data)" tuple is stale.
            if result is None:
                st.info("No genes found for the selected complexes.")
            else:
                st.pyplot(result.figure)
        else:
            st.info("Select at least one complex and one feature.")

# --- Coherence ----------------------------------------------------------
with tabs[4]:
    if coherence is None:
        st.warning("Coherence results not found.")
    else:
        if complex_bootstrap is not None:
            st.plotly_chart(
                build_coherence_scatter(
                    coherence, complex_bootstrap, sig_features or feature_cols
                ),
                use_container_width=True,
            )
        st.dataframe(
            coherence.sort_values("coherence_z", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        if representativeness is not None:
            st.subheader("Member representativeness")
            st.caption(
                "Correlation of each member to the leave-one-out median of the "
                "rest of its complex. A split into two modes is a complex "
                "fragmenting into subcomplexes."
            )
            group = st.selectbox("Complex", selected_groups, key="rep_group")
            sub = representativeness[representativeness[GROUP_COL] == group]
            if sub.empty:
                st.info("No representativeness data for this complex.")
            else:
                sub = sub.sort_values("r_to_loo_median")
                fig = go.Figure(
                    go.Bar(
                        x=sub["r_to_loo_median"],
                        y=sub["gene_name"],
                        orientation="h",
                        marker=dict(color="#756bb1"),
                    )
                )
                fig.update_layout(
                    xaxis_title="r to leave-one-out complex median",
                    yaxis_title="",
                    height=max(240, 32 * len(sub)),
                    margin=dict(t=30, b=40, l=40, r=20),
                )
                st.plotly_chart(fig, use_container_width=True)

# --- Cluster enrichment -------------------------------------------------
with tabs[5]:
    resolutions = available_resolutions(CHANNEL_COMBO, CELL_CLASS)
    if not resolutions:
        st.warning("No clustering output found.")
    else:
        resolution = st.selectbox(
            "Leiden resolution", resolutions, index=min(4, len(resolutions) - 1)
        )
        enrichment = load_tsv(
            os.path.join(
                COMPLEX_OUTPUT_PATH,
                f"complex_cluster_enrichment__ChCo-{CHANNEL_COMBO}_LR-{resolution}.tsv",
            )
        )
        if enrichment is None:
            st.warning(
                "No enrichment table for this resolution. Run "
                "`python -m scripts.complex_analysis.complex_cluster_enrichment`."
            )
        else:
            enrichment = enrichment[enrichment[GROUP_COL].isin(selected_groups)]
            sig = enrichment[enrichment["fdr"] < FDR_THRESHOLD]
            st.metric("Enriched (cluster, gene set) pairs", len(sig))

            if not sig.empty:
                pivot = sig.pivot_table(
                    index=GROUP_COL,
                    columns="cluster",
                    values="fdr",
                    aggfunc="min",
                )
                heat = -np.log10(pivot)
                fig = go.Figure(
                    go.Heatmap(
                        z=heat.values,
                        x=[str(c) for c in heat.columns],
                        y=heat.index,
                        colorscale="Purples",
                        colorbar=dict(title="-log10(FDR)"),
                        hovertemplate=(
                            "cluster %{x}<br>%{y}<br>-log10(FDR)=%{z:.2f}"
                            "<extra></extra>"
                        ),
                    )
                )
                fig.update_layout(
                    title=f"Gene sets enriched per cluster (LR-{resolution})",
                    xaxis_title="cluster",
                    height=max(320, 22 * len(heat)),
                    margin=dict(t=60, b=40, l=40, r=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                enrichment.sort_values("fdr"), use_container_width=True, hide_index=True
            )

# --- Overlap ------------------------------------------------------------
with tabs[6]:
    st.caption(
        "Genes may belong to several gene sets, so two apparently independent "
        "hits can be largely the same genes. High overlap also means the FDR "
        "correction across gene sets is anticonservative."
    )
    if overlap_all is None:
        st.warning("Overlap table not found.")
    else:
        overlap = overlap_all[
            overlap_all["group_a"].isin(selected_groups)
            & overlap_all["group_b"].isin(selected_groups)
        ]
        min_jaccard = st.slider("Minimum Jaccard", 0.0, 1.0, 0.3, step=0.05)
        overlap = overlap[overlap["jaccard"] >= min_jaccard]
        st.metric("Overlapping pairs", len(overlap))
        st.dataframe(
            overlap.sort_values("jaccard", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
