import os

import streamlit as st

st.set_page_config(
    page_title="CellProfiler PCA Embeddings - Brieflow Analysis",
    layout="wide",
)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

# =====================
# CONFIGURATION
#
# Viewer for the Liu et al.-style per-channel-PCA + control-normalized embedding
# built from raw CellProfiler features
# (pca_embedding_analysis/scripts/run_cellprofiler_pca.py).

CELLPROFILER_PCA_OUTPUT_PATH = os.environ.get("CELLPROFILER_PCA_OUTPUT_PATH", "")

LEIDEN_RESOLUTIONS = [1, 3, 5, 7, 9, 11, 13, 15, 20, 50]

# Phagocytosis cell filter: keep only cells with cell_RFP_int at/above a threshold
# set from the Safe/control cell_RFP_int distribution (see
# pca_embedding_analysis/scripts/run_cellprofiler_pca.py + pca_embedding_analysis/lib/rfp_filter.py).
RFP_FILTERS = {
    "All cells": "",
    "Excl. bottom 10% RFP (Safe)": "_rfpExcl10",
    "Excl. bottom 50% RFP (Safe)": "_rfpExcl50",
    "Excl. bottom 70% RFP (Safe)": "_rfpExcl70",
}

HOVER_COLUMNS = ["name", "cluster", "cell_count", "is_control"]
NAME_IDX = 0
CLUSTER_IDX = 1
CELL_COUNT_IDX = 2
IS_CONTROL_IDX = 3


def tsv_path(level_prefix: str, rfp_suffix: str) -> str:
    return os.path.join(
        CELLPROFILER_PCA_OUTPUT_PATH, f"cellprofiler{rfp_suffix}_{level_prefix}_pca_phate_leiden.tsv"
    )


# =====================
# DATA LOADING


@st.cache_data
def load_tsv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def get_name_col(level: str) -> str:
    return "gene_label" if level == "Per Gene" else "guide_label"


def prepare_df(raw: pd.DataFrame, name_col: str, leiden_res: int) -> pd.DataFrame:
    df = raw.copy()
    df["name"] = df[name_col].astype(str)
    df["cluster"] = df[f"leiden_{leiden_res}"].astype(str)
    df["is_control"] = df["is_control"].astype(bool)
    return df


# =====================
# COLOR PALETTE


def turbo_palette(n: int) -> list[str]:
    cmap = plt.get_cmap("turbo")
    return [mcolors.rgb2hex(cmap(i / max(n - 1, 1))) for i in range(n)]


# =====================
# SCATTER PLOT


def make_trace(x, y, marker, customdata, name: str):
    hovertemplate = (
        "PHATE_0=%{x:.3f}<br>"
        "PHATE_1=%{y:.3f}<br>"
        f"name=%{{customdata[{NAME_IDX}]}}<br>"
        f"cluster=%{{customdata[{CLUSTER_IDX}]}}<br>"
        f"cell_count=%{{customdata[{CELL_COUNT_IDX}]}}<br>"
        f"control=%{{customdata[{IS_CONTROL_IDX}]}}<br>"
        "<extra></extra>"
    )
    trace_cls = (
        go.Scatter if st.session_state.get("cellprofiler_pca_vector_export", False) else go.Scattergl
    )
    return trace_cls(
        x=x,
        y=y,
        mode="markers",
        marker=marker,
        customdata=customdata,
        name=name,
        hovertemplate=hovertemplate,
        showlegend=False,
    )


def build_figure(df: pd.DataFrame) -> go.Figure:
    clusters = sorted(df["cluster"].unique(), key=lambda c: int(c) if c.isdigit() else -1)
    palette = turbo_palette(len(clusters))
    color_map = {c: palette[i] for i, c in enumerate(clusters)}

    selected_item = st.session_state.get("cellprofiler_pca_selected_cluster", None)
    selected_name = st.session_state.get("cellprofiler_pca_selected_name", None)

    fig = go.Figure()

    if selected_item is not None:
        unselected = df[df["cluster"] != selected_item]
        if not unselected.empty:
            fig.add_trace(
                make_trace(
                    x=unselected["PHATE_0"],
                    y=unselected["PHATE_1"],
                    marker=dict(color="gray", size=7, opacity=0.25),
                    customdata=unselected[HOVER_COLUMNS],
                    name="unselected",
                )
            )

        sel_df = df[df["cluster"] == selected_item]
        cluster_color = color_map[selected_item]

        highlight = sel_df[sel_df["name"] == selected_name] if selected_name else pd.DataFrame()
        rest = sel_df[sel_df["name"] != selected_name] if selected_name else sel_df

        if not rest.empty:
            fig.add_trace(
                make_trace(
                    x=rest["PHATE_0"],
                    y=rest["PHATE_1"],
                    marker=dict(
                        color=cluster_color,
                        size=10,
                        opacity=1.0,
                        line=dict(width=2, color="black"),
                    ),
                    customdata=rest[HOVER_COLUMNS],
                    name=selected_item,
                )
            )
        if not highlight.empty:
            fig.add_trace(
                make_trace(
                    x=highlight["PHATE_0"],
                    y=highlight["PHATE_1"],
                    marker=dict(
                        color=cluster_color,
                        size=15,
                        opacity=1.0,
                        symbol="circle",
                        line=dict(width=3, color="white"),
                    ),
                    customdata=highlight[HOVER_COLUMNS],
                    name=f"{selected_name} (selected)",
                )
            )
    else:
        for cluster in clusters:
            cdf = df[df["cluster"] == cluster]
            fig.add_trace(
                make_trace(
                    x=cdf["PHATE_0"],
                    y=cdf["PHATE_1"],
                    marker=dict(color=color_map[cluster], size=8, opacity=0.9),
                    customdata=cdf[HOVER_COLUMNS],
                    name=cluster,
                )
            )

    fig.update_layout(
        hovermode="closest",
        showlegend=False,
        width=900,
        height=750,
        xaxis_title="PHATE_0",
        yaxis_title="PHATE_1",
    )

    if st.session_state.get("cellprofiler_pca_zoom_x") and st.session_state.get("cellprofiler_pca_zoom_y"):
        fig.update_layout(
            xaxis=dict(range=st.session_state.cellprofiler_pca_zoom_x),
            yaxis=dict(range=st.session_state.cellprofiler_pca_zoom_y),
        )

    return fig


# =====================
# SESSION STATE


def init_session_state():
    defaults = {
        "cellprofiler_pca_selected_cluster": None,
        "cellprofiler_pca_selected_name": None,
        "cellprofiler_pca_zoom_x": None,
        "cellprofiler_pca_zoom_y": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_selection():
    st.session_state.cellprofiler_pca_selected_cluster = None
    st.session_state.cellprofiler_pca_selected_name = None
    st.session_state.cellprofiler_pca_zoom_x = None
    st.session_state.cellprofiler_pca_zoom_y = None


# =====================
# MAIN APP

init_session_state()

st.title("CellProfiler PCA Embedding Analysis")
st.caption(
    "Liu et al.-style pipeline: per-channel PCA (80% variance) -> average to guide -> "
    "normalize to safe-guide controls -> second PCA (40% variance) -> average to gene."
)

if not CELLPROFILER_PCA_OUTPUT_PATH:
    st.error(
        "CELLPROFILER_PCA_OUTPUT_PATH environment variable is not set. "
        "Add it to 14.run_visualization.sh and restart the server."
    )
    st.stop()

with st.sidebar:
    st.markdown("## CellProfiler PCA Controls")

    level = st.radio(
        "**Aggregation Level**",
        ["Per Gene", "Per Guide"],
        key="cellprofiler_pca_level",
        on_change=reset_selection,
    )

    leiden_res = st.radio(
        "**Leiden Resolution**",
        LEIDEN_RESOLUTIONS,
        key="cellprofiler_pca_leiden_res",
        on_change=reset_selection,
    )

    rfp_filter = st.radio(
        "**Phagocytosis filter (RFP)**",
        list(RFP_FILTERS.keys()),
        key="cellprofiler_pca_rfp_filter",
        on_change=reset_selection,
    )

name_col = get_name_col(level)
level_prefix = "gene" if level == "Per Gene" else "guide"
rfp_suffix = RFP_FILTERS[rfp_filter]
current_tsv = tsv_path(level_prefix, rfp_suffix)
raw = load_tsv(current_tsv)

if raw.empty:
    st.warning(
        f"No data found at `{current_tsv}`. "
        "Run scripts/run_cellprofiler_pca.py (pca_embedding_analysis/) to generate this TSV."
    )
    st.stop()

leiden_col = f"leiden_{leiden_res}"
if leiden_col not in raw.columns:
    st.warning(
        f"Column `{leiden_col}` not found in TSV. "
        f"Available: {[c for c in raw.columns if c.startswith('leiden_')]}"
    )
    st.stop()

df = prepare_df(raw, name_col, leiden_res)

with st.sidebar:
    all_names = sorted(df["name"].unique().tolist())
    search_options = ["— none —"] + all_names
    current_name = st.session_state.cellprofiler_pca_selected_name
    search_idx = search_options.index(current_name) if current_name in search_options else 0

    def on_name_search():
        chosen = st.session_state.cellprofiler_pca_name_search
        if chosen == "— none —":
            st.session_state.cellprofiler_pca_selected_name = None
            st.session_state.cellprofiler_pca_selected_cluster = None
        else:
            st.session_state.cellprofiler_pca_selected_name = chosen
            row = df[df["name"] == chosen]
            if not row.empty:
                st.session_state.cellprofiler_pca_selected_cluster = str(row["cluster"].iloc[0])

    st.selectbox(
        "**Search gene / guide**",
        search_options,
        index=search_idx,
        key="cellprofiler_pca_name_search",
        on_change=on_name_search,
    )

    cluster_options = ["— none —"] + sorted(
        df["cluster"].unique(), key=lambda c: int(c) if c.isdigit() else -1
    )
    cur_cluster = st.session_state.cellprofiler_pca_selected_cluster
    cluster_idx = cluster_options.index(cur_cluster) if cur_cluster in cluster_options else 0

    def on_cluster_select():
        chosen = st.session_state.cellprofiler_pca_cluster_dropdown
        if chosen == "— none —":
            st.session_state.cellprofiler_pca_selected_cluster = None
            st.session_state.cellprofiler_pca_selected_name = None
        else:
            st.session_state.cellprofiler_pca_selected_cluster = chosen
            genes_in_cluster = df[df["cluster"] == chosen]["name"].tolist()
            if genes_in_cluster:
                st.session_state.cellprofiler_pca_selected_name = genes_in_cluster[0]

    st.selectbox(
        "**Select cluster**",
        cluster_options,
        index=cluster_idx,
        key="cellprofiler_pca_cluster_dropdown",
        on_change=on_cluster_select,
    )

    if st.button("Clear selection"):
        reset_selection()
        st.rerun()

n_clusters = df["cluster"].nunique()
n_controls = df["is_control"].sum()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Perturbations", len(df))
col2.metric("Clusters", n_clusters)
col3.metric("Controls", int(n_controls))
col4.metric("Resolution", leiden_res)

st.checkbox(
    "Vectorize points for SVG export (slower render; makes points selectable in Illustrator)",
    key="cellprofiler_pca_vector_export",
)
fig = build_figure(df)
event = st.plotly_chart(
    fig,
    use_container_width=True,
    key="cellprofiler_pca_plot",
    on_select="rerun",
    config={"toImageButtonOptions": {"format": "svg"}},
)

if event.selection and event.selection.points:
    pt = event.selection.points[0]
    if "customdata" in pt and len(pt["customdata"]) > CLUSTER_IDX:
        clicked_cluster = str(pt["customdata"][CLUSTER_IDX])
        clicked_name = str(pt["customdata"][NAME_IDX])
        if (
            st.session_state.cellprofiler_pca_selected_cluster != clicked_cluster
            or st.session_state.cellprofiler_pca_selected_name != clicked_name
        ):
            st.session_state.cellprofiler_pca_selected_cluster = clicked_cluster
            st.session_state.cellprofiler_pca_selected_name = clicked_name
            st.rerun()

selected_cluster = st.session_state.cellprofiler_pca_selected_cluster
if selected_cluster is not None:
    cluster_df = df[df["cluster"] == selected_cluster].copy()
    cluster_df = cluster_df.sort_values("cell_count", ascending=False)
    pct_ctrl = 100 * cluster_df["is_control"].mean()

    st.markdown(
        f"### Cluster {selected_cluster} — {len(cluster_df)} perturbations ({pct_ctrl:.0f}% control)"
    )

    display_cols = (
        ["name", "gene_label", "cell_count", "is_control"]
        if level == "Per Guide"
        else ["name", "n_guides", "cell_count", "is_control"]
    )
    display_cols = [c for c in display_cols if c in cluster_df.columns]
    st.dataframe(cluster_df[display_cols].set_index("name"), use_container_width=True, height=300)
else:
    st.info("Click a point on the plot or use the dropdowns to select a cluster.")
