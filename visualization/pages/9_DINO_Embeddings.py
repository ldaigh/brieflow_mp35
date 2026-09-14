import os

import streamlit as st

st.set_page_config(
    page_title="DINO Embeddings - Brieflow Analysis",
    layout="wide",
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =====================
# CONFIGURATION

# Directory holding the pre-computed TSVs written by
# cs231n_analysis/scripts/run_phate_leiden.py
EMBEDDING_OUTPUT_PATH = os.environ.get("EMBEDDING_OUTPUT_PATH", "")

# Normalization variants of the centroid matrix (the switchable axis here, in
# place of celldino's channel combos — this embedding is a single 384-dim block).
NORM_VARIANTS = {
    "Centered + scaled (default)": "centered_scaled",
    "Control-centered": "centered",
    "Raw (celldino-style)": "raw",
}

LEIDEN_RESOLUTIONS = [1, 3, 5, 7, 9, 11, 13, 15, 20, 50]
TP_COL = "inferred_timepoint"

# Hover column ordering — must match HOVER_INDICES below
HOVER_COLUMNS = ["name", "cluster", "cell_count", "is_control", TP_COL]
NAME_IDX = 0
CLUSTER_IDX = 1
CELL_COUNT_IDX = 2
IS_CONTROL_IDX = 3
TP_IDX = 4


def tsv_path(level_prefix: str, variant: str) -> str:
    return os.path.join(
        EMBEDDING_OUTPUT_PATH, f"{level_prefix}_phate_leiden_{variant}.tsv"
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
    if TP_COL not in df.columns:
        df[TP_COL] = np.nan
    return df


# =====================
# COLOR PALETTE


def turbo_palette(n: int) -> list[str]:
    cmap = plt.get_cmap("turbo")
    return [mcolors.rgb2hex(cmap(i / max(n - 1, 1))) for i in range(n)]


# =====================
# SCATTER PLOT


def hovertemplate() -> str:
    return (
        "PHATE_0=%{x:.3f}<br>"
        "PHATE_1=%{y:.3f}<br>"
        f"name=%{{customdata[{NAME_IDX}]}}<br>"
        f"cluster=%{{customdata[{CLUSTER_IDX}]}}<br>"
        f"cell_count=%{{customdata[{CELL_COUNT_IDX}]}}<br>"
        f"control=%{{customdata[{IS_CONTROL_IDX}]}}<br>"
        f"timepoint=%{{customdata[{TP_IDX}]:.2f}}<br>"
        "<extra></extra>"
    )


def _trace_cls():
    # WebGL (Scattergl) is fast but rasterizes points on SVG export. When the
    # user enables vector export, use go.Scatter (SVG) so each point exports as
    # an individually selectable vector shape.
    return go.Scatter if st.session_state.get("dino_vector_export", False) else go.Scattergl


def make_trace(x, y, marker, customdata, name: str):
    return _trace_cls()(
        x=x, y=y, mode="markers", marker=marker, customdata=customdata,
        name=name, hovertemplate=hovertemplate(), showlegend=False,
    )


def build_cluster_figure(df: pd.DataFrame) -> go.Figure:
    clusters = sorted(df["cluster"].unique(), key=lambda c: int(c) if c.isdigit() else -1)
    palette = turbo_palette(len(clusters))
    color_map = {c: palette[i] for i, c in enumerate(clusters)}

    selected_item = st.session_state.get("dino_selected_cluster", None)
    selected_name = st.session_state.get("dino_selected_name", None)

    fig = go.Figure()

    if selected_item is not None:
        unselected = df[df["cluster"] != selected_item]
        if not unselected.empty:
            fig.add_trace(make_trace(
                unselected["PHATE_0"], unselected["PHATE_1"],
                dict(color="gray", size=7, opacity=0.25),
                unselected[HOVER_COLUMNS], "unselected"))

        sel_df = df[df["cluster"] == selected_item]
        cluster_color = color_map[selected_item]
        highlight = sel_df[sel_df["name"] == selected_name] if selected_name else pd.DataFrame()
        rest = sel_df[sel_df["name"] != selected_name] if selected_name else sel_df

        if not rest.empty:
            fig.add_trace(make_trace(
                rest["PHATE_0"], rest["PHATE_1"],
                dict(color=cluster_color, size=10, opacity=1.0,
                     line=dict(width=2, color="black")),
                rest[HOVER_COLUMNS], selected_item))
        if not highlight.empty:
            fig.add_trace(make_trace(
                highlight["PHATE_0"], highlight["PHATE_1"],
                dict(color=cluster_color, size=15, opacity=1.0, symbol="circle",
                     line=dict(width=3, color="white")),
                highlight[HOVER_COLUMNS], f"{selected_name} (selected)"))
    else:
        for cluster in clusters:
            cdf = df[df["cluster"] == cluster]
            fig.add_trace(make_trace(
                cdf["PHATE_0"], cdf["PHATE_1"],
                dict(color=color_map[cluster], size=8, opacity=0.9),
                cdf[HOVER_COLUMNS], cluster))

    _finalize(fig)
    return fig


def build_timepoint_figure(df: pd.DataFrame) -> go.Figure:
    """Color every point continuously by mean inferred timepoint."""
    fig = go.Figure()
    selected_name = st.session_state.get("dino_selected_name", None)

    fig.add_trace(_trace_cls()(
        x=df["PHATE_0"], y=df["PHATE_1"], mode="markers",
        marker=dict(
            color=df[TP_COL], colorscale="Viridis", size=8, opacity=0.9,
            colorbar=dict(title="inferred<br>timepoint"),
            showscale=True,
        ),
        customdata=df[HOVER_COLUMNS], hovertemplate=hovertemplate(),
        showlegend=False,
    ))

    if selected_name:
        h = df[df["name"] == selected_name]
        if not h.empty:
            fig.add_trace(make_trace(
                h["PHATE_0"], h["PHATE_1"],
                dict(color="black", size=16, opacity=1.0, symbol="circle-open",
                     line=dict(width=3, color="black")),
                h[HOVER_COLUMNS], f"{selected_name} (selected)"))
    _finalize(fig)
    return fig


def _finalize(fig: go.Figure) -> None:
    fig.update_layout(
        hovermode="closest", showlegend=False, width=900, height=750,
        xaxis_title="PHATE_0", yaxis_title="PHATE_1",
    )
    if st.session_state.get("dino_zoom_x") and st.session_state.get("dino_zoom_y"):
        fig.update_layout(
            xaxis=dict(range=st.session_state.dino_zoom_x),
            yaxis=dict(range=st.session_state.dino_zoom_y),
        )


# =====================
# SESSION STATE


def init_session_state():
    defaults = {
        "dino_selected_cluster": None,
        "dino_selected_name": None,
        "dino_zoom_x": None,
        "dino_zoom_y": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_selection():
    st.session_state.dino_selected_cluster = None
    st.session_state.dino_selected_name = None
    st.session_state.dino_zoom_x = None
    st.session_state.dino_zoom_y = None


# =====================
# MAIN APP

init_session_state()

st.title("DINO Embedding Analysis")
st.caption(
    "PHATE + Leiden of the 384-dim DINOv2 (CS231n) per-cell embeddings, averaged "
    "to per-guide / per-gene centroids. Switch normalization, clustering "
    "resolution, and color by cluster or inferred timepoint."
)

if not EMBEDDING_OUTPUT_PATH:
    st.error(
        "EMBEDDING_OUTPUT_PATH environment variable is not set. "
        "Add it to 14.run_visualization.sh and restart the server."
    )
    st.stop()

# ---- Sidebar controls ----
with st.sidebar:
    st.markdown("## DINO Controls")

    norm_variant = st.radio(
        "**Normalization**",
        list(NORM_VARIANTS.keys()),
        key="dino_norm_variant",
        on_change=reset_selection,
    )

    level = st.radio(
        "**Aggregation Level**",
        ["Per Gene", "Per Guide"],
        key="dino_level",
        on_change=reset_selection,
    )

    leiden_res = st.radio(
        "**Leiden Resolution**",
        LEIDEN_RESOLUTIONS,
        key="dino_leiden_res",
        on_change=reset_selection,
    )

    color_mode = st.radio(
        "**Color by**",
        ["Cluster", "Inferred timepoint"],
        key="dino_color_mode",
    )

# ---- Load data ----
name_col = get_name_col(level)
variant_suffix = NORM_VARIANTS[norm_variant]
level_prefix = "gene" if level == "Per Gene" else "guide"
current_tsv = tsv_path(level_prefix, variant_suffix)
raw = load_tsv(current_tsv)

if raw.empty:
    st.warning(
        f"No data found at `{current_tsv}`. "
        "Run scripts/run_phate_leiden.py to generate the TSV files."
    )
    st.stop()

leiden_col = f"leiden_{leiden_res}"
if leiden_col not in raw.columns:
    st.warning(
        f"Column `{leiden_col}` not found. "
        f"Available: {[c for c in raw.columns if c.startswith('leiden_')]}"
    )
    st.stop()

df = prepare_df(raw, name_col, leiden_res)

# ---- Sidebar: name / cluster search ----
with st.sidebar:
    all_names = sorted(df["name"].unique().tolist())
    search_options = ["— none —"] + all_names
    current_name = st.session_state.dino_selected_name
    search_idx = search_options.index(current_name) if current_name in search_options else 0

    def on_name_search():
        chosen = st.session_state.dino_name_search
        if chosen == "— none —":
            st.session_state.dino_selected_name = None
            st.session_state.dino_selected_cluster = None
        else:
            st.session_state.dino_selected_name = chosen
            row = df[df["name"] == chosen]
            if not row.empty:
                st.session_state.dino_selected_cluster = str(row["cluster"].iloc[0])

    st.selectbox(
        "**Search gene / guide**", search_options, index=search_idx,
        key="dino_name_search", on_change=on_name_search,
    )

    cluster_options = ["— none —"] + sorted(
        df["cluster"].unique(), key=lambda c: int(c) if c.isdigit() else -1)
    cur_cluster = st.session_state.dino_selected_cluster
    cluster_idx = cluster_options.index(cur_cluster) if cur_cluster in cluster_options else 0

    def on_cluster_select():
        chosen = st.session_state.dino_cluster_dropdown
        if chosen == "— none —":
            st.session_state.dino_selected_cluster = None
            st.session_state.dino_selected_name = None
        else:
            st.session_state.dino_selected_cluster = chosen
            names_in = df[df["cluster"] == chosen]["name"].tolist()
            if names_in:
                st.session_state.dino_selected_name = names_in[0]

    st.selectbox(
        "**Select cluster**", cluster_options, index=cluster_idx,
        key="dino_cluster_dropdown", on_change=on_cluster_select,
    )

    if st.button("Clear selection"):
        reset_selection()
        st.rerun()

# ---- Summary stats ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Perturbations", len(df))
col2.metric("Clusters", df["cluster"].nunique())
col3.metric("Controls", int(df["is_control"].sum()))
col4.metric("Resolution", leiden_res)

# ---- PHATE plot ----
st.checkbox(
    "Vectorize points for SVG export (slower render; makes points selectable in Illustrator)",
    key="dino_vector_export",
)
if color_mode == "Inferred timepoint":
    fig = build_timepoint_figure(df)
else:
    fig = build_cluster_figure(df)
event = st.plotly_chart(fig, use_container_width=True, key="dino_plot", on_select="rerun", config={"toImageButtonOptions": {"format": "svg"}})

# Handle click events
if event.selection and event.selection.points:
    pt = event.selection.points[0]
    if "customdata" in pt and len(pt["customdata"]) > CLUSTER_IDX:
        clicked_cluster = str(pt["customdata"][CLUSTER_IDX])
        clicked_name = str(pt["customdata"][NAME_IDX])
        if (
            st.session_state.dino_selected_cluster != clicked_cluster
            or st.session_state.dino_selected_name != clicked_name
        ):
            st.session_state.dino_selected_cluster = clicked_cluster
            st.session_state.dino_selected_name = clicked_name
            st.rerun()

# ---- Details panel ----
selected_cluster = st.session_state.dino_selected_cluster
if selected_cluster is not None:
    cluster_df = df[df["cluster"] == selected_cluster].copy()
    cluster_df = cluster_df.sort_values("cell_count", ascending=False)
    pct_ctrl = 100 * cluster_df["is_control"].mean()

    st.markdown(
        f"### Cluster {selected_cluster} — {len(cluster_df)} perturbations "
        f"({pct_ctrl:.0f}% control)"
    )

    base_cols = ["name", "gene_label", "cell_count", TP_COL, "is_control"] \
        if level == "Per Guide" else ["name", "cell_count", "n_guides", TP_COL, "is_control"]
    display_cols = [c for c in base_cols if c in cluster_df.columns]
    st.dataframe(
        cluster_df[display_cols].set_index("name"),
        use_container_width=True, height=300,
    )
else:
    st.info("Click a point on the plot or use the dropdowns to select a cluster.")
