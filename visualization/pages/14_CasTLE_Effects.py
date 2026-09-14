import os
import sys

import streamlit as st

st.set_page_config(
    page_title="casTLE Effects - Brieflow Analysis",
    layout="wide",
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src import castle as ct

# =====================
# CONFIGURATION

# Directory holding the outputs of the sibling castle_analysis project.
CASTLE_OUTPUT_PATH = ct.CASTLE_OUTPUT_PATH

CELL_CLASS = "all"
CHANNEL_COMBO = "DAPI_YFP_RFP_Cy5"

# Which embedding to read. 'meaningful' clusters on the ranked, redundancy-pruned
# features; 'all' uses every FDR-passing feature via 188 PCs, matching
# config.aggregate.variance_or_ncomp so it is comparable to the existing
# PCA-of-medians clustering. Which is better is an empirical question -- see the
# CORUM/KEGG benchmark in castle_analysis/README.md section 10.
FEATURE_SETS = {
    "Meaningful features": "meaningful",
    "All significant features (188 PCs)": "all",
}

# Depth of every feature dropdown, ranked by discriminability rather than
# alphabetically: with ~1474 scored features an alphabetical menu buries the
# useful ones.
#
# "All scored features" is the default and is the only option that can reach every
# feature. The narrower ones are conveniences, not a filter you should have to
# defeat: a feature is left out of the meaningful set either for having few
# significant genes (cell_DAPI_int) or for being redundant with a higher-scoring
# sibling in its |r| > 0.9 group (cell_RFP_int is meaningful but only rank 293, so
# a Top 50 menu hides it), and neither is a reason to be unable to plot it.
MENU_DEPTHS = {
    "Top 25": 25,
    "Top 50": 50,
    "Top 100": 100,
    "All meaningful": None,
    "All FDR-significant": "significant",
    "All scored features": "all_scored",
}
DEFAULT_MENU_DEPTH = "All scored features"

LEIDEN_RESOLUTIONS = [1, 3, 5, 7, 9, 11, 13, 15, 20, 50]

HOVER_COLUMNS = ["name", "cluster", "n_guides", "is_control", "_color_value"]
NAME_IDX = 0
CLUSTER_IDX = 1
N_GUIDES_IDX = 2
IS_CONTROL_IDX = 3
COLOR_IDX = 4

SENTINEL = "— none —"
CLUSTER_DEFAULT = "Cluster (default)"


# =====================
# COLOR PALETTE


def turbo_palette(n: int) -> list[str]:
    cmap = plt.get_cmap("turbo")
    return [mcolors.rgb2hex(cmap(i / max(n - 1, 1))) for i in range(n)]


# =====================
# SCATTER PLOT


def hovertemplate(color_label: str | None) -> str:
    base = (
        "PHATE_0=%{x:.3f}<br>"
        "PHATE_1=%{y:.3f}<br>"
        f"gene=%{{customdata[{NAME_IDX}]}}<br>"
        f"cluster=%{{customdata[{CLUSTER_IDX}]}}<br>"
        f"guides=%{{customdata[{N_GUIDES_IDX}]}}<br>"
        f"control=%{{customdata[{IS_CONTROL_IDX}]}}<br>"
    )
    if color_label:
        base += f"{color_label}=%{{customdata[{COLOR_IDX}]:.3f}}<br>"
    return base + "<extra></extra>"


def _trace_cls():
    # WebGL (Scattergl) is fast but rasterizes points on SVG export. When the user
    # enables vector export, use go.Scatter so each point exports as an
    # individually selectable vector shape.
    return (
        go.Scatter
        if st.session_state.get("castle_vector_export", False)
        else go.Scattergl
    )


def make_trace(x, y, marker, customdata, name: str, color_label=None):
    return _trace_cls()(
        x=x,
        y=y,
        mode="markers",
        marker=marker,
        customdata=customdata,
        name=name,
        hovertemplate=hovertemplate(color_label),
        showlegend=False,
    )


def build_cluster_figure(df: pd.DataFrame) -> go.Figure:
    clusters = sorted(
        df["cluster"].unique(), key=lambda c: int(c) if c.isdigit() else -1
    )
    palette = turbo_palette(len(clusters))
    color_map = {c: palette[i] for i, c in enumerate(clusters)}

    selected_item = st.session_state.get("castle_selected_cluster", None)
    selected_name = st.session_state.get("castle_selected_name", None)

    fig = go.Figure()

    if selected_item is not None:
        unselected = df[df["cluster"] != selected_item]
        if not unselected.empty:
            fig.add_trace(
                make_trace(
                    unselected["PHATE_0"],
                    unselected["PHATE_1"],
                    dict(color="gray", size=7, opacity=0.25),
                    unselected[HOVER_COLUMNS],
                    "unselected",
                )
            )

        sel_df = df[df["cluster"] == selected_item]
        cluster_color = color_map[selected_item]
        highlight = (
            sel_df[sel_df["name"] == selected_name] if selected_name else pd.DataFrame()
        )
        rest = sel_df[sel_df["name"] != selected_name] if selected_name else sel_df

        if not rest.empty:
            fig.add_trace(
                make_trace(
                    rest["PHATE_0"],
                    rest["PHATE_1"],
                    dict(
                        color=cluster_color,
                        size=10,
                        opacity=1.0,
                        line=dict(width=2, color="black"),
                    ),
                    rest[HOVER_COLUMNS],
                    selected_item,
                )
            )
        if not highlight.empty:
            fig.add_trace(
                make_trace(
                    highlight["PHATE_0"],
                    highlight["PHATE_1"],
                    dict(
                        color=cluster_color,
                        size=15,
                        opacity=1.0,
                        symbol="circle",
                        line=dict(width=3, color="white"),
                    ),
                    highlight[HOVER_COLUMNS],
                    f"{selected_name} (selected)",
                )
            )
    else:
        for cluster in clusters:
            cdf = df[df["cluster"] == cluster]
            fig.add_trace(
                make_trace(
                    cdf["PHATE_0"],
                    cdf["PHATE_1"],
                    dict(color=color_map[cluster], size=8, opacity=0.9),
                    cdf[HOVER_COLUMNS],
                    cluster,
                )
            )

    _finalize(fig)
    return fig


def build_continuous_figure(df: pd.DataFrame, label: str, signed: bool) -> go.Figure:
    """Colour every gene continuously by a casTLE quantity.

    Ported from ``Cluster_Analysis.py``'s ``feature_mode`` branch, with one
    substantive change: casTLE effects are **signed**, so a signed metric gets a
    diverging scale forced symmetric about zero. On Viridis a zero-effect gene
    would read as "low" rather than "neutral", which inverts the interpretation.
    """
    values = df["_color_value"]
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()

    if len(finite):
        cmin, cmax = float(finite.quantile(0.02)), float(finite.quantile(0.98))
        if cmin == cmax:
            cmin, cmax = cmin - 0.5, cmax + 0.5
    else:
        cmin, cmax = -3.0, 3.0

    if signed:
        bound = max(abs(cmin), abs(cmax))
        cmin, cmax = -bound, bound
        colorscale = ct.SIGNED_COLORSCALE
    else:
        colorscale = ct.UNSIGNED_COLORSCALE

    fig = go.Figure()
    fig.add_trace(
        _trace_cls()(
            x=df["PHATE_0"],
            y=df["PHATE_1"],
            mode="markers",
            marker=dict(
                color=values,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                size=8,
                opacity=0.9,
                colorbar=dict(title=label.replace(" ", "<br>"), thickness=15),
                showscale=True,
            ),
            customdata=df[HOVER_COLUMNS],
            hovertemplate=hovertemplate(label),
            showlegend=False,
        )
    )

    selected_name = st.session_state.get("castle_selected_name", None)
    if selected_name:
        h = df[df["name"] == selected_name]
        if not h.empty:
            fig.add_trace(
                make_trace(
                    h["PHATE_0"],
                    h["PHATE_1"],
                    dict(
                        color="black",
                        size=16,
                        opacity=1.0,
                        symbol="circle-open",
                        line=dict(width=3, color="black"),
                    ),
                    h[HOVER_COLUMNS],
                    f"{selected_name} (selected)",
                    color_label=label,
                )
            )
    _finalize(fig)
    return fig


def _finalize(fig: go.Figure) -> None:
    fig.update_layout(
        hovermode="closest",
        showlegend=False,
        width=900,
        height=750,
        xaxis_title="PHATE_0",
        yaxis_title="PHATE_1",
    )
    if st.session_state.get("castle_zoom_x") and st.session_state.get("castle_zoom_y"):
        fig.update_layout(
            xaxis=dict(range=st.session_state.castle_zoom_x),
            yaxis=dict(range=st.session_state.castle_zoom_y),
        )


def build_cluster_heatmap(long, genes, features, effect_column) -> go.Figure | None:
    """Gene x feature effect heatmap for a Leiden cluster.

    Written as a plotly ``go.Heatmap`` rather than reusing
    ``cluster_analysis.cluster_heatmap``: that one requires a literal ``cluster``
    column, returns a seaborn ClusterGrid *or None*, and would drag the Style-A
    ``workflow.lib`` import into this standalone page.
    """
    subset = long.loc[long[ct.GENE_COL].isin(genes) & long["feature"].isin(features)]
    if subset.empty:
        return None

    wide = subset.pivot_table(
        index=ct.GENE_COL, columns="feature", values=effect_column, aggfunc="first"
    ).reindex(columns=[f for f in features if f in set(subset["feature"])])
    if wide.empty:
        return None

    bound = float(np.nanmax(np.abs(wide.to_numpy()))) or 1.0
    fig = go.Figure(
        go.Heatmap(
            z=wide.to_numpy(),
            x=list(wide.columns),
            y=list(wide.index),
            colorscale=ct.SIGNED_COLORSCALE,
            zmid=0,
            zmin=-bound,
            zmax=bound,
            colorbar=dict(title="effect", thickness=15),
            hovertemplate="%{y}<br>%{x}<br>effect=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(320, 22 * len(wide.index) + 160),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(t=40, b=160, l=140, r=20),
    )
    return fig


# =====================
# SESSION STATE


def init_session_state():
    defaults = {
        "castle_selected_cluster": None,
        "castle_selected_name": None,
        "castle_zoom_x": None,
        "castle_zoom_y": None,
        "castle_vector_export": False,
        "castle_color_by": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_selection():
    st.session_state.castle_selected_cluster = None
    st.session_state.castle_selected_name = None
    st.session_state.castle_zoom_x = None
    st.session_state.castle_zoom_y = None


# =====================
# MAIN APP

init_session_state()

st.title("casTLE Per-Feature Gene Effects")
st.caption(
    "casTLE mixture-model effect sizes per gene per CellProfiler feature, so "
    "inactive and off-target sgRNAs are modelled rather than averaged in. "
    "Effect is the *maximal* guide effect (the hit component is uniform on "
    "[0, I]), not a mean — larger in magnitude than a gene median z-score and "
    "not interchangeable with one."
)

if not CASTLE_OUTPUT_PATH:
    st.error(
        "CASTLE_OUTPUT_PATH environment variable is not set. "
        "Add it to 14.run_visualization.sh and restart the server."
    )
    st.stop()

long = ct.load_castle_long(CELL_CLASS, CHANNEL_COMBO)
if long.empty:
    st.warning(
        f"No casTLE results found under `{CASTLE_OUTPUT_PATH}`. Run "
        "`castle_analysis/scripts/run_castle.sbatch`, then `rank_features.py` "
        "and `run_phate_leiden.py`."
    )
    st.stop()

ranking = ct.load_feature_ranking(CELL_CLASS, CHANNEL_COMBO)
metadata = ct.load_run_metadata(CELL_CLASS, CHANNEL_COMBO)

# ---- Sidebar controls ----
with st.sidebar:
    st.markdown("## casTLE Controls")

    feature_set_label = st.radio(
        "**Clustered on**",
        list(FEATURE_SETS.keys()),
        key="castle_feature_set",
        on_change=reset_selection,
    )
    feature_set = FEATURE_SETS[feature_set_label]

    effect_label = st.radio(
        "**Effect variant**",
        ["Shrunk (spike-and-slab)", "MLE (casTLE Effect)"],
        key="castle_effect_variant",
        help=(
            "The shrunk effect collapses no-evidence genes toward zero and is the "
            "clustering input. The MLE is the reference's casTLE Effect, and is "
            "noisier for weak genes."
        ),
    )
    effect_column = (
        "castle_effect_shrunk"
        if effect_label.startswith("Shrunk")
        else "castle_effect"
    )

    leiden_res = st.radio(
        "**Leiden resolution**",
        LEIDEN_RESOLUTIONS,
        index=LEIDEN_RESOLUTIONS.index(9),
        key="castle_leiden_res",
        on_change=reset_selection,
    )

    menu_depth_label = st.radio(
        "**Feature menu depth**",
        list(MENU_DEPTHS.keys()),
        index=list(MENU_DEPTHS.keys()).index(DEFAULT_MENU_DEPTH),
        key="castle_menu_depth",
        help=(
            "Features are ranked by discriminability, not alphabetically. The "
            "default reaches every scored feature; the narrower options put the "
            "most discriminating ones at the top of a shorter list."
        ),
    )

    hide_controls = st.toggle(
        "Hide pseudo-gene controls in plots", value=True, key="castle_hide_controls"
    )

# ---- Feature menu ----
# Every scored feature, curated representatives first. Computed unconditionally
# because the per-gene detail tab offers the full list regardless of menu depth --
# that tab is about one gene, and which feature you want to inspect there has
# nothing to do with how the feature ranks across the whole screen.
all_scored = ct.all_scored_features(long, ranking)

depth = MENU_DEPTHS[menu_depth_label]
if depth == "all_scored":
    menu_features = all_scored
    menu_note = f"all {len(menu_features)} scored features, most discriminating first"
elif depth == "significant":
    menu_features = ct.significant_features(long)
    menu_note = f"{len(menu_features)} features with any gene at FDR<0.05"
else:
    menu_features = ct.meaningful_features(ranking, depth)
    menu_note = (
        f"{len(menu_features)} meaningful features by discriminability"
        if menu_features
        else "no feature ranking found"
    )

if not menu_features:
    menu_features = all_scored
    menu_note = (
        f"all {len(menu_features)} scored features "
        "(no ranking table — run rank_features.py)"
    )

# The cluster heatmap keeps the curated representatives whatever the menu depth is:
# its top 40 by raw discriminability would otherwise be 40 flavours of the same
# zernike moment, which is exactly what the redundancy grouping exists to collapse.
heat_features = ct.meaningful_features(ranking, 40) or menu_features[:40]

if metadata is not None and not metadata.empty:
    method = metadata.iloc[0].get("background_method", "?")
    st.caption(
        f"Background: **{method}**"
        + (
            "  ⚠️ the variance-scaled background is an approximation for "
            "development; re-run with `--background resampled` before quoting "
            "these numbers."
            if method == "variance_scaled"
            else ""
        )
        + f"  ·  {int(metadata.iloc[0].get('n_features', 0))} features scored"
        + f"  ·  menu: {menu_note}"
    )

embedding = ct.load_embedding(CELL_CLASS, CHANNEL_COMBO, feature_set)

# ---- Embedding frame, built once in the prelude ----
# This lives outside the tab bodies because the sidebar gene/cluster search needs
# it, and the sidebar cannot be written to from inside a fragment (see the
# RENDERING note below).
leiden_col = f"leiden_{leiden_res}"
embedding_problem = None
if embedding.empty:
    embedding_problem = (
        f"No embedding at `castle_gene_phate_leiden_{feature_set}.tsv`. "
        "Run `castle_analysis/scripts/run_phate_leiden.py`."
    )
    df = pd.DataFrame(columns=["name", "cluster", "is_control"])
elif leiden_col not in embedding.columns:
    embedding_problem = (
        f"Column `{leiden_col}` not found. Available: "
        f"{[c for c in embedding.columns if c.startswith('leiden_')]}"
    )
    df = pd.DataFrame(columns=["name", "cluster", "is_control"])
else:
    df = embedding.copy()
    df["name"] = df["gene_label"].astype(str)
    df["cluster"] = df[leiden_col].astype(str)
    df["is_control"] = df["is_control"].astype(bool)

# ---- Sidebar gene / cluster search ----
# Kept in the prelude rather than the landscape fragment: Streamlit raises
# "Calling `st.sidebar` in a function wrapped with `st.fragment` is not supported"
# (delta_generator.py:451). These are sidebar widgets, so changing them triggers a
# full rerun -- that is fine, it is a deliberate user action, unlike a plot click.
if not df.empty:
    with st.sidebar:
        all_names = sorted(df["name"].unique().tolist())
        search_options = [SENTINEL] + all_names
        current_name = st.session_state.castle_selected_name
        search_idx = (
            search_options.index(current_name)
            if current_name in search_options
            else 0
        )

        def on_name_search():
            chosen = st.session_state.castle_name_search
            if chosen == SENTINEL:
                st.session_state.castle_selected_name = None
                st.session_state.castle_selected_cluster = None
            else:
                st.session_state.castle_selected_name = chosen
                row = df[df["name"] == chosen]
                if not row.empty:
                    st.session_state.castle_selected_cluster = str(
                        row["cluster"].iloc[0]
                    )

        st.selectbox(
            "**Search gene**",
            search_options,
            index=search_idx,
            key="castle_name_search",
            on_change=on_name_search,
        )

        cluster_options = [SENTINEL] + sorted(
            df["cluster"].unique(), key=lambda c: int(c) if c.isdigit() else -1
        )
        cur_cluster = st.session_state.castle_selected_cluster
        cluster_idx = (
            cluster_options.index(cur_cluster)
            if cur_cluster in cluster_options
            else 0
        )

        def on_cluster_select():
            chosen = st.session_state.castle_cluster_dropdown
            if chosen == SENTINEL:
                st.session_state.castle_selected_cluster = None
                st.session_state.castle_selected_name = None
            else:
                st.session_state.castle_selected_cluster = chosen
                names_in = df[df["cluster"] == chosen]["name"].tolist()
                if names_in:
                    st.session_state.castle_selected_name = names_in[0]

        st.selectbox(
            "**Select cluster**",
            cluster_options,
            index=cluster_idx,
            key="castle_cluster_dropdown",
            on_change=on_cluster_select,
        )

        if st.button("Clear selection"):
            reset_selection()
            st.rerun()

tab_landscape, tab_volcano, tab_scatter, tab_gene, tab_features = st.tabs(
    ["Landscape", "Volcano", "Scatter", "Per-gene detail", "Feature ranking"]
)


# =====================
# RENDERING
#
# Every tab body is an @st.fragment. This is load-bearing, not tidiness:
#
# 1. ``st.tabs`` is client-side only -- Streamlit executes *all five* tab bodies on
#    every script run, whichever tab is open. Measured on this screen's outputs, a
#    full run costs 3.3s with labels off and 8.7s with the volcano/scatter label
#    toggles on (two 3-second ``adjust_text`` budgets in src/labels.py). The
#    landscape's own work is 0.2s of that.
# 2. Clicking a point took *two* full runs: one to observe the selection, one to
#    redraw with it. So a click cost 7-17 seconds of recomputing the volcano,
#    scatter, guide-evidence and ranking views, which the click does not affect.
#    That is the freeze.
#
# Inside a fragment, a widget interaction reruns only that fragment, so a landscape
# click no longer touches the other four views. Fragment reruns re-invoke the
# function with the arguments captured at the last full run (runtime/fragment.py
# stores the wrapped call), so everything a fragment needs is passed explicitly
# rather than closed over -- do not read module-level mutable state inside one.
#
# Fragments cannot write to the sidebar or to any container created outside them,
# which is why the gene/cluster search is in the prelude above.


# =====================
# LANDSCAPE


@st.fragment
def render_landscape(
    df, long, menu_features, heat_features, effect_column, leiden_res, problem
):
    if problem:
        st.warning(problem)
        return
    else:

        # ---- Colour-by controls ----
        color_col1, color_col2 = st.columns([1, 1])
        with color_col1:
            color_options = [CLUSTER_DEFAULT] + menu_features
            current = st.session_state.get("castle_color_by") or CLUSTER_DEFAULT
            if current not in color_options:
                current = CLUSTER_DEFAULT
                st.session_state.castle_color_by = None
            chosen_feature = st.selectbox(
                "Color by casTLE value for feature",
                options=color_options,
                index=color_options.index(current),
                key="castle_color_by_select",
            )
            st.session_state.castle_color_by = (
                None if chosen_feature == CLUSTER_DEFAULT else chosen_feature
            )
        with color_col2:
            color_metric_label = st.selectbox(
                "Quantity",
                list(ct.COLOR_METRICS.keys()),
                key="castle_color_metric",
                disabled=st.session_state.castle_color_by is None,
            )

        color_feature = st.session_state.castle_color_by
        if color_feature:
            metric_column, signed = ct.COLOR_METRICS[color_metric_label]
            lookup = ct.effect_lookup(long, color_feature, metric_column)
            df["_color_value"] = df["name"].map(lookup)
        else:
            df["_color_value"] = np.nan

        # ---- Metrics ----
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Genes", int((~df["is_control"]).sum()))
        m2.metric("Clusters", df["cluster"].nunique())
        m3.metric("Pseudo-gene controls", int(df["is_control"].sum()))
        m4.metric("Resolution", leiden_res)

        st.checkbox(
            "Vectorize points for SVG export (slower render; makes points "
            "selectable in Illustrator)",
            key="castle_vector_export",
        )

        if color_feature:
            fig = build_continuous_figure(
                df, f"{color_metric_label}<br>{color_feature}", signed
            )
        else:
            fig = build_cluster_figure(df)

        event = st.plotly_chart(
            fig,
            use_container_width=True,
            key="castle_plot",
            on_select="rerun",
            config={"toImageButtonOptions": {"format": "svg"}},
        )

        if event.selection and event.selection.points:
            pt = event.selection.points[0]
            if "customdata" in pt and len(pt["customdata"]) > CLUSTER_IDX:
                clicked_cluster = str(pt["customdata"][CLUSTER_IDX])
                clicked_name = str(pt["customdata"][NAME_IDX])
                if (
                    st.session_state.castle_selected_cluster != clicked_cluster
                    or st.session_state.castle_selected_name != clicked_name
                ):
                    st.session_state.castle_selected_cluster = clicked_cluster
                    st.session_state.castle_selected_name = clicked_name
                    # Point the per-gene detail tab at the clicked gene. That tab is
                    # its own fragment, so it will not re-render on this click; it
                    # picks this up the next time it runs. Only real genes are
                    # written -- its menu excludes pseudo-gene controls, and
                    # Streamlit raises if a selectbox's session-state value is not
                    # among its options.
                    # Read is_control from the frame rather than the round-tripped
                    # customdata, whose JSON type is not worth relying on.
                    clicked_row = df.loc[df["name"] == clicked_name]
                    if not clicked_row.empty and not bool(
                        clicked_row["is_control"].iloc[0]
                    ):
                        st.session_state.castle_detail_gene = clicked_name
                    # scope="fragment" is the whole point: a plain st.rerun() would
                    # re-execute the other four tab bodies, which is what made a
                    # click take seconds. The sidebar search boxes will catch up on
                    # the next full run.
                    st.rerun(scope="fragment")

        # ---- Cluster detail + heatmap ----
        selected_cluster = st.session_state.castle_selected_cluster
        if selected_cluster is not None:
            cluster_df = df[df["cluster"] == selected_cluster].copy()
            pct_ctrl = 100 * cluster_df["is_control"].mean()
            st.markdown(
                f"### Cluster {selected_cluster} — {len(cluster_df)} genes "
                f"({pct_ctrl:.0f}% pseudo-gene control)"
            )
            if pct_ctrl > 50:
                st.warning(
                    "This cluster is mostly pseudo-gene controls, which means it is "
                    "grouping genes with no detected effect rather than a shared "
                    "phenotype."
                )
            st.dataframe(
                cluster_df[["name", "n_guides", "is_control"]].set_index("name"),
                use_container_width=True,
                height=260,
            )

            heat = build_cluster_heatmap(
                long,
                cluster_df["name"].tolist(),
                heat_features,
                effect_column,
            )
            if heat is not None:
                st.markdown(
                    f"#### Effect heatmap — cluster {selected_cluster} "
                    f"x top {len(heat_features)} features"
                )
                st.plotly_chart(
                    heat,
                    use_container_width=True,
                    key="castle_heatmap",
                    config={"toImageButtonOptions": {"format": "svg"}},
                )
        else:
            st.info(
                "Click a point on the plot or use the sidebar dropdowns to select a "
                "cluster."
            )


# =====================
# VOLCANO


@st.fragment
def render_volcano(long, menu_features, effect_column, hide_controls):
    st.markdown(
        "#### casTLE volcano  \n"
        "The casTLE analogue of the Feature Explorer volcano, which runs off the "
        "bootstrap z-score and FDR. Same visual grammar, different axes — casTLE "
        "effect on x, and real credible intervals available."
    )
    vcol1, vcol2, vcol3 = st.columns([2, 1, 1])
    with vcol1:
        volcano_feature = st.selectbox(
            "Feature", menu_features, key="castle_volcano_feature"
        )
    with vcol2:
        y_metric = st.selectbox(
            "Y axis", list(ct.VOLCANO_Y.keys()), key="castle_volcano_y"
        )
    with vcol3:
        volcano_labels = st.checkbox("Label genes", key="castle_volcano_labels")

    lcol1, lcol2, lcol3 = st.columns(3)
    with lcol1:
        label_fdr = st.number_input(
            "Label FDR ≤",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            key="castle_volcano_label_fdr",
        )
    with lcol2:
        max_labels = st.number_input(
            "Max labels", 1, 300, 50, key="castle_volcano_max_labels"
        )
    with lcol3:
        error_bars = st.checkbox(
            "Show 95% credible intervals", key="castle_volcano_errors"
        )

    vfig = ct.build_castle_volcano(
        long,
        volcano_feature,
        y_metric=y_metric,
        x_column=effect_column,
        show_labels=volcano_labels,
        label_fdr=label_fdr,
        max_labels=int(max_labels),
        show_error_bars=error_bars,
        hide_controls=hide_controls,
    )
    if vfig is None:
        st.warning(f"No plottable data for `{volcano_feature}`.")
    else:
        st.plotly_chart(
            vfig,
            use_container_width=True,
            key="castle_volcano",
            config={"toImageButtonOptions": {"format": "svg"}},
        )

    sub = long.loc[long["feature"] == volcano_feature]
    if hide_controls:
        sub = sub.loc[~sub["is_control"]]
    n_ex = int(sub["castle_pval_extrapolated"].sum())
    st.caption(
        f"{int(sub['significant'].sum())} of {len(sub)} genes at FDR<0.05.  "
        f"{n_ex} p-values are extrapolated beyond the permutation null's support "
        "— reliable for ranking, not exact tail probabilities."
    )


# =====================
# SCATTER


@st.fragment
def render_scatter(long, menu_features, effect_column, hide_controls):
    st.markdown(
        "#### casTLE effect, feature vs feature  \n"
        "The casTLE analogue of Feature Scatter, with the same four significance "
        "categories and palette so the two can be compared side by side."
    )
    scol1, scol2, scol3 = st.columns([2, 2, 1])
    with scol1:
        feature_x = st.selectbox("X feature", menu_features, key="castle_scatter_x")
    with scol2:
        feature_y = st.selectbox(
            "Y feature",
            menu_features,
            index=min(1, len(menu_features) - 1),
            key="castle_scatter_y",
        )
    with scol3:
        scatter_labels = st.checkbox("Label genes", key="castle_scatter_labels")

    sfig = ct.build_castle_scatter(
        long,
        feature_x,
        feature_y,
        effect_column=effect_column,
        show_labels=scatter_labels,
        hide_controls=hide_controls,
    )
    if sfig is None:
        st.warning("No plottable data for that feature pair.")
    else:
        st.plotly_chart(
            sfig,
            use_container_width=True,
            key="castle_scatter",
            config={"toImageButtonOptions": {"format": "svg"}},
        )


# =====================
# PER-GENE DETAIL


@st.fragment
def render_gene_detail(long, all_features):
    gene_options = sorted(long.loc[~long["is_control"], ct.GENE_COL].unique().tolist())
    default_gene = st.session_state.get("castle_selected_name")
    gene_idx = (
        gene_options.index(default_gene) if default_gene in gene_options else 0
    )
    gene = st.selectbox(
        "Gene", gene_options, index=gene_idx, key="castle_detail_gene"
    )
    st.caption(
        "Follows the gene clicked on the Landscape tab. Because each tab renders "
        "independently, a landscape click lands here the next time this tab runs "
        "rather than instantly."
    )

    top = ct.top_features_for_gene(long, gene, top_n=25)
    if top.empty:
        st.info(f"No casTLE results for {gene}.")
    else:
        st.markdown(f"#### {gene}: top features by casTLE score")
        display = top[
            [
                "feature",
                "effect ± 95% CI",
                "castle_score",
                "castle_fdr",
                "hit_rate_map",
                "n_guides_used",
            ]
        ].rename(
            columns={
                "castle_score": "score",
                "castle_fdr": "FDR",
                "hit_rate_map": "hit rate",
                "n_guides_used": "guides",
            }
        )
        st.dataframe(
            display.set_index("feature").style.format(
                {"score": "{:.1f}", "FDR": "{:.2e}", "hit rate": "{:.2f}"}
            ),
            use_container_width=True,
            height=300,
        )

        # This gene's top-scoring features first, then every other scored feature.
        # The table above is a top-25 view, but the guide-evidence plot is valid for
        # any feature -- you often want a specific readout (cell_RFP_int, say) for a
        # gene where it is not among the top hits, and that is a useful negative
        # result rather than something to hide. Offering the full list also keeps
        # the option set identical across genes, so the stored selection survives
        # switching gene instead of Streamlit raising on a value not in options.
        detail_options = top["feature"].tolist()
        _in_top = set(detail_options)
        detail_options += [f for f in all_features if f not in _in_top]

        detail_feature = st.selectbox(
            "Feature for the guide-evidence plot",
            detail_options,
            key="castle_detail_feature",
            help=(
                f"This gene's top {len(_in_top)} features by casTLE score come "
                "first; every scored feature follows. Type to search."
            ),
        )

        rhos = ct.load_castle_rhos(CELL_CLASS, CHANNEL_COMBO)
        background = ct.load_castle_background(CELL_CLASS, CHANNEL_COMBO)
        if rhos.empty:
            st.warning(
                "No `__castle_rho_constructs.parquet`; cannot draw guide evidence."
            )
        elif detail_feature not in rhos.columns:
            st.warning(f"`{detail_feature}` not present in the rho table.")
        else:
            fit_rows = long.loc[
                (long[ct.GENE_COL] == gene) & (long["feature"] == detail_feature)
            ]
            # The table above only covers the top 25, so state the fit for the
            # chosen feature explicitly -- otherwise picking one further down the
            # list leaves its score and FDR nowhere on the page.
            if detail_feature not in _in_top and not fit_rows.empty:
                fit = fit_rows.iloc[0]
                # hit_rate_map is NaN when the fit lands on I = 0: there is no hit
                # component to estimate a rate for. Say so rather than printing
                # "nan", which reads like a bug.
                hit_rate = float(fit["hit_rate_map"])
                hit_text = (
                    f"hit rate {hit_rate:.2f}"
                    if np.isfinite(hit_rate)
                    else "no hit component (effect 0)"
                )
                st.caption(
                    f"`{detail_feature}` is outside {gene}'s top "
                    f"{len(_in_top)} — score {float(fit['castle_score']):.1f}, "
                    f"FDR {float(fit['castle_fdr']):.2e}, {hit_text}, "
                    f"{int(fit['n_guides_used'])} guides used."
                )
            gfig = ct.build_guide_evidence(
                gene,
                detail_feature,
                rhos,
                background,
                fit_rows.iloc[0] if not fit_rows.empty else None,
            )
            st.plotly_chart(
                gfig,
                use_container_width=True,
                key="castle_guides",
                config={"toImageButtonOptions": {"format": "svg"}},
            )
            if background.empty:
                st.caption(
                    "No background summary found — shaded control bands are "
                    "omitted. The variance-scaled background produces no samples; "
                    "re-run with `--background resampled` to get them."
                )
            else:
                st.caption(
                    "Shaded grey bands are the interquartile and 95% ranges of "
                    "**control guide medians resampled at matched cell count** — "
                    "not the single-cell control distribution, which is much wider "
                    "and is the wrong comparator for a guide median. A guide with "
                    "few cells sits against a visibly wider band."
                )


# =====================
# FEATURE RANKING


@st.fragment
def render_feature_ranking(ranking):
    if ranking.empty:
        st.warning(
            "No `__castle_feature_ranking.tsv`. Run "
            "`castle_analysis/scripts/rank_features.py`."
        )
    else:
        st.markdown(
            "#### Feature ranking  \n"
            "`discriminability` is the spread of gene effects divided by the spread "
            "of *pseudo-gene* effects for the same feature. Counting significant "
            "genes is not enough on its own: a feature can have many significant "
            "genes while barely separating them, or few while cleanly splitting the "
            "screen. Features are grouped at |r| > 0.9 and one representative is "
            "kept per group, which is what collapses the near-identical zernike and "
            "texture families."
        )
        r1, r2, r3 = st.columns(3)
        r1.metric("Features scored", len(ranking))
        r2.metric("Eligible", int(ranking.get("is_eligible", pd.Series()).sum()))
        r3.metric("Meaningful representatives", int(ranking["is_meaningful"].sum()))

        only_meaningful = st.checkbox(
            "Show only meaningful representatives", value=True, key="castle_rank_filter"
        )
        table = ranking.loc[ranking["is_meaningful"]] if only_meaningful else ranking
        st.dataframe(
            table.sort_values("discriminability", ascending=False)[
                [
                    "feature",
                    "discriminability",
                    "n_sig_genes",
                    "effect_spread",
                    "null_spread",
                    "top_score",
                    "redundancy_group",
                    "is_meaningful",
                ]
            ].set_index("feature"),
            use_container_width=True,
            height=520,
        )


# =====================
# TAB WIRING
#
# Each fragment is called inside its tab so its output lands there. Arguments are
# passed explicitly because a fragment-only rerun replays the call with the
# arguments from the last full run.

with tab_landscape:
    render_landscape(
        df,
        long,
        menu_features,
        heat_features,
        effect_column,
        leiden_res,
        embedding_problem,
    )

with tab_volcano:
    render_volcano(long, menu_features, effect_column, hide_controls)

with tab_scatter:
    render_scatter(long, menu_features, effect_column, hide_controls)

with tab_gene:
    render_gene_detail(long, all_scored)

with tab_features:
    render_feature_ranking(ranking)
