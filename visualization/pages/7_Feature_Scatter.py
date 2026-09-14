import glob
import io
import os
import sys
import uuid

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Feature Scatter - Brieflow Analysis", layout="wide")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import BRIEFLOW_OUTPUT_PATH, STATIC_ASSET_PATH, STATIC_ASSET_URL_ROOT
from src.config import load_config
from src.filesystem import FileSystem
from src.labels import add_point_labels
from src.rendering import render_composite_montage
from src.scrnaseq_aging import (
    SCRNASEQ_AGING_FEATURE,
    attach_scrnaseq_aging,
    has_scrnaseq_aging_data,
)
from src.external_deg import (
    EXTERNAL_DEG_FEATURES,
    attach_external_deg,
    available_external_deg_features,
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

# =====================
# DATA LOADERS


@st.cache_data
def load_significant_features(cell_class, channel_combo, fdr_threshold=0.05):
    bootstrap_path = os.path.join(
        BRIEFLOW_OUTPUT_PATH,
        "aggregate",
        "bootstrap",
        f"CeCl-{cell_class}_ChCo-{channel_combo}__all_gene_bootstrap_results.tsv",
    )
    if not os.path.exists(bootstrap_path):
        return []
    df = pd.read_csv(bootstrap_path, sep="\t")
    fdr_cols = [c for c in df.columns if c.endswith("_fdr")]
    significant = [
        c[:-4] for c in fdr_cols if (df[c] < fdr_threshold).any()
    ]
    return sorted(significant)


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
def load_uniprot_for_gene(gene, leiden_resolution):
    tsv_path = os.path.join(
        CLUSTER_ROOT, CHANNEL_COMBO, CELL_CLASS, leiden_resolution,
        "phate_leiden_clustering.tsv",
    )
    if not os.path.exists(tsv_path):
        pattern = os.path.join(
            CLUSTER_ROOT, CHANNEL_COMBO, CELL_CLASS, "*", "phate_leiden_clustering.tsv"
        )
        matches = glob.glob(pattern)
        if not matches:
            return None
        tsv_path = matches[0]
    df = pd.read_csv(tsv_path, sep="\t")
    rows = df[df["gene_symbol_0"] == gene]
    return rows.iloc[0] if not rows.empty else None


# =====================
# PLOT BUILDER


def build_scatter_df(feature_x, feature_y):
    gene_df = load_gene_table(CELL_CLASS, CHANNEL_COMBO)
    bootstrap_df = load_bootstrap_results(CELL_CLASS, CHANNEL_COMBO)
    if gene_df is None or bootstrap_df is None:
        return None

    # External scRNA-seq aging values behave like a feature column: the macrophage
    # mean age-coef goes in the gene table, the min FDR in the bootstrap table.
    if SCRNASEQ_AGING_FEATURE in (feature_x, feature_y):
        gene_df = attach_scrnaseq_aging(
            gene_df, gene_col="gene_symbol_0", add_fdr=False
        )
        bootstrap_df = attach_scrnaseq_aging(bootstrap_df, gene_col="gene").drop(
            columns=[SCRNASEQ_AGING_FEATURE]
        )

    # External bulk RNA-seq DEG values behave the same way: value in the gene
    # table, '<feature>_fdr' (padj) in the bootstrap table.
    ext_deg_selected = [f for f in (feature_x, feature_y) if f in EXTERNAL_DEG_FEATURES]
    if ext_deg_selected:
        gene_df = attach_external_deg(
            gene_df, gene_col="gene_symbol_0", features=ext_deg_selected, add_fdr=False
        )
        bootstrap_df = attach_external_deg(
            bootstrap_df, gene_col="gene", features=ext_deg_selected
        ).drop(columns=ext_deg_selected)

    for f in [feature_x, feature_y]:
        if f not in gene_df.columns:
            return None
    bootstrap_df = bootstrap_df.rename(columns={"gene": "gene_symbol_0"})
    cols = ["gene_symbol_0", feature_x, feature_y]
    boot_cols = ["gene_symbol_0"]
    for f in [feature_x, feature_y]:
        fdr_col = f"{f}_fdr"
        if fdr_col in bootstrap_df.columns:
            boot_cols.append(fdr_col)
    df = gene_df[cols].merge(bootstrap_df[boot_cols], on="gene_symbol_0", how="inner").dropna()

    sig_x = df.get(f"{feature_x}_fdr", pd.Series([1.0] * len(df))) < 0.05
    sig_y = df.get(f"{feature_y}_fdr", pd.Series([1.0] * len(df))) < 0.05
    df["sig_x"] = sig_x.values if hasattr(sig_x, "values") else sig_x
    df["sig_y"] = sig_y.values if hasattr(sig_y, "values") else sig_y
    df["significant"] = df["sig_x"] | df["sig_y"]
    return df


def build_feature_scatter(
    df, feature_x, feature_y, show_labels=False, label_fdr=0.05, max_labels=50
):
    # Four categories: neither, x only, y only, both
    neither = df[~df["sig_x"] & ~df["sig_y"]]
    x_only  = df[ df["sig_x"] & ~df["sig_y"]]
    y_only  = df[~df["sig_x"] &  df["sig_y"]]
    both    = df[ df["sig_x"] &  df["sig_y"]]

    # Age-coefficients are small numbers; give them more significant digits on hover
    fmt_x = ".4g" if feature_x == SCRNASEQ_AGING_FEATURE else ".3f"
    fmt_y = ".4g" if feature_y == SCRNASEQ_AGING_FEATURE else ".3f"

    def make_trace(subset, name, color, size, opacity=0.6):
        return go.Scatter(
            x=subset[feature_x],
            y=subset[feature_y],
            mode="markers",
            name=name,
            marker=dict(color=color, size=size, opacity=opacity),
            customdata=subset[["gene_symbol_0"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"{feature_x}: %{{x:{fmt_x}}}<br>"
                f"{feature_y}: %{{y:{fmt_y}}}<extra></extra>"
            ),
        )

    fig = go.Figure()
    fig.add_trace(make_trace(neither, "Neither significant",         "#aaaaaa", 7))
    fig.add_trace(make_trace(x_only,  f"Sig in X only",             "#4575b4", 9))
    fig.add_trace(make_trace(y_only,  f"Sig in Y only",             "#d73027", 9))
    fig.add_trace(make_trace(both,    "Sig in both",                "#f1c40f", 11, opacity=0.85))

    # Shared axis ranges so offscreen label placement maps onto the Plotly figure
    x_vals = df[feature_x].dropna()
    y_vals = df[feature_y].dropna()
    xspan = (x_vals.max() - x_vals.min()) or 1
    yspan = (y_vals.max() - y_vals.min()) or 1
    xlim = (x_vals.min() - 0.05 * xspan, x_vals.max() + 0.05 * xspan)
    ylim = (y_vals.min() - 0.05 * yspan, y_vals.max() + 0.05 * yspan)

    # Toggleable labels: every gene at/below the chosen FDR in either feature
    if show_labels:
        fx = df.get(f"{feature_x}_fdr", pd.Series(1.0, index=df.index))
        fy = df.get(f"{feature_y}_fdr", pd.Series(1.0, index=df.index))
        lab = df.assign(_fdr_min=pd.concat([fx, fy], axis=1).min(axis=1))
        lab = lab[lab["_fdr_min"] <= label_fdr]
        if len(lab) > max_labels:
            lab = lab.nsmallest(max_labels, "_fdr_min")
        if not lab.empty:
            add_point_labels(
                fig,
                lab[feature_x].values,
                lab[feature_y].values,
                lab["gene_symbol_0"].astype(str).values,
                df[feature_x].values,
                df[feature_y].values,
                xlim,
                ylim,
            )

    fig.update_layout(
        title=f"{feature_x}  vs  {feature_y}",
        xaxis_title=feature_x,
        yaxis_title=feature_y,
        xaxis=dict(range=list(xlim)),
        yaxis=dict(range=list(ylim)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40, l=40, r=20),
        height=550,
    )
    return fig


# =====================
# GENE INFO DISPLAYS


def display_uniprot_info_fs(gene):
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

    if f"fs_guide_{gene}" not in st.session_state:
        st.session_state[f"fs_guide_{gene}"] = None

    def on_guide_select():
        st.session_state[f"fs_guide_{gene}"] = st.session_state[f"fs_guide_dropdown_{gene}"]

    selected_guide = st.session_state.get(f"fs_guide_{gene}", None)
    selected_index = 0
    if selected_guide in available_guides:
        selected_index = available_guides.index(selected_guide)
    elif available_guides:
        selected_guide = available_guides[0]
        st.session_state[f"fs_guide_{gene}"] = selected_guide

    selected_guide = st.selectbox(
        "Select Guide",
        available_guides,
        index=selected_index,
        key=f"fs_guide_dropdown_{gene}",
        on_change=on_guide_select,
    )

    filtered_montage_data = montage_data[montage_data["guide"] == selected_guide]

    if len(filtered_montage_data) > 0:
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
                overlay_tiff_path,
                key_prefix=f"fs_{gene}_{selected_guide}",
            )
            if STATIC_ASSET_URL_ROOT and STATIC_ASSET_PATH:
                relative_path = overlay_tiff_path.replace(STATIC_ASSET_PATH, "")
                static_url = f"{STATIC_ASSET_URL_ROOT}{relative_path}"
                st.markdown(f"[Download Overlay TIFF]({static_url})")
            else:
                with open(overlay_tiff_path, "rb") as f:
                    st.download_button(
                        label="Download Overlay TIFF",
                        data=f,
                        file_name=f"{gene}_{selected_guide}_overlay.tiff",
                        key=f"fs_download_{gene}_{selected_guide}_{uuid.uuid4()}",
                    )
        else:
            st.warning(f"No overlay tiff found: {overlay_tiff_path}")
    else:
        st.warning(f"No image found for {gene} - {selected_guide}")


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
        key=f"fs_rep_feature_{gene}_{cell_class}_{channel_combo}",
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
    _svg_buf = io.StringIO()
    fig.savefig(_svg_buf, format="svg", bbox_inches="tight")
    st.download_button(
        label="Download plot as SVG (vector, editable in Illustrator)",
        data=_svg_buf.getvalue(),
        file_name=f"{gene}_{selected_feature}.svg",
        mime="image/svg+xml",
        key=f"svg_dl_{uuid.uuid4()}",
    )


# =====================
# SESSION STATE INIT


def init_state():
    if "fs_feature_x" not in st.session_state:
        st.session_state.fs_feature_x = None
    if "fs_feature_y" not in st.session_state:
        st.session_state.fs_feature_y = None
    if "fs_selected_gene" not in st.session_state:
        st.session_state.fs_selected_gene = None
    if "hide_controls" not in st.session_state:
        st.session_state.hide_controls = False


init_state()

# =====================
# PAGE HEADER

title_col, reset_col = st.columns([4, 1])
with title_col:
    st.title("Feature Scatter")
with reset_col:
    st.write("")
    if st.button("Reset All", type="secondary"):
        st.session_state.fs_feature_x = None
        st.session_state.fs_feature_y = None
        st.session_state.fs_selected_gene = None
        st.rerun()

# =====================
# FEATURE SELECTORS

_control_key = load_config().get("aggregate", {}).get("control_key", "0Safe")
st.sidebar.title("Options")
st.sidebar.toggle(f"Hide controls ({_control_key})", key="hide_controls")

sig_features = load_significant_features(CELL_CLASS, CHANNEL_COMBO)
if not sig_features:
    st.warning(
        f"No significant features found for {CELL_CLASS} / {CHANNEL_COMBO}. "
        "Check that the bootstrap results file exists."
    )
    st.stop()

feature_options = ["Select a feature..."] + sig_features
# External scRNA-seq aging values (macrophage mean age-coef, sig. by macrophage min FDR)
if has_scrnaseq_aging_data():
    feature_options.append(SCRNASEQ_AGING_FEATURE)
# External bulk RNA-seq DEG values (ADCP timepoints, Effero, Super-Effero)
feature_options += available_external_deg_features()

sel_col_x, sel_col_y = st.columns(2)

with sel_col_x:
    cur_x = st.session_state.fs_feature_x
    idx_x = feature_options.index(cur_x) if cur_x in feature_options else 0
    label_x = st.selectbox(
        f"X-axis feature ({CHANNEL_COMBO})",
        options=feature_options,
        index=idx_x,
        key="fs_feature_x_select",
    )

with sel_col_y:
    cur_y = st.session_state.fs_feature_y
    idx_y = feature_options.index(cur_y) if cur_y in feature_options else 0
    label_y = st.selectbox(
        f"Y-axis feature ({CHANNEL_COMBO})",
        options=feature_options,
        index=idx_y,
        key="fs_feature_y_select",
    )

# Clear gene when either feature changes
if label_x != st.session_state.fs_feature_x or label_y != st.session_state.fs_feature_y:
    st.session_state.fs_selected_gene = None

st.session_state.fs_feature_x = label_x if label_x != "Select a feature..." else None
st.session_state.fs_feature_y = label_y if label_y != "Select a feature..." else None

feature_x = st.session_state.fs_feature_x
feature_y = st.session_state.fs_feature_y

if not feature_x or not feature_y:
    st.info("Select both an X and Y feature above to display the scatter plot.")
    st.stop()

if feature_x == feature_y:
    st.warning("X and Y features must be different.")
    st.stop()

# =====================
# SCATTER PLOT

df = build_scatter_df(feature_x, feature_y)
if df is None or df.empty:
    st.warning(f"Could not load data for the selected features.")
    st.stop()

if st.session_state.hide_controls:
    df = df[~df["gene_symbol_0"].str.startswith(_control_key)].reset_index(drop=True)

n_both = (df["sig_x"] & df["sig_y"]).sum()
n_either = df["significant"].sum()
st.caption(
    f"{len(df)} genes · {n_either} significant in at least one feature "
    f"({n_both} in both) · Click a point to select a gene"
)

lab_col, fdr_col_ctrl, _spacer = st.columns([1, 2, 3])
with lab_col:
    show_labels = st.toggle("Label genes", value=False, key="fs_scatter_labels")
with fdr_col_ctrl:
    label_fdr = st.slider(
        "Label genes with FDR ≤ (in either feature)",
        min_value=0.001,
        max_value=0.25,
        value=0.05,
        step=0.001,
        format="%.3f",
        key="fs_scatter_label_fdr",
        disabled=not show_labels,
    )

scatter_fig = build_feature_scatter(
    df, feature_x, feature_y, show_labels=show_labels, label_fdr=label_fdr
)
scatter_event = st.plotly_chart(
    scatter_fig,
    on_select="rerun",
    key="fs_scatter",
    use_container_width=True,
    config={"toImageButtonOptions": {"format": "svg"}},
)

if (
    scatter_event
    and hasattr(scatter_event, "selection")
    and scatter_event.selection
    and scatter_event.selection.points
):
    clicked_gene = scatter_event.selection.points[0]["customdata"][0]
    if clicked_gene != st.session_state.fs_selected_gene:
        st.session_state.fs_selected_gene = clicked_gene
        st.rerun()

# =====================
# GENE INFO PANEL

selected_gene = st.session_state.fs_selected_gene
if not selected_gene:
    st.stop()

st.divider()

gene_title_col, clear_col = st.columns([4, 1])
with gene_title_col:
    st.markdown(f"## Gene: {selected_gene}")
with clear_col:
    st.write("")
    if st.button("Clear Gene", type="secondary"):
        st.session_state.fs_selected_gene = None
        st.rerun()

# UniProt
st.markdown("#### UniProt Summary")
display_uniprot_info_fs(selected_gene)

# Montages
gene_montages_root = os.path.join(
    BRIEFLOW_OUTPUT_PATH, "aggregate", "montages", f"{CELL_CLASS}__montages"
)
if os.path.exists(gene_montages_root):
    st.markdown("#### Gene Montages")
    display_gene_montages(gene_montages_root, selected_gene)

# sgRNA assessment
display_sgrna_representativeness(selected_gene, CELL_CLASS, CHANNEL_COMBO)
