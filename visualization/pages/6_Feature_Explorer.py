import glob
import io
import math
import os
import sys
import uuid

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Feature Explorer - Brieflow Analysis", layout="wide")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import BRIEFLOW_OUTPUT_PATH, STATIC_ASSET_PATH, STATIC_ASSET_URL_ROOT
from src.config import load_config
from src.filesystem import FileSystem
from src.labels import add_point_labels
from src.rendering import render_composite_montage
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
        # Fall back to any available resolution
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
# PLOT BUILDERS


def build_merged_df(feature):
    gene_df = load_gene_table(CELL_CLASS, CHANNEL_COMBO)
    bootstrap_df = load_bootstrap_results(CELL_CLASS, CHANNEL_COMBO)
    if gene_df is None or bootstrap_df is None:
        return None
    if feature not in gene_df.columns:
        return None
    fdr_col = f"{feature}_fdr"
    log10_col = f"{feature}_log10"
    if fdr_col not in bootstrap_df.columns or log10_col not in bootstrap_df.columns:
        return None
    bootstrap_df = bootstrap_df.rename(columns={"gene": "gene_symbol_0"})
    merged = gene_df[["gene_symbol_0", feature]].merge(
        bootstrap_df[["gene_symbol_0", log10_col, fdr_col]],
        on="gene_symbol_0",
        how="inner",
    ).dropna()
    merged["significant"] = merged[fdr_col] < 0.05
    return merged


def build_volcano_plot(merged, feature, show_labels=False, label_fdr=0.05, max_labels=50):
    fdr_col = f"{feature}_fdr"
    log10_col = f"{feature}_log10"

    bg = merged[~merged["significant"]]
    sig = merged[merged["significant"]]

    def make_trace(df, name, color, size):
        return go.Scatter(
            x=df[feature],
            y=df[log10_col],
            mode="markers",
            name=name,
            marker=dict(color=color, size=size, opacity=0.6),
            customdata=df[["gene_symbol_0"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"z-score: %{{x:.3f}}<br>"
                f"-log10(FDR): %{{y:.3f}}<extra></extra>"
            ),
        )

    fig = go.Figure()
    fig.add_trace(make_trace(bg, "Background", "#aaaaaa", 7))
    fig.add_trace(make_trace(sig, "Significant (FDR<0.05)", "#756bb1", 10))

    # Shared axis ranges so offscreen label placement maps onto the Plotly figure
    x_vals = merged[feature].dropna()
    y_vals = merged[log10_col].dropna()
    xspan = (x_vals.max() - x_vals.min()) or 1
    yspan = (y_vals.max() - y_vals.min()) or 1
    xlim = (x_vals.min() - 0.05 * xspan, x_vals.max() + 0.05 * xspan)
    ylim = (y_vals.min() - 0.05 * yspan, y_vals.max() + 0.05 * yspan)

    # Threshold lines
    threshold_y = -math.log10(0.05)
    fig.add_hline(y=threshold_y, line_dash="dash", line_color="#888888", line_width=1)
    if len(x_vals) > 0:
        fig.add_vline(x=2, line_dash="dash", line_color="#888888", line_width=1)
        fig.add_vline(x=-2, line_dash="dash", line_color="#888888", line_width=1)

    # Toggleable gene labels for every gene at/below the chosen FDR threshold
    if show_labels and fdr_col in merged.columns:
        lab = merged[merged[fdr_col] <= label_fdr].copy()
        n_pass = len(lab)
        if n_pass > max_labels:
            lab = lab.nlargest(max_labels, log10_col)
        # Variable FDR threshold line being labeled
        if label_fdr > 0:
            fig.add_hline(
                y=-math.log10(label_fdr),
                line_dash="dot",
                line_color="#2c7fb8",
                line_width=1,
                annotation_text=f"FDR ≤ {label_fdr:g}",
                annotation_position="top left",
            )
        if not lab.empty:
            add_point_labels(
                fig,
                lab[feature].values,
                lab[log10_col].values,
                lab["gene_symbol_0"].astype(str).values,
                merged[feature].values,
                merged[log10_col].values,
                xlim,
                ylim,
            )

    fig.update_layout(
        title=f"Volcano: {feature}",
        xaxis_title="z-score",
        yaxis_title="-log10(FDR)",
        xaxis=dict(range=list(xlim)),
        yaxis=dict(range=list(ylim)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=40, r=20),
        height=450,
    )
    return fig


def build_waterfall_plot(merged, feature):
    sorted_df = merged.sort_values(feature).reset_index(drop=True)
    sorted_df["x_pos"] = range(len(sorted_df))

    bg = sorted_df[~sorted_df["significant"]]
    sig = sorted_df[sorted_df["significant"]]

    def make_trace(df, name, color, size):
        return go.Scatter(
            x=df["x_pos"],
            y=df[feature],
            mode="markers",
            name=name,
            marker=dict(color=color, size=size, opacity=0.6),
            customdata=df[["gene_symbol_0"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"value: %{{y:.3f}}<extra></extra>"
            ),
        )

    fig = go.Figure()
    fig.add_trace(make_trace(bg, "Background", "#aaaaaa", 6))
    fig.add_trace(make_trace(sig, "Significant (FDR<0.05)", "#756bb1", 9))

    y_min = sorted_df[feature].min()
    y_max = sorted_df[feature].max()
    if y_min < 0 < y_max:
        fig.add_hline(y=0, line_dash="solid", line_color="#444444", line_width=1)

    fig.update_layout(
        title=f"Waterfall: {feature}",
        xaxis_title="Rank",
        yaxis_title="Feature value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=40, r=20),
        height=450,
    )
    return fig


# =====================
# GENE INFO DISPLAYS


def display_uniprot_info_fe(gene):
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

    if f"fe_guide_{gene}" not in st.session_state:
        st.session_state[f"fe_guide_{gene}"] = None

    def on_guide_select():
        st.session_state[f"fe_guide_{gene}"] = st.session_state[f"fe_guide_dropdown_{gene}"]

    selected_guide = st.session_state.get(f"fe_guide_{gene}", None)
    selected_index = 0
    if selected_guide in available_guides:
        selected_index = available_guides.index(selected_guide)
    elif available_guides:
        selected_guide = available_guides[0]
        st.session_state[f"fe_guide_{gene}"] = selected_guide

    selected_guide = st.selectbox(
        "Select Guide",
        available_guides,
        index=selected_index,
        key=f"fe_guide_dropdown_{gene}",
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
                key_prefix=f"fe_{gene}_{selected_guide}",
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
                        key=f"fe_download_{gene}_{selected_guide}_{uuid.uuid4()}",
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
        key=f"fe_rep_feature_{gene}_{cell_class}_{channel_combo}",
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
    if "fe_selected_feature" not in st.session_state:
        st.session_state.fe_selected_feature = None
    if "fe_selected_gene" not in st.session_state:
        st.session_state.fe_selected_gene = None
    if "hide_controls" not in st.session_state:
        st.session_state.hide_controls = False


init_state()

# =====================
# PAGE HEADER

title_col, reset_col = st.columns([4, 1])
with title_col:
    st.title("Feature Explorer")
with reset_col:
    st.write("")  # vertical padding
    if st.button("Reset All", type="secondary"):
        st.session_state.fe_selected_feature = None
        st.session_state.fe_selected_gene = None
        st.rerun()

# =====================
# FEATURE DROPDOWN

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
current_feature = st.session_state.fe_selected_feature
current_index = (
    feature_options.index(current_feature)
    if current_feature in feature_options
    else 0
)

selected_label = st.selectbox(
    f"Feature (FDR < 0.05 in {CHANNEL_COMBO})",
    options=feature_options,
    index=current_index,
    key="fe_feature_select",
)

if selected_label == "Select a feature...":
    st.session_state.fe_selected_feature = None
    st.info("Select a feature above to display volcano and waterfall plots.")
    st.stop()

# New feature selection — clear gene
if selected_label != st.session_state.fe_selected_feature:
    st.session_state.fe_selected_gene = None

st.session_state.fe_selected_feature = selected_label
feature = selected_label

# =====================
# PLOTS

merged = build_merged_df(feature)
if merged is None or merged.empty:
    st.warning(f"Could not load data for feature '{feature}'.")
    st.stop()

if st.session_state.hide_controls:
    merged = merged[~merged["gene_symbol_0"].str.startswith(_control_key)].reset_index(drop=True)

n_sig = merged["significant"].sum()
st.caption(
    f"{len(merged)} genes · {n_sig} significant (FDR < 0.05) · "
    f"Click a point to select a gene"
)

col1, col2 = st.columns(2)
with col1:
    lab_col, fdr_col_ctrl = st.columns([1, 2])
    with lab_col:
        show_labels = st.toggle("Label genes", value=False, key="fe_volcano_labels")
    with fdr_col_ctrl:
        label_fdr = st.slider(
            "Label genes with FDR ≤",
            min_value=0.001,
            max_value=0.25,
            value=0.05,
            step=0.001,
            format="%.3f",
            key="fe_volcano_label_fdr",
            disabled=not show_labels,
        )
    vol_fig = build_volcano_plot(
        merged, feature, show_labels=show_labels, label_fdr=label_fdr
    )
    vol_event = st.plotly_chart(
        vol_fig,
        on_select="rerun",
        key="fe_volcano",
        use_container_width=True,
        config={"toImageButtonOptions": {"format": "svg"}},
    )
with col2:
    wfall_fig = build_waterfall_plot(merged, feature)
    wfall_event = st.plotly_chart(
        wfall_fig,
        on_select="rerun",
        key="fe_waterfall",
        use_container_width=True,
        config={"toImageButtonOptions": {"format": "svg"}},
    )

# Handle click events from either plot
for event in [vol_event, wfall_event]:
    if event and hasattr(event, "selection") and event.selection and event.selection.points:
        clicked_gene = event.selection.points[0]["customdata"][0]
        if clicked_gene != st.session_state.fe_selected_gene:
            st.session_state.fe_selected_gene = clicked_gene
            st.rerun()
        break

# =====================
# GENE INFO PANEL

selected_gene = st.session_state.fe_selected_gene
if not selected_gene:
    st.stop()

st.divider()

gene_title_col, clear_col = st.columns([4, 1])
with gene_title_col:
    st.markdown(f"## Gene: {selected_gene}")
with clear_col:
    st.write("")
    if st.button("Clear Gene", type="secondary"):
        st.session_state.fe_selected_gene = None
        st.rerun()

# UniProt
st.markdown("#### UniProt Summary")
display_uniprot_info_fe(selected_gene)

# Montages
gene_montages_root = os.path.join(
    BRIEFLOW_OUTPUT_PATH, "aggregate", "montages", f"{CELL_CLASS}__montages"
)
if os.path.exists(gene_montages_root):
    st.markdown("#### Gene Montages")
    display_gene_montages(gene_montages_root, selected_gene)

# sgRNA assessment
display_sgrna_representativeness(selected_gene, CELL_CLASS, CHANNEL_COMBO)
