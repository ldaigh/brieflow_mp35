import os
import glob

import matplotlib.pyplot as plt
import streamlit as st
from src.filesystem import FileSystem
from src.rendering import VisualizationRenderer
from src.filtering import create_filter_radio, apply_filter
from src.config import BRIEFLOW_OUTPUT_PATH
from src.sbs_abundance import (
    NORMALIZE_MODES,
    build_gdna_scatter,
    build_gdna_scatter_mpl,
    build_gene_symbol_histogram,
    figure_to_vector,
    fit_stats,
    has_gdna_data,
    list_sbs_cell_tables,
    load_sbs_gene_symbols,
    match_gene_counts,
    prepare_axes,
)

st.set_page_config(
    page_title="Quality Control - Brieflow Analysis",
    page_icon=":microscope:",
    layout="wide",
)


def find_eval_files(root_dir):
    pattern = os.path.join(root_dir, "*", "eval", "**", "*")
    all_files = glob.glob(pattern, recursive=True)
    return [f for f in all_files if f.endswith(".png") or f.endswith(".tsv")]


@st.cache_data
def load_data(root_dir):
    global filtered_df
    files = find_eval_files(root_dir)
    filtered_df = FileSystem.extract_features(root_dir, files)
    return filtered_df


# Create filters using the helper function
def apply_all_filters(df, sidebar):
    """Apply all filters in sequence and return the filtered dataframe."""
    filters = [
        ("dir_level_0", "Phase"),
        # Intentionally omitting dir_level_1
        ("dir_level_2", "Subgroup"),
        ("plate_id", "Plate"),
        ("well_id", "Well"),
        ("metric_name", "Metric"),
    ]

    filtered_df = df.copy()
    selected_values = {}

    for column, label in filters:
        # Create a unique key for each filter based on the column name
        key = f"filter_{column}"
        selected_value = create_filter_radio(
            filtered_df, column, sidebar, label, key=key
        )
        filtered_df = apply_filter(filtered_df, column, selected_value)
        selected_values[column] = selected_value

    return filtered_df, selected_values


def render_sbs_count_figures():
    """SBS gene-count figures rendered live from the per-well cell tables.

    Both are driven by the same plate/well selection: the pipeline's gene-symbol
    count histogram (re-rendered so it can be exported as a vector, which the
    stored PNG cannot) and the gene-abundance comparison against the external
    gDNA count library.
    """
    st.divider()
    st.subheader("SBS gene counts")

    tables = list_sbs_cell_tables(BRIEFLOW_OUTPUT_PATH)
    if tables.empty:
        st.warning(
            f"No SBS `*__cells.parquet` tables found under "
            f"`{os.path.join(BRIEFLOW_OUTPUT_PATH, 'sbs', 'parquets')}`."
        )
        return

    sel_a, sel_b = st.columns([1, 1])
    with sel_a:
        plates = st.multiselect(
            "Plate(s)",
            options=sorted(tables["plate"].unique()),
            default=sorted(tables["plate"].unique()),
            key="sbs_counts_plates",
        )
    with sel_b:
        wells = st.multiselect(
            "Well(s)",
            options=sorted(tables["well"].unique()),
            default=sorted(tables["well"].unique()),
            key="sbs_counts_wells",
        )

    # Exported filenames record the selection whenever it is a subset, so a
    # single-well figure is not mistaken for the whole screen.
    tag = ""
    if set(plates) != set(tables["plate"]):
        tag += "_P-" + "-".join(plates)
    if set(wells) != set(tables["well"]):
        tag += "_W-" + "-".join(wells)

    render_gene_symbol_histogram(tuple(plates), tuple(wells), tag)
    render_gdna_comparison(tuple(plates), tuple(wells), tag)


def render_gene_symbol_histogram(plates, wells, tag=""):
    """The pipeline's gene-symbol count histogram, with vector download."""
    st.markdown("#### Gene symbol count histogram")
    st.caption(
        "Same figure as `sbs/eval/mapping/P-*__gene_symbol_histogram*.png`, "
        "re-rendered here so it can be downloaded as a vector."
    )

    hist_a, hist_b = st.columns([1, 1])
    with hist_a:
        exclude_controls = st.toggle(
            "Exclude controls",
            value=True,
            key="sbs_hist_no_controls",
            help="Drops perturbations whose gene symbol starts with the control key.",
        )
    with hist_b:
        x_cutoff = st.number_input(
            "X-axis cutoff (0 = auto)",
            min_value=0,
            value=0,
            step=50,
            key="sbs_hist_cutoff",
            help="Auto uses the pipeline's IQR-based cutoff (Q3 + 1.5 × IQR).",
        )

    cells = load_sbs_gene_symbols(plates, wells)
    if cells.empty:
        st.warning("No mapped cells in the selected wells.")
        return

    outliers, fig = build_gene_symbol_histogram(
        cells,
        exclude_controls=exclude_controls,
        x_cutoff=float(x_cutoff) if x_cutoff else None,
    )
    st.pyplot(fig, use_container_width=False)

    stem = "gene_symbol_histogram" + ("_no_controls" if exclude_controls else "") + tag
    hist_svg, hist_pdf, hist_note = st.columns([1, 1, 2])
    with hist_svg:
        st.download_button(
            "Download histogram (SVG)",
            data=figure_to_vector(fig, "svg"),
            file_name=f"{stem}.svg",
            mime="image/svg+xml",
            key="sbs_hist_dl_svg",
        )
    with hist_pdf:
        st.download_button(
            "Download histogram (PDF)",
            data=figure_to_vector(fig, "pdf"),
            file_name=f"{stem}.pdf",
            mime="application/pdf",
            key="sbs_hist_dl_pdf",
        )
    with hist_note:
        st.caption(f"{len(outliers)} gene(s) above the x-axis cutoff (not shown).")
    plt.close(fig)

    if len(outliers):
        with st.expander(f"Genes above the cutoff ({len(outliers)})"):
            st.dataframe(
                outliers.rename("cells").rename_axis("gene_symbol_0"),
                use_container_width=False,
            )


def render_gdna_comparison(plates, wells, tag=""):
    """SBS gene-abundance scatter against the external gDNA count library."""
    st.markdown("#### Gene abundance vs external gDNA counts")
    st.caption(
        "Per-gene abundance in this screen (mapped cells, `gene_symbol_0`) against "
        "an external gDNA amplicon library, whose guide counts are summed per gene."
    )

    if not has_gdna_data():
        st.info(
            "Set `GDNA_COUNTS_PATH` to the external gDNA count CSV "
            "(headerless `<guide name>,<count>`) to enable this figure."
        )
        return

    normalize = st.selectbox(
        "Counts",
        options=NORMALIZE_MODES,
        key="gdna_normalize",
        help=(
            "Counts per million divides each side by its own library total "
            "(controls included), making the two scales comparable."
        ),
    )

    opt_a, opt_b, opt_c, opt_d = st.columns([1, 1, 1, 1])
    with opt_a:
        log_scale = st.toggle(
            "Log–log axes",
            value=False,
            key="gdna_log",
            help="Fits a power law (OLS on log10 of both axes).",
        )
    with opt_b:
        show_fit = st.toggle("Best fit line", value=True, key="gdna_fit")
    with opt_c:
        show_controls = st.toggle(
            "Show pooled controls",
            value=False,
            key="gdna_controls",
            help=(
                "All control guides pool into one point that would dominate the "
                "fit, so it is always excluded from the fit and its statistics."
            ),
        )
    with opt_d:
        show_labels = st.toggle(
            "Label outliers",
            value=False,
            key="gdna_labels",
            help="Labels the genes furthest from the fitted line.",
        )

    matched, info = match_gene_counts(plates, wells)
    if matched.empty:
        st.warning(
            "No genes matched between the screen and the external library "
            f"({info['screen_genes']} screen genes, {info['external_genes']} "
            "external genes)."
        )
        return

    plot_df, x_title, y_title = prepare_axes(matched, info, normalize)
    genes = plot_df[~plot_df["is_control"]]
    fit = fit_stats(genes["_x"], genes["_y"], log_space=log_scale)

    st.caption(
        f"**{fit['n']} genes** in the fit (of {info['screen_genes']} screen / "
        f"{info['external_genes']} external genes) · "
        f"Pearson r = {fit['r']:.3f} (p = {fit['r_p']:.3g}) · "
        f"Spearman ρ = {fit['rho']:.3f} (p = {fit['rho_p']:.3g}) · "
        f"slope = {fit['slope']:.3g}, R² = {fit['r2']:.3f}"
        + (" (log10–log10)" if log_scale else "")
        + f" · {info['screen_total']:,} mapped cells vs "
        f"{info['external_total']:,} external reads"
    )

    fig_df = plot_df if show_controls else genes
    fig = build_gdna_scatter(
        fig_df,
        info,
        x_title,
        y_title,
        fit,
        log_scale=log_scale,
        show_fit=show_fit,
        show_identity=normalize == "Counts per million",
        show_labels=show_labels,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key="gdna_scatter",
        config={
            "toImageButtonOptions": {
                "format": "svg",
                "filename": "sbs_gene_abundance_vs_gdna",
            }
        },
    )

    # Vector export. Plotly's server-side export needs kaleido (absent here), so
    # the downloadable figure is rendered with matplotlib.
    static_fig = build_gdna_scatter_mpl(
        fig_df,
        x_title,
        y_title,
        fit,
        log_scale=log_scale,
        show_fit=show_fit,
        show_identity=normalize == "Counts per million",
        show_labels=show_labels,
    )
    svg_data = figure_to_vector(static_fig, "svg")
    pdf_data = figure_to_vector(static_fig, "pdf")
    plt.close(static_fig)

    stem = "sbs_gene_abundance_vs_gdna" + ("_loglog" if log_scale else "") + tag
    dl_svg, dl_pdf, dl_csv = st.columns([1, 1, 1])
    with dl_svg:
        st.download_button(
            "Download figure (SVG)",
            data=svg_data,
            file_name=f"{stem}.svg",
            mime="image/svg+xml",
            key="gdna_dl_svg",
        )
    with dl_pdf:
        st.download_button(
            "Download figure (PDF)",
            data=pdf_data,
            file_name=f"{stem}.pdf",
            mime="application/pdf",
            key="gdna_dl_pdf",
        )
    with dl_csv:
        st.download_button(
            "Download gene table (CSV)",
            data=plot_df.drop(columns=["_x", "_y"]).to_csv(index=False),
            file_name=f"{stem}.csv",
            mime="text/csv",
            key="gdna_dl_csv",
        )

    with st.expander("Matched gene counts"):
        st.dataframe(
            plot_df.drop(columns=["_x", "_y"]).set_index("gene_symbol"),
            use_container_width=True,
        )
    if info["missing_in_external"] or info["missing_in_screen"]:
        with st.expander(
            f"Unmatched genes ({len(info['missing_in_external'])} screen-only, "
            f"{len(info['missing_in_screen'])} external-only)"
        ):
            col_screen, col_ext = st.columns(2)
            with col_screen:
                st.markdown("**In this screen, not in the gDNA library**")
                st.write(", ".join(info["missing_in_external"]) or "—")
            with col_ext:
                st.markdown("**In the gDNA library, not mapped in this screen**")
                st.write(", ".join(info["missing_in_screen"]) or "—")


st.title("Quality Control")
st.markdown("Review the quality control metrics from the brieflow pipeline")

# Load the data
filtered_df = load_data(BRIEFLOW_OUTPUT_PATH)

st.sidebar.title("Filters")
filtered_df, selected_values = apply_all_filters(filtered_df, st.sidebar)

VisualizationRenderer.display_plots_and_tables(filtered_df, BRIEFLOW_OUTPUT_PATH)

# SBS-only figures: shown with the SBS eval plots (Phase = sbs, or unfiltered).
if selected_values.get("dir_level_0") in (None, "All", "sbs"):
    render_sbs_count_figures()
