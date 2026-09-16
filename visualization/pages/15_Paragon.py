import os
import sys

import streamlit as st

st.set_page_config(
    page_title="Paragon - Brieflow Analysis",
    layout="wide",
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =====================
# CONFIGURATION

# Root of the paragon_analysis project, whose package is `paragon_lib`. It must
# NOT be called `lib`: brieflow's editable install (`__editable__.brieflow-*.pth`)
# puts `brieflow/workflow` on sys.path for the whole environment, and every page
# that imports `workflow.lib.cluster.*` pulls in brieflow's top-level `lib`, which
# then owns `sys.modules["lib"]` for the rest of the process. In a multipage app
# the home page does that before this page ever runs.
PARAGON_ROOT = os.environ.get("PARAGON_ROOT", "")

PARAGON_AVAILABLE = False
PARAGON_IMPORT_ERROR = None
if PARAGON_ROOT and os.path.isdir(PARAGON_ROOT):
    if PARAGON_ROOT not in sys.path:
        sys.path.insert(0, PARAGON_ROOT)
    try:
        from paragon_lib import axes as pax
        from paragon_lib import crops as pcrops
        from paragon_lib import exemplars as pexe
        from paragon_lib import features as pfeat
        from paragon_lib import paths as ppaths
        from paragon_lib import render as prender
        from paragon_lib import signatures as psig
        from paragon_lib import spaces as pspaces
        from paragon_lib import views as pviews

        PARAGON_AVAILABLE = True
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI below
        PARAGON_IMPORT_ERROR = exc

AXIS_METHOD_LABELS = {
    "Mean of per-gene LDs (Labitigan)": "gene_ld_mean",
    "Centroid difference": "centroid",
    "Cluster-vs-control LDA": "cluster_lda",
}

# Order matters: the first entry is the default. The paper draws landmarks from
# the perturbed population but IMAGES from the entire dataset, and that is not a
# stylistic choice -- a narrow percentile band of a 73-cell gene contains no
# cells of that gene, so a subject-only pool cannot populate a ladder.
POOL_LABELS = {
    "Whole dataset (paper default)": "whole_dataset",
    "Subject cells only": "cluster",
    "Non-targeting only": "control",
}

MODE_LABELS = {
    "Latent axis (VIEWS)": "views",
    "Interpretable features": "features",
}

CHANNEL_LABELS = {0: "DAPI", 1: "CFP", 2: "YFP", 3: "RFP", 4: "Cy5"}


# =====================
# DATA LOADING


@st.cache_data(show_spinner=False)
def load_index():
    """Cell index. Cached as a DataFrame -- ~50 MB, loads in about a second."""
    return pspaces.load_cell_index()


@st.cache_resource(show_spinner=False)
def load_pcs(k):
    """Leading-K PC subspace.

    cache_resource rather than cache_data: this is a 20-200 MB array that never
    needs copying per session, and cache_data would pickle it on every miss.
    """
    return pspaces.load_pc_matrix(k)


@st.cache_data(show_spinner=False)
def load_clustering(channel_combo, resolution, cell_class):
    fp = ppaths.clustering_fp(channel_combo, resolution, cell_class)
    if not os.path.exists(fp):
        return pd.DataFrame()
    return pd.read_csv(fp, sep="\t")


@st.cache_data(show_spinner=False)
def load_axes(channel_combo, resolution, k, method):
    """Precomputed cluster axes, or None if that combination was not built."""
    fp = ppaths.axes_fp(channel_combo, resolution, k, method)
    if not os.path.exists(fp):
        return None
    with np.load(fp, allow_pickle=False) as z:
        return {int(c): z["axes"][i] for i, c in enumerate(z["clusters"])}


@st.cache_data(show_spinner=False)
def load_axes_meta(channel_combo, resolution):
    fp = ppaths.axes_meta_fp(channel_combo, resolution)
    if not os.path.exists(fp):
        return pd.DataFrame()
    return pd.read_csv(fp, sep="\t")


@st.cache_data(show_spinner=False)
def load_signatures(channel_combo, resolution):
    fp = ppaths.signatures_fp(channel_combo, resolution)
    if not os.path.exists(fp):
        return pd.DataFrame()
    return pd.read_csv(fp, sep="\t")


@st.cache_data(show_spinner=False)
def load_gene_ld(channel_combo, k):
    """Per-gene LD vectors and their cross-correlation (Labitigan Figure 6)."""
    fp = ppaths.gene_ld_fp(channel_combo, k)
    names_fp = str(fp).replace(".npz", ".names.npz")
    if not (os.path.exists(fp) and os.path.exists(names_fp)):
        return None, None
    with np.load(fp, allow_pickle=False) as z:
        gene_matrix = z["gene_matrix"]
    with np.load(names_fp, allow_pickle=False) as nz:
        gene_names = [str(s) for s in nz["gene_names"]]
    return gene_names, gene_matrix


@st.cache_data(show_spinner=False)
def load_pc_diagnostics():
    try:
        return pspaces.load_pc_diagnostics()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_gene_features_cached(channel_combo, cell_class):
    return psig.load_gene_features(channel_combo, cell_class)


@st.cache_data(show_spinner=False)
def load_bootstrap_cached(bootstrap_combo, cell_class):
    return psig.load_gene_bootstrap(bootstrap_combo, cell_class)


@st.cache_data(show_spinner=False)
def gene_signature_cached(channel_combo, cell_class, bootstrap_combo, gene):
    """Signal/nuisance partition for one gene, from the bootstrap null."""
    gf = load_gene_features_cached(channel_combo, cell_class)
    bs = load_bootstrap_cached(bootstrap_combo, cell_class)
    feats = psig.feature_columns(gf)
    redundancy = psig.global_redundancy_groups(gf, feats)
    return psig.gene_signature(gf, bs, gene, features=feats, redundancy=redundancy)


@st.cache_data(show_spinner=False)
def load_gene_separation(channel_combo, k):
    """Per-gene axis separation, ranked. Empty if not built."""
    fp = ppaths.gene_separation_fp(channel_combo, k)
    if not os.path.exists(fp):
        return pd.DataFrame()
    return pd.read_csv(fp, sep="\t")


@st.cache_data(show_spinner=False)
def load_ld_crosscorr(channel_combo, k):
    names, matrix = load_gene_ld(channel_combo, k)
    if names is None:
        return None
    return pax.gene_ld_crosscorr(names, matrix)


@st.cache_data(show_spinner=False)
def load_space_meta():
    """What the embedding space was actually built from."""
    import json

    fp = ppaths.CELL_INDEX_META_FP
    if not os.path.exists(fp):
        return {}
    try:
        return json.loads(open(fp).read())
    except Exception:  # noqa: BLE001
        return {}


@st.cache_data(show_spinner=False, max_entries=64)
def cached_crops(cell_keys, box, channels, outline):
    """Crop a set of cells.

    `cell_keys` is a tuple of (plate, well, tile, cell_0, i_0, j_0) tuples so
    the cache key is hashable and stable. Crops come back as a list of dicts
    with numpy arrays, which Streamlit pickles happily.
    """
    df = pd.DataFrame(
        list(cell_keys), columns=["plate", "well", "tile", "cell_0", "i_0", "j_0"]
    )
    return pcrops.crop_cells(df, box=box, channels=list(channels), outline=outline)


def crop_keys(df):
    return tuple(
        df[["plate", "well", "tile", "cell_0", "i_0", "j_0"]]
        .itertuples(index=False, name=None)
    )


# =====================
# HELPERS


def pool_mask_for(choice, index, member_mask, is_control, gate_ok):
    if choice == "cluster":
        return member_mask & gate_ok
    if choice == "control":
        return is_control & gate_ok
    return gate_ok


def format_stats_line(stats):
    line = (
        f"band [{stats['band_lo']:.3f}, {stats['band_hi']:.3f}] | "
        f"{stats['n_candidates']:,} of {stats['n_pool']:,} pool cells in band "
        f"({100 * stats['n_candidates'] / max(stats['n_pool'], 1):.3f}%) | "
        f"showing {stats['n_shown']} from {stats.get('n_tiles_shown', 0)} tiles / "
        f"{stats.get('n_guides_shown', 0)} sgRNAs"
    )
    if stats.get("widened"):
        line += (
            f" | ⚠ band widened {stats['halfwidth']:.2f}→"
            f"{stats['halfwidth_used']:.2f} percentile half-width to find "
            "candidates, so these cells are less tightly localized on the axis"
        )
    return line


def separation_banner(sep, container):
    """State the axis separation prominently, good or bad.

    In OPS data the single-cell spread routinely dwarfs the mean perturbation
    effect, which means median-percentile exemplars can be indistinguishable
    from controls. A viewer must be told that rather than left to infer it.
    """
    d = sep["cohens_d"]
    auc = sep["auc"]
    msg = (
        f"**Axis separation: Cohen's d = {d:.3f}, AUC = {auc:.3f}.** "
        f"The cluster's mean sits at the {sep['cluster_mean_control_percentile']:.1f}th "
        f"percentile of the control distribution "
        f"(cluster SD {sep['cluster_sd']:.2f} vs control SD {sep['control_sd']:.2f})."
    )
    if abs(d) < 0.2:
        container.error(
            msg
            + "\n\nThis is a **very small** separation: the average cluster cell is an "
            "ordinary control cell. Mid-ladder exemplars carry essentially no "
            "phenotype -- only the p1/p99 tails are informative, and even those are "
            "selected from the overlap between the distributions. Treat any apparent "
            "difference with suspicion and check the shuffled-axis control."
        )
    elif abs(d) < 0.5:
        container.warning(
            msg
            + "\n\nSmall separation. The tails of the ladder are where the phenotype "
            "lives; do not read the median rung as characteristic of the cluster."
        )
    else:
        container.success(msg)


# =====================
# PAGE


st.title("Paragon - representative cells for a cluster")

if not PARAGON_ROOT:
    st.error(
        "`PARAGON_ROOT` is not set. Add it to `analysis/14.run_visualization.sh`:\n\n"
        "```bash\n"
        'export PARAGON_ROOT="/scratch/users/ldaigh/brieflow-MP35/'
        'brieflow_mp35_analysis/paragon_analysis"\n'
        "```"
    )
    st.stop()

if not PARAGON_AVAILABLE:
    st.error(f"Could not import the Paragon library from `{PARAGON_ROOT}`.")
    st.exception(PARAGON_IMPORT_ERROR)
    st.stop()

try:
    index = load_index()
except Exception as exc:  # noqa: BLE001
    st.error(
        "Paragon's cell index has not been built yet. Run:\n\n"
        "```bash\n"
        "cd paragon_analysis && sbatch scripts/build_cell_index.sbatch\n"
        "```"
    )
    st.exception(exc)
    st.stop()

st.caption(
    "Exemplar cells chosen to be strongly displaced along the direction that "
    "defines a cluster while remaining unremarkable in every other measured "
    "respect - the VIEWS procedure of Labitigan et al. (eLife reviewed preprint "
    "94964), plus an interpretable-feature variant. Every panel reports the "
    "denominators behind it."
)

# --- sidebar ---------------------------------------------------------------
sb = st.sidebar
sb.header("Paragon")

combos = sorted(
    {
        p
        for p in os.listdir(ppaths.CLUSTER_DIR)
        if os.path.isdir(os.path.join(ppaths.CLUSTER_DIR, p))
    }
) if os.path.isdir(ppaths.CLUSTER_DIR) else []
default_combo = (
    ppaths.CHANNEL_COMBO if ppaths.CHANNEL_COMBO in combos else (combos[0] if combos else "")
)
channel_combo = sb.selectbox(
    "Cluster source (channel combo)",
    combos,
    index=combos.index(default_combo) if default_combo in combos else 0,
    key="paragon_combo",
    help=(
        "This selects which clustering supplies the GENE GROUPS only. The "
        "embedding space the axes live in is fixed by whatever build_cell_index "
        "was run against - shown under 'Embedding space' below. Picking a "
        "different combo here means 'group genes by that clustering, then find "
        "their direction in this space', which is defensible but worth being "
        "deliberate about."
    ),
)
cell_class = ppaths.CELL_CLASS

space_meta = load_space_meta()
if space_meta:
    space_combo = space_meta.get("space_channel_combo", "?")
    sb.caption(
        f"**Embedding space:** {space_combo} | "
        f"{space_meta.get('n_cells', 0):,} cells | "
        f"{space_meta.get('n_pcs', 0)} PCs | "
        f"{space_meta.get('n_targeting_genes', 0)} targeting genes"
    )
    if space_combo != channel_combo:
        sb.warning(
            f"Gene groups come from the **{channel_combo}** clustering but the "
            f"axes are computed in the **{space_combo}** PC space."
        )

level = sb.radio(
    "Axis level",
    ["Gene", "Cluster"],
    key="paragon_level",
    help=(
        "Gene is the unit both source papers actually use, and in this screen "
        "per-gene axes separate from controls 2-3x better than cluster axes: "
        "the best genes reach Cohen's d 0.45-0.66 against a median of 0.20, "
        "while cluster axes sit at 0.12-0.27. Averaging LD vectors across a "
        "cluster only helps to the extent its genes share a direction."
    ),
)

resolutions = ppaths.available_resolutions(channel_combo, cell_class)
if not resolutions:
    # Gene mode needs no clustering at all -- the per-gene LD vectors and the
    # separation ranking are self-contained. Only Cluster mode depends on this
    # tree, which the brieflow pipeline empties and regenerates while running.
    if level == "Gene":
        sb.warning(
            "No clustering on disk for this combo (the pipeline empties and "
            "rebuilds that tree while it runs). Gene mode does not need it."
        )
        resolutions = [0]
    else:
        st.error(
            f"No clustering found under "
            f"`{ppaths.CLUSTER_DIR / channel_combo / cell_class}`. The brieflow "
            "pipeline empties this tree while regenerating it, so this is "
            "usually transient. Switch **Axis level** to *Gene*, which does not "
            "depend on the clustering."
        )
        st.stop()
# Default 11: fine-grained enough to separate individual complexes (HOPS, mTOR)
# rather than lumping unrelated genes, which is what matters. Cluster SIZE is
# not the criterion -- LD coherence is, and it does not track size.
_preferred = [11, 13, 15, 9]
_default_res = next((r for r in _preferred if r in resolutions), resolutions[0])
resolution = sb.selectbox(
    "Leiden resolution",
    resolutions,
    index=resolutions.index(_default_res),
    key="paragon_resolution",
    disabled=(level == "Gene"),
    help=(
        "Only used at the Cluster level. What makes a cluster axis meaningful is "
        "whether its genes share a phenotypic direction (LD coherence, shown "
        "below), not how many genes it has -- a tight 4-gene complex beats a "
        "30-gene grab-bag."
    ),
)

clustering = load_clustering(channel_combo, resolution, cell_class)
if clustering.empty:
    if level == "Cluster":
        st.error(
            f"Clustering table for resolution {resolution} is empty or missing. "
            "Switch **Axis level** to *Gene*, which does not depend on it."
        )
        st.stop()
    targeting = pd.DataFrame({ppaths.PERTURBATION_NAME_COL: [], "cluster": []})
    cluster_sizes = pd.Series(dtype="int64")
else:
    targeting = clustering[
        ~clustering[ppaths.PERTURBATION_NAME_COL]
        .astype(str)
        .str.startswith(ppaths.CONTROL_PREFIX)
    ]
    cluster_sizes = targeting["cluster"].value_counts().sort_index()

# --- subject selection: one gene, or one cluster of genes ------------------
separation = load_gene_separation(channel_combo, ppaths.DEFAULT_K)
cluster_id = None
selected_gene = None

if level == "Gene":
    if not separation.empty:
        order = sb.radio(
            "Order genes by",
            ["Axis separation", "Alphabetical", "Cell count"],
            key="paragon_gene_order",
            help=(
                "Separation-ordered puts the genes whose phenotype is actually "
                "visible at single-cell resolution first. Most genes in this "
                "screen separate only weakly."
            ),
        )
        if order == "Axis separation":
            gene_list = separation.sort_values("cohens_d", ascending=False)
        elif order == "Cell count":
            gene_list = separation.sort_values("n_cells", ascending=False)
        else:
            gene_list = separation.sort_values("gene")
        d_by_gene = separation.set_index("gene")["cohens_d"].to_dict()
        n_by_gene = separation.set_index("gene")["n_cells"].to_dict()
        options = gene_list["gene"].tolist()

        def _fmt(g):
            return f"{g}  (d={d_by_gene.get(g, float('nan')):.2f}, {n_by_gene.get(g, 0):,} cells)"

        selected_gene = sb.selectbox(
            "Gene", options, format_func=_fmt, key="paragon_gene"
        )
        null_d = float(separation["null_cohens_d"].iloc[0])
        sb.caption(
            f"{len(options)} genes with an LD axis | median d "
            f"{separation['cohens_d'].median():.3f} | inflated null d {null_d:.3f}"
        )
    else:
        # No separation ranking: fall back to the genes present in the cell
        # index, which is always available, rather than the clustering.
        genes_avail = sorted(
            index.loc[~index["is_control"], ppaths.PERTURBATION_NAME_COL]
            .astype(str)
            .unique()
        )
        selected_gene = sb.selectbox("Gene", genes_avail, key="paragon_gene")
        sb.info(
            "Separation ranking not built yet - run "
            "`sbatch scripts/build_cluster_axes.sbatch` to order genes by how "
            "visible their phenotype is."
        )
else:
    corr = load_ld_crosscorr(channel_combo, ppaths.DEFAULT_K)
    coherence = {}
    if corr is not None:
        for cid in cluster_sizes.index:
            genes_c = pax.cluster_members(clustering, cid)
            inc = [g for g in genes_c if g in corr.index]
            if len(inc) >= 2:
                w = corr.loc[inc, inc].to_numpy()
                coherence[int(cid)] = float(
                    w[~np.eye(len(inc), dtype=bool)].mean()
                )
    order = sb.radio(
        "Order clusters by",
        ["LD coherence", "Size", "Cluster id"],
        key="paragon_cluster_order",
        help=(
            "Coherence is the mean pairwise correlation between member genes' LD "
            "vectors. It is what determines whether a cluster has one shared "
            "phenotypic direction; cluster size does not track it."
        ),
    )
    cids = [int(c) for c in cluster_sizes.index]
    if order == "LD coherence" and coherence:
        cids = sorted(cids, key=lambda c: -coherence.get(c, float("-inf")))
    elif order == "Size":
        cids = sorted(cids, key=lambda c: -cluster_sizes[c])

    def _fmt_c(c):
        coh = coherence.get(c)
        coh_s = f", coh={coh:.2f}" if coh is not None else ""
        return f"{c}  ({cluster_sizes[c]} genes{coh_s})"

    cluster_id = sb.selectbox("Cluster", cids, format_func=_fmt_c, key="paragon_cluster")
    if coherence:
        sb.caption(
            f"{len(cluster_sizes)} clusters | median coherence "
            f"{np.median(list(coherence.values())):.3f} | "
            f"{sum(1 for v in coherence.values() if v > 0.3)} above 0.3"
        )

method_label = sb.radio(
    "Axis definition",
    list(AXIS_METHOD_LABELS),
    key="paragon_axis_method",
    help=(
        "Mean-of-gene-LDs follows the paper and resists one high-cell-count gene "
        "dominating. Centroid difference is near-optimal here because the PC space "
        "is already whitened to the control covariance. Cluster-vs-control LDA is "
        "the simplest but lets a large gene choose the direction."
    ),
)
method = AXIS_METHOD_LABELS[method_label]

k = sb.slider(
    "Leading PCs (K)",
    min_value=5,
    max_value=ppaths.N_PCS,
    value=ppaths.DEFAULT_K,
    step=5,
    key="paragon_k",
    help=(
        "Labitigan et al. used the first 25 fPCs. Raising K pulls in the trailing "
        "PCs, where tvn_on_controls' control-fitted PCA amplifies a handful of "
        "segmentation artifacts by ~1000x - see the Diagnostics tab."
    ),
)

mode_label = sb.radio("Selection mode", list(MODE_LABELS), key="paragon_mode")
mode = MODE_LABELS[mode_label]

pool_label = sb.radio(
    "Draw images from",
    list(POOL_LABELS),
    key="paragon_pool",
    help=(
        "The paper takes landmarks from the perturbed population but images from "
        "the entire dataset. For a knockout screen you usually want the cluster's "
        "own cells. Whichever you pick is printed on the panel."
    ),
)
pool_choice = POOL_LABELS[pool_label]

n_show = sb.slider("Cells per rung", 1, 8, 3, key="paragon_n_show")

ladder_pcts = sb.multiselect(
    "Ladder rungs (percentile)",
    options=list(ppaths.LADDER_PERCENTILE_CHOICES),
    default=list(ppaths.LADDER_PERCENTILES),
    key="paragon_percentiles",
    disabled=(mode != "views"),
    help=(
        "Percentiles of the subject's own projection onto the axis. The default "
        "five are Labitigan et al. Figure 3. Adding 10 and 90 resolves the "
        "shoulders; dropping to three cuts image loading proportionally."
    ),
)
# multiselect returns click order, and an empty selection would render nothing.
ladder_pcts = tuple(sorted(ladder_pcts)) or ppaths.LADDER_PERCENTILES
if mode == "views":
    # Every rung is loaded for the ladder AND for both Controls-tab panels, and
    # st.tabs runs all four bodies on every rerun -- so the crop count is the
    # honest cost of another rung, not a hypothetical one.
    sb.caption(
        f"{len(ladder_pcts)} rungs x {n_show} cells = "
        f"{len(ladder_pcts) * n_show} crops per panel, "
        f"{len(ladder_pcts) * n_show * 3} per rerun including the two control "
        "panels."
    )
box = sb.select_slider(
    "Crop size (px, native scale)",
    options=[48, 64, 80, 96, 128, 160, 200, 256],
    value=ppaths.DEFAULT_CROP_BOX,
    key="paragon_box",
)

sb.subheader("Channels")
channels = []
colors = {}
for idx, name in CHANNEL_LABELS.items():
    cols = sb.columns([1.2, 1])
    on = cols[0].checkbox(
        name, value=idx in prender.DEFAULT_CHANNELS, key=f"paragon_ch_{idx}"
    )
    if on:
        channels.append(idx)
        color_names = list(prender.COMPOSITE_COLOR_OPTIONS)
        default_color = prender.CHANNEL_DEFAULTS.get(idx, ("", "Gray"))[1]
        colors[idx] = cols[1].selectbox(
            f"{name} colour",
            color_names,
            index=color_names.index(default_color),
            key=f"paragon_color_{idx}",
            label_visibility="collapsed",
        )
if not channels:
    st.warning("Enable at least one channel in the sidebar.")
    st.stop()

sb.subheader("Contrast")
shared_contrast = sb.checkbox(
    "Shared range across panel (recommended)",
    value=True,
    key="paragon_shared_contrast",
    help=(
        "Off means each cell is stretched by its own percentiles, which makes "
        "intensity phenotypes invisible: 'brighter than normal' and 'normal' both "
        "map to full range."
    ),
)
contrast_reference = sb.checkbox(
    "Set range from control cells",
    value=True,
    key="paragon_contrast_ref",
    disabled=not shared_contrast,
)
low_pct, high_pct = sb.slider(
    "Contrast percentiles",
    0.0,
    100.0,
    (1.0, 99.0),
    step=0.5,
    key="paragon_contrast_pct",
)
show_outline = sb.checkbox("Outline the subject cell", value=False, key="paragon_outline")

sb.subheader("Scale bar")
show_scale_bar = sb.checkbox(
    "Draw a scale bar",
    value=True,
    key="paragon_scale_bar",
    help=(
        "A white bar in the bottom-right corner of every crop, no label. The "
        "bar is drawn at an integer number of pixels, and the length that "
        "rounding actually produced is printed below -- quote that, not the "
        "requested value."
    ),
)
um_per_px = sb.number_input(
    "µm per pixel",
    min_value=0.001,
    max_value=100.0,
    value=float(ppaths.PIXEL_SIZE_UM),
    step=0.005,
    format="%.4f",
    key="paragon_um_per_px",
    disabled=not show_scale_bar,
    help=(
        "Defaults to 0.325, which is `pixel_size_x` in every row of "
        "`preprocess/metadata/phenotype/P-*__combined_metadata.parquet`. Crops "
        "are taken at native resolution from the 2048x2048 aligned tiles, so "
        "no resampling intervenes."
    ),
)
scale_bar_um = sb.number_input(
    "Bar length (µm)",
    min_value=0.1,
    max_value=500.0,
    value=float(ppaths.SCALE_BAR_UM),
    step=1.0,
    key="paragon_scale_bar_um",
    disabled=not show_scale_bar,
)
if show_scale_bar:
    _geom = prender.scale_bar_geometry(box, box, scale_bar_um, um_per_px)
    if _geom is None:
        sb.warning(
            f"A {scale_bar_um:g} µm bar is {scale_bar_um / um_per_px:.0f} px, "
            f"which does not fit in a {box} px crop ({box * um_per_px:.1f} µm "
            "across). No bar will be drawn - raise the crop size or shorten "
            "the bar."
        )
    else:
        sb.caption(
            f"{_geom['length_px']} px bar = "
            f"**{_geom['length_um_actual']:.2f} µm** | crop {box} px = "
            f"{box * um_per_px:.1f} µm | bar spans "
            f"{100 * _geom['length_px'] / box:.0f}% of the width"
        )

sb.subheader("Export")
export_on = sb.checkbox(
    "Per-image TIFF downloads",
    value=True,
    key="paragon_export",
    help=(
        "Adds a download button under every crop, plus one zip per panel. "
        "Encoding is done on each rerun whether or not you click, but an 80 px "
        "crop is only ~19 KB."
    ),
)
EXPORT_KINDS = {
    "Displayed composite (8-bit RGB)": "composite",
    "Raw channel stack (16-bit)": "stack",
}
export_kind = EXPORT_KINDS[
    sb.radio(
        "TIFF contents",
        list(EXPORT_KINDS),
        key="paragon_export_kind",
        disabled=not export_on,
        help=(
            "The composite is exactly what is on screen - contrast range, "
            "channel colours and scale bar already applied - and is what a "
            "figure panel needs. The raw stack is the untouched uint16 "
            "channels for requantification; no scale bar is burnt into it, "
            "because that would overwrite real pixel values. Both carry the "
            "µm calibration in their ImageJ tags."
        ),
    )
]
export_scale_bar = sb.checkbox(
    "Burn the scale bar into exported composites",
    value=True,
    key="paragon_export_scale_bar",
    disabled=not (export_on and show_scale_bar and export_kind == "composite"),
)

sb.subheader("Quality gates")
active_gates = [
    g
    for g in ppaths.GATE_COLS
    if sb.checkbox(g, value=True, key=f"paragon_gate_{g}")
]

held_out = sb.checkbox(
    "Held-out split (fit A1, show A2)",
    value=True,
    key="paragon_held_out",
    help=(
        "Choosing exemplars with the same cells that defined the axis is "
        "circular. Fitting on one well and displaying from the other makes each "
        "panel a held-out prediction."
    ),
)
diversify = sb.checkbox(
    "Diversity constraint",
    value=True,
    key="paragon_diversify",
    help="One cell per tile, spread across sgRNAs. Off produces near-duplicates.",
)

# --- shared computation ----------------------------------------------------
with st.spinner("Loading PC space..."):
    pcs = load_pcs(k)
is_control = pspaces.control_mask(index)
gate_ok = pspaces.gate_mask(index, active_gates) if active_gates else np.ones(len(index), bool)

gene_labels = index[ppaths.PERTURBATION_NAME_COL].astype(str).to_numpy()

if level == "Gene":
    genes = [selected_gene]
    subject = f"gene {selected_gene}"
else:
    genes = pax.cluster_members(clustering, cluster_id)
    if not genes:
        st.error(f"Cluster {cluster_id} contains no targeting genes.")
        st.stop()
    subject = f"cluster {cluster_id}"

member_mask = np.isin(gene_labels, genes)
if member_mask.sum() == 0:
    st.error(
        f"None of {subject}'s {len(genes)} gene(s) appear in the cell index. The "
        "clustering and the aligned parquet may be from different runs."
    )
    st.stop()

# Held-out split: fit the axis on one well, draw images from the other.
fit_mask = np.ones(len(index), bool)
show_mask = np.ones(len(index), bool)
if held_out:
    try:
        fit_mask, show_mask = pspaces.well_split_mask(index, fit_well="A1")
    except pspaces.SpaceError as exc:
        st.warning(f"Held-out split unavailable: {exc}")
        held_out = False

gene_names, gene_matrix = load_gene_ld(channel_combo, k)

axis_source = "computed on the fly"
v = None
stored_axes = (
    load_axes(channel_combo, resolution, k, method) if level == "Cluster" else None
)
if (
    level == "Cluster"
    and stored_axes is not None
    and cluster_id in stored_axes
    and not held_out
):
    v = stored_axes[cluster_id]
    axis_source = "precomputed"
else:
    try:
        fit_member = member_mask & fit_mask
        fit_control = is_control & fit_mask
        if method == "centroid":
            v = pax.axis_centroid(pcs, fit_member, fit_control)
        elif method == "cluster_lda":
            # At the gene level this is exactly the paper's per-guide LD, fitted
            # on one gene's cells against all controls.
            v = pax.axis_cluster_lda(pcs, fit_member, fit_control)
            axis_source = (
                "LDA, this gene vs all controls"
                if level == "Gene"
                else "LDA, cluster vs all controls"
            )
        else:
            if gene_names is None:
                st.warning(
                    f"Per-gene LD vectors for K={k} have not been built "
                    "(`sbatch scripts/build_cluster_axes.sbatch`). Falling back to "
                    "the centroid axis."
                )
                v = pax.axis_centroid(pcs, fit_member, fit_control)
                method = "centroid"
            elif held_out:
                # The cached LD vectors were fitted on all cells, so reusing
                # them would leak the held-out well into the axis. Refit.
                v = pax.axis_cluster_lda(pcs, fit_member, fit_control)
                axis_source = "LD refitted on the fit split (held-out mode)"
            else:
                v, n_used = pax.axis_gene_ld_mean(gene_names, gene_matrix, genes)
                axis_source = (
                    "precomputed gene LD vector"
                    if level == "Gene"
                    else f"mean of {n_used}/{len(genes)} gene LD vectors"
                )
    except pax.AxisError as exc:
        st.error(f"Could not build an axis for {subject}: {exc}")
        st.stop()

sep = pax.axis_separation(pcs, v, member_mask, is_control)
pool_mask = pool_mask_for(pool_choice, index, member_mask, is_control, gate_ok)
if held_out:
    pool_mask = pool_mask & show_mask

# --- header metrics --------------------------------------------------------
st.subheader(f"{subject}")

m = st.columns(6)
m[0].metric("Genes" if level == "Cluster" else "sgRNAs",
            len(genes) if level == "Cluster"
            else int(index.loc[member_mask, ppaths.PERTURBATION_ID_COL].nunique()))
m[1].metric("Cells", f"{int(member_mask.sum()):,}")
m[2].metric("Cohen's d", f"{sep['cohens_d']:.3f}")
m[3].metric("AUC", f"{sep['auc']:.3f}")
m[4].metric("Display pool", f"{int(pool_mask.sum()):,}")
m[5].metric("K", k)

# Cluster-level coherence warning. Note the criterion is coherence, not size: a
# tight small complex is a better subject than a large heterogeneous cluster.
if level == "Cluster":
    corr_c = load_ld_crosscorr(channel_combo, ppaths.DEFAULT_K)
    if corr_c is not None:
        inc = [g for g in genes if g in corr_c.index]
        if len(inc) >= 2:
            w = corr_c.loc[inc, inc].to_numpy()
            coh = float(w[~np.eye(len(inc), dtype=bool)].mean())
            if coh < 0.15:
                st.warning(
                    f"Mean pairwise LD correlation among this cluster's genes is "
                    f"{coh:.3f} - they do **not** share a phenotypic direction, "
                    "even though PHATE/Leiden grouped them. Averaging their LD "
                    "vectors produces an axis that represents no member well. "
                    "Inspect the genes individually at the Gene level instead."
                )
            else:
                st.caption(
                    f"LD coherence {coh:.3f} across {len(inc)} genes with an LD "
                    "vector."
                )

separation_banner(sep, st.container())

st.caption(
    f"Axis: **{method}** ({axis_source}) | "
    f"images from **{pool_label.lower()}**"
    + (" | axis fitted on well A1, images from A2" if held_out else "")
    + f" | gates: {', '.join(active_gates) if active_gates else 'none'}"
)

if level == "Cluster":
    with st.expander(f"Genes in cluster {cluster_id} ({len(genes)})", expanded=False):
        st.write(", ".join(sorted(genes)))
else:
    guides_here = sorted(
        index.loc[member_mask, ppaths.PERTURBATION_ID_COL].astype(str).unique()
    )
    with st.expander(f"sgRNAs for {selected_gene} ({len(guides_here)})", expanded=False):
        counts = (
            index.loc[member_mask, ppaths.PERTURBATION_ID_COL]
            .value_counts()
            .rename_axis("sgRNA")
            .reset_index(name="cells")
        )
        st.dataframe(counts, use_container_width=True, hide_index=True)

tab_ladder, tab_controls, tab_attr, tab_diag = st.tabs(
    ["Ladder", "Controls", "Attribution", "Diagnostics"]
)


def cell_tag(row):
    """Compact, unique-per-cell identifier used in keys and filenames."""
    return f"W{row['well']}-T{int(row['tile'])}-c{int(row['cell_0'])}"


def tiff_export(crop, rgb, row, panel):
    """(filename, bytes) for one exported crop, honouring the export mode."""
    provenance = (
        f"{row[ppaths.PERTURBATION_NAME_COL]} {row[ppaths.PERTURBATION_ID_COL]} "
        f"{cell_tag(row)} | {subject} | {panel} | plate {int(row['plate'])} | "
        f"box {box} px | {um_per_px:g} µm/px"
    )
    if export_kind == "stack":
        name = prender.safe_filename(
            "paragon", subject, panel,
            row[ppaths.PERTURBATION_NAME_COL], cell_tag(row), "raw",
            suffix=".tiff",
        )
        return name, prender.stack_tiff_bytes(
            crop,
            um_per_px=um_per_px,
            description=provenance,
            channel_names=CHANNEL_LABELS,
        )
    name = prender.safe_filename(
        "paragon", subject, panel,
        row[ppaths.PERTURBATION_NAME_COL], cell_tag(row), "composite",
        suffix=".tiff",
    )
    return name, prender.composite_tiff_bytes(
        rgb, um_per_px=um_per_px, description=provenance
    )


def render_panel(sel, container, caption_col="residual", title=None, panel="panel"):
    """Crop, composite and display one row of cells.

    `panel` identifies this row within the page; it disambiguates the download
    widgets' keys and names their files, so the same cell appearing in two
    panels does not collide.
    """
    if sel is None or len(sel) == 0:
        container.info("No cells to show.")
        return
    crop_list = cached_crops(crop_keys(sel), box, tuple(channels), show_outline)

    if shared_contrast:
        reference = None
        if contrast_reference:
            ctl_sample = index[is_control & gate_ok].head(12)
            if len(ctl_sample):
                reference = cached_crops(
                    crop_keys(ctl_sample), box, tuple(channels), False
                )
        ranges = prender.panel_display_range(
            crop_list, percentiles=(low_pct, high_pct), reference=reference
        )
        per_cell_ranges = [ranges] * len(crop_list)
    else:
        per_cell_ranges = [
            prender.panel_display_range([c], percentiles=(low_pct, high_pct))
            for c in crop_list
        ]

    if title:
        container.markdown(title)
    cols = container.columns(len(crop_list))
    exports = []
    for i, (crop, rng) in enumerate(zip(crop_list, per_cell_ranges)):
        rgb = prender.composite_rgb(crop, rng, colors=colors, outline=show_outline)
        row = sel.iloc[i]

        # Geometry from this image rather than from `box`: crop_cell always
        # returns box x box, but deriving it here keeps the bar honest if that
        # ever stops being true.
        geom = (
            prender.scale_bar_geometry(
                rgb.shape[0], rgb.shape[1], scale_bar_um, um_per_px
            )
            if show_scale_bar
            else None
        )
        # draw_scale_bar copies, so `rgb` stays available unbarred for an
        # export that should not have the bar burnt in.
        shown = prender.draw_scale_bar(rgb, geom) if geom is not None else rgb

        cols[i].image(shown, use_container_width=True)
        label = f"**{row[ppaths.PERTURBATION_NAME_COL]}**"
        cols[i].caption(
            f"{label}  \n`{row[ppaths.PERTURBATION_ID_COL]}`  \n"
            f"W{row['well']} T{int(row['tile'])} c{int(row['cell_0'])}"
        )
        if caption_col in row.index:
            cols[i].caption(f"{caption_col} {row[caption_col]:.2f}")

        if export_on:
            fname, payload = tiff_export(
                crop, shown if export_scale_bar else rgb, row, panel
            )
            exports.append((fname, payload))
            cols[i].download_button(
                "⬇ TIFF",
                data=payload,
                file_name=fname,
                mime="image/tiff",
                key=f"paragon_dl_{panel}_{i}_{cell_tag(row)}",
                use_container_width=True,
            )

    if export_on and len(exports) > 1:
        container.download_button(
            f"⬇ All {len(exports)} as .zip",
            data=prender.zip_bytes(exports),
            file_name=prender.safe_filename(
                "paragon", subject, panel, export_kind, suffix=".zip"
            ),
            mime="application/zip",
            key=f"paragon_zip_{panel}",
        )


# =====================
# LADDER


with tab_ladder:
    if mode == "views":
        st.subheader("Percentile ladder along the cluster axis")
        st.caption(
            "One row per percentile of the cluster's projection onto the axis. "
            "Within a row, cells are those closest to the axis - i.e. most "
            "typical in every direction other than this one. A real phenotype "
            "reads as a gradient from p1 to p99."
        )

        with st.spinner("Selecting exemplars..."):
            ladder, lstats = pviews.views_ladder(
                pcs,
                index,
                v,
                landmark_mask=member_mask & fit_mask if held_out else member_mask,
                pool_mask=pool_mask,
                percentiles=ladder_pcts,
                n_show=n_show,
                diversify=diversify,
                min_candidates=n_show * 3,
            )

        s_all = pviews.project(pcs, v)
        xs, dens = prender.density_curve(s_all[member_mask])
        xc, densc = prender.density_curve(s_all[is_control])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=xc, y=densc, name="non-targeting", fill="tozeroy",
                       line=dict(color="#888"), opacity=0.5)
        )
        fig.add_trace(
            go.Scatter(x=xs, y=dens, name=subject, fill="tozeroy",
                       line=dict(color="#1f77b4"), opacity=0.6)
        )
        for _, r in lstats.iterrows():
            if r["n_shown"] > 0:
                fig.add_vline(
                    x=0.5 * (r["band_lo"] + r["band_hi"]),
                    line_dash="dot",
                    line_color="#d62728",
                    annotation_text=f"p{int(r['percentile'])}",
                    annotation_position="top",
                )
        fig.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="projection onto the cluster axis (control SD units)",
            yaxis_title="density",
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="paragon_density",
            config={"toImageButtonOptions": {"format": "svg"}},
        )

        for _, r in lstats.iterrows():
            pct = int(r["percentile"])
            rung = ladder[ladder["percentile"] == pct] if len(ladder) else ladder
            with st.container():
                st.markdown(f"#### p{pct}")
                if r["n_shown"] == 0:
                    st.warning(
                        r.get("warning", "no cells in this band after gating")
                    )
                    continue
                st.caption(format_stats_line(r.to_dict()))
                render_panel(
                    rung, st.container(), caption_col="residual",
                    panel=f"ladder-p{pct}",
                )

        with st.expander("Ladder statistics", expanded=False):
            st.dataframe(lstats, use_container_width=True)

    else:
        st.subheader("Interpretable-feature exemplars")
        if not pfeat.available():
            st.error(
                "The single-cell feature matrix has not been built. Run:\n\n"
                "```bash\ncd paragon_analysis && sbatch scripts/build_signatures.sbatch\n```"
            )
            st.stop()
        if level == "Gene":
            try:
                sigs = gene_signature_cached(
                    channel_combo,
                    cell_class,
                    ppaths.BOOTSTRAP_CHANNEL_COMBO,
                    selected_gene,
                )
            except psig.SignatureError as exc:
                st.error(str(exc))
                st.stop()
            subject_key = str(selected_gene)
            st.caption(
                "Significance comes from the per-construct bootstrap null "
                f"({ppaths.BOOTSTRAP_CHANNEL_COMBO}), which resamples control "
                "constructs at this gene's observed sample size - a better null "
                "than a rank test over genes, and already corrected across "
                "features."
            )
        else:
            sigs = load_signatures(channel_combo, resolution)
            if sigs.empty:
                st.error(
                    f"No signature table for {channel_combo} / LR-{resolution}. Run "
                    "`sbatch scripts/build_signatures.sbatch`."
                )
                st.stop()
            subject_key = cluster_id

        signal, _ = psig.partition(sigs, subject_key, pfeat.all_features())
        if signal.empty:
            st.warning(
                f"{subject} has no signal features: no CellProfiler feature "
                "passed both the FDR and effect-size thresholds. That is itself "
                "the finding - no individual interpretable feature defines it. "
                "Use the latent-axis mode, which can pick up a direction "
                "distributed across many weak features."
            )
            st.stop()

        st.caption(
            f"Signal set: {len(signal)} features. Cells are matched to the "
            "target profile on those, then ranked by how ordinary they are "
            "across the remaining features."
        )
        st.dataframe(signal, use_container_width=True, hide_index=True)

        # The gene-level effect is a median difference over a pooled MAD taken
        # ACROSS GENES, and gene-level MADs are small -- so a feature can show
        # |effect| = 8 while its target sits at z = 0.1. Single-cell z-scores
        # have a spread near 1, so matching such a target is nearly the same as
        # matching zero, i.e. selecting an average cell. Say so rather than let
        # the panel imply the cells were chosen for a strong phenotype.
        med_target = float(signal["target"].abs().median())
        if med_target < 0.5:
            st.warning(
                f"Median |target| across the signal set is {med_target:.2f} "
                "z-units, against a single-cell spread of about 1.0. These "
                "features separate the subject from controls at the *gene* level "
                "(large effects, because gene-level MADs are small) but the "
                "absolute shift per cell is small - so matching the target is "
                "close to selecting an ordinary cell. The latent-axis mode is "
                "the better tool here: it sums a weak signal distributed across "
                "many features into one direction, which no single feature "
                "resolves. Use this mode to learn *which* features move, and the "
                "ladder to actually see the shift."
            )

        nuisance = pfeat.nuisance_sample(signal["feature"].tolist())
        rows = np.flatnonzero(pool_mask)
        with st.spinner(f"Scoring {len(rows):,} cells on {len(signal) + len(nuisance)} features..."):
            # Indexed by cell row id and covering only the pool -- a
            # full-length frame would be a ~550 MB allocation per interaction.
            fvals = pfeat.load_feature_values(
                signal["feature"].tolist() + nuisance, rows=rows
            )
            try:
                sel, estats, attribution = pexe.select_exemplars(
                    fvals,
                    index,
                    signal,
                    nuisance,
                    pool_mask=pool_mask,
                    n_show=n_show,
                    max_per_tile=1 if diversify else 10**6,
                    min_guides=3 if diversify else 0,
                )
            except pexe.ExemplarError as exc:
                st.error(str(exc))
                st.stop()

        st.caption(
            f"{estats['n_pool']:,} pool cells -> top {estats['n_stage1']:,} on signal "
            f"match -> {estats['n_shown']} most typical elsewhere, from "
            f"{estats['n_tiles_shown']} tiles / {estats['n_guides_shown']} sgRNAs"
        )
        render_panel(
            sel, st.container(), caption_col="nuisance_norm", panel="features"
        )

        st.markdown("##### Why these cells")
        for _, row in sel.iterrows():
            st.markdown(
                f"**#{int(row['rank'])}** {row[ppaths.PERTURBATION_NAME_COL]} - "
                + pexe.describe_cell(
                    attribution,
                    int(row["rank"]),
                    row["max_nuisance_z"],
                    estats["n_nuisance_available"],
                )
            )
        with st.expander("Per-feature attribution", expanded=False):
            st.dataframe(attribution, use_container_width=True, hide_index=True)


# =====================
# CONTROLS


with tab_controls:
    st.subheader("Negative controls")
    st.caption(
        "Two checks that a panel is showing signal rather than selection "
        "artifacts. If either looks as phenotypic as the real ladder, the axis "
        "is not carrying the phenotype you think it is."
    )

    st.markdown("#### Matched non-targeting cells")
    st.caption(
        "Same axis, same percentile bands, but drawn only from non-targeting "
        "controls. This is the honest side-by-side comparison for the ladder."
    )
    with st.spinner("Selecting control exemplars..."):
        cladder, clstats = pviews.views_ladder(
            pcs,
            index,
            v,
            landmark_mask=member_mask & fit_mask if held_out else member_mask,
            pool_mask=is_control & gate_ok & (show_mask if held_out else True),
            percentiles=ladder_pcts,
            n_show=n_show,
            diversify=diversify,
            min_candidates=n_show * 3,
        )
    for _, r in clstats.iterrows():
        pct = int(r["percentile"])
        rung = cladder[cladder["percentile"] == pct] if len(cladder) else cladder
        st.markdown(f"**p{pct}** (controls)")
        if r["n_shown"] == 0:
            st.warning("no control cells in this band")
            continue
        st.caption(format_stats_line(r.to_dict()))
        render_panel(
            rung, st.container(), caption_col="residual",
            panel=f"control-p{pct}",
        )

    st.divider()
    st.markdown("#### Shuffled axis")
    shuffle_seed = st.number_input(
        "Seed", value=0, step=1, key="paragon_shuffle_seed"
    )
    st.caption(
        "A random unit vector in the same space, run through the identical "
        "pipeline. These cells were selected to be extreme along a direction "
        "with no biological meaning - they are what 'selection artifact' looks "
        "like here."
    )
    v_shuf = pviews.shuffled_axis(k, seed=int(shuffle_seed))
    sep_shuf = pax.axis_separation(pcs, v_shuf, member_mask, is_control)
    st.caption(
        f"Shuffled-axis separation: Cohen's d = {sep_shuf['cohens_d']:.3f}, "
        f"AUC = {sep_shuf['auc']:.3f} "
        f"(real axis: {sep['cohens_d']:.3f} / {sep['auc']:.3f})"
    )
    with st.spinner("Selecting shuffled-axis cells..."):
        sladder, slstats = pviews.views_ladder(
            pcs,
            index,
            v_shuf,
            landmark_mask=member_mask & fit_mask if held_out else member_mask,
            pool_mask=pool_mask,
            percentiles=ladder_pcts,
            n_show=n_show,
            diversify=diversify,
            min_candidates=n_show * 3,
        )
    for _, r in slstats.iterrows():
        pct = int(r["percentile"])
        rung = sladder[sladder["percentile"] == pct] if len(sladder) else sladder
        st.markdown(f"**p{pct}** (shuffled axis)")
        if r["n_shown"] == 0:
            st.warning("no cells in this band")
            continue
        render_panel(
            rung, st.container(), caption_col="residual",
            panel=f"shuffled-p{pct}",
        )


# =====================
# ATTRIBUTION


with tab_attr:
    st.subheader(f"What defines {subject}")
    if level == "Gene":
        try:
            cl = gene_signature_cached(
                channel_combo, cell_class, ppaths.BOOTSTRAP_CHANNEL_COMBO, selected_gene
            )
        except psig.SignatureError as exc:
            st.error(str(exc))
            cl = pd.DataFrame()
        if not cl.empty:
            st.caption(
                "Per-feature effect is this gene's control-referenced z-score "
                "(median of its construct medians). Significance is the "
                "per-construct bootstrap FDR from "
                f"`{ppaths.BOOTSTRAP_CHANNEL_COMBO}`, which resamples control "
                "constructs at the observed sample size and is already corrected "
                "across features."
            )
            if cl["fdr_bootstrap"].isna().all():
                st.warning(
                    f"No bootstrap FDRs found for {selected_gene}. Only "
                    f"{ppaths.BOOTSTRAP_CHANNEL_COMBO} was bootstrapped "
                    "(config.yml `aggregate.bootstrap_combinations`), and this "
                    "gene is absent from that table - signal features are gated "
                    "on effect size alone, so treat them as unvalidated."
                )
    else:
        sigs = load_signatures(channel_combo, resolution)
        if sigs.empty:
            st.info(
                f"No signature table for {channel_combo} / LR-{resolution}. Build it "
                "with `sbatch scripts/build_signatures.sbatch`."
            )
            cl = pd.DataFrame()
        else:
            cl = sigs[sigs["cluster"] == cluster_id].copy()
            if cl.empty:
                st.info(f"Cluster {cluster_id} was not tested.")
            else:
                n_test = int(cl["n_genes_test"].iloc[0])
                n_ctl = int(cl["n_genes_control"].iloc[0])
                st.caption(
                    f"Mann-Whitney U on **gene-level** profiles: {n_test} cluster "
                    f"genes vs {n_ctl} control pseudo-genes, effect = median "
                    "difference over pooled MAD, Benjamini-Hochberg across "
                    f"{len(cl):,} tested features."
                )
                if n_test < 6:
                    st.warning(
                        f"Only {n_test} genes in this cluster, so this gene-level "
                        "rank test has very little power no matter how many cells "
                        "underlie it - absence of signal features here is not "
                        "evidence of absence of a phenotype. The per-gene view "
                        "uses the cell-count-aware bootstrap null instead, which "
                        "does not have this limitation."
                    )

    if not cl.empty:
        n_signal = int(cl["is_signal"].sum())
        st.caption(f"{n_signal} signal feature(s) of {len(cl):,} tested.")
        sig_only = st.checkbox(
            "Signal features only", value=True, key="paragon_sig_only"
        )
        show = cl[cl["is_signal"]] if sig_only else cl
        sort_col = "rank" if sig_only else ("fdr_bh" if level == "Gene" else "p_value")
        st.dataframe(
            show.sort_values(sort_col, na_position="last"),
            use_container_width=True,
            hide_index=True,
        )

        top = cl[cl["is_signal"]].sort_values("rank").head(15)
        if len(top):
            fig = go.Figure(
                go.Bar(
                    x=top["effect"],
                    y=top["feature"],
                    orientation="h",
                    marker_color=[
                        "#d62728" if e > 0 else "#1f77b4" for e in top["effect"]
                    ],
                )
            )
            fig.update_layout(
                height=30 * len(top) + 120,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title=(
                    "control-referenced z-score"
                    if level == "Gene"
                    else "robust-z effect vs non-targeting"
                ),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(
                fig, use_container_width=True, key="paragon_effects",
                config={"toImageButtonOptions": {"format": "svg"}},
            )


# =====================
# DIAGNOSTICS


with tab_diag:
    st.subheader("Diagnostics")

    st.markdown("#### Why K defaults to 25")
    diag = load_pc_diagnostics()
    if diag.empty:
        st.info("PC diagnostics not available; rebuild the cell index.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=diag["pc"], y=diag["control_std"], name="control SD",
                line=dict(color="#888")
            )
        )
        fig.add_trace(
            go.Scatter(
                x=diag["pc"], y=diag["perturbed_std"], name="perturbed SD",
                line=dict(color="#d62728")
            )
        )
        fig.add_vline(
            x=k, line_dash="dash", line_color="#1f77b4",
            annotation_text=f"K = {k}", annotation_position="top",
        )
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="PC index",
            yaxis_title="SD along PC",
            yaxis_type="log",
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(
            fig, use_container_width=True, key="paragon_pc_diag",
            config={"toImageButtonOptions": {"format": "svg"}},
        )
        st.caption(
            "`tvn_on_controls` fits its PCA on control cells and rescales every PC "
            "to unit control SD, so the trailing PCs are the lowest-variance "
            "control directions and that rescaling amplifies a handful of "
            "pathological cells enormously. On PC 187 the top 10 cells carry 99.7% "
            "of the squared magnitude, and PCs 180-187 together account for 73.7% "
            "of the mean squared norm over all 188. A residual computed over the "
            "full space therefore ranks cells by how artifactual they are, not by "
            "how typical. K=25 matches the paper and keeps every included PC's "
            "perturbed SD within 0.80-0.96."
        )
        tail = diag[diag["pc"] >= k]
        if len(tail):
            st.caption(
                f"Excluded by the current K: {len(tail)} PCs, max perturbed SD "
                f"{tail['perturbed_std'].max():.1f}, max |value| "
                f"{tail['abs_max'].max():.0f}."
            )

    st.divider()
    st.markdown("#### Axis agreement across methods")
    st.caption(
        "Cosine similarity between the three axis definitions for this cluster. "
        "Low agreement means the cluster has no single coherent phenotypic "
        "direction, and no exemplar panel built on one of them is trustworthy."
    )
    built = {}
    for m_name in pax.METHODS:
        try:
            if m_name == "centroid":
                built[m_name] = pax.axis_centroid(pcs, member_mask, is_control)
            elif m_name == "cluster_lda":
                built[m_name] = pax.axis_cluster_lda(pcs, member_mask, is_control)
            elif gene_names is not None:
                built[m_name], _ = pax.axis_gene_ld_mean(
                    gene_names, gene_matrix, genes
                )
        except pax.AxisError:
            continue
    if len(built) > 1:
        agree = pax.axis_agreement(built)
        st.dataframe(agree.style.format("{:.3f}"), use_container_width=True)
        offdiag = agree.to_numpy()[~np.eye(len(agree), dtype=bool)]
        if offdiag.min() < 0.5:
            st.warning(
                f"Minimum pairwise cosine is {offdiag.min():.3f}. The three "
                "definitions disagree substantially about this cluster's direction."
            )
        sep_rows = []
        for m_name, vv in built.items():
            s = pax.axis_separation(pcs, vv, member_mask, is_control)
            sep_rows.append({"method": m_name, **{kk: s[kk] for kk in ("cohens_d", "auc")}})
        st.dataframe(pd.DataFrame(sep_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Only one axis method is available for this cluster.")

    st.divider()
    st.markdown("#### Quality gates")
    st.dataframe(
        pspaces.gate_drop_counts(index), use_container_width=True, hide_index=True
    )
    st.caption(
        "`n_dropped_uniquely` is how many cells only that gate rejects - it "
        "identifies which gate is actually binding rather than merely correlated "
        "with the others."
    )

    st.divider()
    st.markdown("#### Gene LD cross-correlation (Labitigan Figure 6)")
    if gene_names is None:
        st.info(
            f"Per-gene LD vectors for K={k} have not been built. Run "
            "`sbatch scripts/build_cluster_axes.sbatch`."
        )
    else:
        st.caption(
            f"{len(gene_names)} genes with an LD vector. Correlation between gene "
            "LD vectors gives a clustering of the screen that is independent of "
            "the PHATE/Leiden one, so the two can be compared."
        )
        scope = st.radio(
            "Scope",
            ["This cluster", "This cluster + nearest genes"],
            horizontal=True,
            key="paragon_ld_scope",
        )
        corr = pax.gene_ld_crosscorr(gene_names, gene_matrix)
        in_cluster = [g for g in genes if g in corr.index]
        if not in_cluster:
            st.info("None of this cluster's genes have an LD vector.")
        else:
            if scope == "This cluster":
                sub = corr.loc[in_cluster, in_cluster]
            else:
                mean_to_cluster = corr.loc[:, in_cluster].mean(axis=1)
                extra = (
                    mean_to_cluster.drop(index=in_cluster)
                    .sort_values(ascending=False)
                    .head(15)
                    .index.tolist()
                )
                keep = in_cluster + extra
                sub = corr.loc[keep, keep]
            fig = go.Figure(
                go.Heatmap(
                    z=sub.to_numpy(),
                    x=sub.columns,
                    y=sub.index,
                    zmin=-1,
                    zmax=1,
                    colorscale="RdBu",
                    reversescale=True,
                )
            )
            fig.update_layout(
                height=max(320, 22 * len(sub) + 140),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(
                fig, use_container_width=True, key="paragon_ld_corr",
                config={"toImageButtonOptions": {"format": "svg"}},
            )
            within = sub.loc[in_cluster, in_cluster].to_numpy()
            offdiag = within[~np.eye(len(in_cluster), dtype=bool)]
            if offdiag.size:
                st.caption(
                    f"Mean within-cluster LD correlation: {offdiag.mean():.3f} "
                    f"(median {np.median(offdiag):.3f}). Near zero means the "
                    "cluster's genes do not share a phenotypic direction even "
                    "though PHATE/Leiden grouped them."
                )

    st.divider()
    st.markdown("#### Axis meta (all clusters at this resolution)")
    meta = load_axes_meta(channel_combo, resolution)
    if meta.empty:
        st.info("Not built yet.")
    else:
        st.dataframe(meta, use_container_width=True, hide_index=True)
