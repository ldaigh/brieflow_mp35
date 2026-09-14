import os
import numpy as np
import pandas as pd
import streamlit as st
import sys
import uuid
import tifffile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from workflow.lib.shared.file_utils import parse_filename
from src.config import STATIC_ASSET_URL_ROOT, STATIC_ASSET_PATH


# Named colors selectable in the composite UI, mapped to additive RGB weights.
COMPOSITE_COLOR_OPTIONS = {
    "Cyan": (0.0, 1.0, 1.0),
    "Yellow": (1.0, 1.0, 0.0),
    "Magenta": (1.0, 0.0, 1.0),
    "Blue": (0.0, 0.0, 1.0),
    "Red": (1.0, 0.0, 0.0),
    "Green": (0.0, 1.0, 0.0),
    "Orange": (1.0, 0.5, 0.0),
    "White": (1.0, 1.0, 1.0),
    "Gray": (0.7, 0.7, 0.7),
}

# Default channels to composite into a single colored montage.
# `index` is the channel position within the overlay_montage.tiff stack,
# which follows config["phenotype"]["channel_names"]:
# [DAPI, CFP, YFP, RFP, Cy5] -> 0, 1, 2, 3, 4.
# `color` is the default color name (user-selectable in the UI).
DEFAULT_COMPOSITE_CHANNELS = [
    {"index": 0, "name": "DAPI", "color": "Cyan"},
    {"index": 2, "name": "YFP", "color": "Yellow"},
    {"index": 3, "name": "RFP", "color": "Magenta"},
    {"index": 4, "name": "Cy5", "color": "Blue"},
]


def _normalize_channel(channel, low_pct, high_pct):
    """Rescale a 2D channel to [0, 1] using percentile clipping.

    Percentiles are computed over non-zero pixels only. The montage canvas is
    zero-padded (edge crops and unfilled grid cells are exactly 0), and those
    zeros otherwise dominate the low percentiles, making the black-point control
    unresponsive. Restricting to signal pixels makes the sliders track the real
    intensity distribution.

    Returns:
        tuple: (normalized 2D array in [0, 1], lo cutoff, hi cutoff).
    """
    channel = channel.astype(np.float32)
    signal = channel[channel > 0]
    if signal.size == 0:
        return np.zeros_like(channel), 0.0, 0.0
    lo, hi = np.percentile(signal, (low_pct, high_pct))
    if hi <= lo:
        return np.zeros_like(channel), float(lo), float(hi)
    norm = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
    return norm, float(lo), float(hi)


def render_composite_montage(
    overlay_tiff_path,
    key_prefix,
    composite_channels=DEFAULT_COMPOSITE_CHANNELS,
    container=None,
):
    """Render an additive colored composite from an overlay_montage.tiff stack.

    Reads the multi-channel montage TIFF, lets the user toggle channels on/off,
    pick a color per channel, and adjust per-channel contrast, then composites
    the selected channels into a single RGB image.

    Args:
        overlay_tiff_path (str): Path to the stacked overlay_montage.tiff.
        key_prefix (str): Unique prefix for Streamlit widget keys (per page/gene/guide).
        composite_channels (list[dict]): Channel specs with 'index', 'name', and
            default 'color' keys. Defaults to DAPI(cyan)/YFP(yellow)/RFP(magenta)/Cy5(blue).
        container: Optional Streamlit container to render into. Use a full-width
            container (created at page top level) to escape a narrow column.
            Defaults to inline rendering.
    """
    target = container if container is not None else st.container()
    with target:
        _render_composite_montage_body(
            overlay_tiff_path, key_prefix, composite_channels
        )


def _render_composite_montage_body(overlay_tiff_path, key_prefix, composite_channels):
    """Body of render_composite_montage (assumes the target container is active)."""
    try:
        stack = tifffile.imread(overlay_tiff_path)
    except Exception as e:
        st.error(f"Could not read overlay TIFF for composite: {e}")
        return

    # Expect (channels, height, width). A single-channel file loads as 2D.
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    n_channels = stack.shape[0]

    color_names = list(COMPOSITE_COLOR_OPTIONS.keys())

    with st.expander("Colored channel composite", expanded=True):
        rgb = np.zeros(stack.shape[1:] + (3,), dtype=np.float32)
        any_selected = False

        for spec in composite_channels:
            idx = spec["index"]
            if idx >= n_channels:
                st.caption(
                    f"Channel {spec['name']} (index {idx}) not in stack "
                    f"({n_channels} channels); skipping."
                )
                continue

            cols = st.columns([1, 1, 3])
            with cols[0]:
                enabled = st.checkbox(
                    spec["name"],
                    value=True,
                    key=f"{key_prefix}_cmp_on_{idx}",
                )
            with cols[1]:
                default_color = spec.get("color", color_names[0])
                color_name = st.selectbox(
                    f"{spec['name']} color",
                    color_names,
                    index=color_names.index(default_color)
                    if default_color in color_names
                    else 0,
                    key=f"{key_prefix}_cmp_color_{idx}",
                    disabled=not enabled,
                    label_visibility="collapsed",
                )
            with cols[2]:
                low_pct, high_pct = st.slider(
                    f"{spec['name']} contrast (percentile of signal)",
                    min_value=0.0,
                    max_value=100.0,
                    value=(1.0, 99.0),
                    step=0.5,
                    key=f"{key_prefix}_cmp_pct_{idx}",
                    disabled=not enabled,
                )

            if not enabled:
                continue

            any_selected = True
            norm, lo, hi = _normalize_channel(stack[idx], low_pct, high_pct)
            rgb += norm[..., np.newaxis] * np.array(
                COMPOSITE_COLOR_OPTIONS[color_name], dtype=np.float32
            )
            st.caption(
                f"{spec['name']} → {color_name} | "
                f"display range [{lo:.0f}, {hi:.0f}]"
            )

        if not any_selected:
            st.info("Enable at least one channel to render the composite.")
            return

        rgb = np.clip(rgb, 0.0, 1.0)
        st.image(rgb, caption="Composite overlay", use_container_width=True)


class VisualizationRenderer:
    @staticmethod
    def display_plots_and_tables(filtered_df, root_dir):
        # Check if the root directory exists
        if not os.path.exists(root_dir):
            st.error(f"Analysis root directory does not exist: {root_dir}")
            return

        if filtered_df.empty:
            st.warning("No data found matching the selected filters.")
            return

        # Group by directory and basename
        grouped = filtered_df.groupby(["dir", "basename"])
        # Iterate through each group
        for (dir_name, base_name), group_df in grouped:
            with st.container():
                attrs, metric_name, _ = parse_filename(base_name)
                metric_title = metric_name.replace("_", " ").title()
                attr_parts = [
                    f"{k.replace('_', ' ').title()}: {v}" for k, v in attrs.items()
                ]
                title = f"{metric_title} - " + ", ".join(attr_parts)
                st.markdown(f"### {title}")

                # Count only the items we'll actually display
                display_items = []
                for _, row in group_df.iterrows():
                    has_png = any(r["ext"] == "png" for _, r in group_df.iterrows())
                    if row["ext"] == "png" or (row["ext"] == "tsv" and not has_png):
                        display_items.append(row)

                # Create columns based on actual display items
                cols = st.columns(min(3, len(display_items)))

                for idx, row in enumerate(display_items):
                    col_idx = idx % len(cols)
                    with cols[col_idx]:
                        # Check if this group has both PNG and TSV
                        has_png = any(r["ext"] == "png" for _, r in group_df.iterrows())
                        has_tsv = any(r["ext"] == "tsv" for _, r in group_df.iterrows())

                        if row["ext"] == "png":
                            # Always show PNG if it exists
                            try:
                                st.image(
                                    os.path.join(root_dir, row["file_path"]),
                                    caption=f"{row['metric_name']} - {row['well_id']}",
                                )
                            except Exception as e:
                                st.error(f"Could not load image: {row['file_path']}")
                                st.error(str(e))

                            # If there's a corresponding TSV, add download link
                            if has_tsv:
                                tsv_row = group_df[group_df["ext"] == "tsv"].iloc[0]
                                tsv_path = os.path.join(root_dir, tsv_row["file_path"])
                                if STATIC_ASSET_URL_ROOT and STATIC_ASSET_PATH:
                                    # Use nginx-served static files when configured
                                    relative_path = tsv_path.replace(
                                        STATIC_ASSET_PATH, ""
                                    )
                                    static_url = (
                                        f"{STATIC_ASSET_URL_ROOT}{relative_path}"
                                    )
                                    st.markdown(f"[Download TSV data]({static_url})")
                                else:
                                    # Fall back to direct download when running locally
                                    with open(tsv_path, "rb") as f:
                                        st.download_button(
                                            label="Download TSV data",
                                            data=f,
                                            file_name=os.path.basename(tsv_path),
                                            key=f"download_{str(uuid.uuid4())}",
                                        )

                        elif row["ext"] == "tsv" and not has_png:
                            # Only show TSV if there's no PNG
                            try:
                                tsv_data = pd.read_csv(
                                    os.path.join(root_dir, row["file_path"]), sep="\t"
                                )
                                st.dataframe(tsv_data)
                            except Exception as e:
                                st.error(f"Error reading TSV file: {e}")

                st.markdown("---")  # Add a separator between groups
