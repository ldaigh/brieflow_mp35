import os
import re

import streamlit as st

st.set_page_config(
    page_title="Timepoint Landscape - Brieflow Analysis",
    layout="wide",
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =====================
# CONFIGURATION

# Directory holding the pre-computed landscape TSVs written by
# cs231n_analysis/scripts/compute_timepoint_landscape.py
TIMEPOINT_OUTPUT_PATH = os.environ.get("TIMEPOINT_OUTPUT_PATH", "")

# Cell-level joined table (for ECDF overlays). Optional — the page still works
# without it, just without the per-cell distribution panel.
JOINED_PARQUET = os.environ.get(
    "TIMEPOINT_JOINED_PARQUET",
    "/scratch/users/ldaigh/brieflow-MP35/brieflow_mp35_analysis/"
    "cs231n_analysis/data/ops_dino_joined.parquet",
)

# Safe controls share a constant gene_id ("0Safe") but a per-guide unique
# gene_symbol, so the control mask uses gene_id.
CONTROL_ID = "0Safe"
TP_COL = "inferred_timepoint"


# =====================
# DATA LOADING


@st.cache_data
def load_tsv(name: str) -> pd.DataFrame:
    path = os.path.join(TIMEPOINT_OUTPUT_PATH, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


@st.cache_data
def load_cell_timepoints() -> pd.DataFrame:
    """Load only (gene_symbol, inferred_timepoint) for ECDF overlays."""
    if not os.path.exists(JOINED_PARQUET):
        return pd.DataFrame()
    return pd.read_parquet(JOINED_PARQUET, columns=["gene_symbol", "gene_id", TP_COL])


# =====================
# HELPERS


def parse_highlight(text: str, names: list[str]) -> set[str]:
    """Match a comma-separated list of terms (substring, case-insensitive, or
    regex) against the gene/guide names and return the matched set."""
    if not text.strip():
        return set()
    hits: set[str] = set()
    for term in [t.strip() for t in text.split(",") if t.strip()]:
        try:
            pat = re.compile(term, re.IGNORECASE)
            hits.update(n for n in names if pat.search(n))
        except re.error:
            low = term.lower()
            hits.update(n for n in names if low in n.lower())
    return hits


# =====================
# MAIN

st.title("Inferred Timepoint Landscape")
st.caption(
    "Where each perturbation sits on the inferred-timepoint axis relative to "
    "the safe-targeting controls. Built for subtle, guide-concordant shifts — "
    "not big single outliers."
)

if not TIMEPOINT_OUTPUT_PATH:
    st.error(
        "TIMEPOINT_OUTPUT_PATH environment variable is not set. "
        "Add it to 14.run_visualization.sh and restart the server."
    )
    st.stop()

gene_df = load_tsv("per_gene_timepoint.tsv")
guide_df = load_tsv("per_guide_timepoint.tsv")
meta_df = load_tsv("landscape_meta.tsv")

if gene_df.empty or guide_df.empty:
    st.warning(
        f"No landscape tables found in `{TIMEPOINT_OUTPUT_PATH}`. "
        "Run scripts/compute_timepoint_landscape.py (sbatch) to generate them."
    )
    st.stop()

safe_mean = float(meta_df["safe_mean"].iloc[0]) if not meta_df.empty else float(
    gene_df.loc[gene_df["is_control"], "gene_mean"].mean()
)
# Safe envelope = spread of the safe guides' means (the real noise floor).
safe_guides = guide_df[guide_df["is_control"]]
safe_env = float(safe_guides["tp_mean"].std(ddof=1)) if len(safe_guides) > 1 else 0.0

# ---- Sidebar controls ----
with st.sidebar:
    st.markdown("## Landscape Controls")

    level = st.radio("**Level**", ["Per Gene", "Per Guide"], key="tp_level")

    if level == "Per Gene":
        min_guides = st.slider(
            "Min guides / gene", 1, 6,
            int(meta_df["min_guides_gene"].iloc[0]) if not meta_df.empty else 2,
        )
    else:
        min_cells = st.slider("Min cells / guide", 20, 500, 20, step=10)

    sort_by = st.selectbox(
        "Sort by", ["Mean timepoint", "Delta vs safe", "FDR q-value"],
        key="tp_sort",
    )

    color_mode = st.radio(
        "**Color points by**", ["Delta (blue→red)", "Significance", "None"],
        key="tp_color",
    )
    fdr_thresh = st.slider("FDR q threshold", 0.0, 0.25, 0.05, step=0.01)

    st.markdown("---")
    highlight_text = st.text_input(
        "**Highlight** (comma-separated, substring or regex)",
        key="tp_highlight",
        placeholder="e.g.  Vps, ^Rab, Atg",
    )
    show_labels = st.checkbox("Label highlighted points", value=True)
    show_safe_band = st.checkbox("Show safe control band", value=True)

# ---- Build the working frame ----
if level == "Per Gene":
    d = gene_df.copy()
    d["name"] = d["gene_symbol"].astype(str)
    d["value"] = d["gene_mean"]
    d = d[(d["n_guides"] >= min_guides) | (d["is_control"])]
    err_lo = d["value"] - d["ci_lo"]
    err_hi = d["ci_hi"] - d["value"]
else:
    d = guide_df.copy()
    d["name"] = d["cell_barcode"].astype(str)
    d["value"] = d["tp_mean"]
    d = d[d["n_cells"] >= min_cells]
    err_lo = err_hi = None

# sort
if sort_by == "Mean timepoint":
    d = d.sort_values("value")
elif sort_by == "Delta vs safe":
    d = d.sort_values("delta")
else:
    key = "q_value" if "q_value" in d.columns else "p_vs_safe"
    d = d.sort_values(key, na_position="last")
d = d.reset_index(drop=True)
d["rank"] = np.arange(len(d))

# ---- Summary metrics ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Items shown", len(d))
if level == "Per Gene":
    sig = int((d["q_value"] < fdr_thresh).sum())
    c2.metric(f"FDR<{fdr_thresh:g}", sig)
else:
    sig = int((d["p_vs_safe"] < 0.05).sum())
    c2.metric("p<0.05 (uncorr.)", sig)
c3.metric("Safe mean", f"{safe_mean:.2f}")
c4.metric("Safe envelope ±", f"{safe_env:.3f}")

# ---- Highlight set ----
highlight = parse_highlight(highlight_text, d["name"].tolist())
if highlight_text.strip():
    st.caption(f"Highlight matched **{len(highlight)}** items.")

# ---- Colors ----
if color_mode == "Delta (blue→red)":
    vmax = float(np.nanpercentile(np.abs(d["delta"]), 98)) or 1.0
    base_color = d["delta"].to_numpy()
    marker_kw = dict(
        color=base_color, colorscale="RdBu_r", cmin=-vmax, cmax=vmax,
        colorbar=dict(title="Δ vs safe"),
    )
elif color_mode == "Significance":
    if level == "Per Gene":
        sigmask = (d["q_value"] < fdr_thresh).fillna(False)
    else:
        sigmask = (d["p_vs_safe"] < 0.05).fillna(False)
    marker_kw = dict(color=np.where(sigmask, "#c0392b", "#b0b0b0"))
else:
    marker_kw = dict(color="#7f8c8d")

# ---- Figure ----
fig = go.Figure()

if show_safe_band and safe_env > 0:
    fig.add_hrect(
        y0=safe_mean - 1.96 * safe_env, y1=safe_mean + 1.96 * safe_env,
        fillcolor="rgba(120,120,120,0.15)", line_width=0,
    )
fig.add_hline(y=safe_mean, line_dash="dash", line_color="gray",
              annotation_text="safe mean", annotation_position="right")

customdata = np.stack([
    d["name"],
    d["delta"].round(3),
    (d["q_value"].round(4) if "q_value" in d.columns else d["p_vs_safe"].round(4)),
    (d["n_guides"] if "n_guides" in d.columns else d["n_cells"]),
], axis=-1)

hovertemplate = (
    "%{customdata[0]}<br>"
    "mean tp=%{y:.3f}<br>"
    "Δ vs safe=%{customdata[1]}<br>"
    + ("q=%{customdata[2]}<br>n_guides=%{customdata[3]}" if level == "Per Gene"
       else "p=%{customdata[2]}<br>n_cells=%{customdata[3]}")
    + "<extra></extra>"
)

error_y = (dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo,
                thickness=0.8, width=0, color="rgba(120,120,120,0.4)")
           if err_lo is not None else None)

# Scattergl (WebGL) is fast but rasterizes points on SVG export. When the
# user enables vector export, use go.Scatter (SVG) so each point exports as
# an individually selectable vector shape.
_trace_cls = go.Scatter if st.session_state.get("tp_vector_export", False) else go.Scattergl

fig.add_trace(_trace_cls(
    x=d["rank"], y=d["value"], mode="markers",
    marker=dict(size=6, **marker_kw),
    error_y=error_y,
    customdata=customdata,
    hovertemplate=hovertemplate,
    showlegend=False,
))

# highlighted overlay
if highlight:
    hd = d[d["name"].isin(highlight)]
    fig.add_trace(_trace_cls(
        x=hd["rank"], y=hd["value"],
        mode="markers+text" if show_labels else "markers",
        marker=dict(size=11, color="#111", symbol="circle-open",
                    line=dict(width=2.5, color="#111")),
        text=hd["name"] if show_labels else None,
        textposition="top center",
        textfont=dict(size=10),
        customdata=np.stack([
            hd["name"], hd["delta"].round(3),
            (hd["q_value"].round(4) if "q_value" in hd.columns else hd["p_vs_safe"].round(4)),
            (hd["n_guides"] if "n_guides" in hd.columns else hd["n_cells"]),
        ], axis=-1),
        hovertemplate=hovertemplate,
        showlegend=False,
    ))

fig.update_layout(
    height=650,
    xaxis_title=f"rank (ordered by {sort_by.lower()})",
    yaxis_title="inferred timepoint",
    hovermode="closest",
    margin=dict(l=60, r=30, t=30, b=50),
)

st.checkbox(
    "Vectorize points for SVG export (slower render; makes points selectable in Illustrator)",
    key="tp_vector_export",
)
st.plotly_chart(fig, use_container_width=True, key="tp_landscape", config={"toImageButtonOptions": {"format": "svg"}})

# ---- Detail panel: pick a gene/guide to inspect ----
st.markdown("### Inspect")
options = ["— none —"] + d["name"].tolist()
sel = st.selectbox("Gene / guide", options, key="tp_inspect")

if sel != "— none —":
    if level == "Per Gene":
        row = gene_df[gene_df["gene_symbol"] == sel]
        st.markdown(f"#### {sel}")
        if not row.empty:
            r = row.iloc[0]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Mean tp", f"{r['gene_mean']:.3f}")
            m2.metric("Δ vs safe", f"{r['delta']:+.3f}")
            m3.metric("Guides", int(r["n_guides"]))
            m4.metric("Concordance", f"{r['concordance']:.2f}"
                      if pd.notna(r["concordance"]) else "—")
            m5.metric("FDR q", f"{r['q_value']:.3g}"
                      if pd.notna(r["q_value"]) else "—")

        # its guides
        gsub = guide_df[guide_df["gene_symbol"] == sel].sort_values("tp_mean")
        st.markdown("**Per-guide means for this gene**")
        st.dataframe(
            gsub[["cell_barcode", "n_cells", "tp_mean", "delta", "z_vs_safe",
                  "p_vs_safe"]].set_index("cell_barcode"),
            use_container_width=True, height=200,
        )

        # ECDF vs safe pool
        cells = load_cell_timepoints()
        if not cells.empty:
            gene_cells = cells.loc[cells["gene_symbol"] == sel, TP_COL].to_numpy()
            safe_cells = cells.loc[cells["gene_id"] == CONTROL_ID, TP_COL].to_numpy()

            def ecdf(a):
                a = np.sort(a)
                return a, np.arange(1, len(a) + 1) / len(a)

            efig = go.Figure()
            xs, ys = ecdf(safe_cells)
            efig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                      name="safe controls", line=dict(color="gray")))
            xs, ys = ecdf(gene_cells)
            efig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                      name=sel, line=dict(color="#c0392b")))
            efig.update_layout(
                height=350, xaxis_title="inferred timepoint",
                yaxis_title="cumulative fraction",
                title=f"Cell-level distribution: {sel} (n={len(gene_cells)}) "
                      f"vs safe (n={len(safe_cells)})",
                margin=dict(l=60, r=30, t=50, b=50),
            )
            st.plotly_chart(efig, use_container_width=True, key="tp_ecdf", config={"toImageButtonOptions": {"format": "svg"}})
    else:
        row = guide_df[guide_df["cell_barcode"] == sel]
        if not row.empty:
            r = row.iloc[0]
            st.markdown(f"#### {sel}  —  {r['gene_symbol']}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean tp", f"{r['tp_mean']:.3f}")
            m2.metric("Δ vs safe", f"{r['delta']:+.3f}")
            m3.metric("Cells", int(r["n_cells"]))
            m4.metric("z vs safe", f"{r['z_vs_safe']:+.2f}")
