import os
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get paths for visualization
BRIEFLOW_OUTPUT_PATH = os.environ["BRIEFLOW_OUTPUT_PATH"]
CONFIG_PATH = os.environ["CONFIG_PATH"]
SCREEN_PATH = os.environ["SCREEN_PATH"]

# External scRNA-seq aging data (TSV export of the 'Ranked_Summary' sheet).
# Optional: when unset/missing, the scRNAseq_aging_data option is hidden.
SCRNASEQ_AGING_PATH = os.environ.get("SCRNASEQ_AGING_PATH", None)

# External bulk RNA-seq DEG data (TSV exports of the CZ ADCP workbooks; see
# external_data/export_deg_sheets.py). Optional: when unset/missing, the
# corresponding feature is hidden from the dropdowns.
ADCP_6HR_DEG_PATH = os.environ.get("ADCP_6HR_DEG_PATH", None)
ADCP_12HR_DEG_PATH = os.environ.get("ADCP_12HR_DEG_PATH", None)
ADCP_24HR_DEG_PATH = os.environ.get("ADCP_24HR_DEG_PATH", None)
ADCP_EFFERO_DEG_PATH = os.environ.get("ADCP_EFFERO_DEG_PATH", None)
EFFERO_SUPEREFFERO_DEG_PATH = os.environ.get("EFFERO_SUPEREFFERO_DEG_PATH", None)

# External CRISPR screen (casTLE) results used by the External Screen Comparison
# page. ADCP is read straight from its CSV; the genome-wide bead screen is read
# from the TSV export of beads_genome_wide.xlsx (see
# external_data/export_screens.py). Optional: when unset/missing, the screen is
# hidden from the dropdown.
ADCP_COMBO_PATH = os.environ.get("ADCP_COMBO_PATH", None)
BEADS_GENOME_WIDE_PATH = os.environ.get("BEADS_GENOME_WIDE_PATH", None)

# External gDNA guide-count CSV (headerless '<guide name>,<count>'), used by the
# gene-abundance comparison on the Quality Control page. Optional: when
# unset/missing, that figure is replaced by a short note.
GDNA_COUNTS_PATH = os.environ.get("GDNA_COUNTS_PATH", None)

# Protein-complex level analysis outputs (gene_sets.tsv, complex_features,
# complex_bootstrap_results, complex_coherence, ...). Produced by
# scripts/complex_analysis/. Defaults to the pipeline's own output tree, so no
# new export is needed in 14.run_visualization.sh; override only if the complex
# outputs live somewhere else.
COMPLEX_OUTPUT_PATH = os.environ.get(
    "COMPLEX_OUTPUT_PATH", os.path.join(BRIEFLOW_OUTPUT_PATH, "complex")
)

# Static asset configuration - these can be None for local development
STATIC_ASSET_URL_ROOT = os.environ.get(
    "STATIC_ASSET_URL_ROOT", None
)  # e.g. "/aconcagua_dataset_static/"
STATIC_ASSET_PATH = os.environ.get(
    "STATIC_ASSET_PATH", None
)  # e.g. "/disk1/brieflow_datasets/aconcagua/"

logger.info(f"CONFIG_PATH: {os.path.abspath(CONFIG_PATH)}")
logger.info(f"SCREEN_PATH: {os.path.abspath(SCREEN_PATH)}")


def load_config():
    """Load the YAML configuration file."""
    try:
        with open(CONFIG_PATH, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Config file not found at: {os.path.abspath(CONFIG_PATH)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        raise
