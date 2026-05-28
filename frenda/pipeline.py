"""
FRENDA - Fast Retrieval of ENzyme DAta
=======================================
Main pipeline: converts proteomics input (EC numbers + abundances) into
human-readable kinetic model parameter tables.

Typical usage (CLI):
    python -m frenda.pipeline proteome.csv
    python -m frenda.pipeline proteome.csv --organism "Escherichia coli" \\
        --first-reaction-only --output-dir results/

Python API:
    from frenda.pipeline import run_pipeline
    rxn_df, sbm_df = run_pipeline("proteome.csv", brenda_file="brenda_download.txt")

Outputs (written to --output-dir):
    Reaction.csv             - Reactions with substrates, products, Km, Kcat, KI, Keq
    SpeciesBaseMechanisms.csv - Enzymes and metabolites with starting concentrations
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from equilibrator_api import ComponentContribution

from .brenda import (
    get_brenda_ki_values,
    get_brenda_parameters,
    get_enzyme_name,
    load_brenda,
    map_compounds_to_kegg,
    map_ki_to_kegg,
)
from .catpred import get_kinetics_catpred, load_cache, save_cache
from .cleaning import calculate_kcat_f_r, clean_reactions
from .concentrations import assign_metabolite_concentrations, build_species_df
from .kegg import get_substrates_and_products, load_compound_names, load_kegg_data
from .thermo import calculate_keq, load_thermo_cache, save_thermo_cache

logger = logging.getLogger(__name__)

_KEGG_ID_RE = re.compile(r"C\d+")


# ---------------------------------------------------------------------------
# Per-EC assembly
# ---------------------------------------------------------------------------

def _kegg_ids_in(cell: str) -> list[str]:
    if not isinstance(cell, str):
        return []
    return _KEGG_ID_RE.findall(cell)


def assemble_ec(
    ec: str,
    brenda,
    ECs: dict,
    RXNs: dict,
    cc,
    compound_names: dict,
    compound_cache: dict,
    sequence_cache: dict,
    smiles_cache: dict,
    use_all_reactions: bool,
    organism: str | None,
    catpred_url: str | None,
) -> pd.DataFrame | None:
    """Build a DataFrame of reaction rows for one EC number.

    For each KEGG reaction annotated to the EC:
      1. Substrates and products are retrieved from KEGG.
      2. Km and kcat are taken from BRENDA (averaged, wild-type preferred).
         Compound names are mapped to KEGG IDs via eQuilibrator.
      3. For any compound still missing a value, CatPred is called using the
         UniProt sequence and PubChem SMILES (if catpred_url is set).
      4. KI values are retrieved from BRENDA and mapped to KEGG IDs.

    Returns None if the EC cannot be found in either KEGG or BRENDA.
    """
    # ---- KEGG substrates / products ----------------------------------------
    try:
        sub_prod = get_substrates_and_products(ec, ECs, RXNs, use_all_reactions)
    except (KeyError, ValueError) as exc:
        logger.warning("EC %s: KEGG lookup failed — %s", ec, exc)
        return None

    # ---- BRENDA kinetics (Km, kcat) ----------------------------------------
    enzyme_name = get_enzyme_name(brenda, ec)
    brenda_params = get_brenda_parameters(brenda, ec, organism)
    km_dict, kcat_dict = map_compounds_to_kegg(brenda_params, cc, compound_cache)

    # ---- BRENDA inhibitors (KI) --------------------------------------------
    ki_params = get_brenda_ki_values(brenda, ec, organism)
    inhibitor_str, ki_str = map_ki_to_kegg(ki_params, cc, compound_cache)

    sub_prod.insert(0, "Enzyme", enzyme_name or ec)
    sub_prod["EC"] = ec
    sub_prod["Mechanism"] = "MRL"  # Modular Rate Law

    # ---- Build Km / Kcat strings for each reaction row ---------------------
    def _kinetics_for_row(row: pd.Series) -> pd.Series:
        kegg_ids = _kegg_ids_in(row["Substrates"]) + _kegg_ids_in(row["Products"])
        km_parts, kcat_parts = [], []

        for cid in kegg_ids:
            km_val = km_dict.get(f"Km_{cid}")
            kcat_val = kcat_dict.get(f"Kcat_{cid}")

            # Fall back to CatPred for any missing value
            if (km_val is None or kcat_val is None) and catpred_url:
                pred_kcat, pred_km = get_kinetics_catpred(
                    ec=ec,
                    kegg_id=cid,
                    catpred_url=catpred_url,
                    organism=organism,
                    compound_names=compound_names,
                    sequence_cache=sequence_cache,
                    smiles_cache=smiles_cache,
                )
                if km_val is None:
                    km_val = pred_km
                if kcat_val is None:
                    kcat_val = pred_kcat

            km_parts.append(f"Km_{cid}: {km_val if km_val is not None else np.nan}")
            kcat_parts.append(f"Kcat_{cid}: {kcat_val if kcat_val is not None else np.nan}")

        return pd.Series({
            "Km": "; ".join(km_parts),
            "Kcat": "; ".join(kcat_parts),
        })

    sub_prod[["Km", "Kcat"]] = sub_prod.apply(_kinetics_for_row, axis=1)
    sub_prod["Inhibitors"] = inhibitor_str if inhibitor_str else np.nan
    sub_prod["KI"] = ki_str if ki_str else np.nan

    return sub_prod


# ---------------------------------------------------------------------------
# Species (enzyme) DataFrame builder
# ---------------------------------------------------------------------------

def _row_organism(row: pd.Series, default: str | None) -> str | None:
    """Return the per-row Species value if present and non-empty, else the default."""
    val = row.get("Species")
    if val is not None and isinstance(val, str) and val.strip():
        return val.strip()
    return default


def _build_enzyme_sbm(proteome_df: pd.DataFrame, brenda, organism: str | None) -> pd.DataFrame:
    """Create the initial SpeciesBaseMechanisms rows for enzymes from the proteome."""
    rows = []
    for _, row in proteome_df.iterrows():
        ec = row["EC"]
        name = get_enzyme_name(brenda, ec) or ec.replace(".", "_")
        effective_organism = _row_organism(row, organism)
        rows.append(
            {
                "Label": name,
                "EC": ec,
                "Type": "Enzyme",
                "StartingConc": row["Abundances"],
                "Conc": np.nan,
                "Mechanisms": np.nan,
                "Parameters": np.nan,
                "Species": effective_organism or np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Manual EC handling
# ---------------------------------------------------------------------------

def _add_manual_ecs(
    manual_ec_csv: str | Path,
    sbm_df: pd.DataFrame,
    reaction_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append manually specified EC entries that BRENDA/KEGG cannot resolve.

    Expected CSV columns: EC, Conc (optional), Accession Number (optional).
    These rows are added with NaN kinetics and must be filled in by hand.
    """
    manual = pd.read_csv(manual_ec_csv).dropna(subset=["EC"])

    sbm_rows, rxn_rows = [], []
    for _, row in manual.iterrows():
        label = row.get("Accession Number", row["EC"])
        sbm_rows.append(
            {
                "Label": label,
                "EC": row["EC"],
                "Type": "Enzyme",
                "StartingConc": row.get("Conc", np.nan),
                "Conc": np.nan,
                "Mechanisms": np.nan,
                "Parameters": np.nan,
                "Species": np.nan,
            }
        )
        rxn_rows.append(
            {
                "Label": label,
                "EC": row["EC"],
                "Enzyme": np.nan,
                "Reaction ID": np.nan,
                "Mechanism": np.nan,
                "Substrates": np.nan,
                "Products": np.nan,
                "Km": np.nan,
                "Kcat": np.nan,
                "Inhibitors": np.nan,
                "KI": np.nan,
            }
        )

    if sbm_rows:
        sbm_df = pd.concat([sbm_df, pd.DataFrame(sbm_rows)], ignore_index=True)
    if rxn_rows:
        reaction_df = pd.concat([reaction_df, pd.DataFrame(rxn_rows)], ignore_index=True)

    return sbm_df, reaction_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    proteome_csv: str | Path,
    brenda_file: str | Path,
    kegg_dir: str | Path,
    conc_ref_csv: str | Path,
    output_dir: str | Path = "frenda_output",
    organism: str | None = None,
    use_all_reactions: bool = True,
    catpred_url: str | None = "https://www.catpred.com",
    thermo_cache_path: str | Path | None = None,
    catpred_cache_path: str | Path | None = None,
    manual_ec_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full FRENDA pipeline.

    Args:
        proteome_csv:       Input CSV with EC and Abundances columns.
        brenda_file:        Path to BRENDA text database (brenda_download.txt).
        kegg_dir:           Directory with kegg_enzymes.json.gz and
                            kegg_reactions.json.gz (and optionally compound_names.csv).
        conc_ref_csv:       Metabolite_Concentrations.csv reference file.
        output_dir:         Directory for output CSV files.
        organism:           Organism name for BRENDA filtering (None = all organisms).
        use_all_reactions:  True = include every KEGG reaction for each EC;
                            False = first reaction only (smaller network).
        catpred_url:        CatPred web server URL; set to None to skip ML fallback.
        thermo_cache_path:  Pickle file to cache eQuilibrator Keq results.
        catpred_cache_path: Pickle file to cache UniProt / PubChem / CatPred results.
        manual_ec_csv:      Optional CSV of EC entries to add without BRENDA/KEGG lookup.

    Returns:
        (reaction_df, sbm_df) — the two output DataFrames, also written to disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load inputs ----------------------------------------------------------
    proteome_df = pd.read_csv(proteome_csv)
    logger.info("Loaded %d enzymes from %s", len(proteome_df), proteome_csv)

    ECs, RXNs = load_kegg_data(kegg_dir)
    compound_names = load_compound_names(kegg_dir)
    logger.info("KEGG: %d ECs, %d reactions, %d compound names", len(ECs), len(RXNs), len(compound_names))

    brenda = load_brenda(brenda_file)
    logger.info("BRENDA database loaded from %s", brenda_file)

    cc = ComponentContribution()

    # Persistent lookup caches to avoid redundant external API calls
    catpred_cache = load_cache(catpred_cache_path)
    sequence_cache: dict = catpred_cache.get("sequences", {})
    smiles_cache: dict = catpred_cache.get("smiles", {})
    compound_cache: dict = {}  # BRENDA name -> KEGG ID (eQuilibrator, rebuilt each run)

    thermo_cache = load_thermo_cache(thermo_cache_path)

    # Load the compound name->KEGG ID cache so lookups persist across runs.
    # This cache is stored inside the CatPred cache pickle for convenience.
    compound_cache_key = "_equilibrator_compound_cache"
    compound_cache: dict = catpred_cache.get(compound_cache_key, {})

    # -- Build initial enzyme SBM table ---------------------------------------
    sbm_df = _build_enzyme_sbm(proteome_df, brenda, organism)

    # -- Process each EC -------------------------------------------------------
    reaction_parts: list[pd.DataFrame] = []

    for idx, row in proteome_df.iterrows():
        ec = row["EC"]
        effective_organism = _row_organism(row, organism)
        logger.info("[%d/%d] Processing EC %s (organism: %s)", idx + 1, len(proteome_df), ec, effective_organism or "any")

        rxn_df = assemble_ec(
            ec=ec,
            brenda=brenda,
            ECs=ECs,
            RXNs=RXNs,
            cc=cc,
            compound_names=compound_names,
            compound_cache=compound_cache,
            sequence_cache=sequence_cache,
            smiles_cache=smiles_cache,
            use_all_reactions=use_all_reactions,
            organism=effective_organism,
            catpred_url=catpred_url,
        )

        if rxn_df is None:
            logger.warning("EC %s could not be assembled; skipping", ec)
            continue

        reaction_parts.append(rxn_df)

    if not reaction_parts:
        logger.error("No reactions assembled. Verify EC numbers and data file paths.")
        sys.exit(1)

    reaction_df = pd.concat(reaction_parts, ignore_index=True)

    # -- Add sequential reaction labels (R1, R2, ...) -------------------------
    reaction_df.insert(0, "Label", [f"R{i + 1}" for i in range(len(reaction_df))])

    # -- Handle manual ECs (appended without BRENDA/KEGG data) ----------------
    if manual_ec_csv:
        sbm_df, reaction_df = _add_manual_ecs(manual_ec_csv, sbm_df, reaction_df)

    # -- Thermodynamic equilibrium constants -----------------------------------
    keqs = []
    for rxn_id in reaction_df["Reaction ID"]:
        keq = calculate_keq(str(rxn_id), RXNs, cc, thermo_cache)
        keqs.append(keq["value"] if keq else np.nan)
    reaction_df["Keq"] = keqs
    save_thermo_cache(thermo_cache, thermo_cache_path)

    # -- Kcat forward/reverse split then full cleaning ------------------------
    reaction_df = calculate_kcat_f_r(reaction_df)
    reaction_df, sbm_df = clean_reactions(reaction_df, sbm_df)

    # -- Build complete metabolite list and assign concentrations --------------
    sbm_df = build_species_df(reaction_df, sbm_df)
    conc_ref_df = pd.read_csv(conc_ref_csv)
    sbm_df = assign_metabolite_concentrations(sbm_df, conc_ref_df)

    # -- Persist all caches ---------------------------------------------------
    catpred_cache["sequences"] = sequence_cache
    catpred_cache["smiles"] = smiles_cache
    catpred_cache[compound_cache_key] = compound_cache  # persist eQuilibrator lookups
    save_cache(catpred_cache, catpred_cache_path)

    # -- Write outputs ---------------------------------------------------------
    out_rxn = output_dir / "Reaction.csv"
    out_sbm = output_dir / "SpeciesBaseMechanisms.csv"
    reaction_df.to_csv(out_rxn, index=False)
    sbm_df.to_csv(out_sbm, index=False)

    logger.info("Wrote %d reactions to %s", len(reaction_df), out_rxn)
    logger.info("Wrote %d species to %s", len(sbm_df), out_sbm)
    return reaction_df, sbm_df


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m frenda.pipeline",
        description=(
            "FRENDA: convert proteomics data (EC numbers + abundances) "
            "into kinetic metabolic model parameter tables."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m frenda.pipeline proteome.csv
  python -m frenda.pipeline proteome.csv --organism "Escherichia coli"
  python -m frenda.pipeline proteome.csv --first-reaction-only --no-catpred
  python -m frenda.pipeline proteome.csv --output-dir results/ --thermo-cache thermo.pkl
        """,
    )
    parser.add_argument("proteome_csv", help="Input CSV with EC and Abundances columns")
    parser.add_argument(
        "--brenda",
        default=None,
        metavar="PATH",
        help="Path to BRENDA database file (brenda_download.txt). "
             "Default: data/brenda_download.txt",
    )
    parser.add_argument(
        "--kegg-dir",
        default=None,
        metavar="DIR",
        help="Directory containing KEGG JSON files. "
             "Default: data/",
    )
    parser.add_argument(
        "--conc-ref",
        default=None,
        metavar="PATH",
        help="Metabolite_Concentrations.csv reference file. "
             "Default: data/Metabolite_Concentrations.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="frenda_output",
        metavar="DIR",
        help="Directory for output files (default: frenda_output/)",
    )
    parser.add_argument(
        "--organism",
        default=None,
        metavar="NAME",
        help="Default organism for BRENDA/UniProt filtering, e.g. 'Escherichia coli'. "
             "Applied to any EC whose Species column is blank or absent. "
             "Omit entirely to use all organisms (broader coverage, more variation).",
    )
    parser.add_argument(
        "--first-reaction-only",
        action="store_true",
        help="Include only the first KEGG-annotated reaction per EC "
             "(reduces network size; useful for exploratory modeling).",
    )
    parser.add_argument(
        "--catpred-url",
        default="https://www.catpred.com",
        metavar="URL",
        help="CatPred web server base URL (default: https://www.catpred.com)",
    )
    parser.add_argument(
        "--no-catpred",
        action="store_true",
        help="Skip CatPred ML predictions; leave missing kinetics as NaN "
             "(will be imputed from dataset averages during cleaning).",
    )
    parser.add_argument(
        "--manual-ec",
        default=None,
        metavar="PATH",
        help="CSV of EC entries (columns: EC, Conc, Accession Number) to add "
             "without BRENDA/KEGG lookup (e.g., proprietary or novel enzymes).",
    )
    parser.add_argument(
        "--thermo-cache",
        default=None,
        metavar="PATH",
        help="Pickle file for persisting eQuilibrator Keq calculations across runs.",
    )
    parser.add_argument(
        "--catpred-cache",
        default=None,
        metavar="PATH",
        help="Pickle file for persisting UniProt/PubChem/CatPred results across runs.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve default data paths relative to this package's parent directory
    pkg_parent = Path(__file__).parent.parent
    brenda_file = args.brenda or (pkg_parent / "data/brenda_download.txt")
    kegg_dir = args.kegg_dir or (pkg_parent / "data")
    conc_ref = args.conc_ref or (pkg_parent / "data/Metabolite_Concentrations.csv")

    for path, label in [(brenda_file, "BRENDA"), (kegg_dir, "KEGG dir"), (conc_ref, "conc ref")]:
        if not Path(path).exists():
            logger.error("%s path not found: %s", label, path)
            sys.exit(1)

    run_pipeline(
        proteome_csv=args.proteome_csv,
        brenda_file=brenda_file,
        kegg_dir=kegg_dir,
        conc_ref_csv=conc_ref,
        output_dir=args.output_dir,
        organism=args.organism,
        use_all_reactions=not args.first_reaction_only,
        catpred_url=None if args.no_catpred else args.catpred_url,
        thermo_cache_path=args.thermo_cache,
        catpred_cache_path=args.catpred_cache,
        manual_ec_csv=args.manual_ec,
    )


if __name__ == "__main__":
    _cli()
