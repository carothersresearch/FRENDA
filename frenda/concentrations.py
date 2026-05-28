"""Metabolite concentration assignment for the SpeciesBaseMechanisms table."""

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_KEGG_ID_RE = re.compile(r"C\d+")

# Applied when a metabolite has no entry in the concentration reference table.
DEFAULT_CONCENTRATION_MM = 0.001


def extract_unique_compounds(reaction_df: pd.DataFrame) -> set[str]:
    """Collect every unique KEGG compound ID from Substrates, Products, and Inhibitors."""
    ids: set[str] = set()
    for col in ("Substrates", "Products", "Inhibitors"):
        if col not in reaction_df.columns:
            continue
        for cell in reaction_df[col].dropna():
            ids.update(_KEGG_ID_RE.findall(str(cell)))
    return ids


def build_species_df(reaction_df: pd.DataFrame, sbm_df: pd.DataFrame) -> pd.DataFrame:
    """Append a Metabolite row for each KEGG compound not already in sbm_df.

    Existing rows (enzymes and pre-existing metabolites) are left unchanged.
    """
    compound_ids = extract_unique_compounds(reaction_df)
    existing_labels = set(sbm_df["Label"].astype(str))

    new_rows = [
        {"Label": cid, "Type": "Metabolite", "StartingConc": np.nan}
        for cid in compound_ids
        if cid not in existing_labels
    ]

    if not new_rows:
        return sbm_df

    return pd.concat([sbm_df, pd.DataFrame(new_rows)], ignore_index=True)


def assign_metabolite_concentrations(
    sbm_df: pd.DataFrame, conc_ref_df: pd.DataFrame
) -> pd.DataFrame:
    """Populate StartingConc for metabolite rows using a concentration reference table.

    Metabolites found in the reference receive the measured concentration;
    all others receive DEFAULT_CONCENTRATION_MM (0.001 mM).

    Args:
        sbm_df: SpeciesBaseMechanisms DataFrame with Label and Type columns
        conc_ref_df: reference with KEGG and Concentration columns
            (see frenda_brenda/Files/Metabolite_Concentrations.csv)

    Returns:
        Updated sbm_df with StartingConc filled for all Metabolite rows.
    """
    kegg_to_conc: dict = dict(
        zip(conc_ref_df["KEGG"].astype(str), conc_ref_df["Concentration"])
    )

    metabolite_mask = sbm_df["Type"] == "Metabolite"
    sbm_df.loc[metabolite_mask, "StartingConc"] = sbm_df.loc[
        metabolite_mask, "Label"
    ].map(lambda label: kegg_to_conc.get(str(label), DEFAULT_CONCENTRATION_MM))

    missing = (
        metabolite_mask
        & sbm_df["StartingConc"].isna()
    ).sum()
    if missing:
        logger.debug("%d metabolites still missing concentration; using default", missing)
        sbm_df["StartingConc"] = sbm_df["StartingConc"].fillna(DEFAULT_CONCENTRATION_MM)

    return sbm_df
