"""KEGG database utilities for enzyme and reaction data."""

import gzip
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_kegg_data(kegg_dir: str | Path):
    """Load KEGG enzyme and reaction databases from gzipped JSON files.

    Args:
        kegg_dir: directory containing kegg_enzymes.json.gz and kegg_reactions.json.gz

    Returns:
        ECs: dict mapping EC number -> list of KEGG reaction IDs
        RXNs: dict mapping KEGG reaction ID -> list of (stoichiometry, compound_ID) tuples
              Negative stoichiometry = substrate; positive = product.
    """
    kegg_dir = Path(kegg_dir)

    with gzip.open(kegg_dir / "kegg_enzymes.json.gz", "r") as f:
        ECs = {e["EC"]: e["reaction_ids"] for e in json.load(f)}

    with gzip.open(kegg_dir / "kegg_reactions.json.gz", "r") as f:
        RXNs = {r["RID"]: r["reaction"] for r in json.load(f)}

    return ECs, RXNs


def load_compound_names(kegg_dir: str | Path) -> dict:
    """Load KEGG compound ID -> preferred name mapping from compound_names.csv.

    Used to look up human-readable names when querying PubChem for SMILES.
    Falls back to an empty dict if the file is absent.
    """
    path = Path(kegg_dir) / "compound_names.csv"
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    # compound_id is like "kegg:C00001"; strip the prefix
    return {
        row["compound_id"].replace("kegg:", ""): row["preferred_name"]
        for _, row in df.iterrows()
    }


def get_substrates_and_products(
    ec: str,
    ECs: dict,
    RXNs: dict,
    use_all_reactions: bool = True,
) -> pd.DataFrame:
    """Look up KEGG reactions for an EC number and extract substrates/products.

    Args:
        ec: EC number (e.g., "1.1.1.1")
        ECs: EC -> reaction_ids dict from load_kegg_data()
        RXNs: reaction_id -> compounds dict from load_kegg_data()
        use_all_reactions: if False, return only the first annotated reaction,
            reducing network size

    Returns:
        DataFrame with columns Reaction ID, Substrates, Products.
        Each compound list is formatted as "stoich KEGG_ID; stoich KEGG_ID".

    Raises:
        KeyError: EC not in KEGG
        ValueError: EC in KEGG but all annotated reactions are missing from RXNs
    """
    if ec not in ECs:
        raise KeyError(f"EC {ec} not found in KEGG enzyme database")

    rxn_ids = ECs[ec] if use_all_reactions else ECs[ec][:1]

    rows = []
    for rxn_id in rxn_ids:
        if rxn_id not in RXNs:
            logger.debug("Reaction %s not in KEGG reaction database, skipping", rxn_id)
            continue

        compounds = RXNs[rxn_id]
        substrates = [f"{abs(c[0])} {c[1]}" for c in compounds if c[0] <= 0]
        products = [f"{abs(c[0])} {c[1]}" for c in compounds if c[0] >= 0]

        rows.append(
            {
                "Reaction ID": rxn_id,
                "Substrates": "; ".join(substrates),
                "Products": "; ".join(products),
            }
        )

    if not rows:
        raise ValueError(f"No usable KEGG reactions found for EC {ec}")

    return pd.DataFrame(rows)
