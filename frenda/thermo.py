"""Thermodynamic equilibrium constant calculation using eQuilibrator."""

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def calculate_keq(
    reaction_id: str,
    RXNs: dict,
    cc,
    cache: dict,
) -> dict | None:
    """Calculate the equilibrium constant (Keq) for a KEGG reaction.

    Uses eQuilibrator's Component Contribution method (Noor et al. 2013) to
    estimate ΔG'° and converts to Keq = exp(-ΔG'°/RT) at pH 7, ionic
    strength 0.25 M (eQuilibrator defaults).

    Args:
        reaction_id: KEGG reaction ID (e.g., "R00001")
        RXNs: reaction_id -> compound list dict from kegg.load_kegg_data()
        cc: equilibrator_api.ComponentContribution instance
        cache: persistent dict for caching results (modified in place)

    Returns:
        dict with 'value' (Keq point estimate) and 'error' (uncertainty),
        or None if the calculation cannot be completed.
    """
    from equilibrator_api import Reaction

    if reaction_id in cache:
        return cache[reaction_id]

    if reaction_id not in RXNs:
        return None

    try:
        # Build the Reaction object: map KEGG IDs to stoichiometries
        rxn = Reaction(
            {
                cc.get_compound(f"kegg:{c[1]}"): c[0]
                for c in RXNs[reaction_id]
            }
        )

        # Attempt to balance via oxidation if needed
        if not rxn.is_balanced():
            try:
                rxn = cc.balance_by_oxidation(rxn)
            except Exception as e:
                logger.debug("Could not balance %s: %s", reaction_id, e)

        dg = cc.standard_dg_prime(rxn)
        # Clip exponent to avoid overflow; reactions with |ΔG'°/RT| > 500 are
        # treated as effectively irreversible (Keq → ∞ or ≈ 0).
        exp_val = float(np.clip((-dg / cc.RT).value, -500, 500))
        exp_err = float(np.clip((-dg / cc.RT).error, -500, 500))
        keq = {
            "value": float(np.exp(exp_val)),
            "error": float(np.exp(exp_err)),
        }
        cache[reaction_id] = keq
        return keq

    except Exception as e:
        logger.warning("Keq calculation failed for %s: %s", reaction_id, e)
        return None


def load_thermo_cache(cache_path: str | Path | None) -> dict:
    """Load persisted thermodynamic cache; returns empty dict on failure."""
    if cache_path is None:
        return {}
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return {}


def save_thermo_cache(cache: dict, cache_path: str | Path | None) -> None:
    """Write thermodynamic cache to disk if a path is provided."""
    if cache_path is None:
        return
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
