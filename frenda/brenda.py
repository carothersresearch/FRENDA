"""BRENDA database utilities for kinetic parameter retrieval."""

import logging
import math
from itertools import filterfalse

import numpy as np

logger = logging.getLogger(__name__)


def load_brenda(brenda_file: str):
    """Load the BRENDA database from the local text file.

    Download the file from https://www.brenda-enzymes.org/download.php
    (BRENDA text database, requires free registration).
    """
    from brendapyrser import BRENDA
    return BRENDA(str(brenda_file))


def get_enzyme_name(brenda, ec: str) -> str | None:
    """Return the canonical enzyme name from BRENDA, or None if not found."""
    try:
        return brenda.reactions.get_by_id(ec).name
    except (ValueError, AttributeError):
        return None


def get_brenda_parameters(brenda, ec: str, organism: str | None = None) -> list:
    """Extract averaged Km and kcat values from BRENDA for each compound.

    When multiple measurements exist, wild-type entries are preferred; otherwise
    all values are averaged.

    Args:
        brenda: BRENDA object from load_brenda()
        ec: EC number
        organism: organism name for filtering (None = use all organisms)

    Returns:
        List of [compound_name, Km (mM), kcat (s^-1)] for each compound that
        has at least one Km or kcat value. Missing values are NaN.
    """
    try:
        rxn = brenda.reactions.get_by_id(ec)
    except ValueError:
        logger.warning("EC %s not found in BRENDA", ec)
        return []

    kmvals = rxn.KMvalues.filter_by_organism(organism) if organism else rxn.KMvalues
    kcatvals = rxn.Kcatvalues.filter_by_organism(organism) if organism else rxn.Kcatvalues

    compounds = list(set(list(kmvals) + list(kcatvals)))

    results = []
    for compound in compounds:
        km = _extract_average(kmvals, compound, exclude=_get_kkm_values(rxn, compound, organism))
        kcat = _extract_average(kcatvals, compound)
        results.append([compound, km, kcat])

    return results


def get_brenda_ki_values(brenda, ec: str, organism: str | None = None) -> list:
    """Extract KI (inhibition constant) values from BRENDA.

    Args:
        brenda: BRENDA object from load_brenda()
        ec: EC number
        organism: organism name for filtering (None = use all organisms)

    Returns:
        List of [inhibitor_name, KI (mM)] pairs; empty list if no KI data.
    """
    try:
        rxn = brenda.reactions.get_by_id(ec)
        ki_attr = getattr(rxn, "KIvalues", None)
        if ki_attr is None:
            return []

        ki_data = ki_attr.filter_by_organism(organism) if organism else ki_attr
        results = []
        for compound in ki_data:
            vals = ki_data.filter_by_compound(compound).get_values()
            if vals:
                results.append([compound, float(np.mean(vals))])
        return results

    except (ValueError, AttributeError) as e:
        logger.debug("KI lookup failed for EC %s: %s", ec, e)
        return []


def map_compounds_to_kegg(
    compound_params: list, cc, compound_cache: dict
) -> tuple[dict, dict]:
    """Translate BRENDA compound names to KEGG IDs via eQuilibrator.

    Args:
        compound_params: list of [name, Km, kcat] from get_brenda_parameters()
        cc: equilibrator_api.ComponentContribution instance
        compound_cache: persistent dict caching name -> KEGG ID (modified in place)

    Returns:
        km_dict: maps 'Km_CXXXXX' -> float value (excludes zeros/NaN)
        kcat_dict: maps 'Kcat_CXXXXX' -> float value (excludes zeros/NaN)
    """
    km_dict, kcat_dict = {}, {}

    for name, km, kcat in compound_params:
        kegg_id = _resolve_kegg_id(name, cc, compound_cache)
        if kegg_id is None:
            continue

        if _is_valid(km):
            km_dict[f"Km_{kegg_id}"] = float(km)
        if _is_valid(kcat):
            kcat_dict[f"Kcat_{kegg_id}"] = float(kcat)

    return km_dict, kcat_dict


def map_ki_to_kegg(ki_params: list, cc, compound_cache: dict) -> tuple[list, list]:
    """Translate BRENDA KI compound names to KEGG IDs.

    Returns:
        (inhibitor_ids, ki_entries) where inhibitor_ids is a semicolon-separated
        string of KEGG IDs and ki_entries is formatted as "CXXXXX_KI: value".
    """
    inhibitor_ids, ki_entries = [], []

    for name, ki_val in ki_params:
        kegg_id = _resolve_kegg_id(name, cc, compound_cache)
        if kegg_id is None or not _is_valid(ki_val):
            continue
        inhibitor_ids.append(kegg_id)
        ki_entries.append(f"{kegg_id}_KI: {ki_val:.6g}")

    return "; ".join(inhibitor_ids), "; ".join(ki_entries)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_kkm_values(rxn, compound: str, organism: str | None) -> list:
    """Return KKM (substrate inhibition) values to exclude from KM."""
    try:
        kkm = rxn.KKMvalues.filter_by_organism(organism) if organism else rxn.KKMvalues
        return kkm.filter_by_compound(compound).get_values()
    except AttributeError:
        return []


def _extract_average(vals_obj, compound: str, exclude: list | None = None) -> float:
    """Average kinetic values for one compound, preferring wild-type entries."""
    try:
        raw = vals_obj.filter_by_compound(compound).get_values()
    except AttributeError:
        return np.nan

    if exclude:
        raw = list(filterfalse(exclude.__contains__, raw))

    if not raw:
        return np.nan

    # When multiple measurements exist, prefer wild-type variants
    if len(raw) > 1:
        wt = []
        try:
            kval = vals_obj.filter_by_compound(compound)
            for k in kval.keys():
                for entry in kval[k]:
                    if "wild" in entry.get("meta", ""):
                        wt.append(entry["value"])
        except (AttributeError, TypeError):
            pass
        if wt:
            raw = wt

    return float(np.mean(raw))


def _resolve_kegg_id(name: str, cc, cache: dict) -> str | None:
    """Look up the KEGG compound ID for a compound name via eQuilibrator."""
    if name in cache:
        return cache[name]

    try:
        for identifier in cc.search_compound(name).identifiers:
            if identifier.registry.namespace == "kegg":
                kegg_id = identifier.accession
                cache[name] = kegg_id
                return kegg_id
    except (ValueError, AttributeError, Exception) as e:
        logger.debug("No KEGG ID for '%s': %s", name, e)

    cache[name] = None
    return None


def _is_valid(val) -> bool:
    """Return True if val is a non-NaN, non-zero float."""
    try:
        return not (math.isnan(float(val)) or math.isclose(float(val), 0.0))
    except (TypeError, ValueError):
        return False
