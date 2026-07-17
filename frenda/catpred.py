"""CatPred kinetic parameter prediction with UniProt and PubChem data fetching.

CatPred (https://catpred.bio) predicts kcat (s^-1) and Km (mM) from protein
sequences and metabolite SMILES strings.

Reference: Singh et al. (2023). CatPred: A comprehensive framework for deep
learning in vitro enzyme kinetic parameters. Nature Computational Science.
GitHub: https://github.com/maranasgroup/CatPred
"""

import json
import logging
import pickle
import re
import time
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

# Public API endpoints
_UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
_PUBCHEM_BY_NAME = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/IsomericSMILES/JSON"
_PUBCHEM_BY_XREF = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RegistryID/{}/property/IsomericSMILES/JSON"
_KEGG_COMPOUND = "https://rest.kegg.jp/get/{}"
_PUBCHEM_SID_TO_CID = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sid/{}/cids/JSON"
_PUBCHEM_BY_CID = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/IsomericSMILES/JSON"
_CATPRED_PREDICT = "{base}/predict"  # POST endpoint; base URL is https://www.catpred.com

# Small ions/cofactors with no real enzyme-substrate structure to learn from.
# CatPred's backend reliably 500s on these (e.g. a bare "[H+]" SMILES crashes
# its featurization step), so we skip the network round-trip entirely.
_TRIVIAL_COMPOUNDS = {
    "C00080",  # H+
    "C00001",  # H2O
    "C00007",  # O2
    "C00011",  # CO2
    "C00014",  # NH3
    "C00009",  # orthophosphate (Pi)
    "C00013",  # diphosphate (PPi)
    "C00238",  # K+
    "C00076",  # Ca2+
    "C00070",  # Cu2+
    "C01330",  # Na+
    "C00291",  # Cl-
    "C00305",  # Mg2+
}


# ---------------------------------------------------------------------------
# UniProt sequence fetching
# ---------------------------------------------------------------------------

def fetch_uniprot_sequence(ec: str, organism: str | None = None) -> str | None:
    """Fetch the first reviewed (Swiss-Prot) protein sequence for an EC number.

    Falls back to unreviewed entries if no reviewed entry is found.

    Args:
        ec: EC number (e.g., "1.1.1.1")
        organism: organism name to narrow results (None = any organism)

    Returns:
        Amino acid sequence string, or None if retrieval fails.
    """
    for reviewed in (True, False):
        query = f"ec:{ec}"
        if organism:
            query += f' AND organism_name:"{organism}"'
        if reviewed:
            query += " AND reviewed:true"

        params = {"query": query, "format": "json", "fields": "sequence", "size": 1}
        try:
            resp = requests.get(_UNIPROT_SEARCH, params=params, timeout=20)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                return results[0]["sequence"]["value"]
        except (requests.RequestException, KeyError, IndexError) as e:
            logger.debug("UniProt lookup failed for EC %s (reviewed=%s): %s", ec, reviewed, e)

    logger.warning("No UniProt sequence found for EC %s", ec)
    return None


# ---------------------------------------------------------------------------
# PubChem SMILES fetching
# ---------------------------------------------------------------------------

def fetch_smiles_for_kegg_id(
    kegg_id: str, compound_names: dict | None = None
) -> str | None:
    """Retrieve an IsomericSMILES string for a KEGG compound ID.

    Tries in order:
      1. PubChem cross-reference by KEGG registry ID
      2. PubChem name lookup using the preferred name from compound_names
      3. KEGG REST API to get the compound name, then PubChem name lookup
      4. KEGG REST API's own PubChem SID cross-reference (DBLINKS), resolved
         to a CID and looked up directly — catches compounds PubChem doesn't
         index under a matching name or "RegistryID" xref (steps 1-3 fail
         for generic/templated KEGG compounds like CoA- or ACP-linked
         intermediates, which have no concrete PubChem CID at all; in that
         case this also returns None).

    Args:
        kegg_id: KEGG compound ID (e.g., "C00001")
        compound_names: optional dict mapping KEGG ID -> preferred name
            (loaded by kegg.load_compound_names)

    Returns:
        SMILES string, or None if all attempts fail.
    """
    # 1. PubChem cross-reference with KEGG ID
    smiles = _pubchem_by_xref(kegg_id)
    if smiles:
        return smiles

    # 2. Use pre-loaded compound name table
    if compound_names and kegg_id in compound_names:
        smiles = _pubchem_by_name(compound_names[kegg_id])
        if smiles:
            return smiles

    # 3 & 4. Fetch the KEGG entry once; try its NAME, then its PubChem SID xref
    entry_text = _kegg_entry_text(kegg_id)
    if entry_text:
        name = _parse_kegg_name(entry_text)
        if name:
            smiles = _pubchem_by_name(name)
            if smiles:
                return smiles

        sid = _parse_kegg_pubchem_sid(entry_text)
        if sid:
            smiles = _pubchem_by_sid(sid)
            if smiles:
                return smiles

    logger.warning("Could not retrieve SMILES for KEGG compound %s", kegg_id)
    return None


def _pubchem_by_xref(kegg_id: str) -> str | None:
    try:
        url = _PUBCHEM_BY_XREF.format(kegg_id)
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            props = resp.json().get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("IsomericSMILES") or props[0].get("SMILES")
    except (requests.RequestException, ValueError, KeyError):
        pass
    return None


def _pubchem_by_name(name: str) -> str | None:
    try:
        url = _PUBCHEM_BY_NAME.format(requests.utils.quote(name))
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            props = resp.json().get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("IsomericSMILES") or props[0].get("SMILES")
    except (requests.RequestException, ValueError, KeyError):
        pass
    return None


def _kegg_entry_text(kegg_id: str) -> str | None:
    """Fetch the raw KEGG flat-file entry text for a compound."""
    try:
        resp = requests.get(_KEGG_COMPOUND.format(kegg_id), timeout=10)
        if resp.status_code == 200:
            return resp.text
    except requests.RequestException:
        pass
    return None


def _parse_kegg_name(entry_text: str) -> str | None:
    """Extract the primary NAME field from a KEGG flat-file entry."""
    for line in entry_text.splitlines():
        if line.startswith("NAME"):
            return line.split(None, 1)[1].strip().rstrip(";")
    return None


def _parse_kegg_pubchem_sid(entry_text: str) -> str | None:
    """Extract the PubChem Substance ID from a KEGG entry's DBLINKS section."""
    match = re.search(r"PubChem:\s*(\d+)", entry_text)
    return match.group(1) if match else None


def _pubchem_by_sid(sid: str) -> str | None:
    """Resolve a PubChem SID to a CID, then fetch its IsomericSMILES."""
    try:
        resp = requests.get(_PUBCHEM_SID_TO_CID.format(sid), timeout=15)
        if resp.status_code != 200:
            return None
        cids = resp.json().get("InformationList", {}).get("Information", [{}])[0].get("CID")
        if not cids:
            return None
        cid = cids[0]

        resp = requests.get(_PUBCHEM_BY_CID.format(cid), timeout=15)
        if resp.status_code == 200:
            props = resp.json().get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("IsomericSMILES") or props[0].get("SMILES")
    except (requests.RequestException, ValueError, KeyError, IndexError):
        pass
    return None


# ---------------------------------------------------------------------------
# CatPred API
# ---------------------------------------------------------------------------

_CATPRED_UNITS = {"kcat": "s^(-1)", "km": "mM", "ki": "mM"}


def predict_catpred(
    sequence: str,
    smiles: str,
    parameter: str,
    catpred_url: str,
) -> float:
    """Call the CatPred web server to predict a single kinetic parameter.

    Args:
        sequence: amino acid sequence of the enzyme
        smiles: SMILES string of the substrate
        parameter: "kcat" (s^-1) or "km" (mM)
        catpred_url: base URL of the CatPred server (e.g., "https://www.catpred.com")

    Returns:
        Predicted value as a float in linear scale, or NaN on failure.
    """
    endpoint = _CATPRED_PREDICT.format(base=catpred_url.rstrip("/"))
    payload = {
        "parameter": parameter,
        "checkpoint_dir": parameter,
        "input_rows": [
            {
                "SMILES": smiles,
                "sequence": sequence,
                "pdbpath": f"frenda_{abs(hash(sequence)) % 10**7}",
            }
        ],
        "results_dir": "frenda_results",
    }

    try:
        resp = requests.post(endpoint, json=payload, timeout=120)
        if not resp.ok:
            logger.warning("CatPred prediction failed (%s): HTTP %s — %s", parameter, resp.status_code, resp.text[:300])
            return np.nan
        rows = resp.json().get("preview_rows", [])
        if not rows:
            return np.nan
        row = rows[0]
        # Primary key is the linear-scale field; fall back to log10 if absent
        unit = _CATPRED_UNITS.get(parameter, "mM")
        linear_key = f"Prediction_({unit})"
        if linear_key in row and row[linear_key] is not None:
            return float(row[linear_key])
        if "Prediction_log10" in row and row["Prediction_log10"] is not None:
            return float(10 ** row["Prediction_log10"])
        return np.nan
    except (requests.RequestException, ValueError, KeyError, IndexError) as e:
        logger.warning("CatPred prediction failed (%s): %s", parameter, e)
        return np.nan


# ---------------------------------------------------------------------------
# High-level convenience function used by the pipeline
# ---------------------------------------------------------------------------

def get_kinetics_catpred(
    ec: str,
    kegg_id: str,
    catpred_url: str,
    organism: str | None = None,
    compound_names: dict | None = None,
    sequence_cache: dict | None = None,
    smiles_cache: dict | None = None,
    prediction_cache: dict | None = None,
) -> tuple[float, float]:
    """Predict kcat and Km for one (enzyme, substrate) pair using CatPred.

    Fetches the UniProt sequence and PubChem SMILES if not already cached,
    then calls CatPred for both kcat and Km.

    Args:
        ec: EC number
        kegg_id: KEGG compound ID of the substrate (e.g., "C00001")
        catpred_url: CatPred web server URL
        organism: organism for UniProt filtering (None = any)
        compound_names: KEGG ID -> name mapping to aid SMILES lookup
        sequence_cache: persistent dict caching EC -> sequence (modified in place)
        smiles_cache: persistent dict caching KEGG ID -> SMILES (modified in place)
        prediction_cache: persistent dict caching (ec, organism, kegg_id) ->
            (kcat, km), so repeat pipeline runs skip the slow CatPred call
            entirely for pairs already predicted (modified in place)

    Returns:
        (kcat, km) tuple; either value may be NaN if prediction fails.
    """
    if kegg_id in _TRIVIAL_COMPOUNDS:
        logger.debug("Skipping CatPred for %s: trivial ion/cofactor, not a real substrate", kegg_id)
        return np.nan, np.nan

    if sequence_cache is None:
        sequence_cache = {}
    if smiles_cache is None:
        smiles_cache = {}
    if prediction_cache is None:
        prediction_cache = {}

    pred_key = f"{ec}|{organism or ''}|{kegg_id}"
    if pred_key in prediction_cache:
        return prediction_cache[pred_key]

    # Resolve protein sequence (cached per EC+organism key)
    seq_key = f"{ec}|{organism or ''}"
    if seq_key not in sequence_cache:
        sequence_cache[seq_key] = fetch_uniprot_sequence(ec, organism)
    sequence = sequence_cache[seq_key]

    # Resolve substrate SMILES (cached per KEGG ID)
    if kegg_id not in smiles_cache:
        smiles_cache[kegg_id] = fetch_smiles_for_kegg_id(kegg_id, compound_names)
    smiles = smiles_cache[kegg_id]

    if not sequence or not smiles:
        logger.debug(
            "Skipping CatPred for EC %s / %s: missing %s",
            ec,
            kegg_id,
            "sequence" if not sequence else "SMILES",
        )
        return np.nan, np.nan

    time.sleep(0.3)  # polite pause between API calls
    kcat = predict_catpred(sequence, smiles, "kcat", catpred_url)
    km = predict_catpred(sequence, smiles, "km", catpred_url)

    prediction_cache[pred_key] = (kcat, km)
    return kcat, km


# ---------------------------------------------------------------------------
# Persistent cache helpers (pickle-backed, shared across pipeline runs)
# ---------------------------------------------------------------------------

def load_cache(cache_path: str | Path | None) -> dict:
    """Load a pickle cache, returning an empty dict if absent or corrupt."""
    if cache_path is None:
        return {}
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return {}


def save_cache(cache: dict, cache_path: str | Path | None) -> None:
    """Persist a dict to a pickle file if a path is given."""
    if cache_path is None:
        return
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
