"""Data cleaning and imputation for assembled reaction tables."""

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_KEGG_ID_RE = re.compile(r"C\d+")


# ---------------------------------------------------------------------------
# Kcat splitting: per-compound -> Kcat_F / Kcat_R
# ---------------------------------------------------------------------------

def calculate_kcat_f_r(df: pd.DataFrame) -> pd.DataFrame:
    """Replace per-compound Kcat entries with forward (Kcat_F) and reverse (Kcat_R) averages.

    Kcat_F is the average kcat for compounds that appear as substrates;
    Kcat_R is the average for compounds that appear as products.
    Compounds present in both roles are excluded from both averages.
    """
    kcat_f, kcat_r = [], []

    for _, row in df.iterrows():
        substrates = set(_kegg_ids_from(row.get("Substrates", "")))
        products = set(_kegg_ids_from(row.get("Products", "")))

        f_vals, r_vals = [], []
        for entry in _split_entries(row.get("Kcat", "")):
            kegg_id, val = _parse_entry(entry, prefix="Kcat_")
            if kegg_id is None or np.isnan(val):
                continue
            if kegg_id in substrates and kegg_id not in products:
                f_vals.append(val)
            elif kegg_id in products and kegg_id not in substrates:
                r_vals.append(val)

        kcat_f.append(np.mean(f_vals) if f_vals else np.nan)
        kcat_r.append(np.mean(r_vals) if r_vals else np.nan)

    df["Kcat"] = [f"Kcat_F: {f}; Kcat_R: {r}" for f, r in zip(kcat_f, kcat_r)]
    return df


# ---------------------------------------------------------------------------
# NaN imputation
# ---------------------------------------------------------------------------

def fill_nan_kcat(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing Kcat_F / Kcat_R with the dataset-wide averages.

    Flags which side(s) were imputed in a separate 'Kcat_Imputed' column
    (e.g. "F", "R", "F; R") so downstream consumers of the numeric Kcat
    string (e.g. odbm's Antimony writer) are unaffected, while the CSV
    still records which values are real measurements/predictions vs.
    dataset-average fallbacks.
    """
    df[["_kf", "_kr"]] = df["Kcat"].str.extract(r"Kcat_F: (.*?); Kcat_R: (.*?)$")
    df["_kf"] = pd.to_numeric(df["_kf"], errors="coerce")
    df["_kr"] = pd.to_numeric(df["_kr"], errors="coerce")

    avg_f = df["_kf"].mean()
    avg_r = df["_kr"].mean()

    was_nan_f = df["_kf"].isna()
    was_nan_r = df["_kr"].isna()
    df["_kf"] = df["_kf"].fillna(avg_f)
    df["_kr"] = df["_kr"].fillna(avg_r)

    # Use apply() to avoid pandas 3.x NaN propagation in string concatenation.
    def _fmt(x) -> str:
        return f"{x:.6g}" if pd.notna(x) else "nan"

    df["Kcat"] = df.apply(
        lambda row: f"Kcat_F: {_fmt(row['_kf'])}; Kcat_R: {_fmt(row['_kr'])}", axis=1
    )
    df["Kcat_Imputed"] = [
        "; ".join(flag for flag, was_nan in (("F", f), ("R", r)) if was_nan) or np.nan
        for f, r in zip(was_nan_f, was_nan_r)
    ]
    df.drop(columns=["_kf", "_kr"], inplace=True)
    return df


def fix_nan_km(df: pd.DataFrame) -> pd.DataFrame:
    """Impute NaN Km values with per-reaction average, then global average fallback.

    Records which compound IDs were imputed in a new 'Km_Imputed' column
    (semicolon-separated KEGG IDs) rather than tagging the Km string itself,
    so the Km column stays a clean 'Km_CXXXXX: <number>' list that downstream
    numeric parsers (e.g. odbm) can consume unchanged.
    """
    all_vals = _collect_float_values(df["Km"])
    global_avg = np.mean(all_vals) if all_vals else 0.1  # 0.1 mM default

    imputed_flags, km_values = [], []
    for s in df["Km"]:
        fixed, imputed_ids = _impute_param_string(s, global_avg)
        km_values.append(fixed)
        imputed_flags.append("; ".join(imputed_ids) if imputed_ids else np.nan)

    df["Km"] = km_values
    df["Km_Imputed"] = imputed_flags
    return df


def fix_nan_ki(df: pd.DataFrame) -> pd.DataFrame:
    """Impute NaN KI values with per-row average, then global average fallback.

    Rows with no KI data at all (NaN) are left unchanged. Imputed compound
    IDs are recorded in a new 'KI_Imputed' column (see fix_nan_km).
    """
    all_vals = _collect_float_values(df["KI"], sep=";")
    global_avg = np.mean(all_vals) if all_vals else 1.0  # 1.0 mM default

    imputed_flags, ki_values = [], []
    for s in df["KI"]:
        fixed, imputed_ids = _impute_param_string(s, global_avg, sep=";")
        ki_values.append(fixed)
        imputed_flags.append("; ".join(imputed_ids) if imputed_ids else np.nan)

    df["KI"] = ki_values
    df["KI_Imputed"] = imputed_flags
    return df


# ---------------------------------------------------------------------------
# Inhibitor filtering
# ---------------------------------------------------------------------------

def remove_non_compounds(df: pd.DataFrame) -> pd.DataFrame:
    """Drop inhibitor entries whose KEGG-style prefix starts with 'D' or 'G'.

    In KEGG nomenclature, 'D' prefixes are drugs and 'G' prefixes are glycans;
    only 'C' (compound/metabolite) entries are relevant for kinetic models.
    """
    df["Inhibitors"] = df["Inhibitors"].apply(_filter_inhibitor_ids)
    df["KI"] = df["KI"].apply(_filter_ki_ids)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate entries within Inhibitors and KI fields."""
    df["Inhibitors"] = df["Inhibitors"].apply(_dedup_field)
    df["KI"] = df["KI"].apply(_dedup_field)
    return df


# ---------------------------------------------------------------------------
# Duplicate reaction / enzyme handling
# ---------------------------------------------------------------------------

def drop_duplicate_reactions(
    reaction_df: pd.DataFrame, sbm_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove duplicate rows and consolidate concentrations.

    Reactions duplicated on (EC, Reaction ID) are dropped (keeps first).
    Enzyme starting concentrations for the same EC are summed (reflecting that
    multiple isoforms contribute additively to total enzyme activity).
    Metabolite concentrations for the same KEGG ID are summed.
    """
    # Sum enzyme concentrations across duplicate EC entries
    enz_mask = sbm_df["Type"] == "Enzyme"
    enzymes = sbm_df[enz_mask].copy()
    enzymes["StartingConc"] = enzymes.groupby("EC")["StartingConc"].transform("sum")
    enzymes = enzymes.drop_duplicates(subset=["EC"])

    # Sum metabolite concentrations across duplicate KEGG IDs
    metabolites = sbm_df[~enz_mask].copy()
    metabolites["StartingConc"] = metabolites.groupby("Label")["StartingConc"].transform("sum")
    metabolites = metabolites.drop_duplicates(subset=["Label"])

    sbm_df = pd.concat([enzymes, metabolites], ignore_index=True)
    reaction_df = reaction_df.drop_duplicates(subset=["EC", "Reaction ID"])
    return reaction_df, sbm_df


# ---------------------------------------------------------------------------
# Master cleaning pipeline
# ---------------------------------------------------------------------------

def clean_reactions(
    reaction_df: pd.DataFrame, sbm_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning pipeline on assembled reaction and species data."""
    reaction_df = fill_nan_kcat(reaction_df)
    reaction_df = fix_nan_km(reaction_df)
    reaction_df = remove_non_compounds(reaction_df)
    reaction_df = remove_duplicates(reaction_df)
    reaction_df = fix_nan_ki(reaction_df)
    reaction_df, sbm_df = drop_duplicate_reactions(reaction_df, sbm_df)
    return reaction_df, sbm_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _kegg_ids_from(cell: str) -> list[str]:
    if not isinstance(cell, str):
        return []
    return _KEGG_ID_RE.findall(cell)


def _split_entries(cell: str, sep: str = "; ") -> list[str]:
    if not isinstance(cell, str):
        return []
    return [e.strip() for e in cell.split(sep) if e.strip()]


def _parse_entry(entry: str, prefix: str) -> tuple[str | None, float]:
    """Split 'Prefix_ID: value' into (ID, float_value)."""
    if ": " not in entry:
        return None, np.nan
    key, val_str = entry.split(": ", 1)
    kegg_id = key.replace(prefix, "")
    try:
        return kegg_id, float(val_str)
    except ValueError:
        return kegg_id, np.nan


def _collect_float_values(series: pd.Series, sep: str = "; ") -> list[float]:
    """Gather all numeric values from a series of 'key: value' strings."""
    vals = []
    for cell in series:
        if not isinstance(cell, str):
            continue
        for entry in cell.split(sep):
            if ": " in entry:
                _, val_str = entry.split(": ", 1)
                try:
                    v = float(val_str)
                    if not np.isnan(v):
                        vals.append(v)
                except ValueError:
                    pass
    return vals


def _impute_param_string(
    cell: str, global_avg: float, sep: str = "; "
) -> tuple[str, list[str]]:
    """Replace 'nan' values in a 'key: value; key: value' string.

    Uses the row-level average of valid values; falls back to global_avg.

    Returns:
        (fixed_string, imputed_keys) — imputed_keys lists the raw keys
        (e.g. "Km_C00080", "C00080_KI") whose value was filled in.
    """
    if not isinstance(cell, str):
        return cell, []

    entries = [e.strip() for e in cell.split(sep) if e.strip()]
    row_vals = []
    for e in entries:
        if ": " in e:
            _, val = e.split(": ", 1)
            try:
                v = float(val)
                if not np.isnan(v):
                    row_vals.append(v)
            except ValueError:
                pass

    row_avg = np.mean(row_vals) if row_vals else global_avg

    fixed, imputed_keys = [], []
    for e in entries:
        if ": " not in e:
            continue
        key, val = e.split(": ", 1)
        try:
            v = float(val)
            if np.isnan(v):
                fixed.append(f"{key}: {row_avg:.6g}")
                imputed_keys.append(key)
            else:
                fixed.append(f"{key}: {v:.6g}")
        except ValueError:
            fixed.append(e)

    return sep.join(fixed), imputed_keys


def _filter_inhibitor_ids(cell) -> object:
    """Keep only semicolon-separated entries that don't start with D or G."""
    if not isinstance(cell, str) or pd.isna(cell):
        return np.nan
    kept = [
        p.strip()
        for p in cell.split(";")
        if p.strip() and not p.strip().startswith(("D", "G"))
    ]
    return "; ".join(kept) if kept else np.nan


def _filter_ki_ids(cell) -> object:
    """Keep KI entries whose compound prefix (before '_KI') doesn't start with D or G."""
    if not isinstance(cell, str) or pd.isna(cell):
        return np.nan
    kept = []
    for entry in cell.split(";"):
        compound_part = entry.split("_KI")[0].strip()
        if compound_part and not compound_part.startswith(("D", "G")):
            kept.append(entry.strip())
    return "; ".join(kept) if kept else np.nan


def _dedup_field(cell) -> object:
    if not isinstance(cell, str) or pd.isna(cell):
        return cell
    seen, out = set(), []
    for item in cell.split(";"):
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return "; ".join(out) if out else np.nan
