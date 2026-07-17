# FRENDA — Fast Retrieval of ENzyme DAta

FRENDA is a Python tool that automates the construction and parameterization of mechanistic kinetic models from proteomics data. Given a list of enzymes (as EC numbers) with measured or relative abundances, FRENDA retrieves the corresponding reactions from KEGG, populates kinetic parameters from BRENDA and CatPred, calculates thermodynamic equilibrium constants with eQuilibrator, and assigns metabolite concentrations — producing a fully parameterized reaction table that can be converted directly into a simulatable ODE model in Antimony/SBML format.

A companion tool, **Antotate**, annotates the resulting model with standardized cross-references from KEGG, ChEBI, HMDB, BiGG, and MetaCyc.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Downloading BRENDA](#downloading-brenda)
5. [Quick Start](#quick-start)
6. [Input Format](#input-format)
7. [Running the Pipeline](#running-the-pipeline)
8. [Output Files](#output-files)
9. [Converting to Antimony/SBML (ODBM)](#converting-to-antimonysml-odbm)
10. [Annotating the Model (Antotate)](#annotating-the-model-antotate)
11. [Full End-to-End Example](#full-end-to-end-example)
12. [Python API](#python-api)
13. [Caching](#caching)
14. [Troubleshooting](#troubleshooting)
15. [References](#references)

---

## Overview

### Workflow

```
Proteomics CSV
(EC numbers + abundances)
        │
        ▼
┌──────────────────┐
│   KEGG lookup    │  Retrieves reactions, substrates, and products
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  BRENDA lookup   │  Fetches measured Km, kcat, KI values
└────────┬─────────┘
         │ (missing values)
         ▼
┌──────────────────┐
│  CatPred (ML)    │  Predicts kcat and Km from sequence + SMILES
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  eQuilibrator    │  Calculates thermodynamic Keq for each reaction
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Concentration   │  Assigns starting metabolite concentrations
│  assignment      │  from literature or imputed averages
└────────┬─────────┘
         │
         ▼
 Reaction.csv  +  SpeciesBaseMechanisms.csv
         │
         ▼
┌──────────────────┐
│  ODBM converter  │  Generates Antimony model string
└────────┬─────────┘
         │
         ▼
    model.ant  (optionally → model.xml via antimony package)
         │
         ▼
┌──────────────────┐
│    Antotate      │  Appends database cross-references
└────────┬─────────┘
         │
         ▼
 model_kegg_chebi.ant
```

---

## Requirements

- Python 3.11
- Conda (recommended for environment management)
- Internet access (for KEGG, UniProt, PubChem, and CatPred API calls)
- BRENDA database file (~250 MB, downloaded separately — see below)

---

## Installation

### 1. Create the conda environment

```bash
conda create -n frenda python=3.11
conda activate frenda
```

### 2. Install dependencies

```bash
pip install pandas numpy requests brendapyrser
pip install equilibrator-api
```

The `equilibrator-api` package bundles a local thermodynamic database (~500 MB) and will download it on first use.

To optionally export models to SBML (the `--sbml` flag in ODBM):

```bash
pip install antimony
```

To optionally annotate models with Antotate:

```bash
# equilibrator-api is already installed above; no additional packages required
```

### 3. Clone or download this repository

```bash
git clone https://github.com/<your-username>/FRENDA.git
cd FRENDA
```

---

## Downloading BRENDA

BRENDA kinetic parameters are retrieved from a local copy of the BRENDA flat-file database. This file is not distributed with FRENDA due to its size (~250 MB) and BRENDA's licensing terms.

**To download it:**

1. Go to [https://www.brenda-enzymes.org/download.php](https://www.brenda-enzymes.org/download.php)
2. Create a free account if you do not already have one
3. Download **"BRENDA text file"** (the full database download)
4. Save it as `data/brenda_download.txt` inside this repository

```
FRENDA/
└── data/
    └── brenda_download.txt   ← place the file here
```

The `data/` directory already contains the KEGG snapshots and metabolite concentration reference used by the pipeline; only `brenda_download.txt` needs to be added manually.

---

## Quick Start

```bash
# from the repository root (the folder containing frenda/, data/, examples/, ...)
conda activate frenda

python -m frenda.pipeline examples/proteome_exe.csv \
    --organism "Escherichia coli" \
    --output-dir frenda_output/
```

This produces `frenda_output/Reaction.csv` and `frenda_output/SpeciesBaseMechanisms.csv`.

To then generate an Antimony model:

```bash
python -m odbm.convert frenda_output/Reaction.csv frenda_output/SpeciesBaseMechanisms.csv
```

---

## Input Format

The pipeline accepts a CSV file with the following columns:

| Column | Required | Description |
|--------|----------|-------------|
| `EC` | Yes | Enzyme Commission number (e.g. `1.1.1.1`) |
| `Abundances` | Yes | Enzyme abundance (any consistent unit; used as starting concentration) |
| `Species` | No | Organism name for this specific enzyme (e.g. `Saccharomyces cerevisiae`). Overrides `--organism` for that row. Leave blank to use the global default. |

### Example (`examples/proteome_exe.csv`)

```
EC,Abundances,Species
1.1.1.1,64.5,
6.2.1.3,47.5,
2.7.1.40,57,
1.1.1.37,150,Saccharomyces cerevisiae
```

- EC `1.1.1.37` will use organism `Saccharomyces cerevisiae` regardless of the global `--organism` flag
- The other three ECs will use the global `--organism` value, or all organisms if `--organism` is not specified

**Notes:**
- EC numbers must match entries in the KEGG database. FRENDA uses a local KEGG snapshot (`data/kegg_enzymes.json.gz`, `data/kegg_reactions.json.gz`) for reaction retrieval.
- Abundances can be relative (e.g. spectral counts) or absolute (e.g. nM). The units are carried through as-is and applied as the enzyme starting concentration in the model.

---

## Running the Pipeline

```
python -m frenda.pipeline <proteome_csv> [options]
```

### Required argument

| Argument | Description |
|----------|-------------|
| `proteome_csv` | Path to the input CSV file |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--organism NAME` | *(all organisms)* | Organism name for BRENDA filtering (e.g. `"Escherichia coli"`). Applied to any row whose `Species` column is blank. If omitted, parameters from all organisms are averaged. |
| `--first-reaction-only` | *(all reactions)* | Include only the first KEGG reaction listed for each enzyme. Useful for producing smaller, more focused models. |
| `--output-dir DIR` | `frenda_output` | Directory to write `Reaction.csv` and `SpeciesBaseMechanisms.csv`. Created if it does not exist. |
| `--brenda PATH` | `data/brenda_download.txt` | Path to the BRENDA flat-file database. |
| `--kegg-dir DIR` | `data/` | Directory containing `kegg_enzymes.json.gz` and `kegg_reactions.json.gz`. |
| `--conc-ref PATH` | `data/Metabolite_Concentrations.csv` | Path to the metabolite concentration reference table. |
| `--catpred-url URL` | `https://www.catpred.com` | Base URL for the CatPred prediction server. |
| `--no-catpred` | *(CatPred enabled)* | Skip CatPred ML predictions. Missing kinetic parameters will remain `NaN` and may be imputed from global averages. |
| `--manual-ec PATH` | *(none)* | Path to a CSV of manually curated entries. Columns: `EC`, `Conc`, `Accession Number`. Rows in this file supplement or override automatic retrieval. |
| `--thermo-cache PATH` | *(none)* | Pickle file for caching eQuilibrator Keq calculations between runs. Significantly speeds up repeated runs on overlapping reaction sets. |
| `--catpred-cache PATH` | *(none)* | Pickle file for caching UniProt, PubChem, and CatPred results. |
| `-v, --verbose` | *(INFO logging)* | Enable DEBUG-level logging. |

### Examples

**Basic run (all organisms, all reactions):**
```bash
python -m frenda.pipeline proteome.csv
```

**Filter to E. coli kinetics, only primary reactions, with caching:**
```bash
python -m frenda.pipeline proteome.csv \
    --organism "Escherichia coli" \
    --first-reaction-only \
    --thermo-cache thermo_cache.pkl \
    --catpred-cache catpred_cache.pkl \
    --output-dir results/
```

**Skip CatPred (faster, more missing values):**
```bash
python -m frenda.pipeline proteome.csv \
    --organism "Homo sapiens" \
    --no-catpred
```

---

## Output Files

### `Reaction.csv`

One row per reaction. Each enzyme may produce multiple rows if it catalyzes multiple KEGG reactions.

| Column | Description |
|--------|-------------|
| `Label` | Reaction identifier (R1, R2, …) |
| `Enzyme` | Human-readable enzyme name from KEGG |
| `Reaction ID` | KEGG reaction ID (e.g. `R07326`) |
| `Substrates` | Semicolon-separated list of `stoichiometry KEGG_ID` pairs (e.g. `1 C00003; 1 C00069`) |
| `Products` | Same format as Substrates |
| `EC` | Enzyme Commission number |
| `Mechanism` | Rate law identifier (default: `MRL` — Common Modular Rate Law) |
| `Km` | Michaelis constants as `key: value` pairs (e.g. `Km_C00003: 1.13; Km_C00069: 0.82`), in mM |
| `Kcat` | Catalytic rates as `Kcat_F: value; Kcat_R: value`, in s⁻¹. `nan` if not found. |
| `Inhibitors` | Semicolon-separated KEGG IDs of known inhibitors |
| `KI` | Inhibition constants as `KEGG_ID_KI: value` pairs, in mM |
| `Keq` | Thermodynamic equilibrium constant (dimensionless), calculated by eQuilibrator |

**Parameter string format:** All kinetic parameter columns use a consistent `key: value; key: value` format that is parsed directly by the ODBM model builder.

---

### `SpeciesBaseMechanisms.csv`

One row per species (enzymes and metabolites combined).

| Column | Description |
|--------|-------------|
| `Label` | Human-readable species name (enzyme name or KEGG compound ID) |
| `EC` | EC number (enzymes only) |
| `Type` | `Enzyme` or `Metabolite` |
| `StartingConc` | Starting concentration (abundance units for enzymes; mM for metabolites) |
| `Conc` | Source of concentration value (experimental reference, literature, or imputed) |
| `Mechanisms` | Rate law override (leave blank to use `Reaction.csv`) |
| `Parameters` | Additional species-level parameters (inhibitor constants, etc.) |
| `Species` | Organism associated with this enzyme |

**Concentration assignment hierarchy for metabolites:**
1. Experimentally measured value from `data/Metabolite_Concentrations.csv` (if available for the compound)
2. Literature value from the same reference file
3. Global average of all measured metabolite concentrations in the reference dataset (~0.001 mM if no data is available)

---

## Converting to Antimony/SBML (ODBM)

The **ODBM** (ODE-Based Model builder) module converts the FRENDA output tables into an executable model in Antimony format. Antimony is a human-readable text format for systems biology models that can be compiled to SBML.

```
python -m odbm.convert <reaction_csv> <species_csv> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `reaction_csv` | Path to `Reaction.csv` from FRENDA output |
| `species_csv` | Path to `SpeciesBaseMechanisms.csv` from FRENDA output |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output, -o PATH` | `<reaction_csv stem>.ant` | Output file path for the Antimony model |
| `--no-enzyme-degradation` | *(degradation included)* | Omit first-order enzyme degradation reactions from the model |
| `--sbml` | *(Antimony only)* | Also write an SBML `.xml` file (requires `pip install antimony`) |
| `-v, --verbose` | *(INFO logging)* | Enable debug logging |

### Example

```bash
python -m odbm.convert frenda_output/Reaction.csv frenda_output/SpeciesBaseMechanisms.csv \
    --output my_model.ant \
    --sbml
```

### Rate law: Common Modular Rate Law (MRL)

The default rate law is the **Common Modular Rate Law** (Liebermeister & Klipp, 2006), which:

- Is thermodynamically consistent via the Haldane relationship (Keq = Kcat_F·∏Km_P / Kcat_R·∏Km_S)
- Handles arbitrary numbers of substrates and products
- Supports non-competitive and competitive inhibition terms
- Reduces to Michaelis-Menten kinetics for single-substrate reactions

The net rate for reaction R is:

```
R = E × [kcat_F × (∏S/∏Km_S) − kcat_R × (∏P/∏Km_P)] / D
```

where `D = ∏(1 + S/Km_S) + ∏(1 + P/Km_P) − 1` is the saturation denominator.

### Model variables

| Variable | Description |
|----------|-------------|
| `dilution_factor` | Global dilution multiplier applied to all starting concentrations (default: 1) |
| `Kcat_V_Rn` | Geometric mean of Kcat_F and Kcat_R for reaction n; used in Haldane-consistent rate expressions |
| `Gnc_X_eECnnnn` | Non-competitive inhibition weight for inhibitor X on enzyme eECnnnn (default: 1) |
| `Gc_X_eECnnnn` | Competitive inhibition weight for inhibitor X on enzyme eECnnnn (default: 1) |
| `kdeg_eECnnnn` | First-order degradation rate constant for enzyme eECnnnn (default: 1×10⁻⁴ s⁻¹) |

### Antimony output structure

```
# Initialize concentrations
species eEC1111;
eEC1111 = 64.5*dilution_factor;
...

# Initialize parameters
Kcat_F_R1 = 0.029;
Km_C00003_eEC1111 = 1.13;
Keq_R1 = 1.08e8;
...

# Initialize variables
dilution_factor = 1;
Kcat_V_R1 = 0.029;
...

# Define reactions
R1 : 1 C00003 + eEC1111 -> eEC1111 + 1 C00004;
R1_D := ...;
R1_f := ...;
R1_r := ...;
R1 = R1_f - R1_r;
...
```

---

## Annotating the Model (Antotate)

**Antotate** appends standardized database cross-references to an Antimony model file, producing a new annotated copy. Metabolites identified by KEGG ID (e.g. `C00003`) are resolved to their database identifiers automatically; free-text species names are matched via eQuilibrator's compound search.

```
python antotate.py <model_file> [--databases kegg chebi hmdb] [--confidence-out metrics.csv]
```

*Antotate is located in the `../annotating/` directory.*

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `model_file` | *(required)* | Path to an Antimony `.ant` or `.txt` model file |
| `--databases` | `kegg chebi` | Space-separated list of databases to include. Choices: `kegg`, `chebi`, `bigg.metabolite`, `hmdb`, `metacyc.compound` |
| `--confidence-out PATH` | `confidence_metrics.csv` | Output path for annotation confidence CSV |
| `-v, --verbose` | *(INFO logging)* | Enable debug logging |

### Supported databases

| Namespace | Description | URL pattern |
|-----------|-------------|-------------|
| `kegg` | KEGG Compound | `http://identifiers.org/kegg/{ID}` |
| `chebi` | ChEBI | `https://www.ebi.ac.uk/chebi/searchId.do?chebiId={ID}` |
| `bigg.metabolite` | BiGG metabolites | `http://bigg.ucsd.edu/universal/metabolites/{ID}` |
| `hmdb` | Human Metabolome Database | `https://hmdb.ca/metabolites/{ID}` |
| `metacyc.compound` | MetaCyc | `https://metacyc.org/compound?orgid=META&id={ID}` |

### Example

```bash
cd annotating/
python antotate.py ../frenda_output/model.ant \
    --databases kegg chebi hmdb \
    --confidence-out annotation_confidence.csv
```

**Output:** `model_kegg_chebi_hmdb.ant` with appended annotation lines:

```
C00003 is "Nicotinamide adenine dinucleotide";
C00003 identity "http://identifiers.org/kegg/C00003";
C00003 identity "https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:15846";
C00003 identity "https://hmdb.ca/metabolites/HMDB01487";
```

### Confidence scores

The confidence CSV reports how well each species name matched the database:

| Column | Description |
|--------|-------------|
| `Given ID` | Species identifier from the model |
| `Display Name` | Resolved common name |
| `Annotated Identities` | All assigned database IDs |
| `Confidence Score` | 1.0 for exact KEGG ID matches; rank-based score (0–1) for text searches |

Scores below ~0.8 may indicate ambiguous matches and should be reviewed manually.

---

## Full End-to-End Example

```bash
# Activate environment
conda activate frenda

# Step 1: Run the FRENDA pipeline
python -m frenda.pipeline examples/proteome_exe.csv \
    --organism "Escherichia coli" \
    --first-reaction-only \
    --thermo-cache thermo_cache.pkl \
    --catpred-cache catpred_cache.pkl \
    --output-dir results/

# Step 2: Convert to Antimony
python -m odbm.convert results/Reaction.csv results/SpeciesBaseMechanisms.csv \
    --output results/model.ant

# Step 3: Annotate
cd annotating/
python antotate.py ../results/model.ant \
    --databases kegg chebi \
    --confidence-out ../results/annotation_confidence.csv
```

**Result:** `results/model_kegg_chebi.ant` — a fully parameterized, annotated ODE model ready for simulation in Tellurium, COPASI, or any SBML-compatible tool.

---

## Python API

All pipeline steps can be called programmatically.

### Running the full pipeline

```python
from frenda.pipeline import run_pipeline

rxn_df, sbm_df = run_pipeline(
    "examples/proteome_exe.csv",
    organism="Escherichia coli",
    first_reaction_only=True,
    output_dir="results/",
    thermo_cache="thermo_cache.pkl",
    catpred_cache="catpred_cache.pkl",
)
```

### Building an Antimony model from DataFrames

```python
import pandas as pd
from odbm.model_builder import ModelBuilder

rxn_df = pd.read_csv("results/Reaction.csv")
spc_df = pd.read_csv("results/SpeciesBaseMechanisms.csv")

mb = ModelBuilder(spc_df, rxn_df)
model_str = mb.compile(enzyme_degradation=True)

with open("results/model.ant", "w") as f:
    f.write(model_str)
```

### Annotating programmatically

```python
from annotating.antotate import Annotate

annotator = Annotate()
out_path = annotator.annotate(
    "results/model.ant",
    databases=["kegg", "chebi"],
)
print(f"Annotated model: {out_path}")
```

---

## Caching

Two caches are available to significantly speed up repeated runs:

### `--thermo-cache` (eQuilibrator)

eQuilibrator compound lookup is the primary bottleneck (~2 s per unique compound name on first load). The thermo cache persists resolved compound objects and Keq values between runs.

```bash
# First run: slow, writes cache
python -m frenda.pipeline proteome.csv --thermo-cache thermo_cache.pkl

# Subsequent runs: fast
python -m frenda.pipeline proteome2.csv --thermo-cache thermo_cache.pkl
```

### `--catpred-cache` (UniProt / PubChem / CatPred)

The CatPred cache persists protein sequences (from UniProt), SMILES strings (from PubChem), and CatPred ML predictions. Each CatPred API call takes several seconds; caching eliminates redundant calls for enzymes already processed.

```bash
python -m frenda.pipeline proteome.csv --catpred-cache catpred_cache.pkl
```

Both caches are standard Python pickle files and can be safely deleted to force a fresh run.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'brendapyrser'`
Install the BRENDA parser: `pip install brendapyrser`

### `No reactions found for EC X.X.X.X`
The EC number is not present in the local KEGG snapshot. Check that the EC number is valid at [https://www.kegg.jp/kegg/enzyme/](https://www.kegg.jp/kegg/enzyme/). The KEGG data files in `data/` are a static snapshot; very recently added entries may be missing.

### `FileNotFoundError: data/brenda_download.txt`
You need to download the BRENDA flat file manually. See [Downloading BRENDA](#downloading-brenda).

### CatPred returns NaN for many entries
CatPred can fail server-side for structurally complex molecules (e.g. CoA, Acyl-CoA). This is expected behavior — FRENDA falls back to imputing missing values from the average of other measured kcat values in the same run. Use `--no-catpred` to skip predictions entirely.

### eQuilibrator Keq is very large or very small
Keq values are computed from ΔG°' at pH 7, 25 °C, ionic strength 0.1 M. Extremely large or small values (e.g. >10¹⁰) indicate highly favorable or unfavorable reactions and are numerically valid. The Haldane relationship in the MRL rate law handles these correctly.

### Antimony model doesn't load in Tellurium
Ensure `dilution_factor` is set to 1 (or your desired value) before simulation. Parameters that are `nan` in the CSV will appear as `nan` in the model and must be set manually before simulating.

### `python -m odbm.convert` finds the wrong `odbm` package
If there is another `odbm` package on your Python path, run the converter via a script from the repository root directory:
```python
# run_convert.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
from odbm.convert import main
main()
```

---

## References

- **KEGG:** Kanehisa, M. et al. KEGG for integration and interpretation of large-scale molecular data sets. *Nucleic Acids Research* (2012). [https://www.kegg.jp](https://www.kegg.jp)
- **BRENDA:** Chang, A. et al. BRENDA, the ELIXIR core data resource in 2021. *Nucleic Acids Research* (2021). [https://www.brenda-enzymes.org](https://www.brenda-enzymes.org)
- **CatPred:** Singh, M. et al. CatPred: A comprehensive framework for deep learning in vitro enzyme kinetic parameters. *bioRxiv* (2024). [https://github.com/maranasgroup/CatPred](https://github.com/maranasgroup/CatPred)
- **eQuilibrator:** Beber, M. et al. eQuilibrator 3.0: a database solution for thermodynamic constant estimation. *Nucleic Acids Research* (2022). [https://equilibrator.weizmann.ac.il](https://equilibrator.weizmann.ac.il)
- **Common Modular Rate Law:** Liebermeister, W. & Klipp, E. Bringing metabolic networks to life: convenience rate law and thermodynamic constraints. *Theoretical Biology and Medical Modelling* (2006).
- **Antimony:** Smith, L.P. et al. Antimony: a modular model definition language. *Bioinformatics* (2009). [https://antimony.sourceforge.net](https://antimony.sourceforge.net)
