"""
FRENDA - Fast Retrieval of ENzyme DAta

Converts proteomics data (EC numbers + abundances) into parameterized kinetic
metabolic model tables suitable for simulation in Antimony/SBML.

Typical usage:
    python -m frenda.pipeline proteome.csv --organism "Escherichia coli"
"""

from .pipeline import run_pipeline

__all__ = ["run_pipeline"]
