"""CLI adapter: convert FRENDA output CSVs to Antimony (and optionally SBML)."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .model_builder import ModelBuilder


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog='python -m odbm.convert',
        description='Convert FRENDA Reaction.csv + SpeciesBaseMechanisms.csv to Antimony/SBML.',
    )
    p.add_argument('reaction_csv', type=Path, help='Path to Reaction.csv from FRENDA output.')
    p.add_argument('species_csv', type=Path, help='Path to SpeciesBaseMechanisms.csv from FRENDA output.')
    p.add_argument(
        '--output', '-o', type=Path, default=None,
        help='Output file path (default: <reaction_csv stem>.ant next to reaction_csv).',
    )
    p.add_argument(
        '--no-enzyme-degradation', action='store_true',
        help='Omit first-order enzyme degradation reactions from the model.',
    )
    p.add_argument(
        '--sbml', action='store_true',
        help='Also write an SBML file (requires the `antimony` Python package).',
    )
    p.add_argument('--verbose', '-v', action='store_true', help='Enable debug logging.')
    return p.parse_args(argv)


def _to_sbml(ant_path: Path) -> None:
    try:
        import antimony as ant
    except ImportError:
        logging.warning('antimony package not found; skipping SBML export. Install with: pip install antimony')
        return

    code = ant_path.read_text()
    ret = ant.loadAntimonyString(code)
    if ret < 0:
        logging.error('Antimony failed to parse model: %s', ant.getLastError())
        return

    sbml = ant.getSBMLString()
    sbml_path = ant_path.with_suffix('.xml')
    sbml_path.write_text(sbml)
    logging.info('SBML written to %s', sbml_path)
    print(f'SBML written to {sbml_path}')


def main(argv=None):
    args = _parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=level)

    rxn_path: Path = args.reaction_csv
    spc_path: Path = args.species_csv

    for p in (rxn_path, spc_path):
        if not p.exists():
            logging.error('File not found: %s', p)
            sys.exit(1)

    rxn_df = pd.read_csv(rxn_path)
    spc_df = pd.read_csv(spc_path)

    logging.info('Loaded %d reactions from %s', len(rxn_df), rxn_path)
    logging.info('Loaded %d species from %s', len(spc_df), spc_path)

    mb = ModelBuilder(spc_df, rxn_df)
    model_str = mb.compile(enzyme_degradation=not args.no_enzyme_degradation)

    out_path: Path = args.output or rxn_path.parent / (rxn_path.stem.replace('Reaction', 'model') + '.ant')
    out_path.write_text(model_str)
    logging.info('Antimony model written to %s', out_path)
    print(f'Antimony model written to {out_path}')

    if args.sbml:
        _to_sbml(out_path)


if __name__ == '__main__':
    main()
