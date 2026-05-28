"""Antotate: annotate metabolites and enzymes in Antimony model files.

Takes an Antimony-format model, identifies all species, and appends
standardized database identifiers (KEGG, ChEBI, HMDB, BiGG, MetaCyc)
to a new annotated copy of the file.

Usage
-----
python antotate.py model.ant [--databases kegg chebi hmdb]
"""

import argparse
import csv
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from equilibrator_api import ComponentContribution
    _HAS_EQUILIBRATOR = True
except ImportError:
    _HAS_EQUILIBRATOR = False
    logger.warning("equilibrator_api not found; compound database lookup disabled.")

# KEGG compound/drug/glycan ID pattern
_KEGG_ID_RE = re.compile(r'^[CDGR]\d{5}$')

# Enzyme label pattern used by FRENDA (eEC1111, hEC27140, etc.)
_ENZYME_LABEL_RE = re.compile(r'^[eh]EC\d+$')

# URL templates per database namespace
_DB_LINKS = {
    'kegg':            'http://identifiers.org/kegg/{}',
    'chebi':           'https://www.ebi.ac.uk/chebi/searchId.do?chebiId={}',
    'bigg.metabolite': 'http://bigg.ucsd.edu/universal/metabolites/{}',
    'hmdb':            'https://hmdb.ca/metabolites/{}',
    'metacyc.compound':'https://metacyc.org/compound?orgid=META&id={}',
}

# Namespaces that may return many aliases — keep only the first
_FIRST_ONLY = {'kegg', 'chebi', 'bigg.metabolite', 'hmdb', 'metacyc.compound'}


class Annotate:
    """Annotate species in an Antimony model with database cross-references."""

    def __init__(self):
        self._cc = None  # lazy-loaded to avoid slow startup when not needed

    def _get_cc(self) -> 'ComponentContribution':
        if self._cc is None:
            if not _HAS_EQUILIBRATOR:
                raise RuntimeError("equilibrator_api is required for compound lookup.")
            self._cc = ComponentContribution()
        return self._cc

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_enzymes(self, file_path: str) -> list:
        """Return species that appear on both sides of any reaction arrow (catalysts)."""
        catalysts = set()
        with open(file_path) as f:
            for line in f:
                if '->' not in line:
                    continue
                parts = line.split(':', 1)
                if len(parts) < 2:
                    continue
                reaction_part = parts[1].strip().rstrip(';')
                left, _, right = reaction_part.partition('->')
                left_tokens = set(left.split()) - {'+'}
                right_tokens = set(right.split()) - {'+'}
                catalysts.update(
                    t for t in left_tokens & right_tokens
                    if not t.isdigit()
                )
        return list(catalysts)

    def parse_species(self, file_path: str) -> list:
        """Return all species IDs from `species X;` declarations in the model."""
        species = []
        with open(file_path) as f:
            for line in f:
                m = re.match(r'^\s*species\s+(\w+)\s*;', line)
                if m:
                    species.append(m.group(1))
        return species

    # ------------------------------------------------------------------
    # Compound lookup
    # ------------------------------------------------------------------

    def _lookup_compound(self, spc: str) -> tuple:
        """
        Return (display_name, {namespace: accession}) for a species.

        For KEGG IDs (C/D/G/R + 5 digits), uses the registry lookup which
        is exact and returns rich cross-references.  For free-text names,
        falls back to eQuilibrator's text search.
        """
        cc = self._get_cc()
        compound = None

        if _KEGG_ID_RE.match(spc):
            compound = cc.ccache.get_compound_from_registry('kegg', spc)

        if compound is None:
            compound = cc.search_compound(spc)

        if compound is None:
            logger.debug("No compound found for %s", spc)
            return spc, {}

        display_name = compound.get_common_name() or spc

        identities = {}
        for ident in compound.identifiers:
            ns = ident.registry.namespace
            if ns in _FIRST_ONLY and ns not in identities:
                identities[ns] = ident.accession
            elif ns not in _FIRST_ONLY:
                pass  # skip verbose namespaces (reactome, seed, metanetx, synonyms, etc.)

        return display_name, identities

    def _confidence_score(self, spc: str) -> float:
        """
        Return a [0, 1] confidence score for the annotation.

        KEGG IDs are exact registry matches → 1.0.
        For text searches, use the rank score from ccache.search().
        """
        if _KEGG_ID_RE.match(spc):
            return 1.0
        try:
            cc = self._get_cc()
            results = cc.ccache.search(spc, page=1, page_size=25)
            result = cc.search_compound(spc)
            if result is None:
                return 0.0
            score = next((s for c, s in results if c.id == result.id), 0.0)
            return float(score)
        except Exception as e:
            logger.debug("Confidence score failed for %s: %s", spc, e)
            return float('nan')

    # ------------------------------------------------------------------
    # Annotation pipeline
    # ------------------------------------------------------------------

    def annotate_species(self, species_list: list, databases: list) -> tuple:
        """
        Look up each species and filter identities to the requested databases.

        Returns
        -------
        annotations : list of (spc, display_name, {db: accession})
        confidence_scores : list of (spc, display_name, {db: accession}, score)
        """
        annotations = []
        confidence_scores = []
        for spc in species_list:
            display_name, all_identities = self._lookup_compound(spc)
            identities = {db: acc for db, acc in all_identities.items() if db in databases}
            score = self._confidence_score(spc)
            annotations.append((spc, display_name, identities))
            confidence_scores.append((spc, display_name, identities, score))
            logger.debug("%s -> %s  (score=%.2f)", spc, display_name, score if score == score else -1)
        return annotations, confidence_scores

    def write_annotations(
        self,
        file_path: str,
        annotations: list,
        enzyme_list: list,
        databases: list,
    ) -> str:
        """
        Append annotation lines to a copy of the model file.

        The output file is named  <stem>_<db1>_<db2><ext>
        (e.g. model_kegg_chebi.ant).

        Returns the path to the new file.
        """
        p = Path(file_path)
        db_suffix = '_'.join(databases)
        out_path = p.parent / f"{p.stem}_{db_suffix}{p.suffix}"

        lines = [p.read_text()]

        for spc, display_name, identities in annotations:
            lines.append(f'{spc} is "{display_name}";\n')
            for db, acc in identities.items():
                url = _DB_LINKS.get(db, '').format(acc)
                if url:
                    lines.append(f'{spc} identity "{url}";\n')

        for enzyme in enzyme_list:
            lines.append(f'{enzyme} is "{enzyme}";\n')

        out_path.write_text(''.join(lines))
        logger.info("Annotated model written to %s", out_path)
        return str(out_path)

    def write_confidence_metrics(
        self,
        confidence_scores: list,
        output_path: str = 'confidence_metrics.csv',
    ):
        """Write per-species annotation confidence to a CSV file."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['Given ID', 'Display Name', 'Annotated Identities', 'Confidence Score'],
            )
            writer.writeheader()
            for spc, display_name, identities, score in confidence_scores:
                writer.writerow({
                    'Given ID': spc,
                    'Display Name': display_name,
                    'Annotated Identities': ', '.join(f'{db}:{acc}' for db, acc in identities.items()),
                    'Confidence Score': f'{score:.4f}' if score == score else 'nan',
                })

    def annotate(self, file_path: str, databases: list) -> str:
        """
        Run the full annotation pipeline.

        Parameters
        ----------
        file_path : path to the Antimony model file
        databases : list of database namespaces to include

        Returns
        -------
        Path to the annotated output file.
        """
        enzyme_list = self.parse_enzymes(file_path)
        all_species = self.parse_species(file_path)
        # Also treat any species whose name matches the FRENDA enzyme label
        # pattern (eEC.../hEC...) as an enzyme even if not found in reactions.
        enzyme_set = set(enzyme_list) | {s for s in all_species if _ENZYME_LABEL_RE.match(s)}
        enzyme_list = list(enzyme_set)
        metabolite_list = [s for s in all_species if s not in enzyme_set]

        logger.info(
            "Found %d metabolite(s) and %d enzyme(s)",
            len(metabolite_list), len(enzyme_list),
        )

        annotations, confidence_scores = self.annotate_species(metabolite_list, databases)
        out_path = self.write_annotations(file_path, annotations, enzyme_list, databases)
        self.write_confidence_metrics(confidence_scores)
        return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Annotate metabolites and enzymes in an Antimony model file.',
    )
    parser.add_argument('file_path', help='Path to the Antimony (.ant or .txt) model file.')
    parser.add_argument(
        '--databases', nargs='+',
        choices=list(_DB_LINKS),
        default=['kegg', 'chebi'],
        help='Databases to annotate against (default: kegg chebi).',
    )
    parser.add_argument(
        '--confidence-out',
        default='confidence_metrics.csv',
        help='Output path for confidence metrics CSV (default: confidence_metrics.csv).',
    )
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    annotator = Annotate()
    out = annotator.annotate(args.file_path, args.databases)
    print(f'Annotated model written to: {out}')
