"""ModelBuilder: converts FRENDA parameter tables into an Antimony model string."""

import logging
from copy import deepcopy

import numpy as np
import pandas as pd

from .mechanisms import MECHANISMS, Mechanism
from .utils import extractParams, fmt

logger = logging.getLogger(__name__)


def _ec_label(rxn: pd.Series) -> str:
    """Return the enzyme variable name (e.g. eEC1111) for a reaction row."""
    accession = rxn.get('Accession Number') if hasattr(rxn, 'get') else None
    prefix = 'hEC' if accession == 'Heterologous' else 'eEC'
    return prefix + str(rxn['EC']).replace('.', '')


class ModelBuilder:
    """Assembles an Antimony kinetic model from FRENDA output DataFrames.

    Parameters
    ----------
    species : pd.DataFrame
        SpeciesBaseMechanisms.csv loaded as a DataFrame.
    reactions : pd.DataFrame
        Reaction.csv loaded as a DataFrame.
    """

    def __init__(self, species: pd.DataFrame, reactions: pd.DataFrame):
        self.mech_dict: dict = {}
        for m in MECHANISMS:
            self.mech_dict[m.name] = m

        self.species = species.copy()
        self.rxns = reactions.copy()

        # Collect all compound IDs that appear as substrates, products, or inhibitors
        self.rxn_species: list[str] = []
        for col in ('Substrates', 'Products'):
            for cell in self.rxns[col]:
                if isinstance(cell, str):
                    for token in cell.split('; '):
                        parts = token.strip().split(' ')
                        if len(parts) == 2:
                            cid = parts[1]
                            if cid not in self.rxn_species:
                                self.rxn_species.append(cid)

        for cell in self.rxns['Inhibitors']:
            if isinstance(cell, str):
                for i in cell.split(';'):
                    i = i.strip()
                    if i and i not in self.rxn_species:
                        self.rxn_species.append(i)

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def _write_species(self, sp: pd.Series) -> str:
        sp_type = str(sp.get('Type', ''))
        ec = str(sp.get('EC', ''))

        if sp_type in ('Endogenous Enzyme', 'Enzyme'):
            label = 'eEC' + ec.replace('.', '')
            present = ''
        elif sp_type == 'Heterologous Enzyme':
            label = 'hEC' + ec.replace('.', '')
            present = ''
        else:
            label = fmt(str(sp['Label']))
            present = ''

        conc = sp['StartingConc']
        s_str = f'species {label};\n'
        s_str += f'{label} = {present}{conc}*dilution_factor; \n'
        return s_str

    def _write_reaction(self, rxn: pd.Series) -> str:
        mech_str = rxn.get('Mechanism', 'MRL')
        if not isinstance(mech_str, str):
            mech_str = 'MRL'
        mechs = [m.strip() for m in mech_str.split(';')]

        try:
            M = self.mech_dict[mechs[0]](rxn)
        except KeyError:
            raise KeyError(f'No mechanism called "{mechs[0]}"')

        eq_str = M.writeEquation() + '; \n'
        rate_str = M.writeRate()

        # Skip any rate variables already written to avoid duplicates
        rate_vars = rate_str.split('; \n')
        rate_str = '; \n'.join(v for v in rate_vars if v not in self.r_str)

        return '\n' + eq_str + rate_str + '; '

    def _write_parameters(self, param_str, label: str, required: bool = True) -> str:
        if pd.isnull(param_str) or not str(param_str).strip():
            if required:
                logger.warning('No parameters found for %s', label)
            return ''
        try:
            pdict = extractParams(str(param_str))
        except Exception as e:
            logger.debug('Could not parse parameters for %s: %s', label, e)
            return ''

        p_str = ''
        for key, value in pdict.items():
            if '$' in key:
                full_key = key.replace('$', '_')
            else:
                full_key = key + '_' + label
            if full_key + ' ' not in self.p_str and full_key + ' ' not in p_str:
                p_str += f'{full_key} = {value}; \n'

        if p_str:
            p_str += '\n'
        return p_str

    def _write_variable(self, name, value=1) -> str:
        if pd.isnull(name) or not str(name).strip():
            return ''
        name = str(name)
        if name + ' ' not in self.v_str:
            return f'{name} = {value}; \n'
        return ''

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile(self, enzyme_degradation: bool = True) -> str:
        """Generate and return the full Antimony model string."""

        self.s_str = '# Initialize concentrations \n'
        self.p_str = '\n# Initialize parameters \n'
        self.v_str = '\n# Initialize variables \n'
        self.r_str = '# Define reactions \n'

        # Write species (enzymes first, then metabolites)
        for _, sp in self.species.iterrows():
            sp_type = str(sp.get('Type', ''))
            label = str(sp['Label'])
            in_rxn_species = label in self.rxn_species
            is_enzyme = 'Enzyme' in sp_type and label in self.rxns['Enzyme'].values

            if in_rxn_species or is_enzyme:
                self.s_str += self._write_species(sp)
                self.s_str += self._write_parameters(sp.get('Parameters'), label, required=False)

        self.v_str += self._write_variable('dilution_factor')

        # Track one representative reaction per enzyme for EED
        enzyme_rxns: dict = {}

        for _, rxn in self.rxns.iterrows():
            EC = _ec_label(rxn)
            if EC not in enzyme_rxns:
                enzyme_rxns[EC] = deepcopy(rxn)

            # Kcat parameters (Kcat_F_R1, Kcat_R_R1)
            self.p_str += self._write_parameters(rxn.get('Kcat'), rxn['Label'])

            # Keq as a named parameter (Keq_R1)
            keq = rxn.get('Keq')
            if pd.notna(keq):
                self.p_str += self._write_parameters(f'Keq: {keq}', rxn['Label'])

            # Kcat_V: geometric mean of Kcat_F and Kcat_R for Haldane consistency
            kcat_v = self._kcat_v(rxn.get('Kcat'))
            self.v_str += self._write_variable(f'Kcat_V_{rxn["Label"]}', kcat_v)

            # Km (and KI) parameters keyed to the enzyme
            km_str = str(rxn.get('Km', '')) if pd.notna(rxn.get('Km')) else ''
            ki_str = ''
            if pd.notna(rxn.get('KI')):
                ki_raw = ';'.join([
                    I for I in np.unique(str(rxn['KI']).split(';'))
                    if np.all([c not in I for c in ['D', 'G']])
                ])
                ki_str = '; ' + ki_raw if ki_raw else ''
            self.p_str += self._write_parameters(km_str + ki_str, EC)

            # Inhibitor auxiliary variables
            if pd.notna(rxn.get('KI')) and pd.notna(rxn.get('Inhibitors')):
                for inh in str(rxn['Inhibitors']).split(';'):
                    inh = inh.strip()
                    if inh and np.all([c not in inh for c in ['D', 'G']]):
                        self.v_str += self._write_variable(f'Gnc_{inh}_{EC}', 1)
                        self.v_str += self._write_variable(f'Gc_{inh}_{EC}', 1)

            self.r_str += self._write_reaction(rxn) + '\n'

        # Enzyme degradation reactions
        if enzyme_degradation:
            for EC, rxn in enzyme_rxns.items():
                rxn = rxn.copy()
                rxn['Mechanism'] = 'EED'
                self.r_str += self._write_reaction(rxn) + '\n'
                self.v_str += self._write_variable(f'kdeg_{EC}', 1e-4)

        return self.s_str + self.p_str + self.v_str + self.r_str

    @staticmethod
    def _kcat_v(kcat_str) -> float:
        """Return Kcat_V as the geometric mean of Kcat_F and Kcat_R (or just Kcat_F)."""
        if not isinstance(kcat_str, str) or ':' not in kcat_str:
            return 1.0
        try:
            params = extractParams(kcat_str)
            kf = float(params.get('Kcat_F', np.nan))
            kr = float(params.get('Kcat_R', np.nan))
            if np.isfinite(kf) and np.isfinite(kr) and kf > 0 and kr > 0:
                return float(np.sqrt(kf * kr))
            if np.isfinite(kf) and kf > 0:
                return kf
        except Exception:
            pass
        return 1.0

    def save_model(self, filename: str) -> None:
        """Compile and write the Antimony model to a file."""
        with open(filename, 'w') as f:
            f.write(self.compile())
        logger.info('Antimony model written to %s', filename)
