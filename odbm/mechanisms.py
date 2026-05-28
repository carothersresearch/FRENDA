"""Kinetic mechanism definitions for ODBM.

Each class represents one rate-law template. New mechanisms should subclass
Mechanism and override writeRate(). The MECHANISMS list at the bottom is the
registry used by ModelBuilder.
"""

import re

import numpy as np
import pandas as pd
from .utils import extractParams, fmt


def overrides(f):
    return f


class InputError(Exception):
    pass


class Mechanism:
    """Base class for all kinetic mechanisms.

    Subclasses must set class-level attributes and override writeRate().
    """

    name = 'base_mechanism'
    required_params = []
    nS = np.nan
    nC = np.nan
    nP = np.nan
    nE = np.nan

    def __init__(self, rxn: pd.Series):
        # Determine enzyme label (endogenous vs heterologous)
        accession = rxn.get('Accession Number') if hasattr(rxn, 'get') else None
        prefix = 'hEC' if accession == 'Heterologous' else 'eEC'
        self.enzyme = prefix + str(rxn['EC']).replace('.', '')

        self.substrates = rxn['Substrates']
        self.products = rxn['Products']
        self.inhibitors = rxn.get('Inhibitors', np.nan)

        # Normalise inhibitor string
        try:
            self.inhibitors = ';'.join([
                I for I in np.unique(str(self.inhibitors).split(';'))
                if np.all([c not in I for c in ['D', 'G']])
            ])
        except Exception:
            self.inhibitors = 'nan'

        # Combine Km and Kcat into a single param string, skipping NaN halves
        km_str = str(rxn['Km']) if pd.notna(rxn.get('Km')) else ''
        kcat_str = str(rxn['Kcat']) if pd.notna(rxn.get('Kcat')) else ''
        parts = [p for p in [km_str, kcat_str] if p]
        self.params = '; '.join(parts) if parts else 'placeholder: 0'

        self.Ki = rxn.get('KI', np.nan)
        try:
            self.Ki = ';'.join([
                I for I in np.unique(str(self.Ki).split(';'))
                if np.all([c not in I for c in ['D', 'G']])
            ])
        except Exception:
            self.Ki = np.nan

        self.label = rxn['Label']
        self.EC = str(rxn['EC'])

        self._processInput()
        self._formatInput()

    def _processInput(self):
        self.params = extractParams(self.params)

        if self.required_params:
            self.relevent_params = sum(
                [[P for P in self.params if re.match(p, P)] for p in self.required_params], []
            )
            if not np.all([
                np.any([re.match(p, P) for P in self.params]) for p in self.required_params
            ]):
                raise InputError(
                    'No ' + ' or '.join(self.required_params) +
                    ' found in parameters for reaction ' + self.label
                )

        # enzyme
        if str(self.enzyme) != 'nan':
            self.enzyme = self.enzyme.split(';')
        else:
            self.enzyme = []
        if not np.isnan(self.nE) and len(self.enzyme) != self.nE:
            raise InputError(
                f'{len(self.enzyme)} enzyme(s) found for a {self.nE}-enzyme mechanism in reaction {self.label}'
            )

        # substrates
        if str(self.substrates) != 'nan':
            self.substrate_stoichiometries = [x.split(' ')[0] for x in self.substrates.split('; ')]
            self.substrates = [x.split(' ')[1] for x in self.substrates.split('; ')]
        else:
            self.substrates, self.substrate_stoichiometries = [], []
        if not np.isnan(self.nS) and len(self.substrates) != self.nS:
            raise InputError(
                f'{len(self.substrates)} substrate(s) found for a {self.nS}-substrate mechanism in reaction {self.label}'
            )

        # products
        if str(self.products) != 'nan':
            self.product_stoichiometries = [x.split(' ')[0] for x in self.products.split('; ')]
            self.products = [x.split(' ')[1] for x in self.products.split('; ')]
        else:
            self.products, self.product_stoichiometries = [], []
        if not np.isnan(self.nP) and len(self.products) != self.nP:
            raise InputError(
                f'{len(self.products)} product(s) found for a {self.nP}-product mechanism in reaction {self.label}'
            )

        # inhibitors
        if str(self.inhibitors) != 'nan':
            self.inhibitors = [i.strip() for i in self.inhibitors.split(';')]
        else:
            self.inhibitors = []

    def _formatInput(self):
        self.products = list(map(fmt, self.products))
        self.substrates = list(map(fmt, self.substrates))
        self.product_stoichiometries = list(map(fmt, self.product_stoichiometries))
        self.substrate_stoichiometries = list(map(fmt, self.substrate_stoichiometries))
        if self.inhibitors and not pd.isnull(self.Ki):
            self.inhibitors = list(map(fmt, self.inhibitors))

    def writeEquation(self) -> str:
        allS = ' + '.join(f'{n} {S}' for n, S in zip(self.substrate_stoichiometries, self.substrates))
        allE = ' + '.join(self.enzyme)
        allP = ' + '.join(f'{n} {P}' for n, P in zip(self.product_stoichiometries, self.products))

        if self.enzyme:
            rxn_str = allS + ' + ' + allE + ' -> ' + allE + ' + ' + allP
        else:
            rxn_str = allS + ' -> ' + allP

        return self.label + ' : ' + rxn_str

    def writeRate(self) -> str:
        pass


# ---------------------------------------------------------------------------
# Concrete mechanisms
# ---------------------------------------------------------------------------

class MichaelisMenten(Mechanism):
    name = 'MM'
    required_params = ['kcat', 'Km']
    nS = 1
    nP = np.nan
    nE = 1

    @overrides
    def writeRate(self) -> str:
        S = self.substrates
        E = self.enzyme[0]
        kcat, Km = [p + '_' + self.label for p in self.relevent_params]
        return self.label + ' = ' + kcat + '*' + E + '*' + S[0] + '/(' + Km + ' + ' + S[0] + ')'


class ModularRateLaw(Mechanism):
    """Common Modular Rate Law (Liebermeister & Klipp 2006)."""

    name = 'MRL'
    required_params = None
    nS = np.nan
    nP = np.nan
    nI = np.nan
    nE = 1
    ignore = ['C00001', 'C00080']  # H2O, H+

    def _active(self, species, stoichiometries=None):
        if stoichiometries is not None:
            return (
                [s for s, c in zip(species, stoichiometries) if s not in self.ignore],
                [c for s, c in zip(species, stoichiometries) if s not in self.ignore],
            )
        return [s for s in species if s not in self.ignore]

    def haldane_kcats(self):
        subs, n_subs = self._active(self.substrates, self.substrate_stoichiometries)
        prods, n_prods = self._active(self.products, self.product_stoichiometries)
        E = self.enzyme[0]
        allKmS = '*'.join(f'(Km_{s}_{E}^{n})' for s, n in zip(subs, n_subs))
        allKmP = '*'.join(f'(Km_{p}_{E}^{n})' for p, n in zip(prods, n_prods))
        Keq = 'Keq_' + self.label
        kcat_F = f'Kcat_V_{self.label}*({Keq}*({allKmS})/({allKmP}))^(0.5)'
        kcat_R = f'Kcat_V_{self.label}*({Keq}*({allKmS})/({allKmP}))^(-0.5)'
        return kcat_F, kcat_R

    def numerators(self):
        subs, n_subs = self._active(self.substrates, self.substrate_stoichiometries)
        prods, n_prods = self._active(self.products, self.product_stoichiometries)
        E = self.enzyme[0]
        allS = '*'.join(f'({s}^{n})' for s, n in zip(subs, n_subs))
        allP = '*'.join(f'({p}^{n})' for p, n in zip(prods, n_prods))
        allKmS = '*'.join(f'(Km_{s}_{E}^{n})' for s, n in zip(subs, n_subs))
        allKmP = '*'.join(f'(Km_{p}_{E}^{n})' for p, n in zip(prods, n_prods))
        kcatF = 'hKcat_F_' + self.label
        kcatR = 'hKcat_R_' + self.label
        return f'({kcatF}*({allS})/({allKmS}))', f'({kcatR}*({allP})/({allKmP}))'

    def denominator(self):
        subs, n_subs = self._active(self.substrates, self.substrate_stoichiometries)
        prods, n_prods = self._active(self.products, self.product_stoichiometries)
        E = self.enzyme[0]
        allKmS = [f'Km_{s}_{E}' for s in subs]
        allKmP = [f'Km_{p}_{E}' for p in prods]
        allS = '*'.join(f'((1+{s}/{Km})^{n})' for s, Km, n in zip(subs, allKmS, n_subs))
        allP = '*'.join(f'((1+{p}/{Km})^{n})' for p, Km, n in zip(prods, allKmP, n_prods))
        return f'({allS} + {allP} -1)'

    def inhibition_nc(self):
        inhibitors = self._active(self.inhibitors)
        E = self.enzyme[0]
        allKi = [f'Ki_{i}_{E}' for i in inhibitors]
        allGnc = [f'Gnc_{i}_{E}' for i in inhibitors]
        return '*'.join(
            f'({Gnc}+(1-{Gnc})*(1/(1+{i}/{Ki})))'
            for i, Ki, Gnc in zip(inhibitors, allKi, allGnc)
        )

    def inhibition_c(self):
        inhibitors = self._active(self.inhibitors)
        E = self.enzyme[0]
        allKi = [f'Ki_{i}_{E}' for i in inhibitors]
        allGc = [f'Gc_{i}_{E}' for i in inhibitors]
        return '+'.join(
            f'(1-{Gc})*({i}/{Ki})'
            for i, Ki, Gc in zip(inhibitors, allKi, allGc)
        )

    @overrides
    def writeRate(self) -> str:
        kcat_F, kcat_R = self.haldane_kcats()
        hkcat_F = f'hKcat_F_{self.label} := {kcat_F}'
        hkcat_R = f'hKcat_R_{self.label} := {kcat_R}'
        u = self.enzyme[0]
        Tf, Tr = self.numerators()
        D = f'{self.label}_D := {self.denominator()}'

        if self.inhibitors:
            fr = f'{u}_fr := {self.inhibition_nc()}'
            Dreg = f'{u}_Dreg := {self.inhibition_c()}'
            rate_f = f'{self.label}_f := {u} * {u}_fr * {Tf}/({self.label}_D + {u}_Dreg)'
            rate_r = f'{self.label}_r := {u} * {u}_fr * {Tr}/({self.label}_D + {u}_Dreg)'
            rate_net = f'{self.label} = {self.label}_f - {self.label}_r'
            return '; \n'.join([fr, Dreg, D, hkcat_F, hkcat_R, rate_f, rate_r, rate_net])
        else:
            rate_f = f'{self.label}_f := {u} * {Tf}/{self.label}_D'
            rate_r = f'{self.label}_r := {u} * {Tr}/{self.label}_D'
            rate_net = f'{self.label} = {self.label}_f - {self.label}_r'
            return '; \n'.join([D, hkcat_F, hkcat_R, rate_f, rate_r, rate_net])


class EnzymeExponentialDecay(Mechanism):
    """First-order enzyme degradation."""

    name = 'EED'
    required_params = None
    nS = np.nan
    nP = np.nan
    nE = np.nan

    @overrides
    def writeEquation(self) -> str:
        return f'D_{self.enzyme[0]} : {self.enzyme[0]} -> '

    @overrides
    def writeRate(self) -> str:
        return f'D_{self.enzyme[0]} = kdeg_{self.enzyme[0]}*{self.enzyme[0]}'


class OrderedBisubstrateBiproduct(Mechanism):
    name = 'OBB'
    required_params = ['kcat', 'Km1', 'Km2', 'K']
    nS = 2
    nP = 2
    nE = 1

    @overrides
    def writeRate(self) -> str:
        S = self.substrates
        E = self.enzyme[0]
        kcat, Km1, Km2, K = [p + '_' + self.label for p in self.relevent_params]
        return (
            f'{self.label} = {kcat}*{E}*{S[0]}*{S[1]}'
            f'/({S[0]}*{S[1]}+{Km1}*{S[1]}+{Km2}*{S[0]}+{K})'
        )


class SimplifiedOBB(Mechanism):
    name = 'SOBB'
    required_params = ['kcat', 'Km1', 'Km2']
    nS = 2
    nE = 1

    @overrides
    def writeRate(self) -> str:
        S = self.substrates
        E = self.enzyme[0]
        kcat, Km1, Km2 = [p + '_' + self.label for p in self.relevent_params]
        return (
            f'{self.label} = {kcat}*{E}*{S[0]}*{S[1]}'
            f'/({S[0]}*{S[1]}+{Km1}*{S[1]}+{Km2}*{S[0]}+{Km1}*{Km2})'
        )


class MassAction(Mechanism):
    name = 'MA'
    required_params = ['k']
    nS = np.nan
    nP = np.nan
    nE = np.nan

    @overrides
    def writeRate(self) -> str:
        S = self.substrates
        power = self.substrate_stoichiometries
        k, = [p + '_' + self.label if '$' not in p else p.replace('$', '_') for p in self.relevent_params]
        allS = '*'.join(s + '^' + c if c else s for s, c in zip(S, power))
        allE = '*'.join(self.enzyme)
        if allE:
            allE = '*' + allE
        if allS:
            allS = '*' + allS + allE
        return f'{self.label} = {k}{allS}'


class MonoMassAction(Mechanism):
    name = 'MMA'
    required_params = ['k']
    nS = np.nan
    nP = np.nan
    nE = np.nan

    @overrides
    def writeRate(self) -> str:
        allS = '*'.join(self.substrates)
        allE = '*'.join(self.enzyme)
        if allE:
            allE = '*' + allE
        k, = [p + '_' + self.label if '$' not in p else p.replace('$', '_') for p in self.relevent_params]
        return f'{self.label} = {k}*{allS}{allE}'


class ConstantRate(Mechanism):
    name = 'CR'
    required_params = ['k']

    @overrides
    def writeRate(self) -> str:
        return f'{self.label} = k_{self.label}'


MECHANISMS = [
    MichaelisMenten,
    ModularRateLaw,
    EnzymeExponentialDecay,
    OrderedBisubstrateBiproduct,
    SimplifiedOBB,
    MassAction,
    MonoMassAction,
    ConstantRate,
]
