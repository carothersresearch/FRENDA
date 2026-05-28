from symbolicSBML import SBMLModel, Parameters
from parameter_balancer import Balancer, ParameterData
from src.thermo_calculations.equilibrium_constants import keq_from_ec
import numpy as np
import tellurium as te

class RRBalancer():
    # Priors are linear scale here and will be converted inside.
    default_priors = {
        # Base quantities
        Parameters.mu: (-880.0, 680.00),
        Parameters.kv: (10.0, 6.26),
        Parameters.km: (0.1, 6.26),
        Parameters.c: (0.1, 10.32), # we might want to change this
        Parameters.u: (0.0001, 10.32), # we might want to change this
        Parameters.ki: (0.1, 6.26),
        Parameters.ka: (0.1, 6.26),
        # Derived quantities
        Parameters.keq: (1.0, 10.32),
        Parameters.kcat_prod: (10.0, 10.32),
        Parameters.kcat_sub: (10.0, 10.32),
        Parameters.vmax: (0.001, 17.01),
        Parameters.A: (0.0, 10.00),
        Parameters.mu_p: (-880.0, 680.00),
    }

    default_data = ParameterData([], [], [], [], [])

    def __init__(self, ModelBuilder=None, rr=None):
        """
        Parameter balancing object capable of extracting and updating paramteres from RoadRunner models.
        If provided a ModelBuilder, also capable of fetching equilibrium constants, and species concentrations (TODO).

        Parameters
        ----------
        ModelBuilder : ModelBuilder, optional
            ModelBuilder containing all model information, by default None
        rr : RoadRunner, optional
            roadrunner model, by default None
        """
        if ModelBuilder and not rr:
            self.model = ModelBuilder
            self.rr = te.loada(self.model.compile())
        if rr and not ModelBuilder:
            self.rr = rr
        if ModelBuilder and rr:
            self.model =  ModelBuilder
            self.rr = rr

        self._get_metabolites()
        self._get_reactions()
        self._get_stoichiometry()
        self.data = {}
        self._parse_parameters()
        self.structure = SBMLModel.from_structure(self.metabolites, self.reactions, self.stoichiometry)
        self.balanced_parameters = None

    def balance(self, priors = default_priors, data = default_data, T = 300., R = 8.314 / 1000.):
        """
        Balance the input data, incorporating the model structure and the priors, using Balancer from parameter_balancer

        Parameters
        ----------
        priors : dict, optional
            Priors for kinetic and thermodynamic parameters, by default uses default_priors based on median of available data
        data : list or ParameterData, optional
            Data for kinetic and thermodynamic parameters, by default uses no data
        T : float, optional
            temperature in Kelvin, by default 300
        R : float, optional
            gas constant in J/mmol/K, by default 8.314/1000.

        Returns
        -------
        DataFrame
            Pandas dataframe containing balanced parameters 
        """
        if type(data) is not type(ParameterData): data = ParameterData(*zip(*data))
        self.balanced_parameters = Balancer(priors, data, self.structure, T=T, R=R, augment=True).balance(sparse=False).to_frame(mean=True)
        return self.balanced_parameters
    
    def _get_metabolites(self):
        self.metabolites = self.rr.model.getFloatingSpeciesIds()
        return

    def _get_reactions(self):
        self.reactions = self.rr.model.getReactionIds()
        return

    def _get_stoichiometry(self):
        self.stoichiometry = np.array(self.rr.getFullStoichiometryMatrix())
        return

    def _parse_parameters(self):
        """
        Get all paramters from RoadRunner and prepare them for parameter balancing.
        Currently best integrated with ModularRateLaws with Km(s) and kcatF/R. 
        Curently defaults to the same std for all parameters...
        """
        p_ids = self.rr.model.getGlobalParameterIds() # Kms, kcats, and other
        p_values = self.rr.model.getGlobalParameterValues()

        params = {}
        for p,v in zip(p_ids, p_values):
            plist = p.split('_')
            if 'km' in p.lower():
                params[p] = (Parameters.km, plist[1], plist[-1], v, 6.26) # could control std based on BRENDA or estimated...
            elif 'kcatf' in p.lower():
                params[p] = (Parameters.kcat_sub, None, plist[-1], v, 6.26)
            elif 'kcatr' in p.lower():
                params[p] = (Parameters.kcat_prod, None, plist[-1], v, 6.26)
            else:
                print(p)
        self.data.update(params)
        return

    def update_parameters(self, update_with = 'median'):
        """
        Update the parameters in RoadRunner with the balanced parameter values

        Parameters
        ----------
        update_with : str, optional
            Which values to get from the posterior distirbution, by default 'median'. Could also sample form the distribution (TODO)

        Raises
        ------
        Exception
            in case parameters have not been balanced
        """
        if self.balanced_parameters is None:
            raise Exception('Must balance parameters first')
        else:
            p_ids = self.rr.model.getGlobalParameterIds()
            p_values = self.rr.model.getGlobalParameterValues()
            p_dict = {key:tuple([x if x is not None else '' for x in parameter[:3]]) for key, parameter in self.data.items()}
            bp_dict = self.balanced_parameters.to_dict()
            new_ps = [bp_dict[p_dict[p]][update_with] if p in p_dict.keys() else p_values[i] for i,p in enumerate(p_ids)]
            self.rr.model.setGlobalParameterValues(new_ps)
      
    def add_thermodynamics(self, keqs=None):
        if keqs:
            #add them to the Parameters Dataframe
            pass
        else:
            keqs = {}
            for i, row in self.model.rxns.iterrows():
                keq = keq_from_ec(row['EC'])
                if len(keq) == 1:
                    keqs['Keq_'+row['Label']] = (Parameters.keq, None, row['Label'], keq[0]['value'], keq[0]['error'])
        self.data.update(keqs)
        return


    # TODO: could get data for nominal concentrations of metabolites and enzymes..
    def add_metabolomics(self, c=None):
        if c:
            pass
        else:
            pass
        """
        import requests
        import json

        # something like this to get physiologically relevant metabolite concentrations 

        url = 'https://api.datanator.info/metabolites/concentration_only/?inchi_key='
        inchi_key = 'VZCYOOQTPOCHFL-OWOJBTEDSA-N'

        try:
            response = requests.get(url+inchi_key)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Parse the JSON data from the response
                data = json.loads(response.text)

                print(data[0]['concentrations'])


            else:
                print(f"Failed to retrieve data. Status code: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
        """
        
    def add_proteomics(self, u=None):
        """
        https://api.datanator.info/#/Proteins/get_proteins_precise_abundance_
        """
        if u:
            pass
        else:
            pass


