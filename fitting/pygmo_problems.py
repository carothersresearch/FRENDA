import os
import tellurium as te
import numpy as np
import pygmo as pg
from copy import deepcopy

class TimeoutError(Exception):
    pass

import signal
import time
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

class SBMLGlobalFit:

    def __init__(self, model, data, parameter_labels, lower_bounds, upper_bounds, settings, variables = {}, scale = False):
        self.model = model
        self.data = data
        self.parameter_labels = parameter_labels # only the ones that are going to be fitted
        self.settings = settings
        self.upperb = upper_bounds
        self.lowerb = lower_bounds
        self.scale = scale
        self.variables = variables # dict of labels and values

    def fitness(self, x):
        if self.scale: x = self._unscale(x)
        r, results = self._simulate(x)
        obj = self._residual(results,self.data)
        return [obj]
            
    def _simulate(self, x):
        from roadrunner import Config, RoadRunner
        Config.setValue(Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES, False)
        Config.setValue(Config.LOADSBMLOPTIONS_RECOMPILE, False) 
        Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, 4)

        id = str(os.getpid())
        try:
            r = RoadRunner()
            r.loadState('/mmfs1/gscratch/cheme/dalba/repos/ECFERS/models/binaries/model_state_'+id+'.b')
        except Exception as e:
            print(e)
            r = te.loadSBMLModel(self.model)
            r.saveState('/mmfs1/gscratch/cheme/dalba/repos/ECFERS/models/binaries/model_state_'+id+'.b')

        # update parameters
        for label, value in zip(self.parameter_labels,x):
            try:
                r[label] = value
            except Exception as e:
                print(e)
                return r, self.data*(-np.inf)

        # set any variable
        for label, value in self.variables.items():
            try:
                r[label] = value
            except Exception as e:
                print(e)
                return r, self.data*(-np.inf)
        try:
            results = r.simulate(**self.settings['simulation'])[:,1:].__array__()
        except:
            results = self.data*(-np.inf)

        return r, results
    
    # def _residual(self,results,data,points):
    #     md = (np.nanmax(data,1,keepdims=True)-np.nanmin(data,1,keepdims=True))/2
    #     mr = (np.nanmax(results,1,keepdims=True)-np.nanmin(results,1,keepdims=True))/2
    #     denom = np.ones(data.shape)
    #     return [np.nansum(((data-results)/(points*(md**2+mr**2)**0.5*denom))**2)]

    def _residual(self,results,data):
        cols = self.settings['fit_to_cols']
        rows = self.settings['fit_to_rows']

        error = (data[:,cols][rows,:]-results[:,cols][rows,:])
        RMSE = np.sqrt(np.nansum(error**2, axis=0)/len(rows))
        NRMSE = RMSE/(np.nanmax(data[:,cols][rows,:], axis=0) - np.nanmin(data[:,cols][rows,:], axis=0) + 1e6)
        return np.nansum(NRMSE)
    
    def _unscale(self, x):
        unscaled = self.lowerb + (self.upperb - self.lowerb) * x
        unscaled[unscaled<0] = self.lowerb[unscaled<0]
        unscaled[unscaled>1] = self.lowerb[unscaled>1]
        return unscaled
    
    def _scale(self, x):
        return (x - self.lowerb) / (self.upperb - self.lowerb)

    def get_bounds(self):
        if self.scale:
            lowerb = [0 for i in self.lowerb]
            upperb = [1 for i in self.upperb]
        else:
            upperb = self.upperb
            lowerb = self.lowerb    
        return (lowerb, upperb)
    
    def get_name(self):
        return 'Global Fitting of Multiple SBML Models'

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x),x)


class SBMLGlobalFit_Constrained:

    def __init__(self, model, data, parameter_labels, lower_bounds, upper_bounds, settings, logKeq_r, variables = {}, scale = False, n_constraints = 0):
        self.model = model
        self.data = data
        self.parameter_labels = parameter_labels # only the ones that are going to be fitted
        self.settings = settings
        self.upperb = upper_bounds
        self.lowerb = lower_bounds
        self.scale = scale
        self.n_constraints = n_constraints
        self.variables = variables # dict of labels and values
        self.logKeq_r = logKeq_r

    def batch_fitness(self, xs):
        view = pg.ipyparallel_bfe().init_view(client_kwargs={'profile':'cheme-ecfers'})
        fs = view.map_sync(self.fitness,xs)
        del view
        return fs

    def fitness(self, x):
        if self.scale: x = self._unscale(x)
        r, results = self._simulate(x)
        obj = self._residual(results,self.data)
        if self.n_constraints > 0:
            constraints = self._haldane(r)
            return [obj, *constraints]
        else:
            constraints = self._haldane(r)
            return [obj + constraints/r.getNumReactions()/10e6]
            
    def _simulate(self, x):
        from roadrunner import Config, RoadRunner
        Config.setValue(Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES, False)
        Config.setValue(Config.LOADSBMLOPTIONS_RECOMPILE, False) 
        Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, 4)

        id = str(os.getpid())
        try:
            r = RoadRunner()
            r.loadState('/mmfs1/gscratch/cheme/dalba/repos/ECFERS/models/binaries/model_state_'+id+'.b')
        except Exception as e:
            print(e)
            r = te.loadSBMLModel(self.model)
            r.saveState('/mmfs1/gscratch/cheme/dalba/repos/ECFERS/models/binaries/model_state_'+id+'.b')

        # r = te.loadSBMLModel(self.model)
        # update parameters
        for label, value in zip(self.parameter_labels,x):
            try:
                r[label] = value
            except Exception as e:
                print(e)
                return r, self.data*(-np.inf)

        # set any variable
        for label, value in self.variables.items():
            try:
                r[label] = value
            except Exception as e:
                print(e)
                return r, self.data*(-np.inf)
        try:
            results = r.simulate(**self.settings['simulation'])[:,1:].__array__()
        except:
            results = self.data*(-np.inf)

        return r, results
    
    # def _residual(self,results,data,points):
    #     md = (np.nanmax(data,1,keepdims=True)-np.nanmin(data,1,keepdims=True))/2
    #     mr = (np.nanmax(results,1,keepdims=True)-np.nanmin(results,1,keepdims=True))/2
    #     denom = np.ones(data.shape)
    #     return [np.nansum(((data-results)/(points*(md**2+mr**2)**0.5*denom))**2)]

    def _residual(self,results,data):
        cols = self.settings['fit_to_cols']
        rows = self.settings['fit_to_rows']

        error = (data[:,cols][rows,:]-results[:,cols][rows,:])
        RMSE = np.sqrt(np.nansum(error**2, axis=0)/len(rows))
        NRMSE = RMSE/(np.nanmax(data[:,cols][rows,:], axis=0) - np.nanmin(data[:,cols][rows,:], axis=0) + 1e6)
        return np.nansum(NRMSE)
    
    def _unscale(self, x):
        unscaled = self.lowerb + (self.upperb - self.lowerb) * x
        unscaled[unscaled<0] = self.lowerb[unscaled<0]
        unscaled[unscaled>1] = self.lowerb[unscaled>1]
        return unscaled
    
    def _scale(self, x):
        return (x - self.lowerb) / (self.upperb - self.lowerb)

    def _haldane(self, r):
        matrix = r.getFullStoichiometryMatrix();
        species_labels = np.array(matrix.rownames)
        metabolites_index = ['EC' not in label for label in species_labels]
        metabolites_labels = species_labels[metabolites_index]
        reaction_labels = np.array(matrix.colnames)
        stoich = matrix[metabolites_index,:]

        kms = np.zeros(shape=stoich.shape)
        for i in range(stoich.shape[0]):
            for j in range(stoich.shape[1]):
                m_id = metabolites_labels[i]
                r_id = reaction_labels[j]
                if stoich[i,j] != 0:
                    kms[i,j] = stoich[i,j]*np.log(r['Km_'+m_id+'_'+r_id])
        
        nlogKm_r = kms.sum(axis=0)
        logKcats_r = np.array([np.log(r['Kcat_F_'+r_id]/r['Kcat_R_'+r_id]) for r_id in r.getReactionIds()])
        logKeq_r = self.logKeq_r # may be a better way to pass this
        return np.sum(((logKeq_r-logKcats_r-nlogKm_r)/logKeq_r)**2)

    def get_bounds(self):
        if self.scale:
            lowerb = [0 for i in self.lowerb]
            upperb = [1 for i in self.upperb]
        else:
            upperb = self.upperb
            lowerb = self.lowerb    
        return (lowerb, upperb)
    
    def get_name(self):
        return 'Global Fitting of Multiple SBML Models'

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x),x)
    
    def get_nec(self):
        return self.n_constraints
    

class SBMLGlobalFit_Multi:

    def __init__(self, model, data, parameter_labels, lower_bounds, upper_bounds, metadata, variables = {}, scale = False):
        self.model = model
        self.path_to_models = model[:model.find('/models')]
        self.parameter_labels = parameter_labels # only the ones that are going to be fitted
        self.upperb = upper_bounds
        self.lowerb = lower_bounds
        self.scale = scale
        self.data = data
        self.metadata = metadata
        self.variables = variables # dict of labels and values
        self.cvode_timepoints = 1000

        r = te.loadSBMLModel(self.model)
        self.species_labels = np.array(r.getFullStoichiometryMatrix().rownames)
        self.global_parameter_labels = np.array(r.getGlobalParameterIds())
        self.cols = {}
        self.rows = {}
        self.data_cols = {}
        for s in self.metadata['sample_labels']:
            measurements = []
            self.data_cols[s] = []
            for i,m in enumerate(self.metadata['measurement_labels']):
                try:    # model may not contain measurement species
                    measurements.append(np.where(self.species_labels==m)[0][0])
                    self.data_cols[s].append(i)
                except:
                    pass
            self.cols[s] = measurements
            self.rows[s] = [np.where(np.linspace(0, self.metadata['timepoints'][s][-1], self.cvode_timepoints) < (t+0.1))[0][-1] for t in self.metadata['timepoints'][s]]


    def fitness(self, x):
        if self.scale: x = self._unscale(x)
        res_dict = self._simulate(x)
        obj = np.nanmean([self._residual(results, self.data[sample], sample) for sample, results in res_dict.items()])
        return [obj]
            
    def _simulate(self, x):
        from roadrunner import Config, RoadRunner
        Config.setValue(Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES, False)
        Config.setValue(Config.LOADSBMLOPTIONS_RECOMPILE, False) 
        Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, 4)

        # load model
        id = str(os.getpid())
        if os.path.isfile(self.path_to_models+'/models/binaries/model_state_'+id+'.b'):
            with open(self.path_to_models+'/models/binaries/model_state_'+id+'.b', 'rb') as f:
                r = RoadRunner()
                r.loadStateS(f.read())
        else:
            r = te.loadSBMLModel(self.model)
            r.integrator.absolute_tolerance = 1e-8
            r.integrator.relative_tolerance = 1e-8
            r.integrator.maximum_num_steps = 2000
            with open(self.path_to_models+'/models/binaries/model_state_'+id+'.b', 'wb') as f:
                f.write(r.saveStateS())
       
        # set new parameters
        for l, v in zip(self.parameter_labels,x):
            if 'Km' in l:
                r.setValue('init('+l+')',v)
            else:
                r.setValue('init('+l+')',v)

        # set variables and simulate
        results = {sample:self.data[sample]*(-np.inf) for sample in self.metadata['sample_labels']}
        rb = r.saveStateS()
        for sample in self.metadata['sample_labels']:
            r2 = RoadRunner()
            r2.loadStateS(rb)
            # set any variable
            for label, value in self.variables[sample].items():
                if not np.isnan(value):
                    if label not in self.species_labels:
                        r2.setValue('init('+label+')', value) # we might need new assignment rules for heterologous enzymes
                    else:
                        if 'EC' in label:
                            r2.setValue('['+label+']', value*r2.getValue(self.parameter_labels[-1])) # this is a bit obtuse
                        else:
                        # r2.removeInitialAssignment(label) theres some bug here? it seems to also mess up with over valuesS
                            r2.setValue('['+label+']', value)
            try:
                results[sample] = r2.simulate(0,self.metadata['timepoints'][sample][-1],self.cvode_timepoints)[:,1:].__array__()
            except Exception as e:
                print(e)
                # break # stop if any fail
        return results
                
    def _residual(self,results,data,sample):
        cols = self.cols[sample]
        rows = self.rows[sample]
        dcols = self.data_cols[sample]
        
        if data.shape == results.shape:
            error = (data[:,dcols]-results[:,dcols])
        else:
            error = (data[:,dcols]-results[:,cols][rows,:])

        RMSE = np.sqrt(np.nansum(error**2, axis=0)/len(rows))
        NRMSE = RMSE/(np.nanmax(data[:,dcols], axis=0) - np.nanmin(data[:,dcols], axis=0) + 1e-6)
        return np.nansum(NRMSE)
    
    def _unscale(self, x):
        unscaled = self.lowerb + (self.upperb - self.lowerb) * x
        # unscaled[unscaled<0] = self.lowerb[unscaled<0]
        # unscaled[unscaled>1] = self.lowerb[unscaled>1]
        return unscaled
    
    def _scale(self, x):
        return (x - self.lowerb) / (self.upperb - self.lowerb)

    def get_bounds(self):
        if self.scale:
            lowerb = [0 for i in self.lowerb]
            upperb = [1 for i in self.upperb]
        else:
            upperb = self.upperb
            lowerb = self.lowerb    
        return (lowerb, upperb)
    
    def get_name(self):
        return 'Global Fitting of Multiple SBML Models'

class SBMLGlobalFit_Multi_Fly:
    class ModelStuff:
        def __init__(self, model, metadata, cvode_timepoints, parameter_labels, variables, extra_residuals):
            r = te.loadSBMLModel(model)
            self.r_parameter_labels = np.array(r.getGlobalParameterIds())
            self.species_labels = np.array(r.getFullStoichiometryMatrix().rownames)
            self.species_init = {k:v for k,v in zip(self.species_labels,r.model.getFloatingSpeciesInitConcentrations())}
            self.species_to_v = {k:v for k,v in zip([s for s in r.getFloatingSpeciesIds() if 'EC' not in s],[p for p in self.r_parameter_labels if 'v' in p])}
            self.parameter_order = np.int32(np.squeeze(np.array([np.where(p == self.r_parameter_labels) for p in parameter_labels if p in self.r_parameter_labels])))
            self.parameter_present = [p in self.r_parameter_labels for p in parameter_labels]
            self.variable_order = {sample:np.int32(np.squeeze(np.array([np.where(p == self.r_parameter_labels) for p in var.keys() if p in self.r_parameter_labels]))) for sample,var in variables.items()}
            self.variable_present = {sample:[p in self.r_parameter_labels for p in var.keys()] for sample,var in variables.items()}
            self.extra_residuals = extra_residuals
            self.cols = {}
            self.rows = {}
            self.data_cols = {}
            for s in metadata['sample_labels']:
                measurements = []
                self.data_cols[s] = []
                for i,m in enumerate(metadata['measurement_labels']):
                    try:    # model may not contain measurement species
                        measurements.append(np.where(self.species_labels==m)[0][0])
                        self.data_cols[s].append(i)
                    except:
                        pass
                self.cols[s] = measurements
                self.rows[s] = [np.where(np.linspace(0, metadata['timepoints'][s][-1], cvode_timepoints) < (t+0.1))[0][-1] for t in metadata['timepoints'][s]]
            del r

        def set_vars(self, new_vars_dict):
            self.variable_order = {sample:np.int32(np.squeeze(np.array([np.where(p == self.r_parameter_labels) for p in var.keys() if p in self.r_parameter_labels]))) for sample,var in new_vars_dict.items()}
            self.variable_present = {sample:[p in self.r_parameter_labels for p in var.keys()] for sample,var in new_vars_dict.items()}

    def __init__(self, model:list, data:list, data_weights:list, parameter_labels, lower_bounds, upper_bounds, metadata:list, variables:list, extra_residuals:list,
                        scale = False, log=None, elambda = 1, dlambda=1, llambda=1, lelambda = 0, ldlambda=0, lllambda=0, elambda2 = 1, dlambda2=1, llambda2=1, lelambda2 = 0, ldlambda2=0, lclambda2=0, lllambda2=0, rmse = 'sum', objf = 'multi', normalize_fitness=True, fitness_std = False):
        self.model = model # now a list of models
        self.parameter_labels = parameter_labels # all parameters across all models, only the ones that are going to be fitted
        self.set_bounds(upper_bounds, lower_bounds)
        self.scale = scale
        self.log = log
        self.data = data # now a list of data
        self.data_weights = data_weights
        self.metadata = metadata # now a list of metadas
        self.variables = variables # # now a list of dict of labels and values

        self.cvode_timepoints = 1000

        self.model_stuff = [self.ModelStuff(m, md, self.cvode_timepoints, self.parameter_labels, var, er) for m,md,var,er in zip(self.model, self.metadata, self.variables, extra_residuals)]
        self.dlambda = dlambda
        self.llambda = llambda
        self.elambda = elambda
        self.ldlambda = ldlambda
        self.lllambda = lllambda
        self.lelambda = lelambda
        self.dlambda2 = dlambda2
        self.llambda2 = llambda2
        self.elambda2 = elambda2
        self.ldlambda2 = ldlambda2
        self.lllambda2 = lllambda2
        self.lelambda2 = lelambda2
        self.lclambda2 = lclambda2
        self.rmse = rmse
        self.objf = objf


        self.current_parameters = None
        self.current_results = None
        self.current_fitness = None
        self.all_fitness = []
        self.nominal_fitness = None
        self.normalize_fitness = normalize_fitness
        self.fitness_std = fitness_std

    def fitness(self, x):
        if self.scale: x = self._unscale(x)
        if self.log: x = np.array([10**v if k else v for k,v in zip(self.log,x)])
        res = self._simulate(x)
        self.current_d_erorr=[]
        self.current_cumd_erorr=[]
        self.current_error=[]
        self.current_log_erorr=[]
        if 'multi' not in self.objf:
            if self.rmse == 'sum':
                self.current_fitness = np.nansum([list(np.nanmean([self._residual(results, data[sample], weights[sample], sample, ms, md) for sample, results in resdict.items()],0)) + [self._extra_residual(resdict,data,weights,ms)] for data,weights,ms,resdict,md in zip(self.data,self.data_weights,self.model_stuff,res,self.metadata)],0)
            if self.rmse == 'mean':
                self.current_fitness = np.nanmean([list(np.nanmean([self._residual(results, data[sample], weights[sample], sample, ms, md) for sample, results in resdict.items()],0))+ [self._extra_residual(resdict,data,weights,ms)] for data,weights,ms,resdict,md in zip(self.data,self.data_weights,self.model_stuff,res,self.metadata)],0)
            if self.rmse == 'sum_range':
                all_res = []
                for data,weights,ms,resdict,md in zip(self.data,self.data_weights,self.model_stuff,res,self.metadata):
                    er = self._extra_residual(resdict,data,weights,ms)
                    res = []
                    for sample, results in resdict.items():
                        res.append(self._residual(results, data[sample], weights[sample], sample, ms, md))
                    mean_res = np.nanmean(res,0)
                    range_res = np.nanmax(res,0)-np.nanmin(res,0)
                    all_res.append(list(mean_res+range_res)+[er])
                self.current_fitness = np.nansum(all_res,0)     
        elif self.objf == 'multi_exp':
            all_residuals = [[self._residual(results, data[sample], weights[sample], sample, ms, md) for sample, results in resdict.items()] for data,weights,ms,resdict,md in zip(self.data,self.data_weights,self.model_stuff,res,self.metadata)][0]
            extra_res = self._extra_residual(res[0],self.data[0],self.data_weights[0],self.model_stuff[0])/self.get_nobj()
            values, indices, counts = np.unique(self.metadata[0]['objective'], return_counts=True, return_index=True)
            subarrays=np.split(all_residuals,indices)
            self.current_fitness = [np.nanmean(subarray)+extra_res for subarray in subarrays[1:]]
        elif self.objf == 'multi_error':
            all_residuals = [[self._residual(results, data[sample], weights[sample], sample, ms, md) for sample, results in resdict.items()] for data,weights,ms,resdict,md in zip(self.data,self.data_weights,self.model_stuff,res,self.metadata)]
            extra_res = self._extra_residual(res[0],self.data[0],self.data_weights[0],self.model_stuff[0])
            self.current_fitness = [*list(np.nanmean(all_residuals, (0,1))), extra_res]
        elif self.objf == 'multi_range':
            all_residuals = [[self._residual(results, data[sample], weights[sample], sample, ms, md) for sample, results in resdict.items()] for data,weights,ms,resdict,md in zip(self.data,self.data_weights,self.model_stuff,res,self.metadata)]
            extra_res = self._extra_residual(res[0],self.data[0],self.data_weights[0],self.model_stuff[0])
            self.current_fitness = [*list(np.nanmean(all_residuals, (0,1))), *list(np.nanmax(all_residuals, (0,1))-np.nanmin(all_residuals, (0,1))), extra_res]
        elif self.objf == 'multi_comb':
            all_residuals = [[self._residual(results, data[sample], weights[sample], sample, ms, md) for sample, results in resdict.items()] for data,weights,ms,resdict,md in zip(self.data,self.data_weights,self.model_stuff,res,self.metadata)]
            extra_res = self._extra_residual(res[0],self.data[0],self.data_weights[0],self.model_stuff[0])
            means = np.nanmean(all_residuals, (0,1))
            self.current_fitness = [means[0], *list(means[1:] + means[0]), extra_res+means[0]]
        else:
            print('Objective function not recognized')
        del res

        if self.normalize_fitness:
            if not self.nominal_fitness:
                raise ValueError('Nominal fitness not set')
        else:
            if not self.nominal_fitness:
                self._set_nominal_fitness([1 for _ in self.current_fitness])
        
        if self.objf == 'multi_range':
            # n = (len(self.current_fitness)-1)//2 if self.model_stuff[0].extra_residuals else len(self.current_fitness)//2
            # means = np.array([f/n for f,n in zip(self.current_fitness[:n],self.nominal_fitness[:n])])
            # ranges = np.array([f/n for f,n in zip(self.current_fitness[n:],self.nominal_fitness[n:])])
            normalized_fitness = self.current_fitness
            print('Normalized fitness with multi_range not implemented')
        else:
            normalized_fitness = np.array([f/n for f,n in zip(self.current_fitness,self.nominal_fitness)])
        
        if self.fitness_std:
            normalized_fitness = np.append(normalized_fitness,np.std(normalized_fitness))
            
        self.all_fitness.append(normalized_fitness)

        if 'multi' not in self.objf:
            if self.rmse == 'sum':
                return [np.nansum(normalized_fitness)]
            if self.rmse == 'mean':
                return [np.nanmean(normalized_fitness)]
            if self.rmse == 'sum_range':
                return [np.nansum(normalized_fitness)]
        elif self.objf == 'multi_range':
            n = (len(self.current_fitness)-1)//2 if self.model_stuff[0].extra_residuals else len(self.current_fitness)//2
            f = [np.sum(self.current_fitness[:n]), np.sum(self.current_fitness[n:-1]), self.current_fitness[-1]] if self.model_stuff[0].extra_residuals else [np.sum(self.current_fitness[:n]), np.sum(self.current_fitness[n:])]
            return f
        else:
            return list(normalized_fitness)


    def _set_nominal_fitness(self, nominal_fitness=None):
            self.nominal_fitness = nominal_fitness
    
    def _setup_rr(self): # run on engine
        from roadrunner import Config, RoadRunner, Logger
        Logger.disableLogging()
        Config.setValue(Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES, True)
        Config.setValue(Config.LOADSBMLOPTIONS_RECOMPILE, False) 
        Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, 4)
        Config.setValue(Config.LLVM_SYMBOL_CACHE, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_GVN, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER, True)
        Config.setValue(Config.SIMULATEOPTIONS_COPY_RESULT, True)
        self.r = []
        for m in self.model:
            r = te.loadSBMLModel(m)
            r.integrator.absolute_tolerance = 1e-8
            r.integrator.relative_tolerance = 1e-8
            r.integrator.maximum_num_steps = 2000
            self.r.append(r)
            
    def _simulate(self, x):
        from roadrunner import Config, RoadRunner, Logger
        Logger.disableLogging()
        Config.setValue(Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES, True)
        Config.setValue(Config.LOADSBMLOPTIONS_RECOMPILE, False) 
        Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, 4)
        Config.setValue(Config.LLVM_SYMBOL_CACHE, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_GVN, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER, True)
        Config.setValue(Config.SIMULATEOPTIONS_COPY_RESULT, True)

        all_results = []
        for r,ms, metadata, variables in zip(self.r, self.model_stuff, self.metadata, self.variables):
            results = {sample:np.ones((self.cvode_timepoints,len(ms.species_labels)))*1e12 for sample in metadata['sample_labels']}
            for sample in metadata['sample_labels']:
                
                x_dict = {k:v for k,v in zip(self.parameter_labels,x)}
                for label, value in variables[sample].items():
                    if not np.isnan(value):
                        if label in ms.species_labels:
                            x_dict[ms.species_to_v[label]]=(value+x_dict[ms.species_to_v[label]]*ms.species_init[label]*variables[sample]['dilution_factor'])/(ms.species_init[label]*variables[sample]['dilution_factor'])

                x2 = np.array(list(x_dict.values()))
                r.model.setGlobalParameterValues([*ms.parameter_order, *ms.variable_order[sample]], [*x2[ms.parameter_present], *np.array(list(variables[sample].values()))[ms.variable_present[sample]]])
                r.reset()

                try:
                    results[sample] = r.simulate(0,metadata['timepoints'][sample][-1],self.cvode_timepoints)[:,1:].__array__()
                except Exception as e:
                    print(e)
                    # break # stop if any fail
                r.resetToOrigin()
            all_results.append(results)
        del Config, RoadRunner, Logger, results

        self.current_x = x
        self.current_x2 = x2
        self.current_results = all_results
        return self.current_results
                
    def _residual(self,results,data,weights,sample,modelstuff,metadata): # per sample
        cols = modelstuff.cols[sample]
        rows = modelstuff.rows[sample]
        dcols = modelstuff.data_cols[sample]
        
        # calculate error
        if data.shape == results.shape:
            error = (data[:,dcols]-results[:,dcols])
            d_error = (np.diff(data[:,dcols],axis=0)-np.diff(results[:,dcols],axis=0))/np.tile(np.diff(rows),(len(dcols),1)).T
            cumd_error = np.sum(np.abs(np.diff(data[:,dcols],axis=0)),axis=0)-np.sum(np.abs(np.diff(results[:,dcols],axis=0)),axis=0)
            log_error = np.log10(data[:,dcols]+1e-12)-np.log10(results[:,dcols]+1e-12)
        else:
            error = (data[:,dcols]-results[:,cols][rows,:])
            d_error = (np.diff(data[:,dcols],axis=0)-np.diff(results[:,cols][rows,:],axis=0))/np.tile(np.diff(rows),(len(dcols),1)).T
            cumd_error = np.sum(np.abs(np.diff(data[:,dcols],axis=0)),axis=0)-np.sum(np.abs(np.diff(results[:,cols][rows,:],axis=0)),axis=0)
            log_error = np.log10(data[:,dcols]+1e-12)-np.log10(results[:,cols][rows,:]+1e-12)

        self.current_d_erorr.append(d_error)
        self.current_cumd_erorr.append(cumd_error)
        self.current_error.append(error)
        self.current_log_erorr.append(log_error)
        
        # appply error function, then weights
        wsq_e = (error**2)*weights[:,dcols]
        wsq_de = (d_error**2)*((weights[:,dcols]*np.roll(weights[:,dcols],1,axis=0))[:-1,:])
        wsq_le = (log_error**2)*weights[:,dcols]

        wl_e = np.log1p(np.abs(error))*weights[:,dcols]
        wl_de = np.log1p(np.abs(d_error))*((weights[:,dcols]+np.roll(weights[:,dcols],1,axis=0))[:-1,:]/2)
        wl_cde = np.log1p(np.abs(cumd_error))*(((weights[:,dcols]+np.roll(weights[:,dcols],1,axis=0))[:-1,:]/2).mean(0))
        wl_le = np.log1p(np.abs(log_error))*weights[:,dcols]

        # # avereage acrross time and apply loss function 
        # wrmse_e = np.sqrt(np.nansum(wsq_e, axis=0)/(np.nansum((~np.isnan(wsq_e)),axis=0)+1e-12))
        # wrmse_de = np.sqrt(np.nansum(wsq_de, axis=0)/(np.nansum((~np.isnan(wsq_de)),axis=0)+1e-12))
        # wrmse_le = np.sqrt(np.nansum(wsq_le, axis=0)/(np.nansum((~np.isnan(wsq_le)),axis=0)+1e-12))

        # wll_e = np.log1p(np.nansum(wl_e, axis=0)/(np.nansum((~np.isnan(wl_e)),axis=0)+1e-12))
        # wll_de = np.log1p(np.nansum(wl_de, axis=0)/(np.nansum((~np.isnan(wl_de)),axis=0)+1e-12))
        # wll_le = np.log1p(np.nansum(wl_le, axis=0)/(np.nansum((~np.isnan(wl_le)),axis=0)+1e-12))

        # RMSE = np.nansum([self.elambda*np.nansum(wrmse_e)/np.count_nonzero(wrmse_e),
        #                 self.dlambda*np.nansum(wrmse_de)/np.count_nonzero(wrmse_de),
        #                 self.llambda*np.nansum(wrmse_le)/np.count_nonzero(wrmse_le),
        #                 self.lelambda*np.nansum(wll_e)/np.count_nonzero(wll_e),
        #                 self.ldlambda*np.nansum(wll_de)/np.count_nonzero(wll_de),
        #                 self.lllambda*np.nansum(wll_le)/np.count_nonzero(wll_le),
        #                 self.elambda2*wrmse_e2,
        #                 self.dlambda2*wrmse_de2,
        #                 self.llambda2*wrmse_le2,
        #                 self.lelambda2*wll_e2,
        #                 self.ldlambda2*wll_de2,
        #                 self.lllambda2*wll_le2])

        # or just sum it up
        wrmse_e2 = np.sqrt(np.nansum(wsq_e)/(np.nansum((~np.isnan(wsq_e)))+1e-12))
        wrmse_de2 = np.ma.masked_invalid(wsq_de).mean()/((1-np.ma.masked_invalid(wsq_de).mask).sum()+1e-12)
        wrmse_le2 = np.sqrt(np.nansum(wsq_le)/(np.nansum((~np.isnan(wsq_le)))+1e-12))

        wll_e2 = np.nansum(wl_e)#/(np.nansum((~np.isnan(wl_e)))+1e-12)
        wll_de2 = np.ma.masked_invalid(wl_de).sum()#/((1-np.ma.masked_invalid(wl_de).mask).sum()+1e-12)
        wll_cde2 = np.ma.masked_invalid(wl_cde).sum()#/((1-np.ma.masked_invalid(wl_cde).mask).sum()+1e-12)
        wll_le2 = np.nansum(wl_le)#/(np.nansum((~np.isnan(wl_le)))+1e-12)

        RMSE = [self.elambda2*wrmse_e2 + self.lelambda2*wll_e2,
                self.dlambda2*wrmse_de2 + self.ldlambda2*wll_de2,
                self.lclambda2*wll_cde2,
                self.llambda2*wrmse_le2 + self.lllambda2*wll_le2]
                 
                # np.nanmax(wsq_e), np.nanmax(wsq_de), np.nanmax(wsq_le)]
        
        return RMSE

    def _extra_residual(self,results:dict,data:dict,weights:dict,modelstuff):
        er = []
        if modelstuff.extra_residuals:
            for d in modelstuff.extra_residuals:
                sample1 = d[0]
                sample2 = d[1]
                metabolite = d[2]
                index1 = np.where(modelstuff.species_labels[modelstuff.cols[sample1]] == metabolite)[0][0]
                index2 = np.where(modelstuff.species_labels[modelstuff.cols[sample2]] == metabolite)[0][0]

                cols1 = modelstuff.cols[sample1]
                rows1 = modelstuff.rows[sample1]
                dcols1 = modelstuff.data_cols[sample1]

                cols2 = modelstuff.cols[sample2]
                rows2 = modelstuff.rows[sample2]
                dcols2 = modelstuff.data_cols[sample2]

                rows = list(np.sort(list(set(rows1) & set(rows2))))
                drows1 = [rows1.index(r) for r in rows]
                drows2 = [rows2.index(r) for r in rows]

                try:
                    error = (data[sample1][drows1,:][:,dcols1][:,index1]-data[sample2][drows2,:][:,dcols2][:,index2]) - (results[sample1][:,dcols1][rows,:][:,index1]-results[sample2][rows,:][:,dcols2][:,index2])
                except Exception as e:
                    error = (data[sample1][:,dcols1][:,index1]-data[sample2][:,dcols2][:,index2]) - (results[sample1][:,dcols1][:,index1]-results[sample2][:,dcols2][:,index2])
                
                weight = (weights[sample1][drows1,:][:,dcols1][:,index1]+weights[sample2][drows2,:][:,dcols2][:,index2])/2
                wsq_e = (error**2)*weight
                wrmse_e2 = np.sqrt(np.nansum(wsq_e)/(np.nansum((~np.isnan(wsq_e)))+1e-12))
                er.append(wrmse_e2)
        else:
            er.append(0)
        return np.nanmean(er)
    
    def _unscale(self, x):
        unscaled = self.lowerb + (self.upperb - self.lowerb) * x
        # unscaled[unscaled<0] = self.lowerb[unscaled<0]
        # unscaled[unscaled>1] = self.lowerb[unscaled>1]
        return unscaled
    
    def _scale(self, x):
        return (x - self.lowerb) / (self.upperb - self.lowerb)

    def get_bounds(self):
        if self.scale:
            lowerb = [0 for i in self.lowerb]
            upperb = [1 for i in self.upperb]
        else:
            upperb = self.upperb
            lowerb = self.lowerb    
        return (lowerb, upperb)
    
    def get_name(self):
        return 'Global Fitting of Multiple SBML Models'
    
    def get_nobj(self):
        if self.objf == 'multi_exp':
            if self.metadata[0]['objective'] is not None:
                return len(np.unique((self.metadata[0]['objective'])))
            return len(self.metadata[0]['sample_labels'])
        elif self.objf == 'multi_comb':
            return 5 # change this based on # error in _residuals
        elif self.objf == 'multi_error':
            return 5 # change this based on # error in _residuals
        elif self.objf == 'multi_range':
            return 3
        else:
            return 1
    
    def set_bounds(self, upper_bounds, lower_bounds):
        self.upperb = upper_bounds # for all parameters
        self.lowerb = lower_bounds # fol all parameters

    def set_var(self, list_new_vars_dict):
        for i,new_vars_dict in enumerate(list_new_vars_dict):
            self.variables[i] = new_vars_dict
            old_variables = self.variables[i]
            new_variables = {}
            for sample, var_dict in old_variables.items():
                for k,v in new_vars_dict.items():
                        var_dict[k] = v
                new_variables[sample] = var_dict

            self.variables[i] = new_variables
            self.model_stuff[i].set_vars(new_variables)

class SBML_Overproduction_Multi_Fly:
    class ModelStuff:
        def __init__(self, model, parameter_labels, variables):
            r = te.loadSBMLModel(model)
            self.species_labels = np.array(r.getFullStoichiometryMatrix().rownames)
            self.r_parameter_labels = np.array(r.getGlobalParameterIds())
            self.parameter_order = np.int32(np.squeeze(np.array([np.where(p == self.r_parameter_labels) for p in parameter_labels if p in self.r_parameter_labels])))
            self.parameter_present = [p in self.r_parameter_labels for p in parameter_labels]
            self.variable_order = {sample:np.int32(np.squeeze(np.array([np.where(p == self.r_parameter_labels) for p in var.keys() if p in self.r_parameter_labels]))) for sample,var in variables.items()}
            self.variable_present = {sample:[p in self.r_parameter_labels for p in var.keys()] for sample,var in variables.items()}
            del r

    def __init__(self, model:list, overproduction_function, parameter_labels, lower_bounds, upper_bounds, metadata:list, variables:list, scale = False, dlambda=1):
        self.model = model # now a list of models
        self.parameter_labels = parameter_labels # all parameters across all models, only the ones that are going to be fitted
        self.set_bounds(upper_bounds, lower_bounds)
        self.scale = scale
        self.overproduction_function = overproduction_function 
        self.metadata = metadata # now a list of metadas
        self.variables = variables # # now a list of dict of labels and values

        self.cvode_timepoints = 1000

        self.model_stuff = [self.ModelStuff(m, self.parameter_labels, var) for m,var in zip(self.model, self.variables)]

    def fitness(self, x):
        if self.scale: x = self._unscale(x)
        res = self._simulate(x)
        obj = np.nansum([np.nansum([self.overproduction_function(results) for _, results in resdict.items()]) for resdict in res])
        del res
        return [obj]
    
    def _setup_rr(self): # run on engine
        from roadrunner import Config, RoadRunner, Logger
        Logger.disableLogging()
        Config.setValue(Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES, True)
        Config.setValue(Config.LOADSBMLOPTIONS_RECOMPILE, False) 
        Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, 4)
        Config.setValue(Config.LLVM_SYMBOL_CACHE, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_GVN, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER, True)
        Config.setValue(Config.SIMULATEOPTIONS_COPY_RESULT, True)
        self.r = []
        for m in self.model:
            r = te.loadSBMLModel(m)
            r.integrator.absolute_tolerance = 1e-8
            r.integrator.relative_tolerance = 1e-8
            r.integrator.maximum_num_steps = 2000
            self.r.append(r)
            
    def _simulate(self, x):
        from roadrunner import Config, RoadRunner, Logger
        Logger.disableLogging()
        Config.setValue(Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES, True)
        Config.setValue(Config.LOADSBMLOPTIONS_RECOMPILE, False) 
        Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, 4)
        Config.setValue(Config.LLVM_SYMBOL_CACHE, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_GVN, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER, True)
        Config.setValue(Config.SIMULATEOPTIONS_COPY_RESULT, True)

        all_results = []
        for r,ms, metadata, variables in zip(self.r, self.model_stuff, self.metadata, self.variables):

            results = {sample:[0] for sample in metadata['sample_labels']}
            for sample in metadata['sample_labels']:
                r.model.setGlobalParameterValues([*ms.parameter_order, *ms.variable_order[sample]], [*x[ms.parameter_present], *np.array(list(variables[sample].values()))[ms.variable_present[sample]]])
                r.reset()

                # set init concentrations
                for label, value in variables[sample].items():
                    if not np.isnan(value):
                        if label in ms.species_labels:
                            if 'EC' not in label:
                                r.setValue('['+label+']', value)
                            else: 
                                r.setValue('['+label+']', value*variables[sample]['dilution_factor']*x[-1]) # this is a bit obtuse: [plasmid] * DF * rel1
                try:
                    results[sample] = r.simulate(0,metadata['timepoints'][sample][-1],self.cvode_timepoints)[:,1:].__array__()
                except Exception as e:
                    print(e)
                    # break # stop if any fail
                r.resetToOrigin()
            all_results.append(results)
        del Config, RoadRunner, Logger, results
        return all_results
    
    def _unscale(self, x):
        unscaled = self.lowerb + (self.upperb - self.lowerb) * x
        # unscaled[unscaled<0] = self.lowerb[unscaled<0]
        # unscaled[unscaled>1] = self.lowerb[unscaled>1]
        return unscaled
    
    def _scale(self, x):
        return (x - self.lowerb) / (self.upperb - self.lowerb)

    def get_bounds(self):
        if self.scale:
            lowerb = [0 for i in self.lowerb]
            upperb = [1 for i in self.upperb]
        else:
            upperb = self.upperb
            lowerb = self.lowerb    
        return (lowerb, upperb)
    
    def get_name(self):
        return 'Global Fitting of Multiple SBML Models'
    
    def set_bounds(self, upper_bounds, lower_bounds):
        self.upperb = upper_bounds # for all parameters
        self.lowerb = lower_bounds # fol all parameters


class SBMLGlobalFit_Multi_Fly_time(SBMLGlobalFit_Multi_Fly):
    def __init__(self, model:list, data:list, data_weights:list, parameter_labels, lower_bounds, upper_bounds, metadata:list, variables:list, extra_residuals:list,
                        scale = False, log=None, elambda = 1, dlambda=1, llambda=1, lelambda = 0, ldlambda=0, lllambda=0, elambda2 = 1, dlambda2=1, llambda2=1, lelambda2 = 0, ldlambda2=0, lllambda2=0, rmse = 'sum', objf = 'multi', normalize_fitness=True, fitness_std = False):
        self.model = model # now a list of models
        self.parameter_labels = parameter_labels[:-1] # all parameters across all models, only the ones that are going to be fitted
        self.set_bounds(upper_bounds, lower_bounds)
        self.scale = scale
        self.log = log
        self.data = data # now a list of data
        self.data_weights = data_weights
        self.metadata = metadata # now a list of metadas
        self.variables = variables # # now a list of dict of labels and values

        self.cvode_timepoints = 1000

        self.model_stuff = [self.ModelStuff(m, md, self.cvode_timepoints, self.parameter_labels, var, er) for m,md,var,er in zip(self.model, self.metadata, self.variables, extra_residuals)]
        self.dlambda = dlambda
        self.llambda = llambda
        self.elambda = elambda
        self.ldlambda = ldlambda
        self.lllambda = lllambda
        self.lelambda = lelambda
        self.dlambda2 = dlambda2
        self.llambda2 = llambda2
        self.elambda2 = elambda2
        self.ldlambda2 = ldlambda2
        self.lllambda2 = lllambda2
        self.lelambda2 = lelambda2
        self.rmse = rmse
        self.objf = objf


        self.current_parameters = None
        self.current_results = None
        self.current_fitness = None
        self.all_fitness = []
        self.nominal_fitness = None
        self.normalize_fitness = normalize_fitness
        self.fitness_std = fitness_std

    def _simulate(self, X):
        x = X[:-1]
        time = X[-1]
        from roadrunner import Config, RoadRunner, Logger
        Logger.disableLogging()
        Config.setValue(Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES, True)
        Config.setValue(Config.LOADSBMLOPTIONS_RECOMPILE, False) 
        Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, 4)
        Config.setValue(Config.LLVM_SYMBOL_CACHE, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_GVN, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER, True)
        Config.setValue(Config.SIMULATEOPTIONS_COPY_RESULT, True)

        all_results = []
        for r,ms, metadata, variables in zip(self.r, self.model_stuff, self.metadata, self.variables):
            results = {sample:np.ones((self.cvode_timepoints,len(ms.species_labels)))*1e12 for sample in metadata['sample_labels']}
            for sample in metadata['sample_labels']:
                
                x_dict = {k:v for k,v in zip(self.parameter_labels,x)}
                for label, value in variables[sample].items():
                    if not np.isnan(value):
                        if label in ms.species_labels:
                            x_dict[ms.species_to_v[label]]=(value+x_dict[ms.species_to_v[label]]*ms.species_init[label]*variables[sample]['dilution_factor'])/(ms.species_init[label]*variables[sample]['dilution_factor'])

                x2 = np.array(list(x_dict.values()))
                r.model.setGlobalParameterValues([*ms.parameter_order, *ms.variable_order[sample]], [*x2[ms.parameter_present], *np.array(list(variables[sample].values()))[ms.variable_present[sample]]])
                r.reset()

                try:
                    results[sample] = r.simulate(0,time,self.cvode_timepoints)[:,1:].__array__()
                except Exception as e:
                    print(e)
                    # break # stop if any fail
                r.resetToOrigin()
            all_results.append(results)
        del Config, RoadRunner, Logger, results

        self.current_x = x
        self.current_x2 = x2
        self.current_results = all_results
        return self.current_results

class SBML_Barebone_Multi_Fly:
    class ModelStuff:
        def __init__(self, model, parameter_labels):
            r = te.loadSBMLModel(model)
            self.species_labels = np.array(r.getFullStoichiometryMatrix().rownames)
            self.r_parameter_labels = np.array(r.getGlobalParameterIds())
            self.parameter_order = np.int32(np.squeeze(np.array([np.where(p == self.r_parameter_labels) for p in parameter_labels if p in self.r_parameter_labels])))
            self.parameter_present = [p in self.r_parameter_labels for p in parameter_labels]
            del r

    def __init__(self, model:list, parameter_labels, timepoint, num_metrics):
        self.model = model # now a list of models
        self.timepoint = timepoint
        self.parameter_labels = parameter_labels # all parameters across all models, only the ones that are going to be fitted
        self.cvode_timepoints = 1000
        self.model_stuff = [self.ModelStuff(m, self.parameter_labels) for m in self.model]
        self.num_metrics = num_metrics

    def _setup_rr(self): # run on engine
        from roadrunner import Config, RoadRunner, Logger
        Logger.disableLogging()
        Config.setValue(Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES, True)
        Config.setValue(Config.LOADSBMLOPTIONS_RECOMPILE, False) 
        Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, 4)
        Config.setValue(Config.LLVM_SYMBOL_CACHE, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_GVN, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER, True)
        Config.setValue(Config.SIMULATEOPTIONS_COPY_RESULT, True)
        self.r = []
        for m in self.model:
            r = te.loadSBMLModel(m)
            r.integrator.absolute_tolerance = 1e-8
            r.integrator.relative_tolerance = 1e-8
            r.integrator.maximum_num_steps = 2000
            self.r.append(r)
            
    def _simulate(self, x):
        from roadrunner import Config, RoadRunner, Logger
        Logger.disableLogging()
        Config.setValue(Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES, True)
        Config.setValue(Config.LOADSBMLOPTIONS_RECOMPILE, False) 
        Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, 4)
        Config.setValue(Config.LLVM_SYMBOL_CACHE, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_GVN, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER, True)
        Config.setValue(Config.SIMULATEOPTIONS_COPY_RESULT, True)

        all_results = []
        for r,ms in zip(self.r, self.model_stuff):

            # this sets the "parameters"
            r.model.setGlobalParameterValues([*ms.parameter_order], [*x[ms.parameter_present]])
            r.reset()

            # this sets species inital concentrations
            try:
                results = r.simulate(0,self.timepoint,self.cvode_timepoints)
            except Exception as e:
                print(e)
                results = np.nan
                # break # stop if any fail
            r.resetToOrigin()
            all_results.append(results)
        del Config, RoadRunner, Logger, results
        return all_results

    def _calculate_metrics(self, x): 
        import pandas as pd
        import numpy as np
        
        all_results = self._simulate(x)  # This returns a list of results
        all_metrics = []

        for result in all_results:
            # Check if the result is np.nan
            if np.isnan(result).all():
                # Append a row of NaN values if the result is np.nan
                rows = []

                for i in range(int(self.num_metrics/9)):
                    rows.append({ 'Final Concentration': np.nan,
                                        'Min Conc': np.nan,
                                        'Max Conc': np.nan,
                                        'Min Time': np.nan,
                                        'Max Time': np.nan,
                                        'Total Production': np.nan,
                                        'Production to Max': np.nan,
                                        'Half Max Time': np.nan,
                                        'Half Max Conc': np.nan })
            else:
                # result is a numpy array 
                # make a dataframe of the simulation results
                df_un = pd.DataFrame(result, columns=result.colnames)
                columns_to_keep = ['time'] + [col for col in df_un.columns if col.startswith('[C')]
                df = df_un[columns_to_keep]
                
                # create a list to store each row as a dictionary
                rows = []
                
                for compound in df.columns:
                    if compound == 'time':
                        continue
                    # calculate the initial concentration
                    initialconc = df[compound].iloc[0]
                    # calculate the final concentration
                    finalconc = df[compound].iloc[-1]
                    # calculates change in malate from start to finish
                    deltatot = finalconc - initialconc
                    # finds the minimum concentration and time at minima
                    minconc = min(df[compound])
                    mintime = df['time'][df[compound].idxmin()]
                    # finds the maximum concentration and time at maximum
                    maxconc = max(df[compound])
                    maxtime = df['time'][df[compound].idxmax()]
                    # calculates change in malate from start to max
                    deltamax = maxconc - initialconc
                    # calculates half of produced malate
                    halfmax = deltamax / 2
                    # finds the concentration and time closest to half max
                    df_closest = df.iloc[(df[compound] - (initialconc + halfmax)).abs().argsort()[:1]]
                    halftime = df_closest['time'].iloc[0]
                    halfconc = df_closest[compound].iloc[0]
                    # append the calculated metrics to the list of rows
                    rows.append({'Final Concentration': finalconc,
                                'Min Conc': minconc,
                                'Max Conc': maxconc,
                                'Min Time': mintime,
                                'Max Time': maxtime,
                                'Total Production': deltatot,
                                'Production to Max': deltamax,
                                'Half Max Time': halftime,
                                'Half Max Conc': halfconc})

            # create a dataframe from the list of rows
            df_final = pd.DataFrame(rows)
            all_metrics.extend(df_final.to_numpy())
            
        return np.array(all_metrics).reshape(-1)

    # gotta keep these around but we dont use them
    def fitness(self, x):
        return self._calculate_metrics(x)

    def get_bounds(self):
        return ([0 for i in self.parameter_labels], [1 for i in self.parameter_labels])
    
    def get_nobj(self):
        return self.num_metrics