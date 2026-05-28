from threading import Lock as _Lock
import pygmo as pg
from src.fitting.fastnumpyio import save, load

def _evolve_func_ipy(algo, udbfe, prob):
    # The evolve function that is actually run from the separate processes
    # in ipyparallel_island.
    from pickle import dumps, loads
    xs = load(udbfe.temp_dv_path+'/dvs_bunch_'+str(udbfe.instance_id)+'.b')
    fs = load(udbfe.temp_dv_path+'/fs_bunch_'+str(udbfe.instance_id)+'.b')
    pop = pg.population(prob, size = 0)  # not the problem with _r initialized
    list(map(pop.push_back, xs, fs))
    # pop = loads(ser_pop)
    new_pop = algo.evolve(pop)
    save(udbfe.temp_dv_path+'/dvs_bunch_'+str(udbfe.instance_id)+'.b', new_pop.get_x())
    save(udbfe.temp_dv_path+'/fs_bunch_'+str(udbfe.instance_id)+'.b', new_pop.get_f())
    return 

def _eval_func_ipy(udbfe, ser_pop_prob, mode):
    # The evolve function that is actually run from the separate processes
    # in ipyparallel_island.
    from pickle import dumps, loads
    import numpy as np
    pop,prob = loads(ser_pop_prob)
    # xs = load(udbfe.temp_dv_path+'/dvs_bunch_'+str(udbfe.instance_id)+'.b')
    fs = udbfe(prob, pop.reshape(-1), mode)[:,np.newaxis]
    return dumps(fs)

class my_ipyparallel_island(object):
    # Static variables for the view.
    _view_lock = _Lock()
    _views = {}
    _cluster = None

    def __init__(self, id, client_kwargs, view_kwargs, bfe_client_kwargs, bfe_view_kwargs, bfe_kwargs, algo_kwargs):
        self.client_kwargs = client_kwargs
        self.view_kwargs = view_kwargs
        self.bfe_client_kwargs = bfe_client_kwargs
        self.bfe_view_kwargs = bfe_view_kwargs
        self.client_size = None
        self.bfe_kwargs = bfe_kwargs
        self.algo_kwargs = algo_kwargs
        self.instance_id = id

    def init_cluster(self):
        with my_ipyparallel_island._view_lock:
            if my_ipyparallel_island._cluster is None:
                import os
                from ipyparallel import Client
                my_ipyparallel_island._cluster = Client(**self.client_kwargs)
                my_ipyparallel_island._cluster.wait_for_engines(n=self.client_kwargs['n'])
                my_ipyparallel_island._cluster[:].apply_sync(os.chdir,'/mmfs1/gscratch/cheme/dalba/repos/ECFERS');
                my_ipyparallel_island._cluster[:].use_cloudpickle()

    def init_view(self):
        if my_ipyparallel_island._cluster is None: self.init_cluster()
        with my_ipyparallel_island._view_lock:
            if self.instance_id not in my_ipyparallel_island._views.keys():
                # Create the new view.
                my_ipyparallel_island._views[self.instance_id] = my_ipyparallel_island._cluster[self.instance_id]
                self.init_algo()

    def init_algo(self):
        make_algo = """
        import pygmo as pg
        from src.fitting.pygmo_bfes import pickleless_bfe2
        pickleless_bfe2.instance_counter = instance_id
        prob = bfe_kwargs['prob']['train']
        udbfe = pickleless_bfe2(client_kwargs=bfe_client_kwargs, view_kwargs=bfe_view_kwargs,**bfe_kwargs)
        udbfe.init_view()
        
        a = algo_kwargs.pop('algo')
        if a == 'pso':
            algo = pg.pso_gen(**algo_kwargs)
        elif a == 'moead':
            algo = pg.moead_gen(**algo_kwargs)

        algo.set_bfe(pg.bfe(udbfe))
        algo = pg.algorithm(algo)
        algo.set_verbosity(1)"""
        my_ipyparallel_island._views[self.instance_id].push(dict(bfe_client_kwargs=self.bfe_client_kwargs, bfe_view_kwargs=self.bfe_view_kwargs, bfe_kwargs=self.bfe_kwargs, algo_kwargs=self.algo_kwargs, instance_id=self.instance_id), block=True)
        my_ipyparallel_island._views[self.instance_id].execute(make_algo, block=False) # maybe this can be not blocking

    @staticmethod
    def shutdown_view(self):
        """Destroy the ipyparallel view.

        .. versionadded:: 2.12

        This method will destroy the :class:`ipyparallel.LoadBalancedView`
        currently being used by the ipyparallel islands for submitting
        evolution tasks to an ipyparallel cluster. The view can be re-inited
        implicitly by submitting a new evolution task, or by invoking
        the :func:`~pygmo.ipyparallel_island.init_view()` method.

        """
        import gc

        with my_ipyparallel_island._view_lock:
            if my_ipyparallel_island._views[self.instance_id] is None:
                return

            old_view = my_ipyparallel_island._views[self.instance_id]
            my_ipyparallel_island._views[self.instance_id] = None
            del old_view
            gc.collect()

    def run_evolve(self, algo, pop, block = True):
        """Evolve population.

        This method will evolve the input :class:`~pygmo.population` *pop* using the input
        :class:`~pygmo.algorithm` *algo*, and return *algo* and the evolved population. The evolution
        task is submitted to the ipyparallel cluster via a global :class:`ipyparallel.LoadBalancedView`
        instance initialised either implicitly by the first invocation of this method,
        or by an explicit call to the :func:`~pygmo.ipyparallel_island.init_view()` method.

        Args:

            pop(:class:`~pygmo.population`): the input population
            algo(:class:`~pygmo.algorithm`): the input algorithm

        Returns:

            :class:`tuple`: a tuple of 2 elements containing *algo* (i.e., the :class:`~pygmo.algorithm` object that was used for the evolution) and the evolved :class:`~pygmo.population`

        Raises:

            unspecified: any exception thrown by the evolution, by the creation of a
              :class:`ipyparallel.LoadBalancedView`, or by the sumission of the evolution task
              to the ipyparallel cluster

        """
        # NOTE: as in the mp_island, we pre-serialize
        # the algo and pop, so that we can catch
        # serialization errors early.
        from pickle import dumps, loads
        import ipyparallel as ipp

        save(self.bfe_kwargs['temp_dv_path']+'/dvs_bunch_'+str(self.instance_id)+'.b', pop.get_x())
        save(self.bfe_kwargs['temp_dv_path']+'/fs_bunch_'+str(self.instance_id)+'.b', pop.get_f())

        with my_ipyparallel_island._view_lock:
            if self.instance_id not in my_ipyparallel_island._views.keys():
                self.init_view()
            ret = my_ipyparallel_island._views[self.instance_id].apply_async(_evolve_func_ipy, ipp.Reference("algo"),ipp.Reference("udbfe"), ipp.Reference("prob"))

        if block: 
            ret.get()
            return (algo, self._load_pop())
        else:
            return ret
    
    def _load_pop(self):
        xs = load(self.bfe_kwargs['temp_dv_path']+'/dvs_bunch_'+str(self.instance_id)+'.b')
        fs = load(self.bfe_kwargs['temp_dv_path']+'/fs_bunch_'+str(self.instance_id)+'.b')
        pop = pg.population(self.bfe_kwargs['prob']['train'], size = 0)
        list(map(pop.push_back, xs, fs))
        return pop
    
    def run_eval(self, prob, xs, mode ='train', block = False):
        """Evaluate a set of decision vectors.

        This method will evaluate the input :class:`~pygmo.problem` *prob* on the input
        decision vectors *xs*, and return the corresponding fitness values. The evaluation
        task is submitted to the ipyparallel cluster via a global :class:`ipyparallel.LoadBalancedView`
        instance initialised either implicitly by the first invocation of this method,
        or by an explicit call to the :func:`~pygmo.ipyparallel_island.init_view()` method.

        Args:

            prob(:class:`~pygmo.problem`): the input problem
            xs(:class:`numpy.ndarray`): the input decision vectors

        Returns:

            :class:`numpy.ndarray`: the fitness values corresponding to the input decision vectors

        Raises:

            unspecified: any exception thrown by the evaluation, by the creation of a
              :class:`ipyparallel.LoadBalancedView`, or by the sumission of the evaluation task
              to the ipyparallel cluster

        """
        # NOTE: as in the mp_island, we pre-serialize
        # the algo and pop, so that we can catch
        # serialization errors early.
        from pickle import dumps, loads
        import ipyparallel as ipp

        # save(self.bfe_kwargs['temp_dv_path']+'/dvs_bunch_'+str(self.instance_id)+'.b', xs)
        ser_pop_prob = dumps((xs,prob))
        with my_ipyparallel_island._view_lock:
            if self.instance_id not in my_ipyparallel_island._views.keys():
                self.init_view()
            ret = my_ipyparallel_island._views[self.instance_id].apply_async(_eval_func_ipy, ipp.Reference("udbfe"), ser_pop_prob, mode)

        if block: return loads(ret.get())
        else: return ret

    def get_name(self):
        """Island's name.

        Returns:
            :class:`str`: ``"Ipyparallel island"``

        """
        return "Ipyparallel island"

    def get_extra_info(self):
        """Island's extra info.

        Returns:
            :class:`str`: a string with extra information about the status of the island

        """
        from copy import deepcopy

        with my_ipyparallel_island._view_lock:
            if my_ipyparallel_island._views[self.instance_id] is None:
                return "\tNo cluster view has been created yet"
            else:
                d = deepcopy(my_ipyparallel_island._views[self.instance_id].queue_status())
        return "\tQueue status:\n\t\n\t" + "\n\t".join(
            ["(" + str(k) + ", " + str(d[k]) + ")" for k in d]
        )