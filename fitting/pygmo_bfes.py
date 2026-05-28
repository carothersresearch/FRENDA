import pygmo as pg
from src.fitting.pygmo_problems import *
from src.fitting.fastnumpyio import save, load
from threading import Lock as _Lock

def get_partition(n_items: int, rank: int, size: int)-> tuple[int, int]: 
    """
    Compute the partition

    Returns (start, end) of partition
    """
    chunk_size = n_items // size
    if n_items % size:
        chunk_size += 1
    start = rank * chunk_size
    if rank + 1 == size:
        end = n_items
    else:
        end = start + chunk_size
    return (start, end)

def _ipy_bfe_func(dv_path, prob, rank, size):
    # The function that will be invoked
    # by the individual processes/nodes of mp/ipy bfe.
    import pickle

    all_dvs = load(dv_path)

    # pick dvs based on engine rank
    n_items = all_dvs.shape[0]
    start, end = get_partition(n_items, rank, size)
    dvs = all_dvs[start:end,:]

    return pickle.dumps(list(map(prob.fitness, dvs)))

def _ipy_bfe_func2(path, prob, rank):
    # The function that will be invoked
    # by the individual processes/nodes of mp/ipy bfe.
    # import pickle

    dvs = load(path+'/dvs_'+str(rank)+'.b')
    f = np.array(list(map(prob.fitness, dvs)))
    save(path+'/f_'+str(rank)+'.b', f)
    return
    # return pickle.dumps(list(map(prob.fitness, dvs)))

class pickleless_bfe(pg.ipyparallel_bfe):
    def __init__(self, client_kwargs, view_kwargs, temp_dv_path, prob:dict):
        self.client_kwargs = client_kwargs
        self.view_kwargs = view_kwargs
        self.temp_dv_path = temp_dv_path
        self.client_size = None
        self.prob = prob
        super().__init__()

    def init_view(self,client_args=[], client_kwargs={}, view_args=[], view_kwargs={}):
        self.client_kwargs = client_kwargs
        self.view_kwargs = view_kwargs
        with pg.ipyparallel_bfe._view_lock:
            if pg.ipyparallel_bfe._view is None:
                # Create the new view.
                from ipyparallel import Client, Reference
                rc = Client(*client_args, **self.client_kwargs)
                rc[:].use_cloudpickle()
                pg.ipyparallel_bfe._view = rc.broadcast_view(*view_args, **self.view_kwargs)
                pg.ipyparallel_bfe._view.is_coalescing = False
                self.client_size = len(rc.ids)
                pg.ipyparallel_bfe._view.scatter("rank", range(len(rc.ids)), flatten=True)
                for k,v in self.prob.items():
                    prob_id = "prob_"+k
                    pg.ipyparallel_bfe._view.push({prob_id: v}, block = True)
                    pg.ipyparallel_bfe._view.apply_sync(lambda x: x.extract(SBMLGlobalFit_Multi_Fly)._setup_rr(), Reference(prob_id))

    def __call__(self, prob, dvs, mode = 'train'):
        import ipyparallel as ipp
        import pickle
        import numpy as np
        prob_id = "prob_"+mode
        # Fetch the dimension and the fitness
        # dimension of the problem.
        ndim = prob.get_nx()
        nf = prob.get_nf()

        # Compute the total number of decision
        # vectors represented by dvs.
        ndvs = len(dvs) // ndim
        # Reshape dvs so that it represents
        # ndvs decision vectors of dimension ndim
        # each.
        dvs.shape = (ndvs, ndim)

        # Save dvs in a binary file
        for i in range(self.client_size):
            # pick dvs based on engine rank
            start, end = get_partition(ndvs, i, self.client_size)
            save(self.temp_dv_path+'/dvs_'+str(i)+'.b', dvs[start:end,:])

        with pg.ipyparallel_bfe._view_lock:

            ar = pg.ipyparallel_bfe._view.apply(_ipy_bfe_func2, self.temp_dv_path, ipp.Reference(prob_id), ipp.Reference("rank"))
            # ret = ipp.AsyncMapResult(pg.ipyparallel_bfe._view.client, ar._children, ipp.client.map.Map())
            
        # Build the vector of fitness vectors as a 2D numpy array.
        fvs = np.array(sum([pickle.loads(fv) for fv in ar.get()],[]))
        # Reshape it so that it is 1D.
        fvs.shape = (ndvs*nf,)
        del ar, dvs, ipp
        return fvs
    
class pickleless_bfe2(object):
    _view_lock = _Lock()
    _views = {}
    instance_counter = 0

    def __init__(self, client_kwargs, view_kwargs, temp_dv_path, prob:dict):
        self.client_kwargs = client_kwargs
        self.view_kwargs = view_kwargs
        self.temp_dv_path = temp_dv_path
        self.client_size = None
        self.prob = prob
        self.instance_id = pickleless_bfe2.instance_counter
        pickleless_bfe2.instance_counter += 1
    
    def init_view(self):
        with pickleless_bfe2._view_lock:
            if self.instance_id not in pickleless_bfe2._views.keys():
                # Create the new view.
                from ipyparallel import Client, Reference
                rc = Client(**self.client_kwargs)
                rc[:].use_cloudpickle()
                pickleless_bfe2._views[self.instance_id] = rc.broadcast_view(**self.view_kwargs)
                pickleless_bfe2._views[self.instance_id].is_coalescing = False
                pickleless_bfe2._views[self.instance_id].scatter("rank", self.view_kwargs['targets'], flatten=True)
                pickleless_bfe2._views[self.instance_id].push({'path':self.temp_dv_path}, block = True)
                for k,v in self.prob.items():
                    prob_id = "prob_"+k
                    pickleless_bfe2._views[self.instance_id].push({prob_id: v}, block = True)
                    pickleless_bfe2._views[self.instance_id].apply_async(lambda x: x.extract(SBMLGlobalFit_Multi_Fly)._setup_rr(), Reference(prob_id))

    def __call__(self, prob, dvs, mode = 'train'):
        import ipyparallel as ipp
        import pickle
        import numpy as np
        prob_id = "prob_"+mode
        # Fetch the dimension and the fitness
        # dimension of the problem.
        ndim = prob.get_nx()
        nf = prob.get_nf()

        # Compute the total number of decision
        # vectors represented by dvs.
        ndvs = len(dvs) // ndim
        # Reshape dvs so that it represents
        # ndvs decision vectors of dimension ndim
        # each.
        dvs.shape = (ndvs, ndim)
        
        ndim = dvs.shape[1]

        # Save dvs in a binary file
        for k,i in enumerate(self.view_kwargs['targets']):
            # pick dvs based on engine rank
            start, end = get_partition(ndvs, k, len(self.view_kwargs['targets']))
            save(self.temp_dv_path+'/dvs_'+str(i)+'.b', dvs[start:end,:])

        with pickleless_bfe2._view_lock:
            if self.instance_id not in pickleless_bfe2._views.keys():
                self.init_view()

            ar = pickleless_bfe2._views[self.instance_id].apply(_ipy_bfe_func2, ipp.Reference('path'), ipp.Reference(prob_id), ipp.Reference("rank"))
            # ret = ipp.AsyncMapResult(pg.ipyparallel_bfe._view.client, ar._children, ipp.client.map.Map())
            
        # Build the vector of fitness vectors as a 2D numpy array.
        ar.get()
        # fvs = np.array(sum([pickle.loads(fv) for fv in ar.get()],[]))
        fvs = np.array([load(self.temp_dv_path+'/f_'+str(i)+'.b') for i in self.view_kwargs['targets']])
        # Reshape it so that it is 1D.
        fvs.shape = (ndvs*nf,)
        return fvs
    
    def get_name(self):
        """Name of the evaluator.

        Returns:
            :class:`str`: ``"Ipyparallel batch fitness evaluator"``

        """
        return "My ipyparallel batch fitness evaluator"

    def get_extra_info(self):
        """Extra info for this evaluator.

        Returns:
            :class:`str`: a string with extra information about the status of the evaluator

        """
        from copy import deepcopy

        with pickleless_bfe2._view_lock:
            if pickleless_bfe2._views[self.instance_id] is None:
                return "\tNo cluster view has been created yet"
            else:
                d = deepcopy(pickleless_bfe2._views[self.instance_id].queue_status())
        return "\tQueue status:\n\t\n\t" + "\n\t".join(
            ["(" + str(k) + ", " + str(d[k]) + ")" for k in d]
        )

    def shutdown_view(self):
        """Destroy the ipyparallel view.

        This method will destroy the :class:`ipyparallel.LoadBalancedView`
        currently being used by the ipyparallel evaluators for submitting
        evaluation tasks to an ipyparallel cluster. The view can be re-inited
        implicitly by submitting a new evaluation task, or by invoking
        the :func:`~pygmo.ipyparallel_bfe.init_view()` method.

        """
        import gc

        with pickleless_bfe2._view_lock:
            if pickleless_bfe2._views[self.instance_id] is None:
                return

            old_view = pickleless_bfe2._views[self.instance_id]
            pickleless_bfe2._views[self.instance_id] = None
            del old_view
            gc.collect()

def _ipy_bfe_metrics(dv_path, prob, rank):
    # The function that will be invoked
    # by the individual processes/nodes of mp/ipy bfe.
    import pickle

    dvs = load(dv_path+'/dvs_'+str(rank)+'.b')
    print([str(rank), ])

    return pickle.dumps(list(map(prob.fitness, dvs)))

class mkcook_bfe(pg.ipyparallel_bfe):
    def __init__(self, client_kwargs, view_kwargs, temp_dv_path, prob):
        self.client_kwargs = client_kwargs
        self.view_kwargs = view_kwargs
        self.temp_dv_path = temp_dv_path
        self.client_size = None
        self.prob = prob
        super().__init__()

    def init_view(self,client_args=[], client_kwargs={}, view_args=[], view_kwargs={}):
        self.client_kwargs = client_kwargs
        self.view_kwargs = view_kwargs
        with pg.ipyparallel_bfe._view_lock:
            if pg.ipyparallel_bfe._view is None:
                # Create the new view.
                from ipyparallel import Client, Reference
                rc = Client(*client_args, **self.client_kwargs)
                rc[:].use_cloudpickle()
                pg.ipyparallel_bfe._view = rc.broadcast_view(*view_args, **self.view_kwargs)
                pg.ipyparallel_bfe._view.is_coalescing = False
                self.client_size = len(rc.ids)
                pg.ipyparallel_bfe._view.scatter("rank", rc.ids, flatten=True)
                pg.ipyparallel_bfe._view.push({'prob': self.prob}, block = True)
                pg.ipyparallel_bfe._view.apply_sync(lambda x: x.extract(SBML_Barebone_Multi_Fly)._setup_rr(), Reference('prob'))

    def __call__(self, prob, dvs):
        import ipyparallel as ipp
        import pickle
        import numpy as np
        # Fetch the dimension and the fitness
        # dimension of the problem.
        ndim = prob.get_nx()
        nf = prob.get_nf()

        # Compute the total number of decision
        # vectors represented by dvs.
        ndvs = len(dvs) // ndim
        # Reshape dvs so that it represents
        # ndvs decision vectors of dimension ndim
        # each.
        dvs.shape = (ndvs, ndim)

        # Save dvs in a binary file
        for i in range(self.client_size):
            # pick dvs based on engine rank
            start, end = get_partition(ndvs, i, self.client_size)
            save(self.temp_dv_path+'/dvs_'+str(i)+'.b', dvs[start:end,:])

        with pg.ipyparallel_bfe._view_lock:

            ar = pg.ipyparallel_bfe._view.apply(_ipy_bfe_metrics, self.temp_dv_path, ipp.Reference('prob'), ipp.Reference("rank"))
            # ret = ipp.AsyncMapResult(pg.ipyparallel_bfe._view.client, ar._children, ipp.client.map.Map())
            
        # Build the vector of fitness vectors as a 2D numpy array.
        fvs = np.array(sum([pickle.loads(fv) for fv in ar.get()],[]))
        # Reshape it so that it is 1D.
        fvs.shape = (ndvs*nf,)
        del ar, dvs, ipp
        return fvs