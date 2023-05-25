import time as python_time
import numpy as np
import sys
from collections import OrderedDict
from pymoo.core.problem import ElementwiseProblem

from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions
from pyLIMA.priors import parameters_boundaries

class MLProblem(ElementwiseProblem):

    def __init__(self, bounds, objective_photometry=None, objective_astrometry=None, **kwargs):

        n_var = len(bounds)
        n_obj = 0

        if objective_photometry is not None:

            n_obj += 1
            self.objective_photometry = objective_photometry

        else:

            self.objective_photometry = None

        if objective_astrometry is not None:

            n_obj += 1
            self.objective_astrometry = objective_astrometry

        else:

            self.objective_astrometry = None


        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=0,
                         xl=np.array([i[0] for i in bounds]),
                         xu=np.array([i[1] for i in bounds]),
                         **kwargs)




    def _evaluate(self, x, out, *args, **kwargs):

        objectives = []

        if self.objective_photometry is not None:

            objectives.append(self.objective_photometry(x))

        if self.objective_astrometry is not None:

            objectives.append(self.objective_astrometry(x))

        out["F"] = objectives


class NGSA2fit(MLfit):

    def fit_type(self):
        return "Non-dominated Sorting Genetic Algorithm"





    def fit(self,computational_pool=None):

        starting_time = python_time.time()

        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.factory import get_sampling, get_crossover, get_mutation
        from pymoo.operators.selection.rnd import RandomSelection
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PolynomialMutation
        from pymoo.operators.repair.rounding import RoundingRepair

#        algorithm = NSGA2(
#            pop_size=40,
#            n_offsprings=10,
#            sampling=RandomSelection(),
#            crossover=SBX(),
#            mutation= PolynomialMutation(prob=1.0, eta=20, repair=RoundingRepair()),
#            eliminate_duplicates=True
#        )

        algorithm = NSGA2(pop_size=100)

        if self.model.astrometry is not None:

            astrometry = self.likelihood_astrometry
        else:
            astrometry = None

        if self.model.photometry is not None:

            photometry = self.likelihood_photometry
        else:
            photometry = None

        bounds = [self.fit_parameters[key][1] for key in self.fit_parameters.keys()]

        problem = MLProblem(bounds, photometry, astrometry)

        from pymoo.optimize import minimize

        res = minimize(problem,
                       algorithm,
                       ('n_gen', 200),
                        seed=1,
                       save_history=True,
                       verbose=True)

        X = res.X
        F = res.F
        n_evals = np.array([e.evaluator.n_eval for e in res.history])
        opt = np.array([e.opt[0].F for e in res.history])

        # mask =np.argmin((F[:,0]/F[:,0].max())**2+(F[:,1]/F[:,1].max())**2)

        import pdb;
        pdb.set_trace()

        print('DE converge to objective function : f(x) = ', str(differential_evolution_estimation['fun']))
        print('DE converge to parameters : = ', differential_evolution_estimation['x'].astype(str))

        fit_results = np.hstack((differential_evolution_estimation['x'],differential_evolution_estimation['fun']))

        computation_time = python_time.time() - starting_time
        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')
        self.DE_population = np.array(self.DE_population)
        self.fit_results = fit_results
        self.fit_time = computation_time