import sys
import time as python_time

import numpy as np
import scipy
from pyLIMA.fits.ML_fit import MLfit


class LMfit(MLfit):
    """
    Standard Levenberg-Marquardt fit

    Attributes
    -----------
    guess : list, the starting point of the fit
    """
    def __init__(self, model, telescopes_fluxes_method='fit', loss_function='chi2'):
        """The fit class has to be intialized with an event object."""

        if loss_function == 'likelihood':
            print(
                'Cannot use likelihood with gradient-like method,switching to chi2 ('
                'default)')
            loss_function = 'chi2'

        super().__init__(model, telescopes_fluxes_method=telescopes_fluxes_method,
                         loss_function=loss_function)

        self.guess = []

    def fit_type(self):
        return "Levenberg-Marquardt"

    def objective_function(self, fit_process_parameters):

        parameters = np.array(fit_process_parameters)

        model_parameters = parameters[self.model_parameters_index]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        residus, err = self.model_residuals(pyLIMA_parameters,
                                            rescaling_photometry_parameters=None,
                                            rescaling_astrometry_parameters=None)

        residuals = []
        errors = []

        for data_type in ['photometry', 'astrometry']:

            try:

                residuals.append(np.concatenate(residus[data_type]))
                errors.append(np.concatenate(err[data_type]))

            except ValueError:

                pass

        residuals = np.concatenate(residuals)
        errors = np.concatenate(errors)

        return residuals / errors

    def fit(self):

        start_time = python_time.time()

        # use the analytical Jacobian (faster) if no second order are present,
        # else let the
        # algorithm find it.
        self.guess = self.initial_guess()

        if self.guess is None:
            return


        n_data = 0
        for telescope in self.model.event.telescopes:
            n_data = n_data + telescope.n_data('flux')

        if self.model.Jacobian_flag != 'No Way':

            jacobian_function = self.residuals_Jacobian

        else:

            jacobian_function = '2-point'

        if self.loss_function == 'soft_l1':

            loss = 'soft_l1'
            #jacobian_function = '2-point'

        else:

            loss = 'linear'

        bounds_min = [self.fit_parameters[key][1][0] for key in
                      self.fit_parameters.keys()]
        bounds_max = [self.fit_parameters[key][1][1] for key in
                      self.fit_parameters.keys()]

        lm_fit = scipy.optimize.least_squares(self.objective_function, self.guess,
                                              method='lm', max_nfev=50000,
                                              jac=jacobian_function, loss=loss,
                                              xtol=10 ** -10, ftol=10 ** -10,
                                              gtol=10 ** -10,
                                              x_scale=(np.array(bounds_max)-bounds_min)/2)

        fit_results = lm_fit['x'].tolist()
        fit_chi2 = lm_fit['cost'] * 2  # chi2

        try:
            # Try to extract the covariance matrix from the levenberg-marquard_fit
            # output
            covariance_matrix = np.linalg.pinv(np.dot(lm_fit['jac'].T, lm_fit['jac']))

        except ValueError:

            covariance_matrix = np.zeros((len(self.model.model_dictionnary),
                                          len(self.model.model_dictionnary)))

        covariance_matrix *= fit_chi2 / (n_data - len(self.model.model_dictionnary))
        computation_time = python_time.time() - start_time

        print(sys._getframe().f_code.co_name, ' : ' + self.fit_type() + ' fit SUCCESS')
        print('best_model:', fit_results, ' chi2:', fit_chi2)

        self.fit_results = {'best_model': fit_results, 'chi2': fit_chi2,
                            'fit_time': computation_time,
                            'covariance_matrix': covariance_matrix}

    def samples_to_plot(self):

        samples = np.random.multivariate_normal(self.fit_results['best_model'],
                                                self.fit_results['covariance_matrix'],
                                                10000)
        return samples
