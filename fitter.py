from functools import partial
import lmfit
import math
import matplotlib.pyplot as plt
import numpy as np


from extraction_functions import EDC_prep, EDC_array_with_SE, symmetrize_EDC, Norman_EDC_array
from general import k_as_index, get_degree_polynomial


class Fitter:

    @staticmethod
    def NormanFit(Z, k, w, a_estimate, c_estimate, kf_index, energy_conv_sigma, temp):
        pars = lmfit.Parameters()
        pars.add('scale', value=70000, min=0, max=1000000)
        pars.add('T', value=7, min=0, max=25)
        pars.add('dk', value=10, min=0, max=25)
        pars.add('s', value=1500, min=1000, max=2000, vary=True)
        pars.add('a', value=a_estimate, min=a_estimate / 1.5, max=a_estimate * 1.5)
        pars.add('c', value=c_estimate, min=c_estimate * 1.5, max=c_estimate / 1.5) #pass ac for a and c
        # low_noise_w, low_noise_slice, _, _, _, _ = EDC_prep(kf_index, Z, w, min_fit_count)
        low_noise_w = w
        low_noise_slice = [Z[i][kf_index] for i in range(len(w))]
        low_noise_w, low_noise_slice = symmetrize_EDC(low_noise_w, low_noise_slice)
        EDC_func = partial(Norman_EDC_array, fixed_k=math.fabs(k[kf_index]),
                           energy_conv_sigma=energy_conv_sigma, temp=temp)

        def calculate_residual(p):
            EDC_residual = EDC_func(
                np.asarray(low_noise_w), p['scale'], p['T'], p['dk'], p['s'], p['a'], p['c']) - low_noise_slice
            weighted_EDC_residual = EDC_residual / np.sqrt(low_noise_slice)
            return weighted_EDC_residual

        mini = lmfit.Minimizer(calculate_residual, pars, nan_policy='omit', calc_covar=True)
        result = mini.minimize(method='leastsq')
        print(lmfit.fit_report(result))
        scale = result.params.get('scale').value
        T = result.params.get('T').value
        dk = result.params.get('dk').value
        s = result.params.get('s').value
        a = result.params.get('a').value
        c = result.params.get('c').value
        plt.title("Norman fit")
        plt.plot()
        plt.plot(low_noise_w, low_noise_slice, label='data')
        plt.plot(low_noise_w, EDC_func(low_noise_w, scale, T, dk, s, a, c), label='fit')
        plt.show()

