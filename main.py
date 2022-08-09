from data_reader import DataReader
from extract_ac import extract_ac
# from extract_k_dependent import KDependentExtractor
from fitter import Fitter
from general import k_as_index
global results


def run():

    # Detector settings
    temperature = 103.63
    energy_conv_sigma = 8 / 2.35482004503
    data = DataReader(
        fileName=r"/Users/eddywu/On my mac/X20141210_far_off_node/OD50_0333_nL.dat")

    # Get initial estimates (a, c, dk, kf, k_error) - Small height and not too wide for lorentz+SEC fits
    data.getZoomedData(width=118, height=38, x_center=359, y_center=65)
    initial_a_estimate, initial_c_estimate, initial_dk_estimate, initial_kf_estimate, initial_k_error = extract_ac(
        data.zoomed_Z,
        data.zoomed_k,
        data.zoomed_w,
        show_results=True)

    # single EDC fits
    data.getZoomedData(width=140, height=50, x_center=360, y_center=50)
    data.zoomed_k -= initial_k_error

    print('FIT with initial_kf_estimate')
    Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                     k_as_index(initial_kf_estimate, data.zoomed_k), energy_conv_sigma, temperature)
    # results = mini.minimize(method='leastsq')
    # dk_estimate = result.params['dk'].value
    # print(dk_estimate)
    # quit()



    # six EDC fits (manual k)
    print('FIT 1')
    Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                     k_as_index(0.093, data.zoomed_k), energy_conv_sigma, temperature)
    # dk1 =
    print('FIT 2')
    Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                     k_as_index(0.091, data.zoomed_k), energy_conv_sigma, temperature)
    # dk2 =
    print('FIT 3')
    Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                     k_as_index(0.092, data.zoomed_k), energy_conv_sigma, temperature)
    # dk3 =
    print('FIT 4')
    Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                     k_as_index(0.094, data.zoomed_k), energy_conv_sigma, temperature)
    # dk4 =
    print('FIT 5')
    Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                     k_as_index(0.096, data.zoomed_k), energy_conv_sigma, temperature)
    # dk5 =
    print('FIT 6')
    Fitter.NormanFit(data.zoomed_Z, data.zoomed_k, data.zoomed_w, initial_a_estimate, initial_c_estimate,
                     k_as_index(0.090, data.zoomed_k), energy_conv_sigma, temperature)
    # dk6 =


if __name__ == '__main__':
    run()
