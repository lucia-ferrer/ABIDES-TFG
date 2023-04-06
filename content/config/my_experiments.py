from adversarial.detectors import *
from adversarial.lucia.my_recover import *

detector_experiments = {
    #'KNN-Hyper': {},
    #'Kernel_Density': {},
    'DBSCAN': {}
    #'Gaussian_Mixture': {'n_clusters': [128, 256, 512, 1024, 2048] }
}

recovery_experiments = {
    'KNNRecovery': {
        'k': [1, 3, 5, 10],
        'consider_next_state': False,
        'consider_transition': [True, False],
        'window' : [3,4],
        'diff_state' : [False, True]
    },
    'TimeSeries': {}

}

DETECTOR_CLASS = {
    'KNN-Hyper': KNNHyper,
    'Kernel_Density': KernelDensity,
    'DBSCAN': DBSCAN,
    'Gaussian_Mixture': GaussianMixture
}

RECOVERY_CLASS = {
    'KNNRecovery': KNNRecovery,
    'TimeSeries': TimeSeries
}


def params_to_str(params, compact=False):
    params_str = params.copy()
    for k, v in params_str.items():
        if '__name__' in dir(v):
            params_str[k] = v.__name__
    params_str = str(params_str).translate({ord(i): None for i in "{}'"})
    if compact:
        params_str = params_str.replace(': ', '')
        if len(params) > 1: params_str.replace(', ', '_')
    return params_str
