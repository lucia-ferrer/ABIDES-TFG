from adversarial.detectors import *
from adversarial.lucia.my_recover import *

detector_experiments = {
    #'KNN-Hyper': {},
    #'Kernel_Density': {},
    #'GainDiscriminator':{}
    'DBSCAN': {}
    #'Gaussian_Mixture': {'n_clusters': [128, 256, 512, 1024, 2048] }
}

recovery_experiments = {
    #'GainDiscriminator':{},
    'KNNRecovery': {
        'k': [1,2,3,5],
        'trans': [True, False],
        'window' : [2,3]
        #'consider_next_state': False,
        #'diff_state' : False
    },
    'None': {}
    #'TimeSeries': {}
}

DETECTOR_CLASS = {
    #'GainDiscriminator' : GainDiscriminator,
    'KNN-Hyper': KNNHyper,
    'Kernel_Density': KernelDensity,
    'DBSCAN': DBSCAN,
    'Gaussian_Mixture': GaussianMixture
}

RECOVERY_CLASS = {
    #'GainRecovery': GainRecovery,
    'KNNRecovery': KNNRecovery,
    'None' : lambda *args: {}
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
