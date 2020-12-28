import json
import numpy as np
from collections import OrderedDict
from functools import partial
from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.surrogates.RandomForest import RandomForest
from pyGPGO.GPGO import GPGO

from src.runner import run_wparams

acq_func = ['ExpectedImprovement', 'ProbabilityImprovement']

def gproc(evaluate_network, params, acquisition_function='ExpectedImprovement'):
    print('init Gaussian process')
    sexp = squaredExponential()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode=acquisition_function)

    gpgo = GPGO(gp, acq, evaluate_network, params)
    try:
        gpgo.run(init_evals=5, max_iter=5)
        res = gpgo.getResult()
    except Exception as err:
        print("Error handled: ", err)
        res = [None, 0]
        raise err
    return res[0], res[1]


def rforest(evaluate_network, params, acquisition_function='ExpectedImprovement'):
    print('init Random forest')
    rf = RandomForest()
    acq = Acquisition(mode=acquisition_function)

    rf1 = GPGO(rf, acq, evaluate_network, params)    
    try:
        rf1.run(init_evals=5, max_iter=5)
        res = rf1.getResult()
    except Exception as err:
        print("Error handled: ", err)
        res = [None, 0]
    return res[0], res[1]
   
def evaluate_model(constants, **sampling_params):
    params = {**sampling_params, **constants}
    # print("Evaluating", sampling_params)
    return run_wparams(verbose=0, t_params=params)

def hp_sampling():
    best_params, best_acc = None, 0
    surrogates = [gproc, rforest]
    params = OrderedDict()
    constants = {}
    with open('auto_config.json') as json_file:
        autoconfig = json.load(json_file)
        constants = autoconfig["constant"]
        for k in autoconfig["sample"].keys():
            params[k] = ('int', autoconfig["sample"][k])
    print('Sampling', params.keys())
    evaluate_model_bound = partial(evaluate_model, constants)
    acquisition_function = 'ExpectedImprovement'
    for surrogate in surrogates:
        next_params, next_acc = surrogate(evaluate_model_bound, params, acquisition_function)
        if best_acc < next_acc:
            best_acc = next_acc
            best_params = next_params
    print('Sampled', best_params, 'with score', best_acc)
    return best_params