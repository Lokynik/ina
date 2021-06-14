import numpy as np
import pandas as pd
import scipy.optimize as scop
import matplotlib.pyplot as plt
import os
import ctypes
import sys
from sklearn.metrics import mean_squared_error as MSE

def scale(x, min_bound, max_bound):
    y = (x.copy() - min_bound)/(max_bound -  min_bound)
    return(y)

def rescale(x, min_bound, max_bound):
    y = x.copy() * (max_bound - min_bound) + min_bound
    return(y)

def calculate_full_trace(y, *args):

    kwargs = args[-1]

    t = kwargs['t']#there should be time for the first step
    v_all = kwargs['v']


    output_S = kwargs['output_S']
    output_A = kwargs['output_A']
    bounds = kwargs['bounds']

    S = kwargs['S']

    t0 = kwargs['t0']
    v0 = kwargs['v0']

    initial_state_S = kwargs['initial_state_S']
    initial_state_A = kwargs['initial_state_A']
    initial_state_len = kwargs['initial_state_len']
    filename_abs = kwargs['filename_abs']

    x = np.concatenate((rescale(y.copy(), *bounds), [18.,-80.]))

    ina = give_me_ina(filename_abs)
    ina.run(S.values.copy(), x,
            t0, v0, initial_state_len,
            initial_state_S.values, initial_state_A.values)

    S0 = initial_state_S.values[-1]

    n_sections = 20
    #split_indices = [0,635 ,1315 ,1995 ,2675 ,3355 ,4035 ,4715 ,5395 ,6075 ,6755 ,7435 ,8115 ,8795 ,9475 ,10155 ,10835 ,11515 ,12195 ,12875 ,13555 ]
    split_indices = np.linspace(0, len(v_all), n_sections + 1).astype(int)

    for k in range(n_sections):
        start, end = split_indices[k], split_indices[k+1]
        v = v_all[start:end]
        t1 = t[start:end] - t[start]
        #print(t1[0])
        len_one_step = split_indices[k+1] - split_indices[k]
        status = ina.run(S0.copy(), x.copy(),
                         t1, v, len_one_step,
                         output_S.values[start:end], output_A.values[start:end])


    #status = ina.run(S0.copy(), x.copy(),
    #                     t, v, len(t),
    #                     output_S.values, output_A.values)
    I_out = output_S.I_out.copy()
    if kwargs.get('interpolate', False):
        time = np.arange(0, t[-1], 5e-5)
        I_out = interp1d(t, output_S.I_out.copy())(time)
        I_out = np.concatenate([I_out, I_out[-1:]])
        #print(time)
    return I_out

def give_me_ina(filename):

    ina = ctypes.CDLL(filename)

    # void initialize_states_default(double *STATES)
    ina.initialize_states_default.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    ]
    ina.initialize_states_default.restype = ctypes.c_void_p


    # void compute_rates(const double time,  double *STATES, double *CONSTANTS,  double *ALGEBRAIC, double *RATES)
    ina.compute_rates.argtypes = [
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    ]
    ina.compute_rates.restype = ctypes.c_void_p


    # void compute_algebraic(const double time,  double *STATES, double *CONSTANTS,  double *ALGEBRAIC)
    ina.compute_algebraic.argtypes = [
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    ]
    ina.compute_algebraic.restype = ctypes.c_void_p


    # int run(double *S, double *C,
    #         double *time_array, double *voltage_command_array, int array_length,
    #         double *output_S, double *output_A)
    ina.run.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
    ]
    ina.run.restype = ctypes.c_int
    return ina
def loss(y, *args):
    kwargs = args[-1]

    data = args[0]


    sample_weight = kwargs.get('sample_weight', None)

    I_out = calculate_full_trace(y, *args)
    if np.any(np.isnan(I_out)):
        return np.inf
    if np.any(np.isinf(I_out)):
        return np.inf

    return MSE(data, I_out, sample_weight=sample_weight)
