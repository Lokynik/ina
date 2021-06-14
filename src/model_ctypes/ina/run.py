import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import os
import ctypes


dirname = './'
filename_so = os.path.join(dirname, 'ina.so')

filename_so_abs = os.path.abspath(filename_so)

ina = ctypes.CDLL(filename_so_abs)


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

legend_constants = pd.read_csv(os.path.join(dirname, "legend_constants.csv"), index_col='name')['value']
legend_states = pd.read_csv(os.path.join(dirname, "legend_states.csv"), index_col='name')['value']
legend_algebraic = pd.read_csv(os.path.join(dirname, "legend_algebraic.csv"), index_col='name')['value']

S = legend_states.copy()
R = S.copy() * 0
C = legend_constants.copy()
A = legend_algebraic.copy()

t = np.arange(0, 1e-3, 5e-5)
v_c = np.full_like(t, -80.0)

# df_protocol = pd.DataFrame(np.vstack([t, v_c]).T, columns=['t', 'v_c'])

filename_protocol = "../../../data/protocols/protocol.csv"
df_protocol = pd.read_csv(filename_protocol)
df_protocol

output_len = len(df_protocol)

output_S = pd.DataFrame(np.zeros((output_len, len(S))), columns=legend_states.index)
output_A = pd.DataFrame(np.zeros((output_len, len(A))), columns=legend_algebraic.index)

ina.run(S.values.copy(), C.values.copy(),
        df_protocol['t'].values, df_protocol['v'].values, output_len,
        output_S.values, output_A.values)
