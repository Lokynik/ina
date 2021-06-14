import numpy as np
import matplotlib.pyplot  as plt

from numba import njit

import ctypes 

def load_ctypes_object(filename_so):

    ctypes_object = ctypes.CDLL(filename_so)
    ctypes_object.calculate_circle.restype = ctypes.c_void_p

    ctypes_object.calculate_circle.argtypes = [ctypes.c_int, ctypes.c_int,ctypes.c_int,
                                      np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                                      np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                                      ctypes.c_double,
                                      np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                                      np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                                      np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                                      np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                                      np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                                      np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                                      np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                                      np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                                      np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                                      np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')]
    
    return ctypes_object

@njit
def rush_larsen_easy_numba_helper(x, y, c):
    for i in range(1, len(x)):
        x[i] = y[i-1] + (x[i-1] - y[i-1]) * np.exp(c)
        

def calculate_I_out(x, *args):
    y = x.copy()
    kwargs = args[-1]
    
    t = kwargs['t']
    dt = t[1] - t[0]
    
    n = int(5e-5 / dt)
    
    v_list = kwargs['v_list']
    k_list = kwargs['k_list'] * n
    
    if kwargs.get('log_scale', False):
        y[:-1] = np.exp(y[:-1])
        assert np.all(y[:-1] > 0) 

    count = np.zeros_like(t)
    count[k_list] = 1
    count = np.cumsum(count) 
    
    
    v_c = np.zeros_like(t)
    v_c = v_list[count.astype(int)]

    v_rev = 18.0

    #
    v_cp = np.zeros_like(t)
    v_p = np.zeros_like(t)
    v_m = np.zeros_like(t)
    v_comp = np.zeros_like(t)
        
    m = np.zeros_like(t)
    h = np.zeros_like(t)
    j = np.zeros_like(t)
    
    I_leak = np.zeros_like(t)
    I_Na = np.zeros_like(t)
    
    v_p[0] = -80.0
    v_m[0] = -80.0
    v_comp[0] =  - 80.0
    v_cp[0] =  - 80.0

    
    m[0] = 0.0
    h[0] = 1.0

    n_start = 15000
    N = len(t)
    I_error = np.zeros(len(t)//n)
    len_one_trace = 25000
    try:
        calculate_circle = kwargs['calculate_circle']
        calculate_circle(N, n_start,len_one_trace, t, v_c, v_rev,v_cp,v_p,v_m,v_comp ,m, h, j,I_leak, I_Na,y)
    
    except ZeroDivisionError:
        I_error+=1e100
        return I_error
    
    c_p = y[0]
    c_m = y[1]
    x_c_comp = y[-4]

    I_c = 1e9 * c_m * (np.diff(v_m) / dt)[::n]
    I_p = 1e9 * c_p * (np.diff(v_p) / dt)[::n]
    I_comp=1e9 * x_c_comp* c_m * (np.diff(v_comp) / dt)[::n]
    
    if len(I_c) != len(I_Na[::n]):
        I_c = np.concatenate((I_c,I_c[-1:]))
        I_p = np.concatenate((I_p,I_p[-1:]))
        I_comp = np.concatenate((I_comp,I_comp[-1:]))
    
    tau_z = 5e-4 # 1e-12 * 5e8
    
    I_in = I_c  + I_leak[::n] + I_Na[::n] + I_p - I_comp
    del I_c, I_leak,I_Na,I_p,I_comp, v_cp, v_p, v_m, v_comp
    #gc.collect()
    I_out = np.zeros_like(I_in)

    I_out[0] = I_in[0]
    rush_larsen_easy_numba_helper(I_out, I_in, - dt * n / tau_z)
    #euler_numba_helper(I_out,I_in,(dt / tau_z))  
    del I_in
    if kwargs.get('graph', True):
        #plt.plot(V_m_list, label = 'command')
        plt.figure()
        
        plt.plot(v_c, label = 'command')
        #plt.plot(v_comp, label = 'compensated')
        plt.plot(v_cp, label = 'prediction')
        plt.plot(v_p, label = 'pipette', ls = '--')
        plt.plot(v_m, label = 'membrane',ls = '-.')
        plt.legend()
        v_graph = np.arange(-95,35)

        plt.figure()
        plt.plot(I_c, label = 'I_c')
        plt.plot(I_p, label = 'I_p')
        plt.plot(I_leak, label = 'I_leak')
        plt.plot(I_Na, label = 'I_Na',ls = '-.')
        plt.legend()
        

        plt.figure()
        #plt.plot(I_in, label = 'I_in')
        plt.plot(I_out, label = 'I_out')
        plt.legend() 


    return I_out
