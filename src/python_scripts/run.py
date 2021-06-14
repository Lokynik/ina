import numpy as np
import pandas as pd
import scipy.optimize as scop
#import matplotlib.pyplot as plt
import os
import ctypes
import sys
from sklearn.metrics import mean_squared_error as MSE
from cmaes import CMA

sys.path.append("../src/python_scripts/")
from functions import scale, rescale, calculate_full_trace, give_me_ina,loss

dirname = '../model_ctypes/ina/'


legend_constants = pd.read_csv(os.path.join(dirname, "legend_constants.csv"), index_col='name')
legend_states = pd.read_csv(os.path.join(dirname, "legend_states.csv"), index_col='name')['value']
legend_algebraic = pd.read_csv(os.path.join(dirname, "legend_algebraic.csv"), index_col='name')['value']

S = legend_states.copy()
R = S.copy() * 0
C = legend_constants.copy()
A = legend_algebraic.copy()

filename_protocol = "../../data/protocols/protocol_79.csv"
df_protocol = pd.read_csv(filename_protocol)

##KWARGS WORKING
t0 = np.arange(0, 0.25, 5e-5)
v0 = np.full_like(t0, -80.0)
initial_state_len = len(t0)

t = df_protocol['t'].values
v = df_protocol['v'].values
output_len = len(t)
initial_state_S = pd.DataFrame(np.zeros((initial_state_len, len(S))), columns=legend_states.index)
initial_state_A = pd.DataFrame(np.zeros((initial_state_len, len(A))), columns=legend_algebraic.index)

output_S = pd.DataFrame(np.zeros((output_len, len(S))), columns=legend_states.index)
output_A = pd.DataFrame(np.zeros((output_len, len(A))), columns=legend_algebraic.index)


bounds = np.array([[ 1e-14 , 1.0000000000000001e-11 ],
    [ 1e-12 , 1e-10 ],
    [ 100.0 , 100000.0 ],
    [ 1.0 , 1000.0 ],
    [ 1.0 , 100.0 ],
    [ 0.1 , 100.0 ],
    [ 0.01 , 10.0 ],
    [ 10.0 , 10000.0 ],
    [ 1.0 , 100.0 ],
    [ 1.0 , 100.0 ],
    [ 0.01 , 10.0 ],
    [ 1000.0 , 100000.0 ],
    [ 0.1 , 100.0 ],
    [ 10.0 , 1000.0 ],
    [ 1e-05 , 0.01 ],
    [ 1000000.0 , 1000000000.0 ],
    [ 1000.0 , 100000000.0 ],
    [ 1000.0 , 10000000.0 ],
    [ 0.01 , 1000.0 ],
    [ 1e-05 , 0.001 ],
    [ 1.0 , 100.0 ],
    [ 1.0 , 100.0 ],
    [ 0.1 , 10.0 ],
    [ 0.1 , 10.0 ],
    [ 0.01 , 1.1 ],
    [ 0.01 , 1.1 ],
    [ 0.01 , 1.0 ],
    [ -25.0 , 25.0 ],
    #[ -50.0 , 50.0 ],
    #[ -100.0 , 100.0 ]
         ])


filename = 'ina.so'
filename_so = os.path.join(dirname, filename)
filename_abs = os.path.abspath(filename_so)
kwargs = dict(S = S,
              t0 = t0,
              v0 = v0,
              initial_state_S = initial_state_S,
              initial_state_A = initial_state_A,
              initial_state_len = initial_state_len,
              t = df_protocol['t'].values,
              v = df_protocol['v'].values,
              filename_abs = filename_abs,
              output_S = output_S,
              output_A = output_A,
              bounds = bounds.T
             )

data = pd.read_csv('../../data/training/2020_12_19_0035 I-V INa 11,65 pF.atf' ,delimiter= '\t', header=None, skiprows = 11)
exp_data = np.concatenate([data[k] for k in range(1,21)])

x0 = scale(C.value.values[:-2],*bounds.T)
scale_bounds = np.array([[0.,1.] for k in range(len(bounds))])
data = calculate_full_trace(x0, kwargs)
#res = scop.differential_evolution(loss, bounds=scale_bounds,args=(exp_data, kwargs), maxiter=10, popsize = 10, workers = 2, seed = 42, tol = 0.1, disp = True)
#print('res = ',res.x, 'loss = ',loss(res.x, exp_data, kwargs))
if __name__ == "__main__":
    mean = x0+0.001
    sigma = 0.5
    optimizer = CMA(mean=mean, sigma=sigma, bounds=scale_bounds, seed=0, population_size=50)
    for generation in range(50):

        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = loss(x,data, kwargs)
            solutions.append((x, value))
            print(f"#{generation} {value} ")

        optimizer.tell(solutions)

#res = calculate_full_trace(scale(C.value.values[:-2],*bounds.T), kwargs)
#print(res)
result = pd.DataFrame(solutions, columns=['x', 'loss'])
name = 'result_1.csv'
result.to_csv(f"../../data/results/{name}")
print('OK')
