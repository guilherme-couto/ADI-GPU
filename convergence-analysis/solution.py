import numpy as np
import matplotlib.pyplot as plt
import os

L = 2.0
pi = 3.14159265358979323846
M = 100     # Number of time iterations

dxs = ['0.010', '0.012', '0.014', '0.016', '0.018', '0.020']
dts_1st = ['0.0100', '0.0144', '0.0196', '0.0256', '0.0324', '0.0400']
dts_2nd = ['0.0100', '0.0120', '0.0140', '0.0160', '0.0180', '0.0200']

thetas = [0.5]

def run_all_simulations():
    # Compile (sm_80 for A100-Ampere; sm_86 for RTX3050-Ampere; sm_89 for RTX 4070-Ada)
    os.system('nvcc -Xcompiler -fopenmp -lpthread -lcusparse convergence.cu -o convergence -O3')

    for i in range(len(dxs)):
        dts = [float(dts_1st[i]), float(dts_2nd[i])]
        dx = float(dxs[i])

        for theta in thetas:
            for dt in dts:
                simulation_line = f'./convergence theta-ADI {dt} {dx} {theta}'
                print(f'Executing {simulation_line}...')
                os.system(simulation_line)
                print('Simulation finished!\n')

def solution(x, y, t):
    return np.sin(pi*t) * np.sin(pi*x/L) * np.sin(pi*y/L)

#run_all_simulations()


