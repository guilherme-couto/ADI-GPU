import numpy as np
import matplotlib.pyplot as plt
import os

L = 1.0
pi = 3.14159265358979323846
M = 100     # Number of time iterations
real_type = 'double'

def run_all_simulations():
    # Compile (sm_80 for A100-Ampere; sm_86 for RTX3050-Ampere; sm_89 for RTX 4070-Ada)
    os.system('nvcc -Xcompiler -fopenmp -lpthread -lcusparse convergence.cu -o convergence -O3 -w')

    for i in range(len(dts)):
        dx = dxs_2nd[i]
        dt = dts[i]
        for theta in thetas:
            simulation_line = f'./convergence theta-ADI {dt} {dx} {theta}'
            print(f'Executing {simulation_line}...')
            os.system(simulation_line)
            print('Simulation finished!\n')

def solution(x, y, t):
    return np.sin(pi*t) * np.sin(pi*x/L) * np.sin(pi*y/L)

# Function to read files and save in a dictionary
def read_files():
    data = {}
    for theta in thetas:
        data[theta] = {}
        for index in range(len(dts)):
            dx = dxs_2nd[index]
            dt = dts[index]
            data[theta][dt] = {}
            data[theta][dt][dx] = {}
            simulation_minus_solution = []
            
            # Save data and analytical solution
            filename = f'./simulation-files/{real_type}/AFHN/theta-ADI/{theta}/last-{dt}-{dx}.txt'
            i = 0
            for line in open(filename, 'r'):
                line = line.split()
                for j in range(len(line)):
                    simulation_minus_solution.append(float(line[j]) - solution(i*float(dx), j*float(dx), (M)*float(dt)))
                i += 1
            
            # Calculate the error with norm 2
            number_of_points = len(simulation_minus_solution)
            error = np.linalg.norm(np.array(simulation_minus_solution)) / np.sqrt(number_of_points) # RMSE
            data[theta][dt][dx]['error'] = error
            print(f'Error for theta = {theta}, dx = {dx} and dt = {dt}: {error}') 
    return data

# Function to plot the convergence analysis
def plot_convergence(data, alpha):
    for theta in thetas:
        errors_2nd = []
        for dt in dts:
            errors_2nd.append(data[theta][dt][dxs_2nd[dts.index(dt)]]['error'])
        plt.loglog([float(dt) for dt in dts], errors_2nd, '-o', label=f'theta {theta}')
    
    plt.xlabel('dt')
    plt.ylabel('Error')
    plt.title(f'Convergence Analysis - 2nd Order (a = {(alpha):.3f})')
    plt.legend()
    plt.savefig(f'./convergence-analysis.png')
    plt.show()

# Function to calculate the slope of the convergence analysis
def calculate_slope(data, alpha):
    print(f'Slopes for 2nd Order (a = {(alpha):.3f})')
    for theta in thetas:
        errors_2nd = []
        slopes = []
        for dt in dts:
            errors_2nd.append(data[theta][dt][dxs_2nd[dts.index(dt)]]['error'])
        for index in range(1, len(errors_2nd)):
            slopes.append(np.log(errors_2nd[index] / errors_2nd[index-1]) / np.log(float(dts[index]) / float(dts[index-1])))

        slope_2nd = np.log(errors_2nd[-1] / errors_2nd[0]) / np.log(float(dts[-1]) / float(dts[0]))
        print(f'With theta {theta}: {slope_2nd} (mean: {np.mean(np.array(slopes))})')
        

# 1st order (dt = a*dx²)
# 2nd order (dt = a*dx)
alpha = 0.06
thetas = ['0.50', '0.67', '1.00']
values = np.linspace(0.000013*6.0, 0.0001*6.0, 10)
dts = [f'{value:.8f}' for value in values]

dxs_2nd = [f'{(value / alpha):.6f}' for value in values]
# dxs_2nd = [f'{np.sqrt(value / alpha):.6f}' for value in values]
dxs_2nd = ['0.001300','0.002267', '0.003233','0.004200','0.005167','0.006133','0.007100','0.008067','0.009033','0.010000']
# dxs_2nd = ['0.000650','0.001133', '0.001617','0.002100','0.002583','0.003067','0.003550','0.004033','0.004517','0.005000']
# dxs_2nd = ['0.002600', '0.004533', '0.006467', '0.008400', '0.010333', '0.012267', '0.014200', '0.016133', '0.018067', '0.020000']
dxs_2nd = ['0.000500', '0.001000', '0.002000', '0.002500', '0.003125', '0.004000', '0.005000']
alpha = 0.001
dts = [f'{(float(dxs_2nd[i]) * alpha):.8f}' for i in range(len(dxs_2nd))]

run_all_simulations()
data = read_files()
plot_convergence(data, alpha)
calculate_slope(data, alpha)