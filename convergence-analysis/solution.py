import numpy as np
import matplotlib.pyplot as plt
import os

L = 1.0
pi = 3.14159265358979323846
M = 100     # Number of time iterations

dts = ['0.00000144', '0.00000256', '0.00000400'] 
dxs_1st = ['0.006000', '0.008000', '0.010000']          # alpha = 0.04 (dt = a*dx^2)
dxs_2nd = ['0.000144', '0.000256', '0.000400']          # alpha = 0.01 (dt = a*dx)

thetas = ['0.50']

real_type = 'double'

def run_all_simulations():
    # Compile (sm_80 for A100-Ampere; sm_86 for RTX3050-Ampere; sm_89 for RTX 4070-Ada)
    os.system('nvcc -Xcompiler -fopenmp -lpthread -lcusparse convergence.cu -o convergence -O3 -w')

    for i in range(len(dts)):
        dxs = [dxs_1st[i], dxs_2nd[i]]
        dt = dts[i]

        for theta in thetas:
            for dx in dxs:
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
            dxs = [dxs_1st[index], dxs_2nd[index]]
            dt = dts[index]
            data[theta][dt] = {}
            for dx in dxs:
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
                error = np.linalg.norm(np.array(simulation_minus_solution)) / np.sqrt(number_of_points)
                data[theta][dt][dx]['error'] = error
                
                print(f'Error for theta = {theta}, dx = {dx} and dt = {dt}: {error}')
                
                
    return data

# Function to plot the convergence analysis
def plot_convergence(data):
    for theta in thetas:
        errors_1st = []
        errors_2nd = []
        for dt in dts:
            errors_1st.append(data[theta][dt][dxs_1st[dts.index(dt)]]['error'])
            errors_2nd.append(data[theta][dt][dxs_2nd[dts.index(dt)]]['error'])

        plt.loglog([float(dt) for dt in dts], errors_1st, '-o', label=f'1st Order')
        plt.loglog([float(dt) for dt in dts], errors_2nd, '-o', label=f'2nd Order')
        plt.xlabel('dt')
        plt.ylabel('Error')
        plt.title(f'Convergence Analysis - Theta = {theta}')
        plt.legend()
        plt.savefig(f'./convergence-analysis.png')
        plt.show()

# Function to calculate the slope of the convergence analysis
def calculate_slope(data):
    for theta in thetas:
        errors_1st = []
        errors_2nd = []
        for dt in dts:
            errors_1st.append(data[theta][dt][dxs_1st[dts.index(dt)]]['error'])
            errors_2nd.append(data[theta][dt][dxs_2nd[dts.index(dt)]]['error'])

        slope_1st = np.log(errors_1st[-1] / errors_1st[0]) / np.log(float(dts[-1]) / float(dts[0]))
        slope_2nd = np.log(errors_2nd[-1] / errors_2nd[0]) / np.log(float(dts[-1]) / float(dts[0]))
        
        print(f'Slope for 1st Order: {slope_1st}')
        print(f'Slope for 2nd Order: {slope_2nd}')
        
# run_all_simulations()
data = read_files()
plot_convergence(data)
calculate_slope(data)