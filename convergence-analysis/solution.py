import numpy as np
import matplotlib.pyplot as plt
import os

L = 1.0
T = 1.0
pi = 3.14159265358979323846
real_type = 'double'

def run_all_simulations():
    # Compile (sm_80 for A100-Ampere; sm_86 for RTX3050-Ampere; sm_89 for RTX 4070-Ada)
    os.system('nvcc -Xcompiler -fopenmp -lpthread -lcusparse convergence.cu -o convergence -O3 -w')

    for i in range(len(dts)):
        dx = dxs_2nd[i]
        dt = dts[i]
        tts = []
        for method in methods:
            if method != 'theta-ADI':
                tts = ['0.00']
            else:
                tts = thetas 
            for theta in tts:
                simulation_line = f'./convergence {method} {dt} {dx} {theta}'
                print(f'Executing {simulation_line}...')
                os.system(simulation_line)
                print('Simulation finished!\n')

def solution(x, y, t):
    return np.sin(pi*t) * np.sin(pi*x/L) * np.sin(pi*y/L)

# Function to read files and save in a dictionary
def read_files():
    data = {}
    for method in methods:
        data[method] = {}
        if method != 'theta-ADI':
            tts = ['0.00']
        else:
            tts = thetas 
        for theta in tts:
            data[method][theta] = {}
            for index in range(len(dts)):
                dx = dxs_2nd[index]
                dt = dts[index]
                data[method][theta][dt] = {}
                data[method][theta][dt][dx] = {}
                simulation_minus_solution = []
                
                # Save data and analytical solution
                filename = f'./simulation-files/{real_type}/AFHN/{method}/last-{dt}-{dx}.txt'
                if method == 'theta-ADI':
                    filename = f'./simulation-files/{real_type}/AFHN/{method}/{theta}/last-{dt}-{dx}.txt'
                i = 0
                for line in open(filename, 'r'):
                    line = line.split()
                    for j in range(len(line)):
                        simulation_minus_solution.append(float(line[j]) - solution(j*float(dx), i*float(dx), T))
                    i += 1
                
                # Calculate the error with norm 2
                number_of_points = len(simulation_minus_solution)
                error = (np.linalg.norm(np.array(simulation_minus_solution))) / (np.sqrt(number_of_points)) # RMSE
                data[method][theta][dt][dx]['error'] = error
                if method != 'theta-ADI':
                    print(f'Error for method = {method}, dx = {dx} and dt = {dt}: {error}') 
                    analysis_file.write(f'Error for method = {method}, dx = {dx} and dt = {dt}: {error}\n')
                else:
                    print(f'Error for method = {method}, theta = {theta}, dx = {dx} and dt = {dt}: {error}') 
                    analysis_file.write(f'Error for method = {method}, theta = {theta}, dx = {dx} and dt = {dt}: {error}\n')
            analysis_file.write(f'\n')
    return data

# Function to plot the convergence analysis
def plot_convergence(data, plot_path):
    for method in methods:
        if method != 'theta-ADI':
            tts = ['0.00']
        else:
            tts = thetas 
        for theta in tts:
            errors_2nd = []
            for dt in dts:
                errors_2nd.append(data[method][theta][dt][dxs_2nd[dts.index(dt)]]['error'])
            if method != 'theta-ADI':
                plt.loglog([float(dt) for dt in dts], errors_2nd, '-x', label=f'{method}')
            else:
                plt.loglog([float(dt) for dt in dts], errors_2nd, '-o', label=f'{method}-{theta}')
    
    plt.xlabel('dt')
    plt.ylabel('Error')
    plt.title(f'Convergence Analysis - 2nd Order (a = {(alpha):.3f})')
    plt.legend()
    plt.savefig(plot_path)
    #plt.show()
    plt.close()

# Function to calculate the slope of the convergence analysis
def calculate_slope(data, alpha):
    print(f'Slopes for 2nd Order (a = {(alpha):.3f})')
    for method in methods:
        if method != 'theta-ADI':
            tts = ['0.00']
        else:
            tts = thetas 
        for theta in tts:
            errors_2nd = []
            slopes = []
            for dt in dts:
                errors_2nd.append(data[method][theta][dt][dxs_2nd[dts.index(dt)]]['error'])
            for index in range(1, len(errors_2nd)):
                slopes.append(np.log(errors_2nd[index] / errors_2nd[index-1]) / np.log(float(dts[index]) / float(dts[index-1])))

            slope_2nd = np.log(errors_2nd[-1] / errors_2nd[0]) / np.log(float(dts[-1]) / float(dts[0]))
            if method != 'theta-ADI':
                print(f'For {method}: {slope_2nd} (mean: {np.mean(np.array(slopes))})')
                analysis_file.write(f'For {method}: {slope_2nd} (mean: {np.mean(np.array(slopes))})\n')
            else:
                print(f'For {method} {theta}: {slope_2nd} (mean: {np.mean(np.array(slopes))})')
                analysis_file.write(f'For {method} {theta}: {slope_2nd} (mean: {np.mean(np.array(slopes))})\n')
        

# 1st order (dt = a*dx²)
# 2nd order (dt = a*dx)
alpha = 0.1
thetas = ['0.50', '1.00']
methods = ['theta-ADI']

# Create directories
if not os.path.exists(f'./simulation-files/simulation-graphs'):
    os.makedirs(f'./simulation-files/simulation-graphs')
if not os.path.exists(f'./simulation-files/simulation-analysis'):
    os.makedirs(f'./simulation-files/simulation-analysis')

# values = np.linspace(0.0001, 0.01, 10)

# start = 100
# end = 1000
# values = []

# for i in np.arange(0.0001, 0.001, 0.00005):
#     value = float(f'{i:.5f}')
#     if abs(1.0/value - round(1.0/value)) < 1e-9:
#         values.append(value)
#     if len(values) == 10:
#         break

values = [0.000100, 0.000125, 0.000160, 0.000200, 0.000250, 0.000400, 0.000500, 0.000625, 0.000800, 0.001000]
dts = [f'{value:.8f}' for value in values]
# dxs_2nd = [f'{(value / alpha):.6f}' for value in values]

# values = [0.000100, 0.000125, 0.000160, 0.000200, 0.000250, 0.000400, 0.000500, 0.000625, 0.000800, 0.001000]
dxs_2nd = [f'{value:.6f}' for value in values]

# run_all_simulations()

analysis_file = open(f'./simulation-files/simulation-analysis/analysis.txt', 'w')
# data = read_files()
plot_path = f'./simulation-files/simulation-graphs/convergence-analysis.png'
# plot_convergence(data, plot_path)

# analysis_file.write('\n\n')

# calculate_slope(data, alpha)

# analysis_file.close()

os.system('nvcc -Xcompiler -fopenmp -lpthread -lcusparse convergence.cu -o convergence -O3 -w')
terminal_outputs = []
dts = [0.01, 0.005, 0.001, 0.0005, 0.0001]
for dt in [f'{value:.8f}' for value in dts]:
    dxs = [0.0008, 0.000625, 0.0005, 0.0004, 0.00025, 0.0002, 0.0001, 0.00005]
    for dx in [f'{value:.6f}' for value in dxs]:
        simulation_line = f'./convergence theta-ADI {dt} {dx} 0.50'
        print(f'Executing {simulation_line}...')
        os.system(simulation_line)
        print('Simulation finished!\n')
        # Save in the terminal output the value of the first element of the output file
        output_file = open(f'./simulation-files/double/AFHN/theta-ADI/0.50/last-{dt}-{dx}.txt', 'r')
        first_element = output_file.readline().split()[0]
        terminal_outputs.append(f'For dt = {dt} and dx = {dx}, the first element is {first_element}')
        output_file.close()

# Print the terminal outputs
for output in terminal_outputs:
    print(output)