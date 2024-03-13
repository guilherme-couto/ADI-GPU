from execution import *

# Compile (sm_80 for A100-Ampere; sm_86 for RTX3050-Ampere; sm_89 for RTX 4070-Ada)
os.system(f'nvcc -Xcompiler -fopenmp -lpthread -lcusparse main.cu -o main -O3 -arch=sm_89 -w')

# Check arguments
for dx in dxs:
    for cell_model in cell_models:
        for num_threads in numbers_threads:
            for method in methods:
                for dt in dts:
                    for mode in modes:
                        if method == 'theta-ADI':
                            for theta in thetas:
                                for exec_number in range(number_of_executions):
                                    execution_line = ''
                                    if cell_model.find('Fibro') != -1:
                                        execution_line = f'./main {num_threads} {dt} {dx} {mode} {method} 1 {theta} 0 {exec_number}'
                                    else:
                                        execution_line = f'./main {num_threads} {dt} {dx} {mode} {method} 0 {theta} 0 {exec_number}'
                                    
                                    print(f'Executing {execution_line}')
                                    os.system(f'{execution_line}')
                                    print(f'Simultation {execution_line} finished!\n')
                        else:
                            for exec_number in range(number_of_executions):
                                execution_line = ''
                                if cell_model.find('Fibro') != -1:
                                    execution_line = f'./main {num_threads} {dt} {dx} {mode} {method} 1 0 0 {exec_number}'
                                else:
                                    execution_line = f'./main {num_threads} {dt} {dx} {mode} {method} 0 0 0 {exec_number}'
                                
                                print(f'Executing {execution_line}')
                                os.system(f'{execution_line}')
                                print(f'Simultation {execution_line} finished!\n')
                            
                    
