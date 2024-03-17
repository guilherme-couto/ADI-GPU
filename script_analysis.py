from execution import *

# Program to read the files from script-simulation-files folder, calculate speedups for each mode and write the informations and results to a csv file

for cell_model in cell_models:
    for mode in modes:
        if mode == 'All-GPU':
            numbers_threads = gpu_threads
        else:
            numbers_threads = cpu_threads
        output_file = open(f'analysis-{mode}.csv', 'w')
        
        for method in methods:
            for dx in dxs:
                for number_of_threads in numbers_threads:
                    for theta in thetas:
                        output_file.write('dx,number_of_threads,theta\n')
                        output_file.write(f'{dx},{number_of_threads},{theta}\n')
                        output_file.write('exec_number,')
                        if mode == 'All-GPU':
                            output_file.write('S1 velocity,ODE time,PDE time,Total time,1st Thomas,2nd Thomas,Transpose,1st RHS,2nd RHS,Total memory copy time\n')
                        elif mode == 'CPU':
                            output_file.write('S1 velocity,ODE time,PDE time,Total time,1st Thomas,2nd Thomas\n')
                    
                        for dt in dts:                              
                            s1_velocities = []
                            odes = []
                            pdes = []
                            totals = []
                            thomas_1s = []
                            thomas_2s = []
                            transposes = []
                            rhs_1s = []
                            rhs_2s = []
                            memory_copys = []
                            
                            for exec_number in range(number_of_executions):
                                filename =f'script-simulation-files/double/{cell_model}/{mode}/{method}/{dx}/{number_of_threads}/{exec_number}/infos-{number_of_threads}-{dt}-{theta}.txt'
                                
                                if mode == 'All-GPU':
                                    # Read the file looking for 'S1 velocity:', 'ODE execution time:', 'PDE execution time:', 'Total execution time with writings:',
                                    # '1st Thomas algorithm time:', '2nd Thomas algorithm time:', 'Transpose time:'
                                    # '1st RHS preparation time:', '2nd RHS preparation time:', 'Total memory copy time:'
                                    s1_velocity, ode, pde, total, thomas_1, thomas_2, transpose, rhs_1, rhs_2, memory_copy = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                    with open(filename, 'r') as file:
                                        lines = file.readlines()
                                        for line in lines:
                                            if 'S1 velocity:' in line:
                                                s1_velocity = float(line.split(' ')[-2])
                                                s1_velocities.append(s1_velocity)
                                            if 'ODE execution time:' in line:
                                                ode = float(line.split(' ')[-2])
                                                odes.append(ode)
                                            if 'PDE execution time:' in line:
                                                pde = float(line.split(' ')[-2])
                                                pdes.append(pde)
                                            if 'Total execution time with writings:' in line:
                                                total = float(line.split(' ')[-2])
                                                totals.append(total)
                                            if '1st Thomas algorithm time:' in line:
                                                thomas_1 = float(line.split(' ')[-2])
                                                thomas_1s.append(thomas_1)
                                            if '2nd Thomas algorithm time:' in line:
                                                thomas_2 = float(line.split(' ')[-2])
                                                thomas_2s.append(thomas_2)
                                            if 'Transpose time:' in line:
                                                transpose = float(line.split(' ')[-2])
                                                transposes.append(transpose)
                                            if '1st RHS preparation time:' in line:
                                                rhs_1 = float(line.split(' ')[-2])
                                                rhs_1s.append(rhs_1)
                                            if '2nd RHS preparation time:' in line:
                                                rhs_2 = float(line.split(' ')[-2])
                                                rhs_2s.append(rhs_2)
                                            if 'Total memory copy time:' in line:
                                                memory_copy = float(line.split(' ')[-2])
                                                memory_copys.append(memory_copy)
                                                            
                                    output_file.write(f'{exec_number},{s1_velocity},{ode},{pde},{total},{thomas_1},{thomas_2},{transpose},{rhs_1},{rhs_2},{memory_copy}\n')
                                
                                elif mode == 'CPU':
                                    # Read the file looking for 'S1 velocity:', 'ODE execution time:', 'PDE execution time:', 'Total execution time with writings:',
                                    # '1st Thomas algorithm (+ transpose) execution time:', '2nd Thomas algorithm execution time:'
                                    s1_velocity, ode, pde, total, thomas_1, thomas_2 = 0, 0, 0, 0, 0, 0
                                    with open(filename, 'r') as file:
                                        lines = file.readlines()
                                        for line in lines:
                                            if 'S1 velocity:' in line:
                                                s1_velocity = float(line.split(' ')[-2])
                                                s1_velocities.append(s1_velocity)
                                            if 'ODE execution time:' in line:
                                                ode = float(line.split(' ')[-2])
                                                odes.append(ode)
                                            if 'PDE execution time:' in line:
                                                pde = float(line.split(' ')[-2])
                                                pdes.append(pde)
                                            if 'Total execution time with writings:' in line:
                                                total = float(line.split(' ')[-2])
                                                totals.append(total)
                                            if '1st Thomas algorithm (+ transpose) execution time:' in line:
                                                thomas_1 = float(line.split(' ')[-2])
                                                thomas_1s.append(thomas_1)
                                            if '2nd Thomas algorithm execution time:' in line:
                                                thomas_2 = float(line.split(' ')[-2])
                                                thomas_2s.append(thomas_2)
                                    
                                    output_file.write(f'{exec_number},{s1_velocity},{ode},{pde},{total},{thomas_1},{thomas_2}\n')
                            
                            # Write average values
                            output_file.write('Average,')
                            if mode == 'All-GPU':
                                output_file.write(f'{(np.mean(s1_velocities)):.6f},{(np.mean(odes)):.6f},{(np.mean(pdes)):.6f},{(np.mean(totals)):.6f},{(np.mean(thomas_1s)):.6f},{(np.mean(thomas_2s)):.6f},{(np.mean(transposes)):.6f},{(np.mean(rhs_1s)):.6f},{(np.mean(rhs_2s)):.6f},{(np.mean(memory_copys)):.6f}\n') 
                            elif mode == 'CPU':
                                output_file.write(f'{(np.mean(s1_velocities)):.6f},{(np.mean(odes)):.6f},{(np.mean(pdes)):.6f},{(np.mean(totals)):.6f},{(np.mean(thomas_1s)):.6f},{(np.mean(thomas_2s)):.6f}\n')
                            output_file.write('\n')

        output_file.close()
        print(f'File analysis-{mode}.csv created')

print('Start calculating speedups')

# Read csv files created, catch the average values and calculate the speedups. Write the results to a new csv file where the two modes are compared
output_file = open('speedups.csv', 'w')

for mode in modes:
    if mode == 'All-GPU':
        numbers_threads = gpu_threads
    else:
        numbers_threads = cpu_threads
    for dx in dxs:
        for theta in thetas:
            output_file.write('\n')
            output_file.write('dx,theta,mode\n')
            output_file.write(f'{dx},{theta},{mode}\n')
            output_file.write('number_of_threads,')
            if mode == 'All-GPU':
                output_file.write('ODE time,PDE time,Total time,1st Thomas,2nd Thomas,Transpose,1st RHS,2nd RHS,Total memory copy time\n')
            elif mode == 'CPU':
                output_file.write('ODE time,PDE time,Total time,1st Thomas,2nd Thomas\n')
            
            odes, pdes, totals, thomas_1s, thomas_2s, transposes, rhs_1s, rhs_2s, memory_copys = [], [], [], [], [], [], [], [], []
            for number_of_threads in numbers_threads:
                filename = f'analysis-{mode}.csv'
                with open(filename, 'r') as file:
                    lines = file.readlines()
                    for i in range(len(lines)):
                        line = lines[i]
                        if f'{dx},{number_of_threads},{theta}' in line:
                            for j in range(i+1, i+3+number_of_executions):
                                line = lines[j]
                                if 'Average' in line:
                                    values = line.split(',')
                                    if mode == 'All-GPU':
                                        odes.append(float(values[2]))
                                        pdes.append(float(values[3]))
                                        totals.append(float(values[4]))
                                        thomas_1s.append(float(values[5]))
                                        thomas_2s.append(float(values[6]))
                                        transposes.append(float(values[7]))
                                        rhs_1s.append(float(values[8]))
                                        rhs_2s.append(float(values[9]))
                                        memory_copys.append(float(values[10]))
                                    elif mode == 'CPU':
                                        odes.append(float(values[2]))
                                        pdes.append(float(values[3]))
                                        totals.append(float(values[4]))
                                        thomas_1s.append(float(values[5]))
                                        thomas_2s.append(float(values[6]))
            
            for i in range(len(numbers_threads)):
                output_file.write(f'{numbers_threads[i]},')
                if mode == 'All-GPU':
                    output_file.write(f'{(odes[0]/odes[i]):.6f},{(pdes[0]/pdes[i]):.6f},{(totals[0]/totals[i]):.6f},{(thomas_1s[0]/thomas_1s[i]):.6f},{(thomas_2s[0]/thomas_2s[i]):.6f},{(transposes[0]/transposes[i]):.6f},{(rhs_1s[0]/rhs_1s[i]):.6f},{(rhs_2s[0]/rhs_2s[i]):.6f},{(memory_copys[0]/memory_copys[i]):.6f}\n')
                elif mode == 'CPU':
                    output_file.write(f'{(odes[0]/odes[i]):.6f},{(pdes[0]/pdes[i]):.6f},{(totals[0]/totals[i]):.6f},{(thomas_1s[0]/thomas_1s[i]):.6f},{(thomas_2s[0]/thomas_2s[i]):.6f}\n')                
                
output_file.close()
print('File speedups.csv created')  

                    
        
         