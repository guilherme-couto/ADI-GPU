import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Possible parameters
dxs = ['0.020', '0.010', '0.002', '0.001']
numbers_threads = [6, 8, 16, 32, 64, 128]
dts = [0.02, 0.08]
methods = ['OS-ADI']
modes = ['CPU', 'GPU']

counter = 0

for dx in dxs:
    for number_threads in numbers_threads:
        for dt in dts:
            for method in methods:
                for mode in modes:
                    # Copy the template
                    os.system(f'cp base_script.pbs ./job-scripts/script{counter}.pbs')
                    
                    # Replace the parameters
                    f = open(f'./job-scripts/script{counter}.pbs', 'rt')
                    filedata = f.read()
                    filedata = filedata.replace('<job_name>', f'job{counter}')
                    filedata = filedata.replace('<dx>', dx)
                    filedata = filedata.replace('<number_threads>', str(number_threads))
                    filedata = filedata.replace('<dt>', str(dt))
                    filedata = filedata.replace('<method>', method)
                    filedata = filedata.replace('<mode>', mode)
                    f.close()
                    f = open(f'./job-scripts/script{counter}.pbs', 'wt')
                    f.write(filedata)
                    f.close()
                    counter += 1
                        