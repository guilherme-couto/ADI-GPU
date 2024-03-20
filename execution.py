import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Possible parameters
dxs = ['0.005', '0.008', '0.010','0.016', '0.020']
cell_models = ['AFHN']
numbers_threads = [1, 2, 4, 6]
dts = ['0.020']
methods = ['theta-ADI']
modes = ['All-GPU', 'CPU']
thetas = ['0.50', '0.67', '1.00']
number_of_executions = 5

dxs = ['0.005']
cell_models = ['TT2']
numbers_threads = [6]
dts = ['0.020']
methods = ['MOSI-2-ADI']#, 'MOSI-ADI']
modes = ['All-GPU']
thetas = ['0.50']
number_of_executions = 1

gpu_threads = [max(numbers_threads)]
cpu_threads = numbers_threads