import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Possible parameters
dxs = ['0.010', '0.020']
cell_models = ['AFHN']
numbers_threads = [1, 2, 4, 6]
dts = [0.02]
methods = ['theta-ADI']
modes = ['All-GPU', 'CPU']
thetas = [0.5, 2/3, 1.0]
number_of_executions = 5