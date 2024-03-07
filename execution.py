import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Possible parameters
dxs = ['0.010', '0.020', '0.002', '0.001']
cell_models = ['AFHN']
numbers_threads = [4]
dts = [0.02]
methods = ['theta-ADI']
modes = ['All-GPU']
thetas = [0.5, 1.0]
dxs = ['0.010', '0.001']