#import numpy as np
#import matplotlib.pyplot as plt
import os
#import math

# Possible parameters
dxs = ['0.020', '0.010', '0.002', '0.001']
cell_models = ['AFHN']
numbers_threads = [6, 8, 16, 32, 64, 128]
dts = [0.02]
methods = ['OS-ADI']
modes = ['GPU', 'CPU']