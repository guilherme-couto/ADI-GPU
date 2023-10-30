import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Possible parameters
dxs = ['0.020', '0.010', '0.002', '0.001']
cell_models = ['AFHN']
numbers_threads = [4, 6]
dts = [0.02, 0.04, 0.08]
methods = ['OS-ADI']
modes = ['CPU', 'GPU']