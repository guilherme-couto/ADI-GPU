import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Possible parameters
dxs = ['0.010', '0.020', '0.002', '0.001']
cell_models = ['AFHN']
numbers_threads = [6]
dts = [0.02]
methods = ['MOSI-ADI', 'SSI-ADI']
modes = ['All-GPU']