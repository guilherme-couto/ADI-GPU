#ifndef INCLUDES_H
#define INCLUDES_H

#define MAX_STRING_SIZE 100

// Sizes of shared memory tile
#define BDIMX 16
#define BDIMY 16
#define BDIMZ 4
#define BLOCK_SIZE 32

// Define real type
typedef double real;
#define REAL_TYPE "double"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cusparse.h>
#include <cusolverSp.h>

#include "cell_model.h"
#include "parameters.h"
#include "auxfuncs.h"
#include "functions.h"
#include "cuda_functions.h"
#include "methods.h"

#endif // INCLUDES_H

// nvcc -Xcompiler -fopenmp -lpthread -lcusparse main.cu -o main -O3