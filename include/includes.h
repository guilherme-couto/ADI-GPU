#ifndef INCLUDES_H
#define INCLUDES_H

#define MAX_STRING_SIZE 100
#define TILE_DIM 16
#define BLOCK_ROWS 8
#define BLOCK_SIZE 32
#define NUM_BLOCKS 10
/*size X of shared memory tile*/
#define BDIMX 16
/*size Y of shared memory tile*/
#define BDIMY 16

// Define real type
typedef float real;

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