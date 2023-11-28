#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include "includes.h"

__constant__ real d_stimStrength = 100.0;  

__constant__ real d_stim1Begin = 0.0;            // Stimulation start time -> ms
__constant__ real d_stim1Duration = 2.0;         // Stimulation duration -> ms

__constant__ real d_stim2Begin = 120.0;          // Stimulation start time -> ms
__constant__ real d_stim2Duration = 2.0;         // Stimulation duration -> ms

__constant__ real d_G = 1.5;         // omega^-1 * cm^-2
__constant__ real d_eta1 = 4.4;      // omega^-1 * cm^-1
__constant__ real d_eta2 = 0.012;    // dimensionless
__constant__ real d_eta3 = 1.0;      // dimensionless
__constant__ real d_vth = 13.0;      // mV
__constant__ real d_vp = 100.0;      // mV
__constant__ real d_sigma = 1.2e-3;  // omega^-1 * cm^-1

__constant__ real d_chi = 1.0e3;     // cm^-1
__constant__ real d_Cm = 1.0e-3;     // mF * cm^-2

// From GLOSTER, Andrew et al. Efficient Interleaved Batch Matrix Solvers for CUDA. arXiv preprint arXiv:1909.04539, 2019.
__global__ void parallelThomas(real *d, unsigned long N, real *la, real *lb, real *lc)
{ 
    int previousRow, nextRow;
    int currentRow = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 0;

    if (currentRow < N)
    {    
        // 1st: update auxiliary arrays
        d[currentRow] = d[currentRow] / lb[i];
        
        #pragma unroll
        for (i = 1; i < N; i++)
        {
            previousRow = currentRow;
            currentRow += N;

            d[currentRow] = (d[currentRow] - la[i] * d[previousRow]) / (lb[i]);
        }
        
        // 2nd: update solution
        d[currentRow] = d[currentRow];
        
        #pragma unroll
        for (i = N - 2; i >= 0; i--)
        {
            nextRow = currentRow;
            currentRow -= N;

            d[currentRow] = d[currentRow] - lc[i] * d[nextRow];
        }
    }
}

__device__ real d_stimulus(int i, int j, int timeStep, int discS1xLimit, int discS1yLimit, int discS2xMin, int discS2xMax, int discS2yMin, int discS2yMax)
{
   // Stimulus 1
    if (timeStep >= d_stim1Begin && timeStep <= d_stim1Begin + d_stim1Duration && j <= discS1xLimit)
    {
        return d_stimStrength;
    }
    // Stimulus 2
    else if (timeStep >= d_stim2Begin && timeStep <= d_stim2Begin + d_stim2Duration && j >= discS2xMin && j <= discS2xMax && i >= discS2yMin && i <= discS2yMax)
    {
        return d_stimStrength;
    }
    return 0.0;
}

#if defined(AFHN)
__device__ real d_reactionV(real v, real w)
{
    return (1.0 / (d_Cm * d_chi)) * ((-d_G * v * (1.0 - (v / d_vth)) * (1.0 - (v / d_vp))) + (-d_eta1 * v * w));
}

__device__ real d_reactionW(real v, real w)
{
    return d_eta2 * ((v / d_vp) - (d_eta3 * w));
}
#endif // AFHN

__global__ void parallelODE(real *d_V, real *d_W, real *d_rightside, unsigned long N, real timeStep, real deltat, int discS1xLimit, int discS1yLimit, int discS2xMin, int discS2xMax, int discS2yMin, int discS2yMax)
{
    // Naive
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // Diagonal / Column
    // unsigned int blk_y = blockIdx.x;
    // unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    // unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    // unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < N && iy < N)
    {
        unsigned int index = ix * N + iy;
        
        real actualV = d_V[index];
        real actualW = d_W[index];

        d_V[index] = actualV + deltat * (d_reactionV(actualV, actualW) + d_stimulus(ix, iy, timeStep, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax));
        d_W[index] = actualW + deltat * d_reactionW(actualV, actualW);

        d_rightside[iy*N+ix] = d_V[index];
    }
}

__global__ void transposeDiagonalCol(real *in, real *out, unsigned int nx, unsigned int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

//=======================================
//      3D functions
//=======================================
__global__ void parallelODE3D(real *d_V, real *d_W, real *d_rightside, unsigned long N, real timeStep, real deltat, int discS1xLimit, int discS1yLimit, int discS2xMin, int discS2xMax, int discS2yMin, int discS2yMax)
{
    // Naive
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix < N && iy < N)
    {
        unsigned int index = (ix * N) + iy + (iz * N * N);
        
        real actualV = d_V[index];
        real actualW = d_W[index];

        d_V[index] = actualV + deltat * (d_reactionV(actualV, actualW) + d_stimulus(ix, iy, timeStep, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax));
        d_W[index] = actualW + deltat * d_reactionW(actualV, actualW);

        d_rightside[iy*N+ix+(iz*N*N)] = d_V[index];
    }
}

__global__ void parallelThomas3D(real *d, unsigned long N, real *la, real *lb, real *lc)
{ 
    int previousRow, nextRow;
    int currentRow = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 0;

    if (currentRow < N*N && currentRow % (N*N) == 0)
    {    
        // 1st: update auxiliary arrays
        d[currentRow] = d[currentRow] / lb[i];
        
        #pragma unroll
        for (i = 1; i < N; i++)
        {
            previousRow = currentRow;
            currentRow += N;

            d[currentRow] = (d[currentRow] - la[i] * d[previousRow]) / (lb[i]);
        }
        
        // 2nd: update solution
        d[currentRow] = d[currentRow];
        
        #pragma unroll
        for (i = N - 2; i >= 0; i--)
        {
            nextRow = currentRow;
            currentRow -= N;

            d[currentRow] = d[currentRow] - lc[i] * d[nextRow];
        }
    }
}

__global__ void mapping1(real *in, real *out, unsigned int nx, unsigned int ny, unsigned int nz)
{
    // Naive
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix < nx && iy < ny && iz < nz)
    {
        out[ix*nx + iy + iz*nx*nz] = in[iy*ny + ix + iz*nx*nz];
    }
}

__global__ void mapping2(real *in, real *out, unsigned int nx, unsigned int ny, unsigned int nz)
{
    // Naive
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix < nx && iy < ny && iz < nz)
    {
        out[iz*nx + ix + iy*nx*nz] = in[ix*ny + iy + iz*nx*nz];
    }
}

#endif // CUDA_FUNCTIONS_H