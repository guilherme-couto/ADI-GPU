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
__global__ void parallelThomas(real *d, unsigned int N, real *la, real *lb, real *lc)
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

__global__ void parallelODE(real *d_V, real *d_W, real *d_rightside, unsigned int N, real timeStep, real deltat, int discS1xLimit, int discS1yLimit, int discS2xMin, int discS2xMax, int discS2yMin, int discS2yMax)
{
    // Naive
    //unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    //unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // Diagonal / Column
    //unsigned int blk_y = blockIdx.x;
    //unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    //unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    //unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    //if (ix < N && iy < N)
    if (i < N*N)
    {
        unsigned int ix = i / N;
        unsigned int iy = i % N;
        
        real actualV = d_V[i];
        real actualW = d_W[i];

        d_V[i] = actualV + deltat * (d_reactionV(actualV, actualW) + d_stimulus(ix, iy, timeStep, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax));
        //d_V[i] = actualV + deltat * (d_reactionV(actualV, actualW));
        d_W[i] = actualW + deltat * d_reactionW(actualV, actualW);

        d_rightside[iy*N+ix] = d_V[i];
    }
}

__device__ real d_iDiffusion(int i, int j, unsigned int N, real *V, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = i * N + j;

    real result = 0.0;
    if (i == 0)
    {
        result = - 2.0*V[index] + 2.0*V[index + N];  
    }
    else if (i == N - 1)
    {
        result = 2.0*V[index - N] - 2.0*V[index]; 
    }
    else
    {
        result = V[index - N] - 2.0*V[index] + V[index + N];
    }

    if ((i >= discFibyMin && i <= discFibyMax) && (j >= discFibxMin && j <= discFibxMax))
    {
        result *= fibrosisFactor;
    }
    return result;
}

__device__ real d_jDiffusion(int i, int j, unsigned int N, real *V, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = i * N + j;

    real result = 0.0;
    if (j == 0)
    {
        result = - 2.0*V[index] + 2.0*V[index + 1];  
    }
    else if (j == N - 1)
    {
        result = 2.0*V[index - 1] - 2.0*V[index]; 
    }
    else
    {
        result = V[index - 1] - 2.0*V[index] + V[index + 1];
    }

    if ((i >= discFibyMin && i <= discFibyMax) && (j >= discFibxMin && j <= discFibxMax))
    {
        result *= fibrosisFactor;
    }
    return result;
}

__global__ void parallelODE_SSI(real *d_V, real *d_W, real *d_Rv, unsigned int N, real timeStep, real deltat, real phi, int discS1xLimit, int discS1yLimit, int discS2xMin, int discS2xMax, int discS2yMin, int discS2yMax, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N*N)
    {
        unsigned int ix = i / N;
        unsigned int iy = i % N;
        
        real actualV = d_V[i];
        real actualW = d_W[i];

        real stim = d_stimulus(ix, iy, timeStep, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax);

        // Update V with diffusion (RK2) and W without diffusion
        real Vtilde, Wtilde;
        real diffusion = d_iDiffusion(ix, iy, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor) + d_jDiffusion(ix, iy, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
        
        Vtilde = actualV + (0.5 * deltat * (d_reactionV(actualV, actualW) + stim)) + (0.5 * phi * diffusion);
        Wtilde = actualW + 0.5 * deltat * d_reactionW(actualV, actualW);

        // Update V reaction term
        d_Rv[i] = deltat * (d_reactionV(Vtilde, Wtilde) + stim);

        // Update W explicitly (RK2)
        d_W[i] = actualW + deltat * d_reactionW(Vtilde, Wtilde);
    }
}

__global__ void parallelODE_MOSI(real *d_V, real *d_W, real *d_Rv, unsigned int N, real timeStep, real deltat, real phi, int discS1xLimit, int discS1yLimit, int discS2xMin, int discS2xMax, int discS2yMin, int discS2yMax, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N*N)
    {
        unsigned int ix = i / N;
        unsigned int iy = i % N;
        
        real actualV = d_V[i];
        real actualW = d_W[i];

        real stim = d_stimulus(ix, iy, timeStep, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax);

        // Update V with diffusion (RK2) and W without diffusion
        real Vtilde, Wtilde;
        real diffusion = d_iDiffusion(ix, iy, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor) + d_jDiffusion(ix, iy, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
        
        Vtilde = actualV + (0.5 * deltat * (d_reactionV(actualV, actualW) + stim)) + (0.5 * phi * diffusion);
        
        real Rw = deltat * d_reactionW(actualV, actualW);
        Wtilde = actualW + 0.5 * Rw;

        // Update V reaction term
        d_Rv[i] = deltat * (d_reactionV(Vtilde, Wtilde) + stim);

        // Update W explicitly
        d_W[i] = actualW + Rw;
    }
}

__global__ void transposeDiagonalCol(real *in, real *out, unsigned int nx, unsigned int ny)
{
    //unsigned int blk_y = blockIdx.x;
    //unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    //unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    //unsigned int iy = blockDim.y * blk_y + threadIdx.y;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    //if (ix < nx && iy < ny)
    if (i < nx*ny)
    {
        unsigned int ix = i / nx;
        unsigned int iy = i % ny;
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

__global__ void prepareRighthandSide_iDiffusion(real *d_V, real *d_rightside, real *d_Rv, unsigned int N, real phi, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N*N)
    {
        int i = index / N;
        int j = index % N;
        d_rightside[index] = d_V[index] + (0.5 * phi * d_iDiffusion(i, j, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor)) + (0.5 * d_Rv[index]);
    }
}

__global__ void prepareRighthandSide_jDiffusion(real *d_V, real *d_rightside, real *d_Rv, unsigned int N, real phi, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N*N)
    {
        int i = index / N;
        int j = index % N;
        int transposedIndex = j * N + i;
        d_rightside[transposedIndex] = d_V[index] + (0.5 * phi * d_jDiffusion(i, j, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor)) + (0.5 * d_Rv[index]);
    }
}

//=======================================
//      3D functions
//=======================================
__global__ void parallelODE3D(real *d_V, real *d_W, real *d_rightside, unsigned int N, real timeStep, real deltat, int discS1xLimit, int discS1yLimit, int discS2xMin, int discS2xMax, int discS2yMin, int discS2yMax)
{
    // Naive
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix < N && iy < N && iz < N)
    {
        unsigned int index = (ix * N) + iy + (iz * N * N);
        unsigned int index2 = (iy * N) + ix + (iz * N * N);
        
        real actualV = d_V[index];
        real actualW = d_W[index];

        d_V[index] = actualV + deltat * (d_reactionV(actualV, actualW) + d_stimulus(ix, iy, timeStep, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax));
        d_W[index] = actualW + deltat * d_reactionW(actualV, actualW);

        d_rightside[index2] = d_V[index];

        // if (index == 8000)
        // {
        //     printf("\nparallelODE3D\n");
        //     printf("value_in = %f\n", actualV);
        //     printf("ix = %d, iy = %d, iz = %d\n", ix, iy, iz);
        //     printf("index_out = %d\n", index2);
        //     printf("value_out = %f\n", d_rightside[index2]);
        // }
    }
}

__global__ void parallelThomas3D(real *d, unsigned int N, real *la, real *lb, real *lc)
{ 
    int previousRow, nextRow;
    int currentRow = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 0;
    int remainder = currentRow % (N*N); 

    if (remainder < N && currentRow < (N*N*N))
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
        out[ix*nx + iy + iz*nx*nz] = in[ix + iy*ny + iz*nx*nz];

        // if (ix + iy*ny + iz*nx*nz == 2200)
        // {
        //     printf("mapping1\n");
        //     printf("value_in = %f\n", in[ix + iy*ny + iz*nx*nz]);
        //     printf("ix = %d, iy = %d, iz = %d\n", ix, iy, iz);
        //     printf("index_out = %d\n", ix*nx + iy + iz*nx*nz);
        //     printf("value_out = %f\n", out[ix*nx + iy + iz*nx*nz]);
        // }
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
        out[iy + iz*ny + ix*ny*ny] = in[iy + ix*ny + iz*nx*nz];

        // if (iy + ix*ny + iz*nx*nz == 8000)
        // {
        //     printf("mapping2\n");
        //     printf("value_in = %f\n", in[iy + ix*ny + iz*nx*nz]);
        //     printf("ix = %d, iy = %d, iz = %d\n", ix, iy, iz);
        //     printf("index_out = %d\n", iy + iz*ny + ix*ny*ny);
        //     printf("value_out = %f\n", out[iy + iz*ny + ix*ny*ny]);
        // }
    }
}

__global__ void mapping3(real *in, real *out, unsigned int nx, unsigned int ny, unsigned int nz)
{
    // Naive
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix < nx && iy < ny && iz < nz)
    {
        out[iy + ix*ny + iz*ny*ny] = in[iy + iz*ny + ix*ny*ny];

        // if (iy + iz*ny + ix*ny*ny == 805900)
        // {
        //     printf("mapping3\n");
        //     printf("value_in = %f\n", in[iy + iz*ny + ix*ny*ny]);
        //     printf("ix = %d, iy = %d, iz = %d\n", ix, iy, iz);
        //     printf("index_out = %d\n", iy + ix*ny + iz*ny*ny);
        //     printf("value_out = %f\n", out[iy + ix*ny + iz*ny*ny]);
        // }
    }
}

#endif // CUDA_FUNCTIONS_H