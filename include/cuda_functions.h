#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include "includes.h"

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

__global__ void transposeDiagonal(real *odata, real *idata, int width, int height)
{
    __shared__ real tile[TILE_DIM][TILE_DIM+1];
    int blockIdx_x, blockIdx_y;

    // diagonal reordering
    if (width == height)
    {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    }
    else
    {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid%gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }

    int xIndex = blockIdx_x*TILE_DIM + threadIdx.x;
    int yIndex = blockIdx_y*TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex = blockIdx_y*TILE_DIM + threadIdx.x;
    yIndex = blockIdx_x*TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
    {
        tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
    }

    __syncthreads();

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
    {
        odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
    }
    
}

__global__ void transposeNaive(real *odata, real* idata, int width, int height)
{
    int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
    int index_in = xIndex + width * yIndex;
    int index_out = yIndex + height * xIndex;

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
    {
        odata[index_out+i] = idata[index_in+i*width];
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

#endif // CUDA_FUNCTIONS_H