#ifndef CONVERGENCE_FUNCTIONS_H
#define CONVERGENCE_FUNCTIONS_H

#include "../include/includes.h"

__constant__ real d_L = 1.0;
__constant__ real d_pi = 3.14159265358979323846;

#if defined(AFHN)
__device__ real d_forcingTerm(unsigned int i, unsigned int j, real delta_x, real t)
{
    real x = j * delta_x;
    real y = i * delta_x;

    return cos(d_pi*x/d_L) * cos(d_pi*y/d_L) * (d_pi*cos(d_pi*t) + ((2*d_pi*d_pi*d_sigma)/(d_chi*d_Cm*d_L*d_L))*sin(d_pi*t) + (d_G/d_Cm)*sin(d_pi*t)); 
}

__global__ void parallelRHSForcing_theta(real *d_V, real *d_Rv, unsigned int N, real t, real delta_t, real delta_x, real phi, real theta, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N*N)
    {
        unsigned int i = index / N;
        unsigned int j = index % N;
        
        real actualV = d_V[index];
        real Vtilde;
        real diffusion = d_iDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor) + d_jDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
        
        real actualRHS_V = d_forcingTerm(i, j, delta_x, t) - (d_G*actualV/d_Cm);
        
        Vtilde = actualV + (delta_t * actualRHS_V) + (phi * diffusion);

        real tildeRHS_V = d_forcingTerm(i, j, delta_x, t+(1.0*delta_t)) - (d_G*Vtilde/d_Cm);     // alpha = 1.0 (Heun's) / 0.5 (midpoint) / 2/3 (Ralston's)

        // Update V reaction term
        d_Rv[index] = delta_t * (((1.0 - theta) * actualRHS_V) + (theta * tildeRHS_V));
    }
}

__global__ void parallelRHSForcing_SSI(real *d_V, real *d_Rv, unsigned int N, real t, real delta_t, real delta_x, real phi, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N*N)
    {
        unsigned int i = index / N;
        unsigned int j = index % N;
        
        real actualV = d_V[index];
        real Vtilde;
        real diffusion = d_iDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor) + d_jDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
        
        real actualRHS_V = d_forcingTerm(i, j, delta_x, t) - (d_G*actualV/d_Cm);
        
        Vtilde = actualV + (0.5 * delta_t * actualRHS_V) + (((delta_t * d_sigma) / (2*delta_x*delta_x*d_chi*d_Cm)) * diffusion);

        real tildeRHS_V = d_forcingTerm(i, j, delta_x, t+(0.5*delta_t)) - (d_G*Vtilde/d_Cm);
        // real tildeRHS_V = (d_forcingTerm(i, j, delta_x, t+(delta_t)))- (d_G*Vtilde/d_Cm);

        // Update V reaction term
        d_Rv[index] = delta_t * tildeRHS_V;
    }
}

__global__ void parallelFE(real *d_V, real *d_Rv, unsigned int N, real t, real delta_t, real delta_x, real phi, real theta, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N*N)
    {
        unsigned int i = index / N;
        unsigned int j = index % N;
        
        real actualV = d_V[index];
        real diffusion = d_iDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor) + d_jDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
        real linearReaction = d_G*actualV;
        real forcingTerm = d_forcingTerm(i, j, delta_x, t);

        d_Rv[index] = actualV + (phi * diffusion) - (delta_t * linearReaction) + (delta_t * forcingTerm);
    }
}

__global__ void cuThomasConstantBatch(real* la, real* lb, real* lc, real* d, unsigned long n) {

	int rowCurrent;
	int rowPrevious;

	int rowAhead;

	// set the current row
	rowCurrent = threadIdx.x + blockDim.x*blockIdx.x;

	int i = 0;

	if ( rowCurrent < n ) 
	{

		//----- Forward Sweep
		d[rowCurrent] = d[rowCurrent] / lb[i];

		#pragma unroll
		for (i = 1; i < n; ++i) {
			rowPrevious = rowCurrent;
			rowCurrent += n;

			d[rowCurrent] = (d[rowCurrent] - la[i]*d[rowPrevious]) / (lb[i]);
		
		}


		//----- Back Sub
		d[rowCurrent] = d[rowCurrent];

		#pragma unroll
		for (i = n - 2; i >= 0; --i) {
			rowAhead    = rowCurrent;
			rowCurrent -= n;

			d[rowCurrent] = d[rowCurrent] - lc[i] * d[rowAhead];
		}
	}
}

__global__ void prepareRHS_theta(real *d_V, real *d_rightside, real *d_Rv, unsigned int N, real phi, real theta, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N * N)
    {
        int i = index / N;
        int j = index % N;
        d_rightside[index] = d_V[index] + ((1-theta) * phi * d_jDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor)) + (0.5 * d_Rv[index]);
    }
}

__global__ void prepareRHS(real *d_V, real *d_RHS, real *d_Rv, unsigned int N, real phi, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N * N)
    {
        int i = index / N;
        int j = index % N;
        d_RHS[index] = d_V[index] + (phi * d_iDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor)) + (0.5 * d_Rv[index]);
    }
}

__global__ void prepareRHS2(real *d_V, real *d_RHS, real *d_Rv, unsigned int N, real phi, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N * N)
    {
        int i = index / N;
        int j = index % N;
        d_RHS[index] = d_V[index] + (phi * d_jDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor)) + (0.5 * d_Rv[index]);
    }
}


#endif

#endif // CONVERGENCE_FUNCTIONS_H