#ifndef CONVERGENCE_FUNCTIONS_H
#define CONVERGENCE_FUNCTIONS_H

#include "../include/includes.h"

__constant__ real d_L = 2.0;

#if defined(AFHN)
__device__ real d_forcingTerm(unsigned int i, unsigned int j, real delta_x, real t)
{
    real x = i * delta_x;
    real y = j * delta_x;

    return (-M_PI + (2*M_PI*M_PI/(d_L*d_L))) * sin(M_PI*x/d_L) * sin(M_PI*y/d_L) * sin(M_PI*t);
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
        
        real actualRHS_V = (1.0 / (d_Cm * d_chi)) * d_forcingTerm(i, j, delta_x, t);
        
        Vtilde = actualV + (delta_t * actualRHS_V) + (phi * diffusion);

        real tildeRHS_V = actualRHS_V;

        // Update V reaction term
        d_Rv[index] = delta_t * (((1.0 - theta) * actualRHS_V) + (theta * tildeRHS_V));
    }
}
#endif

#endif // CONVERGENCE_FUNCTIONS_H