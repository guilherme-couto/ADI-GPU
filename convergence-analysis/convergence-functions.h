#ifndef CONVERGENCE_FUNCTIONS_H
#define CONVERGENCE_FUNCTIONS_H

#include "../include/includes.h"

__constant__ real d_L = 1.0;
__constant__ real d_pi = 3.14159265358979323846;

#if defined(AFHN)
__device__ real d_forcingTerm(unsigned int i, unsigned int j, real delta_x, real t)
{
    real x = i * delta_x;
    real y = j * delta_x;

    return (d_chi*d_Cm*d_pi*cos(d_pi*t) + (2*d_pi*d_pi*d_sigma*sin(d_pi*t)/(d_L*d_L)) + (-d_G*sin(d_pi*t))) * sin(d_pi*x/d_L) * sin(d_pi*y/d_L);
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
        
        real actualRHS_V = (1.0 / (d_Cm * d_chi)) * (-d_G*actualV + d_forcingTerm(i, j, delta_x, t));
        
        Vtilde = actualV + (delta_t * actualRHS_V) + (phi * diffusion);
        // Vtilde = actualV + (theta * delta_t * actualRHS_V) + (theta * phi * diffusion); // promising

        // real tildeRHS_V = (1.0 / (d_Cm * d_chi)) * (-d_G*Vtilde + d_forcingTerm(i, j, delta_x, t+(1.0*delta_t)));     // alpha = 1.0 (Heun's) / 0.5 (midpoint) / 2/3 (Ralston's)
        real tildeRHS_V = (1.0 / (d_Cm * d_chi)) * (-d_G*Vtilde + d_forcingTerm(i, j, delta_x, t+(theta*delta_t)));     // promising

        // Update V reaction term
        d_Rv[index] = delta_t * (((1.0 - theta) * actualRHS_V) + (theta * tildeRHS_V));
    }
}
#endif

#endif // CONVERGENCE_FUNCTIONS_H