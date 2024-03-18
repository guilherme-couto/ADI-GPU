#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include "cuda_constants.h"
#include "cuda_devicefuncs.h"


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

__global__ void transposeDiagonalCol(real *in, real *out, unsigned int nx, unsigned int ny)
{
    // unsigned int blk_y = blockIdx.x;
    // unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    // unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    // unsigned int iy = blockDim.y * blk_y + threadIdx.y;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    // if (ix < nx && iy < ny)
    if (i < nx * ny)
    {
        unsigned int ix = i / nx;
        unsigned int iy = i % ny;
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

__global__ void prepareRighthandSide_iDiffusion(real *d_V, real *d_rightside, real *d_Rv, unsigned int N, real phi, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N * N)
    {
        int i = index / N;
        int j = index % N;
        d_rightside[index] = d_V[index] + (0.5 * phi * d_iDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor)) + (0.5 * d_Rv[index]);
    }
}

__global__ void prepareRighthandSide_jDiffusion(real *d_V, real *d_rightside, real *d_Rv, unsigned int N, real phi, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N * N)
    {
        int i = index / N;
        int j = index % N;
        int transposedIndex = j * N + i;
        d_rightside[transposedIndex] = d_V[index] + (0.5 * phi * d_jDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor)) + (0.5 * d_Rv[index]);
    }
}

__global__ void prepareRighthandSide_iDiffusion_theta(real *d_V, real *d_rightside, real *d_Rv, unsigned int N, real phi, real theta, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N * N)
    {
        int i = index / N;
        int j = index % N;
        d_rightside[index] = d_V[index] + ((1-theta) * phi * d_iDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor)) + (0.5 * d_Rv[index]);
    }
}

__global__ void prepareRighthandSide_jDiffusion_theta(real *d_V, real *d_rightside, real *d_Rv, unsigned int N, real phi, real theta, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N * N)
    {
        int i = index / N;
        int j = index % N;
        int transposedIndex = j * N + i;
        d_rightside[transposedIndex] = d_V[index] + ((1-theta) * phi * d_jDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor)) + (0.5 * d_Rv[index]);
    }
}

//############################################
//##                                        ##
//##     Adapted FitzHugh-Nagumo (AFHN)     ##
//##                                        ##
//############################################
#if defined(AFHN)
__global__ void parallelODE(real *d_V, real *d_W, real *d_rightside, unsigned int N, real timeStep, real deltat, int discS1xLimit, int discS1yLimit, int discS2xMin, int discS2xMax, int discS2yMin, int discS2yMax)
{
    // Naive
    // unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    // unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // Diagonal / Column
    // unsigned int blk_y = blockIdx.x;
    // unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    // unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    // unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    // if (ix < N && iy < N)
    if (i < N * N)
    {
        unsigned int ix = i / N;
        unsigned int iy = i % N;

        real actualV = d_V[i];
        real actualW = d_W[i];

        d_V[i] = actualV + deltat * (d_reactionV(actualV, actualW) + d_stimulus(ix, iy, timeStep, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax));
        // d_V[i] = actualV + deltat * (d_reactionV(actualV, actualW));
        d_W[i] = actualW + deltat * d_reactionW(actualV, actualW);

        d_rightside[iy * N + ix] = d_V[i];
    }
}

__global__ void parallelODE_SSI(real *d_V, real *d_W, real *d_Rv, unsigned int N, real timeStep, real deltat, real phi, int discS1xLimit, int discS1yLimit, int discS2xMin, int discS2xMax, int discS2yMin, int discS2yMax, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N * N)
    {
        unsigned int i = index / N;
        unsigned int j = index % N;

        real actualV = d_V[index];
        real actualW = d_W[index];

        real stim = d_stimulus(i, j, timeStep, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax);

        // Update V with diffusion and W without diffusion
        real Vtilde, Wtilde;
        real diffusion = d_iDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor) + d_jDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);

        Vtilde = actualV + (0.5 * deltat * (d_reactionV(actualV, actualW) + stim)) + (0.5 * phi * diffusion);
        Wtilde = actualW + 0.5 * deltat * d_reactionW(actualV, actualW);

        // Update V reaction term
        d_Rv[index] = deltat * (d_reactionV(Vtilde, Wtilde) + stim);

        // Update W explicitly (RK2)
        d_W[index] = actualW + deltat * d_reactionW(Vtilde, Wtilde);
    }
}

__global__ void parallelODE_theta(real *d_V, real *d_W, real *d_Rv, unsigned int N, real timeStep, real deltat, real phi, real theta, int discS1xLimit, int discS1yLimit, int discS2xMin, int discS2xMax, int discS2yMin, int discS2yMax, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N * N)
    {
        unsigned int i = index / N;
        unsigned int j = index % N;

        real actualV = d_V[index];
        real actualW = d_W[index];

        real stim = d_stimulus(i, j, timeStep, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax);

        // Update V with diffusion (RK2) and W without diffusion
        real Vtilde, Wtilde;
        real diffusion = d_iDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor) + d_jDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);

        real actualRHS_V = d_reactionV(actualV, actualW);
        real actualRHS_W = d_reactionW(actualV, actualW);

        Vtilde = actualV + (deltat * (actualRHS_V + stim)) + (phi * diffusion);
        //Vtilde = actualV + (deltat * (actualRHS_V + stim));
        Wtilde = actualW + deltat * actualRHS_W;

        real tildeRHS_V = d_reactionV(Vtilde, Wtilde);
        real tildeRHS_W = d_reactionW(Vtilde, Wtilde);

        // Update V reaction term
        // First approach of theta method
        // d_Rv[index] = deltat * (((1.0 - theta) * actualRHS_V) + (theta * tildeRHS_V) + stim); // old
        // d_W[index] = actualW + deltat * (((1.0 - theta) * actualRHS_W) + (theta * tildeRHS_W)); // old

        // Second approach (that works implicitly with theta=1, but it is not the common way to do it)
        // d_Rv[index] = deltat * ((theta * actualRHS_V) + ((1.0 - theta) * tildeRHS_V) + stim); // works implicitly with theta=1
        // d_W[index] = actualW + deltat * ((theta * actualRHS_W) + ((1.0 - theta) * tildeRHS_W)); // works implicitly with theta=1

        // Theta method
        d_Rv[index] = deltat * (((1.0 - theta) * actualRHS_V) + (theta * tildeRHS_V) + stim);
        d_W[index] = actualW + deltat * tildeRHS_W;
    }
}

__global__ void parallelODE_MOSI(real *d_V, real *d_W, real *d_Rv, unsigned int N, real timeStep, real deltat, real phi, int discS1xLimit, int discS1yLimit, int discS2xMin, int discS2xMax, int discS2yMin, int discS2yMax, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N * N)
    {
        unsigned int i = index / N;
        unsigned int j = index % N;

        real actualV = d_V[index];
        real actualW = d_W[index];

        real stim = d_stimulus(i, j, timeStep, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax);

        // Update V with diffusion and W without diffusion
        real Vtilde, Wtilde;
        real diffusion = d_iDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor) + d_jDiffusion(i, j, index, N, d_V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);

        Vtilde = actualV + (0.5 * deltat * (d_reactionV(actualV, actualW) + stim)) + (0.5 * phi * diffusion);

        real Rw = deltat * d_reactionW(actualV, actualW);
        Wtilde = actualW + 0.5 * Rw;

        // Update V reaction term
        d_Rv[index] = deltat * (d_reactionV(Vtilde, Wtilde) + stim);

        // Update W explicitly
        d_W[index] = actualW + Rw;
    }
}
#endif // AFHN

//###########################################
//##                                       ##
//##     ten Tusscher 2006 model (TT2)     ##
//##                                       ##
//###########################################
#if defined(TT2)
__global__ void parallelODE_MOSI(real *V, real *X_r1, real *X_r2, real *X_s, real *m, real *h, real *j, real *d, real *f, real *f2, real *fCass, real *s, real *r, real *Ca_i, real *Ca_SR, real *Ca_SS, real *R_prime, real *Na_i, real *K_i, real *d_Rv, unsigned int N, real timeStep, real deltat, real phi, int discS1xLimit, int discS1yLimit, int discS2xMin, int discS2xMax, int discS2yMin, int discS2yMax, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N * N)
    {
        unsigned int i = index / N;
        unsigned int jj = index % N;

        real actualV = V[index];
        real actualX_r1 = X_r1[index];
        real actualX_r2 = X_r2[index];
        real actualX_s = X_s[index];
        real actualm = m[index];
        real actualh = h[index];
        real actualj = j[index];
        real actuald = d[index];
        real actualf = f[index];
        real actualf2 = f2[index];
        real actualfCass = fCass[index];
        real actuals = s[index];
        real actualr = r[index];
        real actualCa_i = Ca_i[index];
        real actualCa_SR = Ca_SR[index];
        real actualCa_SS = Ca_SS[index];
        real actualR_prime = R_prime[index];
        real actualNa_i = Na_i[index];
        real actualK_i = K_i[index];

        real stim = d_stimulus(i, jj, timeStep, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax);

        // Itotal
        real Itotal = d_Itotal(stim, actualV, actualm, actualh, actualj, actualNa_i, actualK_i, actualr, actuals, actualX_r1, actualX_r2, actualX_s, actuald, actualf, actualf2, actualfCass, actualCa_SS, actualCa_i);

        // Update V with diffusion and W without diffusion
        real Vtilde, X_r1tilde, X_r2tilde, X_stilde, mtilde, htilde, jtilde, dtilde, ftilde, f2tilde, fCasstilde, stilde, rtilde, Ca_itilde, Ca_SRtilde, Ca_SStilde, R_primetilde, Na_itilde, K_itilde;

        // Diffusion for Vtilde
        real diffusion = d_iDiffusion(i, jj, index, N, V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor) + d_jDiffusion(i, jj, index, N, V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
        Vtilde = actualV + (0.5 * deltat * (-Itotal)) + (0.5 * phi * diffusion);

        // Reactions and predictions for state variables
        X_r1tilde = d_updateXr1(actualX_r1, actualV, 0.5*deltat);
        X_r2tilde = d_updateXr2(actualX_r2, actualV, 0.5*deltat);
        X_stilde = d_updateXs(actualX_s, actualV, 0.5*deltat);
        mtilde = d_updatem(actualm, actualV, 0.5*deltat);
        htilde = d_updateh(actualh, actualV, 0.5*deltat);
        jtilde = d_updatej(actualj, actualV, 0.5*deltat);
        dtilde = d_updated(actuald, actualV, 0.5*deltat);
        ftilde = d_updatef(actualf, actualV, 0.5*deltat);
        f2tilde = d_updatef2(actualf2, actualV, 0.5*deltat);
        fCasstilde = d_updatefCass(actualfCass, actualV, 0.5*deltat);
        stilde = d_updates(actuals, actualV, 0.5*deltat);
        rtilde = d_updater(actualr, actualV, 0.5*deltat);

        R_primetilde = actualR_prime + 0.5 * deltat * d_dRprimedt(actualCa_SS, actualR_prime);
        Ca_itilde = actualCa_i + 0.5 * deltat * d_dCaidt(actualCa_i, actualCa_SR, actualCa_SS, actualV, actualNa_i);
        Ca_SRtilde = actualCa_SR + 0.5 * deltat * d_dCaSRdt(actualCa_SR, actualCa_i, actualCa_SS, actualR_prime);
        Ca_SStilde = actualCa_SS + 0.5 * deltat * d_dCaSSdt(actualCa_SS, actualV, actuald, actualf, actualf2, actualfCass, actualCa_SR, actualR_prime, actualCa_i);
        Na_itilde = actualNa_i + 0.5 * deltat * d_dNaidt(actualV, actualm, actualh, actualj, actualNa_i, actualCa_i);
        K_itilde = actualK_i + 0.5 * deltat * d_dKidt(stim, actualV, actualK_i, actualr, actuals, actualX_r1, actualX_r2, actualX_s, actualNa_i);

        // Update V reaction term
        real Itotaltilde = d_Itotal(stim, Vtilde, mtilde, htilde, jtilde, Na_itilde, K_itilde, rtilde, stilde, X_r1tilde, X_r2tilde, X_stilde, dtilde, ftilde, f2tilde, fCasstilde, Ca_SStilde, Ca_itilde);
        d_Rv[index] = deltat * (-Itotaltilde);

        // Update state variables
        X_r1[index] = d_updateXr1(actualX_r1, actualV, deltat);
        X_r2[index] = d_updateXr2(actualX_r2, actualV, deltat);
        X_s[index] = d_updateXs(actualX_s, actualV, deltat);
        m[index] = d_updatem(actualm, actualV, deltat);
        h[index] = d_updateh(actualh, actualV, deltat);
        j[index] = d_updatej(actualj, actualV, deltat);
        d[index] = d_updated(actuald, actualV, deltat);
        f[index] = d_updatef(actualf, actualV, deltat);
        f2[index] = d_updatef2(actualf2, actualV, deltat);
        fCass[index] = d_updatefCass(actualfCass, actualV, deltat);
        s[index] = d_updates(actuals, actualV, deltat);
        r[index] = d_updater(actualr, actualV, deltat);

        R_prime[index] = actualR_prime + deltat * d_dRprimedt(actualCa_SS, actualR_prime);
        Ca_i[index] = actualCa_i + deltat * d_dCaidt(actualCa_i, actualCa_SR, actualCa_SS, actualV, actualNa_i);
        Ca_SR[index] = actualCa_SR + deltat * d_dCaSRdt(actualCa_SR, actualCa_i, actualCa_SS, actualR_prime);
        Ca_SS[index] = actualCa_SS + deltat * d_dCaSSdt(actualCa_SS, actualV, actuald, actualf, actualf2, actualfCass, actualCa_SR, actualR_prime, actualCa_i);
        Na_i[index] = actualNa_i + deltat * d_dNaidt(actualV, actualm, actualh, actualj, actualNa_i, actualCa_i);
        K_i[index] = actualK_i + deltat * d_dKidt(stim, actualV, actualK_i, actualr, actuals, actualX_r1, actualX_r2, actualX_s, actualNa_i);
    }
}

__global__ void parallelODE_MOSI_2(real *V, real *X_r1, real *X_r2, real *X_s, real *m, real *h, real *j, real *d, real *f, real *f2, real *fCass, real *s, real *r, real *Ca_i, real *Ca_SR, real *Ca_SS, real *R_prime, real *Na_i, real *K_i, real *d_Rv, unsigned int N, real timeStep, real deltat, real phi, int discS1xLimit, int discS1yLimit, int discS2xMin, int discS2xMax, int discS2yMin, int discS2yMax, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N * N)
    {
        unsigned int i = index / N;
        unsigned int jj = index % N;

        real actualV = V[index];
        real actualX_r1 = X_r1[index];
        real actualX_r2 = X_r2[index];
        real actualX_s = X_s[index];
        real actualm = m[index];
        real actualh = h[index];
        real actualj = j[index];
        real actuald = d[index];
        real actualf = f[index];
        real actualf2 = f2[index];
        real actualfCass = fCass[index];
        real actuals = s[index];
        real actualr = r[index];
        real actualCa_i = Ca_i[index];
        real actualCa_SR = Ca_SR[index];
        real actualCa_SS = Ca_SS[index];
        real actualR_prime = R_prime[index];
        real actualNa_i = Na_i[index];
        real actualK_i = K_i[index];

        real stim = d_stimulus(i, jj, timeStep, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax);

        // Itotal
        real Itotal = d_Itotal(stim, actualV, actualm, actualh, actualj, actualNa_i, actualK_i, actualr, actuals, actualX_r1, actualX_r2, actualX_s, actuald, actualf, actualf2, actualfCass, actualCa_SS, actualCa_i);

        // Update V with diffusion and W without diffusion
        real Vtilde, X_r1tilde, X_r2tilde, X_stilde, mtilde, htilde, jtilde, dtilde, ftilde, f2tilde, fCasstilde, stilde, rtilde, Ca_itilde, Ca_SRtilde, Ca_SStilde, R_primetilde, Na_itilde, K_itilde;

        // Diffusion for Vtilde
        real diffusion = d_iDiffusion(i, jj, index, N, V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor) + d_jDiffusion(i, jj, index, N, V, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
        Vtilde = actualV + (0.5 * deltat * (-Itotal)) + (0.5 * phi * diffusion);

        // Reactions and predictions for state variables
        X_r1tilde = d_updateXr1(actualX_r1, actualV, 0.5*deltat);
        X_r2tilde = d_updateXr2(actualX_r2, actualV, 0.5*deltat);
        X_stilde = d_updateXs(actualX_s, actualV, 0.5*deltat);
        mtilde = d_updatem(actualm, actualV, 0.5*deltat);
        htilde = d_updateh(actualh, actualV, 0.5*deltat);
        jtilde = d_updatej(actualj, actualV, 0.5*deltat);
        dtilde = d_updated(actuald, actualV, 0.5*deltat);
        ftilde = d_updatef(actualf, actualV, 0.5*deltat);
        f2tilde = d_updatef2(actualf2, actualV, 0.5*deltat);
        fCasstilde = d_updatefCass(actualfCass, actualV, 0.5*deltat);
        stilde = d_updates(actuals, actualV, 0.5*deltat);
        rtilde = d_updater(actualr, actualV, 0.5*deltat);

        R_primetilde = actualR_prime + 0.5 * deltat * d_dRprimedt(actualCa_SS, actualR_prime);
        Ca_itilde = actualCa_i + 0.5 * deltat * d_dCaidt(actualCa_i, actualCa_SR, actualCa_SS, actualV, actualNa_i);
        Ca_SRtilde = actualCa_SR + 0.5 * deltat * d_dCaSRdt(actualCa_SR, actualCa_i, actualCa_SS, actualR_prime);
        Ca_SStilde = actualCa_SS + 0.5 * deltat * d_dCaSSdt(actualCa_SS, actualV, actuald, actualf, actualf2, actualfCass, actualCa_SR, actualR_prime, actualCa_i);
        Na_itilde = actualNa_i + 0.5 * deltat * d_dNaidt(actualV, actualm, actualh, actualj, actualNa_i, actualCa_i);
        K_itilde = actualK_i + 0.5 * deltat * d_dKidt(stim, actualV, actualK_i, actualr, actuals, actualX_r1, actualX_r2, actualX_s, actualNa_i);

        // Update V reaction term
        real Itotaltilde = d_Itotal(stim, Vtilde, mtilde, htilde, jtilde, Na_itilde, K_itilde, rtilde, stilde, X_r1tilde, X_r2tilde, X_stilde, dtilde, ftilde, f2tilde, fCasstilde, Ca_SStilde, Ca_itilde);
        d_Rv[index] = deltat * (-Itotaltilde);

        // Update state variables
        X_r1[index] = d_updateXr1(actualX_r1, actualV, deltat);
        X_r2[index] = d_updateXr2(actualX_r2, actualV, deltat);
        X_s[index] = d_updateXs(actualX_s, actualV, deltat);
        m[index] = d_updatem(actualm, actualV, deltat);
        h[index] = d_updateh(actualh, actualV, deltat);
        j[index] = d_updatej(actualj, actualV, deltat);
        d[index] = d_updated(actuald, actualV, deltat);
        f[index] = d_updatef(actualf, actualV, deltat);
        f2[index] = d_updatef2(actualf2, actualV, deltat);
        fCass[index] = d_updatefCass(actualfCass, actualV, deltat);
        s[index] = d_updates(actuals, actualV, deltat);
        r[index] = d_updater(actualr, actualV, deltat);

        R_prime[index] = actualR_prime + deltat * d_dRprimedt(actualCa_SS, actualR_prime);
        Ca_i[index] = actualCa_i + deltat * d_dCaidt(actualCa_i, actualCa_SR, actualCa_SS, actualV, actualNa_i);
        Ca_SR[index] = actualCa_SR + deltat * d_dCaSRdt(actualCa_SR, actualCa_i, actualCa_SS, actualR_prime);
        Ca_SS[index] = actualCa_SS + deltat * d_dCaSSdt(actualCa_SS, actualV, actuald, actualf, actualf2, actualfCass, actualCa_SR, actualR_prime, actualCa_i);
        Na_i[index] = actualNa_i + deltat * d_dNaidt(actualV, actualm, actualh, actualj, actualNa_i, actualCa_i);
        K_i[index] = actualK_i + deltat * d_dKidt(stim, actualV, actualK_i, actualr, actuals, actualX_r1, actualX_r2, actualX_s, actualNa_i);
    }
}

__global__ void parallelODE_MOSI_USV(real *V, real *X_r1, real *X_r2, real *X_s, real *m, real *h, real *j, real *d, real *f, real *f2, real *fCass, real *s, real *r, real *Ca_i, real *Ca_SR, real *Ca_SS, real *R_prime, real *Na_i, real *K_i, real *d_Rv, unsigned int N, real timeStep, real deltat, real phi, int discS1xLimit, int discS1yLimit, int discS2xMin, int discS2xMax, int discS2yMin, int discS2yMax, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N * N)
    {
        unsigned int i = index / N;
        unsigned int jj = index % N;

        real actualV = V[index];
        real actualX_r1 = X_r1[index];
        real actualX_r2 = X_r2[index];
        real actualX_s = X_s[index];
        real actualm = m[index];
        real actualh = h[index];
        real actualj = j[index];
        real actuald = d[index];
        real actualf = f[index];
        real actualf2 = f2[index];
        real actualfCass = fCass[index];
        real actuals = s[index];
        real actualr = r[index];
        real actualCa_i = Ca_i[index];
        real actualCa_SR = Ca_SR[index];
        real actualCa_SS = Ca_SS[index];
        real actualR_prime = R_prime[index];
        real actualNa_i = Na_i[index];
        real actualK_i = K_i[index];

        real stim = d_stimulus(i, jj, timeStep, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax);

        // Update state variables
        X_r1[index] = d_updateXr1(actualX_r1, actualV, deltat);
        X_r2[index] = d_updateXr2(actualX_r2, actualV, deltat);
        X_s[index] = d_updateXs(actualX_s, actualV, deltat);
        m[index] = d_updatem(actualm, actualV, deltat);
        h[index] = d_updateh(actualh, actualV, deltat);
        j[index] = d_updatej(actualj, actualV, deltat);
        d[index] = d_updated(actuald, actualV, deltat);
        f[index] = d_updatef(actualf, actualV, deltat);
        f2[index] = d_updatef2(actualf2, actualV, deltat);
        fCass[index] = d_updatefCass(actualfCass, actualV, deltat);
        s[index] = d_updates(actuals, actualV, deltat);
        r[index] = d_updater(actualr, actualV, deltat);

        R_prime[index] = actualR_prime + deltat * d_dRprimedt(actualCa_SS, actualR_prime);
        Ca_i[index] = actualCa_i + deltat * d_dCaidt(actualCa_i, actualCa_SR, actualCa_SS, actualV, actualNa_i);
        Ca_SR[index] = actualCa_SR + deltat * d_dCaSRdt(actualCa_SR, actualCa_i, actualCa_SS, actualR_prime);
        Ca_SS[index] = actualCa_SS + deltat * d_dCaSSdt(actualCa_SS, actualV, actuald, actualf, actualf2, actualfCass, actualCa_SR, actualR_prime, actualCa_i);
        Na_i[index] = actualNa_i + deltat * d_dNaidt(actualV, actualm, actualh, actualj, actualNa_i, actualCa_i);
        K_i[index] = actualK_i + deltat * d_dKidt(stim, actualV, actualK_i, actualr, actuals, actualX_r1, actualX_r2, actualX_s, actualNa_i);
    }
}
#endif // TT2

//=======================================
//      3D functions
//=======================================
#if defined(AFHN)
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
#endif // AFHN

__global__ void parallelThomas3D(real *d, unsigned int N, real *la, real *lb, real *lc)
{
    int previousRow, nextRow;
    int currentRow = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 0;
    int remainder = currentRow % (N * N);

    if (remainder < N && currentRow < (N * N * N))
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
        out[ix * nx + iy + iz * nx * nz] = in[ix + iy * ny + iz * nx * nz];

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
        out[iy + iz * ny + ix * ny * ny] = in[iy + ix * ny + iz * nx * nz];

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
        out[iy + ix * ny + iz * ny * ny] = in[iy + iz * ny + ix * ny * ny];

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