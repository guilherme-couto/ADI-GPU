#ifndef CUDA_DEVICEFUNCS_H
#define CUDA_DEVICEFUNCS_H

#include "cuda_constants.h"
#include "includes.h"

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

__device__ real d_iDiffusion(unsigned int i, unsigned int j, unsigned int index, unsigned int N, real *V, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    // unsigned int index = i * N + j;

    real result = 0.0;
    if (i == 0)
    {
        result = -2.0 * V[index] + 2.0 * V[index + N];
    }
    else if (i == N - 1)
    {
        result = 2.0 * V[index - N] - 2.0 * V[index];
    }
    else
    {
        result = V[index - N] - 2.0 * V[index] + V[index + N];
    }

    if ((i >= discFibyMin && i <= discFibyMax) && (j >= discFibxMin && j <= discFibxMax))
    {
        result *= fibrosisFactor;
    }
    return result;
}

__device__ real d_jDiffusion(unsigned int i, unsigned int j, unsigned int index, unsigned int N, real *V, int discFibxMax, int discFibxMin, int discFibyMax, int discFibyMin, real fibrosisFactor)
{
    // unsigned int index = i * N + j;

    real result = 0.0;
    if (j == 0)
    {
        result = -2.0 * V[index] + 2.0 * V[index + 1];
    }
    else if (j == N - 1)
    {
        result = 2.0 * V[index - 1] - 2.0 * V[index];
    }
    else
    {
        result = V[index - 1] - 2.0 * V[index] + V[index + 1];
    }

    if ((i >= discFibyMin && i <= discFibyMax) && (j >= discFibxMin && j <= discFibxMax))
    {
        result *= fibrosisFactor;
    }
    return result;
}

//############################################
//##                                        ##
//##     Adapted FitzHugh-Nagumo (AFHN)     ##
//##                                        ##
//############################################
#if defined(AFHN)
__device__ real d_reactionV(real v, real w)
{
    return (1.0 / (d_Cm * d_chi)) * ((-d_G * v * (1.0 - (v / d_vth)) * (1.0 - (v / d_vp))) + (-d_eta1 * v * w));
}

__device__ real d_reactionW(real v, real w)
{
    return d_eta2 * ((v / d_vp) - (d_eta3 * w));
}
#endif //AFHN

//###########################################
//##                                       ##
//##     ten Tusscher 2006 model (TT2)     ##
//##                                       ##
//###########################################
#if defined(TT2)
#if defined(EPI) || defined(M) || defined(ENDO)
/*---------------------------------------------------------------------------------------------------------------------------------------------------
Functions for ten Tusscher model 2006 (https://journals.physiology.org/doi/full/10.1152/ajpheart.00109.2006)
from https://tbb.bio.uu.nl/khwjtuss/SourceCodes/HVM2/Source/Main.cc - ten Tusscher code
and https://github.com/rsachetto/MonoAlg3D_C/blob/master/src/models_library/ten_tusscher/ten_tusscher_2006_RS_CPU.c - Sachetto MonoAlg3D
-----------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------
Currents functions
----------------------*/
// Reversal potentials for Na+, K+ and Ca++
__device__ real d_E_Na(real Na_i)
{
    return d_RTONF * log(d_Na_o / Na_i);
}
__device__ real d_E_K(real K_i)
{
    return d_RTONF * log(d_K_o / K_i);
}
__device__ real d_E_Ca(real Ca_i)
{
    return 0.5 * d_RTONF * log(d_Ca_o / Ca_i);
}

// Reversal potential for Ks
__device__ real d_E_Ks(real K_i, real Na_i)
{
    return d_RTONF * log((d_K_o + d_p_KNa * d_Na_o) / (K_i + d_p_KNa * Na_i));
}

// Fast sodium (Na+) current
__device__ real d_I_Na(real V, real m, real h, real j, real Na_i)
{
    return d_G_Na * (m*m*m) * h * j * (V - d_E_Na(Na_i));
}
__device__ real d_m_inf(real V)
{
    return 1.0 / ((1.0 + exp((-56.86 - V) / 9.03))*(1.0 + exp((-56.86 - V) / 9.03)));
}
__device__ real d_alpha_m(real V)
{
    return 1.0 / (1.0 + exp((-60.0 - V) / 5.0));
}
__device__ real d_beta_m(real V)
{
    return (0.1 / (1.0 + exp((V + 35.0) / 5.0))) + (0.1 / (1.0 + exp((V - 50.0) / 200.0)));
}
__device__ real d_tau_m(real V)
{
    return d_alpha_m(V) * d_beta_m(V);
}
__device__ real d_h_inf(real V)
{
    return 1.0 / ((1.0 + exp((V + 71.55) / 7.43))*(1.0 + exp((V + 71.55) / 7.43)));
}
__device__ real d_alpha_h(real V)
{
    if (V >= -40.0)
    {
        return 0.0;
    }
    else
    {
        return 0.057 * exp(-(80.0 + V) / 6.8);
    }
}
__device__ real d_beta_h(real V)
{
    if (V >= -40.0)
    {
        return 0.77 / (0.13 * (1.0 + exp((V + 10.66) / (-11.1))));
    }
    else
    {
        return 2.7 * exp(0.079 * V) + 3.1e5 * exp(0.3485 * V);
    }
}
__device__ real d_tau_h(real V)
{
    return 1.0 / (d_alpha_h(V) + d_beta_h(V));
}
__device__ real d_j_inf(real V)
{
    return 1.0 / ((1.0 + exp((V + 71.55) / 7.43))*(1.0 + exp((V + 71.55) / 7.43)));
}
__device__ real d_alpha_j(real V)
{
    if (V >= -40.0)
    {
        return 0.0;
    }
    else
    {
        return ((-25428.0 * exp(0.2444 * V) - (6.948e-6 * exp((-0.04391) * V))) * (V + 37.78)) / (1.0 + exp(0.311 * (V + 79.23)));
    }
}
__device__ real d_beta_j(real V)
{
    if (V >= -40.0)
    {
        return (0.6 * exp(0.057 * V)) / (1.0 + exp(-0.1 * (V + 32.0)));
    }
    else
    {
        return (0.02424 * exp(-0.01052 * V)) / (1.0 + exp(-0.1378 * (V + 40.14)));
    }
}
__device__ real d_tau_j(real V)
{
    return 1.0 / (d_alpha_j(V) + d_beta_j(V));
}

// L-type Ca2+ current
__device__ real d_I_CaL(real V, real d, real f, real f2, real fCass, real Ca_SS)   // !!!
{
    if (V < 15.0 - 1.0e-5)
    {
        return d_G_CaL * d * f * f2 * fCass * 4.0 * (V - 15.0) * (d_F*d_F) * (0.25 * Ca_SS * exp(2 * (V - 15.0) * d_FONRT) - d_Ca_o) / (d_R * d_T * (exp(2.0 * (V - 15.0) * d_FONRT) - 1.0));
    }
    else if (V > 15.0 + 1.0e-5)
    {
        return d_G_CaL * d * f * f2 * fCass * 2.0 * d_F * (0.25 * Ca_SS - d_Ca_o);
    }
}
__device__ real d_d_inf(real V)
{
    return 1.0 / (1.0 + exp((-8.0 - V) / 7.5));
}
__device__ real d_alpha_d(real V)
{
    return (1.4 / (1.0 + exp((-35.0 - V) / 13.0))) + 0.25;
}
__device__ real d_beta_d(real V)
{
    return 1.4 / (1.0 + exp((V + 5.0) / 5.0));
}
__device__ real d_gamma_d(real V)
{
    return 1.0 / (1.0 + exp((50.0 - V) / 20.0));
}
__device__ real d_tau_d(real V)
{
    return d_alpha_d(V) * d_beta_d(V) + d_gamma_d(V);
}
__device__ real d_f_inf(real V)
{
    return 1.0 / (1.0 + exp((V + 20.0) / 7.0));
}
__device__ real d_alpha_f(real V)
{
    return 1102.5 * exp(-(((V + 27.0)*(V + 27.0))) / 225.0);
}
__device__ real d_beta_f(real V)
{
    return 200.0 / (1.0 + exp((13.0 - V) / 10.0));
}
__device__ real d_gamma_f(real V)
{
    return 180.0 / (1.0 + exp((V + 30.0) / 10.0)) + 20.0;
}
__device__ real d_tau_f(real V)
{
    return d_alpha_f(V) + d_beta_f(V) + d_gamma_f(V);
}
__device__ real d_f2_inf(real V)
{
    return 0.67 / (1.0 + exp((V + 35.0) / 7.0)) + 0.33;
}
__device__ real d_alpha_f2(real V)   // !!!
{
    return 562.0 * exp(-(((V + 27.0)*(V + 27.0))) / 240.0);
}
__device__ real d_beta_f2(real V)
{
    return 31.0 / (1.0 + exp((25.0 - V) / 10.0));
}
__device__ real d_gamma_f2(real V)   // !!!
{
    return 80.0 / (1.0 + exp((V + 30.0) / 10.0));
}
__device__ real d_tau_f2(real V)
{
    return d_alpha_f2(V) + d_beta_f2(V) + d_gamma_f2(V);
}
__device__ real d_fCass_inf(real Ca_SS)
{
    return 0.6 / (1.0 + ((Ca_SS / 0.05)*(Ca_SS / 0.05))) + 0.4;
}
__device__ real d_tau_fCass(real Ca_SS)
{
    return 80.0 / (1.0 + ((Ca_SS / 0.05)*(Ca_SS / 0.05))) + 2.0;
}

// Transient outward current
__device__ real d_I_to(real V, real r, real s, real K_i)
{
    return d_G_to * r * s * (V - d_E_K(K_i));
}
__device__ real d_r_inf(real V)
{
    return 1.0 / (1.0 + exp((20.0 - V) / 6.0));
}
__device__ real d_tau_r(real V)
{
    return 9.5 * exp(-(((V + 40.0)*(V + 40.0))) / 1800.0) + 0.8;
}
#if defined(EPI) || defined(M)  // for epicardial and M cells
__device__ real d_s_inf(real V)
{
    return 1.0 / (1.0 + exp((V + 20.0) / 5.0));
}
__device__ real d_tau_s(real V)
{
    return 85.0 * exp(-(((V + 45.0)*(V + 45.0))) / 320.0) + 5.0 / (1.0 + exp((V - 20.0) / 5.0)) + 3.0;
}
#endif
#ifdef ENDO  // for endocardial cells
__device__ real d_s_inf(real V)
{
    return 1.0 / (1.0 + exp((V + 28.0) / 5.0));
}
__device__ real d_tau_s(real V)
{
    return 1000.0 * exp(-(((V + 67.0)*(V + 67.0))) / 1000.0) + 8.0;
}
#endif

// Slow delayed rectifier current
__device__ real d_I_Ks(real V, real X_s, real K_i, real Na_i)
{
    return d_G_Ks * (X_s*X_s) * (V - d_E_Ks(K_i, Na_i));
}
__device__ real d_x_s_inf(real V)
{
    return 1.0 / (1.0 + exp((-5.0 - V) / 14.0));
}
__device__ real d_alpha_x_s(real V)
{
    return 1400.0 / sqrt(1.0 + exp((5.0 - V) / 6.0));
}
__device__ real d_beta_x_s(real V)
{
    return 1.0 / (1.0 + exp((V - 35.0) / 15.0));
}
__device__ real d_tau_x_s(real V)
{
    return d_alpha_x_s(V) * d_beta_x_s(V) + 80.0;
}

// Rapid delayed rectifier current
__device__ real d_I_Kr(real V, real X_r1, real X_r2, real K_i)
{
    return d_G_Kr * sqrt(d_K_o / 5.4) * X_r1 * X_r2 * (V - d_E_K(K_i));
}
__device__ real d_x_r1_inf(real V)
{
    return 1.0 / (1.0 + exp((-26.0 - V) / 7.0));
}
__device__ real d_alpha_x_r1(real V)
{
    return 450.0 / (1.0 + exp((-45.0 - V) / 10.0));
}
__device__ real d_beta_x_r1(real V)
{
    return 6.0 / (1.0 + exp((V + 30.0) / 11.5));
}
__device__ real d_tau_x_r1(real V)
{
    return d_alpha_x_r1(V) * d_beta_x_r1(V);
}
__device__ real d_x_r2_inf(real V)
{
    return 1.0 / (1.0 + exp((V + 88.0) / 24.0));
}
__device__ real d_alpha_x_r2(real V)
{
    return 3.0 / (1.0 + exp((-60.0 - V) / 20.0));
}
__device__ real d_beta_x_r2(real V)
{
    return 1.12 / (1.0 + exp((V - 60.0) / 20.0));
}
__device__ real d_tau_x_r2(real V)
{
    return d_alpha_x_r2(V) * d_beta_x_r2(V);
}

// Inward rectifier K+ current
__device__ real d_alpha_K1(real V, real K_i)
{
    return 0.1 / (1.0 + exp(0.06 * (V - d_E_K(K_i) - 200.0)));
}
__device__ real d_beta_K1(real V, real K_i)
{
    real ek = d_E_K(K_i);
    return (3.0 * exp(0.0002 * (V - ek + 100.0)) + exp(0.1 * (V - ek - 10.0))) / (1.0 + exp(-0.5 * (V - ek)));
    // return (3.0 * exp(0.0002 * (V - d_E_K(K_i) + 100.0)) + exp(0.1 * (V - d_E_K(K_i) - 10.0))) / (1.0 + exp(-0.5 * (V - d_E_K(K_i))));
}
__device__ real d_x_K1_inf(real V, real K_i)
{
    return d_alpha_K1(V, K_i) / (d_alpha_K1(V, K_i) + d_beta_K1(V, K_i));
}
__device__ real d_I_K1(real V, real K_i)
{
    return d_G_K1 * d_x_K1_inf(V, K_i) * (V - d_E_K(K_i));
}

// Na+/Ca++ exchanger current
__device__ real d_I_NaCa(real V, real Na_i, real Ca_i)   // !!!
{
    return (d_k_NaCa * ((exp((d_gamma_I_NaCa * V * d_FONRT)) * (Na_i*Na_i*Na_i) * d_Ca_o) - (exp(((d_gamma_I_NaCa - 1.0) * V * d_FONRT)) * (d_Na_o*d_Na_o*d_Na_o) * Ca_i * d_alpha))) / (((d_K_mNa_i*d_K_mNa_i*d_K_mNa_i) + (d_Na_o*d_Na_o*d_Na_o)) * (d_K_mCa + d_Ca_o) * (1.0 + (d_k_sat * exp(((d_gamma_I_NaCa) * V * d_FONRT)))));
}

// Na+/K+ pump current
__device__ real d_I_NaK(real V, real Na_i) // !!!
{
    return ((((d_p_KNa * d_K_o) / (d_K_o + d_K_mK)) * Na_i) / (Na_i + d_K_mNa)) / (1.0 + (0.1245 * exp(((-0.1) * V * d_FONRT))) + (0.0353 * exp(((-V) * d_FONRT))));
}

// I_pCa
__device__ real d_I_pCa(real V, real Ca_i)
{
    return (d_G_pCa * Ca_i) / (d_K_pCa + Ca_i);
}

// I_pK
__device__ real d_I_pK(real V, real K_i)
{
    return (d_G_pK * (V - d_E_K(K_i))) / (1.0 + exp((25.0 - V) / 5.98));
}

// Background currents
__device__ real d_I_bNa(real V, real Na_i)
{
    return d_G_bNa * (V - d_E_Na(Na_i));
}
__device__ real d_I_bCa(real V, real Ca_i)
{
    return d_G_bCa * (V - d_E_Ca(Ca_i));
}

// Calcium dynamics
__device__ real d_I_leak(real Ca_SR, real Ca_i)
{
    return d_V_leak * (Ca_SR - Ca_i);
}
__device__ real d_I_up(real Ca_i)
{
    return d_V_maxup / (1.0 + ((d_K_up*d_K_up) / (Ca_i*Ca_i)));
}
__device__ real d_k_casr(real Ca_SR)
{
    return d_max_SR - ((d_max_SR - d_min_SR) / (1.0 + ((d_EC / Ca_SR)*(d_EC / Ca_SR))));
}
__device__ real d_k1(real Ca_SR)
{
    return d_k1_prime / d_k_casr(Ca_SR);
}
__device__ real d_O(real Ca_SR, real Ca_SS, real R_prime)
{
    return (d_k1(Ca_SR) * (Ca_SS*Ca_SS) * R_prime) / (d_k3 + (d_k1(Ca_SR) * (Ca_SS*Ca_SS)));
}
__device__ real d_I_rel(real Ca_SR, real Ca_SS, real R_prime)
{
    return d_V_rel * d_O(Ca_SR, Ca_SS, R_prime) * (Ca_SR - Ca_SS);
}
__device__ real d_I_xfer(real Ca_SS, real Ca_i)
{
    return d_V_xfer * (Ca_SS - Ca_i);
}
__device__ real d_k2(real Ca_SR)
{
    return d_k2_prime * d_k_casr(Ca_SR);
}
__device__ real d_Ca_ibufc(real Ca_i)    // !!!
{
    return 1.0 / (1.0 + ((d_Buf_C * d_K_bufc) / ((Ca_i + d_K_bufc)*(Ca_i + d_K_bufc))));
}
__device__ real d_Ca_srbufsr(real Ca_SR) // !!!
{
    return 1.0 / (1.0 + ((d_Buf_SR * d_K_bufsr) / ((Ca_SR + d_K_bufsr)*(Ca_SR + d_K_bufsr))));
}
__device__ real d_Ca_ssbufss(real Ca_SS) // !!!
{
    return 1.0 / (1.0 + ((d_Buf_SS * d_K_bufss) / ((Ca_SS + d_K_bufss)*(Ca_SS + d_K_bufss))));
}


/*-----------------------------------------------------
Differential equations for each variable
-----------------------------------------------------*/
__device__ real d_Itotal(real I_stim, real V, real m, real h, real j, real Na_i, real K_i, real r, real s, real X_r1, real X_r2, real X_s, real d, real f, real f2, real fCass, real Ca_SS, real Ca_i)
{
    real VmENa = V - d_E_Na(Na_i);
    real VmEK = V - d_E_K(K_i);

    real INa = d_G_Na * (m*m*m) * h * j * VmENa;
    real IbNa = d_G_bNa * VmENa;
    real IK1 = d_G_K1 * d_x_K1_inf(V, K_i) * VmEK;
    real Ito = d_G_to * r * s * VmEK;
    real IKr = d_G_Kr * sqrt(d_K_o / 5.4) * X_r1 * X_r2 * VmEK;
    real IKs = d_I_Ks(V, X_s, K_i, Na_i);
    real ICaL = d_I_CaL(V, d, f, f2, fCass, Ca_SS);
    real INaK = d_I_NaK(V, Na_i);
    real INaCa = d_I_NaCa(V, Na_i, Ca_i);
    real IpCa = d_I_pCa(V, Ca_i);
    real IpK = (d_G_pK * VmEK) / (1.0 + exp((25.0 - V) / 5.98));
    real IbCa = d_I_bCa(V, Ca_i);

    return I_stim + INa + IbNa + IK1 + Ito + IKr + IKs + ICaL + INaK + INaCa + IpCa + IpK + IbCa;
}
__device__ real d_dRprimedt(real Ca_SS, real R_prime)
{
    return ((-d_k2(Ca_SS)) * Ca_SS * R_prime) + (d_k4 * (1.0 - R_prime));
}
__device__ real d_dCaidt(real Ca_i, real Ca_SR, real Ca_SS, real V, real Na_i)
{
    return d_Ca_ibufc(Ca_i) * (((((d_I_leak(Ca_SR, Ca_i) - d_I_up(Ca_i)) * d_V_SR) / d_V_C) + d_I_xfer(Ca_SS, Ca_i)) - ((((d_I_bCa(V, Ca_i) + d_I_pCa(V, Ca_i)) - (2.0 * d_I_NaCa(V, Na_i, Ca_i))) * d_Cm) / (2.0 * d_V_C * d_F)));
}
__device__ real d_dCaSRdt(real Ca_SR, real Ca_i, real Ca_SS, real R_prime)
{
    return d_Ca_srbufsr(Ca_SR) * (d_I_up(Ca_i) - (d_I_rel(Ca_SR, Ca_SS, R_prime) + d_I_leak(Ca_SR, Ca_i)));
}
__device__ real d_dCaSSdt(real Ca_SS, real V, real d, real f, real f2, real fCass, real Ca_SR, real R_prime, real Ca_i)
{
    return d_Ca_ssbufss(Ca_SS) * (((((-d_I_CaL(V, d, f, f2, fCass, Ca_SS)) * d_Cm) / (2.0 * d_V_SS * d_F)) + ((d_I_rel(Ca_SR, Ca_SS, R_prime) * d_V_SR) / d_V_SS)) - ((d_I_xfer(Ca_SS, Ca_i) * d_V_C) / d_V_SS));
}
__device__ real d_dNaidt(real V, real m, real h, real j, real Na_i, real Ca_i)
{
    return ((-(d_I_Na(V, m, h, j, Na_i) + d_I_bNa(V, Na_i) + (3.0 * d_I_NaK(V, Na_i)) + (3.0 * d_I_NaCa(V, Na_i, Ca_i)))) / (d_V_C * d_F)) * d_Cm;
}
__device__ real d_dKidt(real I_stim, real V, real K_i, real r, real s, real X_r1, real X_r2, real X_s, real Na_i)
{
    return ((-((I_stim + d_I_K1(V, K_i) + d_I_to(V, r, s, K_i) + d_I_Kr(V, X_r1, X_r2, K_i) + d_I_Ks(V, X_s, K_i, Na_i) + d_I_pK(V, K_i)) - (2.0 * d_I_NaK(V, Na_i)))) / (d_V_C * d_F)) * d_Cm;
}


/*-----------------------------------------------------
Differential equations for each variable
-----------------------------------------------------*/
__device__ real d_updateXr1(real X_r1, real V, real dt)
{
    real xr1inf = d_x_r1_inf(V);
    return xr1inf - (xr1inf - X_r1) * exp(-dt / d_tau_x_r1(V));
}
__device__ real d_updateXr2(real X_r2, real V, real dt)
{
    real xr2inf = d_x_r2_inf(V);
    return xr2inf - (xr2inf - X_r2) * exp(-dt / d_tau_x_r2(V));
}
__device__ real d_updateXs(real X_s, real V, real dt)
{
    real xsinf = d_x_s_inf(V);
    return xsinf - (xsinf - X_s) * exp(-dt / d_tau_x_s(V));
}
__device__ real d_updater(real r, real V, real dt)
{
    real rinf = d_r_inf(V);
    return rinf - (rinf - r) * exp(-dt / d_tau_r(V));
}
__device__ real d_updates(real s, real V, real dt)
{
    real sinf = d_s_inf(V);
    return sinf - (sinf - s) * exp(-dt / d_tau_s(V));
}
__device__ real d_updatem(real m, real V, real dt)
{
    real minf = d_m_inf(V);
    return minf - (minf - m) * exp(-dt / d_tau_m(V));
}
__device__ real d_updateh(real h, real V, real dt)
{
    real hinf = d_h_inf(V);
    return hinf - (hinf - h) * exp(-dt / d_tau_h(V));
}
__device__ real d_updatej(real j, real V, real dt)
{
    real jinf = d_j_inf(V);
    return jinf - (jinf - j) * exp(-dt / d_tau_j(V));
}
__device__ real d_updated(real d, real V, real dt)
{
    real dinf = d_d_inf(V);
    return dinf - (dinf - d) * exp(-dt / d_tau_d(V));
}
__device__ real d_updatef(real f, real V, real dt)
{
    real finf = d_f_inf(V);
    return finf - (finf - f) * exp(-dt / d_tau_f(V));
}
__device__ real d_updatef2(real f2, real V, real dt)
{
    real f2inf = d_f2_inf(V);
    return f2inf - (f2inf - f2) * exp(-dt / d_tau_f2(V));
}
__device__ real d_updatefCass(real fCass, real V, real dt)
{
    real fCassinf = d_fCass_inf(V);
    return fCassinf - (fCassinf - fCass) * exp(-dt / d_tau_fCass(V));
}
#endif  // EPI || M || ENDO
#endif //TT2


#endif //CUDA_DEVICEFUNCS_H