#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "includes.h"


//############################################
//##                                        ##
//##     Adapted FitzHugh-Nagumo (AFHN)     ##
//##                                        ##
//############################################
#if defined(AFHN)
real reactionV(real v, real w)
{
    return (1.0 / (Cm * chi)) * ((-G * v * (1.0 - (v / vth)) * (1.0 - (v / vp))) + (-eta1 * v * w));
}

real reactionW(real v, real w)
{
    return eta2 * ((v / vp) - (eta3 * w));
}
#endif // AFHN



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
real E_Na(real Na_i)
{
    return RTONF * log(Na_o / Na_i);
}
real E_K(real K_i)
{
    return RTONF * log(K_o / K_i);
}
real E_Ca(real Ca_i)
{
    return 0.5 * RTONF * log(Ca_o / Ca_i);
}

// Reversal potential for Ks
real E_Ks(real K_i, real Na_i)
{
    return RTONF * log((K_o + p_KNa * Na_o) / (K_i + p_KNa * Na_i));
}

// Fast sodium (Na+) current
real I_Na(real V, real m, real h, real j, real Na_i)
{
    return G_Na * (m*m*m) * h * j * (V - E_Na(Na_i));
}
real m_inf(real V)
{
    return 1.0 / ((1.0 + exp((-56.86 - V) / 9.03))*(1.0 + exp((-56.86 - V) / 9.03)));
}
real alpha_m(real V)
{
    return 1.0 / (1.0 + exp((-60.0 - V) / 5.0));
}
real beta_m(real V)
{
    return (0.1 / (1.0 + exp((V + 35.0) / 5.0))) + (0.1 / (1.0 + exp((V - 50.0) / 200.0)));
}
real tau_m(real V)
{
    return alpha_m(V) * beta_m(V);
}
real h_inf(real V)
{
    return 1.0 / ((1.0 + exp((V + 71.55) / 7.43))*(1.0 + exp((V + 71.55) / 7.43)));
}
real alpha_h(real V)
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
real beta_h(real V)
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
real tau_h(real V)
{
    return 1.0 / (alpha_h(V) + beta_h(V));
}
real j_inf(real V)
{
    return 1.0 / ((1.0 + exp((V + 71.55) / 7.43))*(1.0 + exp((V + 71.55) / 7.43)));
}
real alpha_j(real V)
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
real beta_j(real V)
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
real tau_j(real V)
{
    return 1.0 / (alpha_j(V) + beta_j(V));
}

// L-type Ca2+ current
real I_CaL(real V, real d, real f, real f2, real fCass, real Ca_SS)   // !!!
{
    if (V < 15.0 - 1.0e-5)
    {
        return G_CaL * d * f * f2 * fCass * 4.0 * (V - 15.0) * (F*F) * (0.25 * Ca_SS * exp(2 * (V - 15.0) * FONRT) - Ca_o) / (R * T * (exp(2.0 * (V - 15.0) * FONRT) - 1.0));
    }
    else if (V > 15.0 + 1.0e-5)
    {
        return G_CaL * d * f * f2 * fCass * 2.0 * F * (0.25 * Ca_SS - Ca_o);
    }
}
real d_inf(real V)
{
    return 1.0 / (1.0 + exp((-8.0 - V) / 7.5));
}
real alpha_d(real V)
{
    return (1.4 / (1.0 + exp((-35.0 - V) / 13.0))) + 0.25;
}
real beta_d(real V)
{
    return 1.4 / (1.0 + exp((V + 5.0) / 5.0));
}
real gamma_d(real V)
{
    return 1.0 / (1.0 + exp((50.0 - V) / 20.0));
}
real tau_d(real V)
{
    return alpha_d(V) * beta_d(V) + gamma_d(V);
}
real f_inf(real V)
{
    return 1.0 / (1.0 + exp((V + 20.0) / 7.0));
}
real alpha_f(real V)
{
    return 1102.5 * exp(-(((V + 27.0)*(V + 27.0))) / 225.0);
}
real beta_f(real V)
{
    return 200.0 / (1.0 + exp((13.0 - V) / 10.0));
}
real gamma_f(real V)
{
    return 180.0 / (1.0 + exp((V + 30.0) / 10.0)) + 20.0;
}
real tau_f(real V)
{
    return alpha_f(V) + beta_f(V) + gamma_f(V);
}
real f2_inf(real V)
{
    return 0.67 / (1.0 + exp((V + 35.0) / 7.0)) + 0.33;
}
real alpha_f2(real V)   // !!!
{
    return 562.0 * exp(-(((V + 27.0)*(V + 27.0))) / 240.0);
}
real beta_f2(real V)
{
    return 31.0 / (1.0 + exp((25.0 - V) / 10.0));
}
real gamma_f2(real V)   // !!!
{
    return 80.0 / (1.0 + exp((V + 30.0) / 10.0));
}
real tau_f2(real V)
{
    return alpha_f2(V) + beta_f2(V) + gamma_f2(V);
}
real fCass_inf(real Ca_SS)
{
    return 0.6 / (1.0 + ((Ca_SS / 0.05)*(Ca_SS / 0.05))) + 0.4;
}
real tau_fCass(real Ca_SS)
{
    return 80.0 / (1.0 + ((Ca_SS / 0.05)*(Ca_SS / 0.05))) + 2.0;
}

// Transient outward current
real I_to(real V, real r, real s, real K_i)
{
    return G_to * r * s * (V - E_K(K_i));
}
real r_inf(real V)
{
    return 1.0 / (1.0 + exp((20.0 - V) / 6.0));
}
real tau_r(real V)
{
    return 9.5 * exp(-(((V + 40.0)*(V + 40.0))) / 1800.0) + 0.8;
}
#if defined(EPI) || defined(M)  // for epicardial and M cells
real s_inf(real V)
{
    return 1.0 / (1.0 + exp((V + 20.0) / 5.0));
}
real tau_s(real V)
{
    return 85.0 * exp(-(((V + 45.0)*(V + 45.0))) / 320.0) + 5.0 / (1.0 + exp((V - 20.0) / 5.0)) + 3.0;
}
#endif
#ifdef ENDO  // for endocardial cells
real s_inf(real V)
{
    return 1.0 / (1.0 + exp((V + 28.0) / 5.0));
}
real tau_s(real V)
{
    return 1000.0 * exp(-(((V + 67.0)*(V + 67.0))) / 1000.0) + 8.0;
}
#endif

// Slow delayed rectifier current
real I_Ks(real V, real X_s, real K_i, real Na_i)
{
    return G_Ks * (X_s*X_s) * (V - E_Ks(K_i, Na_i));
}
real x_s_inf(real V)
{
    return 1.0 / (1.0 + exp((-5.0 - V) / 14.0));
}
real alpha_x_s(real V)
{
    return 1400.0 / sqrt(1.0 + exp((5.0 - V) / 6.0));
}
real beta_x_s(real V)
{
    return 1.0 / (1.0 + exp((V - 35.0) / 15.0));
}
real tau_x_s(real V)
{
    return alpha_x_s(V) * beta_x_s(V) + 80.0;
}

// Rapid delayed rectifier current
real I_Kr(real V, real X_r1, real X_r2, real K_i)
{
    return G_Kr * sqrt(K_o / 5.4) * X_r1 * X_r2 * (V - E_K(K_i));
}
real x_r1_inf(real V)
{
    return 1.0 / (1.0 + exp((-26.0 - V) / 7.0));
}
real alpha_x_r1(real V)
{
    return 450.0 / (1.0 + exp((-45.0 - V) / 10.0));
}
real beta_x_r1(real V)
{
    return 6.0 / (1.0 + exp((V + 30.0) / 11.5));
}
real tau_x_r1(real V)
{
    return alpha_x_r1(V) * beta_x_r1(V);
}
real x_r2_inf(real V)
{
    return 1.0 / (1.0 + exp((V + 88.0) / 24.0));
}
real alpha_x_r2(real V)
{
    return 3.0 / (1.0 + exp((-60.0 - V) / 20.0));
}
real beta_x_r2(real V)
{
    return 1.12 / (1.0 + exp((V - 60.0) / 20.0));
}
real tau_x_r2(real V)
{
    return alpha_x_r2(V) * beta_x_r2(V);
}

// Inward rectifier K+ current
real alpha_K1(real V, real K_i)
{
    return 0.1 / (1.0 + exp(0.06 * (V - E_K(K_i) - 200.0)));
}
real beta_K1(real V, real K_i)
{
    return (3.0 * exp(0.0002 * (V - E_K(K_i) + 100.0)) + exp(0.1 * (V - E_K(K_i) - 10.0))) / (1.0 + exp(-0.5 * (V - E_K(K_i))));
}
real x_K1_inf(real V, real K_i)
{
    return alpha_K1(V, K_i) / (alpha_K1(V, K_i) + beta_K1(V, K_i));
}
real I_K1(real V, real K_i)
{
    return G_K1 * x_K1_inf(V, K_i) * (V - E_K(K_i));
}

// Na+/Ca++ exchanger current
real I_NaCa(real V, real Na_i, real Ca_i)   // !!!
{
    return (k_NaCa * ((exp((gamma_I_NaCa * V * FONRT)) * (Na_i*Na_i*Na_i) * Ca_o) - (exp(((gamma_I_NaCa - 1.0) * V * FONRT)) * (Na_o*Na_o*Na_o) * Ca_i * alpha))) / (((K_mNa_i*K_mNa_i*K_mNa_i) + (Na_o*Na_o*Na_o)) * (K_mCa + Ca_o) * (1.0 + (k_sat * exp(((gamma_I_NaCa) * V * FONRT)))));
}

// Na+/K+ pump current
real I_NaK(real V, real Na_i) // !!!
{
    return ((((p_KNa * K_o) / (K_o + K_mK)) * Na_i) / (Na_i + K_mNa)) / (1.0 + (0.1245 * exp(((-0.1) * V * FONRT))) + (0.0353 * exp(((-V) * FONRT))));
}

// I_pCa
real I_pCa(real V, real Ca_i)
{
    return (G_pCa * Ca_i) / (K_pCa + Ca_i);
}

// I_pK
real I_pK(real V, real K_i)
{
    return (G_pK * (V - E_K(K_i))) / (1.0 + exp((25.0 - V) / 5.98));
}

// Background currents
real I_bNa(real V, real Na_i)
{
    return G_bNa * (V - E_Na(Na_i));
}
real I_bCa(real V, real Ca_i)
{
    return G_bCa * (V - E_Ca(Ca_i));
}

// Calcium dynamics
real I_leak(real Ca_SR, real Ca_i)
{
    return V_leak * (Ca_SR - Ca_i);
}
real I_up(real Ca_i)
{
    return V_maxup / (1.0 + ((K_up*K_up) / (Ca_i*Ca_i)));
}
real k_casr(real Ca_SR)
{
    return max_SR - ((max_SR - min_SR) / (1.0 + ((EC / Ca_SR)*(EC / Ca_SR))));
}
real k1(real Ca_SR)
{
    return k1_prime / k_casr(Ca_SR);
}
real O(real Ca_SR, real Ca_SS, real R_prime)
{
    return (k1(Ca_SR) * (Ca_SS*Ca_SS) * R_prime) / (k3 + (k1(Ca_SR) * (Ca_SS*Ca_SS)));
}
real I_rel(real Ca_SR, real Ca_SS, real R_prime)
{
    return V_rel * O(Ca_SR, Ca_SS, R_prime) * (Ca_SR - Ca_SS);
}
real I_xfer(real Ca_SS, real Ca_i)
{
    return V_xfer * (Ca_SS - Ca_i);
}
real k2(real Ca_SR)
{
    return k2_prime * k_casr(Ca_SR);
}
real Ca_ibufc(real Ca_i)    // !!!
{
    return 1.0 / (1.0 + ((Buf_C * K_bufc) / ((Ca_i + K_bufc)*(Ca_i + K_bufc))));
}
real Ca_srbufsr(real Ca_SR) // !!!
{
    return 1.0 / (1.0 + ((Buf_SR * K_bufsr) / ((Ca_SR + K_bufsr)*(Ca_SR + K_bufsr))));
}
real Ca_ssbufss(real Ca_SS) // !!!
{
    return 1.0 / (1.0 + ((Buf_SS * K_bufss) / ((Ca_SS + K_bufss)*(Ca_SS + K_bufss))));
}


/*-----------------------------------------------------
Differential equations for each variable
-----------------------------------------------------*/
real Itotal(real I_stim, real V, real m, real h, real j, real Na_i, real K_i, real r, real s, real X_r1, real X_r2, real X_s, real d, real f, real f2, real fCass, real Ca_SS, real Ca_i)
{
    real VmENa = V - E_Na(Na_i);
    real VmEK = V - E_K(K_i);

    real INa = G_Na * (m*m*m) * h * j * VmENa;
    real IbNa = G_bNa * VmENa;
    real IK1 = G_K1 * x_K1_inf(V, K_i) * VmEK;
    real Ito = G_to * r * s * VmEK;
    real IKr = G_Kr * sqrt(K_o / 5.4) * X_r1 * X_r2 * VmEK;
    real IKs = I_Ks(V, X_s, K_i, Na_i);
    real ICaL = I_CaL(V, d, f, f2, fCass, Ca_SS);
    real INaK = I_NaK(V, Na_i);
    real INaCa = I_NaCa(V, Na_i, Ca_i);
    real IpCa = I_pCa(V, Ca_i);
    real IpK = (G_pK * VmEK) / (1.0 + exp((25.0 - V) / 5.98));
    real IbCa = I_bCa(V, Ca_i);

    return I_stim + INa + IbNa + IK1 + Ito + IKr + IKs + ICaL + INaK + INaCa + IpCa + IpK + IbCa;
}

real dRprimedt(real Ca_SS, real R_prime)
{
    return ((-k2(Ca_SS)) * Ca_SS * R_prime) + (k4 * (1.0 - R_prime));
}

real dCaidt(real Ca_i, real Ca_SR, real Ca_SS, real V, real Na_i)
{
    return Ca_ibufc(Ca_i) * (((((I_leak(Ca_SR, Ca_i) - I_up(Ca_i)) * V_SR) / V_C) + I_xfer(Ca_SS, Ca_i)) - ((((I_bCa(V, Ca_i) + I_pCa(V, Ca_i)) - (2.0 * I_NaCa(V, Na_i, Ca_i))) * Cm) / (2.0 * V_C * F)));
}

real dCaSRdt(real Ca_SR, real Ca_i, real Ca_SS, real R_prime)
{
    return Ca_srbufsr(Ca_SR) * (I_up(Ca_i) - (I_rel(Ca_SR, Ca_SS, R_prime) + I_leak(Ca_SR, Ca_i)));
}

real dCaSSdt(real Ca_SS, real V, real d, real f, real f2, real fCass, real Ca_SR, real R_prime, real Ca_i)
{
    return Ca_ssbufss(Ca_SS) * (((((-I_CaL(V, d, f, f2, fCass, Ca_SS)) * Cm) / (2.0 * V_SS * F)) + ((I_rel(Ca_SR, Ca_SS, R_prime) * V_SR) / V_SS)) - ((I_xfer(Ca_SS, Ca_i) * V_C) / V_SS));
}

real dNaidt(real V, real m, real h, real j, real Na_i, real Ca_i)
{
    return ((-(I_Na(V, m, h, j, Na_i) + I_bNa(V, Na_i) + (3.0 * I_NaK(V, Na_i)) + (3.0 * I_NaCa(V, Na_i, Ca_i)))) / (V_C * F)) * Cm;
}

real dKidt(real I_stim, real V, real K_i, real r, real s, real X_r1, real X_r2, real X_s, real Na_i)
{
    return ((-((I_stim + I_K1(V, K_i) + I_to(V, r, s, K_i) + I_Kr(V, X_r1, X_r2, K_i) + I_Ks(V, X_s, K_i, Na_i) + I_pK(V, K_i)) - (2.0 * I_NaK(V, Na_i)))) / (V_C * F)) * Cm;
}


/*-----------------------------------------------------
Differential equations for each variable
-----------------------------------------------------*/
real updateXr1(real X_r1, real V, real dt)
{
    real xr1inf = x_r1_inf(V);
    return xr1inf - (xr1inf - X_r1) * exp(-dt / tau_x_r1(V));
}

real updateXr2(real X_r2, real V, real dt)
{
    real xr2inf = x_r2_inf(V);
    return xr2inf - (xr2inf - X_r2) * exp(-dt / tau_x_r2(V));
}

real updateXs(real X_s, real V, real dt)
{
    real xsinf = x_s_inf(V);
    return xsinf - (xsinf - X_s) * exp(-dt / tau_x_s(V));
}

real updater(real r, real V, real dt)
{
    real rinf = r_inf(V);
    return rinf - (rinf - r) * exp(-dt / tau_r(V));
}

real updates(real s, real V, real dt)
{
    real sinf = s_inf(V);
    return sinf - (sinf - s) * exp(-dt / tau_s(V));
}

real updatem(real m, real V, real dt)
{
    real minf = m_inf(V);
    return minf - (minf - m) * exp(-dt / tau_m(V));
}

real updateh(real h, real V, real dt)
{
    real hinf = h_inf(V);
    return hinf - (hinf - h) * exp(-dt / tau_h(V));
}

real updatej(real j, real V, real dt)
{
    real jinf = j_inf(V);
    return jinf - (jinf - j) * exp(-dt / tau_j(V));
}

real updated(real d, real V, real dt)
{
    real dinf = d_inf(V);
    return dinf - (dinf - d) * exp(-dt / tau_d(V));
}

real updatef(real f, real V, real dt)
{
    real finf = f_inf(V);
    return finf - (finf - f) * exp(-dt / tau_f(V));
}

real updatef2(real f2, real V, real dt)
{
    real f2inf = f2_inf(V);
    return f2inf - (f2inf - f2) * exp(-dt / tau_f2(V));
}

real updatefCass(real fCass, real V, real dt)
{
    real fCassinf = fCass_inf(V);
    return fCassinf - (fCassinf - fCass) * exp(-dt / tau_fCass(V));
}
#endif  // EPI || M || ENDO
#endif  // TT2


#endif // FUNCTIONS_H