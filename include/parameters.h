#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "includes.h"


//############################################
//##                                        ##
//##         Simulation parameters          ##
//##                                        ##
//############################################
real L = 2;             // Length of each side (cm)
real deltax = 0.002;    // Spatial step (x) -> cm
real deltay = 0.002;    // Spatial step (y) -> cm
real deltaz = 0.002;    // Spatial step (z) -> cm
#if defined(AFHN)
real T = 300;
#endif // AFHN
#if defined(TT2)
real Time = 600.0;         // Simulation time -> ms
#endif // TT2


//############################################
//##                                        ##
//##         Stimulation parameters         ##
//##                                        ##
//############################################
#if defined(AFHN)
real stimStrength = 100.0;
#endif // AFHN
#if defined(TT2)
real stimStrength = -38.0;         
#endif //TT2

real stim1Begin = 0.0;            // Stimulation start time -> ms
real stim1Duration = 2.0;         // Stimulation duration -> ms
real stim1xLimit = 0.2;           // Stimulation x limit -> cm
real stim1yLimit = 2.0;           // Stimulation y limit -> cm ( = L)

#if defined(AFHN)
real stim2Begin = 120.0;
#endif // AFHN
#if defined(TT2)
real stim2Begin = 310.0;          // Stimulation start time -> ms
#endif //TT2
real stim2Duration = 2.0;         // Stimulation duration -> ms
real stim2xMax = 1.0;             // Stimulation x max -> cm
real stim2yMax = 1.0;             // Stimulation y max -> cm
real stim2xMin = 0.0;             // Stimulation x min -> cm
real stim2yMin = 0.0;             // Stimulation y min -> cm



//############################################
//##                                        ##
//##         Fibrosis parameters            ##
//##                                        ##
//############################################
real fibrosisFactor = 0.2;        // Fibrosis rate -> dimensionless
real fibrosisMinX = 0.7;          // Fibrosis x min -> cm
real fibrosisMaxX = 1.3;          // Fibrosis x max -> cm
real fibrosisMinY = 0.7;          // Fibrosis y min -> cm
real fibrosisMaxY = 1.3;          // Fibrosis y max -> cm



//############################################
//##                                        ##
//##     Adapted FitzHugh-Nagumo (AFHN)     ##
//##                                        ##
//############################################
#if defined(AFHN)
/*----------------------------
Model parameters
Based on Gerardo_Giorda 2007
----------------------------*/
real G = 1.5;         // omega^-1 * cm^-2
real eta1 = 4.4;      // omega^-1 * cm^-1
real eta2 = 0.012;    // dimensionless
real eta3 = 1.0;      // dimensionless
real vth = 13.0;      // mV
real vp = 100.0;      // mV
real sigma = 1.2e-3;  // omega^-1 * cm^-1

real chi = 1.0e3;     // cm^-1
real Cm = 1.0e-3;     // mF * cm^-2

/*----------------
Initial Conditions
----------------*/
real V_init = 0.0;    // Initial membrane potential -> mV
real W_init = 0.0;    // Initial recovery variable -> dimensionless
#endif // AFHN



//###########################################
//##                                       ##
//##     ten Tusscher 2006 model (TT2)     ##
//##                                       ##
//###########################################
#if defined(TT2)
#if defined(EPI) || defined(M) || defined(ENDO)
/*------------------------------------------------------------------------------------------------------------------------------------------------------
Parameters for ten Tusscher model 2006 (https://journals.physiology.org/doi/full/10.1152/ajpheart.00109.2006)
from https://tbb.bio.uu.nl/khwjtuss/SourceCodes/HVM2/Source/Main.cc - ten Tusscher code
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3263775/ - Benchmark
and https://github.com/rsachetto/MonoAlg3D_C/blob/master/src/models_library/ten_tusscher/ten_tusscher_2006_RS_CPU.c - Sachetto MonoAlg3D
--------------------------------------------------------------------------------------------------------------------------------------------------------*/

/*----------------
Model parameters
----------------*/
real chi = 1400.0;        // Surface area-to-volume ratio -> cm^-1
real Cm = 0.185;          // Cell capacitance per unit surface area -> uF/ (???)^2 (ten Tusscher)
real sigma = 1.171;               // Diffusion coefficient -> cm²/s

/*----------------
Parameters
----------------*/
// Constants
real R = 8314.472;        // Gas constant -> (???) [8.314472 J/(K*mol)]
real T = 310.0;           // Temperature -> K
real F = 96485.3415;      // Faraday constant -> (???) [96.4867 C/mmol]
real RTONF = 26.713761;   // R*T/F -> (???)
real FONRT = 0.037434;    // F/(R*T) -> (???)

// Intracellular volumes
real V_C = 0.016404;      // Cellular volume -> (???) [16404 um^3]
real V_SR = 0.001094;     // Sarcoplasmic reticulum volume -> (???) [1094 um^3]
real V_SS = 0.00005468;   // Subsarcolemmal space volume -> (???) [54.68 um^3]

// External concentrations
real K_o = 5.4;           // Extracellular potassium (K+) concentration -> mM
real Na_o = 140;          // Extracellular sodium (Na+) concentration -> mM
real Ca_o = 2.0;          // Extracellular calcium (Ca++) concentration -> mM

// Parameters for currents
real G_Na = 14.838;       // Maximal I_Na (sodium current) conductance -> nS/pF
real G_K1 = 5.405;        // Maximal I_K1 (late rectifier potassium current) conductance -> nS/pF
#if defined(EPI) || defined(M)
real G_to = 0.294;        // Maximal I_to (transient outward potassium current) conductance -> nS/pF (epi and M cells)
#endif
#ifdef ENDO
real G_to = 0.073;        // Maximal I_to (transient outward potassium current) conductance -> nS/pF (endo cells)
#endif
real G_Kr = 0.153;        // Maximal I_Kr (rapidly activating delayed rectifier potassium current) conductance -> nS/pF
#if defined(EPI) || defined(ENDO)
real G_Ks = 0.392;        // Maximal I_Ks (slowly activating delayed rectifier potassium current) conductance -> nS/pF (epi and endo cells)
#endif
#ifdef M
real G_Ks = 0.098;        // Maximal I_Ks (slowly activating delayed rectifier potassium current) conductance -> nS/pF (M cells)
#endif
real p_KNa = 0.03;        // Relative I_Ks permeability to Na+ over K+ -> dimensionless
real G_CaL = 3.98e-5;     // Maximal I_CaL (L-type calcium current) conductance -> cm/ms/uF
real k_NaCa = 1000.0;     // Maximal I_NaCa (Na+/Ca++ exchanger current) -> pA/pF
real gamma_I_NaCa = 0.35; // Voltage dependence parameter of I_NaCa -> dimensionless
real K_mCa = 1.38;        // Half-saturation constant of I_NaCa for intracellular Ca++ -> mM
real K_mNa_i = 87.5;      // Half-saturation constant of I_NaCa for intracellular Na+ -> mM
real k_sat = 0.1;         // Saturation factor for I_NaCa -> dimensionless
real alpha = 2.5;         // Factor enhancing outward nature of I_NaCa -> dimensionless
real P_NaK = 2.724;       // Maximal I_NaK (Na+/K+ pump current) -> pA/pF
real K_mK = 1.0;          // Half-saturation constant of I_NaK for Ko -> mM
real K_mNa = 40.0;        // Half-saturation constant of I_NaK for intracellular Na+ -> mM
real G_pK = 0.0146;       // Maximal I_pK (plateau potassium current) conductance -> nS/pF
real G_pCa = 0.1238;      // Maximal I_pCa (plateau calcium current) conductance -> nS/pF
real K_pCa = 0.0005;      // Half-saturation constant of I_pCa for intracellular Ca++ -> mM
real G_bNa = 0.00029;     // Maximal I_bNa (sodium background current) conductance -> nS/pF
real G_bCa = 0.000592;    // Maximal I_bCa (calcium background current) conductance -> nS/pF

// Intracellular calcium flux dynamics
real V_maxup = 0.006375;  // Maximal I_up -> mM/ms
real K_up = 0.00025;      // Half-saturation constant of I_up -> mM
real V_rel = 0.102;       // Maximal I_rel conductance -> mM/ms
real k1_prime = 0.15;     // R to O and RI to I I_rel transition rate -> mM^-2*ms^-1
real k2_prime = 0.045;    // O to I  and R to RI I_rel transition rate -> mM^-1*ms^-1
real k3 = 0.06;           // O to R and I to RI I_rel transition rate -> ms^-1
real k4 = 0.005;          // I to O and RI to I I_rel transition rate -> ms^-1
real EC = 1.5;            // Half-saturation constant of k_Ca_SR -> mM
real max_SR = 2.5;        // Maximum value of k_Ca_SR -> dimensionless
real min_SR = 1.0;        // Minimum value of k_Ca_SR -> dimensionless
real V_leak = 0.00036;    // Maximal I_leak conductance -> mM/ms
real V_xfer = 0.0038;     // Maximal I_xfer conductance -> mM/ms

// Calcium buffering dynamics
real Buf_C = 0.2;         // Total cytoplasmic buffer concentration -> mM
real K_bufc = 0.001;      // Half-saturation constant of cytoplasmic buffers -> mM
real Buf_SR = 10.0;       // Total sarcoplasmic reticulum buffer concentration -> mM
real K_bufsr = 0.3;       // Half-saturation constant of sarcoplasmic reticulum buffers -> mM
real Buf_SS = 0.4;        // Total subspace buffer concentration -> mM
real K_bufss = 0.00025;   // Half-saturation constant of subspace buffer -> mM


/*-----------------------------------------------------------
Initial Conditions for epicardium cells
from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3263775/
------------------------------------------------------------*/
#ifdef EPI
real V_init = -85.23;       // Initial membrane potential -> mV
real X_r1_init = 0.00621;   // Initial rapid time-dependent potassium current Xr1 gate -> dimensionless
real X_r2_init = 0.4712;    // Initial rapid time-dependent potassium current Xr2 gate -> dimensionless
real X_s_init = 0.0095;     // Initial slow time-dependent potassium current Xs gate -> dimensionless
real m_init = 0.00172;      // Initial fast sodium current m gate -> dimensionless
real h_init = 0.7444;       // Initial fast sodium current h gate -> dimensionless
real j_init = 0.7045;       // Initial fast sodium current j gate -> dimensionless
real d_init = 3.373e-5;     // Initial L-type calcium current d gate -> dimensionless
real f_init = 0.7888;       // Initial L-type calcium current f gate -> dimensionless
real f2_init = 0.9755;      // Initial L-type calcium current f2 gate -> dimensionless
real fCass_init = 0.9953;   // Initial L-type calcium current fCass gate -> dimensionless
real s_init = 0.999998;     // Initial transient outward current s gate -> dimensionless
real r_init = 2.42e-8;      // Initial transient outward current r gate -> dimensionless
real Ca_i_init = 0.000126;  // Initial intracellular Ca++ concentration -> mM
real Ca_SR_init = 3.64;     // Initial sarcoplasmic reticulum Ca++ concentration -> mM
real Ca_SS_init = 0.00036;  // Initial subspace Ca++ concentration -> mM
real R_prime_init = 0.9073; // Initial ryanodine receptor -> dimensionless
real Na_i_init = 8.604;     // Initial intracellular Na+ concentration -> mM
real K_i_init = 136.89;     // Initial intracellular K+ concentration -> mM
#endif

/*-----------------------------------------------------
Initial Conditions for endocardium or M cells
from https://tbb.bio.uu.nl/khwjtuss/SourceCodes/HVM2/
-----------------------------------------------------*/
#if defined(ENDO) || defined(M)
real V_init = -86.2;        // Initial membrane potential -> mV
real X_r1_init = 0.0;       // Initial rapid time-dependent potassium current Xr1 gate -> dimensionless
real X_r2_init = 1.0;       // Initial rapid time-dependent potassium current Xr2 gate -> dimensionless
real X_s_init = 0.0;        // Initial slow time-dependent potassium current Xs gate -> dimensionless
real m_init = 0.0;          // Initial fast sodium current m gate -> dimensionless
real h_init = 0.75;         // Initial fast sodium current h gate -> dimensionless
real j_init = 0.75;         // Initial fast sodium current j gate -> dimensionless
real d_init = 0.0;          // Initial L-type calcium current d gate -> dimensionless
real f_init = 1.0;          // Initial L-type calcium current f gate -> dimensionless
real f2_init = 1.0;         // Initial L-type calcium current f2 gate -> dimensionless
real fCass_init = 1.0;      // Initial L-type calcium current fCass gate -> dimensionless
real s_init = 1.0;          // Initial transient outward current s gate -> dimensionless
real r_init = 0.0;          // Initial transient outward current r gate -> dimensionless
real Ca_i_init = 0.00007;   // Initial intracellular Ca++ concentration -> mM
real Ca_SR_init = 1.3;      // Initial sarcoplasmic reticulum Ca++ concentration -> mM
real Ca_SS_init = 0.00007;  // Initial subspace Ca++ concentration -> mM
real R_prime_init = 1.0;    // Initial ryanodine receptor -> dimensionless
real Na_i_init = 7.67;      // Initial intracellular Na+ concentration -> mM
real K_i_init = 138.3;      // Initial intracellular K+ concentration -> mM
#endif
#endif // EPI || M || ENDO
#endif // TT2

#endif // PARAMETERS_H