#ifndef CUDA_CONSTANTS_H
#define CUDA_CONSTANTS_H

__constant__ real d_stimStrength = 100.0;

__constant__ real d_stim1Begin = 0.0;    // Stimulation start time -> ms
__constant__ real d_stim1Duration = 2.0; // Stimulation duration -> ms

__constant__ real d_stim2Begin = 120.0;  // Stimulation start time -> ms
__constant__ real d_stim2Duration = 2.0; // Stimulation duration -> ms

//############################################
//##                                        ##
//##     Adapted FitzHugh-Nagumo (AFHN)     ##
//##                                        ##
//############################################
#if defined(AFHN)
__constant__ real d_G = 1.5;        // omega^-1 * cm^-2
__constant__ real d_eta1 = 4.4;     // omega^-1 * cm^-1
__constant__ real d_eta2 = 0.012;   // dimensionless
__constant__ real d_eta3 = 1.0;     // dimensionless
__constant__ real d_vth = 13.0;     // mV
__constant__ real d_vp = 100.0;     // mV
__constant__ real d_sigma = 1.2e-3; // omega^-1 * cm^-1

__constant__ real d_chi = 1.0e3; // cm^-1
__constant__ real d_Cm = 1.0e-3; // mF * cm^-2
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
__constant__ real d_chi = 1400.0;        // Surface area-to-volume ratio -> cm^-1
__constant__ real d_Cm = 0.185;          // Cell capacitance per unit surface area -> uF/ (???)^2 (ten Tusscher)


/*----------------
Parameters
----------------*/
// Constants
__constant__ real d_R = 8314.472;        // Gas constant -> (???) [8.314472 J/(K*mol)]
__constant__ real d_T = 310.0;           // Temperature -> K
__constant__ real d_F = 96485.3415;      // Faraday constant -> (???) [96.4867 C/mmol]
__constant__ real d_RTONF = 26.713761;   // R*T/F -> (???)
__constant__ real d_FONRT = 0.037434;    // F/(R*T) -> (???)

// Intracellular volumes
__constant__ real d_V_C = 0.016404;      // Cellular volume -> (???) [16404 um^3]
__constant__ real d_V_SR = 0.001094;     // Sarcoplasmic reticulum volume -> (???) [1094 um^3]
__constant__ real d_V_SS = 0.00005468;   // Subsarcolemmal space volume -> (???) [54.68 um^3]

// External concentrations
__constant__ real d_K_o = 5.4;           // Extracellular potassium (K+) concentration -> mM
__constant__ real d_Na_o = 140;          // Extracellular sodium (Na+) concentration -> mM
__constant__ real d_Ca_o = 2.0;          // Extracellular calcium (Ca++) concentration -> mM

// Parameters for currents
__constant__ real d_G_Na = 14.838;       // Maximal I_Na (sodium current) conductance -> nS/pF
__constant__ real d_G_K1 = 5.405;        // Maximal I_K1 (late rectifier potassium current) conductance -> nS/pF
#if defined(EPI) || defined(M)
__constant__ real d_G_to = 0.294;        // Maximal I_to (transient outward potassium current) conductance -> nS/pF (epi and M cells)
#endif
#ifdef ENDO
__constant__ real d_G_to = 0.073;        // Maximal I_to (transient outward potassium current) conductance -> nS/pF (endo cells)
#endif
__constant__ real d_G_Kr = 0.153;        // Maximal I_Kr (rapidly activating delayed rectifier potassium current) conductance -> nS/pF
#if defined(EPI) || defined(ENDO)
__constant__ real d_G_Ks = 0.392;        // Maximal I_Ks (slowly activating delayed rectifier potassium current) conductance -> nS/pF (epi and endo cells)
#endif
#ifdef M
__constant__ real d_G_Ks = 0.098;        // Maximal I_Ks (slowly activating delayed rectifier potassium current) conductance -> nS/pF (M cells)
#endif
__constant__ real d_p_KNa = 0.03;        // Relative I_Ks permeability to Na+ over K+ -> dimensionless
__constant__ real d_G_CaL = 3.98e-5;     // Maximal I_CaL (L-type calcium current) conductance -> cm/ms/uF
__constant__ real d_k_NaCa = 1000.0;     // Maximal I_NaCa (Na+/Ca++ exchanger current) -> pA/pF
__constant__ real d_gamma_I_NaCa = 0.35; // Voltage dependence parameter of I_NaCa -> dimensionless
__constant__ real d_K_mCa = 1.38;        // Half-saturation constant of I_NaCa for intracellular Ca++ -> mM
__constant__ real d_K_mNa_i = 87.5;      // Half-saturation constant of I_NaCa for intracellular Na+ -> mM
__constant__ real d_k_sat = 0.1;         // Saturation factor for I_NaCa -> dimensionless
__constant__ real d_alpha = 2.5;         // Factor enhancing outward nature of I_NaCa -> dimensionless
__constant__ real d_P_NaK = 2.724;       // Maximal I_NaK (Na+/K+ pump current) -> pA/pF
__constant__ real d_K_mK = 1.0;          // Half-saturation constant of I_NaK for Ko -> mM
__constant__ real d_K_mNa = 40.0;        // Half-saturation constant of I_NaK for intracellular Na+ -> mM
__constant__ real d_G_pK = 0.0146;       // Maximal I_pK (plateau potassium current) conductance -> nS/pF
__constant__ real d_G_pCa = 0.1238;      // Maximal I_pCa (plateau calcium current) conductance -> nS/pF
__constant__ real d_K_pCa = 0.0005;      // Half-saturation constant of I_pCa for intracellular Ca++ -> mM
__constant__ real d_G_bNa = 0.00029;     // Maximal I_bNa (sodium background current) conductance -> nS/pF
__constant__ real d_G_bCa = 0.000592;    // Maximal I_bCa (calcium background current) conductance -> nS/pF

// Intracellular calcium flux dynamics
__constant__ real d_V_maxup = 0.006375;  // Maximal I_up -> mM/ms
__constant__ real d_K_up = 0.00025;      // Half-saturation constant of I_up -> mM
__constant__ real d_V_rel = 0.102;       // Maximal I_rel conductance -> mM/ms
__constant__ real d_k1_prime = 0.15;     // R to O and RI to I I_rel transition rate -> mM^-2*ms^-1
__constant__ real d_k2_prime = 0.045;    // O to I  and R to RI I_rel transition rate -> mM^-1*ms^-1
__constant__ real d_k3 = 0.06;           // O to R and I to RI I_rel transition rate -> ms^-1
__constant__ real d_k4 = 0.005;          // I to O and RI to I I_rel transition rate -> ms^-1
__constant__ real d_EC = 1.5;            // Half-saturation constant of k_Ca_SR -> mM
__constant__ real d_max_SR = 2.5;        // Maximum value of k_Ca_SR -> dimensionless
__constant__ real d_min_SR = 1.0;        // Minimum value of k_Ca_SR -> dimensionless
__constant__ real d_V_leak = 0.00036;    // Maximal I_leak conductance -> mM/ms
__constant__ real d_V_xfer = 0.0038;     // Maximal I_xfer conductance -> mM/ms

// Calcium buffering dynamics
__constant__ real d_Buf_C = 0.2;         // Total cytoplasmic buffer concentration -> mM
__constant__ real d_K_bufc = 0.001;      // Half-saturation constant of cytoplasmic buffers -> mM
__constant__ real d_Buf_SR = 10.0;       // Total sarcoplasmic reticulum buffer concentration -> mM
__constant__ real d_K_bufsr = 0.3;       // Half-saturation constant of sarcoplasmic reticulum buffers -> mM
__constant__ real d_Buf_SS = 0.4;        // Total subspace buffer concentration -> mM
__constant__ real d_K_bufss = 0.00025;   // Half-saturation constant of subspace buffer -> mM
#endif // EPI || M || ENDO
#endif // TT2

#endif // CUDA_CONSTANTS_H