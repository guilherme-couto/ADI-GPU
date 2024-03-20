#ifndef METHODS_H
#define METHODS_H

#include "includes.h"

// For stimulation
real Istim = 0.0;

// For time step
int timeStepCounter = 0;
real timeStep = 0.0;

// For execution time
real startTotal = 0.0, finishTotal = 0.0, elapsedTotal = 0.0;
real startPartial = 0.0, finishPartial = 0.0;
real elapsedODE = 0.0, elapsedPDE = 0.0, elapsedMemCopy = 0.0;
real elapsed1stMemCopy = 0.0;
real elapsed2ndMemCopy = 0.0;
real elapsed3rdMemCopy = 0.0;
real elapsed4thMemCopy = 0.0;
real elapsed1stThomas = 0.0;
real elapsed2ndThomas = 0.0;
real elapsed3rdThomas = 0.0;
real elapsedTranspose = 0.0;
real elapsedTranspose2 = 0.0;
real elapsedTranspose3 = 0.0;
real elapsedWriting = 0.0;
real elapsed1stRHS = 0.0, elapsed2ndRHS = 0.0;
real lastCheckpointTime = 0.0;

// For velocity
bool S1VelocityTag = true;
real S1Velocity = 0.0;

// For vulnerability window
bool VWTag = false;
real measureVWFrom = 117.0, measureVWTo = 132.0;
real lowerVWBound = 999.0, upperVWBound = 0.0;

// ############################################
// ##                                        ##
// ##     Adapted FitzHugh-Nagumo (AFHN)     ##
// ##                                        ##
// ############################################
#if defined(AFHN)
void runAllinCPU(bool options[], char *method, real deltat, int numberThreads, real delta_x, char *mode, real theta, int number_of_exec)
{
    // Get options
    bool haveFibrosis = options[0];
    bool measureTotalTime = options[1];
    bool saveDataToError = options[2];
    bool saveDataToGif = options[3];
    bool measureTimeParts = options[4];
    bool measureS1Velocity = options[5];

    deltax = delta_x;
    deltay = delta_x;

    // Number of steps
    int N = round(L / deltax) + 1;                  // Spatial steps (square tissue)
    int M = round(T / deltat) + 1;               // Number of time steps
    int PdeOdeRatio = round(deltat / deltat); // Ratio between PDE and ODE time steps

    // Allocate and populate time array
    real *time;
    time = (real *)malloc(M * sizeof(real));
    for (int i = 0; i < M; i++)
    {
        time[i] = i * deltat;
    }

    // Allocate and initialize variables
    real **V, **W;
    V = (real **)malloc(N * sizeof(real *));
    W = (real **)malloc(N * sizeof(real *));

    for (int i = 0; i < N; i++)
    {
        V[i] = (real *)malloc(N * sizeof(real));
        W[i] = (real *)malloc(N * sizeof(real));
    }
    initializeVariables(N, V, W);

    // Diffusion coefficient - isotropic
    real D = sigma / (chi * Cm);
    real phi = D * deltat / (deltax * deltax); // For Thomas algorithm - isotropic

    // Variables
    int i, j; // i for y-axis and j for x-axis
    real actualV, actualW, diffusion, actualRHS_V, actualRHS_W, tildeRHS_V, tildeRHS_W;
    real Rw;
    real **Vtilde, **Wtilde, **Rv, **rightside, **solution;
    Vtilde = (real **)malloc(N * sizeof(real *));
    Wtilde = (real **)malloc(N * sizeof(real *));
    Rv = (real **)malloc(N * sizeof(real *));
    rightside = (real **)malloc(N * sizeof(real *));
    solution = (real **)malloc(N * sizeof(real *));

    // Auxiliary arrays for Thomas algorithm 2nd order approximation
    real **c_ = (real **)malloc((N) * sizeof(real *));
    real **d_ = (real **)malloc((N) * sizeof(real *));
    for (int i = 0; i < N; i++)
    {
        Vtilde[i] = (real *)malloc(N * sizeof(real));
        Wtilde[i] = (real *)malloc(N * sizeof(real));
        Rv[i] = (real *)malloc(N * sizeof(real));
        rightside[i] = (real *)malloc(N * sizeof(real));
        solution[i] = (real *)malloc(N * sizeof(real));
        c_[i] = (real *)malloc(N * sizeof(real));
        d_[i] = (real *)malloc(N * sizeof(real));
    }

    // Discretized limits of stimulation area
    int discS1xLimit = round(stim1xLimit / deltax);
    int discS1yLimit = round(stim1yLimit / deltay);
    int discS2xMax = round(stim2xMax / deltax);
    int discS2xMin = round(stim2xMin / deltax);
    int discS2yMax = N;
    int discS2yMin = N - round(stim2yMax / deltay);

    // Discritized limits of fibrotic area
    int discFibxMax = round(fibrosisMaxX / deltax);
    int discFibxMin = round(fibrosisMinX / deltax);
    int discFibyMax = N - round(fibrosisMinY / deltay);
    int discFibyMin = N - round(fibrosisMaxY / deltay);

    // File names
    char framesFileName[MAX_STRING_SIZE], infosFileName[MAX_STRING_SIZE];
    sprintf(framesFileName, "frames-%d-%.3lf.txt", numberThreads, deltat);
    sprintf(infosFileName, "infos-%d-%.3lf-%.2lf.txt", numberThreads, deltat, theta);
    int saverate = ceil(M / 100.0);
    FILE *fpFrames, *fpInfos;

    // Create directories and files
    char pathToSaveData[MAX_STRING_SIZE];
    if (haveFibrosis)
    {
        // createDirectories(pathToSaveData, method, "AFHN-Fibro", mode);
        createDirectoriesScript(pathToSaveData, method, "AFHN-Fibro", mode, numberThreads, number_of_exec);
    }
    else
    {
        // createDirectories(pathToSaveData, method, "AFHN", mode);
        createDirectoriesScript(pathToSaveData, method, "AFHN", mode, numberThreads, number_of_exec);
    }

    // File pointers
    char aux[MAX_STRING_SIZE];
    if (VWTag == false)
    {

        sprintf(aux, "%s/%s", pathToSaveData, infosFileName);
        fpInfos = fopen(aux, "w");
    }
    else
    {
        sprintf(aux, "%s/%s", pathToSaveData, infosFileName);
        fpInfos = fopen(aux, "a");
    }
    if (saveDataToGif == false)
    {
        sprintf(aux, "%s/%s", pathToSaveData, framesFileName);
        fpFrames = fopen(aux, "a");
    }
    else
    {
        sprintf(aux, "%s/%s", pathToSaveData, framesFileName);
        fpFrames = fopen(aux, "w");
    }

    /*--------------------
    --  ADI 1st order   --
    ----------------------*/
    if (strcmp(method, "OS-ADI") == 0)
    {
        // Start measuring total execution time
        startTotal = omp_get_wtime();

        #pragma omp parallel num_threads(numberThreads) default(none) private(i, j, Istim, actualV, actualW) \
        shared(V, W, N, M, L, T, D, phi, deltat, time, timeStep, timeStepCounter, PdeOdeRatio, \
        fibrosisFactor, stimStrength, stim1Duration, stim2Duration, stim1Begin, stim2Begin, stim1xLimit, stim1yLimit, \
        discS1xLimit, discS1yLimit, discS2xMax, discS2yMax, discS2xMin, discS2yMin, discFibxMax, discFibxMin, discFibyMax, discFibyMin, \
        c_, d_, Vtilde, Wtilde, S1Velocity, S1VelocityTag, VWTag, saverate, fpFrames, saveDataToGif, \
        Rv, rightside, solution, startPartial, finishPartial, elapsedODE, elapsedPDE, \
        elapsedWriting, elapsed1stThomas, elapsed2ndThomas)
        {
            while (timeStepCounter < M)
            {
                // Get time step
                timeStep = time[timeStepCounter];

                // Start measuring ODE execution time
                #pragma omp master
                {
                    startPartial = omp_get_wtime();
                }

                // Resolve ODEs
                #pragma omp for collapse(2)
                for (i = 0; i < N; i++)
                {
                    for (j = 0; j < N; j++)
                    {
                        // Stimulus
                        Istim = stimulus(i, j, timeStep, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax);

                        // Get actual V and W
                        actualV = V[i][j];
                        actualW = W[i][j];

                        // Update V and W without diffusion
                        V[i][j] = actualV + deltat * (reactionV(actualV, actualW) + Istim);
                        W[i][j] = actualW + deltat * reactionW(actualV, actualW);

                        // Update right side of Thomas algorithm
                        rightside[j][i] = V[i][j];
                    }
                }
                

                // Finish measuring ODE execution time
                #pragma omp master
                {
                    finishPartial = omp_get_wtime();
                    elapsedODE += finishPartial - startPartial;
                    startPartial = omp_get_wtime();
                }

                // Resolve PDEs (Diffusion)
                // 1st: Implicit y-axis diffusion (lines)
                #pragma omp barrier
                #pragma omp for
                for (i = 0; i < N; i++)
                {
                    // Check if i is in fibrotic region
                    if (i >= discFibyMin && i <= discFibyMax)
                    {
                        ThomasAlgorithm2ndCPU(rightside[i], solution[i], N, phi, c_[i], d_[i], discFibxMin, discFibxMax, fibrosisFactor);
                    }
                    else
                    {
                        ThomasAlgorithm2ndCPU(rightside[i], solution[i], N, phi, c_[i], d_[i], N, 0, fibrosisFactor);
                    }

                    // Update V
                    for (j = 0; j < N; j++)
                    {
                        Vtilde[j][i] = solution[i][j];
                    }
                }

                // Finish measuring 1st Thomas algorithm execution time and start measuring 2nd Thomas algorithm execution time
                #pragma omp master
                {
                    finishPartial = omp_get_wtime();
                    elapsed1stThomas += finishPartial - startPartial;
                    startPartial = omp_get_wtime();
                }

                // 2nd: Implicit x-axis diffusion (columns)
                #pragma omp barrier
                #pragma omp for
                for (i = 0; i < N; i++)
                {
                    // Check if i is in fibrotic region
                    if (i >= discFibxMin && i <= discFibxMax)
                    {
                        ThomasAlgorithm2ndCPU(Vtilde[i], V[i], N, phi, c_[i], d_[i], discFibyMin, discFibyMax, fibrosisFactor);
                    }
                    else
                    {
                        ThomasAlgorithm2ndCPU(Vtilde[i], V[i], N, phi, c_[i], d_[i], N, 0, fibrosisFactor);
                    }
                }
                
                // Finish measuring PDE execution time
                #pragma omp master
                {                  
                    finishPartial = omp_get_wtime();
                    elapsed2ndThomas += finishPartial - startPartial;
                }

                // Save frames
                #pragma omp master
                {
                    if (VWTag == false)
                    {
                        // Write frames to file
                        // startPartial = omp_get_wtime();
                        // if (timeStepCounter % saverate == 0 && saveDataToGif == true)
                        // {
                        //     fprintf(fpFrames, "%lf\n", time[timeStepCounter]);
                        //     for (i = 0; i < N; i++)
                        //     {
                        //         for (j = 0; j < N; j++)
                        //         {
                        //             fprintf(fpFrames, "%lf ", V[i][j]);
                        //         }
                        //         fprintf(fpFrames, "\n");
                        //     }
                        // }
                        // finishPartial = omp_get_wtime();
                        // elapsedWriting += finishPartial - startPartial;

                        // Check S1 velocity
                        if (S1VelocityTag)
                        {
                            if (V[0][N - 1] >= 80)
                            {
                                S1Velocity = ((10 * (L - stim1xLimit)) / (time[timeStepCounter]));
                                S1VelocityTag = false;
                            }
                        }
                    }
                }

                // Update time step counter
                #pragma omp master
                {
                    timeStepCounter++;
                }
                #pragma omp barrier
            }
        }

        // PDE execution time
        elapsedPDE = elapsed1stThomas + elapsed2ndThomas;
        
        // Finish measuring total execution time
        finishTotal = omp_get_wtime();
        elapsedTotal = finishTotal - startTotal;
    }

    else if (strcmp(method, "theta-ADI") == 0)
    {
        // Start measuring total execution time
        startTotal = omp_get_wtime();

        #pragma omp parallel num_threads(numberThreads) default(none) private(i, j, Istim, actualV, actualW, diffusion, actualRHS_V, actualRHS_W, tildeRHS_V, tildeRHS_W) \
        shared(V, W, N, M, L, T, D, phi, deltat, time, timeStep, timeStepCounter, PdeOdeRatio, \
        fibrosisFactor, stimStrength, stim1Duration, stim2Duration, stim1Begin, stim2Begin, stim1xLimit, stim1yLimit, \
        discS1xLimit, discS1yLimit, discS2xMax, discS2yMax, discS2xMin, discS2yMin, discFibxMax, discFibxMin, discFibyMax, discFibyMin, \
        c_, d_, Vtilde, Wtilde, S1Velocity, S1VelocityTag, VWTag, saverate, fpFrames, saveDataToGif, \
        Rv, rightside, solution, startPartial, finishPartial, elapsedODE, elapsedPDE, \
        elapsedWriting, elapsed1stThomas, elapsed2ndThomas, theta)
        {
            while (timeStepCounter < M)
            {
                // Get time step
                timeStep = time[timeStepCounter];

                // Start measuring ODE execution time
                #pragma omp master
                {
                    startPartial = omp_get_wtime();
                }

                // Resolve ODEs
                #pragma omp for collapse(2)
                for (i = 0; i < N; i++)
                {
                    for (j = 0; j < N; j++)
                    {
                        // Stimulus
                        Istim = stimulus(i, j, timeStep, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax);

                        // Get actual V and W
                        actualV = V[i][j];
                        actualW = W[i][j];

                        diffusion = iDiffusion2nd(i, j, N, V, discFibxMax, discFibxMin, discFibyMax, discFibyMin) + jDiffusion2nd(i, j, N, V, discFibxMax, discFibxMin, discFibyMax, discFibyMin);

                        actualRHS_V = reactionV(actualV, actualW);
                        actualRHS_W = reactionW(actualV, actualW);

                        Vtilde[i][j] = actualV + (deltat * (actualRHS_V + Istim)) + (phi * diffusion);
                        Wtilde[i][j] = actualW + deltat * actualRHS_W;

                        tildeRHS_V = reactionV(Vtilde[i][j], Wtilde[i][j]);
                        tildeRHS_W = reactionW(Vtilde[i][j], Wtilde[i][j]);

                        Rv[i][j] = deltat * (((1.0 - theta) * actualRHS_V) + (theta * tildeRHS_V) + Istim);
                        W[i][j] = actualW + deltat * tildeRHS_W;
                    }
                }
                

                // Finish measuring ODE execution time
                #pragma omp master
                {
                    finishPartial = omp_get_wtime();
                    elapsedODE += finishPartial - startPartial;
                    startPartial = omp_get_wtime();
                }

                // Resolve PDEs (Diffusion)
                // 1st: Implicit y-axis diffusion (lines): right side with explicit x-axis diffusion (columns)
                // Update right side of Thomas algorithm
                #pragma omp barrier
                #pragma omp for collapse(2)
                for (i = 0; i < N; i++)
                {
                    for (j = 0; j < N; j++)
                    {
                        rightside[j][i] = (V[i][j] + ((1-theta) * phi * jDiffusion2nd(i, j, N, V, discFibxMax, discFibxMin, discFibyMax, discFibyMin))) + (0.5 * Rv[i][j]);
                    }
                }

                // Resolve PDEs (Diffusion)
                // 1st: Implicit y-axis diffusion (lines)
                #pragma omp barrier
                #pragma omp for
                for (i = 0; i < N; i++)
                {
                    // Check if i is in fibrotic region
                    if (i >= discFibyMin && i <= discFibyMax)
                    {
                        ThomasAlgorithm2ndCPU(rightside[i], solution[i], N, (theta * phi), c_[i], d_[i], discFibxMin, discFibxMax, fibrosisFactor);
                    }
                    else
                    {
                        ThomasAlgorithm2ndCPU(rightside[i], solution[i], N, (theta * phi), c_[i], d_[i], N, 0, fibrosisFactor);
                    }

                    // Update V
                    for (j = 0; j < N; j++)
                    {
                        V[j][i] = solution[i][j];
                    }
                }

                // Finish measuring 1st Thomas algorithm execution time and start measuring 2nd Thomas algorithm execution time
                #pragma omp master
                {
                    finishPartial = omp_get_wtime();
                    elapsed1stThomas += finishPartial - startPartial;
                    startPartial = omp_get_wtime();
                }

                // 2nd: Implicit x-axis diffusion (columns): right side with explicit y-axis diffusion (lines)
                // Update right side of Thomas algorithm
                #pragma omp barrier
                #pragma omp for collapse(2)
                for (i = 0; i < N; i++)
                {
                    for (j = 0; j < N; j++)
                    {
                        rightside[i][j] = (V[i][j] + ((1-theta) * phi * iDiffusion2nd(i, j, N, V, discFibxMax, discFibxMin, discFibyMax, discFibyMin))) + (0.5 * Rv[i][j]);
                    }
                }

                // 2nd: Implicit x-axis diffusion (columns)
                #pragma omp barrier
                #pragma omp for
                for (i = 0; i < N; i++)
                {
                    // Check if i is in fibrotic region
                    if (i >= discFibxMin && i <= discFibxMax)
                    {
                        ThomasAlgorithm2ndCPU(rightside[i], V[i], N, (theta * phi), c_[i], d_[i], discFibyMin, discFibyMax, fibrosisFactor);
                    }
                    else
                    {
                        ThomasAlgorithm2ndCPU(rightside[i], V[i], N, (theta * phi), c_[i], d_[i], N, 0, fibrosisFactor);
                    }
                }
                
                // Finish measuring PDE execution time
                #pragma omp master
                {                  
                    finishPartial = omp_get_wtime();
                    elapsed2ndThomas += finishPartial - startPartial;
                }

                // Save frames
                #pragma omp master
                {
                    if (VWTag == false)
                    {
                        // Write frames to file
                        // startPartial = omp_get_wtime();
                        // if (timeStepCounter % saverate == 0 && saveDataToGif == true)
                        // {
                        //     fprintf(fpFrames, "%lf\n", time[timeStepCounter]);
                        //     for (i = 0; i < N; i++)
                        //     {
                        //         for (j = 0; j < N; j++)
                        //         {
                        //             fprintf(fpFrames, "%lf ", V[i][j]);
                        //         }
                        //         fprintf(fpFrames, "\n");
                        //     }
                        // }
                        // finishPartial = omp_get_wtime();
                        // elapsedWriting += finishPartial - startPartial;

                        // Check S1 velocity
                        if (S1VelocityTag)
                        {
                            if (V[0][N - 1] >= 80)
                            {
                                S1Velocity = ((10 * (L - stim1xLimit)) / (time[timeStepCounter]));
                                S1VelocityTag = false;
                            }
                        }
                    }
                }

                // Update time step counter
                #pragma omp master
                {
                    timeStepCounter++;
                }
                #pragma omp barrier
            }
        }

        // PDE execution time
        elapsedPDE = elapsed1stThomas + elapsed2ndThomas;

        // Finish measuring total execution time
        finishTotal = omp_get_wtime();
        elapsedTotal = finishTotal - startTotal;
    }

    // Write infos to file
    fprintf(fpInfos, "S1 velocity: %lf m/s\n", S1Velocity);
    fprintf(fpInfos, "S2 begin: %.1lf ms\n", stim2Begin);
    fprintf(fpInfos, "PDE/ODE ratio: %d\n", PdeOdeRatio);
    fprintf(fpInfos, "ODE execution time: %lf seconds\n", elapsedODE);
    fprintf(fpInfos, "PDE execution time: %lf seconds\n", elapsedPDE);
    fprintf(fpInfos, "Writing time: %lf seconds\n", elapsedWriting);
    fprintf(fpInfos, "Total execution time with writings: %lf seconds\n", elapsedTotal);

    fprintf(fpInfos, "\n1st Thomas algorithm (+ transpose) execution time: %lf seconds\n", elapsed1stThomas);
    fprintf(fpInfos, "2nd Thomas algorithm execution time: %lf seconds\n", elapsed2ndThomas);
    
    fprintf(fpInfos, "\ntheta = %lf\n", theta);
    if (haveFibrosis)
    {
        fprintf(fpInfos, "Fibrosis factor: %.2lf\n", fibrosisFactor);
        fprintf(fpInfos, "Fibrosis region: (%.2lf, %.2lf) to (%.2lf, %.2lf)\n", fibrosisMinX, fibrosisMinY, fibrosisMaxX, fibrosisMaxY);
    }

    /*if (saveDataToError == true)
    {
        
        char lastFrameFileName[MAX_STRING_SIZE];
        sprintf(lastFrameFileName, "last-%d-%.3lf-%.3lf.txt", numberThreads, deltat, deltat);
        FILE *fpLast;
        sprintf(aux, "%s/%s", pathToSaveData, lastFrameFileName);
        fpLast = fopen(aux, "w");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                fprintf(fpLast, "%lf ", V[i][j]);
            }
            fprintf(fpLast, "\n");
        }
        fclose(fpLast);
        
    }*/
    
    // Close files
    fclose(fpFrames);
    fclose(fpInfos);

    // Free memory
    free(time);

    // Free memory from host (2D arrays)
    for (int i = 0; i < N; i++)
    {
        free(V[i]);
        free(W[i]);
        free(Vtilde[i]);
        free(Wtilde[i]);
        free(Rv[i]);
        free(rightside[i]);
        free(solution[i]);
        free(c_[i]);
        free(d_[i]);
    }
    free(V);
    free(W);
    free(Vtilde);
    free(Wtilde);
    free(Rv);
    free(rightside);
    free(solution);
    free(c_);
    free(d_);

}

void runODEinCPUandPDEinGPU(bool options[], char *method, real deltat, int numberThreads, real delta_x, char *mode)
{
    // Get options
    bool haveFibrosis = options[0];
    bool measureTotalTime = options[1];
    bool saveDataToError = options[2];
    bool saveDataToGif = options[3];
    bool measureTimeParts = options[4];
    bool measureS1Velocity = options[5];

    deltax = delta_x;
    deltay = delta_x;

    // Number of steps
    int N = round(L / deltax) + 1;                  // Spatial steps (square tissue)
    int M = round(T / deltat) + 1;               // Number of time steps
    int PdeOdeRatio = round(deltat / deltat); // Ratio between PDE and ODE time steps

    // Allocate and populate time array
    real *time;
    time = (real *)malloc(M * sizeof(real));
    for (int i = 0; i < M; i++)
    {
        time[i] = i * deltat;
    }

    // Allocate and initialize variables
    real *V, *W;
    V = (real *)malloc(N * N * sizeof(real));
    W = (real *)malloc(N * N * sizeof(real));
    initializeVariablesGPU(N, V, W);

    // Diffusion coefficient - isotropic
    real D = sigma / (chi * Cm);
    real phi = D * deltat / (deltax * deltax); // For Thomas algorithm - isotropic

    // Variables
    int i, j; // i for y-axis and j for x-axis
    real actualV, actualW;
    real Rw;
    real *Vtilde, *Wtilde, *Rv, *rightside, *solution;
    Vtilde = (real *)malloc(N * N * sizeof(real));
    Wtilde = (real *)malloc(N * N * sizeof(real));
    Rv = (real *)malloc(N * N * sizeof(real));
    rightside = (real *)malloc(N * N * sizeof(real));
    solution = (real *)malloc(N * N * sizeof(real));

    // Auxiliary arrays for Thomas algorithm 2nd order approximation
    real *la = (real *)malloc(N * sizeof(real));
    real *lb = (real *)malloc(N * sizeof(real));
    real *lc = (real *)malloc(N * sizeof(real));
    populateDiagonalThomasAlgorithm(la, lb, lc, N, phi);

    // Discretized limits of stimulation area
    int discS1xLimit = round(stim1xLimit / deltax);
    int discS1yLimit = round(stim1yLimit / deltay);
    int discS2xMax = round(stim2xMax / deltax);
    int discS2xMin = round(stim2xMin / deltax);
    int discS2yMax = N;
    int discS2yMin = N - round(stim2yMax / deltay);

    // Discritized limits of fibrotic area
    int discFibxMax = round(fibrosisMaxX / deltax);
    int discFibxMin = round(fibrosisMinX / deltax);
    int discFibyMax = N - round(fibrosisMinY / deltay);
    int discFibyMin = N - round(fibrosisMaxY / deltay);

    // File names
    char framesFileName[MAX_STRING_SIZE], infosFileName[MAX_STRING_SIZE];
    sprintf(framesFileName, "frames-%d-%.3lf-%.3lf.txt", numberThreads, deltat, deltat);
    sprintf(infosFileName, "infos-%d-%.3lf-%.3lf.txt", numberThreads, deltat, deltat);
    int saverate = ceil(M / 100.0);
    FILE *fpFrames, *fpInfos;

    // Create directories and files
    char pathToSaveData[MAX_STRING_SIZE];
    if (haveFibrosis)
    {
        createDirectories(pathToSaveData, method, "AFHN-Fibro", mode);
    }
    else
    {
        createDirectories(pathToSaveData, method, "AFHN", mode);
    }

    // File pointers
    char aux[MAX_STRING_SIZE];
    if (VWTag == false)
    {

        sprintf(aux, "%s/%s", pathToSaveData, infosFileName);
        fpInfos = fopen(aux, "w");
    }
    else
    {
        sprintf(aux, "%s/%s", pathToSaveData, infosFileName);
        fpInfos = fopen(aux, "a");
    }
    if (saveDataToGif == false)
    {
        sprintf(aux, "%s/%s", pathToSaveData, framesFileName);
        fpFrames = fopen(aux, "a");
    }
    else
    {
        sprintf(aux, "%s/%s", pathToSaveData, framesFileName);
        fpFrames = fopen(aux, "w");
    }

    // CUDA variables and allocation
    int numBlocks = N / 100; 
    int blockSize = round(N / numBlocks) + 1;
    if (blockSize % 32 != 0)
    {
        blockSize = 32 * ((blockSize / 32) + 1);
    }
    printf("NumBlock: %d, BlockSize: %d\n", numBlocks, blockSize);
    
    // Diagonal kernel parameters
    dim3 block (BDIMX, BDIMY);
    dim3 grid  ((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    real *d_V, *d_Vtilde, *d_rightside, *d_solution;
    real *d_la, *d_lb, *d_lc;
    cudaError_t cudaStatus1, cudaStatus2, cudaStatus3, cudaStatus4, cudaStatus5, cudaStatus6, cudaStatus7;

    cudaStatus1 = cudaMalloc(&d_V, N * N * sizeof(real));
    cudaStatus2 = cudaMalloc(&d_Vtilde, N * N * sizeof(real));
    cudaStatus3 = cudaMalloc(&d_rightside, N * N * sizeof(real));
    cudaStatus4 = cudaMalloc(&d_solution, N * N * sizeof(real));
    cudaStatus5 = cudaMalloc(&d_la, N * sizeof(real));
    cudaStatus6 = cudaMalloc(&d_lb, N * sizeof(real));
    cudaStatus7 = cudaMalloc(&d_lc, N * sizeof(real));
    if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess || cudaStatus3 != cudaSuccess || cudaStatus4 != cudaSuccess || cudaStatus5 != cudaSuccess || cudaStatus6 != cudaSuccess || cudaStatus7 != cudaSuccess)
    {
        printf("cudaMalloc failed 1st call!\n");
        exit(EXIT_FAILURE);
    }
    printf("All cudaMallocs done!\n");

    // Prefactorization
    prefactorizationThomasAlgorithm(la, lb, lc, N);

    // Copy memory of diagonals from host to device
    cudaStatus1 = cudaMemcpy(d_la, la, N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus2 = cudaMemcpy(d_lb, lb, N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus3 = cudaMemcpy(d_lc, lc, N * sizeof(real), cudaMemcpyHostToDevice);
    if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess || cudaStatus3 != cudaSuccess)
    {
        printf("cudaMemcpy failed 1st call!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize cuSPARSE
    
	// cusparseHandle_t handle;
    // cusparseCreate(&handle);
    // size_t buffer_size_in_bytes = 0;
    // if (cusparseDgtsv2_nopivot_bufferSizeExt(handle, N, N, d_la, d_lb, d_lc, d_rightside, N, &buffer_size_in_bytes))
    // {
    //     printf("Determining temporary buffer size failed\n");
    // }
    // //buffer_size_in_bytes = buffer_size_in_bytes + 128;
    // //printf("cusparseDgtsv2_bufferSizeExt = %d\n", buffer_size_in_bytes);
    // void * p_buffer;
    // if (cudaMalloc(&p_buffer, buffer_size_in_bytes))
    // {
    //     printf("Allocating temprary buffer failed");
    // }
    


    /*--------------------
    --  ADI 1st order   --
    ----------------------*/
    int index;
    if (strcmp(method, "OS-ADI") == 0)
    {
        // Start measuring total execution time
        startTotal = omp_get_wtime();
        
        while (timeStepCounter < M)
        {
            // Get time step
            timeStep = time[timeStepCounter];

            #pragma omp parallel num_threads(numberThreads) default(none) private(i, j, Istim, actualV, actualW, index) \
            shared(V, W, N, T, D, phi, deltat, time, timeStep, \
            discS1xLimit, discS1yLimit, discS2xMax, discS2yMax, discS2xMin, discS2yMin, discFibxMax, discFibxMin, discFibyMax, discFibyMin, \
            rightside, startPartial, finishPartial, elapsedODE, elapsedPDE)
            {
                // Start measuring ODE execution time
                #pragma omp master
                {
                    startPartial = omp_get_wtime();
                }

                // Resolve ODEs
                #pragma omp for collapse(2)
                for (i = 0; i < N; i++)
                {
                    for (j = 0; j < N; j++)
                    {
                        // Stimulus
                        Istim = stimulus(i, j, timeStep, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax);

                        index = i * N + j;

                        // Get actual V and W
                        actualV = V[index];
                        actualW = W[index];

                        // Update V and W without diffusion
                        V[index] = actualV + deltat * (reactionV(actualV, actualW) + Istim);
                        W[index] = actualW + deltat * reactionW(actualV, actualW);

                        // Update right side of Thomas algorithm
                        rightside[j*N+i] = V[index];
                    }
                }

                // Finish measuring ODE execution time
                #pragma omp master
                {
                    finishPartial = omp_get_wtime();
                    elapsedODE += finishPartial - startPartial;
                }
                #pragma omp barrier
            }
            
            // Resolve PDEs (Diffusion)
            // 1st: Implicit y-axis diffusion (lines)                
            // Copy memory from host to device of the matrices (2D arrays)
            startPartial = omp_get_wtime();
            cudaStatus1 = cudaMemcpy(d_rightside, rightside, N * N * sizeof(real), cudaMemcpyHostToDevice);
            if (cudaStatus1 != cudaSuccess)
            {
                printf("cudaMemcpy failed 2nd call!\n");
                exit(EXIT_FAILURE);
            }
            finishPartial = omp_get_wtime();
            elapsed1stMemCopy += finishPartial - startPartial;
            elapsedMemCopy += elapsed1stMemCopy;
                        
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas<<<numBlocks, blockSize>>>(d_rightside, N, d_la, d_lb, d_lc);
            // if (cusparseDgtsv2_nopivot(handle, N, N, d_la, d_lb, d_lc, d_rightside, N, p_buffer))
            // {
            //     printf("Solving TDS1 failed!\n");
            // }
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed1stThomas += finishPartial - startPartial;

            // Call the transpose kernel
            startPartial = omp_get_wtime();
            transposeDiagonalCol<<<grid, block>>>(d_rightside, d_solution, N, N);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsedTranspose += finishPartial - startPartial;

            // 2nd: Implicit x-axis diffusion (columns)                
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas<<<numBlocks, blockSize>>>(d_solution, N, d_la, d_lb, d_lc);
            // if (cusparseDgtsv2_nopivot(handle, N, N, d_la, d_lb, d_lc, d_solution, N, p_buffer))
            // {
            //     printf("Solving TDS2 failed!\n");
            // }
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed2ndThomas += finishPartial - startPartial;

            // Copy memory from device to host of the matrices (2D arrays)
            startPartial = omp_get_wtime();
            cudaStatus1 = cudaMemcpy(V, d_solution, N * N * sizeof(real), cudaMemcpyDeviceToHost);
            if (cudaStatus1 != cudaSuccess)
            {
                printf("cudaMemcpy failed 5th call!\n");
                exit(EXIT_FAILURE);
            }
            finishPartial = omp_get_wtime();
            elapsed4thMemCopy += finishPartial - startPartial;
            elapsedMemCopy += elapsed4thMemCopy;
            
            // Finish measuring PDE execution time
            finishPartial = omp_get_wtime();
            elapsedPDE += elapsed1stThomas + elapsed2ndThomas + elapsedTranspose + elapsedMemCopy; 
            
            // Save frames
            if (VWTag == false)
            {
                // Write frames to file
                startPartial = omp_get_wtime();
                if (timeStepCounter % saverate == 0 && saveDataToGif == true)
                {
                    fprintf(fpFrames, "%lf\n", time[timeStepCounter]);
                    for (i = 0; i < N; i++)
                    {
                        for (j = 0; j < N; j++)
                        {
                            index = i * N + j;
                            fprintf(fpFrames, "%lf ", V[index]);
                        }
                        fprintf(fpFrames, "\n");
                    }
                }
                finishPartial = omp_get_wtime();
                elapsedWriting += finishPartial - startPartial;
               
                // Check S1 velocity
                if (S1VelocityTag)
                {
                    if (V[N - 1] >= 80)
                    {
                        S1Velocity = ((10 * (L - stim1xLimit)) / (time[timeStepCounter]));
                        S1VelocityTag = false;
                    }
                }
            }
    

            // Update time step counter
            timeStepCounter++;
        }
        
        // Finish measuring total execution time
        finishTotal = omp_get_wtime();
        elapsedTotal = finishTotal - startTotal;
    }

    // Write infos to file
    fprintf(fpInfos, "S1 velocity: %lf m/s\n", S1Velocity);
    fprintf(fpInfos, "S2 begin: %.1lf ms\n", stim2Begin);
    fprintf(fpInfos, "PDE/ODE ratio: %d\n", PdeOdeRatio);
    fprintf(fpInfos, "ODE execution time: %lf seconds\n", elapsedODE);
    fprintf(fpInfos, "PDE execution time: %lf seconds\n", elapsedPDE);
    fprintf(fpInfos, "Writing time: %lf seconds\n", elapsedWriting);
    fprintf(fpInfos, "Total execution time with writings: %lf seconds\n", elapsedTotal);
    
    fprintf(fpInfos, "\nFor PDE -> Grid size: %d, Block size: %d\n", numBlocks, blockSize);
    fprintf(fpInfos, "Total threads: %d\n", numBlocks*blockSize);
    fprintf(fpInfos, "1st Thomas algorithm time: %lf seconds\n", elapsed1stThomas);
    fprintf(fpInfos, "2nd Thomas algorithm time: %lf seconds\n", elapsed2ndThomas);
    fprintf(fpInfos, "Transpose time: %lf seconds\n", elapsedTranspose);
    fprintf(fpInfos, "1st memory copy time: %lf seconds\n", elapsed1stMemCopy);
    fprintf(fpInfos, "Last memory copy time: %lf seconds\n", elapsed4thMemCopy);
    fprintf(fpInfos, "Total memory copy time: %lf seconds\n", elapsedMemCopy);
    
    // fprintf(fpInfos, "\ncusparseDgtsv2_bufferSizeExt = %d\n", buffer_size_in_bytes);

    if (haveFibrosis)
    {
        fprintf(fpInfos, "Fibrosis factor: %.2lf\n", fibrosisFactor);
        fprintf(fpInfos, "Fibrosis region: (%.2lf, %.2lf) to (%.2lf, %.2lf)\n", fibrosisMinX, fibrosisMinY, fibrosisMaxX, fibrosisMaxY);
    }

    if (saveDataToError == true)
    {
        /*
        char lastFrameFileName[MAX_STRING_SIZE];
        sprintf(lastFrameFileName, "last-%d-%.3lf-%.3lf.txt", numberThreads, deltat, deltat);
        FILE *fpLast;
        sprintf(aux, "%s/%s", pathToSaveData, lastFrameFileName);
        fpLast = fopen(aux, "w");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                index = i * N + j;
                fprintf(fpLast, "%lf ", V[index]);
            }
            fprintf(fpLast, "\n");
        }
        fclose(fpLast);
        */
    }

    // Close files
    fclose(fpFrames);
    fclose(fpInfos);

    // Free memory
    free(time);

    // Free memory from host
    free(V);
    free(W);
    free(Vtilde);
    free(Wtilde);
    free(Rv);
    free(rightside);
    free(solution);
    free(la);
    free(lb);
    free(lc);

    // Free memory from device
    cudaFree(d_V);
    cudaFree(d_Vtilde);
    cudaFree(d_rightside);
    cudaFree(d_solution);
    cudaFree(d_la);
    cudaFree(d_lb);
    cudaFree(d_lc);
    // cudaFree(p_buffer);

}

void runAllinGPU(bool options[], char *method, real deltat, int numberThreads, real delta_x, char *mode, real theta, int number_of_exec)
{
    // Get options
    bool haveFibrosis = options[0];
    bool measureTotalTime = options[1];
    bool saveDataToError = options[2];
    bool saveDataToGif = options[3];
    bool measureTimeParts = options[4];
    bool measureS1Velocity = options[5];

    deltax = delta_x;
    deltay = delta_x;

    // Number of steps
    int N = round(L / deltax) + 1;                  // Spatial steps (square tissue)
    int M = round(T / deltat) + 1;                  // Number of time steps
    int PdeOdeRatio = round(deltat / deltat);       // Ratio between PDE and ODE time steps

    // Allocate and populate time array
    real *time;
    time = (real *)malloc(M * sizeof(real));
    for (int i = 0; i < M; i++)
    {
        time[i] = i * deltat;
    }

    // Allocate and initialize variables
    real *V, *W;
    V = (real *)malloc(N * N * sizeof(real));
    W = (real *)malloc(N * N * sizeof(real));
    initializeVariablesGPU(N, V, W);

    // Diffusion coefficient - isotropic
    real D = sigma / (chi * Cm);
    real phi = D * deltat / (deltax * deltax);      // For Thomas algorithm - isotropic

    // Variables
    int i, j;                                       // i for y-axis and j for x-axis
    real actualV, actualW;
    real Rw;
    real *Vtilde, *Wtilde, *Rv, *rightside, *solution;
    Vtilde = (real *)malloc(N * N * sizeof(real));
    Wtilde = (real *)malloc(N * N * sizeof(real));
    Rv = (real *)malloc(N * N * sizeof(real));
    rightside = (real *)malloc(N * N * sizeof(real));
    solution = (real *)malloc(N * N * sizeof(real));

    // Auxiliary arrays for Thomas algorithm 2nd order approximation
    real *la = (real *)malloc(N * sizeof(real));
    real *lb = (real *)malloc(N * sizeof(real));
    real *lc = (real *)malloc(N * sizeof(real));

    // For the Theta method, it will be necessary new auxiliary arrays to be multiplied by 1-theta
    real *la2, *lb2, *lc2;

    if (strcmp(method, "OS-ADI") == 0)
        populateDiagonalThomasAlgorithm(la, lb, lc, N, phi);
    else if (strcmp(method, "SSI-ADI") == 0 || strcmp(method, "MOSI-ADI") == 0)
        populateDiagonalThomasAlgorithm(la, lb, lc, N, 0.5*phi);
    else if (strcmp(method, "theta-ADI") == 0)
    {
        populateDiagonalThomasAlgorithm(la, lb, lc, N, theta*phi);

        // Mallloc new arrays
        la2 = (real *)malloc(N * sizeof(real));
        lb2 = (real *)malloc(N * sizeof(real));
        lc2 = (real *)malloc(N * sizeof(real));
        populateDiagonalThomasAlgorithm(la2, lb2, lc2, N, (1-theta)*phi);
    }
        
    // Prefactorization
    prefactorizationThomasAlgorithm(la, lb, lc, N);
    if (strcmp(method, "theta-ADI") == 0)
        prefactorizationThomasAlgorithm(la2, lb2, lc2, N);

    // Discretized limits of stimulation area
    int discS1xLimit = round(stim1xLimit / deltax);
    int discS1yLimit = round(stim1yLimit / deltay);
    int discS2xMax = round(stim2xMax / deltax);
    int discS2xMin = round(stim2xMin / deltax);
    int discS2yMax = N;
    int discS2yMin = N - round(stim2yMax / deltay);
    
    // Discritized limits of fibrotic area
    int discFibxMax = round(fibrosisMaxX / deltax);
    int discFibxMin = round(fibrosisMinX / deltax);
    int discFibyMax = N - round(fibrosisMinY / deltay);
    int discFibyMin = N - round(fibrosisMaxY / deltay);

    // File names
    char framesFileName[MAX_STRING_SIZE], infosFileName[MAX_STRING_SIZE];
    sprintf(framesFileName, "frames-%d-%.3lf.txt", numberThreads, deltat);
    sprintf(infosFileName, "infos-%d-%.3lf-%.2lf.txt", numberThreads, deltat, theta);
    int saverate = ceil(M / 100.0);
    FILE *fpFrames, *fpInfos;

    // Create directories and files
    char pathToSaveData[MAX_STRING_SIZE];
    if (haveFibrosis)
    {
        // createDirectories(pathToSaveData, method, "AFHN-Fibro", mode);
        createDirectoriesScript(pathToSaveData, method, "AFHN-Fibro", mode, numberThreads, number_of_exec);
    }
    else
    {
        // createDirectories(pathToSaveData, method, "AFHN", mode);
        createDirectoriesScript(pathToSaveData, method, "AFHN", mode, numberThreads, number_of_exec);
    }
    
    // File pointers
    char aux[MAX_STRING_SIZE];
    sprintf(aux, "%s/%s", pathToSaveData, infosFileName);
    if (VWTag == false)
        fpInfos = fopen(aux, "w");
    else
        fpInfos = fopen(aux, "a");
    
    sprintf(aux, "%s/%s", pathToSaveData, framesFileName);
    if (saveDataToGif == false)
        fpFrames = fopen(aux, "a");
    else
        fpFrames = fopen(aux, "w");
    
    // CUDA variables and allocation
    int numBlocks = N / 100; 
    int blockSize = round(N / numBlocks) + 1;
    if (blockSize % 32 != 0)
        blockSize = 32 * ((blockSize / 32) + 1);
    printf("NumBlock: %d, BlockSize: %d\n", numBlocks, blockSize);
    
    // Diagonal kernel parameters
    dim3 block (BDIMX, BDIMY);
    dim3 grid  ((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    real *d_V, *d_W, *d_rightside, *d_solution, *d_Rv;
    real *d_la, *d_lb, *d_lc;
    real *d_la2, *d_lb2, *d_lc2;
    cudaError_t cudaStatus1, cudaStatus2, cudaStatus3, cudaStatus4, cudaStatus5, cudaStatus6, cudaStatus7, cudaStatus8;
    
    cudaStatus1 = cudaMalloc(&d_V, N * N * sizeof(real));
    cudaStatus2 = cudaMalloc(&d_W, N * N * sizeof(real));
    cudaStatus3 = cudaMalloc(&d_rightside, N * N * sizeof(real));
    cudaStatus4 = cudaMalloc(&d_solution, N * N * sizeof(real));
    cudaStatus5 = cudaMalloc(&d_Rv, N * N * sizeof(real));
    cudaStatus6 = cudaMalloc(&d_la, N * sizeof(real));
    cudaStatus7 = cudaMalloc(&d_lb, N * sizeof(real));
    cudaStatus8 = cudaMalloc(&d_lc, N * sizeof(real));
    if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess || cudaStatus3 != cudaSuccess || cudaStatus4 != cudaSuccess || cudaStatus5 != cudaSuccess || cudaStatus6 != cudaSuccess || cudaStatus7 != cudaSuccess || cudaStatus8 != cudaSuccess)
    {
        printf("cudaMalloc failed 1st call!\n");
        exit(EXIT_FAILURE);
    }
    printf("All cudaMallocs done!\n");
    
    if (strcmp(method, "theta-ADI") == 0)
    {
        cudaStatus1 = cudaMalloc(&d_la2, N * sizeof(real));
        cudaStatus2 = cudaMalloc(&d_lb2, N * sizeof(real));
        cudaStatus3 = cudaMalloc(&d_lc2, N * sizeof(real));
        if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess || cudaStatus3 != cudaSuccess)
        {
            printf("cudaMalloc failed for theta-Method second aux arrays for Thomas!\n");
            exit(EXIT_FAILURE);
        }
    }

    // Copy memory from host to device of the matrices (2D arrays)
    cudaStatus1 = cudaMemcpy(d_V, V, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus2 = cudaMemcpy(d_W, W, N * N * sizeof(real), cudaMemcpyHostToDevice);
    if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess)
    {
        printf("cudaMemcpy failed 1st call!\n");
        exit(EXIT_FAILURE);
    }

    // Copy memory of diagonals from host to device
    cudaStatus1 = cudaMemcpy(d_la, la, N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus2 = cudaMemcpy(d_lb, lb, N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus3 = cudaMemcpy(d_lc, lc, N * sizeof(real), cudaMemcpyHostToDevice);
    if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess || cudaStatus3 != cudaSuccess)
    {
        printf("cudaMemcpy failed 2nd call!\n");
        exit(EXIT_FAILURE);
    }

    if (strcmp(method, "theta-ADI") == 0)
    {
        cudaStatus1 = cudaMemcpy(d_la2, la2, N * sizeof(real), cudaMemcpyHostToDevice);
        cudaStatus2 = cudaMemcpy(d_lb2, lb2, N * sizeof(real), cudaMemcpyHostToDevice);
        cudaStatus3 = cudaMemcpy(d_lc2, lc2, N * sizeof(real), cudaMemcpyHostToDevice);
        if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess || cudaStatus3 != cudaSuccess)
        {
            printf("cudaMemcpy failed for theta-Method second aux arrays for Thomas!\n");
            exit(EXIT_FAILURE);
        }
    }

    int GRID_SIZE = ceil((N*N*1.0) / (BLOCK_SIZE*1.0));

    /*--------------------
    --  ADI 1st order   --
    ----------------------*/
    int index;
    if (strcmp(method, "OS-ADI") == 0)
    {
        // Start measuring total execution time
        startTotal = omp_get_wtime();
        
        while (timeStepCounter < M)
        {
            // Get time step
            timeStep = time[timeStepCounter];

            // Start measuring ODE execution time
            startPartial = omp_get_wtime();

            // Resolve ODEs
            parallelODE<<<GRID_SIZE, BLOCK_SIZE>>>(d_V, d_W, d_rightside, N, timeStep, deltat, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax);
            cudaDeviceSynchronize();

            // Finish measuring ODE execution time
            finishPartial = omp_get_wtime();
            elapsedODE += finishPartial - startPartial;
            
            // Resolve PDEs (Diffusion)
            // 1st: Implicit y-axis diffusion (lines)
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas<<<numBlocks, blockSize>>>(d_rightside, N, d_la, d_lb, d_lc);
            //cuThomasVBatch<<<numBlocks, blockSize>>>(d_la, d_lb, d_lc, d_rightside, N, N);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed1stThomas += finishPartial - startPartial;

            // Call the transpose kernel
            startPartial = omp_get_wtime();
            transposeDiagonalCol<<<GRID_SIZE, BLOCK_SIZE>>>(d_rightside, d_V, N, N);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsedTranspose += finishPartial - startPartial;

            // 2nd: Implicit x-axis diffusion (columns)                
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas<<<numBlocks, blockSize>>>(d_V, N, d_la, d_lb, d_lc);
            //cuThomasVBatch<<<numBlocks, blockSize>>>(d_la, d_lb, d_lc, d_V, N, N);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed2ndThomas += finishPartial - startPartial;

            
            // Save frames
            if (VWTag == false)
            {
                // Write frames to file
                startPartial = omp_get_wtime();
                if (timeStepCounter % saverate == 0 && saveDataToGif == true)
                {
                   // Copy memory from device to host of the matrices (2D arrays)
                   startPartial = omp_get_wtime();
                   cudaStatus1 = cudaMemcpy(V, d_V, N * N * sizeof(real), cudaMemcpyDeviceToHost);
                   if (cudaStatus1 != cudaSuccess)
                  {
                       printf("cudaMemcpy failed 5th call!\n");
                       exit(EXIT_FAILURE);
                   }
                   finishPartial = omp_get_wtime();
                   elapsedMemCopy += finishPartial - startPartial;
                   elapsed4thMemCopy += finishPartial - startPartial;

                   fprintf(fpFrames, "%lf\n", time[timeStepCounter]);
                   for (i = 0; i < N; i++)
                   {
                       for (j = 0; j < N; j++)
                       {
                           index = i * N + j;
                           fprintf(fpFrames, "%lf ", V[index]);
                       }
                       fprintf(fpFrames, "\n");
                   }
                }
                finishPartial = omp_get_wtime();
                elapsedWriting += finishPartial - startPartial;
               

                // Check S1 velocity
                if (S1VelocityTag)
                {
                    // Copy memory from device to host of the matrices (2D arrays)
                    startPartial = omp_get_wtime();
                    cudaStatus1 = cudaMemcpy(V, d_V, N * N * sizeof(real), cudaMemcpyDeviceToHost);
                    if (cudaStatus1 != cudaSuccess)
                    {
                        printf("cudaMemcpy failed 5th call!\n");
                        exit(EXIT_FAILURE);
                    }
                    finishPartial = omp_get_wtime();
                    elapsedMemCopy += finishPartial - startPartial;
                    elapsed4thMemCopy += finishPartial - startPartial;

                    if (V[N - 1] >= 80)
                    {
                        S1Velocity = ((10 * (L - stim1xLimit)) / (time[timeStepCounter]));
                        S1VelocityTag = false;
                    }
                }
            }
            
            // Update time step counter
            timeStepCounter++;
        }

        // PDE execution time
        elapsedPDE = elapsed1stThomas + elapsed2ndThomas + elapsedTranspose;
        
        // Finish measuring total execution time
        finishTotal = omp_get_wtime();
        elapsedTotal = finishTotal - startTotal;
    }

    else if (strcmp(method, "SSI-ADI") == 0)
    {
        // Start measuring total execution time
        startTotal = omp_get_wtime();
        
        while (timeStepCounter < M)
        {
            // Get time step
            timeStep = time[timeStepCounter];

            // Start measuring ODE execution time
            startPartial = omp_get_wtime();

            // Resolve ODEs
            parallelODE_SSI<<<GRID_SIZE, BLOCK_SIZE>>>(d_V, d_W, d_Rv, N, timeStep, deltat, phi, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
            cudaDeviceSynchronize();

            // Finish measuring ODE execution time
            finishPartial = omp_get_wtime();
            elapsedODE += finishPartial - startPartial;

            // Prepare right side of Thomas algorithm with explicit diffusion on j
            // Call the kernel
            startPartial = omp_get_wtime();
            prepareRighthandSide_jDiffusion<<<GRID_SIZE, BLOCK_SIZE>>>(d_V, d_rightside, d_Rv, N, phi, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor); 
            cudaDeviceSynchronize();     
            finishPartial = omp_get_wtime();
            elapsed1stRHS += finishPartial - startPartial;

            // 1st: Implicit y-axis diffusion (lines)
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas<<<numBlocks, blockSize>>>(d_rightside, N, d_la, d_lb, d_lc);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed1stThomas += finishPartial - startPartial;

            // Call the transpose kernel
            startPartial = omp_get_wtime();
            transposeDiagonalCol<<<GRID_SIZE, BLOCK_SIZE>>>(d_rightside, d_V, N, N);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsedTranspose += finishPartial - startPartial;

            // Prepare right side of Thomas algorithm with explicit diffusion on i
            // Call the kernel
            startPartial = omp_get_wtime();
            prepareRighthandSide_iDiffusion<<<GRID_SIZE, BLOCK_SIZE>>>(d_V, d_rightside, d_Rv, N, phi, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed2ndRHS += finishPartial - startPartial;

            // 2nd: Implicit x-axis diffusion (columns)                
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas<<<numBlocks, blockSize>>>(d_rightside, N, d_la, d_lb, d_lc);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed2ndThomas += finishPartial - startPartial;

            // Copy d_rightside to d_V
            startPartial = omp_get_wtime();
            cudaStatus1 = cudaMemcpy(d_V, d_rightside, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
            if (cudaStatus1 != cudaSuccess)
            {
                printf("cudaMemcpy failed device to device!\n");
                exit(EXIT_FAILURE);
            }
            finishPartial = omp_get_wtime();
            elapsedMemCopy += finishPartial - startPartial;
            elapsed2ndMemCopy += finishPartial - startPartial;

            
            // Save frames
            if (VWTag == false)
            {
                // Write frames to file
                startPartial = omp_get_wtime();
                if (timeStepCounter % saverate == 0 && saveDataToGif == true)
                {
                  //Copy memory from device to host of the matrices (2D arrays)
                  startPartial = omp_get_wtime();
                  cudaStatus1 = cudaMemcpy(V, d_V, N * N * sizeof(real), cudaMemcpyDeviceToHost);
                  if (cudaStatus1 != cudaSuccess)
                  {
                       printf("cudaMemcpy failed 5th call!\n");
                       exit(EXIT_FAILURE);
                  }
                  finishPartial = omp_get_wtime();
                  elapsedMemCopy += finishPartial - startPartial;
                  elapsed4thMemCopy += finishPartial - startPartial;

                  fprintf(fpFrames, "%lf\n", time[timeStepCounter]);
                  for (i = 0; i < N; i++)
                  {
                      for (j = 0; j < N; j++)
                      {
                           index = i * N + j;
                           fprintf(fpFrames, "%lf ", V[index]);
                      }
                      fprintf(fpFrames, "\n");
                  }
                }
                finishPartial = omp_get_wtime();
                elapsedWriting += finishPartial - startPartial;


                // Check S1 velocity
                if (S1VelocityTag)
                {
                    // Copy memory from device to host of the matrices (2D arrays)
                    startPartial = omp_get_wtime();
                    cudaStatus1 = cudaMemcpy(V, d_V, N * N * sizeof(real), cudaMemcpyDeviceToHost);
                    if (cudaStatus1 != cudaSuccess)
                    {
                        printf("cudaMemcpy failed 5th call!\n");
                        exit(EXIT_FAILURE);
                    }
                    finishPartial = omp_get_wtime();
                    elapsedMemCopy += finishPartial - startPartial;
                    elapsed4thMemCopy += finishPartial - startPartial;

                    if (V[N - 1] >= 80)
                    {
                        S1Velocity = ((10 * (L - stim1xLimit)) / (time[timeStepCounter]));
                        S1VelocityTag = false;
                    }
                }
            }
            
            // Update time step counter
            timeStepCounter++;
        }

        // PDE execution
        elapsedPDE = elapsed1stThomas + elapsed2ndThomas + elapsedTranspose + elapsed1stRHS + elapsed2ndRHS;
        
        // Finish measuring total execution time
        finishTotal = omp_get_wtime();
        elapsedTotal = finishTotal - startTotal;
    }

    else if (strcmp(method, "MOSI-ADI") == 0)
    {
        // Start measuring total execution time
        startTotal = omp_get_wtime();
        
        while (timeStepCounter < M)
        {
            // Get time step
            timeStep = time[timeStepCounter];

            // Start measuring ODE execution time
            startPartial = omp_get_wtime();

            // Resolve ODEs
            parallelODE_MOSI<<<GRID_SIZE, BLOCK_SIZE>>>(d_V, d_W, d_Rv, N, timeStep, deltat, phi, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
            cudaDeviceSynchronize();

            // Finish measuring ODE execution time
            finishPartial = omp_get_wtime();
            elapsedODE += finishPartial - startPartial;

            // Prepare right side of Thomas algorithm with explicit diffusion on j
            // Call the kernel
            startPartial = omp_get_wtime();
            prepareRighthandSide_jDiffusion<<<GRID_SIZE, BLOCK_SIZE>>>(d_V, d_rightside, d_Rv, N, phi, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor); 
            cudaDeviceSynchronize();     
            finishPartial = omp_get_wtime();
            elapsed1stRHS += finishPartial - startPartial;

            // 1st: Implicit y-axis diffusion (lines)
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas<<<numBlocks, blockSize>>>(d_rightside, N, d_la, d_lb, d_lc);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed1stThomas += finishPartial - startPartial;

            // Call the transpose kernel
            startPartial = omp_get_wtime();
            transposeDiagonalCol<<<GRID_SIZE, BLOCK_SIZE>>>(d_rightside, d_V, N, N);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsedTranspose += finishPartial - startPartial;

            // Prepare right side of Thomas algorithm with explicit diffusion on i
            // Call the kernel
            startPartial = omp_get_wtime();
            prepareRighthandSide_iDiffusion<<<GRID_SIZE, BLOCK_SIZE>>>(d_V, d_rightside, d_Rv, N, phi, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed2ndRHS += finishPartial - startPartial;

            // 2nd: Implicit x-axis diffusion (columns)                
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas<<<numBlocks, blockSize>>>(d_rightside, N, d_la, d_lb, d_lc);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed2ndThomas += finishPartial - startPartial;

            // Copy d_rightside to d_V
            startPartial = omp_get_wtime();
            cudaStatus1 = cudaMemcpy(d_V, d_rightside, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
            if (cudaStatus1 != cudaSuccess)
            {
                printf("cudaMemcpy failed device to device!\n");
                exit(EXIT_FAILURE);
            }
            finishPartial = omp_get_wtime();
            elapsedMemCopy += finishPartial - startPartial;
            elapsed2ndMemCopy += finishPartial - startPartial;

            
            // Save frames
            if (VWTag == false)
            {
                // Write frames to file
                startPartial = omp_get_wtime();
                if (timeStepCounter % saverate == 0 && saveDataToGif == true)
                {
                    //Copy memory from device to host of the matrices (2D arrays)
                    startPartial = omp_get_wtime();
                    cudaStatus1 = cudaMemcpy(V, d_V, N * N * sizeof(real), cudaMemcpyDeviceToHost);
                    if (cudaStatus1 != cudaSuccess)
                    {
                        printf("cudaMemcpy failed 5th call!\n");
                        exit(EXIT_FAILURE);
                    }
                    finishPartial = omp_get_wtime();
                    elapsedMemCopy += finishPartial - startPartial;
                    elapsed4thMemCopy += finishPartial - startPartial;

                    fprintf(fpFrames, "%lf\n", time[timeStepCounter]);
                    for (i = 0; i < N; i++)
                    {
                        for (j = 0; j < N; j++)
                        {
                            index = i * N + j;
                            fprintf(fpFrames, "%lf ", V[index]);
                        }
                        fprintf(fpFrames, "\n");
                    }
                }
                finishPartial = omp_get_wtime();
                elapsedWriting += finishPartial - startPartial;


                // Check S1 velocity
                if (S1VelocityTag)
                {
                    // Copy memory from device to host of the matrices (2D arrays)
                    startPartial = omp_get_wtime();
                    cudaStatus1 = cudaMemcpy(V, d_V, N * N * sizeof(real), cudaMemcpyDeviceToHost);
                    if (cudaStatus1 != cudaSuccess)
                    {
                        printf("cudaMemcpy failed 5th call!\n");
                        exit(EXIT_FAILURE);
                    }
                    finishPartial = omp_get_wtime();
                    elapsedMemCopy += finishPartial - startPartial;
                    elapsed4thMemCopy += finishPartial - startPartial;

                    if (V[N - 1] >= 80)
                    {
                        S1Velocity = ((10 * (L - stim1xLimit)) / (time[timeStepCounter]));
                        S1VelocityTag = false;
                    }
                }
            }
            
            // Update time step counter
            timeStepCounter++;
        }

        // PDE execution
        elapsedPDE = elapsed1stThomas + elapsed2ndThomas + elapsedTranspose + elapsed1stRHS + elapsed2ndRHS;
        
        // Finish measuring total execution time
        finishTotal = omp_get_wtime();
        elapsedTotal = finishTotal - startTotal;
    }

    else if (strcmp(method, "theta-ADI") == 0)
    {
        // Start measuring total execution time
        startTotal = omp_get_wtime();
        
        while (timeStepCounter < M)
        {
            // Get time step
            timeStep = time[timeStepCounter];

            // Start measuring ODE execution time
            startPartial = omp_get_wtime();

            // Resolve ODEs
            parallelODE_theta<<<GRID_SIZE, BLOCK_SIZE>>>(d_V, d_W, d_Rv, N, timeStep, deltat, phi, theta, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
            cudaDeviceSynchronize();

            // Finish measuring ODE execution time
            finishPartial = omp_get_wtime();
            elapsedODE += finishPartial - startPartial;

            // Prepare right side of Thomas algorithm with explicit diffusion on j
            // Call the kernel
            startPartial = omp_get_wtime();
            prepareRighthandSide_jDiffusion_theta<<<GRID_SIZE, BLOCK_SIZE>>>(d_V, d_rightside, d_Rv, N, phi, theta, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor); 
            cudaDeviceSynchronize();     
            finishPartial = omp_get_wtime();
            elapsed1stRHS += finishPartial - startPartial;

            // 1st: Implicit y-axis diffusion (lines)
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas<<<numBlocks, blockSize>>>(d_rightside, N, d_la, d_lb, d_lc);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed1stThomas += finishPartial - startPartial;

            // Call the transpose kernel
            startPartial = omp_get_wtime();
            transposeDiagonalCol<<<GRID_SIZE, BLOCK_SIZE>>>(d_rightside, d_V, N, N);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsedTranspose += finishPartial - startPartial;

            // Prepare right side of Thomas algorithm with explicit diffusion on i
            // Call the kernel
            startPartial = omp_get_wtime();
            prepareRighthandSide_iDiffusion_theta<<<GRID_SIZE, BLOCK_SIZE>>>(d_V, d_rightside, d_Rv, N, phi, theta, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed2ndRHS += finishPartial - startPartial;

            // 2nd: Implicit x-axis diffusion (columns)                
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas<<<numBlocks, blockSize>>>(d_rightside, N, d_la, d_lb, d_lc); 
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed2ndThomas += finishPartial - startPartial;

            // Copy d_rightside to d_V
            startPartial = omp_get_wtime();
            cudaStatus1 = cudaMemcpy(d_V, d_rightside, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
            if (cudaStatus1 != cudaSuccess)
            {
                printf("cudaMemcpy failed device to device!\n");
                exit(EXIT_FAILURE);
            }
            finishPartial = omp_get_wtime();
            elapsedMemCopy += finishPartial - startPartial;
            elapsed2ndMemCopy += finishPartial - startPartial;

            
            // Save frames
            if (VWTag == false)
            {
                // Write frames to file
                // startPartial = omp_get_wtime();
                // if (timeStepCounter % saverate == 0 && saveDataToGif == true)
                // {
                //     //Copy memory from device to host of the matrices (2D arrays)
                //     startPartial = omp_get_wtime();
                //     cudaStatus1 = cudaMemcpy(V, d_V, N * N * sizeof(real), cudaMemcpyDeviceToHost);
                //     if (cudaStatus1 != cudaSuccess)
                //     {
                //         printf("cudaMemcpy failed 5th call!\n");
                //         exit(EXIT_FAILURE);
                //     }
                //     finishPartial = omp_get_wtime();
                //     elapsedMemCopy += finishPartial - startPartial;
                //     elapsed4thMemCopy += finishPartial - startPartial;

                //     fprintf(fpFrames, "%lf\n", time[timeStepCounter]);
                //     for (i = 0; i < N; i++)
                //     {
                //         for (j = 0; j < N; j++)
                //         {
                //             index = i * N + j;
                //             fprintf(fpFrames, "%lf ", V[index]);
                //         }
                //         fprintf(fpFrames, "\n");
                //     }
                // }
                // finishPartial = omp_get_wtime();
                // elapsedWriting += finishPartial - startPartial;
                

                // Check S1 velocity
                if (S1VelocityTag)
                {
                    // Copy memory from device to host of the matrices (2D arrays)
                    startPartial = omp_get_wtime();
                    cudaStatus1 = cudaMemcpy(V, d_V, N * N * sizeof(real), cudaMemcpyDeviceToHost);
                    if (cudaStatus1 != cudaSuccess)
                    {
                        printf("cudaMemcpy failed 5th call!\n");
                        exit(EXIT_FAILURE);
                    }
                    finishPartial = omp_get_wtime();
                    elapsedMemCopy += finishPartial - startPartial;
                    elapsed4thMemCopy += finishPartial - startPartial;

                    if (V[N - 1] >= 80)
                    {
                        S1Velocity = ((10 * (L - stim1xLimit)) / (time[timeStepCounter]));
                        S1VelocityTag = false;
                    }
                }
            }
            
            // Update time step counter
            timeStepCounter++;
        }

        // PDE execution
        elapsedPDE = elapsed1stThomas + elapsed2ndThomas + elapsedTranspose + elapsed1stRHS + elapsed2ndRHS;
        
        // Finish measuring total execution time
        finishTotal = omp_get_wtime();
        elapsedTotal = finishTotal - startTotal;
    }


    // Write infos to file
    fprintf(fpInfos, "S1 velocity: %lf m/s\n", S1Velocity);
    fprintf(fpInfos, "S2 begin: %.1lf ms\n", stim2Begin);
    fprintf(fpInfos, "PDE/ODE ratio: %d\n", PdeOdeRatio);
    fprintf(fpInfos, "ODE execution time: %lf seconds\n", elapsedODE);
    fprintf(fpInfos, "PDE execution time: %lf seconds\n", elapsedPDE);
    fprintf(fpInfos, "Writing time: %lf seconds\n", elapsedWriting);
    fprintf(fpInfos, "Total execution time with writings: %lf seconds\n", elapsedTotal);
    
    fprintf(fpInfos, "\nFor ODE and Transpose -> Grid size %d, Block size %d\n", GRID_SIZE, BLOCK_SIZE);
    fprintf(fpInfos, "Total threads: %d\n", GRID_SIZE*BLOCK_SIZE);

    fprintf(fpInfos, "\nFor PDE -> Grid size: %d, Block size: %d\n", numBlocks, blockSize);
    fprintf(fpInfos, "Total threads: %d\n", numBlocks*blockSize);
    fprintf(fpInfos, "1st Thomas algorithm time: %lf seconds\n", elapsed1stThomas);
    fprintf(fpInfos, "2nd Thomas algorithm time: %lf seconds\n", elapsed2ndThomas);
    fprintf(fpInfos, "Transpose time: %lf seconds\n", elapsedTranspose);
    fprintf(fpInfos, "1st RHS preparation time: %lf seconds\n", elapsed1stRHS);
    fprintf(fpInfos, "2nd RHS preparation time: %lf seconds\n", elapsed2ndRHS);
    fprintf(fpInfos, "Memory copy time (device to device): %lf seconds\n", elapsed2ndMemCopy);
    fprintf(fpInfos, "Memory copy time for velocity: %lf seconds\n", elapsed4thMemCopy);
    fprintf(fpInfos, "Total memory copy time: %lf seconds\n", elapsedMemCopy);
    
    fprintf(fpInfos, "\ntheta = %lf\n", theta);

    if (haveFibrosis)
    {
        fprintf(fpInfos, "Fibrosis factor: %.2lf\n", fibrosisFactor);
        fprintf(fpInfos, "Fibrosis region: (%.2lf, %.2lf) to (%.2lf, %.2lf)\n", fibrosisMinX, fibrosisMinY, fibrosisMaxX, fibrosisMaxY);
    }

    // if (saveDataToError == true)
    // {
    //     char lastFrameFileName[MAX_STRING_SIZE];
    //     sprintf(lastFrameFileName, "last-%d-%.3lf.txt", numberThreads, deltat);
    //     FILE *fpLast;
    //     sprintf(aux, "%s/%s", pathToSaveData, lastFrameFileName);
    //     fpLast = fopen(aux, "w");
    //     for (int i = 0; i < N; i++)
    //     {
    //         for (int j = 0; j < N; j++)
    //         {
    //             index = i * N + j;
    //             fprintf(fpLast, "%lf ", V[index]);
    //         }
    //         fprintf(fpLast, "\n");
    //     }
    //     fclose(fpLast);
    // }

    // Close files
    fclose(fpFrames);
    fclose(fpInfos);

    // Free memory
    free(time);

    // Free memory from host
    free(V);
    free(W);
    free(Vtilde);
    free(Wtilde);
    free(Rv);
    free(rightside);
    free(solution);
    free(la);
    free(lb);
    free(lc);
    if (strcmp(method, "theta-ADI") == 0)
    {
        free(la2);
        free(lb2);
        free(lc2);
    }
    

    // Free memory from device
    cudaFree(d_V);
    cudaFree(d_W);
    cudaFree(d_Rv);
    cudaFree(d_rightside);
    cudaFree(d_solution);
    cudaFree(d_la);
    cudaFree(d_lb);
    cudaFree(d_lc);
    if (strcmp(method, "theta-ADI") == 0)
    {
        cudaFree(d_la2);
        cudaFree(d_lb2);
        cudaFree(d_lc2);
    }
    // cudaFree(p_buffer);

}

void runAllinGPU3D(bool options[], char *method, real deltat, int numberThreads, real delta_x, char *mode)
{
    // Get options
    bool haveFibrosis = options[0];
    bool measureTotalTime = options[1];
    bool saveDataToError = options[2];
    bool saveDataToGif = options[3];
    bool measureTimeParts = options[4];
    bool measureS1Velocity = options[5];

    deltax = delta_x;
    deltay = delta_x;
    deltaz = delta_x;

    // Number of steps
    int N = round(L / deltax) + 1;                  // Spatial steps (square tissue)
    int H = round(L / deltax) + 1;                  // Spatial steps for height
    int M = round(T / deltat) + 1;                  // Number of time steps
    int PdeOdeRatio = round(deltat / deltat);       // Ratio between PDE and ODE time steps

    // Allocate and populate time array
    real *time;
    time = (real *)malloc(M * sizeof(real));
    for (int i = 0; i < M; i++)
    {
        time[i] = i * deltat;
    }

    // Allocate and initialize variables
    real *V, *W;
    V = (real *)malloc(N * N * H * sizeof(real));
    W = (real *)malloc(N * N * H * sizeof(real));
    initializeVariablesGPU3D(N, H, V, W);

    // Diffusion coefficient - isotropic
    real D = sigma / (chi * Cm);
    real phi = D * deltat / (deltax * deltax);      // For Thomas algorithm - isotropic

    // Variables
    int i, j, k;                                       // i for y-axis, j for x-axis and k for z-axis
    real actualV, actualW;
    real *Vtilde, *Wtilde, *rightside;
    Vtilde = (real *)malloc(N * N * H * sizeof(real));
    Wtilde = (real *)malloc(N * N * H * sizeof(real));
    rightside = (real *)malloc(N * N * H * sizeof(real));

    // Auxiliary arrays for Thomas algorithm 2nd order approximation
    real *la = (real *)malloc(N * sizeof(real));
    real *lb = (real *)malloc(N * sizeof(real));
    real *lc = (real *)malloc(N * sizeof(real));
    populateDiagonalThomasAlgorithm(la, lb, lc, N, phi);

    // Prefactorization
    prefactorizationThomasAlgorithm(la, lb, lc, N);

    // Discretized limits of stimulation area
    int discS1xLimit = round(stim1xLimit / deltax);
    int discS1yLimit = round(stim1yLimit / deltay);
    int discS2xMax = round(stim2xMax / deltax);
    int discS2xMin = round(stim2xMin / deltax);
    int discS2yMax = N;
    int discS2yMin = N - round(stim2yMax / deltay);
    
    // Discritized limits of fibrotic area
    int discFibxMax = round(fibrosisMaxX / deltax);
    int discFibxMin = round(fibrosisMinX / deltax);
    int discFibyMax = N - round(fibrosisMinY / deltay);
    int discFibyMin = N - round(fibrosisMaxY / deltay);

    // File names
    char framesFileName[MAX_STRING_SIZE], infosFileName[MAX_STRING_SIZE];
    sprintf(framesFileName, "frames-%d-%.3lf-%.3lf.txt", numberThreads, deltat, deltat);
    sprintf(infosFileName, "infos-%d-%.3lf-%.3lf.txt", numberThreads, deltat, deltat);
    int saverate = ceil(M / 100.0);
    FILE *fpFrames, *fpInfos;

    // Create directories and files
    char pathToSaveData[MAX_STRING_SIZE];
    if (haveFibrosis)
    {
        createDirectories(pathToSaveData, method, "AFHN-Fibro", mode);
    }
    else
    {
        createDirectories(pathToSaveData, method, "AFHN", mode);
    }
    
    // File pointers
    char aux[MAX_STRING_SIZE];
    if (VWTag == false)
    {

        sprintf(aux, "%s/%s", pathToSaveData, infosFileName);
        fpInfos = fopen(aux, "w");
    }
    else
    {
        sprintf(aux, "%s/%s", pathToSaveData, infosFileName);
        fpInfos = fopen(aux, "a");
    }
    if (saveDataToGif == false)
    {
        sprintf(aux, "%s/%s", pathToSaveData, framesFileName);
        fpFrames = fopen(aux, "a");
    }
    else
    {
        sprintf(aux, "%s/%s", pathToSaveData, framesFileName);
        fpFrames = fopen(aux, "w");
    }
    
    // CUDA variables and allocation
    int numBlocks = N * N / 100; 
    int blockSize = round(N * N / numBlocks) + 1;
    if (blockSize % 32 != 0)
    {
        blockSize = 32 * ((blockSize / 32) + 1);
    }
    printf("NumBlock: %d, BlockSize: %d\n", numBlocks, blockSize);
    
    // Diagonal kernel parameters
    dim3 block (BDIMX, BDIMY, BDIMZ);
    dim3 grid  (33, 33, 33);

    real *d_V, *d_W, *d_rightside;
    real *d_la, *d_lb, *d_lc;
    cudaError_t cudaStatus1, cudaStatus2, cudaStatus3, cudaStatus4, cudaStatus5, cudaStatus6;
    
    cudaStatus1 = cudaMalloc(&d_V, N * N * H * sizeof(real));
    cudaStatus2 = cudaMalloc(&d_W, N * N * H * sizeof(real));
    cudaStatus3 = cudaMalloc(&d_rightside, N * N * H * sizeof(real));
    cudaStatus5 = cudaMalloc(&d_la, N * sizeof(real));
    cudaStatus6 = cudaMalloc(&d_lb, N * sizeof(real));
    cudaStatus6 = cudaMalloc(&d_lc, N * sizeof(real));
    if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess || cudaStatus3 != cudaSuccess || cudaStatus5 != cudaSuccess || cudaStatus6 != cudaSuccess)
    {
        printf("cudaMalloc failed!\n");
        exit(EXIT_FAILURE);
    }
    printf("All cudaMallocs done!\n");

    // Copy memory from host to device of the matrices (2D arrays)
    cudaStatus1 = cudaMemcpy(d_V, V, N * N * H * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus2 = cudaMemcpy(d_W, W, N * N * H * sizeof(real), cudaMemcpyHostToDevice);
    if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess)
    {
        printf("cudaMemcpy failed!\n");
        exit(EXIT_FAILURE);
    }

    // Copy memory of diagonals from host to device
    cudaStatus1 = cudaMemcpy(d_la, la, N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus2 = cudaMemcpy(d_lb, lb, N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus3 = cudaMemcpy(d_lc, lc, N * sizeof(real), cudaMemcpyHostToDevice);
    if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess || cudaStatus3 != cudaSuccess)
    {
        printf("cudaMemcpy failed!\n");
        exit(EXIT_FAILURE);
    }

    /*--------------------
    --  ADI 1st order   --
    ----------------------*/
    int index;
    if (strcmp(method, "OS-ADI") == 0)
    {
        // Start measuring total execution time
        startTotal = omp_get_wtime();
        
        while (timeStepCounter < M)
        {
            // Get time step
            timeStep = time[timeStepCounter];

            // Start measuring ODE execution time
            startPartial = omp_get_wtime();

            // Resolve ODEs
            parallelODE3D<<<grid, block>>>(d_V, d_W, d_rightside, N, timeStep, deltat, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax);
            cudaDeviceSynchronize();

            // Finish measuring ODE execution time
            finishPartial = omp_get_wtime();
            elapsedODE += finishPartial - startPartial;

            // Resolve PDEs (Diffusion)
            // 1st: Implicit y-axis diffusion (lines)
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas3D<<<numBlocks, blockSize>>>(d_rightside, N, d_la, d_lb, d_lc);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed1stThomas += finishPartial - startPartial;

            // Call the mapping kernel
            startPartial = omp_get_wtime();
            mapping1<<<grid, block>>>(d_rightside, d_V, N, N, N);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsedTranspose += finishPartial - startPartial;

            // 2nd: Implicit x-axis diffusion (columns)                
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas3D<<<numBlocks, blockSize>>>(d_V, N, d_la, d_lb, d_lc);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed2ndThomas += finishPartial - startPartial;

            // Call the mapping kernel
            startPartial = omp_get_wtime();
            mapping2<<<grid, block>>>(d_V, d_rightside, N, N, N);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsedTranspose2 += finishPartial - startPartial;

            // 3rd: Implicit z-axis diffusion (height)                
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas3D<<<numBlocks, blockSize>>>(d_rightside, N, d_la, d_lb, d_lc);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed3rdThomas += finishPartial - startPartial;

            // Call the mapping kernel
            startPartial = omp_get_wtime();
            mapping3<<<grid, block>>>(d_rightside, d_V, N, N, N);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsedTranspose3 += finishPartial - startPartial;

            // Finish measuring PDE execution time
            elapsedPDE += elapsed1stThomas + elapsed2ndThomas + elapsed3rdThomas + elapsedTranspose + elapsedTranspose2 + elapsedTranspose3;
            
            // Save frames
            if (VWTag == false)
            {
                // Write frames to file
                startPartial = omp_get_wtime();
                if (timeStepCounter % saverate == 0 && saveDataToGif == true)
                {
                    // Copy memory from device to host of the matrices (2D arrays)
                    startPartial = omp_get_wtime();
                    cudaStatus1 = cudaMemcpy(V, d_V, N * N * H * sizeof(real), cudaMemcpyDeviceToHost);
                    if (cudaStatus1 != cudaSuccess)
                    {
                        printf("cudaMemcpy failed writing!\n");
                        exit(EXIT_FAILURE);
                    }
                    finishPartial = omp_get_wtime();
                    elapsedMemCopy += finishPartial - startPartial;
                    elapsed4thMemCopy += finishPartial - startPartial;

                    fprintf(fpFrames, "%lf\n", time[timeStepCounter]);
                    for (i = 0; i < N; i++)
                    {
                        for (j = 0; j < N; j++)
                        {
                            index = i * N + j + (((N-1))*N*N);
                            fprintf(fpFrames, "%lf ", V[index]);
                        }
                        fprintf(fpFrames, "\n");
                    }
                }
                finishPartial = omp_get_wtime();
                elapsedWriting += finishPartial - startPartial;
               

                // Check S1 velocity
                if (S1VelocityTag)
                {
                    // Copy memory from device to host of the matrices (2D arrays)
                    startPartial = omp_get_wtime();
                    cudaStatus1 = cudaMemcpy(V, d_V, N * N * H * sizeof(real), cudaMemcpyDeviceToHost);
                    if (cudaStatus1 != cudaSuccess)
                    {
                        printf("cudaMemcpy failed velocity!\n");
                        exit(EXIT_FAILURE);
                    }
                    finishPartial = omp_get_wtime();
                    elapsedMemCopy += finishPartial - startPartial;
                    elapsed4thMemCopy += finishPartial - startPartial;

                    if (V[(N - 1)+ (((N-1))*N*N)] >= 80)
                    {
                        S1Velocity = ((10 * (L - stim1xLimit)) / (time[timeStepCounter]));
                        S1VelocityTag = false;
                    }
                }
            }
            
            // Update time step counter
            timeStepCounter++;
        }
        
        // Finish measuring total execution time
        finishTotal = omp_get_wtime();
        elapsedTotal = finishTotal - startTotal;
    }

    // Write infos to file
    fprintf(fpInfos, "S1 velocity: %lf m/s\n", S1Velocity);
    fprintf(fpInfos, "S2 begin: %.1lf ms\n", stim2Begin);
    fprintf(fpInfos, "PDE/ODE ratio: %d\n", PdeOdeRatio);
    fprintf(fpInfos, "ODE execution time: %lf seconds\n", elapsedODE);
    fprintf(fpInfos, "PDE execution time: %lf seconds\n", elapsedPDE);
    fprintf(fpInfos, "Writing time: %lf seconds\n", elapsedWriting);
    fprintf(fpInfos, "Total execution time with writings: %lf seconds\n", elapsedTotal);
    
    fprintf(fpInfos, "\nFor ODE and Transpose -> Grid size (%d, %d, %d): %d, Block size (%d, %d, %d): %d\n", grid.x, grid.y, grid.z, grid.x*grid.y*grid.z, block.x, block.y, block.z, block.x*block.y*block.z);
    fprintf(fpInfos, "Total threads: %d\n", grid.x*grid.y*grid.z*block.x*block.y*block.z);

    fprintf(fpInfos, "\nFor PDE -> Grid size: %d, Block size: %d\n", numBlocks, blockSize);
    fprintf(fpInfos, "Total threads: %d\n", numBlocks*blockSize);
    fprintf(fpInfos, "1st Thomas algorithm time: %lf seconds\n", elapsed1stThomas);
    fprintf(fpInfos, "2nd Thomas algorithm time: %lf seconds\n", elapsed2ndThomas);
    fprintf(fpInfos, "3rd Thomas algorithm time: %lf seconds\n", elapsed3rdThomas);
    fprintf(fpInfos, "1st Mapping time: %lf seconds\n", elapsedTranspose);
    fprintf(fpInfos, "2nd Mapping time: %lf seconds\n", elapsedTranspose2);
    fprintf(fpInfos, "3rd Mapping time: %lf seconds\n", elapsedTranspose3);
    fprintf(fpInfos, "Memory copy time (vel): %lf seconds\n", elapsed4thMemCopy);
    fprintf(fpInfos, "Total memory copy time: %lf seconds\n", elapsedMemCopy);
    
    if (haveFibrosis)
    {
        fprintf(fpInfos, "Fibrosis factor: %.2lf\n", fibrosisFactor);
        fprintf(fpInfos, "Fibrosis region: (%.2lf, %.2lf) to (%.2lf, %.2lf)\n", fibrosisMinX, fibrosisMinY, fibrosisMaxX, fibrosisMaxY);
    }

    // if (saveDataToError == true)
    // {
    //     char lastFrameFileName[MAX_STRING_SIZE];
    //     sprintf(lastFrameFileName, "last-%d-%.3lf-%.3lf.txt", numberThreads, deltat, deltat);
    //     FILE *fpLast;
    //     sprintf(aux, "%s/%s", pathToSaveData, lastFrameFileName);
    //     fpLast = fopen(aux, "w");
    //     for (int i = 0; i < N; i++)
    //     {
    //         for (int j = 0; j < N; j++)
    //         {
    //             index = i * N + j;
    //             fprintf(fpLast, "%lf ", V[index]);
    //         }
    //         fprintf(fpLast, "\n");
    //     }
    //     fclose(fpLast);
    // }

    // Close files
    fclose(fpFrames);
    fclose(fpInfos);

    // Free memory
    free(time);

    // Free memory from host
    free(V);
    free(W);
    free(Vtilde);
    free(Wtilde);
    free(rightside);
    free(la);
    free(lb);
    free(lc);

    // Free memory from device
    cudaFree(d_V);
    cudaFree(d_W);
    cudaFree(d_rightside);
    cudaFree(d_la);
    cudaFree(d_lb);
    cudaFree(d_lc);

}

#endif // AFHN

//###########################################
//##                                       ##
//##     ten Tusscher 2006 model (TT2)     ##
//##                                       ##
//###########################################
#if defined(TT2)
void runAllinCPU(bool options[], char *method, real deltat, int numberThreads, real delta_x, char *mode, real theta, int number_of_exec){}
void runODEinCPUandPDEinGPU(bool options[], char *method, real deltat, int numberThreads, real delta_x, char *mode){}
void runAllinGPU3D(bool options[], char *method, real deltat, int numberThreads, real delta_x, char *mode){}
void runAllinGPU(bool options[], char *method, real deltat, int numberThreads, real delta_x, char *mode, real theta, int number_of_exec)
{
    // Get options
    bool haveFibrosis = options[0];
    bool measureTotalTime = options[1];
    bool saveDataToError = options[2];
    bool saveDataToGif = options[3];
    bool measureTimeParts = options[4];
    bool measureS1Velocity = options[5];

    deltax = delta_x;
    deltay = delta_x;

    // Number of steps
    int N = round(L / deltax) + 1;                  // Spatial steps (square tissue)
    int M = round(Time / deltat) + 1;                  // Number of time steps
    int PdeOdeRatio = round(deltat / deltat);       // Ratio between PDE and ODE time steps

    // Allocate and populate time array
    real *time;
    time = (real *)malloc(M * sizeof(real));
    for (int i = 0; i < M; i++)
    {
        time[i] = i * deltat;
    }

    // Allocate and initialize variables
    real *V, *X_r1, *X_r2, *X_s, *m, *h, *j, *d, *f, *f2, *fCass, *s, *r, *Ca_i, *Ca_SR, *Ca_SS, *R_prime, *Na_i, *K_i;
    V = (real *)malloc(N * N * sizeof(real));
    X_r1 = (real *)malloc(N * N * sizeof(real));
    X_r2 = (real *)malloc(N * N * sizeof(real));
    X_s = (real *)malloc(N * N * sizeof(real));
    m = (real *)malloc(N * N * sizeof(real));
    h = (real *)malloc(N * N * sizeof(real));
    j = (real *)malloc(N * N * sizeof(real));
    d = (real *)malloc(N * N * sizeof(real));
    f = (real *)malloc(N * N * sizeof(real));
    f2 = (real *)malloc(N * N * sizeof(real));
    fCass = (real *)malloc(N * N * sizeof(real));
    s = (real *)malloc(N * N * sizeof(real));
    r = (real *)malloc(N * N * sizeof(real));
    Ca_i = (real *)malloc(N * N * sizeof(real));
    Ca_SR = (real *)malloc(N * N * sizeof(real));
    Ca_SS = (real *)malloc(N * N * sizeof(real));
    R_prime = (real *)malloc(N * N * sizeof(real));
    Na_i = (real *)malloc(N * N * sizeof(real));
    K_i = (real *)malloc(N * N * sizeof(real));
    initializeVariablesGPU(N, V, X_r1, X_r2, X_s, m, h, j, d, f, f2, fCass, s, r, Ca_i, Ca_SR, Ca_SS, R_prime, Na_i, K_i);

    // Diffusion coefficient - isotropic
    // real D = sigma / (chi * Cm);
    real D = sigma / chi; // As in the paper
    real phi = D * deltat / (deltax * deltax);      // For Thomas algorithm - isotropic

    // Variables
    int i, jj;
    real *Rv, *rightside, *solution;
    Rv = (real *)malloc(N * N * sizeof(real));
    rightside = (real *)malloc(N * N * sizeof(real));
    solution = (real *)malloc(N * N * sizeof(real));

    // Auxiliary arrays for Thomas algorithm 2nd order approximation
    real *la = (real *)malloc(N * sizeof(real));
    real *lb = (real *)malloc(N * sizeof(real));
    real *lc = (real *)malloc(N * sizeof(real));

    // For the Theta method, it will be necessary new auxiliary arrays to be multiplied by 1-theta
    real *la2, *lb2, *lc2;

    if (strcmp(method, "OS-ADI") == 0)
        populateDiagonalThomasAlgorithm(la, lb, lc, N, phi);
    else if (strcmp(method, "SSI-ADI") == 0 || strcmp(method, "MOSI-ADI") == 0 || strcmp(method, "MOSI-2-ADI") == 0)
        populateDiagonalThomasAlgorithm(la, lb, lc, N, 0.5*phi);
    else if (strcmp(method, "theta-ADI") == 0)
    {
        populateDiagonalThomasAlgorithm(la, lb, lc, N, theta*phi);

        // Mallloc new arrays
        la2 = (real *)malloc(N * sizeof(real));
        lb2 = (real *)malloc(N * sizeof(real));
        lc2 = (real *)malloc(N * sizeof(real));
        populateDiagonalThomasAlgorithm(la2, lb2, lc2, N, (1-theta)*phi);
    }
        
    // Prefactorization
    prefactorizationThomasAlgorithm(la, lb, lc, N);
    if (strcmp(method, "theta-ADI") == 0)
        prefactorizationThomasAlgorithm(la2, lb2, lc2, N);

    // Discretized limits of stimulation area
    int discS1xLimit = round(stim1xLimit / deltax);
    int discS1yLimit = round(stim1yLimit / deltay);
    int discS2xMax = round(stim2xMax / deltax);
    int discS2xMin = round(stim2xMin / deltax);
    int discS2yMax = N;
    int discS2yMin = N - round(stim2yMax / deltay);
    
    // Discritized limits of fibrotic area
    int discFibxMax = round(fibrosisMaxX / deltax);
    int discFibxMin = round(fibrosisMinX / deltax);
    int discFibyMax = N - round(fibrosisMinY / deltay);
    int discFibyMin = N - round(fibrosisMaxY / deltay);

    // File names
    char framesFileName[MAX_STRING_SIZE], infosFileName[MAX_STRING_SIZE];
    sprintf(framesFileName, "frames-%d-%.3lf.txt", numberThreads, deltat);
    sprintf(infosFileName, "infos-%d-%.3lf-%.2lf.txt", numberThreads, deltat, theta);
    int saverate = ceil(M / 100.0);
    FILE *fpFrames, *fpInfos;

    // Create directories and files
    char pathToSaveData[MAX_STRING_SIZE];
    if (haveFibrosis)
    {
        createDirectories(pathToSaveData, method, "TT2-Fibro", mode);
        // createDirectoriesScript(pathToSaveData, method, "TT2-Fibro", mode, numberThreads, number_of_exec);
    }
    else
    {
        createDirectories(pathToSaveData, method, "TT2", mode);
        // createDirectoriesScript(pathToSaveData, method, "TT2", mode, numberThreads, number_of_exec);
    }
    
    // File pointers
    char aux[MAX_STRING_SIZE];
    sprintf(aux, "%s/%s", pathToSaveData, infosFileName);
    if (VWTag == false)
        fpInfos = fopen(aux, "w");
    else
        fpInfos = fopen(aux, "a");
    
    sprintf(aux, "%s/%s", pathToSaveData, framesFileName);
    if (saveDataToGif == false)
        fpFrames = fopen(aux, "a");
    else
        fpFrames = fopen(aux, "w");
    
    // CUDA variables and allocation
    int numBlocks = N / 100; 
    int blockSize = round(N / numBlocks) + 1;
    if (blockSize % 32 != 0)
        blockSize = 32 * ((blockSize / 32) + 1);
    printf("NumBlock: %d, BlockSize: %d\n", numBlocks, blockSize);
    
    // Diagonal kernel parameters
    dim3 block (BDIMX, BDIMY);
    dim3 grid  ((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    real *d_V, *d_X_r1, *d_X_r2, *d_X_s, *d_m, *d_h, *d_j, *d_d, *d_f, *d_f2, *d_fCass, *d_s, *d_r, *d_Ca_i, *d_Ca_SR, *d_Ca_SS, *d_R_prime, *d_Na_i, *d_K_i;
    real *d_rightside, *d_solution, *d_Rv;
    real *d_la, *d_lb, *d_lc;
    real *d_la2, *d_lb2, *d_lc2;
    cudaError_t cudaStatus1, cudaStatus2, cudaStatus3, cudaStatus4, cudaStatus5, cudaStatus6, cudaStatus7, cudaStatus8;
    
    cudaStatus1 = cudaMalloc(&d_V, N * N * sizeof(real));
    cudaStatus1 = cudaMalloc(&d_X_r1, N * N * sizeof(real));
    cudaStatus2 = cudaMalloc(&d_X_r2, N * N * sizeof(real));
    cudaStatus3 = cudaMalloc(&d_X_s, N * N * sizeof(real));
    cudaStatus4 = cudaMalloc(&d_m, N * N * sizeof(real));
    cudaStatus5 = cudaMalloc(&d_h, N * N * sizeof(real));
    cudaStatus6 = cudaMalloc(&d_j, N * N * sizeof(real));
    cudaStatus1 = cudaMalloc(&d_d, N * N * sizeof(real));
    cudaStatus1 = cudaMalloc(&d_f, N * N * sizeof(real));
    cudaStatus1 = cudaMalloc(&d_f2, N * N * sizeof(real));
    cudaStatus2 = cudaMalloc(&d_fCass, N * N * sizeof(real));
    cudaStatus3 = cudaMalloc(&d_s, N * N * sizeof(real));
    cudaStatus4 = cudaMalloc(&d_r, N * N * sizeof(real));
    cudaStatus5 = cudaMalloc(&d_Ca_i, N * N * sizeof(real));
    cudaStatus6 = cudaMalloc(&d_Ca_SR, N * N * sizeof(real));
    cudaStatus1 = cudaMalloc(&d_Ca_SS, N * N * sizeof(real));
    cudaStatus1 = cudaMalloc(&d_R_prime, N * N * sizeof(real));
    cudaStatus1 = cudaMalloc(&d_Na_i, N * N * sizeof(real));
    cudaStatus2 = cudaMalloc(&d_K_i, N * N * sizeof(real));
    cudaStatus3 = cudaMalloc(&d_rightside, N * N * sizeof(real));
    cudaStatus4 = cudaMalloc(&d_solution, N * N * sizeof(real));
    cudaStatus5 = cudaMalloc(&d_Rv, N * N * sizeof(real));
    cudaStatus6 = cudaMalloc(&d_la, N * sizeof(real));
    cudaStatus7 = cudaMalloc(&d_lb, N * sizeof(real));
    cudaStatus8 = cudaMalloc(&d_lc, N * sizeof(real));
    if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess || cudaStatus3 != cudaSuccess || cudaStatus4 != cudaSuccess || cudaStatus5 != cudaSuccess || cudaStatus6 != cudaSuccess || cudaStatus7 != cudaSuccess || cudaStatus8 != cudaSuccess)
    {
        printf("cudaMalloc failed call!\n");
        exit(EXIT_FAILURE);
    }
    printf("All cudaMallocs done!\n");
    
    if (strcmp(method, "theta-ADI") == 0)
    {
        cudaStatus1 = cudaMalloc(&d_la2, N * sizeof(real));
        cudaStatus2 = cudaMalloc(&d_lb2, N * sizeof(real));
        cudaStatus3 = cudaMalloc(&d_lc2, N * sizeof(real));
        if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess || cudaStatus3 != cudaSuccess)
        {
            printf("cudaMalloc failed for theta-Method second aux arrays for Thomas!\n");
            exit(EXIT_FAILURE);
        }
    }

    // Copy memory from host to device of the matrices (2D arrays)
    cudaStatus1 = cudaMemcpy(d_V, V, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_X_r1, X_r1, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_X_r2, X_r2, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_X_s, X_s, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_m, m, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_h, h, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_j, j, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_d, d, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_f, f, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_f2, f2, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_fCass, fCass, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_s, s, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_r, r, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_Ca_i, Ca_i, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_Ca_SR, Ca_SR, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_Ca_SS, Ca_SS, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_R_prime, R_prime, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_Na_i, Na_i, N * N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus1 = cudaMemcpy(d_K_i, K_i, N * N * sizeof(real), cudaMemcpyHostToDevice);
    if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess)
    {
        printf("cudaMemcpy failed 1st call!\n");
        exit(EXIT_FAILURE);
    }
    // Free memory
    free(X_r1);
    free(X_r2);
    free(X_s);
    free(m);
    free(h);
    free(j);
    free(d);
    free(f);
    free(f2);
    free(fCass);
    free(s);
    free(r);
    free(Ca_i);
    free(Ca_SR);
    free(Ca_SS);
    free(R_prime);
    free(Na_i);
    free(K_i);

    // For MOSI-2
    real *d_V_2, *d_X_r1_2, *d_X_r2_2, *d_X_s_2, *d_m_2, *d_h_2, *d_j_2, *d_d_2, *d_f_2, *d_f2_2, *d_fCass_2, *d_s_2, *d_r_2, *d_Ca_i_2, *d_Ca_SR_2, *d_Ca_SS_2, *d_R_prime_2, *d_Na_i_2, *d_K_i_2;
    cudaStatus1 = cudaMalloc(&d_V_2, N * N * sizeof(real));
    cudaStatus1 = cudaMalloc(&d_X_r1_2, N * N * sizeof(real));
    cudaStatus2 = cudaMalloc(&d_X_r2_2, N * N * sizeof(real));
    cudaStatus3 = cudaMalloc(&d_X_s_2, N * N * sizeof(real));
    cudaStatus4 = cudaMalloc(&d_m_2, N * N * sizeof(real));
    cudaStatus5 = cudaMalloc(&d_h_2, N * N * sizeof(real));
    cudaStatus6 = cudaMalloc(&d_j_2, N * N * sizeof(real));
    cudaStatus1 = cudaMalloc(&d_d_2, N * N * sizeof(real));
    cudaStatus1 = cudaMalloc(&d_f_2, N * N * sizeof(real));
    cudaStatus1 = cudaMalloc(&d_f2_2, N * N * sizeof(real));
    cudaStatus2 = cudaMalloc(&d_fCass_2, N * N * sizeof(real));
    cudaStatus3 = cudaMalloc(&d_s_2, N * N * sizeof(real));
    cudaStatus4 = cudaMalloc(&d_r_2, N * N * sizeof(real));
    cudaStatus5 = cudaMalloc(&d_Ca_i_2, N * N * sizeof(real));
    cudaStatus6 = cudaMalloc(&d_Ca_SR_2, N * N * sizeof(real));
    cudaStatus1 = cudaMalloc(&d_Ca_SS_2, N * N * sizeof(real));
    cudaStatus1 = cudaMalloc(&d_R_prime_2, N * N * sizeof(real));
    cudaStatus1 = cudaMalloc(&d_Na_i_2, N * N * sizeof(real));
    cudaStatus2 = cudaMalloc(&d_K_i_2, N * N * sizeof(real));
    cudaStatus1 = cudaMemcpy(d_V_2, d_V, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_X_r1_2, d_X_r1, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_X_r2_2, d_X_r2, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_X_s_2, d_X_s, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_m_2, d_m, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_h_2, d_h, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_j_2, d_j, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_d_2, d_d, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_f_2, d_f, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_f2_2, d_f2, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_fCass_2, d_fCass, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_s_2, d_s, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_r_2, d_r, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_Ca_i_2, d_Ca_i, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_Ca_SR_2, d_Ca_SR, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_Ca_SS_2, d_Ca_SS, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_R_prime_2, d_R_prime, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_Na_i_2, d_Na_i, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaStatus1 = cudaMemcpy(d_K_i_2, d_K_i, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
    if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess)
    {
        printf("cudaMalloc MOSI2 failed!\n");
        exit(EXIT_FAILURE);
    }

    // Copy memory of diagonals from host to device
    cudaStatus1 = cudaMemcpy(d_la, la, N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus2 = cudaMemcpy(d_lb, lb, N * sizeof(real), cudaMemcpyHostToDevice);
    cudaStatus3 = cudaMemcpy(d_lc, lc, N * sizeof(real), cudaMemcpyHostToDevice);
    if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess || cudaStatus3 != cudaSuccess)
    {
        printf("cudaMemcpy failed 2nd call!\n");
        exit(EXIT_FAILURE);
    }

    if (strcmp(method, "theta-ADI") == 0)
    {
        cudaStatus1 = cudaMemcpy(d_la2, la2, N * sizeof(real), cudaMemcpyHostToDevice);
        cudaStatus2 = cudaMemcpy(d_lb2, lb2, N * sizeof(real), cudaMemcpyHostToDevice);
        cudaStatus3 = cudaMemcpy(d_lc2, lc2, N * sizeof(real), cudaMemcpyHostToDevice);
        if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess || cudaStatus3 != cudaSuccess)
        {
            printf("cudaMemcpy failed for theta-Method second aux arrays for Thomas!\n");
            exit(EXIT_FAILURE);
        }
    }

    int GRID_SIZE = ceil((N*N*1.0) / (BLOCK_SIZE*1.0));

    // Streams for parallel execution
    cudaStream_t stream1, stream2;
    cudaStatus1 = cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStatus1 = cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

    int index;
    if (strcmp(method, "MOSI-ADI") == 0)
    {
        // Start measuring total execution time
        startTotal = omp_get_wtime();
        
        while (timeStepCounter < M)
        {
            // Get time step
            timeStep = time[timeStepCounter];

            // Start measuring ODE execution time
            startPartial = omp_get_wtime();

            // Resolve ODEs
            parallelODE_MOSI<<<GRID_SIZE, BLOCK_SIZE, 0, stream1>>>(d_V, d_X_r1, d_X_r2, d_X_s, d_m, d_h, d_j, d_d, d_f, d_f2, d_fCass, d_s, d_r, d_Ca_i, d_Ca_SR, d_Ca_SS, d_R_prime, d_Na_i, d_K_i, d_Rv, N, timeStep, deltat, phi, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
            cudaDeviceSynchronize();

            // Finish measuring ODE execution time
            finishPartial = omp_get_wtime();
            elapsedODE += finishPartial - startPartial;

            // Prepare right side of Thomas algorithm with explicit diffusion on j
            // Call the kernel
            startPartial = omp_get_wtime();
            prepareRighthandSide_jDiffusion<<<GRID_SIZE, BLOCK_SIZE, 0, stream1>>>(d_V, d_rightside, d_Rv, N, phi, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor); 
            cudaDeviceSynchronize();     
            finishPartial = omp_get_wtime();
            elapsed1stRHS += finishPartial - startPartial;

            // 1st: Implicit y-axis diffusion (lines)
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas<<<numBlocks, blockSize, 0, stream1>>>(d_rightside, N, d_la, d_lb, d_lc);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed1stThomas += finishPartial - startPartial;

            // Call the transpose kernel
            startPartial = omp_get_wtime();
            transposeDiagonalCol<<<GRID_SIZE, BLOCK_SIZE, 0, stream1>>>(d_rightside, d_V, N, N);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsedTranspose += finishPartial - startPartial;

            // Prepare right side of Thomas algorithm with explicit diffusion on i
            // Call the kernel
            startPartial = omp_get_wtime();
            prepareRighthandSide_iDiffusion<<<GRID_SIZE, BLOCK_SIZE, 0, stream1>>>(d_V, d_rightside, d_Rv, N, phi, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed2ndRHS += finishPartial - startPartial;

            // 2nd: Implicit x-axis diffusion (columns)                
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas<<<numBlocks, blockSize, 0, stream1>>>(d_rightside, N, d_la, d_lb, d_lc);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed2ndThomas += finishPartial - startPartial;

            // Copy d_rightside to d_V
            startPartial = omp_get_wtime();
            cudaStatus1 = cudaMemcpy(d_V, d_rightside, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
            if (cudaStatus1 != cudaSuccess)
            {
                printf("cudaMemcpy failed device to device!\n");
                exit(EXIT_FAILURE);
            }
            finishPartial = omp_get_wtime();
            elapsedMemCopy += finishPartial - startPartial;
            elapsed2ndMemCopy += finishPartial - startPartial;

            
            // Save frames
            if (VWTag == false)
            {
                // Write frames to file
                startPartial = omp_get_wtime();
                if (timeStepCounter % saverate == 0 && saveDataToGif == true)
                {
                    //Copy memory from device to host of the matrices (2D arrays)
                    startPartial = omp_get_wtime();
                    cudaStatus1 = cudaMemcpy(V, d_V, N * N * sizeof(real), cudaMemcpyDeviceToHost);
                    if (cudaStatus1 != cudaSuccess)
                    {
                        printf("cudaMemcpy failed 5th call!\n");
                        exit(EXIT_FAILURE);
                    }
                    finishPartial = omp_get_wtime();
                    elapsedMemCopy += finishPartial - startPartial;
                    elapsed4thMemCopy += finishPartial - startPartial;

                    fprintf(fpFrames, "%lf\n", time[timeStepCounter]);
                    for (i = 0; i < N; i++)
                    {
                        for (jj = 0; jj < N; jj++)
                        {
                            index = i * N + jj;
                            fprintf(fpFrames, "%lf ", V[index]);
                        }
                        fprintf(fpFrames, "\n");
                    }
                }
                finishPartial = omp_get_wtime();
                elapsedWriting += finishPartial - startPartial;


                // Check S1 velocity
                if (S1VelocityTag)
                {
                    // Copy memory from device to host of the matrices (2D arrays)
                    startPartial = omp_get_wtime();
                    cudaStatus1 = cudaMemcpy(V, d_V, N * N * sizeof(real), cudaMemcpyDeviceToHost);
                    if (cudaStatus1 != cudaSuccess)
                    {
                        printf("cudaMemcpy failed 5th call!\n");
                        exit(EXIT_FAILURE);
                    }
                    finishPartial = omp_get_wtime();
                    elapsedMemCopy += finishPartial - startPartial;
                    elapsed4thMemCopy += finishPartial - startPartial;

                    if (V[N - 1] >= 10)
                    {
                        S1Velocity = ((10 * (L - stim1xLimit)) / (time[timeStepCounter]));
                        S1VelocityTag = false;
                    }
                }
            }
            
            // Update time step counter
            timeStepCounter++;
        }

        // PDE execution
        elapsedPDE = elapsed1stThomas + elapsed2ndThomas + elapsedTranspose + elapsed1stRHS + elapsed2ndRHS;
        
        // Finish measuring total execution time
        finishTotal = omp_get_wtime();
        elapsedTotal = finishTotal - startTotal;
    }

    else if (strcmp(method, "MOSI-2-ADI") == 0)
    {
        // Start measuring total execution time
        startTotal = omp_get_wtime();
        
        while (timeStepCounter < M)
        {
            // Get time step
            timeStep = time[timeStepCounter];

            // Start measuring ODE execution time
            startPartial = omp_get_wtime();

            // Resolve ODEs
            parallelODE_MOSI_2<<<GRID_SIZE, BLOCK_SIZE, 0, stream1>>>(d_V, d_X_r1, d_X_r2, d_X_s, d_m, d_h, d_j, d_d, d_f, d_f2, d_fCass, d_s, d_r, d_Ca_i, d_Ca_SR, d_Ca_SS, d_R_prime, d_Na_i, d_K_i, d_Rv, N, timeStep, deltat, phi, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
            // cudaStreamSynchronize(stream1);
            cudaDeviceSynchronize();

            // Finish measuring ODE execution time
            finishPartial = omp_get_wtime();
            elapsedODE += finishPartial - startPartial;

            // Update state variables
            startPartial = omp_get_wtime();
            parallelODE_MOSI_USV<<<GRID_SIZE, BLOCK_SIZE, 0, stream2>>>(d_V, d_X_r1, d_X_r2, d_X_s, d_m, d_h, d_j, d_d, d_f, d_f2, d_fCass, d_s, d_r, d_Ca_i, d_Ca_SR, d_Ca_SS, d_R_prime, d_Na_i, d_K_i, N, timeStep, deltat, phi, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
            cudaStreamSynchronize(stream2);
            finishPartial = omp_get_wtime();
            elapsedTranspose2 += finishPartial - startPartial; 

            // Update state variables
            // parallelODE_MOSI_USV<<<GRID_SIZE, BLOCK_SIZE, 0, stream2>>>(d_V_2, d_X_r1_2, d_X_r2_2, d_X_s_2, d_m_2, d_h_2, d_j_2, d_d_2, d_f_2, d_f2_2, d_fCass_2, d_s_2, d_r_2, d_Ca_i_2, d_Ca_SR_2, d_Ca_SS_2, d_R_prime_2, d_Na_i_2, d_K_i_2, N, timeStep, deltat, phi, discS1xLimit, discS1yLimit, discS2xMin, discS2xMax, discS2yMin, discS2yMax, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
            // cudaStreamSynchronize(stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_X_r1, d_X_r1_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_X_r2, d_X_r2_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_X_s, d_X_s_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_m, d_m_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_h, d_h_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_j, d_j_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_d, d_d_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_f, d_f_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_f2, d_f2_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_fCass, d_fCass_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_s, d_s_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_r, d_r_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_Ca_i, d_Ca_i_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_Ca_SR, d_Ca_SR_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_Ca_SS, d_Ca_SS_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_R_prime, d_R_prime_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_Na_i, d_Na_i_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // cudaStatus1 = cudaMemcpyAsync(d_K_i, d_K_i_2, N * N * sizeof(real), cudaMemcpyDeviceToDevice, stream2);
            // if (cudaStatus1 != cudaSuccess)
            // {
            //     printf("cudaMemcpy failed assynch device to device!\n");
            //     exit(EXIT_FAILURE);
            // }

            // Prepare right side of Thomas algorithm with explicit diffusion on j
            // Call the kernel
            startPartial = omp_get_wtime();
            prepareRighthandSide_jDiffusion<<<GRID_SIZE, BLOCK_SIZE, 0, stream1>>>(d_V, d_rightside, d_Rv, N, phi, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor); 
            cudaStreamSynchronize(stream1);
            // cudaDeviceSynchronize();
            
            
            finishPartial = omp_get_wtime();
            elapsed1stRHS += finishPartial - startPartial;

            // 1st: Implicit y-axis diffusion (lines)
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas<<<numBlocks, blockSize, 0, stream1>>>(d_rightside, N, d_la, d_lb, d_lc);
            cudaStreamSynchronize(stream1);
            // cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed1stThomas += finishPartial - startPartial;

            // Call the transpose kernel
            startPartial = omp_get_wtime();
            transposeDiagonalCol<<<GRID_SIZE, BLOCK_SIZE, 0, stream1>>>(d_rightside, d_V, N, N);
            cudaStreamSynchronize(stream1);
            // cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsedTranspose += finishPartial - startPartial;

            // Prepare right side of Thomas algorithm with explicit diffusion on i
            // Call the kernel
            startPartial = omp_get_wtime();
            prepareRighthandSide_iDiffusion<<<GRID_SIZE, BLOCK_SIZE, 0, stream1>>>(d_V, d_rightside, d_Rv, N, phi, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
            cudaStreamSynchronize(stream1);
            // cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed2ndRHS += finishPartial - startPartial;

            // 2nd: Implicit x-axis diffusion (columns)                
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas<<<numBlocks, blockSize, 0, stream1>>>(d_rightside, N, d_la, d_lb, d_lc);
            cudaStreamSynchronize(stream1);
            // cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed2ndThomas += finishPartial - startPartial;

            // Copy d_rightside to d_V
            startPartial = omp_get_wtime();
            cudaStatus1 = cudaMemcpy(d_V, d_rightside, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
            if (cudaStatus1 != cudaSuccess)
            {
                printf("cudaMemcpy failed device to device!\n");
                exit(EXIT_FAILURE);
            }
            finishPartial = omp_get_wtime();
            elapsedMemCopy += finishPartial - startPartial;
            elapsed2ndMemCopy += finishPartial - startPartial;

            // Save frames
            if (VWTag == false)
            {
                // Write frames to file
                startPartial = omp_get_wtime();
                if (timeStepCounter % saverate == 0 && saveDataToGif == true)
                {
                    //Copy memory from device to host of the matrices (2D arrays)
                    startPartial = omp_get_wtime();
                    cudaStatus1 = cudaMemcpy(V, d_V, N * N * sizeof(real), cudaMemcpyDeviceToHost);
                    if (cudaStatus1 != cudaSuccess)
                    {
                        printf("cudaMemcpy failed 5th call!\n");
                        exit(EXIT_FAILURE);
                    }
                    finishPartial = omp_get_wtime();
                    elapsedMemCopy += finishPartial - startPartial;
                    elapsed4thMemCopy += finishPartial - startPartial;

                    fprintf(fpFrames, "%lf\n", time[timeStepCounter]);
                    for (i = 0; i < N; i++)
                    {
                        for (jj = 0; jj < N; jj++)
                        {
                            index = i * N + jj;
                            fprintf(fpFrames, "%lf ", V[index]);
                        }
                        fprintf(fpFrames, "\n");
                    }
                }
                finishPartial = omp_get_wtime();
                elapsedWriting += finishPartial - startPartial;


                // Check S1 velocity
                if (S1VelocityTag)
                {
                    // Copy memory from device to host of the matrices (2D arrays)
                    startPartial = omp_get_wtime();
                    cudaStatus1 = cudaMemcpy(V, d_V, N * N * sizeof(real), cudaMemcpyDeviceToHost);
                    if (cudaStatus1 != cudaSuccess)
                    {
                        printf("cudaMemcpy failed 5th call!\n");
                        exit(EXIT_FAILURE);
                    }
                    finishPartial = omp_get_wtime();
                    elapsedMemCopy += finishPartial - startPartial;
                    elapsed4thMemCopy += finishPartial - startPartial;

                    if (V[N - 1] >= 10)
                    {
                        S1Velocity = ((10 * (L - stim1xLimit)) / (time[timeStepCounter]));
                        S1VelocityTag = false;
                    }
                }
            }
            
            // Update time step counter
            timeStepCounter++;
        }

        // PDE execution
        elapsedPDE = elapsed1stThomas + elapsed2ndThomas + elapsedTranspose + elapsed1stRHS + elapsed2ndRHS;
        
        // Finish measuring total execution time
        finishTotal = omp_get_wtime();
        elapsedTotal = finishTotal - startTotal;
    }

    // Write infos to file
    fprintf(fpInfos, "S1 velocity: %lf m/s\n", S1Velocity);
    fprintf(fpInfos, "S2 begin: %.1lf ms\n", stim2Begin);
    fprintf(fpInfos, "PDE/ODE ratio: %d\n", PdeOdeRatio);
    fprintf(fpInfos, "ODE execution time: %lf seconds\n", elapsedODE);
    fprintf(fpInfos, "PDE execution time: %lf seconds\n", elapsedPDE);
    fprintf(fpInfos, "Writing time: %lf seconds\n", elapsedWriting);
    fprintf(fpInfos, "Total execution time with writings: %lf seconds\n", elapsedTotal);
    
    fprintf(fpInfos, "\nFor ODE and Transpose -> Grid size %d, Block size %d\n", GRID_SIZE, BLOCK_SIZE);
    fprintf(fpInfos, "Total threads: %d\n", GRID_SIZE*BLOCK_SIZE);

    fprintf(fpInfos, "\nFor PDE -> Grid size: %d, Block size: %d\n", numBlocks, blockSize);
    fprintf(fpInfos, "Total threads: %d\n", numBlocks*blockSize);
    fprintf(fpInfos, "1st Thomas algorithm time: %lf seconds\n", elapsed1stThomas);
    fprintf(fpInfos, "2nd Thomas algorithm time: %lf seconds\n", elapsed2ndThomas);
    fprintf(fpInfos, "Transpose time: %lf seconds\n", elapsedTranspose);
    fprintf(fpInfos, "1st RHS preparation time: %lf seconds\n", elapsed1stRHS);
    fprintf(fpInfos, "2nd RHS preparation time: %lf seconds\n", elapsed2ndRHS);
    fprintf(fpInfos, "Memory copy time (device to device): %lf seconds\n", elapsed2ndMemCopy);
    fprintf(fpInfos, "Memory copy time for velocity: %lf seconds\n", elapsed4thMemCopy);
    fprintf(fpInfos, "Total memory copy time: %lf seconds\n", elapsedMemCopy);
    
    fprintf(fpInfos, "\ntheta = %lf\n", theta);

    fprintf(fpInfos, "\nElapsed update state variables = %lf\n", elapsedTranspose2);

    if (haveFibrosis)
    {
        fprintf(fpInfos, "Fibrosis factor: %.2lf\n", fibrosisFactor);
        fprintf(fpInfos, "Fibrosis region: (%.2lf, %.2lf) to (%.2lf, %.2lf)\n", fibrosisMinX, fibrosisMinY, fibrosisMaxX, fibrosisMaxY);
    }

    // if (saveDataToError == true)
    // {
    //     char lastFrameFileName[MAX_STRING_SIZE];
    //     sprintf(lastFrameFileName, "last-%d-%.3lf.txt", numberThreads, deltat);
    //     FILE *fpLast;
    //     sprintf(aux, "%s/%s", pathToSaveData, lastFrameFileName);
    //     fpLast = fopen(aux, "w");
    //     for (int i = 0; i < N; i++)
    //     {
    //         for (int j = 0; j < N; j++)
    //         {
    //             index = i * N + j;
    //             fprintf(fpLast, "%lf ", V[index]);
    //         }
    //         fprintf(fpLast, "\n");
    //     }
    //     fclose(fpLast);
    // }

    // Close files
    fclose(fpFrames);
    fclose(fpInfos);

    // Free memory
    free(time);

    // Free memory from host
    free(V);
    free(Rv);
    free(rightside);
    free(solution);
    free(la);
    free(lb);
    free(lc);
    if (strcmp(method, "theta-ADI") == 0)
    {
        free(la2);
        free(lb2);
        free(lc2);
    }
    
    // Free memory from device
    cudaFree(d_V);
    cudaFree(d_X_r1);
    cudaFree(d_X_r2);
    cudaFree(d_X_s);
    cudaFree(d_m);
    cudaFree(d_h);
    cudaFree(d_j);
    cudaFree(d_d);
    cudaFree(d_f);
    cudaFree(d_f2);
    cudaFree(d_fCass);
    cudaFree(d_s);
    cudaFree(d_r);
    cudaFree(d_Ca_i);
    cudaFree(d_Ca_SR);
    cudaFree(d_Ca_SS);
    cudaFree(d_R_prime);
    cudaFree(d_Na_i);
    cudaFree(d_K_i);
    cudaFree(d_Rv);
    cudaFree(d_rightside);
    cudaFree(d_solution);
    cudaFree(d_la);
    cudaFree(d_lb);
    cudaFree(d_lc);
    if (strcmp(method, "theta-ADI") == 0)
    {
        cudaFree(d_la2);
        cudaFree(d_lb2);
        cudaFree(d_lc2);
    }

    cudaFree(d_V_2);
    cudaFree(d_X_r1_2);
    cudaFree(d_X_r2_2);
    cudaFree(d_X_s_2);
    cudaFree(d_m_2);
    cudaFree(d_h_2);
    cudaFree(d_j_2);
    cudaFree(d_d_2);
    cudaFree(d_f_2);
    cudaFree(d_f2_2);
    cudaFree(d_fCass_2);
    cudaFree(d_s_2);
    cudaFree(d_r_2);
    cudaFree(d_Ca_i_2);
    cudaFree(d_Ca_SR_2);
    cudaFree(d_Ca_SS_2);
    cudaFree(d_R_prime_2);
    cudaFree(d_Na_i_2);
    cudaFree(d_K_i_2);
    
    // Destroy streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}
#endif // TT2

#endif // METHODS_H