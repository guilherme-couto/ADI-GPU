#ifndef CONVERGENCE_METHODS_EXPLICIT_H
#define CONVERGENCE_METHODS_EXPLICIT_H

#define AFHN

// Define real type
#define MAX_STRING_SIZE 100

typedef double real;
#define REAL_TYPE "real"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>

// For time step
int timeStepCounter = 0;
real timeStep = 0.0;

real startTotal = 0.0;
real finishTotal = 0.0;
real elapsedTotal = 0.0;

real G = 1.5;         // omega^-1 * cm^-2
real sigma = 1.2e-3;  // omega^-1 * cm^-1

real chi = 1.0e3;     // cm^-1
real Cm = 1.0e-3;     // mF * cm^-2

real V_init = 0.0;    // Initial membrane potential -> mV

#if defined(AFHN)
void runSimulation(char *method, real delta_t, real delta_x, int number_of_threads)
{
    // Number of steps
    real L = 1.0;
    real T = 1.0;
    int N = round(L / delta_x) + 1;               // Spatial steps (square tissue)
    int M = round(T / delta_t) + 1;                // Number of time steps

    // Allocate and populate time array
    real *time;
    time = (real *)malloc(M * sizeof(real));
    for (int i = 0; i < M; i++)
        time[i] = i * delta_t;
    
    // Allocate and initialize variables
    real **V, **Vaux;
    V = (real **)malloc(N * sizeof(real *));
    Vaux = (real **)malloc(N * sizeof(real *));

    for (int i = 0; i < N; i++)
    {
        V[i] = (real *)malloc(N * sizeof(real));
        Vaux[i] = (real *)malloc(N * sizeof(real));
    }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            V[i][j] = V_init;
            Vaux[i][j] = V_init;
        }

    // Variables
    int i, j;

    /*--------------------
    --    theta ADI     --
    ----------------------*/
    printf("Spatial discretization N = %d!\n", N);
    printf("(2D) N * N = %d!\n", N*N);
    printf("Time discretization M = %d!\n", M);

    int numberThreads = number_of_threads;
    printf("Number of threads: %d\n", numberThreads);
    if (strcmp(method, "FE") == 0)
    {
        // Start measuring total execution time
        startTotal = omp_get_wtime();

        real actualV = 0.0;
        real idiff = 0.0;
        real jdiff = 0.0;
        real linearReaction = 0.0;
        real forcingTerm = 0.0;
        real aux_pi = 3.14159265358979323846;
        real x = 0.0;
        real y = 0.0;

        #pragma omp parallel num_threads(numberThreads) default(none) private(i, j, actualV, idiff, jdiff, linearReaction, forcingTerm, x, y) \
        shared(V, Vaux, N, M, L, delta_t, delta_x, time, timeStep, timeStepCounter, aux_pi, Cm, chi, sigma, G)
        {
            while (timeStepCounter < M)
            {
                // Get time step
                timeStep = time[timeStepCounter];

                // Resolve ODEs
                #pragma omp for collapse(2)
                for (i = 0; i < N; i++)
                {
                    for (j = 0; j < N; j++)
                    {
                        // Get actual V and W
                        actualV = V[i][j];

                        // Get diffusion terms
                        if (i == 0)
                            idiff = (2*V[i+1][j] - 2*actualV) / (delta_x * delta_x);
                        else if (i == N-1)
                            idiff = (2*V[i-1][j] - 2*actualV) / (delta_x * delta_x);
                        else
                            idiff = (V[i+1][j] - 2*actualV + V[i-1][j]) / (delta_x * delta_x);
                        
                        if (j == 0)
                            jdiff = (2*V[i][j+1] - 2*actualV) / (delta_x * delta_x);
                        else if (j == N-1)
                            jdiff = (2*V[i][j-1] - 2*actualV) / (delta_x * delta_x);
                        else
                            jdiff = (V[i][j+1] - 2*actualV + V[i][j-1]) / (delta_x * delta_x);

                        // Get linear reaction term
                        linearReaction = G*actualV;

                        // Get forcing term
                        x = j * delta_x;
                        y = i * delta_x;
                        forcingTerm = cos(aux_pi*x/L) * cos(aux_pi*y/L) * ((aux_pi*cos(aux_pi*timeStep)) + (((2*aux_pi*aux_pi*sigma)/(chi*Cm*L*L))*sin(aux_pi*timeStep)) + ((G/Cm)*sin(aux_pi*timeStep)));

                        // Update V
                        Vaux[i][j] = actualV + delta_t * (((sigma/(chi*Cm)) * (idiff + jdiff)) - (linearReaction/Cm) + forcingTerm); 
                    }
                }

                // Copy Vaux to V
                #pragma omp for collapse(2)
                for (i = 0; i < N; i++)
                    for (j = 0; j < N; j++)
                        V[i][j] = Vaux[i][j];

                // Update time step counter
                #pragma omp master
                {
                    timeStepCounter++;
                }
            }
        }

        // Finish measuring total execution time
        finishTotal = omp_get_wtime();
        elapsedTotal = finishTotal - startTotal;    
    }

    // Create directories
    char pathToSaveData[MAX_STRING_SIZE];
    char command[MAX_STRING_SIZE];
    sprintf(command, "%s", "mkdir -p");
    char aux[MAX_STRING_SIZE];
    char path[MAX_STRING_SIZE] = "./simulation-files/";
    strcat(path, REAL_TYPE);
    sprintf(aux, "%s %s", command, path);
    system(aux);
    sprintf(pathToSaveData, "%s/%s", path, "AFHN");
    sprintf(aux, "%s %s", command, pathToSaveData);
    system(aux);
    sprintf(pathToSaveData, "%s/%s", pathToSaveData, method);
    sprintf(aux, "%s %s", command, pathToSaveData);
    system(aux);

    // File names
    char infosFileName[MAX_STRING_SIZE];
    sprintf(infosFileName, "infos-%.8lf-%.6lf.txt", delta_t, delta_x);
    char lastFrameFileName[MAX_STRING_SIZE];
    sprintf(lastFrameFileName, "last-%.8lf-%.6lf.txt", delta_t, delta_x);

    // Infos file pointer
    FILE *fpInfos;
    sprintf(aux, "%s/%s", pathToSaveData, infosFileName);
    fpInfos = fopen(aux, "w");

    // Save infos
    fprintf(fpInfos, "Elapsed time: %e\n", elapsedTotal);
    fprintf(fpInfos, "Number of threads: %d\n", numberThreads);
    fprintf(fpInfos, "N: %d\n", N);
    fprintf(fpInfos, "M: %d\n", M);
    fprintf(fpInfos, "delta_t: %e\n", delta_t);
    fprintf(fpInfos, "delta_x: %e\n", delta_x);

    // Close files
    fclose(fpInfos);

    // Save last frame
    FILE *fpLast;
    sprintf(aux, "%s/%s", pathToSaveData, lastFrameFileName);
    fpLast = fopen(aux, "w");
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            fprintf(fpLast, "%e ", V[i][j]);
        }
        fprintf(fpLast, "\n");
    }
    fclose(fpLast);

    // Free memory
    free(time);

    // Free memory
    for (int i = 0; i < N; i++)
    {
        free(V[i]);
        free(Vaux[i]);
    }
    free(V);
    free(Vaux);

    return;
}

#endif // AFHN

#endif // CONVERGENCE_METHODS_EXPLICIT_H