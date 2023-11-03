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
real startODE = 0.0, finishODE = 0.0, elapsedODE = 0.0;
real startPDE = 0.0, finishPDE = 0.0, elapsedPDE = 0.0;
real startMemCopy = 0.0, finishMemCopy = 0.0, elapsedMemCopy = 0.0;
real elapsed1stMemCopy = 0.0;
real elapsed2ndMemCopy = 0.0;
real elapsed3rdMemCopy = 0.0;
real elapsed4thMemCopy = 0.0;
real start1stThomas = 0.0, finish1stThomas = 0.0, elapsed1stThomas = 0.0;
real start2ndThomas = 0.0, finish2ndThomas = 0.0, elapsed2ndThomas = 0.0;
real startTranspose = 0.0, finishTranspose = 0.0, elapsedTranspose = 0.0;
real startWriting = 0.0, finishWriting = 0.0, elapsedWriting = 0.0;
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
void runMethodCPU(bool options[], char *method, real deltat, int numberThreads, real delta_x)
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
    real actualV, actualW;
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
    sprintf(framesFileName, "frames-%d-%.3lf-%.3lf.txt", numberThreads, deltat, deltat);
    sprintf(infosFileName, "infos-%d-%.3lf-%.3lf.txt", numberThreads, deltat, deltat);
    int saverate = ceil(M / 100.0);
    FILE *fpFrames, *fpInfos;

    // Create directories and files
    char pathToSaveData[MAX_STRING_SIZE];
    if (haveFibrosis)
    {
        createDirectories(pathToSaveData, method, "AFHN-Fibro", "CPU");
    }
    else
    {
        createDirectories(pathToSaveData, method, "AFHN", "CPU");
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
        Rv, rightside, solution, startODE, finishODE, elapsedODE, startPDE, finishPDE, elapsedPDE, \
        startWriting, finishWriting, elapsedWriting, start1stThomas, finish1stThomas, elapsed1stThomas, start2ndThomas, finish2ndThomas, elapsed2ndThomas)
        {
            while (timeStepCounter < M)
            {
                // Get time step
                timeStep = time[timeStepCounter];

                // Start measuring ODE execution time
                #pragma omp master
                {
                    startODE = omp_get_wtime();
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
                

                // Finish measuring ODE execution time and start measuring PDE execution time
                #pragma omp master
                {
                    finishODE = omp_get_wtime();
                    elapsedODE += finishODE - startODE;
                    startPDE = omp_get_wtime();
                    start1stThomas = omp_get_wtime();
                }

                // Resolve PDEs (Diffusion)
                // 1st: Implicit y-axis diffusion (lines)
                #pragma omp barrier
                #pragma omp for nowait
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
                    finish1stThomas = omp_get_wtime();
                    elapsed1stThomas += finish1stThomas - start1stThomas;
                    start2ndThomas = omp_get_wtime();
                }

                // 2nd: Implicit x-axis diffusion (columns)
                #pragma omp barrier
                #pragma omp for nowait
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
                    finishPDE = omp_get_wtime();
                    elapsedPDE += finishPDE - startPDE;
                    finish2ndThomas = omp_get_wtime();
                    elapsed2ndThomas += finish2ndThomas - start2ndThomas;
                }

                // Save frames
                #pragma omp master
                {
                    startWriting = omp_get_wtime();
                    if (VWTag == false)
                    {
                        // Write frames to file
                        /*
                        if (timeStepCounter % saverate == 0 && saveDataToGif == true)
                        {
                            fprintf(fpFrames, "%lf\n", time[timeStepCounter]);
                            for (i = 0; i < N; i++)
                            {
                                for (j = 0; j < N; j++)
                                {
                                    fprintf(fpFrames, "%lf ", V[i][j]);
                                }
                                fprintf(fpFrames, "\n");
                            }
                        }
                        */

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
                    finishWriting = omp_get_wtime();
                    elapsedWriting += finishWriting - startWriting;
                }

                // Update time step counter
                #pragma omp master
                {
                    timeStepCounter++;
                }
                #pragma omp barrier
            }
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

    fprintf(fpInfos, "\n1st Thomas algorithm (+ transpose) execution time: %lf seconds\n", elapsed1stThomas);
    fprintf(fpInfos, "2nd Thomas algorithm execution time: %lf seconds\n", elapsed2ndThomas);

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
                fprintf(fpLast, "%lf ", V[i][j]);
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

void runMethodGPU(bool options[], char *method, real deltat, int numberThreads, real delta_x)
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
        createDirectories(pathToSaveData, method, "AFHN-Fibro", "GPU");
    }
    else
    {
        createDirectories(pathToSaveData, method, "AFHN", "GPU");
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
    cudaError_t cudaStatus1, cudaStatus2, cudaStatus3, cudaStatus4, cudaStatus5, cudaStatus6;

    cudaStatus1 = cudaMalloc(&d_V, N * N * sizeof(real));
    cudaStatus2 = cudaMalloc(&d_Vtilde, N * N * sizeof(real));
    cudaStatus3 = cudaMalloc(&d_rightside, N * N * sizeof(real));
    cudaStatus4 = cudaMalloc(&d_solution, N * N * sizeof(real));
    cudaStatus5 = cudaMalloc(&d_la, N * sizeof(real));
    cudaStatus6 = cudaMalloc(&d_lb, N * sizeof(real));
    cudaStatus6 = cudaMalloc(&d_lc, N * sizeof(real));
    if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess || cudaStatus3 != cudaSuccess || cudaStatus4 != cudaSuccess || cudaStatus5 != cudaSuccess || cudaStatus6 != cudaSuccess)
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
            rightside, startODE, finishODE, elapsedODE, startPDE, finishPDE, elapsedPDE)
            {
                // Start measuring ODE execution time
                #pragma omp master
                {
                    startODE = omp_get_wtime();
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

                // Finish measuring ODE execution time and start measuring PDE execution time
                #pragma omp master
                {
                    finishODE = omp_get_wtime();
                    elapsedODE += finishODE - startODE;
                    startPDE = omp_get_wtime();
                }
                #pragma omp barrier
            }
            
            // Resolve PDEs (Diffusion)
            // 1st: Implicit y-axis diffusion (lines)                
            // Copy memory from host to device of the matrices (2D arrays)
            startMemCopy = omp_get_wtime();
            cudaStatus1 = cudaMemcpy(d_rightside, rightside, N * N * sizeof(real), cudaMemcpyHostToDevice);
            if (cudaStatus1 != cudaSuccess)
            {
                printf("cudaMemcpy failed 2nd call!\n");
                exit(EXIT_FAILURE);
            }
            finishMemCopy = omp_get_wtime();
            elapsedMemCopy += finishMemCopy - startMemCopy;
            elapsed1stMemCopy += finishMemCopy - startMemCopy;
                        
            // Call the kernel
            start1stThomas = omp_get_wtime();
            parallelKernel1<<<numBlocks, blockSize>>>(d_rightside, N, d_la, d_lb, d_lc);
            // if (cusparseDgtsv2_nopivot(handle, N, N, d_la, d_lb, d_lc, d_rightside, N, p_buffer))
            // {
            //     printf("Solving TDS1 failed!\n");
            // }
            cudaDeviceSynchronize();
            finish1stThomas = omp_get_wtime();
            elapsed1stThomas += finish1stThomas - start1stThomas;

            // Call the transpose kernel
            startTranspose = omp_get_wtime();
            transposeDiagonalCol<<<grid, block>>>(d_rightside, d_solution, N, N);
            cudaDeviceSynchronize();
            finishTranspose = omp_get_wtime();
            elapsedTranspose += finishTranspose - startTranspose;

            // 2nd: Implicit x-axis diffusion (columns)                
            // Call the kernel
            start2ndThomas = omp_get_wtime();
            parallelKernel1<<<numBlocks, blockSize>>>(d_solution, N, d_la, d_lb, d_lc);
            // if (cusparseDgtsv2_nopivot(handle, N, N, d_la, d_lb, d_lc, d_solution, N, p_buffer))
            // {
            //     printf("Solving TDS2 failed!\n");
            // }
            cudaDeviceSynchronize();
            finish2ndThomas = omp_get_wtime();
            elapsed2ndThomas += finish2ndThomas - start2ndThomas;

            // Copy memory from device to host of the matrices (2D arrays)
            startMemCopy = omp_get_wtime();
            cudaStatus1 = cudaMemcpy(V, d_solution, N * N * sizeof(real), cudaMemcpyDeviceToHost);
            if (cudaStatus1 != cudaSuccess)
            {
                printf("cudaMemcpy failed 5th call!\n");
                exit(EXIT_FAILURE);
            }
            finishMemCopy = omp_get_wtime();
            elapsedMemCopy += finishMemCopy - startMemCopy;
            elapsed4thMemCopy += finishMemCopy - startMemCopy;
            
            // Finish measuring PDE execution time
            finishPDE = omp_get_wtime();
            elapsedPDE += finishPDE - startPDE;
            
            // Save frames
            startWriting = omp_get_wtime();
            if (VWTag == false)
            {
                // Write frames to file
                /*
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
                */

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
            finishWriting = omp_get_wtime();
            elapsedWriting += finishWriting - startWriting;
            

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
    
    fprintf(fpInfos, "\nNumber of blocks: %d\n", numBlocks);
    fprintf(fpInfos, "Block size: %d\n", blockSize);
    fprintf(fpInfos, "1st Thomas algorithm time: %lf seconds\n", elapsed1stThomas);
    fprintf(fpInfos, "2nd Thomas algorithm time: %lf seconds\n", elapsed2ndThomas);
    fprintf(fpInfos, "Transpose time: %lf seconds\n", elapsedTranspose);
    fprintf(fpInfos, "1st memory copy time: %lf seconds\n", elapsed1stMemCopy);
    fprintf(fpInfos, "2nd memory copy time: %lf seconds\n", elapsed2ndMemCopy);
    fprintf(fpInfos, "3rd memory copy time: %lf seconds\n", elapsed3rdMemCopy);
    fprintf(fpInfos, "4th memory copy time: %lf seconds\n", elapsed4thMemCopy);
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

#endif // AFHN

#endif // METHODS_H