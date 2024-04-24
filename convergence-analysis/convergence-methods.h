#ifndef CONVERGENCE_METHODS_H
#define CONVERGENCE_METHODS_H

#include "../include/includes.h"
#include "convergence-functions.h"

real elapsed1stPart = 0.0, elapsed2ndPart = 0.0;

#if defined(AFHN)
void runSimulation(char *method, real delta_t, real delta_x, real theta)
{
    // Number of steps
    L = 1.0;
    T = 1.0;
    int N = round(L / delta_x) + 1;               // Spatial steps (square tissue)
    int M = round(T / delta_t) + 1;                // Number of time steps

    // Allocate and populate time array
    real *time;
    time = (real *)malloc(M * sizeof(real));
    for (int i = 0; i < M; i++)
        time[i] = i * delta_t;
    
    // Allocate and initialize the state variable
    real *V;
    V = (real *)malloc(N * N * sizeof(real));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            V[i * N + j] = V_init;

    // Diffusion coefficient (isotropic)
    real D = sigma / (chi * Cm);
    real phi = D * delta_t / (delta_x * delta_x);       // For Thomas algorithm

    // Variables
    int i, j;

    // Auxiliary arrays for Thomas algorithm
    real *la = (real *)malloc(N * sizeof(real));
    real *lb = (real *)malloc(N * sizeof(real));
    real *lc = (real *)malloc(N * sizeof(real));

    // Populate auxiliary arrays for Thomas algorithm
    real theta_implicit = 0.0;
    real theta_explicit = 0.0;
    if (strcmp(method, "theta-ADI") == 0)
    {
        populateDiagonalThomasAlgorithm(la, lb, lc, N, theta*phi);
    }
    else if (strcmp(method, "SSI-ADI") == 0)
    {
        populateDiagonalThomasAlgorithm(la, lb, lc, N, 0.5*phi);
    } 

    // Prefactorization
    thomasFactorConstantBatch(la, lb, lc, N);

    // Discritized limits of fibrotic area
    int discFibxMax = -1;
    int discFibxMin = 0;
    int discFibyMax = -1;
    int discFibyMin = 0;
    
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
    if (strcmp(method, "theta-ADI") == 0)
    {
        sprintf(pathToSaveData, "%s/%.2lf", pathToSaveData, theta);
        sprintf(aux, "%s %s", command, pathToSaveData);
        system(aux);
    }

    // File names
    char infosFileName[MAX_STRING_SIZE];
    sprintf(infosFileName, "infos-%.8lf-%.6lf.txt", delta_t, delta_x);
    char lastFrameFileName[MAX_STRING_SIZE];
    sprintf(lastFrameFileName, "last-%.8lf-%.6lf.txt", delta_t, delta_x);

    // Infos file pointer
    FILE *fpInfos;
    sprintf(aux, "%s/%s", pathToSaveData, infosFileName);
    fpInfos = fopen(aux, "w");

    // Save data rate
    int saverate = ceil(M / 100.0);    

    // CUDA variables and allocation
    real *d_V, *d_RHS, *d_Rv;
    real *d_la, *d_lb, *d_lc;
    cudaError_t cudaStatus1, cudaStatus2, cudaStatus3, cudaStatus5, cudaStatus6, cudaStatus7, cudaStatus8;
    
    cudaStatus1 = cudaMalloc(&d_V, N * N * sizeof(real));
    cudaStatus2 = cudaMalloc(&d_RHS, N * N * sizeof(real));
    cudaStatus5 = cudaMalloc(&d_Rv, N * N * sizeof(real));
    cudaStatus6 = cudaMalloc(&d_la, N * sizeof(real));
    cudaStatus7 = cudaMalloc(&d_lb, N * sizeof(real));
    cudaStatus8 = cudaMalloc(&d_lc, N * sizeof(real));
    if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess || cudaStatus5 != cudaSuccess || cudaStatus6 != cudaSuccess || cudaStatus7 != cudaSuccess || cudaStatus8 != cudaSuccess)
    {
        printf("cudaMalloc failed!\n");
        exit(EXIT_FAILURE);
    }

    // Copy memory from host to device of the matrices (2D arrays)
    cudaStatus1 = cudaMemcpy(d_V, V, N * N * sizeof(real), cudaMemcpyHostToDevice);
    if (cudaStatus1 != cudaSuccess)
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
    printf("All cudaMallocs done!\n");

    // Block and grid size
    // For parallel Thomas
    printf("N = %d\n", N);
    int numBlocks = N / 100;
    if (numBlocks == 0)
        numBlocks = 1;
    int blockSize = round(N / numBlocks) + 1;
        
    if (blockSize % 32 != 0)
        blockSize = 32 * ((blockSize / 32) + 1);
    
    // For other kernels
    int GRID_SIZE = ceil((N*N*1.0) / (BLOCK_SIZE*1.0));
    if (GRID_SIZE == 0)
        GRID_SIZE = 1;

    printf("For 1st Part and Transpose -> Grid size %d, Block size %d\n", GRID_SIZE, BLOCK_SIZE);
    printf("Total for 1st Part and Transpose: %d\n", GRID_SIZE*BLOCK_SIZE);
    printf("For Thomas Algorithm -> Grid size: %d, Block size: %d\n", numBlocks, blockSize);
    printf("Total for Thomas Algorithm: %d\n", numBlocks*blockSize);

    /*--------------------
    --    theta ADI     --
    ----------------------*/
    printf("Spatial discretization N = %d!\n", N);
    printf("(2D) N * N = %d!\n", N*N);
    printf("Time discretization M = %d!\n", M);

    int index;
    if (strcmp(method, "theta-ADI") == 0)
    {
        // Start measuring total execution time
        startTotal = omp_get_wtime();

        while (timeStepCounter < M)
        {
            // Get time step
            timeStep = time[timeStepCounter];

            // Start measuring 1st part execution time
            startPartial = omp_get_wtime();

            // Solve the reaction and forcing term part
            parallelRHSForcing_theta<<<GRID_SIZE, BLOCK_SIZE>>>(d_V, d_Rv, N, timeStep, delta_t, delta_x, phi, theta, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
            cudaDeviceSynchronize();

            // Finish measuring 1st part execution time
            finishPartial = omp_get_wtime();
            elapsed1stPart += finishPartial - startPartial;

            // Prepare right side of Thomas algorithm with explicit diffusion on j
            // Call the kernel
            startPartial = omp_get_wtime();
            prepareRHS_theta<<<GRID_SIZE, BLOCK_SIZE>>>(d_V, d_RHS, d_Rv, N, phi, theta, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor); 
            cudaDeviceSynchronize();     
            finishPartial = omp_get_wtime();
            elapsed1stRHS += finishPartial - startPartial;

            // 1st: Implicit y-axis diffusion (lines)
            // Call the kernel
            startPartial = omp_get_wtime();
            cuThomasConstantBatch<<<numBlocks, blockSize>>>(d_la, d_lb, d_lc, d_RHS, N);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed1stThomas += finishPartial - startPartial;

            // Call the transpose kernel
            startPartial = omp_get_wtime();
            transposeDiagonalCol<<<GRID_SIZE, BLOCK_SIZE>>>(d_RHS, d_V, N, N);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsedTranspose += finishPartial - startPartial;

            // Prepare right side of Thomas algorithm with explicit diffusion on i
            // Call the kernel
            startPartial = omp_get_wtime();
            prepareRHS_theta<<<GRID_SIZE, BLOCK_SIZE>>>(d_V, d_RHS, d_Rv, N, phi, theta, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed2ndRHS += finishPartial - startPartial;

            // 2nd: Implicit x-axis diffusion (columns)                
            // Call the kernel
            startPartial = omp_get_wtime();
            cuThomasConstantBatch<<<numBlocks, blockSize>>>(d_la, d_lb, d_lc, d_RHS, N);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed2ndThomas += finishPartial - startPartial;

            // Call the transpose kernel --- TEST
            startPartial = omp_get_wtime();
            transposeDiagonalCol<<<GRID_SIZE, BLOCK_SIZE>>>(d_RHS, d_V, N, N);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsedTranspose += finishPartial - startPartial;

            // Copy d_RHS to d_V
            // startPartial = omp_get_wtime();
            // cudaStatus1 = cudaMemcpy(d_V, d_RHS, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
            // if (cudaStatus1 != cudaSuccess)
            // {
            //     printf("cudaMemcpy failed device to device!\n");
            //     exit(EXIT_FAILURE);
            // }
            // finishPartial = omp_get_wtime();
            // elapsedMemCopy += finishPartial - startPartial;
            // elapsed2ndMemCopy += finishPartial - startPartial;

            // Update time step counter
            timeStepCounter++;
        }

        // 2nd Part execution
        elapsed2ndPart = elapsed1stThomas + elapsed2ndThomas + elapsedTranspose + elapsed1stRHS + elapsed2ndRHS;
        
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

            // Start measuring 1st part execution time
            startPartial = omp_get_wtime();

            // Solve the reaction and forcing term part
            parallelRHSForcing_SSI<<<GRID_SIZE, BLOCK_SIZE>>>(d_V, d_Rv, N, timeStep, delta_t, delta_x, phi, theta, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
            cudaDeviceSynchronize();

            // Finish measuring 1st part execution time
            finishPartial = omp_get_wtime();
            elapsedODE += finishPartial - startPartial;

            // Prepare right side of Thomas algorithm with explicit diffusion on j
            // Call the kernel
            startPartial = omp_get_wtime();
            prepareRighthandSide_jDiffusion<<<GRID_SIZE, BLOCK_SIZE>>>(d_V, d_RHS, d_Rv, N, phi, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor); 
            cudaDeviceSynchronize();     
            finishPartial = omp_get_wtime();
            elapsed1stRHS += finishPartial - startPartial;

            // 1st: Implicit y-axis diffusion (lines)
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas<<<numBlocks, blockSize>>>(d_RHS, N, d_la, d_lb, d_lc);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed1stThomas += finishPartial - startPartial;

            // Call the transpose kernel
            startPartial = omp_get_wtime();
            transposeDiagonalCol<<<GRID_SIZE, BLOCK_SIZE>>>(d_RHS, d_V, N, N);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsedTranspose += finishPartial - startPartial;

            // Prepare right side of Thomas algorithm with explicit diffusion on i
            // Call the kernel
            startPartial = omp_get_wtime();
            prepareRighthandSide_iDiffusion<<<GRID_SIZE, BLOCK_SIZE>>>(d_V, d_RHS, d_Rv, N, phi, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed2ndRHS += finishPartial - startPartial;

            // 2nd: Implicit x-axis diffusion (columns)                
            // Call the kernel
            startPartial = omp_get_wtime();
            parallelThomas<<<numBlocks, blockSize>>>(d_RHS, N, d_la, d_lb, d_lc);
            cudaDeviceSynchronize();
            finishPartial = omp_get_wtime();
            elapsed2ndThomas += finishPartial - startPartial;

            // Copy d_RHS to d_V
            startPartial = omp_get_wtime();
            cudaStatus1 = cudaMemcpy(d_V, d_RHS, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
            if (cudaStatus1 != cudaSuccess)
            {
                printf("cudaMemcpy failed device to device!\n");
                exit(EXIT_FAILURE);
            }
            finishPartial = omp_get_wtime();
            elapsedMemCopy += finishPartial - startPartial;
            elapsed2ndMemCopy += finishPartial - startPartial;

            // Update time step counter
            timeStepCounter++;
        }

        // 2nd Part execution
        elapsed2ndPart = elapsed1stThomas + elapsed2ndThomas + elapsedTranspose + elapsed1stRHS + elapsed2ndRHS;
        
        // Finish measuring total execution time
        finishTotal = omp_get_wtime();
        elapsedTotal = finishTotal - startTotal;  
    }
    
    else if (strcmp(method, "FE") == 0)
    {
        // Start measuring total execution time
        startTotal = omp_get_wtime();

        while (timeStepCounter < M)
        {
            // Get time step
            timeStep = time[timeStepCounter];

            // Start measuring 1st part execution time
            startPartial = omp_get_wtime();

            // Solve the reaction and forcing term part
            parallelFE<<<GRID_SIZE, BLOCK_SIZE>>>(d_V, d_Rv, N, timeStep, delta_t, delta_x, phi, theta, discFibxMax, discFibxMin, discFibyMax, discFibyMin, fibrosisFactor);
            cudaDeviceSynchronize();

            // Finish measuring 1st part execution time
            finishPartial = omp_get_wtime();
            elapsedODE += finishPartial - startPartial;

            // Copy d_Rv to d_V
            startPartial = omp_get_wtime();
            cudaStatus1 = cudaMemcpy(d_V, d_Rv, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
            if (cudaStatus1 != cudaSuccess)
            {
                printf("cudaMemcpy failed device to device!\n");
                exit(EXIT_FAILURE);
            }
            finishPartial = omp_get_wtime();
            elapsedMemCopy += finishPartial - startPartial;
            elapsed2ndMemCopy += finishPartial - startPartial;

            // Update time step counter
            timeStepCounter++;
        }

        // 2nd Part execution
        elapsed2ndPart = elapsed1stThomas + elapsed2ndThomas + elapsedTranspose + elapsed1stRHS + elapsed2ndRHS;
        
        // Finish measuring total execution time
        finishTotal = omp_get_wtime();
        elapsedTotal = finishTotal - startTotal; 
    }

    //Copy memory from device to host of the matrices (2D arrays)
    startPartial = omp_get_wtime();
    cudaStatus1 = cudaMemcpy(V, d_V, N * N * sizeof(real), cudaMemcpyDeviceToHost);
    if (cudaStatus1 != cudaSuccess)
    {
        printf("cudaMemcpy failed!\n");
        exit(EXIT_FAILURE);
    }
    finishPartial = omp_get_wtime();
    elapsedMemCopy += finishPartial - startPartial;
    elapsed4thMemCopy += finishPartial - startPartial;

    // Save last frame
    FILE *fpLast;
    sprintf(aux, "%s/%s", pathToSaveData, lastFrameFileName);
    fpLast = fopen(aux, "w");
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            index = i * N + j;
            fprintf(fpLast, "%e ", V[index]);
        }
        fprintf(fpLast, "\n");
    }
    fclose(fpLast);

    // Write infos to file
    fprintf(fpInfos, "1st Part execution time: %lf seconds\n", elapsed1stPart);
    fprintf(fpInfos, "2nd Part execution time: %lf seconds\n", elapsed2ndPart);
    fprintf(fpInfos, "Writing time: %lf seconds\n", elapsedWriting);
    fprintf(fpInfos, "Total execution time with writings: %lf seconds\n", elapsedTotal);
    ////////////////////////////////////////////////////
    fprintf(fpInfos, "\nFor 1st Part and Transpose -> Grid size %d, Block size %d\n", GRID_SIZE, BLOCK_SIZE);
    fprintf(fpInfos, "Total threads: %d\n", GRID_SIZE*BLOCK_SIZE);
    ////////////////////////////////////////////////////
    fprintf(fpInfos, "\nFor 2nd Part -> Grid size: %d, Block size: %d\n", numBlocks, blockSize);
    fprintf(fpInfos, "Total threads: %d\n", numBlocks*blockSize);
    fprintf(fpInfos, "1st Thomas algorithm time: %lf seconds\n", elapsed1stThomas);
    fprintf(fpInfos, "2nd Thomas algorithm time: %lf seconds\n", elapsed2ndThomas);
    fprintf(fpInfos, "Transpose time: %lf seconds\n", elapsedTranspose);
    fprintf(fpInfos, "1st RHS preparation time: %lf seconds\n", elapsed1stRHS);
    fprintf(fpInfos, "2nd RHS preparation time: %lf seconds\n", elapsed2ndRHS);
    fprintf(fpInfos, "Memory copy time (device to device): %lf seconds\n", elapsed2ndMemCopy);
    fprintf(fpInfos, "Memory copy time for velocity: %lf seconds\n", elapsed4thMemCopy);
    fprintf(fpInfos, "Total memory copy time: %lf seconds\n", elapsedMemCopy);
    ////////////////////////////////////////////////////
    fprintf(fpInfos, "\ntheta = %lf\n", theta);
    fprintf(fpInfos, "L = %lf, T = %lf, N = %d, N*N = %d, M = %d\n", L, T, N, N*N, M);

    // Close files
    fclose(fpInfos);

    // Free memory
    free(time);

    // Free memory from host
    free(V);
    free(la);
    free(lb);
    free(lc);

    // Free memory from device
    cudaFree(d_V);
    cudaFree(d_Rv);
    cudaFree(d_RHS);
    cudaFree(d_la);
    cudaFree(d_lb);
    cudaFree(d_lc);
}

#endif // AFHN

#endif // CONVERGENCE_METHODS_H