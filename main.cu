#include "./include/includes.h"

void resetSimulationParameters()
{
    // For stimulation
    Istim = 0.0;

    // For time step
    timeStepCounter = 0;
    timeStep = 0.0;

    // For execution time
    startTotal = 0.0;
    finishTotal = 0.0;
    elapsedTotal = 0.0;
    startPartial = 0.0;
    finishPartial = 0.0;
    elapsedODE = 0.0;
    elapsedPDE = 0.0;
    elapsedMemCopy = 0.0;

    // For velocity
    S1VelocityTag = true;
    S1Velocity = 0.0;
}

/*-------------
Main function
--------------*/
int main(int argc, char *argv[])
{
    // Read parameters from command line
    if (argc != 9)
    {
        printf("Usage: %s <num_threads> <delta_t (ms)> <delta_x (cm)> <mode (CPU or GPU)> <method> <fibrosis (0-false or 1-true)> <vulnerability window (0-false or 1-true)> <only speed (0-false or 1-true)>\n", argv[0]);
        exit(1);
    }
    
    // Get values from command line
    int numberThreads = atoi(argv[1]);
    real deltat = atof(argv[2]);
    real delta_x = atof(argv[3]);
    char *mode = argv[4];
    char *method = argv[5];
    bool fibrosis = atoi(argv[6]);
    bool VWmeasure = atoi(argv[7]);
    bool onlySpeed = atoi(argv[8]);

    // Call method
    if (!fibrosis)
    {
        fibrosisFactor = 1.0;
    }

    bool options[] = {fibrosis, true, true, true, true, true};

    if (onlySpeed == false)
    {
        if (strcmp(mode, "CPU") == 0)
        {
            runAllinCPU(options, method, deltat, numberThreads, delta_x, mode);
        }
        else if (strcmp(mode, "GPU") == 0)
        {
            runODEinCPUandPDEinGPU(options, method, deltat, numberThreads, delta_x, mode);
        } 
        else if (strcmp(mode, "All-GPU") == 0)
        {
            runAllinGPU(options, method, deltat, numberThreads, delta_x, mode);
        }
        else if (strcmp(mode, "All-GPU-3D") == 0)
        {
            runAllinGPU3D(options, method, deltat, numberThreads, delta_x, mode);
        }
        resetSimulationParameters();
    }
    

    // Measure Vulnerable Window
    if (VWmeasure == true)
    {
        bool options[] = {fibrosis, true, false, false, true, true};

        // Update VWTag
        if (VWTag == false)
        {
            VWTag = true;
            stim2Begin = measureVWFrom;
            resetSimulationParameters();
            if (strcmp(mode, "CPU") == 0)
            {
                runAllinCPU(options, method, deltat, numberThreads, delta_x, mode);
            }
            else if (strcmp(mode, "GPU") == 0)
            {
                runODEinCPUandPDEinGPU(options, method, deltat, numberThreads, delta_x, mode);
            }
            else if (strcmp(mode, "All-GPU") == 0)
            {
                runAllinGPU(options, method, deltat, numberThreads, delta_x, mode);
            }
        }
        // Update S2 begin
        while (stim2Begin + 1.0 <= measureVWTo)
        {
            stim2Begin += 1.0;
            resetSimulationParameters();
            if (strcmp(mode, "CPU") == 0)
            {
                runAllinCPU(options, method, deltat, numberThreads, delta_x, mode);
            }
            else if (strcmp(mode, "GPU") == 0)
            {
                runODEinCPUandPDEinGPU(options, method, deltat, numberThreads, delta_x, mode);
            }
            else if (strcmp(mode, "All-GPU") == 0)
            {
                runAllinGPU(options, method, deltat, numberThreads, delta_x, mode);
            }
        }
    }

    return 0;
}

// bool haveFibrosis = options[0];
// bool measureTotalTime = options[1];
// bool saveDataToError = options[2];
// bool saveDataToGif = options[3];
// bool measureTimeParts = options[4];
// bool measureS1Velocity = options[5];
