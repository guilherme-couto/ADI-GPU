#include "../include/includes.h"
#include "convergence-methods.h"

int main(int argc, char *argv[])
{
    // Parameters
    char *method;
    real delta_t;
    real delta_x;
    real theta;

    // Read parameters
    if (argc != 5)
    {
        printf("Usage: %s method delta_t delta_x theta\n", argv[0]);
        return 1;
    }
    method = argv[1];
    delta_t = atof(argv[2]);
    delta_x = atof(argv[3]);
    theta = atof(argv[4]);

    // Call function
    runSimulation(method, delta_t, delta_x, theta);

    return 0;

}