#include "convergence-methods-explicit.h"

int main(int argc, char *argv[])
{
    // Parameters
    char *method;
    real delta_t;
    real delta_x;
    int number_of_threads;

    // Read parameters
    if (argc != 5)
    {
        printf("Usage: %s method delta_t delta_x number_of_threads\n", argv[0]);
        return 1;
    }
    method = argv[1];
    delta_t = atof(argv[2]);
    delta_x = atof(argv[3]);
    number_of_threads = atoi(argv[4]);

    // Call function
    runSimulation(method, delta_t, delta_x, number_of_threads);

    return 0;
}