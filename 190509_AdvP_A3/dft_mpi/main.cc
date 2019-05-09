#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include "complex.h"
#include "dft.h"
#include "file_handler.h"

using namespace std;

file_handler_t file_handler;        // File handler
int num_ranks = 0;                  // Number of ranks
int rank_id = 0;                    // Rank ID
complex_t *data = 0;                // Data to perform DFT
unsigned width = 0, height = 0;     // Data dimension

int main(int argc, char **argv) {
    // Initialize MPI.
    MPI_Init(&argc, &argv);

    // Get the communicator size and rank information.
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);

    // Print usage message.
    if(argc != 1) {
        if(rank_id == 0) { cerr << "Usage: mpirun -np [num_ranks] " << argv[0] << endl; }
        MPI_Finalize(); return 0;
    }

    int val = num_ranks;
    // Max num_ranks is 1024.
    if(val > 1024) {
        if(rank_id == 0) { cerr << "Error: too many ranks" << endl; }
        MPI_Finalize(); return 0;
    }
    // Check if num_threads is power of two.
    while(!(val%2)) { val = val >> 1; }
    if(val != 1) {
        if(rank_id == 0) { cerr << "Error: num_threads must be a power of two." << endl; }
        MPI_Finalize(); return 0;
    }

    // Load input data file.
    file_handler.read_data("input.txt", data, width, height);

    // Perform 2-D DFT.
    dft2d();

    // Save the result to a file.
    if(rank_id == 0) { file_handler.save_data("output-2d.txt", data, width, height); }

    // Deallocate data.
    delete [] data;

    // Finalize MPI.
    MPI_Finalize();

    return 0;
}

