#include <cstdlib>
#include <stdio.h>
#include "complex.h"
#include "dft.h"
#include "file_handler.h"

using namespace std;

file_handler_t file_handler;        // File handler
unsigned num_threads = 0;           // Number of threads
complex_t *data = 0;                // Data to perform DFT
unsigned width = 0, height = 0;     // Data dimension

int main(int argc, char **argv) {
    // Print usage message.
    if(argc != 2) {
        cerr << "Usage: " << argv[0] << " [num_threads]" << endl;
        exit(1);
    }

    // Get number of threads.
    num_threads = unsigned(atoi(argv[1]));

    unsigned val = num_threads;
    // Max num_threads is 1024.
    if(val > 1024) {
        cerr << "Error: too many threads" << endl;
        exit(1);
    }
    // Check if num_threads is power of two.
    while(!(val%2)) { val = val >> 1; }
    if(val != 1) {
        cerr << "Error: num_threads must be a power of two." << endl;
        exit(1);
    }

    // Load input data file a file.
    file_handler.read_data("input.txt", data, width, height);

    // Perform 2-D DFT.
    dft2d();

    // Save the result to a file.
    file_handler.save_data("output-2d.txt", data, width, height);

    // Deallocate data.
    delete [] data;

    return 0;
}
