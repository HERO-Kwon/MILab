#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include "dft.h"

using namespace std;

// Custom thread exit function.
void thread_exit() {
    // Use pthread mutex and condition variables to exit the child function,
    // and signal the parent thread that this thread is done.
    /* Assignment */
}

// Custom thread join function.
void thread_join() {
    // Use pthread mutex and condition variables to wait until all children
    // threads complete their executions.
    /* Assignment */
}

// Perform 1-D DFT.
void dft1d(complex_t *h, const unsigned N) {
    // Use Cooley-Tuckey algorithm to perform row-wise 1-D DFT.
    // Step 1: The elements of row input, h, have to be first hashed in the
    //         bit-reversed order.
    // Step 2: Follow Danielson-Lanczos Lemma to perform the DFT.
    /* Assignment */
}

// 1-D DFT Thread function
void* dft_thread(void *arg) {
    // This thread function calls dft1d() function above for data rows.
    // Each thread performs 1-D DFT for height/num_threads number of rows.
    /* Assignment */

    return 0;
}

// Perform 2-D DFT.
void dft2d() {
    // Perform 2-D DFT with the data.
    // Step 1: Pre-calculate weight values to avoid the repeated calculations
    //         of the weight. Apparently, the pre-calculation is traded width
    //         extra memory usage.
    // Step 2: Initialize pthread-related variables, e.g., mutex and cond
    //         variables, as well as other necessary parameters.
    // Step 3: Create threads to run dft_thread() for row-wise DFTs.
    // Step 4: Call thread_join() to wait for all threads to complete.
    //         Do not use pthread_join().
    // Step 5: Transpose the data matrix so that column-wise DFT can be
    //         performed by the same row-wise DFT function.
    // Step 6: Create threads to run dft_thread() for row-wise DFTs.
    //         It is actually column-wise DFT after transpose.
    // Step 7: Call thread_join() to wait for all threads to complete.
    //         Do not use pthread_join().
    // Step 8: Transpose the data matrix back to the original orientation.
    // Step 9: Destroy pthread-related variables.
    // Step 10: Deallocated heap memory blocks if any.
    /* Assignment */
}

