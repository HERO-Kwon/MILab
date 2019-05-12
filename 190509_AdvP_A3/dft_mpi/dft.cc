#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include "dft.h"

using namespace std;

// Custom MPI_Allgather function with complex_t data type
void mpi_allgather(complex_t *m_send_buffer, int m_send_count,
                   complex_t *m_recv_buffer, int m_recv_count,
                   MPI_Comm m_comm) {
    // Use a group of non-blocking receive (i.e., MPI_Irecv) and blocking
    // send (i.e., MPI_Send) along with MPI_Wait() to implement the custom
    // MPI_Allgather function. Do not use collective communication functions
    // such as MPI_Allgather(), MPI_Gather, MPI_Scatter, or MPI_Broadcast.
    /* Assignment */

    // byte 단위로 보내고 계산해야 함
    
}

// Transpose a matrix.
void transpose_matrix(complex_t *h, const unsigned m_width, const unsigned m_height) {
    // Transpose the matrix h that has the dimension of m_width * m_height.
    
    // make copy of old array
    complex_t *old_data1 = h;
    // make new data array
    h = new complex_t[width * height];
    
    // Copy elements from the transposed value from old array
    for(unsigned i=0; i<height; i++)
    {
        for(unsigned j=0; j<width; j++)
        {
            //transpose
            new (&h[i*height+j]) complex_t(old_data1[i+j*width]);
            old_data1[i+j*width].~complex_t();
        }
    }
    // Deallocate the old array.
    delete[] old_data1;
}

// Perform 1-D DFT.
void dft1d(complex_t *h, const unsigned N) {
    // This function implements 1-D DFT for the array h of length N.
    // You may reuse the Cooley-Tuckey algorithm that you wrote for the
    // previous pthread assignment, or you may directly calculate the DFT
    // equation shown in the lecture note.
    /* Assignment */
}

// Perform 2-D DFT.
void dft2d() {
    // This function performs 2-D DFT with the data.
    // Step 1: Calculate 1-D DFT for the given rows per process.
    // Step 2: Use the custom mpi_allgather() as a barrier to gather the
    //         1-D DFT result.
    // Step 3: Transpose the matrix.
    // Step 4: Calculate 1-D DFT again, which is technically column-wise DFT.
    // Step 5: Use again the custom mpi_allgather() to gather the 2-D DFT
    //         results calculated in the previous step.
    // Step 6: Transpose the matrix back to the original orientation.
    /* Assignment */
}

