#include <cstdlib>
#include <cstring>
#include <iostream>
#include "mergesort.h"

// Device kernel function
__device__ void Merge(int* source, int* dest, int start, int middle, int end) {
    int i = start; 
    int j = middle;
    // iterate through points of target vector
    for (int k = start; k < end; k++) {
        // if left part of target vector is smaller than right
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        // if right part of target vector is larger than left
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}
// setting values for target vector
__global__ void SetMerge(int* source, int* dest, int size, int width, int slices) {
    // make 1d index
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = width*idx*slices;
    int middle; 
    int end;

    // iterate through all slices of target vectors
    for (int slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width / 2), size);
        end = min(start + width, size);
        // call merge function
        Merge(source, dest, start, middle, end);
        // set starting points for next target vector
        start += width;
    }
}

void mergesort(unsigned *m_data) {
    /* Assignment */

    using namespace std;   

    // set block size
    int blocksPerGrid = 8;

    // get vector vector_size
    unsigned vector_size = num_data;

    // Host code
    int *source_vec = 0, *dest_vec = 0;
    int *source_dev = 0, *dest_dev = 0;

    // Host data
    source_vec = (int*)malloc(vector_size*sizeof(int));
    dest_vec = (int*)malloc(vector_size*sizeof(int));

    for(unsigned i = 0; i < vector_size; i++){
        source_vec[i] = m_data[i];
    }
    memset(dest_vec,0,vector_size*sizeof(int));

    // Device memory allocation
    cudaMalloc(&source_dev, vector_size * sizeof(int));
    cudaMalloc(&dest_dev, vector_size * sizeof(int));

    // Memory copy from host to device memory
    cudaMemcpy(source_dev, source_vec, vector_size * sizeof(int), cudaMemcpyHostToDevice);

    // make mergesort
    for (int width = 2; width < num_data*2; width *= 2) {
        int slices = num_data / (blocksPerGrid*block_size * width)+1;

        // call the kernel
        SetMerge<<<blocksPerGrid, block_size>>>(source_dev, dest_dev, num_data, width, slices);

        // swap values of dest vector to source vector
        cudaMemcpy(dest_vec,dest_dev,vector_size * sizeof(int), cudaMemcpyDeviceToHost);

        // copy dest values to source
        for(unsigned i = 0; i < vector_size; i++){
            source_vec[i] = dest_vec[i];
        }
        // copy to device memory
        cudaMemcpy(source_dev, source_vec, vector_size * sizeof(int), cudaMemcpyHostToDevice);
    }

    for(unsigned i=0; i< vector_size; i++){
        data[i] = dest_vec[i];
    }
    
    // Host memory deallocation
    free(source_vec);free(dest_vec);

    // Device memory deallocation
    cudaFree(source_dev); cudaFree(dest_dev);

}

