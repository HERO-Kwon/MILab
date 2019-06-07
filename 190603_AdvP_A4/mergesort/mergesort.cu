#include <cstdlib>
#include <cstring>
#include <iostream>
#include "mergesort.h"

// Device kernel function
__global__ void vector_mul(int *a_dev, int *b_dev, int *c_dev){
    //Element-wise vector multiplication
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    c_dev[idx] = a_dev[idx] + b_dev[idx];
}
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    long start = width*idx*slices, 
         middle, 
         end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width / 2), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}

void mergesort(unsigned *m_data) {
    /* Assignment */

    using namespace std;   
    // get vector vector_size
    unsigned vector_size = num_data;


    // Host code
    int *a = 0, *b = 0, *c = 0;
    int *a_dev = 0, *b_dev = 0, *c_dev = 0;

    // Host data
    a = (int*)malloc(vector_size*sizeof(int));
    b = (int*)malloc(vector_size*sizeof(int));
    c = (int*)malloc(vector_size*sizeof(int));

    for(unsigned i = 0; i < vector_size; i++){
        a[i] = m_data[i];
        b[i] = m_data[i];
    }
    memset(c,0,vector_size*sizeof(int));

    // Device memory allocation
    cudaMalloc(&a_dev, vector_size * sizeof(int));
    cudaMalloc(&b_dev, vector_size * sizeof(int));
    cudaMalloc(&c_dev, vector_size * sizeof(int));

    // Memory copy from host to device memory
    cudaMemcpy(a_dev, a, vector_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b, vector_size * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel launch
    //vector_mul<<<vector_size/block_size,block_size>>>(a_dev,b_dev,c_dev);

    // Actually call the kernel
    //gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);

    for (int width = 2; width < num_data*2; width *= 2) {
        long slices = num_data / (block_size * width) + 1;

        // Actually call the kernel
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(a_dev, c_dev, num_data, width, slices);

        // memory copy from device to host memory
        cudaMemcpy(c,c_dev,vector_size * sizeof(int), cudaMemcpyDeviceToHost);

        for(unsigned i = 0; i < vector_size; i++){
            a[i] = c[i];
        }
        cudaMemcpy(a_dev, a, vector_size * sizeof(int), cudaMemcpyHostToDevice);
    }

    // memory copy from device to host memory
    cudaMemcpy(c,c_dev,vector_size * sizeof(int), cudaMemcpyDeviceToHost);

   
    
    // print vector add results
    cout << "a = [";
    for(unsigned i=0; i<10; i++) {cout << a[i] << " "; }
    cout << "\b]" << endl;
    cout << "*" << endl;
    cout << "b = [";
    for(unsigned i=0; i<10; i++) {cout << b[i] << " "; }
    cout << "\b]" << endl;
    cout << "=" << endl;
    cout << "c = [";
    for(unsigned i=0; i<10; i++) {cout << c[i] << " "; }
    cout << "\b]" << endl;
    

    // Host memory deallocation
    free(a);free(b);free(c);

    // Device memory deallocation
    cudaFree(a_dev); cudaFree(b_dev); cudaFree(c_dev);

}

