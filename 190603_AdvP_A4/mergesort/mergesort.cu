#include "mergesort.h"

using namespace std;

// Device kernel function
__global__ void vector_mul(int *a_dev, int *b_dev, int *c_dev){
    //Element-wise vector multiplication
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    c_dev[idx] = a_dev[idx] * b_dev[idx];
}

void mergesort(unsigned *m_data) {
    /* Assignment */

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
    cudaMemcpy(a_dev, a, vector_size * sizeof(int), cudaMemcpyHostToDeivce);
    cudaMemcpy(b_dev, b, vector_size * sizeof(int), cudaMemcpyHostToDeivce);

    // Kernel launch
    vector_mul<<<vector_size/block_size,block_size>>>(a_dev,b_dev,c_dev);

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

