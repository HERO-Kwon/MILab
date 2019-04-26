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
void dft1d(complex_t *h, complex_t *w, const unsigned N) {
    // Use Cooley-Tuckey algorithm to perform row-wise 1-D DFT.
    // Step 1: The elements of row input, h, have to be first hashed in the
    //         bit-reversed order.
    // Step 2: Follow Danielson-Lanczos Lemma to perform the DFT.
    /* Assignment */
    //block
    for(int r = 0; r < int(1024*1024/N) ; r++)
    {
        int n_start = N*r;
        //calc
        for(int k=0; k < int(N/2); k++)
        {
            //dft_calc2(&h[n_start+k],&h[n_start+k + int(N/2)],w[k*int(1024/2)]);
            //complex_t h1 = h[n_start+k];
            //complex_t h2 = h[n_start+k + int(N/2)];
            //complex_t w_val = w[k*int(1024/2)];
            
            //complex_t h1_new = h[n_start+k] + w[k*int(1024/2)] * h[n_start+k + int(N/2)];
            //complex_t h2_new = h[n_start+k] - w[k*int(1024/2)] * h[n_start+k + int(N/2)];
            complex_t h1_new, h2_new, wval;

            int w_num = k*int(1024/N);
            if(w_num > 512) wval = complex_t(-1.0,0.0) * w[w_num-512];
            else wval = w[w_num];

            h1_new = h[n_start+k] + wval * h[n_start+k + int(N/2)];
            h2_new = h[n_start+k] - wval * h[n_start+k + int(N/2)];
            
            //h1_new = new complex_t *(h[n_start+k] + w[k*int(1024/2)] * h[n_start+k + int(N/2)]);
            //h2_new = new complex_t *(h[n_start+k] - w[k*int(1024/2)] * h[n_start+k + int(N/2)])

            h[n_start+k] = h1_new;
            h[n_start+k + int(N/2)] = h2_new;

        }
    }

}

// 1-D DFT Thread function
void* dft_thread(void *arg) {
    // This thread function calls dft1d() function above for data rows.
    // Each thread performs 1-D DFT for height/num_threads number of rows.
    /* Assignment */

    // beginning of critical section
    // pthread_mutex_lock(&lock);
    // use condition var to wait turn
    


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

    // Step1: Pre-Calc Weight
    cout << "Step1" << endl;
    // Allocate data.
    unsigned N = width; //width=1024

    //make array
    complex_t *w_arr = (complex_t*)malloc(N * sizeof(complex_t));
    // Pre-Calc Weights.
    for(unsigned r = 0; r < N/2-1; r++)
    {
        //make W value.
        // W^n = cos(2n*pi/N) - jsin(2n*pi/N)
        w_arr[r] = * new complex_t(cos(2*r*M_PI/N),-1*sin(2*r*M_PI/N));
    }

    cout << "w[0].re:"<< w_arr[0].re << endl;
    cout << "w[0].im:"<< w_arr[0].im << endl;
    cout << "w[1].re:"<< w_arr[1].re << endl;
    cout << "w[1].im:"<< w_arr[1].im << endl;

    // Step1: Shuffle inputs to binary order
    complex_t *old_data0 = data;
    //data = (complex_t*)malloc(N*N * sizeof(complex_t));
    data = new complex_t[width * height];
    
    //make array
    //complex_t *x_sh = (complex_t*)malloc(N*N * sizeof(complex_t));
    // row loop
    for(unsigned row=0; row<N; row++)
    {
        for(unsigned r=0; r<N; r++)
        {
            //calc shuffle order
            int quo_2 = r;
            int rem_2 = 0;
            int new_ind = 0;
            for(int k = int(log2(N))-1; k>=0 ; k--)
            {
                rem_2 = quo_2 % 2;
                quo_2 = int(quo_2 / 2);
                new_ind += pow(2,k) * rem_2;
            }
            data[height*row + new_ind] = old_data0[height*row + r];
            old_data0[height*row + r].~complex_t();
        }
    }
    delete[] old_data0;


    // Save the result to a file.
    file_handler.save_data("temp-shuffle.txt", data, width, height);
    //Step2: initialize variables
    //cout << "Step2" << endl;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    
    // Do it without thread
    
    //dft calc
    for(int r=1; r<=10; r++)
    {
        dft1d(data,w_arr,pow(2,r));
    }
    // Save the result to a file.
    file_handler.save_data("temp-dft.txt", data, width, height);
    
    
    //transpose array

    complex_t *old_data1 = data;
    //data = (complex_t*)malloc(N*N * sizeof(complex_t));
    data = new complex_t[width * height];
    
    // Copy elements from the old array.
    for(unsigned i = 0; i < N; i++)
    {
        for(unsigned j=0;j<N;j++)
        {
            new (&data[i*N+j]) complex_t(old_data1[i+j*N]);
            old_data1[i+j*N].~complex_t();
        }

    }
    // Deallocate the old array.
    delete[] old_data1;
    
    // Step1: Shuffle inputs to binary order
    complex_t *old_data2 = data;
    //data = (complex_t*)malloc(N*N * sizeof(complex_t));
    data = new complex_t[width * height];
    
    //make array
    //complex_t *x_sh = (complex_t*)malloc(N*N * sizeof(complex_t));
    // row loop
    for(unsigned row=0; row<N; row++)
    {
        for(unsigned r=0; r<N; r++)
        {
            //calc shuffle order
            int quo_2 = r;
            int rem_2 = 0;
            int new_ind = 0;
            for(int k = int(log2(N))-1; k>=0 ; k--)
            {
                rem_2 = quo_2 % 2;
                quo_2 = int(quo_2 / 2);
                new_ind += pow(2,k) * rem_2;
            }
            data[height*row + new_ind] = old_data2[height*row + r];
            old_data2[height*row + r].~complex_t();
        }
    }
    delete[] old_data2;
    
    // dft again
    for(int r=1; r<=10; r++)
    {
        dft1d(data,w_arr,pow(2,r));
    }

    // transpose again

    complex_t *old_data3 = data;
    //data = (complex_t*)malloc(N*N * sizeof(complex_t));
    data = new complex_t[width * height];
    
    // Copy elements from the old array.
    for(unsigned i = 0; i < N; i++)
    {
        for(unsigned j=0;j<N;j++)
        {
            new (&data[i*N+j]) complex_t(old_data3[i+j*N]);
            old_data3[i+j*N].~complex_t();
        }

    }
    // Deallocate the old array.
    delete[] old_data3;

    // Save the result to a file.
    file_handler.save_data("temp-dft2.txt", data, width, height);
    
    //Step3: make thread
    
}

