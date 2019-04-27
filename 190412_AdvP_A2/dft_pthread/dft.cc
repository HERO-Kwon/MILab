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
            complex_t h1_new, h2_new, wval;

            int w_num = k*int(1024/N);
            if(w_num > 512) wval = complex_t(-1.0,0.0) * w[w_num-512];
            else wval = w[w_num];

            h1_new = h[n_start+k] + wval * h[n_start+k + int(N/2)];
            h2_new = h[n_start+k] - wval * h[n_start+k + int(N/2)];

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

    //cast and dereference the input argument.
    unsigned tid = * (unsigned*) arg;

    //beginning of the critical section
    pthread_mutex_lock(&lock);
    //use condition variable to wait for thread's turn.
    while(tid > turn)
    {
        pthread_cond_wait(&cond, &lock);
    }    

    //accumulate thread ID values
    sum += tid;
    //update the turn
    turn++;
    //printf() is in the critical section only for demonstration purpose.
    printf("thread %d: %u\n",tid,sum-tid,sum);

    //wake up all other threads to check if they are the next one to go.
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&lock);

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

    cout << "Step1-1: Pre-Calc Weight" << endl;
    //make w array
    complex_t *w_arr = (complex_t*)malloc(width * sizeof(complex_t));
    
    for(unsigned r = 0; r < width/2-1; r++) //Loop for N/2-1 times
    {
        w_arr[r] = * new complex_t(cos(2*r*M_PI/width),-1*sin(2*r*M_PI/width)); //make w value.
    }
    

    cout << "dft1d-Step1: Shuffle inputs to binary order" << endl;
    //make new data array
    complex_t *old_data = data;
    data = new complex_t[width * height];
    
    for(unsigned row=0; row<height; row++) // Do it for every row
    {
        for(unsigned r=0; r<width; r++) // do it for every value
        {
            //calc shuffle order
            int quo_2 = r;
            int rem_2 = 0;
            int new_ind = 0;
            for(int k = int(log2(width))-1; k>=0 ; k--)
            {
                rem_2 = quo_2 % 2;
                quo_2 = int(quo_2 / 2);
                new_ind += pow(2,k) * rem_2;
            }
            //put reordered data to array
            data[height*row + new_ind] = old_data[height*row + r];
            old_data[height*row + r].~complex_t();
        }
    }
    delete[] old_data;
        
    cout << "Step2: Initialize thread variables" << endl;
    unsigned sum = 0;
    unsigned turn  = 0;
    pthread_mutex_t lock;
    pthread_cond_t cond;

    // number of threads to create -> main
    //initialize mutex lock and condition variable.
    assert(!pthread_mutex_init(&lock,0));
    assert(!pthread_cond_init(&cond,0));

    //allocate pthreads.
    pthread_t *threads = new pthread_t[num_threads];
    unsigned *tid = new unsigned[num_threads];

    //create pthreads and make each thread run func() above.
    for(unsigned t=0; t<num_threads; t++)
    {
        tid[t] = t;
        assert(!pthread_create(&threads[t],0,&dft_thread,&tid[t]));
    }

    //join pthreads/
    for(unsigned t=0; t<num_threads; t++)
    {
        pthread_join(threads[t],0);
        printf("thread %u done\n",t);
    }

    // deallocate pthreads.
    delete [] threads;
    delete [] tid;

    // destroy mutex lock
    assert(!pthread_mutex_destroy(&lock));
    assert(!pthread_cond_destroy(&cond));

    /*
    
    // Do it without thread
    //dft calc
    for(int r=1; r<=10; r++)
    {
        dft1d(data,w_arr,pow(2,r));
    }
    //transpose array
    complex_t *old_data1 = data;
    //data = (complex_t*)malloc(N*N * sizeof(complex_t));
    data = new complex_t[width * height];
    
    // Copy elements from the old array.
    for(unsigned i=0; i<height; i++)
    {
        for(unsigned j=0; j<width; j++)
        {
            new (&data[i*height+j]) complex_t(old_data1[i+j*width]);
            old_data1[i+j*width].~complex_t();
        }

    }
    // Deallocate the old array.
    delete[] old_data1;
    
    // Shuffle input to binary
    //make new data array
    complex_t *old_data2 = data;
    data = new complex_t[width * height];
    
    for(unsigned row=0; row<height; row++) // Do it for every row
    {
        for(unsigned r=0; r<width; r++) // do it for every value
        {
            //calc shuffle order
            int quo_2 = r;
            int rem_2 = 0;
            int new_ind = 0;
            for(int k = int(log2(width))-1; k>=0 ; k--)
            {
                rem_2 = quo_2 % 2;
                quo_2 = int(quo_2 / 2);
                new_ind += pow(2,k) * rem_2;
            }
            //put reordered data to array
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
    for(unsigned i=0; i<height; i++)
    {
        for(unsigned j=0; j<width; j++)
        {
            new (&data[i*height+j]) complex_t(old_data3[i+j*width]);
            old_data3[i+j*width].~complex_t();
        }

    }
    // Deallocate the old array.
    delete[] old_data3;

    //Step3: make thread
    */
}

