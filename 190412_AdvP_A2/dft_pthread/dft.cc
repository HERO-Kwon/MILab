#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include "dft.h"
#include <stdio.h>
using namespace std;
        
unsigned turn  = 0;
unsigned running = 0;
pthread_mutex_t lock;
pthread_cond_t cond;
pthread_mutex_t lock_join;
pthread_cond_t cond_join;

//make w array
complex_t *w_arr = (complex_t*)malloc(511 * sizeof(complex_t));


            
void thread_exit() {
    // Use pthread mutex and condition variables to exit the child function,
    // and signal the parent thread that this thread is done.
    /* Assignment */
    
    pthread_mutex_lock(&lock_join); //beginning of the critical section
    running = 0; // condition variable to check whether this thread is running
    pthread_cond_broadcast(&cond_join); //signal the parent thread that this thread is done
    pthread_mutex_unlock(&lock_join); //ending of the critical section

}

// Custom thread join function.
void thread_join() {
    // Use pthread mutex and condition variables to wait until all children
    // threads complete their executions.
    /* Assignment */
    //beginning of the critical section
    //pthread_mutex_lock(&lock);
    //printf("thread join\n");

    //beginning of the critical section
    pthread_mutex_lock(&lock_join);
    //use condition variable to wait for thread's turn.
    while(running == 1)
    {
        pthread_cond_wait(&cond_join, &lock_join);
    }    
    pthread_mutex_unlock(&lock_join);
}



// Perform 1-D DFT.
void dft1d(complex_t *h, const unsigned N) {
    // Use Cooley-Tuckey algorithm to perform row-wise 1-D DFT.
    // Step 1: The elements of row input, h, have to be first hashed in the
    //         bit-reversed order.
    // Step 2: Follow Danielson-Lanczos Lemma to perform the DFT.
    /* Assignment */
    
    //printf("h[5]: %f\n",h[5].re);
    //printf("N: %d\n",N);

    // Step 1
    //cout << "DFT:Step1: Shuffle inputs to binary order" << endl;
    //make new data array
    complex_t *dft_data;
    dft_data = new complex_t[width * height/num_threads];
    
    for(unsigned row=0; row<height/num_threads; row++) // Do it for every row
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
            new (&dft_data[height*row + new_ind]) complex_t(h[height*row + r]);
            //dft_data[height*row + new_ind] = h[height*row + r];
            //old_data[height*row + r].~complex_t();
        }
    }
    
    // Step 2
    //block
    for(int p=1; p<=int(log2(N)); p++)
    {
        int size = pow(2,p);
            for(int r = 0; r < int(width*(height/num_threads)/size) ; r++)
            {
            //int n_start = 1024*1024*(tid/num_threads)*(r/N);
            //calc
            for(int k=0; k < int(size/2); k++)
            {
                complex_t h1_new, h2_new, wval;

                int w_num = k*int(1024/size);
                if(w_num >= 512) wval =  complex_t(-1.0,0.0) * w_arr[w_num-512];
                else wval = w_arr[w_num];

                h1_new = dft_data[r*size+k] + wval * dft_data[r*size+k + int(size/2)];
                h2_new = dft_data[r*size+k] - wval * dft_data[r*size+k + int(size/2)];

                dft_data[r*size+k] = h1_new;
                dft_data[r*size+k + int(size/2)] = h2_new;
                h[r*size+k] = h1_new;
                h[r*size+k + int(size/2)] = h2_new;

                //h[r*N+k].~complex_t();
                //new (&h[r*N+k]) complex_t(h1_new);
                //h[r*N+k + int(N/2)].~complex_t();
                //new (&h[r*N+k + int(N/2)]) complex_t(h2_new);


            }
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
    running=1;
    //use condition variable to wait for thread's turn.
    while(tid > turn)
    {
        pthread_cond_wait(&cond, &lock);
    }    

    //accumulate thread ID values
    //sum += tid;

    //dft calc

    int n_start = width*(height/num_threads)*tid;
            
    //printf("tid: %d\n",tid);
    //printf("n_st: %d\n",n_start);
    /*
    for(int p=1; p<=10; p++)
    {
        dft1d(&data[n_start],pow(2,p));
    }
    */
    dft1d(&data[n_start],width);
    //update the turn
    turn++;
    

    //printf() is in the critical section only for demonstration purpose.
    printf("thread %d:\n",tid);


    //wake up all other threads to check if they are the next one to go.
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&lock);
    
    thread_exit();
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
    
    for(unsigned r = 0; r < unsigned(width/2-1); r++) //Loop for N/2-1 times
    {
        w_arr[r] = * new complex_t(cos(2*r*M_PI/width),-1*sin(2*r*M_PI/width)); //make w value.

        //new (&w_arr[r]) complex_t(cos(2*r*M_PI/width),-1*sin(2*r*M_PI/width));
        
    }

    cout << "Step2: Initialize thread variables" << endl;
    // number of threads to create -> main
    //initialize mutex lock and condition variable.
    assert(!pthread_mutex_init(&lock,0));
    assert(!pthread_cond_init(&cond,0));

    //allocate pthreads.
    pthread_t *threads = new pthread_t[num_threads];
    unsigned *tid = new unsigned[num_threads];

    cout << "Step 3: Create threads to run dft_thread() for row-wise DFTs." << endl;
    //create pthreads and make each thread run func() above.
    for(unsigned t=0; t<num_threads; t++)
    {
        tid[t] = t;
        assert(!pthread_create(&threads[t],0,&dft_thread,&tid[t]));
    }
    cout << "Step 4: Call thread_join() to wait for all threads to complete." << endl;
    //join pthreads/
    for(unsigned t=0; t<num_threads; t++)
    {
        thread_join();
        printf("thread %u done\n",t);
    }

    cout << "Step 5: Transpose the data matrix so that column-wise DFT can be performed" << endl;
    
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
    

    cout << "Step 6: Create threads to run dft_thread() for row-wise DFTs." << endl;
    //create pthreads and make each thread run func() above.
    for(unsigned t=0; t<num_threads; t++)
    {
        tid[t] = t;
        assert(!pthread_create(&threads[t],0,&dft_thread,&tid[t]));
    }
    cout << "Step 7: Call thread_join() to wait for all threads to complete." << endl;
    //join pthreads/
    for(unsigned t=0; t<num_threads; t++)
    {
        thread_join();
        printf("thread %u done\n",t);
    }


    
    cout << "Step 8: Transpose the data matrix back to the original orientation." << endl;
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

    cout << "Step 9: Destroy pthread-related variables." << endl;
    // deallocate pthreads.
    delete [] threads;
    delete [] tid;

    // destroy mutex lock
    assert(!pthread_mutex_destroy(&lock));
    assert(!pthread_cond_destroy(&cond));
    assert(!pthread_mutex_destroy(&lock_join));
    assert(!pthread_cond_destroy(&cond_join));
    
    // Step 10: Deallocated heap memory blocks if any.
    free(w_arr);

}

