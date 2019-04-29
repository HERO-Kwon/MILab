#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include "dft.h"
#include <stdio.h>
using namespace std;
        
// conditional variables
unsigned turn  = 0;
unsigned running = 1;
unsigned exit_var = 0;
pthread_mutex_t lock;
pthread_cond_t cond;
pthread_mutex_t lock_join;
pthread_cond_t cond_join;

//make global w array
complex_t *w_arr = (complex_t*)malloc(512 * sizeof(complex_t));

void thread_exit() {
    // Use pthread mutex and condition variables to exit the child function,
    // and signal the parent thread that this thread is done.
    /* Assignment */

    pthread_mutex_lock(&lock_join); //beginning of the critical section
    running = 0; // condition variable to check whether this thread is running
    pthread_cond_broadcast(&cond_join); //signal the parent thread that this thread is done
    while(exit_var == 0) // wait for response of join function
    {
        pthread_cond_wait(&cond_join, &lock_join);
    }
    exit_var = 0; // lock condition variable
    pthread_mutex_unlock(&lock_join); //ending of the critical section
}

// Custom thread join function.
void thread_join() {
    // Use pthread mutex and condition variables to wait until all children
    // threads complete their executions.
    /* Assignment */

    //beginning of the critical section
    pthread_mutex_lock(&lock_join);
    // if thread is already running, wait for it.
    while(running == 1)
    {
        pthread_cond_wait(&cond_join, &lock_join);
    }
  
    exit_var = 1;
    running = 1;
    pthread_cond_broadcast(&cond_join); // broadcast for thread function to exit
    pthread_mutex_unlock(&lock_join);  //ending of critical section
}

// Perform 1-D DFT.
void dft1d(complex_t *h, const unsigned N) {
    // Use Cooley-Tuckey algorithm to perform row-wise 1-D DFT.
    // Step 1: The elements of row input, h, have to be first hashed in the
    //         bit-reversed order.
    // Step 2: Follow Danielson-Lanczos Lemma to perform the DFT.
    /* Assignment */
    
    // Step 1
    //cout << "DFT:Step1: Shuffle inputs to binary order" << endl;
    //make new data array
    complex_t *dft_data;
    dft_data = new complex_t[width * height/num_threads];
    
    for(unsigned c=0; c<height/num_threads; c++) // Do it for every row
    {
        for(unsigned r=0; r<width; r++) // do it for every value
        {
            // initialize variables
            int quo_2 = r; 
            int rem_2 = 0;
            int new_ind = 0;
            //calc binary shuffle order by dividing 2
            for(int k = int(log2(width))-1; k>=0 ; k--)
            {
                rem_2 = quo_2 % 2;
                quo_2 = int(quo_2 / 2);
                new_ind += pow(2,k) * rem_2; //order number in decimal value
            }
            //put reordered data to array
            new (&dft_data[height*c + new_ind]) complex_t(h[height*c + r]);
        }
    }
    
    // Step 2
    //cout << "DFT:Step2: Perform DFT" << endl;
    // iterate power of 2
    for(int p=1; p<=int(log2(N)); p++)
    {
        int size = pow(2,p);
        // do it for every row
        for(int r = 0; r < int(width*(height/num_threads)/size) ; r++)
        {
            // calculation of DFT value
            for(int k=0; k < int(size/2); k++)
            {
                complex_t h1_new, h2_new, wval;

                //get w index
                int w_num = k*int(1024/size);
                if(w_num >= 512) wval =  complex_t(-1.0,0.0) * w_arr[w_num-512];
                else wval = w_arr[w_num];

                //perform DFT
                h1_new = dft_data[r*size+k] + wval * dft_data[r*size+k + int(size/2)];
                h2_new = dft_data[r*size+k] - wval * dft_data[r*size+k + int(size/2)];

                //put new value into data array
                dft_data[r*size+k] = h1_new;
                dft_data[r*size+k + int(size/2)] = h2_new;
                h[r*size+k] = h1_new;
                h[r*size+k + int(size/2)] = h2_new;
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

    pthread_mutex_lock(&lock);
    
    //use condition variable to wait for thread's turn.
    while(tid > turn)
    {
        pthread_cond_wait(&cond, &lock);
    }
    //printf() is in the critical section only for demonstration.
    printf("thread %d:\n",tid);
    
    int n_start = width*(height/num_threads)*tid; //starting position of this data block
    dft1d(&data[n_start],width); // calculate DFT
    //update the turn
    turn++;
    
    thread_exit(); // inform join that this thread is finished
    // unlock condition variable    
    pthread_mutex_unlock(&lock);
    pthread_cond_broadcast(&cond);
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
    }

    cout << "Step2: Initialize thread variables" << endl;
    //initialize mutex lock and condition variable.
    assert(!pthread_mutex_init(&lock,0));
    assert(!pthread_cond_init(&cond,0));

    cout << "Step 3: Create threads to run dft_thread() for row-wise DFTs." << endl;
    //allocate pthreads.
    pthread_t *threads = new pthread_t[num_threads];
    unsigned *tid = new unsigned[num_threads];
    //create pthreads and make each thread run func() above.
    for(unsigned t=0; t<num_threads; t++)
    {
        tid[t] = t;
        assert(!pthread_create(&threads[t],0,&dft_thread,&tid[t])); // make thread
    }
    cout << "Step 4: Call thread_join() to wait for all threads to complete." << endl;
    for(unsigned t=0; t<num_threads; t++)
    {
        thread_join();
        printf("thread %u done\n",t);
    }

    cout << "Step 5: Transpose the data matrix so that column-wise DFT can be performed" << endl;
    // make copy of old array
    complex_t *old_data1 = data;
    // make new data array
    data = new complex_t[width * height];
    
    // Copy elements from the transposed value from old array
    for(unsigned i=0; i<height; i++)
    {
        for(unsigned j=0; j<width; j++)
        {
            //transpose
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
        assert(!pthread_create(&threads[t],0,&dft_thread,&tid[t])); //make thread
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
    data = new complex_t[width * height];
    
    // Copy elements from the old array.
    for(unsigned i=0; i<height; i++)
    {
        for(unsigned j=0; j<width; j++)
        {
            //transpose
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
    
    cout << "Step 10: Deallocated heap memory blocks if any." << endl;
    //delete w values
    free(w_arr);

}

