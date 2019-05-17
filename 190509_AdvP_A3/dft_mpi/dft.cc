#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include "dft.h"

using namespace std;

//make global w array
complex_t *w_arr = (complex_t*)malloc(512 * sizeof(complex_t));


// Custom MPI_Allgather function with complex_t data type
void mpi_allgather(complex_t *m_send_buffer, int m_send_count,
                   complex_t *m_recv_buffer, int m_recv_count,
                   MPI_Comm m_comm) {
    // Use a group of non-blocking receive (i.e., MPI_Irecv) and blocking
    // send (i.e., MPI_Send) along with MPI_Wait() to implement the custom
    // MPI_Allgather function. Do not use collective communication functions
    // such as MPI_Allgather(), MPI_Gather, MPI_Scatter, or MPI_Broadcast.
    /* Assignment */

    //MPI_Allgather(&data[n_start],width*(height/num_ranks)*sizeof(complex_t),MPI_BYTE,
    //recv_buffer,width*(height/num_ranks)*sizeof(complex_t),MPI_BYTE,MPI_COMM_WORLD);
    
    int n_start = width*(height/num_ranks) * rank_id; //starting position of this data block
    //complex_t *temp_recv_buffer = (complex_t*)malloc(m_recv_count * sizeof(complex_t));

    // copy recv buffer to data
    for(int i = 0; i < int(m_send_count/sizeof(complex_t)); i++) {
        m_recv_buffer[n_start+i].~complex_t();
        new (&m_recv_buffer[n_start+i]) complex_t(m_send_buffer[i]);   
    }

    MPI_Request mpi_recv_request;
    for(int r = 0; r < num_ranks; r++) 
    {
        if(r!=rank_id)
        {
            // receive a message from the previous process
            MPI_Irecv(&m_recv_buffer[width*(height/num_ranks)*r],m_recv_count,MPI_BYTE,MPI_ANY_SOURCE,
            r, MPI_COMM_WORLD, &mpi_recv_request);
            // send the message to the next process
            MPI_Send(m_send_buffer,m_send_count,MPI_BYTE,
            r,rank_id, MPI_COMM_WORLD);
        }
        
    }
    
    for(int k = 0; k < num_ranks-1 ; k++)
    {
        // receive a message from the last process
        MPI_Status mpi_status;    
        MPI_Wait(&mpi_recv_request, &mpi_status);
        
        cout << "MPI Rank " << rank_id << " start from "<< n_start << " received a message from rank "
        << mpi_status.MPI_SOURCE << ": " << m_recv_buffer[n_start].re << endl;
    }

    
    //int temp_start = m_recv_count*mpi_status.MPI_SOURCE; //starting position of this data block
    /*
    for(int i = 0; i < m_recv_count; i++) {
        m_recv_buffer[temp_start+i].~complex_t();
        new (&m_recv_buffer[temp_start+i]) complex_t(temp_recv_buffer[i]);   
        temp_recv_buffer[i].~complex_t();
    }
    */
}

// Transpose a matrix.
void transpose_matrix(complex_t *h, const unsigned m_width, const unsigned m_height) {
    
    // make new data array
    data = new complex_t[width * height];
    
    // Copy elements from the transposed value from old array
    for(unsigned i=0; i<height; i++)
    {
        for(unsigned j=0; j<width; j++)
        {
            //transpose
            new (&data[i*height+j]) complex_t(h[i+j*width]);
        }
    }
}

// Perform 1-D DFT.
void dft1d(complex_t *h, const unsigned N) {
    // This function implements 1-D DFT for the array h of length N.
    // You may reuse the Cooley-Tuckey algorithm that you wrote for the
    // previous pthread assignment, or you may directly calculate the DFT
    // equation shown in the lecture note.
    /* Assignment */

    // Step 1
    //cout << "DFT:Step1: Shuffle inputs to binary order" << endl;
    //make new data array
    complex_t *dft_data;
    dft_data = new complex_t[width * height/num_ranks];
    
    for(unsigned c=0; c<unsigned(height/num_ranks); c++) // Do it for every row
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
        for(int r = 0; r < int(width*(height/num_ranks)/size) ; r++)
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


    cout << "Step0: Pre-Calc Weight" << endl;
    for(unsigned r = 0; r < unsigned(width/2); r++) //Loop for N/2-1 times
    {
        w_arr[r] = * new complex_t(cos(2*r*M_PI/width),-1*sin(2*r*M_PI/width)); //make w value.        
    }

    // print the number of ranks and rank ID
    cout << "Temp: Number of ranks = " << num_ranks
    << ", rank ID = " << rank_id << endl;

    unsigned buffer_size = width*height;
    complex_t *recv_buffer = (complex_t*)malloc(buffer_size * sizeof(complex_t));
    
    printf("rank %d:\n",rank_id);
    
    int n_start = width*(height/num_ranks)*rank_id; //starting position of this data block

    dft1d(&data[n_start],width); // calculate DFT

    // the message is gathered across all process
    //MPI_Allgather(&data[n_start],width*(height/num_ranks)*sizeof(complex_t),MPI_BYTE,
    //recv_buffer,width*(height/num_ranks)*sizeof(complex_t),MPI_BYTE,MPI_COMM_WORLD);

    // Custom MPI_Allgather function with complex_t data type
    mpi_allgather(&data[n_start], width*(height/num_ranks)*sizeof(complex_t),
                   recv_buffer, width*(height/num_ranks)*sizeof(complex_t),
                   MPI_COMM_WORLD);

    
    // copy recv buffer to data
    for(size_t i = 0; i < width*height; i++) {
        data[i].~complex_t();
        new (&data[i]) complex_t(recv_buffer[i]);   
        recv_buffer[i].~complex_t();
    }

    // transpose
    transpose_matrix(data,width,height);
    
    // do dft
    dft1d(&data[n_start],width); // calculate DFT

    // the message is gathered across all process
    //MPI_Allgather(&data[n_start],width*(height/num_ranks)*sizeof(complex_t),MPI_BYTE,
    //recv_buffer,width*(height/num_ranks)*sizeof(complex_t),MPI_BYTE,MPI_COMM_WORLD);
    
    mpi_allgather(&data[n_start], width*(height/num_ranks)*sizeof(complex_t),
                recv_buffer, width*(height/num_ranks)*sizeof(complex_t),
                MPI_COMM_WORLD);

    
    // copy recv buffer to data
    for(size_t i = 0; i < width*height; i++) {
        data[i].~complex_t();
        new (&data[i]) complex_t(recv_buffer[i]);   
        recv_buffer[i].~complex_t();
    }
    
    // transpose
    transpose_matrix(data,width,height);
    
    // print the information of gathered message.
    cout << "MPI Rank " << rank_id << " has the array of [";
    //for(unsigned i=0; i<buffer_size; i++){
        cout << recv_buffer[rank_id*width*(height/num_ranks)].re << " " ;
        cout << recv_buffer[rank_id*width*(height/num_ranks)].im << " " ;
        cout << "\b]" << endl;
    //}

    //delete variables
    delete(recv_buffer);
    free(w_arr);
}

