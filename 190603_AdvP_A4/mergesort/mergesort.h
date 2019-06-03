#ifndef __MERGESORT_H__
#define __MERGESORT_H__

#define num_data 1000000            // Number of data points
#define block_size 32               // Thread block size

void mergesort(unsigned *m_data);   // Mergesort called by main()
extern unsigned *data;              // Data array to sort

#endif

