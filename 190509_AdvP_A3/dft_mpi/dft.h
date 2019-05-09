#ifndef __DFT_H__
#define __DFT_H__

#include "complex.h"
#include "file_handler.h"

// Perform 2-D DFT.
void dft2d();

// File handler
extern file_handler_t file_handler;
// Number of ranks
extern int num_ranks;
// Rank ID
extern int rank_id;
// Data to perform DFT.
extern complex_t *data;
// Data dimension
extern unsigned width, height;

#endif

