#ifndef _DEFS_H
#define _DEFS_H

#include "fast_float.h"
#include "ac_channel.h"
#include "mc_scverify.h"

typedef ffp16b btype;

// size of the problem (matrix dimensions)
static const int N = 4;
static const int I_F = 4;
static const int O_F1 = 4;
static const int O_F2 = 4;

// systolic array size
static const int K = 4;
static const int M = 4;

// number of non-zero elements of the sparse matrix
static const int nZ = 8;

static const int low = -10;
static const int high = 10;

// size of iput and output buffer
static const int I_B = 4;
static const int O_B = 4;

// size of memory in each PE
static const int MEM = 1;

#endif
