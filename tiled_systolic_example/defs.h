// Copyright 2022 Democritus University of Thrace
// Integrated Circuits Lab
// 
#ifndef _DEFS_H
#define _DEFS_H

#include "fast_float.h"
#include "ac_channel.h"
#include "matrix.h"
#include "mc_scverify.h"

typedef ffp16b btype;

// size of the problem (matrix dimensions)
/*
// for cora dataset
static const int N = 2708;  // number of nodes
static const int I_F = 1433;  // number of input features
static const int O_F1 = 64;  // number of features in the hidden layer
static const int O_F2 = 7;  // number of output features
static const int nZ = 13264;  // number of non-zero values of adjacency
*/

// for citeseer dataset
static const int N = 3327;  // number of nodes
static const int I_F = 3703;  // number of input features
static const int O_F1 = 21;  // number of features in the hidden layer
static const int O_F2 = 6;  // number of output features
static const int nZ = 12431;  // number of non-zero values of adjacency

/*
// for pubmed dataset
static const int N = 19717;  // number of nodes
static const int I_F = 500;  // number of input features
static const int O_F1 = 18;  // number of features in the hidden layer
static const int O_F2 = 3;  // number of output features
static const int nZ = 108365;  // number of non-zero values of adjacency 
*/

// systolic array size
static const int K = 16;
static const int M = 16;

// size of input and output buffer
static const int I_B = 32;
static const int O_B = 32;

// size of memory in each PE
static const int MEM = 512;

#endif
