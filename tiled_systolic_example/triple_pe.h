// Copyright 2022 Democritus University of Thrace
// Integrated Circuits Lab
// 
#ifndef _TRIPLE_PE_H
#define _TRIPLE_PE_H

#include <iostream>

#include "mc_scverify.h"

template <typename T, int MEM>
class PE{
  private:
    T local_w[MEM];  // local weight memory

  public:
    PE(){}
    
    // This function represents the functionality of each PE
    void compute(T a, T c_in, T &out, int r_ptr, int c_ptr, const int num_h_tiles, bool write_en) {
      if (write_en) {
        // load local memory with weight values
        local_w[(r_ptr*num_h_tiles) + c_ptr] = a;
      } else {
        // compute the output
        out = c_in +  a * local_w[(r_ptr * num_h_tiles) + c_ptr];
      }
    }
};

#endif
