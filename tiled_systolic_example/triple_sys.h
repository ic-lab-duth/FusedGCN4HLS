#ifndef _TRIPLE_SYS_H
#define _TRIPLE_SYS_H

#include "triple_pe.h"
#include "defs.h"
#include "mc_scverify.h"

#pragma hls_design
template< typename T, int K, int M , int I_B, int O_B, int MEM>
class triple_sys{
private:

  // instantiate a systolic array of size K by M
  PE< T, MEM > sys_array[K][M];

  void relu (T a[O_B], T b[O_B]) {
    #pragma unroll yes
    COL: for (int j = 0; j < O_B; j++) {
        b[j] = (a[j] > 0) ? a[j] : 0;
    }
  }

  void argmax (T in[O_B], T out[O_B]) {
    T maxVal = -65535.0;
    T curVal;
    MAX_ITER: for (int j=0; j < O_B; j++) {
      curVal = in[j];
      if (curVal > maxVal) {
        maxVal = curVal;
      }
    }
    
    COL: for (int j=0; j < O_B; j++) {
      out[j] = (in[j]==maxVal) ? (T)1.0 : (T)0.0;
    }
}

public:

  triple_sys() {

  }

  /* Multiply a[n][n] * h_i[n][i] * w[i][o] to get h_o[n][o] */
  /* Triple matrix multiplication using systolic array */
  #pragma hls_design interface
  void CCS_BLOCK(systolic_top) (T a_val, T h_i[I_B], ac_channel<T> w[K], T h_o[O_B], int active_row, int active_col, const int num_v_tiles, const int num_h_tiles, bool loadW, bool change_row, bool last_layer, bool apply_activation) {

    const int v_tiles = ((I_B%K)==0) ? (I_B/K) : (I_B/K)+1;
    const int h_tiles = ((O_B%M)==0) ? (O_B/M) : (O_B/M)+1;
    
    static T acc[M];
    static T acc_buf[M];
    
    T prodAH[K];
    T wires[K][M];

    static T out_row_buf[O_B];
    static T in_row_buf[I_B];

    
    for (int i=0; i < M; i++) {
      acc[i] = 0.0;
      acc_buf[i] = 0.0;
    }

    // load weights before the computation
    if (loadW) {
      for (int p=0; p < v_tiles; p++) {
        if (p < num_v_tiles) {
          for (int q=0; q < h_tiles; q++) {
            if (q < num_h_tiles) {
              #pragma unroll yes
              RD_W_ROW:for (int i=0; i < K; i++) {
                RD_W_COL:for (int j=0; j < M; j++) {
                  #ifndef __SYNTHESIS__
                  if (w[i].available(1)) 
                  #endif
                  {
                    sys_array[i][j].compute(w[i].read(), 0.0, wires[0][j], p, q, num_h_tiles, true);
                  }
                }
              }
            }
          }
        }
      }
    } else {
      // initialize local buffers
      if (change_row) {
        for (int i=0; i < O_B; i++) {
          out_row_buf[i] = 0.0;
        }
      }

      for (int i=0; i < I_B; i++) {
        if (i < active_row)
          in_row_buf[i] = h_i[i];
      }

      for (int i=0; i < K; i++) {
        for (int j=0; j < M; j++) {
          wires[i][j] = 0.0;
        }
      }

      // start the systolic computation
      ROW_TILE: for (int p=0; p < v_tiles; p++) {
        if (p < num_v_tiles) {
          // multiply sparse-matrix element with correspoding row of H
          #pragma unroll yes
          MUL_AH: for (int i=0; i < K; i++) {
            prodAH[i] = a_val * in_row_buf[(p*K) + i];
          }

          COL_TILE: for (int q=0; q < h_tiles; q++) {
            if (q < num_h_tiles) {
              #pragma unroll yes
              RD_BUF: for (int k=0; k < M; k++) {
                acc_buf[k]  = out_row_buf[(q*M) + k];
              }
              
              // spatial unroll corresponds to each PE
              #pragma unroll yes
              GRID_ROW: for (int i=0; i < K; i++) {
                
                #pragma unroll yes
                GRID_COL: for (int j=0; j < M; j++) {
                  if(i==0){
                    // compute for the first row of PEs
                    sys_array[0][j].compute(prodAH[0], 0.0, wires[0][j], p, q, num_h_tiles, false);
                  }else{
                    sys_array[i][j].compute(prodAH[i], wires[i-1][j], wires[i][j], p, q, num_h_tiles, false);
                  }
                  if (i == K-1) {
                    acc[j] = acc_buf[j] + wires[i][j];
                  }
                }
              }
                
              #pragma unroll yes
              WR_BUF: for (int k=0; k < M; k++) {
                out_row_buf[(q*M) + k] = acc[k];
              } 
            } 
          }
        }
      }

      if (apply_activation) {
        if (!last_layer) {
          relu(out_row_buf, h_o);
        } else {
          argmax(out_row_buf, h_o);
        }
      }

    }
  }

};
#endif
