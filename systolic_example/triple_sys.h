#ifndef _TRIPLE_SYS_H
#define _TRIPLE_SYS_H

#include "defs.h"
#include "ac_int.h"
#include "triple_pe.h"
#include "mc_scverify.h"

#pragma hls_design
template< typename T, int N, int I_F, int O_F1, int O_F2, int nZ, int K, int M , int I_B, int O_B, int MEM>
class triple_sys{
private:

  // instantiate a systolic array of size K by M
  PE< T, MEM > sys_array[K][M];

  void relu (T a[O_F1], T b[O_F1]) {
    #pragma unroll yes
    COL: for (int j = 0; j < O_F1; j++) {
      b[j] = (a[j] > 0) ? a[j] : 0;
    }
  }

  void argmax (T in[O_F2], T out[O_F2]) {
    T maxVal = -65535.0;
    T curVal;
    MAX_ITER: for (int j=0; j < O_F2; j++) {
      curVal = in[j];
      if (curVal > maxVal) {
        maxVal = curVal;
      }
    }
    
    COL: for (int j=0; j < O_F2; j++) {
      out[j] = (in[j]==maxVal) ? (T)1.0 : (T)0.0;
    }
  }

  /* Triple matrix multiplication using systolic array */
  void systolic (T a_val, T h_i[I_B], ac_channel<T> w[K], T h_o[O_B], int active_row, int active_col, const int num_v_tiles, const int num_h_tiles, bool loadW, bool change_row) {

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
      
      WR_OUT: for (int j=0; j < O_B; j++) {
        if (j <active_col)
          h_o[j] = out_row_buf[j];
      }

    }
  }

public:

  triple_sys() {

  }

  /* Top module */
  #pragma hls_design interface
  void CCS_BLOCK(run)(int a_row[N+1], int a_col[nZ], T a_val[nZ], T h_i[N][I_F], ac_channel<T> w1[K], ac_channel<T> w2[K], T h_o[N][O_F2]/*, const int ver, const int hor*/){
    bool change_row;

    // number of vertical and horizontal tiles of weight matrix for the first layer
    const int ver = ((I_F%K)==0) ? (I_F/K) : (I_F/K)+1;
    const int hor = ((O_F1%M)==0) ? (O_F1/M) : (O_F1/M)+1;

    // number of vertical and horizontal tiles of weight matrix for the second layer
    const int ver2 = ((O_F1%K)==0) ? (O_F1/K) : (O_F1/K)+1;
    const int hor2 = ((O_F2%M)==0) ? (O_F2/M) : (O_F2/M)+1;

    static T inter[N][O_F1];
    
    ac_int<32, false> lambda;
    btype a_lambda;

    static T in_row[I_B];
    static T out_row[O_B];

    // first GCN layer

    // load weights to PEs
    systolic(a_lambda, in_row, w1, out_row, I_F, O_F1, ver, hor, true, true);

    // push different tiles in the systolic array
    ITER: for (int i=0; i < N; i++) {
      change_row = true;

      INTER_ROW_ITER: for (int j=a_row[i]; j < a_row[i+1]; j++) {
        lambda = a_col[j];
        a_lambda = a_val[j];
        
        for (int k=0; k < I_B; k++) {
          if (k < I_F)
            in_row[k] = h_i[lambda][k];
          else
            in_row[k] = (btype)0;
        }

        systolic(a_lambda, in_row, w1, out_row, I_F, O_F1, ver, hor, false, change_row);
        change_row = false;

      }
      
      relu(out_row, inter[i]);
      
    }

    // second GCN layer

    // load weights to PEs
    systolic(a_lambda, in_row, w2, out_row, O_F1, O_F2, ver2, hor2, true, true);

    // push different tiles in the systolic array
    ITER_2: for (int i=0; i < N; i++) {
      change_row = true;

      INTER_ROW_ITER_2: for (int j=a_row[i]; j < a_row[i+1]; j++) {
        lambda = a_col[j];
        a_lambda = a_val[j];
        
        for (int k=0; k < I_B; k++) {
          if (k < O_F1)
            in_row[k] = inter[lambda][k];
          else
            in_row[k] = (btype)0;
        }

        systolic(a_lambda, in_row, w2, out_row, O_F1, O_F2, ver2, hor2, false, change_row);
        change_row = false;

      }

      argmax(out_row, h_o[i]);
            
    }

  }

};
#endif
