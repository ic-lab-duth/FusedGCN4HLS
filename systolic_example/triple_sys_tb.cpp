#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <fstream>
#include <istream>
#include <sstream>
#include <string>

#include <vector>
#include <cstdlib>
#include <ctime>

#include "defs.h"
#include "triple_sys.h"
#include "helper.h"
#include "mc_scverify.h"


CCS_MAIN(int argc, char** argv) {

  triple_sys< btype, K, M, I_B, O_B, MEM > sys_array;

  // number of vertical and horizontal tiles of weight matrix for the first layer
  const int ver = ((I_F%K)==0) ? (I_F/K) : (I_F/K)+1;
  const int hor = ((O_F1%M)==0) ? (O_F1/M) : (O_F1/M)+1;

  // number of vertical and horizontal tiles of weight matrix for the second layer
  const int ver2 = ((O_F1%K)==0) ? (O_F1/K) : (O_F1/K)+1;
  const int hor2 = ((O_F2%M)==0) ? (O_F2/M) : (O_F2/M)+1;
  
  // create random inputs
  srand(123);
  btype h[N][I_F];
  btype w1[I_F][O_F1];
  btype w2[O_F1][O_F2];
  btype a_val[nZ];

  btype a[N][N];
  
  for (int i=0; i < N; i++) {
    for (int j=0; j < N; j++) {
      a[i][j] = 0.0;
    }
  }

  int a_row[N+1];
  int a_col[nZ];

  int t = 0;
  int row, col;
  while (t < nZ){
    row = rand() % N;
    col = rand() % N;

    btype temp;
    temp = 0.0;
    if (a[row][col]==temp){
      a[row][col] = low + static_cast<dtype>(rand()) * static_cast<dtype>(high - low) / RAND_MAX;
      t++;
    }
  }

  print2d<N,N> (a);
  std::cout << "===========================================" <<std::endl;
  encode_csr<btype, N, N, nZ>(a, a_row, a_col, a_val);
  
  for (int i=0; i < N+1; i++) {
    a_row[i] = i;
  }
  
  for (int i=0; i < nZ; i++) {
    a_col[i] = i;
    a_val[i] = (dtype)1;
  }
  
  for (int i=0; i < N; i++) {
    for (int j=0; j < I_F; j++) {
      h[i][j] = low + static_cast<dtype>(rand()) * static_cast<dtype>(high - low) / RAND_MAX;
    }
  }
  
  for (int i=0; i < I_F; i++) {
    for (int j=0; j < O_F1; j++) {
      w1[i][j] = low + static_cast<dtype>(rand()) * static_cast<dtype>(high - low) / RAND_MAX;
    }
  }

  for (int i=0; i < O_F1; i++) {
    for (int j=0; j < O_F2; j++) {
      w2[i][j] = low + static_cast<dtype>(rand()) * static_cast<dtype>(high - low) / RAND_MAX;
    }
  }

  btype h_o[N][O_F1];
  btype final[N][O_F2];

  ac_channel<btype> loadW[K];
  
  // first GCN layer
  std::cout << "Executing first GCN layer..." << std::endl;
  int part_K = K;
  int part_M = M;
  
  // write weights to appropriate channels to store correct in each PE
  for (int p=0; p < ver; p++) {
    if ((p+1)*K < I_F) {
      part_K = K;
    } else {
      part_K = K - ((p+1)*K - I_F);
    }
    
    for (int q=0; q < hor; q++) {
      if ((q+1)*M < O_F1) {
        part_M = M;
      } else {
        part_M = M - ((q+1)*M - O_F1);
      }      
      
      
      for (int i=0; i < K; i++) {
        if (i < part_K) {
          for (int j=0; j < M; j++) {
            if (j < part_M) {
              loadW[i].write(w1[(p*K)+i][(q*M)+j]);
            }
          }
        }
      }
    }
  }
  

  //initialize output
  for (int i=0; i < N; i++) {
    for (int j=0; j < O_F1; j++) {
      h_o[i][j] = (btype)0;
    }
  }

  bool change_row;
  bool last_layer = false;
  bool apply_activation;
  
  int lambda;
  btype a_lambda;

  btype in_row[I_B];
  btype out_row[O_B];

  // load weights to PEs
  sys_array.systolic_top(a_lambda, in_row, loadW, out_row, I_F, O_F1, ver, hor, true, true, false, false);

  // push different tiles in the systolic array
  ITER: for (int i=0; i < N; i++) {
    change_row = true;

    INTER_ROW_ITER: for (int j=a_row[i]; j < a_row[i+1]; j++) {
      lambda = a_col[j];
      a_lambda = a_val[j];
      
      for (int k=0; k < I_B; k++) {
        if (k < I_F)
          in_row[k] = h[lambda][k];
        else
          in_row[k] = (btype)0;
      }
      
      if(j==a_row[i+1]-1)
        apply_activation = true;

      sys_array.systolic_top(a_lambda, in_row, loadW, out_row, I_F, O_F1, ver, hor, false, change_row, last_layer, apply_activation);
      change_row = false;
      apply_activation = false;

    }
    
    WR_OUT: for (int j=0; j < O_F1; j++) {
      // write the output
      h_o[i][j] = out_row[j];
    }
    
  }

  print2d<N, O_F1> (h_o);
  std::cout << "===========================================" <<std::endl;

  // second GCN layer
  std::cout << "Executing second GCN layer..." << std::endl;
  
  // write weights to appropriate channels to store correct in each PE
  for (int p=0; p < ver2; p++) {
    if ((p+1)*K < O_F1) {
      part_K = K;
    } else {
      part_K = K - ((p+1)*K - O_F1);
    }
    
    for (int q=0; q < hor2; q++) {
      if ((q+1)*M < O_F2) {
        part_M = M;
      } else {
        part_M = M - ((q+1)*M - O_F2);
      }      
      
      
      for (int i=0; i < K; i++) {
        if (i < part_K) {
          for (int j=0; j < M; j++) {
            if (j < part_M) {
              loadW[i].write(w2[(p*K)+i][(q*M)+j]);
            }
          }
        }
      }
    }
  }
  

  //initialize output
  for (int i=0; i < N; i++) {
    for (int j=0; j < O_F2; j++) {
      final[i][j] = (btype)0;
    }
  }

  last_layer = true;

  // load weights to PEs
  sys_array.systolic_top(a_lambda, in_row, loadW, out_row, O_F1, O_F2, ver, hor, true, true, false, false);

  // push different tiles in the systolic array
  ITER_2: for (int i=0; i < N; i++) {
    change_row = true;

    INTER_ROW_ITER_2: for (int j=a_row[i]; j < a_row[i+1]; j++) {
      lambda = a_col[j];
      a_lambda = a_val[j];
      
      for (int k=0; k < I_B; k++) {
        if (k < O_F1)
          in_row[k] = h_o[lambda][k];
        else
          in_row[k] = (btype)0;
      }

      if(j==a_row[i+1]-1)
        apply_activation = true;

      sys_array.systolic_top(a_lambda, in_row, loadW, out_row, O_F1, O_F2, ver, hor, false, change_row, last_layer, apply_activation);
      change_row = false;
      apply_activation = false;

    }
    
    WR_OUT_2: for (int j=0; j < O_F2; j++) {
      // write the output
      final[i][j] = out_row[j];
    }
    
  }

  print2d<N, O_F2> (final);
  std::cout << "===========================================" <<std::endl;

  std::cout << "END!!!!" << std::endl;

  CCS_RETURN(0);
  
}
