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

  triple_sys< btype, N, I_F, O_F1, O_F2, nZ, K, M, I_B, O_B, MEM > gcn;

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
      a[row][col] = low + static_cast<float>(rand()) * static_cast<float>(high - low) / RAND_MAX;
      t++;
    }
  }

  encode_csr<btype, N, N, nZ>(a, a_row, a_col, a_val);
  
  for (int i=0; i < N+1; i++) {
    a_row[i] = i;
  }
  
  for (int i=0; i < nZ; i++) {
    a_col[i] = i;
    a_val[i] = (float)1;
  }
  
  for (int i=0; i < N; i++) {
    for (int j=0; j < I_F; j++) {
      h[i][j] = low + static_cast<float>(rand()) * static_cast<float>(high - low) / RAND_MAX;
    }
  }
  
  for (int i=0; i < I_F; i++) {
    for (int j=0; j < O_F1; j++) {
      w1[i][j] = low + static_cast<float>(rand()) * static_cast<float>(high - low) / RAND_MAX;
    }
  }

  for (int i=0; i < O_F1; i++) {
    for (int j=0; j < O_F2; j++) {
      w2[i][j] = low + static_cast<float>(rand()) * static_cast<float>(high - low) / RAND_MAX;
    }
  }

  btype final[N][O_F2];

  ac_channel<btype> loadW1[K];
  
  // first GCN layer
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
              loadW1[i].write(w1[(p*K)+i][(q*M)+j]);
            }
          }
        }
      }
    }
  }
  
  // second GCN layer
  ac_channel<btype> loadW2[K];
  
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
              loadW2[i].write(w2[(p*K)+i][(q*M)+j]);
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

  gcn.run(a_row, a_col, a_val, h, loadW1, loadW2, final);

  std::cout << "===========================================" <<std::endl;
  print2d<N, O_F2> (final);
  std::cout << "===========================================" <<std::endl;

  std::cout << "END!!!!" << std::endl;

  CCS_RETURN(0);
  
}
