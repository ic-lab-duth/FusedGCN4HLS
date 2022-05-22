// Copyright 2022 Democritus University of Thrace
// Integrated Circuits Lab
// 
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <fstream>
#include <istream>
#include <sstream>
#include <string>


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
  
#ifdef __SCVERIFY__
  Matrix<btype> final(N, O_F2);
  Matrix<btype> h(N, I_F);
  Matrix<float> H(N, I_F);
  Matrix<float> w1(I_F, O_F1);
  Matrix<float> w2(O_F1, O_F2);
  Array< int > A_row(N+1);
  Array< int > A_col(nZ);
  Array< btype > A_val(nZ);
  Array< ac_int<32, false> > a_row(N+1);
  Array< ac_int<32, false> > a_col(nZ);
  Array< btype > a_val(nZ);
#else
  btype final2[N*O_F2];
  btype h2[N*I_F];
  float H_[N*I_F];
  float w1_[I_F*O_F1];
  float w2_[O_F1*O_F2];
  int A_row_[N+1];
  int A_col_[nZ];
  btype A_val_[nZ];
  ac_int<32, false> a_row2[N+1];
  ac_int<32, false> a_col2[nZ];
  btype a_val2[nZ];

  Matrix<float> H(N, I_F, H_);
  Matrix<float> w1(I_F, O_F1, w1_);
  Matrix<float> w2(O_F1, O_F2, w2_);
  Array< int > A_row(N+1, A_row_);
  Array< int > A_col(nZ, A_col_);
  Array< btype > A_val(nZ, A_val_);
#endif

  ac_channel<btype> loadW1[K];
  ac_channel<btype> loadW2[K];


  // read input matrices from txt files
  read_adj<float, N, nZ>(A_row, A_col, A_val, "../matrices/citeseer_adj.txt");
  read_data<float, N, I_F>(H, "../matrices/citeseer_feat.txt");
  read_data<float, I_F, O_F1>(w1, "../matrices/citeseer_weights.txt");
  read_data<float, O_F1, O_F2>(w2, "../matrices/citeseer_weights2.txt");


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

#ifdef __SCVERIFY__
  //initialize output
  for (int i=0; i < N; i++) {
    for (int j=0; j < O_F2; j++) {
      final[i][j] = (btype)0;
    }
  }
#else
  for (int i=0; i < N; i++) {
    for (int j=0; j < O_F2; j++) {
      final2[i*O_F2 + j] = (btype)0;
    }
  }
#endif

#ifdef __SCVERIFY__
  for (int i=0; i < N+1; i++) {
    a_row[0][i] = A_row[0][i];
  }

  for (int i=0; i < nZ; i++) {
    a_col[0][i] = A_col[0][i];
    a_val[0][i] = (btype)A_val[0][i];
  }
#else
  for (int i=0; i < N+1; i++) {
    a_row2[i] = A_row[0][i];
  }

  for (int i=0; i < nZ; i++) {
    a_col2[i] = A_col[0][i];
    a_val2[i] = (btype)A_val[0][i];
  }
#endif

#ifdef __SCVERIFY__
  for (int i=0; i < N; i++) {
    for (int j=0; j < I_F; j++) {
      h[i][j] = H[i][j];
    }
  }
#else
  for (int i=0; i < N; i++) {
    for (int j=0; j < I_F; j++) {
      h2[i*I_F + j] = H[i][j];
    }
  }
#endif

#ifdef __SCVERIFY__
  gcn.run(a_row, a_col, a_val, h, loadW1, loadW2, final);
#else
  gcn.run(a_row2, a_col2, a_val2, h2, loadW1, loadW2, final2);
#endif

  std::cout << "==========================" << std::endl;
  std::cout << "END!!!!" << std::endl;

  CCS_RETURN(0);
  
}
