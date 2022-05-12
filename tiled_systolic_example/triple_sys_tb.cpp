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
  

  btype h_o[N][O_F1];
  btype final[N][O_F2];

  ac_channel<btype> loadW[K];
  
  std::vector<int> a_row;
  std::vector<int> a_col;
  std::vector<float> a_val;
  std::vector< std::vector<float> > h;
  std::vector< std::vector<float> > w1;
  std::vector< std::vector<float> > w2;

  // read input matrices from txt files
  read_adj<float, N, nZ>(a_row, a_col, a_val, "../matrices/citeseer_adj.txt");
  read_data<float, N, I_F>(h, "../matrices/citeseer_feat.txt");
  read_data<float, I_F, O_F1>(w1, "../matrices/citeseer_weights.txt");
  read_data<float, O_F1, O_F2>(w2, "../matrices/citeseer_weights2.txt");


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

  std::cout << "===========================================" <<std::endl;

  std::cout << "END!!!!" << std::endl;

  CCS_RETURN(0);
  
}
