// Copyright 2022 Democritus University of Thrace
// Integrated Circuits Lab
// 
#include <iostream>
#include "defs.h"

template <typename Type, int N>
void print1d (Array<Type> &a) {
  for (int i=0; i < N; i++) {
    std::cout << a[0][i] << "\t";
  }
  std::cout << std::endl;
}


template <int N>
void print1d (btype a[N]) {
  for (int i=0; i < N; i++) {
    std::cout << a[i].to_float() << "\t";
  }
  std::cout << std::endl;
}


template <typename Type, int N, int L>
void print2d (Matrix<Type> &a) {
  for (int i=0; i < N; i++) {
    for (int j=0; j < L; j++) {
      std::cout << a[i][j] << "\t";
    }
    std::cout << std::endl;
  }
}

template <int N, int L>
void print2d (Matrix<btype> &a) {
  for (int i=0; i < N; i++) {
    for (int j=0; j < L; j++) {
      std::cout << a[i][j].to_float() << "\t";
    }
    std::cout << std::endl;
  }
}


template<typename Type, int N, int L>
void read_data(Matrix<Type> &img, std::string filename) {
  std::ifstream myFile(filename);
  if(myFile.is_open()) { //throw std::runtime_error("Could not open file");  // Make sure the file is open

    std::string line;
    Type val;
    int rowIdx = 0;

    // Read data, line by line
    while(std::getline(myFile, line)) {
      std::stringstream ss(line);  // Create a stringstream of the current line
      int colIdx = 0;  // Keep track of the current column index


      // Extract each integer
      while(ss >> val){
        img[rowIdx][colIdx] = val;  // Write current input value
        if(ss.peek() == ',') ss.ignore(); // If the next token is a comma, ignore it and move on

        colIdx++;  // Increment the Column index
        if (colIdx == L) break;
      }
      rowIdx++;  // Increment the Row index
      if (rowIdx == N) break;
    }
    myFile.close();  // Close file
  }
}


template<typename Type, int N, int nZ>
void read_adj(Array<int> &a_row, Array<int> &a_col, Array<Type> &a_val, std::string filename) {
  std::ifstream myFile(filename);
  if(myFile.is_open()) { //throw std::runtime_error("Could not open file");  // Make sure the file is open

    std::string line;
    int int_val;
    Type def_val;
    int rowIdx = 0;

    // Read the first line -- row_ind for CSR
    std::getline(myFile, line);
    std::stringstream ss1(line);  // Create a stringstream of the current line
    int colIdx = 0;  // Keep track of the current column index

    // Extract each integer
    while(ss1 >> int_val){
      a_row[colIdx][0] = int_val;
      if(ss1.peek() == ',') ss1.ignore(); // If the next token is a comma, ignore it and move on
      colIdx++;  // Increment the Column index
    }

    // Read the second line -- col_ind for CSR
    std::getline(myFile, line);
    std::stringstream ss2(line);  // Create a stringstream of the current line
    colIdx = 0;  // Keep track of the current column index

    // Extract each integer
    while(ss2 >> int_val){
      a_col[colIdx][0] = int_val;
      if(ss2.peek() == ',') ss2.ignore(); // If the next token is a comma, ignore it and move on
      colIdx++;  // Increment the Column index
    }

    // Read the third line -- value for CSR
    std::getline(myFile, line);
    std::stringstream ss3(line);  // Create a stringstream of the current line
    colIdx = 0;  // Keep track of the current column index

    // Extract each integer
    while(ss3 >> def_val){
      a_col[colIdx][0] = def_val;
      if(ss3.peek() == ',') ss3.ignore(); // If the next token is a comma, ignore it and move on
      colIdx++;  // Increment the Column index
    }

    myFile.close();  // Close file
  }
}
