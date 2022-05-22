// Copyright 2022 Democritus University of Thrace
// Integrated Circuits Lab
// 
#ifndef _MATRIX_H
#define _MATRIX_H

template <typename T>
class Matrix {
private:
  unsigned _rows, _cols;
  T *_data;

public:

  #ifdef __SCVERIFY__
  Matrix(int rows, int cols) : _rows(rows), _cols(cols), _data(new T[rows * cols]) {}

  ~Matrix() {
    delete[] _data;
  }
  #else
  Matrix(int rows, int cols, T *data) : _rows(rows), _cols(cols), _data(data) {}
  #endif

  T *operator[](int row) {
    return _data + (row * _cols);
  }

  unsigned rows() const {
    return _rows;
  }

  unsigned cols() const {
    return _cols;
  }
};

template <typename T>
class Array : public Matrix<T> {
public:
  #ifdef __SCVERIFY__
  Array(int _dim) : Matrix<T>(1, _dim) {}
  #else
  Array(int _dim, T *data) : Matrix<T>(1, _dim, data) {}
  #endif
};
