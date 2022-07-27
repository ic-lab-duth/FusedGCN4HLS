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

  Matrix(int rows, int cols, T *data) : _rows(rows), _cols(cols), _data(data) {}

  T *operator[](int row) {
    return _data + (row * _cols);
  }

};

template <typename T>
class Array : public Matrix<T> {
public:

  Array(int _dim, T *data) : Matrix<T>(1, _dim, data) {}
  
};

#endif
