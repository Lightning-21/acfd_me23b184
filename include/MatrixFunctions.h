#ifndef MATRIXFUNCTIONS_H
#define MATRIXFUNCTIONS_H

#include <vector>

//--------------------------------------------------
// Vector Helpers
//
// dot   : a . b
// norm2 : Euclidean norm of a
// axpy  : returns y + a*x
//--------------------------------------------------

double dot
(
    const std::vector<double>& a,
    const std::vector<double>& b
);

double norm2(const std::vector<double>& a);

std::vector<double> axpy
(
    double a,
    const std::vector<double>& x,
    const std::vector<double>& y
);

//--------------------------------------------------
// Dense Helpers
//
// zeros          : n x n zero matrix
// identityMatrix : n x n identity matrix
// transposeMatrix: A^T
// matMul         : C = A B (square matrices)
// matVec         : y = A x
// forwardSolve   : solves L z = r, L lower triangular
// backwardSolve  : solves U z = r, U upper triangular
// diagonalSolve  : solves D z = r, uses diag(D) only
// requireSquare  : throws std::invalid_argument unless every
//                  row of A has A.size() entries
//--------------------------------------------------

std::vector<std::vector<double>> zeros(int n);

std::vector<std::vector<double>> identityMatrix(int n);

std::vector<std::vector<double>> transposeMatrix
(
    const std::vector<std::vector<double>>& A
);

std::vector<std::vector<double>> matMul
(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B
);

std::vector<double> matVec
(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& x
);

std::vector<double> forwardSolve
(
    const std::vector<std::vector<double>>& L,
    const std::vector<double>& r
);

std::vector<double> backwardSolve
(
    const std::vector<std::vector<double>>& U,
    const std::vector<double>& r
);

std::vector<double> diagonalSolve
(
    const std::vector<std::vector<double>>& D,
    const std::vector<double>& r
);

void requireSquare(const std::vector<std::vector<double>>& A);

//--------------------------------------------------
// Cholesky Factor
//
// choleskyFactor returns the lower triangle L with
// A = L L^T, reading only the lower part of A.
//
// usePattern = true drops fill outside the lower
// pattern of A (IC(0)).
//
// Throws std::runtime_error at the first pivot that is
// not strictly positive (or is NaN), i.e. when A is not
// SPD.
//--------------------------------------------------

std::vector<std::vector<double>> choleskyFactor
(
    const std::vector<std::vector<double>>& A,
    bool usePattern
);

#endif