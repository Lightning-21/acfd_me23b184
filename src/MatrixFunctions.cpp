//------------------------------------------------------------//
// File: MatrixFunctions.cpp
//
// Description:
//
// General dense matrix and vector helpers shared by the
// preconditioners and the Krylov solvers: dot product, 2-norm,
// axpy, zero / identity / transpose, matrix-matrix and
// matrix-vector products, and diagonal / triangular solves.
//
//------------------------------------------------------------//

#include "MatrixFunctions.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

using namespace std;

//------------------------------------------------------------//
// Vector Helpers
//------------------------------------------------------------//

double dot(const vector<double>& a, const vector<double>& b)
{
    double s = 0.0;

    for(size_t i = 0; i < a.size(); i++)
    {
        s += a[i]*b[i];
    }

    return s;
}

double norm2(const vector<double>& a)
{
    return sqrt(dot(a, a));
}

// Returns y + a*x
vector<double> axpy
(
    double a,
    const vector<double>& x,
    const vector<double>& y
)
{
    vector<double> z(y);

    for(size_t i = 0; i < z.size(); i++)
    {
        z[i] += a*x[i];
    }

    return z;
}

//------------------------------------------------------------//
// Dense Helpers
//------------------------------------------------------------//

vector<vector<double>> zeros(int n)
{
    return vector<vector<double>>(n, vector<double>(n, 0.0));
}

vector<vector<double>> identityMatrix(int n)
{
    vector<vector<double>> I = zeros(n);

    for(int i = 0; i < n; i++)
    {
        I[i][i] = 1.0;
    }

    return I;
}

vector<vector<double>> transposeMatrix
(
    const vector<vector<double>>& A
)
{
    int n = static_cast<int>(A.size());

    vector<vector<double>> T = zeros(n);

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            T[j][i] = A[i][j];
        }
    }

    return T;
}

vector<vector<double>> matMul
(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B
)
{
    int n = static_cast<int>(A.size());

    vector<vector<double>> C = zeros(n);

    for(int i = 0; i < n; i++)
    {
        for(int k = 0; k < n; k++)
        {
            if(A[i][k] != 0.0)
            {
                for(int j = 0; j < n; j++)
                {
                    C[i][j] += A[i][k]*B[k][j];
                }
            }
        }
    }

    return C;
}

vector<double> matVec
(
    const vector<vector<double>>& A,
    const vector<double>& x
)
{
    for(const auto& row : A)
    {
        if(row.size() != x.size())
        {
            throw invalid_argument("matVec: size mismatch");
        }
    }

    vector<double> y(A.size(), 0.0);

    for(size_t i = 0; i < A.size(); i++)
    {
        for(size_t j = 0; j < x.size(); j++)
        {
            y[i] += A[i][j]*x[j];
        }
    }

    return y;
}

vector<double> forwardSolve
(
    const vector<vector<double>>& L,
    const vector<double>& r
)
{
    int n = static_cast<int>(r.size());

    vector<double> z(n);

    for(int i = 0; i < n; i++)
    {
        double s = r[i];

        for(int j = 0; j < i; j++)
        {
            s -= L[i][j]*z[j];
        }

        z[i] = s/L[i][i];
    }

    return z;
}

vector<double> backwardSolve
(
    const vector<vector<double>>& U,
    const vector<double>& r
)
{
    int n = static_cast<int>(r.size());

    vector<double> z(n);

    for(int i = n-1; i >= 0; i--)
    {
        double s = r[i];

        for(int j = i+1; j < n; j++)
        {
            s -= U[i][j]*z[j];
        }

        z[i] = s/U[i][i];
    }

    return z;
}

vector<double> diagonalSolve
(
    const vector<vector<double>>& D,
    const vector<double>& r
)
{
    vector<double> z(r.size());

    for(size_t i = 0; i < r.size(); i++)
    {
        z[i] = r[i]/D[i][i];
    }

    return z;
}