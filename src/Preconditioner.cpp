//------------------------------------------------------------//
// File: Preconditioner.cpp
//
// Description:
//
// Dense preconditioners for Krylov solvers: Jacobi,
// Gauss-Seidel, symmetric Gauss-Seidel, SOR, LU, Cholesky,
// ILU(0) and IC(0), each usable as a Left, Right or Split
// preconditioner.
//
// Every setup function builds a factor pair M = F1 * F2 and
// assembles M1, M2 for the effective mode. Every apply
// function applies M1^-1 or M2^-1 with diagonal or
// triangular solves only (no explicit inverse).
//
// ILU(0) and IC(0) keep the nonzero pattern of A, so no fill
// is created. LU is unpivoted: a zero pivot throws.
//
//------------------------------------------------------------//

#include "Preconditioner.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

using namespace std;

//------------------------------------------------------------//
// File-Local Helpers
//------------------------------------------------------------//

// Guards against division by zero only
static const double kTiny = 1e-300;

static vector<vector<double>> zeros(int n)
{
    return vector<vector<double>>(n, vector<double>(n, 0.0));
}

static void requireSquare(const vector<vector<double>>& A)
{
    for(const auto& row : A)
    {
        if(row.size() != A.size())
        {
            throw invalid_argument("matrix must be square");
        }
    }
}

static vector<vector<double>> matMul
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

static vector<double> diagonalOf(const vector<vector<double>>& A)
{
    vector<double> d(A.size());

    for(size_t i = 0; i < A.size(); i++)
    {
        d[i] = A[i][i];

        if(fabs(d[i]) < kTiny)
        {
            throw runtime_error("zero diagonal entry");
        }
    }

    return d;
}

static bool allPositive(const vector<double>& d)
{
    for(double v : d)
    {
        if(v <= 0.0)
        {
            return false;
        }
    }

    return true;
}

static void requireSize(const Preconditioner& P, const vector<double>& r)
{
    if(static_cast<int>(r.size()) != P.n)
    {
        throw invalid_argument("vector size does not match preconditioner");
    }
}

static PrecondMode noSplit(PrecondMode mode)
{
    return (mode == PrecondMode::Split) ? PrecondMode::Left : mode;
}

static Preconditioner makePrecond
(
    PrecondMethod method,
    PrecondMode mode,
    const vector<vector<double>>& A,
    double omega
)
{
    requireSquare(A);

    Preconditioner P;

    P.method = method;
    P.requestedMode = mode;
    P.mode = mode;
    P.n = static_cast<int>(A.size());
    P.omega = omega;

    return P;
}

// Assemble M1, M2 from the factor pair for the effective mode
static void finalize(Preconditioner& P)
{
    switch(P.mode)
    {
        case PrecondMode::Left:
            P.M1 = matMul(P.F1, P.F2);
            P.M2 = identityMatrix(P.n);
            break;

        case PrecondMode::Right:
            P.M1 = identityMatrix(P.n);
            P.M2 = matMul(P.F1, P.F2);
            break;

        case PrecondMode::Split:
            P.M1 = P.F1;
            P.M2 = P.F2;
            break;
    }
}

// Which factors the requested side applies.
// M^-1 r = F2^-1 F1^-1 r, so F1 goes first.
static void activeFactors
(
    PrecondMode mode,
    PrecondSide side,
    bool& useF1,
    bool& useF2
)
{
    bool m1 = (side == PrecondSide::M1);

    useF1 = false;
    useF2 = false;

    switch(mode)
    {
        case PrecondMode::Left:
            useF1 = m1;
            useF2 = m1;
            break;

        case PrecondMode::Right:
            useF1 = !m1;
            useF2 = !m1;
            break;

        case PrecondMode::Split:
            useF1 = m1;
            useF2 = !m1;
            break;
    }
}

// A = L U with unit-diagonal L. usePattern = true restricts
// updates to the nonzero pattern of A (ILU(0)).
static void luFactor
(
    const vector<vector<double>>& A,
    bool usePattern,
    vector<vector<double>>& L,
    vector<vector<double>>& U
)
{
    int n = static_cast<int>(A.size());

    vector<vector<double>> W = A;

    for(int i = 0; i < n; i++)
    {
        for(int k = 0; k < i; k++)
        {
            if(usePattern && A[i][k] == 0.0)
            {
                continue;
            }

            W[i][k] /= W[k][k];

            for(int j = k+1; j < n; j++)
            {
                if(usePattern && A[i][j] == 0.0)
                {
                    continue;
                }

                W[i][j] -= W[i][k]*W[k][j];
            }
        }

        if(fabs(W[i][i]) < kTiny)
        {
            throw runtime_error("zero pivot in LU factorisation");
        }
    }

    L = identityMatrix(n);
    U = zeros(n);

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(j < i)
            {
                L[i][j] = W[i][j];
            }
            else
            {
                U[i][j] = W[i][j];
            }
        }
    }
}

// A = L L^T, using the lower part of A. usePattern = true
// drops fill outside the lower pattern of A (IC(0)).
static vector<vector<double>> choleskyFactor
(
    const vector<vector<double>>& A,
    bool usePattern
)
{
    int n = static_cast<int>(A.size());

    vector<vector<double>> L = zeros(n);

    for(int j = 0; j < n; j++)
    {
        double s = A[j][j];

        for(int k = 0; k < j; k++)
        {
            s -= L[j][k]*L[j][k];
        }

        if(s <= 0.0)
        {
            throw runtime_error("non-positive pivot: matrix is not SPD");
        }

        L[j][j] = sqrt(s);

        for(int i = j+1; i < n; i++)
        {
            if(usePattern && A[i][j] == 0.0)
            {
                continue;
            }

            double t = A[i][j];

            for(int k = 0; k < j; k++)
            {
                t -= L[i][k]*L[j][k];
            }

            L[i][j] = t/L[j][j];
        }
    }

    return L;
}

// Shared by every method whose factors are (lower, upper):
// forward solve with F1, backward solve with F2.
static vector<double> applyLowerUpper
(
    const Preconditioner& P,
    PrecondSide side,
    const vector<double>& r
)
{
    requireSize(P, r);

    bool useF1;
    bool useF2;

    activeFactors(P.mode, side, useF1, useF2);

    vector<double> z = r;

    if(useF1)
    {
        z = forwardSolve(P.F1, z);
    }

    if(useF2)
    {
        z = backwardSolve(P.F2, z);
    }

    return z;
}

//------------------------------------------------------------//
// Dense Helpers
//------------------------------------------------------------//

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

vector<vector<double>> identityMatrix(int n)
{
    vector<vector<double>> I = zeros(n);

    for(int i = 0; i < n; i++)
    {
        I[i][i] = 1.0;
    }

    return I;
}

//------------------------------------------------------------//
// Setup
//------------------------------------------------------------//

Preconditioner setupJacobi
(
    const vector<vector<double>>& A,
    PrecondMode mode
)
{
    Preconditioner P = makePrecond(PrecondMethod::Jacobi, mode, A, 1.0);

    vector<double> d = diagonalOf(A);

    if(P.mode == PrecondMode::Split && !allPositive(d))
    {
        P.mode = PrecondMode::Left;
    }

    bool split = (P.mode == PrecondMode::Split);

    P.F1 = zeros(P.n);
    P.F2 = identityMatrix(P.n);

    for(int i = 0; i < P.n; i++)
    {
        P.F1[i][i] = split ? sqrt(d[i]) : d[i];

        if(split)
        {
            P.F2[i][i] = P.F1[i][i];
        }
    }

    finalize(P);

    return P;
}

Preconditioner setupGaussSeidel
(
    const vector<vector<double>>& A,
    PrecondMode mode
)
{
    Preconditioner P = makePrecond(PrecondMethod::GaussSeidel, mode, A, 1.0);

    P.mode = noSplit(mode);

    // Zero-diagonal guard
    diagonalOf(A);

    // M = D + L
    P.F1 = zeros(P.n);

    for(int i = 0; i < P.n; i++)
    {
        for(int j = 0; j <= i; j++)
        {
            P.F1[i][j] = A[i][j];
        }
    }

    P.F2 = identityMatrix(P.n);

    finalize(P);

    return P;
}

Preconditioner setupSymmetricGaussSeidel
(
    const vector<vector<double>>& A,
    PrecondMode mode
)
{
    Preconditioner P = makePrecond(PrecondMethod::SymmetricGaussSeidel, mode, A, 1.0);

    vector<double> d = diagonalOf(A);

    if(P.mode == PrecondMode::Split && !allPositive(d))
    {
        P.mode = PrecondMode::Left;
    }

    bool split = (P.mode == PrecondMode::Split);

    P.F1 = zeros(P.n);
    P.F2 = zeros(P.n);

    for(int i = 0; i < P.n; i++)
    {
        // F1 = (D+L) D^-1, or (D+L) D^-1/2 when split
        for(int j = 0; j <= i; j++)
        {
            P.F1[i][j] = A[i][j]/(split ? sqrt(d[j]) : d[j]);
        }

        // F2 = (D+U), or D^-1/2 (D+U) when split
        for(int j = i; j < P.n; j++)
        {
            P.F2[i][j] = split ? A[i][j]/sqrt(d[i]) : A[i][j];
        }
    }

    finalize(P);

    return P;
}

Preconditioner setupSOR
(
    const vector<vector<double>>& A,
    PrecondMode mode,
    double omega
)
{
    if(omega <= 0.0)
    {
        throw invalid_argument("SOR omega must be positive");
    }

    Preconditioner P = makePrecond(PrecondMethod::SOR, mode, A, omega);

    P.mode = noSplit(mode);

    diagonalOf(A);

    // M = D/omega + L
    P.F1 = zeros(P.n);

    for(int i = 0; i < P.n; i++)
    {
        for(int j = 0; j < i; j++)
        {
            P.F1[i][j] = A[i][j];
        }

        P.F1[i][i] = A[i][i]/omega;
    }

    P.F2 = identityMatrix(P.n);

    finalize(P);

    return P;
}

Preconditioner setupLU
(
    const vector<vector<double>>& A,
    PrecondMode mode
)
{
    Preconditioner P = makePrecond(PrecondMethod::LU, mode, A, 1.0);

    luFactor(A, false, P.F1, P.F2);

    finalize(P);

    return P;
}

Preconditioner setupCholesky
(
    const vector<vector<double>>& A,
    PrecondMode mode
)
{
    Preconditioner P = makePrecond(PrecondMethod::Cholesky, mode, A, 1.0);

    P.F1 = choleskyFactor(A, false);
    P.F2 = transposeMatrix(P.F1);

    finalize(P);

    return P;
}

Preconditioner setupILU0
(
    const vector<vector<double>>& A,
    PrecondMode mode
)
{
    Preconditioner P = makePrecond(PrecondMethod::ILU0, mode, A, 1.0);

    luFactor(A, true, P.F1, P.F2);

    finalize(P);

    return P;
}

Preconditioner setupIC0
(
    const vector<vector<double>>& A,
    PrecondMode mode
)
{
    Preconditioner P = makePrecond(PrecondMethod::IC0, mode, A, 1.0);

    P.F1 = choleskyFactor(A, true);
    P.F2 = transposeMatrix(P.F1);

    finalize(P);

    return P;
}

//------------------------------------------------------------//
// Apply
//------------------------------------------------------------//

vector<double> applyJacobiPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const vector<double>& r
)
{
    requireSize(P, r);

    bool useF1;
    bool useF2;

    activeFactors(P.mode, side, useF1, useF2);

    vector<double> z = r;

    if(useF1)
    {
        z = diagonalSolve(P.F1, z);
    }

    if(useF2)
    {
        z = diagonalSolve(P.F2, z);
    }

    return z;
}

// F2 is the identity for Gauss-Seidel and SOR, so only F1 is
// ever applied.
vector<double> applyGaussSeidelPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const vector<double>& r
)
{
    requireSize(P, r);

    bool useF1;
    bool useF2;

    activeFactors(P.mode, side, useF1, useF2);

    return useF1 ? forwardSolve(P.F1, r) : r;
}

vector<double> applySORPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const vector<double>& r
)
{
    return applyGaussSeidelPreconditioner(P, side, r);
}

vector<double> applySymmetricGaussSeidelPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const vector<double>& r
)
{
    return applyLowerUpper(P, side, r);
}

vector<double> applyLUPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const vector<double>& r
)
{
    return applyLowerUpper(P, side, r);
}

vector<double> applyCholeskyPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const vector<double>& r
)
{
    return applyLowerUpper(P, side, r);
}

vector<double> applyILUPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const vector<double>& r
)
{
    return applyLowerUpper(P, side, r);
}

vector<double> applyICPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const vector<double>& r
)
{
    return applyLowerUpper(P, side, r);
}

//------------------------------------------------------------//
// Dispatchers
//------------------------------------------------------------//

Preconditioner buildPreconditioner
(
    PrecondMethod method,
    PrecondMode mode,
    const vector<vector<double>>& A,
    double omega
)
{
    switch(method)
    {
        case PrecondMethod::Jacobi:
            return setupJacobi(A, mode);

        case PrecondMethod::GaussSeidel:
            return setupGaussSeidel(A, mode);

        case PrecondMethod::SymmetricGaussSeidel:
            return setupSymmetricGaussSeidel(A, mode);

        case PrecondMethod::SOR:
            return setupSOR(A, mode, omega);

        case PrecondMethod::LU:
            return setupLU(A, mode);

        case PrecondMethod::Cholesky:
            return setupCholesky(A, mode);

        case PrecondMethod::ILU0:
            return setupILU0(A, mode);

        case PrecondMethod::IC0:
            return setupIC0(A, mode);
    }

    throw invalid_argument("unknown preconditioner method");
}

vector<double> applyPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const vector<double>& r
)
{
    switch(P.method)
    {
        case PrecondMethod::Jacobi:
            return applyJacobiPreconditioner(P, side, r);

        case PrecondMethod::GaussSeidel:
            return applyGaussSeidelPreconditioner(P, side, r);

        case PrecondMethod::SymmetricGaussSeidel:
            return applySymmetricGaussSeidelPreconditioner(P, side, r);

        case PrecondMethod::SOR:
            return applySORPreconditioner(P, side, r);

        case PrecondMethod::LU:
            return applyLUPreconditioner(P, side, r);

        case PrecondMethod::Cholesky:
            return applyCholeskyPreconditioner(P, side, r);

        case PrecondMethod::ILU0:
            return applyILUPreconditioner(P, side, r);

        case PrecondMethod::IC0:
            return applyICPreconditioner(P, side, r);
    }

    throw invalid_argument("unknown preconditioner method");
}

//------------------------------------------------------------//
// Validation
//------------------------------------------------------------//

double checkPreconditioner
(
    const vector<vector<double>>& A,
    const Preconditioner& P
)
{
    if(static_cast<int>(A.size()) != P.n || P.n == 0)
    {
        throw invalid_argument("checkPreconditioner: size mismatch");
    }

    double s = 0.0;

    for(int j = 0; j < P.n; j++)
    {
        vector<double> e(P.n, 0.0);

        e[j] = 1.0;

        vector<double> y = applyPreconditioner(P, PrecondSide::M2, e);

        vector<double> z = applyPreconditioner(P, PrecondSide::M1, matVec(A, y));

        for(int i = 0; i < P.n; i++)
        {
            s += (z[i]-e[i])*(z[i]-e[i]);
        }
    }

    return sqrt(s/P.n);
}