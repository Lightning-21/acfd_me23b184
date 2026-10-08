#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include <vector>

enum class PrecondMethod
{
    Jacobi,
    GaussSeidel,
    SymmetricGaussSeidel,
    SOR,
    LU,
    Cholesky,
    ILU0,
    IC0
};

enum class PrecondMode
{
    Left,
    Right,
    Split
};

enum class PrecondSide
{
    M1,
    M2
};

//--------------------------------------------------
// Preconditioner
//
// Convention: the preconditioned operator is
// M1^-1 * A * M2^-1, with M = M1 * M2.
//
//   Left  : M1 = M,  M2 = I
//   Right : M1 = I,  M2 = M
//   Split : M1 * M2 = M (natural factor pair of M)
//
// Internally M = F1 * F2 (factor pair); M1 and M2 are
// assembled from it for the effective mode.
//--------------------------------------------------

struct Preconditioner
{
    PrecondMethod method = PrecondMethod::Jacobi;

    PrecondMode requestedMode = PrecondMode::Left;

    // Effective mode (Split may fall back to Left)
    PrecondMode mode = PrecondMode::Left;

    int n = 0;

    // SOR relaxation parameter
    double omega = 1.0;

    // Factor pair, M = F1 * F2
    std::vector<std::vector<double>> F1;
    std::vector<std::vector<double>> F2;

    // Output matrices for the effective mode
    std::vector<std::vector<double>> M1;
    std::vector<std::vector<double>> M2;
};

//--------------------------------------------------
// Dense Helpers
//
// matVec        : y = A x
// forwardSolve  : solves L z = r, L lower triangular
// backwardSolve : solves U z = r, U upper triangular
// diagonalSolve : solves D z = r, uses diag(D) only
//--------------------------------------------------

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

std::vector<std::vector<double>> transposeMatrix
(
    const std::vector<std::vector<double>>& A
);

std::vector<std::vector<double>> identityMatrix(int n);

//--------------------------------------------------
// Setup
//
// Each function returns a Preconditioner holding M1 and
// M2 for the requested mode.
//
// Plain Gauss-Seidel and SOR have no natural split, so
// Split falls back to Left. Jacobi and symmetric
// Gauss-Seidel split need a positive diagonal, otherwise
// they also fall back to Left.
//
//   Jacobi               : M = D
//   Gauss-Seidel         : M = D + L
//   Symmetric GS         : M = (D + L) D^-1 (D + U)
//   SOR                  : M = D/omega + L
//   LU                   : A = L U (no pivoting)
//   Cholesky             : A = L L^T
//   ILU(0)               : LU on the pattern of A
//   IC(0)                : Cholesky on the pattern of A
//--------------------------------------------------

Preconditioner setupJacobi
(
    const std::vector<std::vector<double>>& A,
    PrecondMode mode
);

Preconditioner setupGaussSeidel
(
    const std::vector<std::vector<double>>& A,
    PrecondMode mode
);

Preconditioner setupSymmetricGaussSeidel
(
    const std::vector<std::vector<double>>& A,
    PrecondMode mode
);

Preconditioner setupSOR
(
    const std::vector<std::vector<double>>& A,
    PrecondMode mode,
    double omega
);

Preconditioner setupLU
(
    const std::vector<std::vector<double>>& A,
    PrecondMode mode
);

Preconditioner setupCholesky
(
    const std::vector<std::vector<double>>& A,
    PrecondMode mode
);

Preconditioner setupILU0
(
    const std::vector<std::vector<double>>& A,
    PrecondMode mode
);

Preconditioner setupIC0
(
    const std::vector<std::vector<double>>& A,
    PrecondMode mode
);

//--------------------------------------------------
// Apply
//
// Computes z = M1^-1 r or z = M2^-1 r (selected by side)
// using triangular / diagonal solves only, never an
// explicit inverse.
//--------------------------------------------------

std::vector<double> applyJacobiPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const std::vector<double>& r
);

std::vector<double> applyGaussSeidelPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const std::vector<double>& r
);

std::vector<double> applySymmetricGaussSeidelPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const std::vector<double>& r
);

std::vector<double> applySORPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const std::vector<double>& r
);

std::vector<double> applyLUPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const std::vector<double>& r
);

std::vector<double> applyCholeskyPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const std::vector<double>& r
);

std::vector<double> applyILUPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const std::vector<double>& r
);

std::vector<double> applyICPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const std::vector<double>& r
);

//--------------------------------------------------
// Dispatchers
//
// The only entry points the Krylov solvers need.
//--------------------------------------------------

Preconditioner buildPreconditioner
(
    PrecondMethod method,
    PrecondMode mode,
    const std::vector<std::vector<double>>& A,
    double omega = 1.0
);

std::vector<double> applyPreconditioner
(
    const Preconditioner& P,
    PrecondSide side,
    const std::vector<double>& r
);

//--------------------------------------------------
// Validation
//
// Returns ||I - M1^-1 A M2^-1||_F / sqrt(n). About 0 for
// exact LU / Cholesky; smaller means a better
// approximation of A.
//--------------------------------------------------

double checkPreconditioner
(
    const std::vector<std::vector<double>>& A,
    const Preconditioner& P
);

#endif