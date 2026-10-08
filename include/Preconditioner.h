#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;   // dense, row-major

enum class PrecondMethod { Jacobi, GaussSeidel, SymmetricGaussSeidel, SOR, LU, Cholesky, ILU0, IC0 };
enum class PrecondMode   { Left, Right, Split };
enum class PrecondSide   { M1, M2 };

// Convention: preconditioned operator = M1^-1 * A * M2^-1,  M = M1 * M2
//   Left : M1 = M,  M2 = I
//   Right: M1 = I,  M2 = M
//   Split: M1 * M2 = M  (natural factor pair of M)
// Internally M = F1 * F2 (factor pair); M1/M2 are assembled from it for the effective mode.
struct Preconditioner {
    PrecondMethod method        = PrecondMethod::Jacobi;
    PrecondMode   requestedMode = PrecondMode::Left;
    PrecondMode   mode          = PrecondMode::Left;   // effective mode (Split may fall back to Left)
    int    n     = 0;
    double omega = 1.0;                                // SOR parameter
    Matrix F1, F2;                                     // factor pair, M = F1 * F2
    Matrix M1, M2;                                     // output matrices for the effective mode
};

// ---- Dense helpers ----
Vector matVec(const Matrix& A, const Vector& x);           // y = A x
Vector forwardSolve(const Matrix& L, const Vector& r);     // solve L z = r, L lower triangular
Vector backwardSolve(const Matrix& U, const Vector& r);    // solve U z = r, U upper triangular
Vector diagonalSolve(const Matrix& D, const Vector& r);    // solve D z = r, uses diag(D) only
Matrix transposeMatrix(const Matrix& A);
Matrix identityMatrix(int n);

// ---- Setup: each returns a Preconditioner holding M1, M2 for the requested mode ----
// Plain Gauss-Seidel and SOR have no natural split, so Split falls back to Left.
// Jacobi and SGS split need a positive diagonal, otherwise they also fall back to Left.
Preconditioner setupJacobi(const Matrix& A, PrecondMode mode);                 // M = D
Preconditioner setupGaussSeidel(const Matrix& A, PrecondMode mode);            // M = D + L
Preconditioner setupSymmetricGaussSeidel(const Matrix& A, PrecondMode mode);   // M = (D+L) D^-1 (D+U)
Preconditioner setupSOR(const Matrix& A, PrecondMode mode, double omega);      // M = D/omega + L
Preconditioner setupLU(const Matrix& A, PrecondMode mode);                     // A = L U (no pivoting)
Preconditioner setupCholesky(const Matrix& A, PrecondMode mode);               // A = L L^T
Preconditioner setupILU0(const Matrix& A, PrecondMode mode);                   // LU on A's pattern
Preconditioner setupIC0(const Matrix& A, PrecondMode mode);                    // Cholesky on A's pattern

// ---- Apply: z = M1^-1 r or z = M2^-1 r (by triangular/diagonal solves, no explicit inverse) ----
Vector applyJacobiPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r);
Vector applyGaussSeidelPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r);
Vector applySymmetricGaussSeidelPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r);
Vector applySORPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r);
Vector applyLUPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r);
Vector applyCholeskyPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r);
Vector applyILUPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r);
Vector applyICPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r);

// ---- Dispatchers: the only entry points the Krylov solvers need ----
Preconditioner buildPreconditioner(PrecondMethod method, PrecondMode mode, const Matrix& A, double omega = 1.0);
Vector applyPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r);

// ---- Validation ----
// Returns ||I - M1^-1 A M2^-1||_F / sqrt(n). ~0 for exact LU/Cholesky; smaller = better approximation.
double checkPreconditioner(const Matrix& A, const Preconditioner& P);

#endif