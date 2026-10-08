#include "Preconditioner.h"

#include <cmath>
#include <stdexcept>

namespace {

const double kTiny = 1e-300;   // guards against division by zero only

Matrix zeros(int n) { return Matrix(n, Vector(n, 0.0)); }

void requireSquare(const Matrix& A) {
    for (const auto& row : A)
        if (row.size() != A.size()) throw std::invalid_argument("matrix must be square");
}

Matrix matMul(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C = zeros(n);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            if (A[i][k] != 0.0)
                for (int j = 0; j < n; ++j) C[i][j] += A[i][k] * B[k][j];
    return C;
}

Vector diagonalOf(const Matrix& A) {
    Vector d(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
        d[i] = A[i][i];
        if (std::abs(d[i]) < kTiny) throw std::runtime_error("zero diagonal entry");
    }
    return d;
}

bool allPositive(const Vector& d) {
    for (double v : d) if (v <= 0.0) return false;
    return true;
}

PrecondMode noSplit(PrecondMode m) { return m == PrecondMode::Split ? PrecondMode::Left : m; }

Preconditioner makePrecond(PrecondMethod method, PrecondMode mode, const Matrix& A, double omega) {
    requireSquare(A);
    Preconditioner P;
    P.method = method;
    P.requestedMode = P.mode = mode;
    P.n = A.size();
    P.omega = omega;
    return P;
}

// Assemble M1, M2 from the factor pair for the effective mode.
void finalize(Preconditioner& P) {
    switch (P.mode) {
        case PrecondMode::Left:  P.M1 = matMul(P.F1, P.F2); P.M2 = identityMatrix(P.n); break;
        case PrecondMode::Right: P.M1 = identityMatrix(P.n); P.M2 = matMul(P.F1, P.F2); break;
        case PrecondMode::Split: P.M1 = P.F1; P.M2 = P.F2; break;
    }
}

// Which factors the requested side applies (M^-1 r = F2^-1 F1^-1 r, so F1 goes first).
void activeFactors(PrecondMode mode, PrecondSide side, bool& useF1, bool& useF2) {
    bool m1 = (side == PrecondSide::M1);
    useF1 = useF2 = false;
    switch (mode) {
        case PrecondMode::Left:  useF1 = useF2 = m1;  break;
        case PrecondMode::Right: useF1 = useF2 = !m1; break;
        case PrecondMode::Split: useF1 = m1; useF2 = !m1; break;
    }
}

// A = L U with unit-diagonal L; usePattern = true restricts updates to A's nonzero pattern (ILU(0)).
void luFactor(const Matrix& A, bool usePattern, Matrix& L, Matrix& U) {
    int n = A.size();
    Matrix W = A;
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < i; ++k) {
            if (usePattern && A[i][k] == 0.0) continue;
            W[i][k] /= W[k][k];
            for (int j = k + 1; j < n; ++j) {
                if (usePattern && A[i][j] == 0.0) continue;
                W[i][j] -= W[i][k] * W[k][j];
            }
        }
        if (std::abs(W[i][i]) < kTiny) throw std::runtime_error("zero pivot in LU factorisation");
    }
    L = identityMatrix(n);
    U = zeros(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) (j < i ? L[i][j] : U[i][j]) = W[i][j];
}

// A = L L^T; usePattern = true drops fill outside A's lower pattern (IC(0)). Uses the lower part of A.
Matrix choleskyFactor(const Matrix& A, bool usePattern) {
    int n = A.size();
    Matrix L = zeros(n);
    for (int j = 0; j < n; ++j) {
        double s = A[j][j];
        for (int k = 0; k < j; ++k) s -= L[j][k] * L[j][k];
        if (s <= 0.0) throw std::runtime_error("non-positive pivot: matrix is not SPD");
        L[j][j] = std::sqrt(s);
        for (int i = j + 1; i < n; ++i) {
            if (usePattern && A[i][j] == 0.0) continue;
            double t = A[i][j];
            for (int k = 0; k < j; ++k) t -= L[i][k] * L[j][k];
            L[i][j] = t / L[j][j];
        }
    }
    return L;
}

// Shared by all methods whose factors are (lower, upper): forward solve with F1, backward with F2.
Vector applyLowerUpper(const Preconditioner& P, PrecondSide side, const Vector& r) {
    bool u1, u2;
    activeFactors(P.mode, side, u1, u2);
    Vector z = r;
    if (u1) z = forwardSolve(P.F1, z);
    if (u2) z = backwardSolve(P.F2, z);
    return z;
}

}  // namespace

// ---------------- Dense helpers ----------------

Vector matVec(const Matrix& A, const Vector& x) {
    Vector y(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < x.size(); ++j) y[i] += A[i][j] * x[j];
    return y;
}

Vector forwardSolve(const Matrix& L, const Vector& r) {
    int n = r.size();
    Vector z(n);
    for (int i = 0; i < n; ++i) {
        double s = r[i];
        for (int j = 0; j < i; ++j) s -= L[i][j] * z[j];
        z[i] = s / L[i][i];
    }
    return z;
}

Vector backwardSolve(const Matrix& U, const Vector& r) {
    int n = r.size();
    Vector z(n);
    for (int i = n - 1; i >= 0; --i) {
        double s = r[i];
        for (int j = i + 1; j < n; ++j) s -= U[i][j] * z[j];
        z[i] = s / U[i][i];
    }
    return z;
}

Vector diagonalSolve(const Matrix& D, const Vector& r) {
    Vector z(r.size());
    for (size_t i = 0; i < r.size(); ++i) z[i] = r[i] / D[i][i];
    return z;
}

Matrix transposeMatrix(const Matrix& A) {
    int n = A.size();
    Matrix T = zeros(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) T[j][i] = A[i][j];
    return T;
}

Matrix identityMatrix(int n) {
    Matrix I = zeros(n);
    for (int i = 0; i < n; ++i) I[i][i] = 1.0;
    return I;
}

// ---------------- Setup ----------------

Preconditioner setupJacobi(const Matrix& A, PrecondMode mode) {
    Preconditioner P = makePrecond(PrecondMethod::Jacobi, mode, A, 1.0);
    Vector d = diagonalOf(A);
    if (P.mode == PrecondMode::Split && !allPositive(d)) P.mode = PrecondMode::Left;
    bool split = (P.mode == PrecondMode::Split);
    P.F1 = zeros(P.n);
    P.F2 = identityMatrix(P.n);
    for (int i = 0; i < P.n; ++i) {
        P.F1[i][i] = split ? std::sqrt(d[i]) : d[i];
        if (split) P.F2[i][i] = P.F1[i][i];
    }
    finalize(P);
    return P;
}

Preconditioner setupGaussSeidel(const Matrix& A, PrecondMode mode) {
    Preconditioner P = makePrecond(PrecondMethod::GaussSeidel, mode, A, 1.0);
    P.mode = noSplit(mode);
    diagonalOf(A);   // zero-diagonal guard
    P.F1 = zeros(P.n);
    for (int i = 0; i < P.n; ++i)
        for (int j = 0; j <= i; ++j) P.F1[i][j] = A[i][j];   // D + L
    P.F2 = identityMatrix(P.n);
    finalize(P);
    return P;
}

Preconditioner setupSymmetricGaussSeidel(const Matrix& A, PrecondMode mode) {
    Preconditioner P = makePrecond(PrecondMethod::SymmetricGaussSeidel, mode, A, 1.0);
    Vector d = diagonalOf(A);
    if (P.mode == PrecondMode::Split && !allPositive(d)) P.mode = PrecondMode::Left;
    bool split = (P.mode == PrecondMode::Split);
    P.F1 = zeros(P.n);
    P.F2 = zeros(P.n);
    for (int i = 0; i < P.n; ++i) {
        for (int j = 0; j <= i; ++j)                       // (D+L) D^-1  or  (D+L) D^-1/2
            P.F1[i][j] = A[i][j] / (split ? std::sqrt(d[j]) : d[j]);
        for (int j = i; j < P.n; ++j)                      // (D+U)       or  D^-1/2 (D+U)
            P.F2[i][j] = split ? A[i][j] / std::sqrt(d[i]) : A[i][j];
    }
    finalize(P);
    return P;
}

Preconditioner setupSOR(const Matrix& A, PrecondMode mode, double omega) {
    if (omega <= 0.0) throw std::invalid_argument("SOR omega must be positive");
    Preconditioner P = makePrecond(PrecondMethod::SOR, mode, A, omega);
    P.mode = noSplit(mode);
    diagonalOf(A);
    P.F1 = zeros(P.n);
    for (int i = 0; i < P.n; ++i) {
        for (int j = 0; j < i; ++j) P.F1[i][j] = A[i][j];
        P.F1[i][i] = A[i][i] / omega;                      // D/omega + L
    }
    P.F2 = identityMatrix(P.n);
    finalize(P);
    return P;
}

Preconditioner setupLU(const Matrix& A, PrecondMode mode) {
    Preconditioner P = makePrecond(PrecondMethod::LU, mode, A, 1.0);
    luFactor(A, false, P.F1, P.F2);
    finalize(P);
    return P;
}

Preconditioner setupCholesky(const Matrix& A, PrecondMode mode) {
    Preconditioner P = makePrecond(PrecondMethod::Cholesky, mode, A, 1.0);
    P.F1 = choleskyFactor(A, false);
    P.F2 = transposeMatrix(P.F1);
    finalize(P);
    return P;
}

Preconditioner setupILU0(const Matrix& A, PrecondMode mode) {
    Preconditioner P = makePrecond(PrecondMethod::ILU0, mode, A, 1.0);
    luFactor(A, true, P.F1, P.F2);
    finalize(P);
    return P;
}

Preconditioner setupIC0(const Matrix& A, PrecondMode mode) {
    Preconditioner P = makePrecond(PrecondMethod::IC0, mode, A, 1.0);
    P.F1 = choleskyFactor(A, true);
    P.F2 = transposeMatrix(P.F1);
    finalize(P);
    return P;
}

// ---------------- Apply ----------------

Vector applyJacobiPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r) {
    bool u1, u2;
    activeFactors(P.mode, side, u1, u2);
    Vector z = r;
    if (u1) z = diagonalSolve(P.F1, z);
    if (u2) z = diagonalSolve(P.F2, z);
    return z;
}

// F2 is the identity for Gauss-Seidel and SOR, so only F1 is ever applied.
Vector applyGaussSeidelPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r) {
    bool u1, u2;
    activeFactors(P.mode, side, u1, u2);
    return u1 ? forwardSolve(P.F1, r) : r;
}

Vector applySORPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r) {
    return applyGaussSeidelPreconditioner(P, side, r);
}

Vector applySymmetricGaussSeidelPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r) {
    return applyLowerUpper(P, side, r);
}
Vector applyLUPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r) {
    return applyLowerUpper(P, side, r);
}
Vector applyCholeskyPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r) {
    return applyLowerUpper(P, side, r);
}
Vector applyILUPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r) {
    return applyLowerUpper(P, side, r);
}
Vector applyICPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r) {
    return applyLowerUpper(P, side, r);
}

// ---------------- Dispatchers ----------------

Preconditioner buildPreconditioner(PrecondMethod method, PrecondMode mode, const Matrix& A, double omega) {
    switch (method) {
        case PrecondMethod::Jacobi:               return setupJacobi(A, mode);
        case PrecondMethod::GaussSeidel:          return setupGaussSeidel(A, mode);
        case PrecondMethod::SymmetricGaussSeidel: return setupSymmetricGaussSeidel(A, mode);
        case PrecondMethod::SOR:                  return setupSOR(A, mode, omega);
        case PrecondMethod::LU:                   return setupLU(A, mode);
        case PrecondMethod::Cholesky:             return setupCholesky(A, mode);
        case PrecondMethod::ILU0:                 return setupILU0(A, mode);
        case PrecondMethod::IC0:                  return setupIC0(A, mode);
    }
    throw std::invalid_argument("unknown preconditioner method");
}

Vector applyPreconditioner(const Preconditioner& P, PrecondSide side, const Vector& r) {
    switch (P.method) {
        case PrecondMethod::Jacobi:               return applyJacobiPreconditioner(P, side, r);
        case PrecondMethod::GaussSeidel:          return applyGaussSeidelPreconditioner(P, side, r);
        case PrecondMethod::SymmetricGaussSeidel: return applySymmetricGaussSeidelPreconditioner(P, side, r);
        case PrecondMethod::SOR:                  return applySORPreconditioner(P, side, r);
        case PrecondMethod::LU:                   return applyLUPreconditioner(P, side, r);
        case PrecondMethod::Cholesky:             return applyCholeskyPreconditioner(P, side, r);
        case PrecondMethod::ILU0:                 return applyILUPreconditioner(P, side, r);
        case PrecondMethod::IC0:                  return applyICPreconditioner(P, side, r);
    }
    throw std::invalid_argument("unknown preconditioner method");
}

// ---------------- Validation ----------------

double checkPreconditioner(const Matrix& A, const Preconditioner& P) {
    double s = 0.0;
    for (int j = 0; j < P.n; ++j) {
        Vector e(P.n, 0.0);
        e[j] = 1.0;
        Vector y = applyPreconditioner(P, PrecondSide::M2, e);
        Vector z = applyPreconditioner(P, PrecondSide::M1, matVec(A, y));
        for (int i = 0; i < P.n; ++i) s += (z[i] - e[i]) * (z[i] - e[i]);
    }
    return std::sqrt(s / P.n);
}