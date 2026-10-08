#ifndef KRYLOVSOLVERS_H
#define KRYLOVSOLVERS_H

#include <vector>

#include "Preconditioner.h"

enum class KrylovMethod { CG, SteepestDescent, BiCGSTAB, Auto };
enum class KrylovStatus { Converged, MaxIterations, Breakdown };

struct KrylovOptions {
    double tol = 1e-10;   // relative residual tolerance
    int maxIter = 10000;
};

// Residual that is monitored (history, relResidual, stopping test):
//   CG, Steepest Descent : true residual ||b - A x|| / ||b||
//   BiCGSTAB             : residual of the transformed system M1^-1 A M2^-1 y = M1^-1 (b - A x0),
//                          relative to ||M1^-1 b||. Equals the true residual for Right mode
//                          or no preconditioner.
// trueRelResidual is always the explicit ||b - A x|| / ||b|| computed after the solve.
struct KrylovResult {
    Vector x;
    KrylovMethod method = KrylovMethod::CG;        // method actually used (Auto resolved)
    KrylovStatus status = KrylovStatus::MaxIterations;
    int iterations = 0;
    int matVecs = 0;                               // products with A inside the solver
    int precondApplies = 0;                        // non-trivial M1^-1 / M2^-1 applications
    double relResidual = 0.0;                      // monitored residual at exit
    double trueRelResidual = 0.0;
    double seconds = 0.0;
    std::vector<double> history;                   // monitored relative residual; entry 0 = initial
};

struct MatrixInfo {
    int n = 0;
    bool symmetric = false;
    bool spd = false;                              // symmetric and Cholesky succeeds
    double conditionEstimate = -1.0;               // 2-norm estimate; -1 if not computed or failed
};

// ---- Matrix checks (used to pick a method) ----
// SPD test is a dense Cholesky attempt, O(n^3/3).
MatrixInfo analyzeMatrix(const Matrix& A, bool estimateCondition = false);
// 2-norm condition number estimate via power / inverse iteration on A^T A (uses an unpivoted LU,
// O(n^3)). Returns -1 if the LU hits a zero pivot.
double estimateConditionNumber(const Matrix& A);
// SPD -> CG, otherwise BiCGSTAB. Steepest Descent is only used when requested explicitly.
KrylovMethod chooseMethod(const MatrixInfo& info);

// ---- Solvers ----
// P = nullptr means no preconditioning. x0 may be empty (zero initial guess).
// CG and Steepest Descent use M^-1 = M2^-1 M1^-1 (identical for Left/Right/Split) and throw
// std::invalid_argument for Gauss-Seidel and SOR, whose M is not symmetric.
// BiCGSTAB honours Left/Right/Split via the operator M1^-1 A M2^-1.
KrylovResult steepestDescent(const Matrix& A, const Vector& b, const Vector& x0,
                             const Preconditioner* P, const KrylovOptions& opt);
KrylovResult conjugateGradient(const Matrix& A, const Vector& b, const Vector& x0,
                               const Preconditioner* P, const KrylovOptions& opt);
KrylovResult biCGStab(const Matrix& A, const Vector& b, const Vector& x0,
                      const Preconditioner* P, const KrylovOptions& opt);

// Single entry point. KrylovMethod::Auto runs analyzeMatrix and chooseMethod.
KrylovResult solveKrylov(KrylovMethod method, const Matrix& A, const Vector& b, const Vector& x0,
                         const Preconditioner* P, const KrylovOptions& opt);

const char* krylovMethodName(KrylovMethod m);
const char* krylovStatusName(KrylovStatus s);

#endif