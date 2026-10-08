#ifndef KRYLOVSOLVERS_H
#define KRYLOVSOLVERS_H

#include <vector>

#include "Preconditioner.h"

enum class KrylovMethod
{
    CG,
    SteepestDescent,
    BiCGSTAB,
    Auto
};

enum class KrylovStatus
{
    Converged,
    MaxIterations,
    Breakdown
};

struct KrylovOptions
{
    // Relative residual tolerance
    double tol = 1e-10;

    int maxIter = 10000;
};

//--------------------------------------------------
// Result
//
// Monitored residual (history, relResidual, stopping
// test):
//
//   CG, Steepest Descent : true residual
//                          ||b - A x|| / ||b||
//   BiCGSTAB             : residual of the transformed
//                          system M1^-1 A M2^-1 y =
//                          M1^-1 (b - A x0), relative to
//                          ||M1^-1 b||. Equals the true
//                          residual for Right mode or no
//                          preconditioner.
//
// trueRelResidual is always the explicit
// ||b - A x|| / ||b|| computed after the solve.
//--------------------------------------------------

struct KrylovResult
{
    std::vector<double> x;

    // Method actually used (Auto resolved)
    KrylovMethod method = KrylovMethod::CG;

    KrylovStatus status = KrylovStatus::MaxIterations;

    int iterations = 0;

    // Products with A inside the solver
    int matVecs = 0;

    // Non-trivial M1^-1 / M2^-1 applications
    int precondApplies = 0;

    // Monitored residual at exit
    double relResidual = 0.0;

    double trueRelResidual = 0.0;

    double seconds = 0.0;

    // Monitored relative residual; entry 0 = initial
    std::vector<double> history;
};

struct MatrixInfo
{
    int n = 0;

    bool symmetric = false;

    // Symmetric and Cholesky succeeds
    bool spd = false;

    // 2-norm estimate; -1 if not computed or failed
    double conditionEstimate = -1.0;
};

//--------------------------------------------------
// Matrix Checks
//
// analyzeMatrix tests symmetry and SPD (a dense Cholesky
// attempt, O(n^3/3)).
//
// estimateConditionNumber gives a 2-norm estimate by
// power / inverse iteration on A^T A, using an unpivoted
// LU (O(n^3)). Returns -1 if the LU hits a zero pivot.
//
// chooseMethod returns CG for SPD matrices and BiCGSTAB
// otherwise. Steepest Descent is only used when
// requested explicitly.
//--------------------------------------------------

MatrixInfo analyzeMatrix
(
    const std::vector<std::vector<double>>& A,
    bool estimateCondition = false
);

double estimateConditionNumber(const std::vector<std::vector<double>>& A);

KrylovMethod chooseMethod(const MatrixInfo& info);

//--------------------------------------------------
// Solvers
//
// P = nullptr means no preconditioning. x0 may be empty
// (zero initial guess).
//
// CG and Steepest Descent use M^-1 = M2^-1 M1^-1, which
// is identical for Left / Right / Split, and throw
// std::invalid_argument for Gauss-Seidel and SOR, whose
// M is not symmetric.
//
// BiCGSTAB honours Left / Right / Split through the
// operator M1^-1 A M2^-1.
//--------------------------------------------------

KrylovResult steepestDescent
(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& b,
    const std::vector<double>& x0,
    const Preconditioner* P,
    const KrylovOptions& opt
);

KrylovResult conjugateGradient
(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& b,
    const std::vector<double>& x0,
    const Preconditioner* P,
    const KrylovOptions& opt
);

KrylovResult biCGStab
(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& b,
    const std::vector<double>& x0,
    const Preconditioner* P,
    const KrylovOptions& opt
);

//--------------------------------------------------
// Dispatcher
//
// Single entry point. KrylovMethod::Auto runs
// analyzeMatrix and chooseMethod.
//--------------------------------------------------

KrylovResult solveKrylov
(
    KrylovMethod method,
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& b,
    const std::vector<double>& x0,
    const Preconditioner* P,
    const KrylovOptions& opt
);

const char* krylovMethodName(KrylovMethod m);

const char* krylovStatusName(KrylovStatus s);

#endif