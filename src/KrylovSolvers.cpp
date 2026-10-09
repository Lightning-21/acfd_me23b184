//------------------------------------------------------------//
// File: KrylovSolvers.cpp
//
// Description:
//
// Krylov subspace solvers for dense Ax = b: Steepest Descent,
// Conjugate Gradient (CG) and Bi-Conjugate Gradient Stabilized
// (BiCGSTAB), all usable with the preconditioners in
// Preconditioner.h. A single entry point, solveKrylov, picks
// the method from the user's choice or, for Auto, from the
// matrix itself via MatrixAnalysis.h (SPD -> CG, otherwise
// BiCGSTAB).
//
// Steepest Descent and CG are written in preconditioned form
// with z = M^-1 r, so Left / Right / Split give identical
// iterates and the monitored residual is the true residual.
// BiCGSTAB solves M1^-1 A M2^-1 y = M1^-1 (b - A x0) and sets
// x = x0 + M2^-1 y, so it honours the preconditioning mode.
//
//------------------------------------------------------------//

#include "KrylovSolvers.h"

#include "MatrixAnalysis.h"
#include "MatrixFunctions.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

using namespace std;

//------------------------------------------------------------//
// File-Local Helpers
//------------------------------------------------------------//

// True if |d| is negligible compared with the product of the
// two norms, i.e. the cosine between the vectors is about 0
static bool nearlyOrthogonal(double d, double na, double nb)
{
    return fabs(d) <= 1e-14*na*nb;
}

namespace
{

// Problem operators with operation counters. An identity side
// costs nothing and is not counted.
struct KrylovWork
{
    const vector<vector<double>>& A;

    const Preconditioner* P;

    int matVecs;

    int precondApplies;

    KrylovWork(const vector<vector<double>>& A_, const Preconditioner* P_)
    :
        A(A_),
        P(P_),
        matVecs(0),
        precondApplies(0)
    {
    }

    vector<double> applyA(const vector<double>& x)
    {
        matVecs++;

        return matVec(A, x);
    }

    bool isIdentity(PrecondSide side) const
    {
        if(!P)
        {
            return true;
        }

        if(P->mode == PrecondMode::Left)
        {
            return side == PrecondSide::M2;
        }

        if(P->mode == PrecondMode::Right)
        {
            return side == PrecondSide::M1;
        }

        return false;
    }

    vector<double> applyInv(PrecondSide side, const vector<double>& v)
    {
        if(isIdentity(side))
        {
            return v;
        }

        precondApplies++;

        return applyPreconditioner(*P, side, v);
    }

    // M^-1 v = M2^-1 (M1^-1 v) in every mode
    vector<double> applyMinv(const vector<double>& v)
    {
        return applyInv(PrecondSide::M2, applyInv(PrecondSide::M1, v));
    }
};

}

static vector<double> initialGuess
(
    const vector<vector<double>>& A,
    const vector<double>& b,
    const vector<double>& x0,
    const Preconditioner* P
)
{
    size_t n = A.size();

    requireSquare(A);

    if(b.size() != n)
    {
        throw invalid_argument("right-hand side size does not match matrix");
    }

    if(!x0.empty() && x0.size() != n)
    {
        throw invalid_argument("initial guess size does not match matrix");
    }

    if(P && static_cast<size_t>(P->n) != n)
    {
        throw invalid_argument("preconditioner size does not match matrix");
    }

    return x0.empty() ? vector<double>(n, 0.0) : x0;
}

static void requireSymmetricPrecond(const Preconditioner* P, const char* solver)
{
    if(P && (P->method == PrecondMethod::GaussSeidel || P->method == PrecondMethod::SOR))
    {
        throw invalid_argument
        (
            string(solver) +
            " needs a symmetric preconditioner; Gauss-Seidel and SOR are not symmetric (use SGS)"
        );
    }
}

// b = 0  ->  x = 0
static KrylovResult zeroSolution(size_t n, KrylovResult R)
{
    R.x = vector<double>(n, 0.0);
    R.status = KrylovStatus::Converged;
    R.history.assign(1, 0.0);

    return R;
}

static void finish
(
    KrylovResult& R,
    const KrylovWork& work,
    const vector<vector<double>>& A,
    const vector<double>& b,
    const vector<double>& x,
    chrono::steady_clock::time_point t0
)
{
    R.x = x;
    R.matVecs = work.matVecs;
    R.precondApplies = work.precondApplies;

    // Diagnostic residual, not counted in matVecs
    R.trueRelResidual = norm2(axpy(-1.0, matVec(A, x), b))/norm2(b);

    R.seconds = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
}

//------------------------------------------------------------//
// Steepest Descent
//
// z = M^-1 r, alpha = r^T z / z^T A z, x += alpha z,
// r -= alpha A z   (z = r when unpreconditioned)
//------------------------------------------------------------//

KrylovResult steepestDescent
(
    const vector<vector<double>>& A,
    const vector<double>& b,
    const vector<double>& x0,
    const Preconditioner* P,
    const KrylovOptions& opt
)
{
    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();

    vector<double> x = initialGuess(A, b, x0, P);

    requireSymmetricPrecond(P, "Steepest Descent");

    KrylovResult R;

    R.method = KrylovMethod::SteepestDescent;

    double bnorm = norm2(b);

    if(bnorm == 0.0)
    {
        return zeroSolution(A.size(), R);
    }

    KrylovWork work(A, P);

    vector<double> r = axpy(-1.0, work.applyA(x), b);

    double rel = norm2(r)/bnorm;

    R.history.push_back(rel);

    if(rel <= opt.tol)
    {
        R.status = KrylovStatus::Converged;
    }
    else
    {
        for(int k = 1; k <= opt.maxIter; k++)
        {
            vector<double> z = work.applyMinv(r);

            vector<double> Az = work.applyA(z);

            double denom = dot(z, Az);

            if(!(denom > 0.0))
            {
                R.status = KrylovStatus::Breakdown;
                break;
            }

            double alpha = dot(r, z)/denom;

            x = axpy(alpha, z, x);
            r = axpy(-alpha, Az, r);

            rel = norm2(r)/bnorm;

            R.history.push_back(rel);
            R.iterations = k;

            if(rel <= opt.tol)
            {
                R.status = KrylovStatus::Converged;
                break;
            }
        }
    }

    R.relResidual = rel;

    finish(R, work, A, b, x, t0);

    return R;
}

//------------------------------------------------------------//
// Conjugate Gradient (preconditioned form)
//
// r0 = b - A x0, z0 = M^-1 r0, p0 = z0
// alpha = r^T z / p^T A p, x += alpha p, r -= alpha A p
// z = M^-1 r, beta = r_new^T z_new / r^T z, p = z + beta p
//
// With M = I this is the plain CG of the lecture notes.
//------------------------------------------------------------//

KrylovResult conjugateGradient
(
    const vector<vector<double>>& A,
    const vector<double>& b,
    const vector<double>& x0,
    const Preconditioner* P,
    const KrylovOptions& opt
)
{
    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();

    vector<double> x = initialGuess(A, b, x0, P);

    requireSymmetricPrecond(P, "CG");

    KrylovResult R;

    R.method = KrylovMethod::CG;

    double bnorm = norm2(b);

    if(bnorm == 0.0)
    {
        return zeroSolution(A.size(), R);
    }

    KrylovWork work(A, P);

    vector<double> r = axpy(-1.0, work.applyA(x), b);

    double rel = norm2(r)/bnorm;

    R.history.push_back(rel);

    if(rel <= opt.tol)
    {
        R.status = KrylovStatus::Converged;
    }
    else
    {
        vector<double> z = work.applyMinv(r);

        vector<double> p = z;

        double rz = dot(r, z);

        for(int k = 1; k <= opt.maxIter; k++)
        {
            vector<double> Ap = work.applyA(p);

            double pAp = dot(p, Ap);

            // A is not SPD
            if(!(pAp > 0.0))
            {
                R.status = KrylovStatus::Breakdown;
                break;
            }

            double alpha = rz/pAp;

            x = axpy(alpha, p, x);
            r = axpy(-alpha, Ap, r);

            rel = norm2(r)/bnorm;

            R.history.push_back(rel);
            R.iterations = k;

            if(rel <= opt.tol)
            {
                R.status = KrylovStatus::Converged;
                break;
            }

            z = work.applyMinv(r);

            double rzNew = dot(r, z);

            // M is not SPD
            if(!(rzNew > 0.0))
            {
                R.status = KrylovStatus::Breakdown;
                break;
            }

            p = axpy(rzNew/rz, p, z);

            rz = rzNew;
        }
    }

    R.relResidual = rel;

    finish(R, work, A, b, x, t0);

    return R;
}

//------------------------------------------------------------//
// BiCGSTAB (van der Vorst)
//
// Solves At y = bt with At = M1^-1 A M2^-1,
// bt = M1^-1 (b - A x0), y0 = 0, and x = x0 + M2^-1 y.
//
// rho = rhat^T r, beta = (rho/rho_old)(alpha/omega)
// p = r + beta (p - omega v), v = At p
// alpha = rho / rhat^T v, s = r - alpha v, t = At s
// omega = t^T s / t^T t
// y += alpha p + omega s, r = s - omega t
//
// Early exit with y += alpha p when ||s|| is small.
//------------------------------------------------------------//

KrylovResult biCGStab
(
    const vector<vector<double>>& A,
    const vector<double>& b,
    const vector<double>& x0,
    const Preconditioner* P,
    const KrylovOptions& opt
)
{
    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();

    vector<double> x = initialGuess(A, b, x0, P);

    KrylovResult R;

    R.method = KrylovMethod::BiCGSTAB;

    if(norm2(b) == 0.0)
    {
        return zeroSolution(A.size(), R);
    }

    size_t n = A.size();

    KrylovWork work(A, P);

    vector<double> r = work.applyInv(PrecondSide::M1, axpy(-1.0, work.applyA(x), b));

    double bnorm = norm2(work.applyInv(PrecondSide::M1, b));

    auto op = [&](const vector<double>& v)
    {
        return work.applyInv(PrecondSide::M1, work.applyA(work.applyInv(PrecondSide::M2, v)));
    };

    vector<double> y(n, 0.0);
    vector<double> p(n, 0.0);
    vector<double> v(n, 0.0);

    const vector<double> rhat = r;

    const double rhatNorm = norm2(rhat);

    double rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    double rel = norm2(r)/bnorm;

    R.history.push_back(rel);

    if(rel <= opt.tol)
    {
        R.status = KrylovStatus::Converged;
    }
    else
    {
        for(int k = 1; k <= opt.maxIter; k++)
        {
            double rhoNew = dot(rhat, r);

            if(nearlyOrthogonal(rhoNew, rhatNorm, norm2(r)))
            {
                R.status = KrylovStatus::Breakdown;
                break;
            }

            double beta = (rhoNew/rho)*(alpha/omega);

            rho = rhoNew;

            p = axpy(beta, axpy(-omega, v, p), r);

            v = op(p);

            double denom = dot(rhat, v);

            if(nearlyOrthogonal(denom, rhatNorm, norm2(v)))
            {
                R.status = KrylovStatus::Breakdown;
                break;
            }

            alpha = rho/denom;

            vector<double> s = axpy(-alpha, v, r);

            double sRel = norm2(s)/bnorm;

            if(sRel <= opt.tol)
            {
                y = axpy(alpha, p, y);
                r = s;
                rel = sRel;

                R.history.push_back(rel);
                R.iterations = k;
                R.status = KrylovStatus::Converged;
                break;
            }

            vector<double> t = op(s);

            double ts = dot(t, s);

            if(nearlyOrthogonal(ts, norm2(t), norm2(s)))
            {
                R.status = KrylovStatus::Breakdown;
                break;
            }

            omega = ts/dot(t, t);

            y = axpy(omega, s, axpy(alpha, p, y));
            r = axpy(-omega, t, s);

            rel = norm2(r)/bnorm;

            R.history.push_back(rel);
            R.iterations = k;

            if(rel <= opt.tol)
            {
                R.status = KrylovStatus::Converged;
                break;
            }
        }
    }

    if(R.iterations > 0)
    {
        x = axpy(1.0, work.applyInv(PrecondSide::M2, y), x);
    }

    R.relResidual = rel;

    finish(R, work, A, b, x, t0);

    return R;
}

//------------------------------------------------------------//
// Dispatcher and Names
//------------------------------------------------------------//

KrylovResult solveKrylov
(
    KrylovMethod method,
    const vector<vector<double>>& A,
    const vector<double>& b,
    const vector<double>& x0,
    const Preconditioner* P,
    const KrylovOptions& opt
)
{
    if(method == KrylovMethod::Auto)
    {
        method = chooseMethod(analyzeMatrix(A));
    }

    switch(method)
    {
        case KrylovMethod::CG:
            return conjugateGradient(A, b, x0, P, opt);

        case KrylovMethod::SteepestDescent:
            return steepestDescent(A, b, x0, P, opt);

        case KrylovMethod::BiCGSTAB:
            return biCGStab(A, b, x0, P, opt);

        case KrylovMethod::Auto:
            break;
    }

    throw invalid_argument("unknown Krylov method");
}

const char* krylovMethodName(KrylovMethod m)
{
    switch(m)
    {
        case KrylovMethod::CG:
            return "CG";

        case KrylovMethod::SteepestDescent:
            return "SteepestDescent";

        case KrylovMethod::BiCGSTAB:
            return "BiCGSTAB";

        case KrylovMethod::Auto:
            return "Auto";
    }

    return "?";
}

const char* krylovStatusName(KrylovStatus s)
{
    switch(s)
    {
        case KrylovStatus::Converged:
            return "Converged";

        case KrylovStatus::MaxIterations:
            return "MaxIterations";

        case KrylovStatus::Breakdown:
            return "Breakdown";
    }

    return "?";
}