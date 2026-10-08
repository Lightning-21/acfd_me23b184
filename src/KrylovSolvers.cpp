#include "KrylovSolvers.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>

namespace {

typedef std::chrono::steady_clock Clock;

double dot(const Vector& a, const Vector& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

double norm2(const Vector& a) { return std::sqrt(dot(a, a)); }

Vector axpy(double a, const Vector& x, const Vector& y) {   // y + a x
    Vector z(y);
    for (size_t i = 0; i < z.size(); ++i) z[i] += a * x[i];
    return z;
}

// True if |d| is negligible compared with the product of the two norms (cosine ~ 0).
bool nearlyOrthogonal(double d, double na, double nb) { return std::abs(d) <= 1e-14 * na * nb; }

// Problem operators with operation counters. An identity side costs nothing and is not counted.
struct Work {
    const Matrix& A;
    const Preconditioner* P;
    int matVecs;
    int precondApplies;

    Work(const Matrix& a, const Preconditioner* p) : A(a), P(p), matVecs(0), precondApplies(0) {}

    Vector applyA(const Vector& x) { ++matVecs; return matVec(A, x); }

    bool isIdentity(PrecondSide s) const {
        if (!P) return true;
        if (P->mode == PrecondMode::Left) return s == PrecondSide::M2;
        if (P->mode == PrecondMode::Right) return s == PrecondSide::M1;
        return false;
    }

    Vector applyInv(PrecondSide s, const Vector& v) {
        if (isIdentity(s)) return v;
        ++precondApplies;
        return applyPreconditioner(*P, s, v);
    }

    Vector applyMinv(const Vector& v) {   // M^-1 v = M2^-1 (M1^-1 v) in every mode
        return applyInv(PrecondSide::M2, applyInv(PrecondSide::M1, v));
    }
};

Vector initialGuess(const Matrix& A, const Vector& b, const Vector& x0, const Preconditioner* P) {
    size_t n = A.size();
    for (const auto& row : A)
        if (row.size() != n) throw std::invalid_argument("matrix must be square");
    if (b.size() != n) throw std::invalid_argument("right-hand side size does not match matrix");
    if (!x0.empty() && x0.size() != n) throw std::invalid_argument("initial guess size does not match matrix");
    if (P && static_cast<size_t>(P->n) != n) throw std::invalid_argument("preconditioner size does not match matrix");
    return x0.empty() ? Vector(n, 0.0) : x0;
}

void requireSymmetricPrecond(const Preconditioner* P, const char* solver) {
    if (P && (P->method == PrecondMethod::GaussSeidel || P->method == PrecondMethod::SOR))
        throw std::invalid_argument(std::string(solver) +
            " needs a symmetric preconditioner; Gauss-Seidel and SOR are not symmetric (use SGS)");
}

KrylovResult zeroSolution(size_t n, KrylovResult R) {   // b = 0  ->  x = 0
    R.x = Vector(n, 0.0);
    R.status = KrylovStatus::Converged;
    R.history.assign(1, 0.0);
    return R;
}

void finish(KrylovResult& R, const Work& w, const Matrix& A, const Vector& b, const Vector& x,
            Clock::time_point t0) {
    R.x = x;
    R.matVecs = w.matVecs;
    R.precondApplies = w.precondApplies;
    R.trueRelResidual = norm2(axpy(-1.0, matVec(A, x), b)) / norm2(b);   // diagnostic, not counted
    R.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
}

bool isSPD(const Matrix& A) {   // dense Cholesky attempt on the lower triangle
    int n = static_cast<int>(A.size());
    Matrix L(A.size(), Vector(A.size(), 0.0));
    for (int j = 0; j < n; ++j) {
        double s = A[j][j];
        for (int k = 0; k < j; ++k) s -= L[j][k] * L[j][k];
        if (!(s > 0.0)) return false;
        L[j][j] = std::sqrt(s);
        for (int i = j + 1; i < n; ++i) {
            double t = A[i][j];
            for (int k = 0; k < j; ++k) t -= L[i][k] * L[j][k];
            L[i][j] = t / L[j][j];
        }
    }
    return true;
}

// Largest eigenvalue of a symmetric positive semi-definite operator by power iteration
// (Rayleigh quotient, deterministic start vector).
double dominantEigenvalue(const std::function<Vector(const Vector&)>& op, int n) {
    Vector v(n);
    unsigned s = 12345u;
    for (int i = 0; i < n; ++i) {
        s = s * 1103515245u + 12345u;
        v[i] = 0.5 + static_cast<double>((s >> 8) & 0xFFFF) / 65536.0;
    }
    double nv = norm2(v);
    for (double& x : v) x /= nv;
    double lambda = 0.0;
    for (int k = 0; k < 2000; ++k) {
        Vector w = op(v);
        double lambdaNew = dot(v, w);
        double nw = norm2(w);
        if (nw == 0.0) return 0.0;
        for (int i = 0; i < n; ++i) v[i] = w[i] / nw;
        if (k > 0 && std::abs(lambdaNew - lambda) <= 1e-9 * std::abs(lambdaNew)) return lambdaNew;
        lambda = lambdaNew;
    }
    return lambda;
}

}  // namespace

// ---------------- Matrix checks ----------------

MatrixInfo analyzeMatrix(const Matrix& A, bool estimateCondition) {
    MatrixInfo info;
    size_t n = A.size();
    for (const auto& row : A)
        if (row.size() != n) throw std::invalid_argument("matrix must be square");
    info.n = static_cast<int>(n);
    double amax = 0.0, asym = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            amax = std::max(amax, std::abs(A[i][j]));
            asym = std::max(asym, std::abs(A[i][j] - A[j][i]));
        }
    info.symmetric = (asym <= 1e-12 * amax);
    info.spd = info.symmetric && isSPD(A);
    if (estimateCondition) info.conditionEstimate = estimateConditionNumber(A);
    return info;
}

double estimateConditionNumber(const Matrix& A) {
    int n = static_cast<int>(A.size());
    if (n == 0) return -1.0;
    Preconditioner lu;
    try {
        lu = setupLU(A, PrecondMode::Split);   // M1 = L (unit lower), M2 = U
    } catch (const std::runtime_error&) {
        return -1.0;
    }
    Matrix At = transposeMatrix(A), Ut = transposeMatrix(lu.F2), Lt = transposeMatrix(lu.F1);

    // sigma_max^2 = lambda_max(A^T A)
    double lamMax = dominantEigenvalue(
        [&](const Vector& v) { return matVec(At, matVec(A, v)); }, n);
    // 1 / sigma_min^2 = lambda_max((A^T A)^-1),  (A^T A)^-1 = A^-1 A^-T,  A^-T = L^-T U^-T
    double invLamMin = dominantEigenvalue(
        [&](const Vector& v) {
            Vector w = backwardSolve(Lt, forwardSolve(Ut, v));   // A^-T v
            return backwardSolve(lu.F2, forwardSolve(lu.F1, w)); // A^-1 w
        }, n);
    if (!(lamMax > 0.0) || !(invLamMin > 0.0)) return -1.0;
    return std::sqrt(lamMax * invLamMin);
}

KrylovMethod chooseMethod(const MatrixInfo& info) {
    return info.spd ? KrylovMethod::CG : KrylovMethod::BiCGSTAB;
}

// ---------------- Steepest Descent ----------------
// z = M^-1 r, alpha = r^T z / z^T A z, x += alpha z, r -= alpha A z   (z = r when unpreconditioned)

KrylovResult steepestDescent(const Matrix& A, const Vector& b, const Vector& x0,
                             const Preconditioner* P, const KrylovOptions& opt) {
    Clock::time_point t0 = Clock::now();
    Vector x = initialGuess(A, b, x0, P);
    requireSymmetricPrecond(P, "Steepest Descent");
    KrylovResult R;
    R.method = KrylovMethod::SteepestDescent;
    double bnorm = norm2(b);
    if (bnorm == 0.0) return zeroSolution(A.size(), R);

    Work w(A, P);
    Vector r = axpy(-1.0, w.applyA(x), b);
    double rel = norm2(r) / bnorm;
    R.history.push_back(rel);
    if (rel <= opt.tol) {
        R.status = KrylovStatus::Converged;
    } else {
        for (int k = 1; k <= opt.maxIter; ++k) {
            Vector z = w.applyMinv(r);
            Vector Az = w.applyA(z);
            double denom = dot(z, Az);
            if (!(denom > 0.0)) { R.status = KrylovStatus::Breakdown; break; }
            double alpha = dot(r, z) / denom;
            x = axpy(alpha, z, x);
            r = axpy(-alpha, Az, r);
            rel = norm2(r) / bnorm;
            R.history.push_back(rel);
            R.iterations = k;
            if (rel <= opt.tol) { R.status = KrylovStatus::Converged; break; }
        }
    }
    R.relResidual = rel;
    finish(R, w, A, b, x, t0);
    return R;
}

// ---------------- Conjugate Gradient (preconditioned form) ----------------
// r0 = b - A x0, z0 = M^-1 r0, p0 = z0
// alpha = r^T z / p^T A p, x += alpha p, r -= alpha A p, z = M^-1 r, beta = r_new^T z_new / r^T z
// p = z + beta p.   With M = I this is the plain CG of the lecture notes.

KrylovResult conjugateGradient(const Matrix& A, const Vector& b, const Vector& x0,
                               const Preconditioner* P, const KrylovOptions& opt) {
    Clock::time_point t0 = Clock::now();
    Vector x = initialGuess(A, b, x0, P);
    requireSymmetricPrecond(P, "CG");
    KrylovResult R;
    R.method = KrylovMethod::CG;
    double bnorm = norm2(b);
    if (bnorm == 0.0) return zeroSolution(A.size(), R);

    Work w(A, P);
    Vector r = axpy(-1.0, w.applyA(x), b);
    double rel = norm2(r) / bnorm;
    R.history.push_back(rel);
    if (rel <= opt.tol) {
        R.status = KrylovStatus::Converged;
    } else {
        Vector z = w.applyMinv(r);
        Vector p = z;
        double rz = dot(r, z);
        for (int k = 1; k <= opt.maxIter; ++k) {
            Vector Ap = w.applyA(p);
            double pAp = dot(p, Ap);
            if (!(pAp > 0.0)) { R.status = KrylovStatus::Breakdown; break; }   // A not SPD
            double alpha = rz / pAp;
            x = axpy(alpha, p, x);
            r = axpy(-alpha, Ap, r);
            rel = norm2(r) / bnorm;
            R.history.push_back(rel);
            R.iterations = k;
            if (rel <= opt.tol) { R.status = KrylovStatus::Converged; break; }
            z = w.applyMinv(r);
            double rzNew = dot(r, z);
            if (!(rzNew > 0.0)) { R.status = KrylovStatus::Breakdown; break; } // M not SPD
            p = axpy(rzNew / rz, p, z);
            rz = rzNew;
        }
    }
    R.relResidual = rel;
    finish(R, w, A, b, x, t0);
    return R;
}

// ---------------- BiCGSTAB (van der Vorst) ----------------
// Solves  At y = bt  with  At = M1^-1 A M2^-1,  bt = M1^-1 (b - A x0),  y0 = 0,  x = x0 + M2^-1 y.
// rho = rhat^T r, beta = (rho/rho_old)(alpha/omega), p = r + beta (p - omega v), v = At p,
// alpha = rho / rhat^T v, s = r - alpha v, t = At s, omega = t^T s / t^T t,
// y += alpha p + omega s, r = s - omega t.   Early exit with y += alpha p when ||s|| is small.

KrylovResult biCGStab(const Matrix& A, const Vector& b, const Vector& x0,
                      const Preconditioner* P, const KrylovOptions& opt) {
    Clock::time_point t0 = Clock::now();
    Vector x = initialGuess(A, b, x0, P);
    KrylovResult R;
    R.method = KrylovMethod::BiCGSTAB;
    if (norm2(b) == 0.0) return zeroSolution(A.size(), R);

    size_t n = A.size();
    Work w(A, P);
    Vector r = w.applyInv(PrecondSide::M1, axpy(-1.0, w.applyA(x), b));
    double bnorm = norm2(w.applyInv(PrecondSide::M1, b));
    auto op = [&](const Vector& v) {
        return w.applyInv(PrecondSide::M1, w.applyA(w.applyInv(PrecondSide::M2, v)));
    };

    Vector y(n, 0.0), p(n, 0.0), v(n, 0.0);
    const Vector rhat = r;
    const double rhatNorm = norm2(rhat);
    double rho = 1.0, alpha = 1.0, omega = 1.0;
    double rel = norm2(r) / bnorm;
    R.history.push_back(rel);
    if (rel <= opt.tol) {
        R.status = KrylovStatus::Converged;
    } else {
        for (int k = 1; k <= opt.maxIter; ++k) {
            double rhoNew = dot(rhat, r);
            if (nearlyOrthogonal(rhoNew, rhatNorm, norm2(r))) { R.status = KrylovStatus::Breakdown; break; }
            double beta = (rhoNew / rho) * (alpha / omega);
            rho = rhoNew;
            p = axpy(beta, axpy(-omega, v, p), r);
            v = op(p);
            double denom = dot(rhat, v);
            if (nearlyOrthogonal(denom, rhatNorm, norm2(v))) { R.status = KrylovStatus::Breakdown; break; }
            alpha = rho / denom;
            Vector s = axpy(-alpha, v, r);
            double sRel = norm2(s) / bnorm;
            if (sRel <= opt.tol) {
                y = axpy(alpha, p, y);
                r = s;
                rel = sRel;
                R.history.push_back(rel);
                R.iterations = k;
                R.status = KrylovStatus::Converged;
                break;
            }
            Vector t = op(s);
            double ts = dot(t, s);
            if (nearlyOrthogonal(ts, norm2(t), norm2(s))) { R.status = KrylovStatus::Breakdown; break; }
            omega = ts / dot(t, t);
            y = axpy(omega, s, axpy(alpha, p, y));
            r = axpy(-omega, t, s);
            rel = norm2(r) / bnorm;
            R.history.push_back(rel);
            R.iterations = k;
            if (rel <= opt.tol) { R.status = KrylovStatus::Converged; break; }
        }
    }
    if (R.iterations > 0) x = axpy(1.0, w.applyInv(PrecondSide::M2, y), x);
    R.relResidual = rel;
    finish(R, w, A, b, x, t0);
    return R;
}

// ---------------- Dispatcher and names ----------------

KrylovResult solveKrylov(KrylovMethod method, const Matrix& A, const Vector& b, const Vector& x0,
                         const Preconditioner* P, const KrylovOptions& opt) {
    if (method == KrylovMethod::Auto) method = chooseMethod(analyzeMatrix(A));
    switch (method) {
        case KrylovMethod::CG:              return conjugateGradient(A, b, x0, P, opt);
        case KrylovMethod::SteepestDescent: return steepestDescent(A, b, x0, P, opt);
        case KrylovMethod::BiCGSTAB:        return biCGStab(A, b, x0, P, opt);
        case KrylovMethod::Auto:            break;
    }
    throw std::invalid_argument("unknown Krylov method");
}

const char* krylovMethodName(KrylovMethod m) {
    switch (m) {
        case KrylovMethod::CG:              return "CG";
        case KrylovMethod::SteepestDescent: return "SteepestDescent";
        case KrylovMethod::BiCGSTAB:        return "BiCGSTAB";
        case KrylovMethod::Auto:            return "Auto";
    }
    return "?";
}

const char* krylovStatusName(KrylovStatus s) {
    switch (s) {
        case KrylovStatus::Converged:     return "Converged";
        case KrylovStatus::MaxIterations: return "MaxIterations";
        case KrylovStatus::Breakdown:     return "Breakdown";
    }
    return "?";
}