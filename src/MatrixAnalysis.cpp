//------------------------------------------------------------//
// File: MatrixAnalysis.cpp
//
// Description:
//
// Matrix diagnostics used to choose a Krylov method: symmetry
// and SPD tests (analyzeMatrix), a 2-norm condition number
// estimate by power / inverse iteration (estimateConditionNumber)
// and the Auto selection rule (chooseMethod). Split out of
// KrylovSolvers.cpp since none of it is a Krylov iteration.
//
//------------------------------------------------------------//

#include "MatrixAnalysis.h"

#include "MatrixFunctions.h"
#include "Preconditioner.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>

using namespace std;

//------------------------------------------------------------//
// File-Local Helpers
//------------------------------------------------------------//

// SPD test by a dense Cholesky attempt; choleskyFactor throws
// at the first non-positive pivot
static bool isSPD(const vector<vector<double>>& A)
{
    try
    {
        choleskyFactor(A, false);
    }
    catch(const runtime_error&)
    {
        return false;
    }

    return true;
}

// Largest eigenvalue of a symmetric positive semi-definite
// operator by power iteration (Rayleigh quotient, fixed
// pseudo-random start vector so results are reproducible)
static double dominantEigenvalue
(
    const function<vector<double>(const vector<double>&)>& op,
    int n
)
{
    vector<double> v(n);

    unsigned s = 12345u;

    for(int i = 0; i < n; i++)
    {
        s = s*1103515245u + 12345u;

        v[i] = 0.5 + static_cast<double>((s >> 8) & 0xFFFF)/65536.0;
    }

    double nv = norm2(v);

    for(double& x : v)
    {
        x /= nv;
    }

    double lambda = 0.0;

    for(int k = 0; k < 2000; k++)
    {
        vector<double> w = op(v);

        double lambdaNew = dot(v, w);

        double nw = norm2(w);

        if(nw == 0.0)
        {
            return 0.0;
        }

        for(int i = 0; i < n; i++)
        {
            v[i] = w[i]/nw;
        }

        if(k > 0 && fabs(lambdaNew - lambda) <= 1e-9*fabs(lambdaNew))
        {
            return lambdaNew;
        }

        lambda = lambdaNew;
    }

    return lambda;
}

//------------------------------------------------------------//
// Matrix Checks
//------------------------------------------------------------//

MatrixInfo analyzeMatrix
(
    const vector<vector<double>>& A,
    bool estimateCondition
)
{
    MatrixInfo info;

    size_t n = A.size();

    requireSquare(A);

    info.n = static_cast<int>(n);

    double amax = 0.0;
    double asym = 0.0;

    for(size_t i = 0; i < n; i++)
    {
        for(size_t j = 0; j < n; j++)
        {
            amax = max(amax, fabs(A[i][j]));
            asym = max(asym, fabs(A[i][j] - A[j][i]));
        }
    }

    info.symmetric = (asym <= 1e-12*amax);

    info.spd = info.symmetric && isSPD(A);

    if(estimateCondition)
    {
        info.conditionEstimate = estimateConditionNumber(A);
    }

    return info;
}

double estimateConditionNumber(const vector<vector<double>>& A)
{
    int n = static_cast<int>(A.size());

    if(n == 0)
    {
        return -1.0;
    }

    Preconditioner lu;

    try
    {
        // M1 = L (unit lower), M2 = U
        lu = setupLU(A, PrecondMode::Split);
    }
    catch(const runtime_error&)
    {
        return -1.0;
    }

    vector<vector<double>> At = transposeMatrix(A);
    vector<vector<double>> Ut = transposeMatrix(lu.F2);
    vector<vector<double>> Lt = transposeMatrix(lu.F1);

    // sigma_max^2 = lambda_max(A^T A)
    double lamMax = dominantEigenvalue
    (
        [&](const vector<double>& v)
        {
            return matVec(At, matVec(A, v));
        },
        n
    );

    // 1 / sigma_min^2 = lambda_max((A^T A)^-1), with
    // (A^T A)^-1 = A^-1 A^-T and A^-T = L^-T U^-T
    double invLamMin = dominantEigenvalue
    (
        [&](const vector<double>& v)
        {
            vector<double> w = backwardSolve(Lt, forwardSolve(Ut, v));

            return backwardSolve(lu.F2, forwardSolve(lu.F1, w));
        },
        n
    );

    if(!(lamMax > 0.0) || !(invLamMin > 0.0))
    {
        return -1.0;
    }

    return sqrt(lamMax*invLamMin);
}

KrylovMethod chooseMethod(const MatrixInfo& info)
{
    return info.spd ? KrylovMethod::CG : KrylovMethod::BiCGSTAB;
}