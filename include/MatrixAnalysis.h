#ifndef MATRIXANALYSIS_H
#define MATRIXANALYSIS_H

#include <vector>

// KrylovMethod, for chooseMethod
#include "KrylovSolvers.h"

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

#endif