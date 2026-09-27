//------------------------------------------------------------//
// File: FiniteDifference.cpp
//
// Description:
//
// Generates central finite-difference stencils (first
// derivative only) for a requested order of accuracy, and
// provides the linear-algebra / extrapolation helpers shared
// with LaplacianOperator and GaussSeidel.
//
//------------------------------------------------------------//

#include "FiniteDifference.h"

#include <cmath>

using namespace std;

//------------------------------------------------------------//
// Stencil Construction
//------------------------------------------------------------//

FiniteDifference::FDStencil FiniteDifference::buildStencil(int orderAcc)
{
    int halfWidth = 1;

    while(true)
    {
        vector<int> offsets = centralOffsets(halfWidth);

        int achieved = verifyOrder(offsets);

        if(achieved >= orderAcc)
        {
            FDStencil stencil;
            stencil.offsets = offsets;
            stencil.achievedOrder = achieved;

            return stencil;
        }

        halfWidth++;
    }
}

vector<double> FiniteDifference::computeWeights
(
    const vector<int>& offsets,
    double h
)
{
    vector<double> weights = fdWeights(offsets);

    for(double& w : weights)
    {
        w /= h;
    }

    return weights;
}

vector<double> FiniteDifference::fdWeights(const vector<int>& offsets)
{
    int n = (int)offsets.size();

    //--------------------------------------------------
    // Taylor-moment conditions: sum_i c_i * offset_i^k
    // equals k! for k == derivOrder (1), 0 otherwise
    //--------------------------------------------------

    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n, 0.0);

    for(int row = 0; row < n; row++)
    {
        for(int col = 0; col < n; col++)
        {
            A[row][col] = pow((double)offsets[col], (double)row);
        }
    }

    b[1] = 1.0;

    return solveLinearSystem(A, b, n);
}

//------------------------------------------------------------//
// Shared Helpers
//------------------------------------------------------------//

vector<int> FiniteDifference::centralOffsets(int halfWidth)
{
    vector<int> offsets;

    for(int k = -halfWidth; k <= halfWidth; k++)
    {
        offsets.push_back(k);
    }

    return offsets;
}

vector<double> FiniteDifference::solveLinearSystem
(
    vector<vector<double>> A,
    vector<double> b,
    int n
)
{
    //--------------------------------------------------
    // Gaussian elimination with partial pivoting
    //--------------------------------------------------

    for(int col = 0; col < n; col++)
    {
        int pivotRow = col;

        for(int row = col+1; row < n; row++)
        {
            if(fabs(A[row][col]) > fabs(A[pivotRow][col]))
            {
                pivotRow = row;
            }
        }

        swap(A[col], A[pivotRow]);
        swap(b[col], b[pivotRow]);

        for(int row = col+1; row < n; row++)
        {
            double factor = A[row][col]/A[col][col];

            for(int k = col; k < n; k++)
            {
                A[row][k] -= factor*A[col][k];
            }

            b[row] -= factor*b[col];
        }
    }

    vector<double> x(n, 0.0);

    for(int row = n-1; row >= 0; row--)
    {
        double sum = b[row];

        for(int col = row+1; col < n; col++)
        {
            sum -= A[row][col]*x[col];
        }

        x[row] = sum/A[row][row];
    }

    return x;
}

double FiniteDifference::factorial(int k)
{
    double result = 1.0;

    for(int i = 2; i <= k; i++)
    {
        result *= i;
    }

    return result;
}

//------------------------------------------------------------//
// Ghost-Node Extrapolation
//------------------------------------------------------------//

double FiniteDifference::extrapolate
(
    const vector<double>& knownValues,
    const vector<double>& knownPositions,
    double targetPosition
)
{
    int n = (int)knownValues.size();

    //--------------------------------------------------
    // Fit the unique degree-(n-1) polynomial through the
    // known (position, value) pairs, then evaluate it at
    // targetPosition.
    //--------------------------------------------------

    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);

    for(int row = 0; row < n; row++)
    {
        for(int col = 0; col < n; col++)
        {
            A[row][col] = pow(knownPositions[row], (double)col);
        }

        b[row] = knownValues[row];
    }

    vector<double> coeffs = solveLinearSystem(A, b, n);

    double value = 0.0;

    for(int col = 0; col < n; col++)
    {
        value += coeffs[col]*pow(targetPosition, (double)col);
    }

    return value;
}

//------------------------------------------------------------//
// Order Verification
//------------------------------------------------------------//

int FiniteDifference::verifyOrder(const vector<int>& offsets)
{
    double h1 = 0.05;
    double h2 = 0.025;

    vector<double> w1 = computeWeights(offsets, h1);
    vector<double> w2 = computeWeights(offsets, h2);

    double x0 = 0.6;

    auto approxDeriv = [&](const vector<double>& w, double h)
    {
        double sum = 0.0;

        for(int i = 0; i < (int)offsets.size(); i++)
        {
            sum += w[i]*sin(x0+offsets[i]*h);
        }

        return sum;
    };

    double d1 = approxDeriv(w1, h1);
    double d2 = approxDeriv(w2, h2);

    double exact = cos(x0);

    double e1 = fabs(d1-exact);
    double e2 = fabs(d2-exact);

    if(e1 < 1e-12 && e2 < 1e-12)
    {
        return (int)offsets.size()-1;
    }

    double ratio = e1/e2;

    return (int)round(log(ratio)/log(2.0));
}