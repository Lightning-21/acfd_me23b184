//------------------------------------------------------------//
// File: LaplacianOperator.cpp
//
// Description:
//
// Generic 2D Laplacian operator on a uniform grid, built from a
// central finite-difference stencil of a requested order of
// accuracy. Reuses FiniteDifference's linear-solve / factorial /
// centralOffsets / extrapolate helpers instead of duplicating
// them.
//
// Fix for the near-boundary gap: a node closer than halfWidth to
// the i = 0/Nx-1 or j = 0/Ny-1 edge does not have real neighbours
// on one side. Previously apply()/residual() (and
// GaussSeidel::sweep, which shares this code) simply skipped a
// margin of halfWidth nodes at the edge, which left interior
// nodes unupdated for any stencil wider than the 3-point 2nd-order
// one. ghostAwareX/ghostAwareY now fill the missing neighbour by
// extrapolating a polynomial through the boundary node and the
// halfWidth nearest interior nodes on that side, so every
// physical interior node is handled, regardless of stencil width.
//
//------------------------------------------------------------//

#include "LaplacianOperator.h"
#include "FiniteDifference.h"

#include <cmath>

using namespace std;

//------------------------------------------------------------//
// Stencil Construction
//------------------------------------------------------------//

LaplacianOperator::LaplacianStencil LaplacianOperator::buildStencil(int orderAcc)
{
    int halfWidth = 1;

    while(true)
    {
        vector<int> offsets = FiniteDifference::centralOffsets(halfWidth);

        int achieved = verifyOrder(offsets);

        if(achieved >= orderAcc)
        {
            LaplacianStencil stencil;
            stencil.offsets = offsets;
            stencil.achievedOrder = achieved;

            return stencil;
        }

        halfWidth++;
    }
}

vector<double> LaplacianOperator::computeWeights
(
    const vector<int>& offsets,
    double h
)
{
    vector<double> weights = fdWeights(offsets);

    for(double& w : weights)
    {
        w /= (h*h);
    }

    return weights;
}

vector<double> LaplacianOperator::fdWeights(const vector<int>& offsets)
{
    int n = (int)offsets.size();

    //--------------------------------------------------
    // Taylor-moment conditions: sum_i c_i * offset_i^k
    // equals k! for k == 2 (second derivative), 0 otherwise
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

    b[2] = 2.0;

    return FiniteDifference::solveLinearSystem(A, b, n);
}

//------------------------------------------------------------//
// Ghost-Aware Neighbour Access
//------------------------------------------------------------//

double LaplacianOperator::ghostAwareX
(
    const vector<double>& phi,
    int i,
    int j,
    int Nx,
    int Ny,
    int offset,
    int halfWidth
)
{
    int target = i+offset;

    if(target >= 0 && target < Nx)
    {
        return phi[j*Nx+target];
    }

    int numKnown = halfWidth+1;

    vector<double> knownValues(numKnown);
    vector<double> knownPositions(numKnown);

    if(target < 0)
    {
        //--------------------------------------------------
        // Extrapolate past i = 0 using nodes 0, 1, ..., halfWidth
        //--------------------------------------------------

        for(int k = 0; k < numKnown; k++)
        {
            knownPositions[k] = k;
            knownValues[k] = phi[j*Nx+k];
        }
    }
    else
    {
        //--------------------------------------------------
        // Extrapolate past i = Nx-1 using nodes Nx-1, Nx-2, ...
        //--------------------------------------------------

        for(int k = 0; k < numKnown; k++)
        {
            knownPositions[k] = (Nx-1)-k;
            knownValues[k] = phi[j*Nx+(Nx-1-k)];
        }
    }

    return FiniteDifference::extrapolate(knownValues, knownPositions, (double)target);
}

double LaplacianOperator::ghostAwareY
(
    const vector<double>& phi,
    int i,
    int j,
    int Nx,
    int Ny,
    int offset,
    int halfWidth
)
{
    int target = j+offset;

    if(target >= 0 && target < Ny)
    {
        return phi[target*Nx+i];
    }

    int numKnown = halfWidth+1;

    vector<double> knownValues(numKnown);
    vector<double> knownPositions(numKnown);

    if(target < 0)
    {
        for(int k = 0; k < numKnown; k++)
        {
            knownPositions[k] = k;
            knownValues[k] = phi[k*Nx+i];
        }
    }
    else
    {
        for(int k = 0; k < numKnown; k++)
        {
            knownPositions[k] = (Ny-1)-k;
            knownValues[k] = phi[(Ny-1-k)*Nx+i];
        }
    }

    return FiniteDifference::extrapolate(knownValues, knownPositions, (double)target);
}

//------------------------------------------------------------//
// Operator Application
//------------------------------------------------------------//

vector<double> LaplacianOperator::apply
(
    const vector<double>& phi,
    int Nx,
    int Ny,
    double dx,
    double dy,
    const vector<int>& offsets
)
{
    int halfWidth = ((int)offsets.size()-1)/2;

    vector<double> wx = computeWeights(offsets, dx);
    vector<double> wy = computeWeights(offsets, dy);

    vector<double> result(Nx*Ny, 0.0);

    for(int j = 1; j < Ny-1; j++)
    {
        for(int i = 1; i < Nx-1; i++)
        {
            double lap = 0.0;

            for(int k = 0; k < (int)offsets.size(); k++)
            {
                lap += wx[k]*ghostAwareX(phi, i, j, Nx, Ny, offsets[k], halfWidth);
                lap += wy[k]*ghostAwareY(phi, i, j, Nx, Ny, offsets[k], halfWidth);
            }

            result[j*Nx+i] = lap;
        }
    }

    return result;
}

vector<double> LaplacianOperator::residual
(
    const vector<double>& phi,
    const vector<double>& f,
    int Nx,
    int Ny,
    double dx,
    double dy,
    const vector<int>& offsets
)
{
    vector<double> Aphi = apply(phi, Nx, Ny, dx, dy, offsets);

    vector<double> r(Nx*Ny, 0.0);

    for(int j = 1; j < Ny-1; j++)
    {
        for(int i = 1; i < Nx-1; i++)
        {
            int idx = j*Nx+i;

            r[idx] = f[idx]-Aphi[idx];
        }
    }

    return r;
}

//------------------------------------------------------------//
// Order Verification
//------------------------------------------------------------//

int LaplacianOperator::verifyOrder(const vector<int>& offsets)
{
    double h1 = 0.05;
    double h2 = 0.025;

    vector<double> w1 = computeWeights(offsets, h1);
    vector<double> w2 = computeWeights(offsets, h2);

    double x0 = 0.6;

    auto approxSecondDeriv = [&](const vector<double>& w, double h)
    {
        double sum = 0.0;

        for(int i = 0; i < (int)offsets.size(); i++)
        {
            sum += w[i]*sin(x0+offsets[i]*h);
        }

        return sum;
    };

    double d1 = approxSecondDeriv(w1, h1);
    double d2 = approxSecondDeriv(w2, h2);

    double exact = -sin(x0);

    double e1 = fabs(d1-exact);
    double e2 = fabs(d2-exact);

    if(e1 < 1e-12 && e2 < 1e-12)
    {
        return (int)offsets.size()-1;
    }

    double ratio = e1/e2;

    return (int)round(log(ratio)/log(2.0));
}