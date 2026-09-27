//------------------------------------------------------------//
// File: LaplacianOperator.cpp
//
// Description:
//
// Generic 2D Laplacian operator on a uniform structured grid.
// Builds a central second-derivative stencil at a requested
// order of accuracy, and applies it (dimensional splitting:
// d2/dx2 + d2/dy2) over every physical interior node. Neighbour
// access goes through ghostAwareX/Y, which extrapolate past the
// boundary instead of skipping near-boundary nodes, so 4th/6th-
// order stencils are handled correctly right up to the edge.
// Reuses FiniteDifference for the shared linear-algebra and
// extrapolation machinery.
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
    // equals k! for k == derivOrder (2), 0 otherwise
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

    b[2] = FiniteDifference::factorial(2);

    return FiniteDifference::solveLinearSystem(A, b, n);
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
    int center = halfWidth;

    vector<double> wx = computeWeights(offsets, dx);
    vector<double> wy = computeWeights(offsets, dy);

    double diag = wx[center]+wy[center];

    vector<double> lap(Nx*Ny, 0.0);

    for(int j = 1; j < Ny-1; j++)
    {
        for(int i = 1; i < Nx-1; i++)
        {
            double sum = diag*phi[j*Nx+i];

            for(int k = 0; k < (int)offsets.size(); k++)
            {
                if(offsets[k] == 0)
                {
                    continue;
                }

                sum += wx[k]*ghostAwareX(phi, i, j, Nx, Ny, offsets[k], halfWidth);
                sum += wy[k]*ghostAwareY(phi, i, j, Nx, Ny, offsets[k], halfWidth);
            }

            lap[j*Nx+i] = sum;
        }
    }

    return lap;
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
    int ii = i+offset;

    if(ii >= 0 && ii < Nx)
    {
        return phi[j*Nx+ii];
    }

    int n = halfWidth+1;

    vector<double> knownValues(n);
    vector<double> knownPositions(n);

    if(ii < 0)
    {
        for(int k = 0; k < n; k++)
        {
            knownPositions[k] = k;
            knownValues[k] = phi[j*Nx+k];
        }
    }
    else
    {
        for(int k = 0; k < n; k++)
        {
            int pos = Nx-n+k;

            knownPositions[k] = pos;
            knownValues[k] = phi[j*Nx+pos];
        }
    }

    return FiniteDifference::extrapolate(knownValues, knownPositions, (double)ii);
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
    int jj = j+offset;

    if(jj >= 0 && jj < Ny)
    {
        return phi[jj*Nx+i];
    }

    int n = halfWidth+1;

    vector<double> knownValues(n);
    vector<double> knownPositions(n);

    if(jj < 0)
    {
        for(int k = 0; k < n; k++)
        {
            knownPositions[k] = k;
            knownValues[k] = phi[k*Nx+i];
        }
    }
    else
    {
        for(int k = 0; k < n; k++)
        {
            int pos = Ny-n+k;

            knownPositions[k] = pos;
            knownValues[k] = phi[pos*Nx+i];
        }
    }

    return FiniteDifference::extrapolate(knownValues, knownPositions, (double)jj);
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