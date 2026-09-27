//------------------------------------------------------------//
// File: GaussSeidel.cpp
//
// Description:
//
// Lexicographic Gauss-Seidel, generic for any central Laplacian
// stencil order. Reuses LaplacianOperator::computeWeights and
// LaplacianOperator::residual instead of duplicating them. Used
// both as a standalone baseline solver and as the smoother
// inside MultiGrid.
//
// sweep() previously extracted the diagonal term and updated
// only over an interior region shrunk by halfWidth on every
// side, which left near-boundary nodes unupdated for 4th/6th-
// order stencils. It now updates every physical interior node
// (i, j != 0 and != Nx-1/Ny-1) and goes through
// LaplacianOperator::ghostAwareX/Y for any neighbour that falls
// outside the domain, matching the fix in LaplacianOperator.
//
//------------------------------------------------------------//

#include "GaussSeidel.h"
#include "LaplacianOperator.h"

#include <cmath>

using namespace std;

//------------------------------------------------------------//
// Smoothing / Solving
//------------------------------------------------------------//

void GaussSeidel::sweep
(
    vector<double>& phi,
    const vector<double>& f,
    int Nx,
    int Ny,
    double dx,
    double dy,
    const vector<int>& offsets
)
{
    int halfWidth = ((int)offsets.size()-1)/2;
    int center = halfWidth;

    vector<double> wx = LaplacianOperator::computeWeights(offsets, dx);
    vector<double> wy = LaplacianOperator::computeWeights(offsets, dy);

    double diag = wx[center]+wy[center];

    for(int j = 1; j < Ny-1; j++)
    {
        for(int i = 1; i < Nx-1; i++)
        {
            double sum = 0.0;

            for(int k = 0; k < (int)offsets.size(); k++)
            {
                if(offsets[k] == 0)
                {
                    continue;
                }

                sum += wx[k]*LaplacianOperator::ghostAwareX(phi, i, j, Nx, Ny, offsets[k], halfWidth);
                sum += wy[k]*LaplacianOperator::ghostAwareY(phi, i, j, Nx, Ny, offsets[k], halfWidth);
            }

            int idx = j*Nx+i;

            phi[idx] = (f[idx]-sum)/diag;
        }
    }
}

double GaussSeidel::residualNorm
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
    vector<double> r = LaplacianOperator::residual(phi, f, Nx, Ny, dx, dy, offsets);

    double sumSq = 0.0;
    int count = 0;

    for(int j = 1; j < Ny-1; j++)
    {
        for(int i = 1; i < Nx-1; i++)
        {
            double val = r[j*Nx+i];

            sumSq += val*val;
            count++;
        }
    }

    return sqrt(sumSq/count);
}

int GaussSeidel::solve
(
    vector<double>& phi,
    const vector<double>& f,
    int Nx,
    int Ny,
    double dx,
    double dy,
    const vector<int>& offsets,
    int maxIter,
    double tol
)
{
    for(int iter = 0; iter < maxIter; iter++)
    {
        sweep(phi, f, Nx, Ny, dx, dy, offsets);

        if(residualNorm(phi, f, Nx, Ny, dx, dy, offsets) < tol)
        {
            return iter+1;
        }
    }

    return maxIter;
}