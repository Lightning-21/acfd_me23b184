//------------------------------------------------------------//
// File: PoissonProblem.cpp
//
// Description:
// Standalone test problem, NOT part of src/ — a specific use
// case built on top of FiniteDifference, LaplacianOperator,
// GaussSeidel and MultiGrid, the same way 02_BackwardFacingStep
// is a specific use case built on top of Mesh/Block.
//
// Solves the 2D Poisson equation on the unit square [0,1]x[0,1]
// with a manufactured solution:
//
//     phi_exact(x,y) = sin(pi*x) * sin(pi*y)
//     f(x,y)         = Laplacian(phi_exact) = -2*pi^2*phi_exact
//
// Dirichlet BC: phi = phi_exact on the boundary. For this choice
// of phi_exact, sin(pi*0) = sin(pi*1) = 0, so the boundary value
// is exactly zero everywhere — the zero-initialised phi array
// already satisfies the BC with no extra boundary-setting code.
//
// Runs MultiGrid::solve, then GaussSeidel::solve (no MG) on a
// separate copy as a baseline, and reports iterations and error
// against phi_exact for both.
//
// Compilation Instruction:
// g++ PoissonProblem.cpp src/*.cpp -Iinclude -std=c++17 -o poissonProblem && ./poissonProblem
//------------------------------------------------------------//

#include "FiniteDifference.h"
#include "LaplacianOperator.h"
#include "GaussSeidel.h"
#include "MultiGrid.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>

using namespace std;

const double PI = 3.14159265358979323846;


//------------------------------------------------------------//
// Manufactured Solution
//------------------------------------------------------------//

double phiExact(double x, double y)
{
    return sin(PI*x)*sin(PI*y);
}

double sourceTerm(double x, double y)
{
    return -2.0*PI*PI*phiExact(x, y);
}


//------------------------------------------------------------//
// Error Norms
//------------------------------------------------------------//

void computeErrors(const vector<double>& phi, int Nx, int Ny, double dx, double dy, double& maxErr, double& l2Err)
{
    maxErr = 0.0;
    double sumSq = 0.0;

    for(int j = 0; j < Ny; j++)
    {
        for(int i = 0; i < Nx; i++)
        {
            double x = i*dx;
            double y = j*dy;

            double err = fabs(phi[j*Nx+i] - phiExact(x, y));

            maxErr = max(maxErr, err);
            sumSq += err*err;
        }
    }

    l2Err = sqrt(sumSq/(Nx*Ny));
}


//------------------------------------------------------------//
// Cycle Type Name
//------------------------------------------------------------//

string cycleTypeName(CycleType cycleType)
{
    switch(cycleType)
    {
        case CycleType::V: return "V-cycle";
        case CycleType::W: return "W-cycle";
        case CycleType::F: return "F-cycle";
    }

    return "unknown";
}


//------------------------------------------------------------//
// Main
//------------------------------------------------------------//

int main()
{
    // Problem setup -------------------------------------------------

    int Nx = 65;
    int Ny = 65;

    double dx = 1.0/(Nx-1);
    double dy = 1.0/(Ny-1);

    int orderAcc = 2;
    int numLevels = 4;

    CycleType cycleType = CycleType::V;

    int maxCycles = 50;
    double tol = 1.0e-8;
    int nu1 = 2;
    int nu2 = 2;

    vector<double> f(Nx*Ny, 0.0);

    for(int j = 0; j < Ny; j++)
    {
        for(int i = 0; i < Nx; i++)
        {
            f[j*Nx+i] = sourceTerm(i*dx, j*dy);
        }
    }

    try
    {
        LaplacianOperator::LaplacianStencil stencil = LaplacianOperator::buildStencil(orderAcc);

        cout << "Stencil: requested order = " << orderAcc
             << ", achieved order = " << stencil.achievedOrder << endl << endl;

        cout << "Cycle type : " << cycleTypeName(cycleType) << endl << endl;

        // MultiGrid solve -------------------------------------------

        vector<double> phiMG(Nx*Ny, 0.0);

        int mgCycles = MultiGrid::solve
        (
            cycleType, numLevels,
            phiMG, f,
            Nx, Ny, dx, dy,
            stencil.offsets,
            maxCycles, tol,
            nu1, nu2
        );

        double mgMaxErr, mgL2Err;
        computeErrors(phiMG, Nx, Ny, dx, dy, mgMaxErr, mgL2Err);

        cout << "MultiGrid  : cycles = " << mgCycles
             << ", max error = " << scientific << setprecision(4) << mgMaxErr
             << ", L2 error = " << mgL2Err << endl;

        // Baseline: plain Gauss-Seidel, no multigrid -----------------

        vector<double> phiGS(Nx*Ny, 0.0);

        int gsIters = GaussSeidel::solve
        (
            phiGS, f,
            Nx, Ny, dx, dy,
            stencil.offsets,
            20000, tol
        );

        double gsMaxErr, gsL2Err;
        computeErrors(phiGS, Nx, Ny, dx, dy, gsMaxErr, gsL2Err);

        cout << "Plain GS   : iters  = " << gsIters
             << ", max error = " << scientific << setprecision(4) << gsMaxErr
             << ", L2 error = " << gsL2Err << endl;
    }
    catch(const exception& e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}