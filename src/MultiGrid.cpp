//------------------------------------------------------------//
// File: MultiGrid.cpp
//
// Description:
//
// V/W/F-cycle multigrid on top of GaussSeidel (smoother) and
// LaplacianOperator (residual). Coarsens Nx, Ny by (N-1)/2+1 and
// dx, dy by x2 per level.
//
// Previously, restrictField/prolongField decided which coarse
// nodes were "boundary" (direct injection) versus "interior"
// (9-point / bilinear) using a halfWidth-wide margin, inherited
// from the same incorrect assumption fixed in LaplacianOperator
// and GaussSeidel. Now that those two treat every node except the
// true edge (index 0 or N-1) as ordinary interior, restriction
// and prolongation do the same: boundary detection here is just
// "is this the edge node", regardless of stencil width.
//
//------------------------------------------------------------//

#include "MultiGrid.h"
#include "LaplacianOperator.h"
#include "GaussSeidel.h"

using namespace std;

//------------------------------------------------------------//
// Grid Hierarchy
//------------------------------------------------------------//

vector<GridLevel> MultiGrid::buildHierarchy
(
    int numLevels,
    const vector<double>& phi,
    const vector<double>& f,
    int Nx,
    int Ny,
    double dx,
    double dy
)
{
    vector<GridLevel> levels(numLevels);

    levels[0].Nx = Nx;
    levels[0].Ny = Ny;
    levels[0].dx = dx;
    levels[0].dy = dy;
    levels[0].phi = phi;
    levels[0].f = f;
    levels[0].r.assign(Nx*Ny, 0.0);

    for(int l = 1; l < numLevels; l++)
    {
        int NxPrev = levels[l-1].Nx;
        int NyPrev = levels[l-1].Ny;

        int NxCoarse = (NxPrev-1)/2+1;
        int NyCoarse = (NyPrev-1)/2+1;

        levels[l].Nx = NxCoarse;
        levels[l].Ny = NyCoarse;
        levels[l].dx = levels[l-1].dx*2.0;
        levels[l].dy = levels[l-1].dy*2.0;

        levels[l].phi.assign(NxCoarse*NyCoarse, 0.0);
        levels[l].f.assign(NxCoarse*NyCoarse, 0.0);
        levels[l].r.assign(NxCoarse*NyCoarse, 0.0);
    }

    return levels;
}

//------------------------------------------------------------//
// Restriction / Prolongation
//------------------------------------------------------------//

vector<double> MultiGrid::restrictField
(
    const vector<double>& fine,
    int NxFine,
    int NyFine,
    int NxCoarse,
    int NyCoarse
)
{
    vector<double> coarse(NxCoarse*NyCoarse, 0.0);

    for(int J = 0; J < NyCoarse; J++)
    {
        int j = 2*J;

        for(int I = 0; I < NxCoarse; I++)
        {
            int i = 2*I;

            bool onBoundary =
                (I == 0 || I == NxCoarse-1 ||
                 J == 0 || J == NyCoarse-1);

            if(onBoundary)
            {
                //--------------------------------------------------
                // Direct injection at the physical boundary
                //--------------------------------------------------

                coarse[J*NxCoarse+I] = fine[j*NxFine+i];
            }
            else
            {
                //--------------------------------------------------
                // 9-point full weighting
                //--------------------------------------------------

                double sum = 4.0*fine[j*NxFine+i];

                sum += 2.0*fine[j*NxFine+(i-1)];
                sum += 2.0*fine[j*NxFine+(i+1)];
                sum += 2.0*fine[(j-1)*NxFine+i];
                sum += 2.0*fine[(j+1)*NxFine+i];

                sum += 1.0*fine[(j-1)*NxFine+(i-1)];
                sum += 1.0*fine[(j-1)*NxFine+(i+1)];
                sum += 1.0*fine[(j+1)*NxFine+(i-1)];
                sum += 1.0*fine[(j+1)*NxFine+(i+1)];

                coarse[J*NxCoarse+I] = sum/16.0;
            }
        }
    }

    return coarse;
}

vector<double> MultiGrid::prolongField
(
    const vector<double>& coarse,
    int NxCoarse,
    int NyCoarse,
    int NxFine,
    int NyFine
)
{
    vector<double> fine(NxFine*NyFine, 0.0);

    for(int j = 0; j < NyFine; j++)
    {
        bool jEven = (j%2 == 0);
        int J = j/2;

        for(int i = 0; i < NxFine; i++)
        {
            bool iEven = (i%2 == 0);
            int I = i/2;

            if(iEven && jEven)
            {
                fine[j*NxFine+i] = coarse[J*NxCoarse+I];
            }
            else if(!iEven && jEven)
            {
                fine[j*NxFine+i] =
                    0.5*(coarse[J*NxCoarse+I]+coarse[J*NxCoarse+(I+1)]);
            }
            else if(iEven && !jEven)
            {
                fine[j*NxFine+i] =
                    0.5*(coarse[J*NxCoarse+I]+coarse[(J+1)*NxCoarse+I]);
            }
            else
            {
                fine[j*NxFine+i] =
                    0.25*(coarse[J*NxCoarse+I]+coarse[J*NxCoarse+(I+1)]+
                          coarse[(J+1)*NxCoarse+I]+coarse[(J+1)*NxCoarse+(I+1)]);
            }
        }
    }

    return fine;
}

//------------------------------------------------------------//
// Coarsest-Level Solve
//------------------------------------------------------------//

void MultiGrid::solveCoarsest
(
    GridLevel& level,
    const vector<int>& offsets
)
{
    GaussSeidel::solve
    (
        level.phi,
        level.f,
        level.Nx,
        level.Ny,
        level.dx,
        level.dy,
        offsets,
        1000,
        1e-12
    );
}

//------------------------------------------------------------//
// Cycle (recursive V/W engine)
//------------------------------------------------------------//

void MultiGrid::cycle
(
    vector<GridLevel>& levels,
    int level,
    int gamma,
    const vector<int>& offsets,
    int nu1,
    int nu2
)
{
    GridLevel& fine = levels[level];

    if(level == (int)levels.size()-1)
    {
        solveCoarsest(fine, offsets);
        return;
    }

    //--------------------------------------------------
    // Pre-smoothing
    //--------------------------------------------------

    for(int s = 0; s < nu1; s++)
    {
        GaussSeidel::sweep(fine.phi, fine.f, fine.Nx, fine.Ny, fine.dx, fine.dy, offsets);
    }

    //--------------------------------------------------
    // Residual and restriction
    //--------------------------------------------------

    fine.r = LaplacianOperator::residual(fine.phi, fine.f, fine.Nx, fine.Ny, fine.dx, fine.dy, offsets);

    GridLevel& coarse = levels[level+1];

    coarse.f = restrictField(fine.r, fine.Nx, fine.Ny, coarse.Nx, coarse.Ny);
    coarse.phi.assign(coarse.Nx*coarse.Ny, 0.0);

    //--------------------------------------------------
    // Recurse: gamma = 1 -> V-cycle, gamma = 2 -> W-cycle
    //--------------------------------------------------

    for(int g = 0; g < gamma; g++)
    {
        cycle(levels, level+1, gamma, offsets, nu1, nu2);
    }

    //--------------------------------------------------
    // Prolongation and correction
    //--------------------------------------------------

    vector<double> correction = prolongField(coarse.phi, coarse.Nx, coarse.Ny, fine.Nx, fine.Ny);

    for(int idx = 0; idx < fine.Nx*fine.Ny; idx++)
    {
        fine.phi[idx] += correction[idx];
    }

    //--------------------------------------------------
    // Post-smoothing
    //--------------------------------------------------

    for(int s = 0; s < nu2; s++)
    {
        GaussSeidel::sweep(fine.phi, fine.f, fine.Nx, fine.Ny, fine.dx, fine.dy, offsets);
    }
}

//------------------------------------------------------------//
// Full Multigrid (FMG)
//------------------------------------------------------------//

void MultiGrid::fmg
(
    vector<GridLevel>& levels,
    const vector<int>& offsets,
    int nu1,
    int nu2
)
{
    int numLevels = (int)levels.size();

    //--------------------------------------------------
    // Restrict f all the way down to the coarsest level
    //--------------------------------------------------

    for(int l = 1; l < numLevels; l++)
    {
        levels[l].f = restrictField
        (
            levels[l-1].f,
            levels[l-1].Nx,
            levels[l-1].Ny,
            levels[l].Nx,
            levels[l].Ny
        );
    }

    levels[numLevels-1].phi.assign(levels[numLevels-1].Nx*levels[numLevels-1].Ny, 0.0);

    solveCoarsest(levels[numLevels-1], offsets);

    //--------------------------------------------------
    // Walk back up: prolong as an initial guess, then
    // polish with one V-cycle per level
    //--------------------------------------------------

    for(int l = numLevels-2; l >= 0; l--)
    {
        levels[l].phi = prolongField
        (
            levels[l+1].phi,
            levels[l+1].Nx,
            levels[l+1].Ny,
            levels[l].Nx,
            levels[l].Ny
        );

        cycle(levels, l, 1, offsets, nu1, nu2);
    }
}

//------------------------------------------------------------//
// Solve
//------------------------------------------------------------//

int MultiGrid::solve
(
    CycleType cycleType,
    int numLevels,
    vector<double>& phi,
    const vector<double>& f,
    int Nx,
    int Ny,
    double dx,
    double dy,
    const vector<int>& offsets,
    int maxCycles,
    double tol,
    int nu1,
    int nu2
)
{
    vector<GridLevel> levels = buildHierarchy(numLevels, phi, f, Nx, Ny, dx, dy);

    int gamma = (cycleType == CycleType::W) ? 2 : 1;

    for(int iter = 0; iter < maxCycles; iter++)
    {
        if(cycleType == CycleType::F)
        {
            fmg(levels, offsets, nu1, nu2);
        }
        else
        {
            cycle(levels, 0, gamma, offsets, nu1, nu2);
        }

        double resNorm = GaussSeidel::residualNorm
        (
            levels[0].phi,
            levels[0].f,
            levels[0].Nx,
            levels[0].Ny,
            levels[0].dx,
            levels[0].dy,
            offsets
        );

        if(resNorm < tol)
        {
            phi = levels[0].phi;
            return iter+1;
        }
    }

    phi = levels[0].phi;
    return maxCycles;
}