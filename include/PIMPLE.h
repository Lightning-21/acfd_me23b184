#ifndef PIMPLE_H
#define PIMPLE_H

#include <vector>

#include "NavierStokes2D.h"

enum class PIMPLEStatus
{
    Converged,
    MaxSteps,
    Diverged
};

//--------------------------------------------------
// Options
//
// scheme                 : convection scheme for momentum
// dt                     : pseudo-time step; if <= 0 it is
//                          cfl min(dx, dy) / uRef
// cfl                    : used only when dt <= 0
// uRef                   : reference speed; if <= 0 the largest
//                          wall speed is used
// outerIterations        : outer (SIMPLE-like) iterations per
//                          time step; 1 gives PISO
// correctors             : PISO pressure correctors per outer
//                          iteration
// alphaU, alphaP         : velocity (inside the momentum
//                          equation) and pressure under-
//                          relaxation of every outer iteration
//                          except the last, both in (0, 1]
// alphaUFinal            : velocity relaxation of the last outer
//                          iteration, in (0, 1]; the last
//                          iteration uses no pressure relaxation
// momentumSweeps         : Gauss-Seidel sweeps per predictor
// pressureTol            : relative tolerance of every p' solve
// pressureMaxIter        : iteration cap of every p' solve
// pressurePreconditioner : symmetric Gauss-Seidel on / off
// tol                    : stop when the scaled u, v and mass
//                          residuals are all below tol
// maxSteps               : time-step cap
// printEvery             : print a line every n steps (0 = silent)
//--------------------------------------------------

struct PIMPLEOptions
{
    ConvectionScheme scheme = ConvectionScheme::Hybrid;

    double dt = 0.0;

    double cfl = 16.0;

    double uRef = 0.0;

    int outerIterations = 2;

    int correctors = 2;

    double alphaU = 0.8;

    double alphaP = 0.5;

    double alphaUFinal = 1.0;

    int momentumSweeps = 3;

    double pressureTol = 1e-4;

    int pressureMaxIter = 1000;

    bool pressurePreconditioner = true;

    double tol = 1e-6;

    int maxSteps = 50000;

    int printEvery = 0;
};

//--------------------------------------------------
// Result
//
// Residuals are defined exactly as for PISO (see PISO.h), from
// the last outer iteration of each time step, so SIMPLE, PISO
// and PIMPLE histories are comparable.
//
// pressureIterations is the total number of CG iterations over
// all outer iterations and correctors.
//--------------------------------------------------

struct PIMPLEResult
{
    PIMPLEStatus status = PIMPLEStatus::MaxSteps;

    int steps = 0;

    int pressureIterations = 0;

    double dt = 0.0;

    // Pseudo-time reached, steps * dt
    double time = 0.0;

    double seconds = 0.0;

    double resU = 0.0;

    double resV = 0.0;

    double resMass = 0.0;

    std::vector<double> historyU;

    std::vector<double> historyV;

    std::vector<double> historyMass;
};

//--------------------------------------------------
// PIMPLE
//
// Merged PISO-SIMPLE, marched in pseudo-time to the steady
// state on the staggered grid in NavierStokes2D. Starts from
// the current u, v and p stored in ns and leaves the final
// fields there. Per time step u^n is stored once, then for
// every outer iteration:
//
//   1. assemble the momentum equations with the time term and
//      the fluxes of the latest iterate, with implicit
//      under-relaxation alphaU (alphaUFinal in the last outer
//      iteration), and solve them with the current p
//   2. pisoCorrect: the PISO pressure correctors
//   3. relax the pressure change of this outer iteration by
//      alphaP (skipped in the last outer iteration)
//
// With outerIterations = 1 the last-iteration rule makes every
// factor 1 and this is PISO.
//
// Throws std::invalid_argument for non-positive counts, a
// relaxation factor outside (0, 1], dt <= 0 with cfl <= 0, or
// when no reference speed is available (uRef <= 0 and every
// wall speed zero) while dt <= 0. Status Diverged is returned
// (fields may then hold NaN) when a residual is not finite or
// exceeds 1e6.
//--------------------------------------------------

PIMPLEResult solvePIMPLE
(
    NavierStokes2D& ns,
    const PIMPLEOptions& opt
);

const char* pimpleStatusName(PIMPLEStatus s);

#endif