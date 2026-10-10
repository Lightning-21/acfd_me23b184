#ifndef PISO_H
#define PISO_H

#include <vector>

#include "NavierStokes2D.h"

enum class PISOStatus
{
    Converged,
    MaxSteps,
    Diverged
};

//--------------------------------------------------
// Corrector Stage
//
// pisoCorrect performs the pressure-correction stage of one
// PISO time step. On entry ns holds the predictor solution
// (u*, v*) in ns.u, ns.v and p* in ns.p, and the momentum
// coefficients cu, cv of that step have been assembled
// (assembleMomentum) and swept (sweepMomentum). It then
//
//   corrector 1 : source = mass imbalance of (u*, v*)
//                 u** = u* + d (p'_W - p'_E),  p** = p* + p'
//   corrector k : u^ = u^(k-1) + sum a_nb (u^(k-1) - u^(k-2)) / aP
//                 source = mass imbalance of u^
//                 u^(k) = u^ + d (p'_W - p'_E), p += p'
//
// with u^(0) = u*, so correctors = 2 is the PISO of the lecture
// notes (u***, v***, p***). The final velocities and the
// pressure are left in ns.u, ns.v and ns.p. No pressure
// under-relaxation is applied.
//
// massRaw is sum |b| of the first corrector (the mass
// imbalance of the predictor); pressureIterations is the total
// CG iterations over all correctors.
//
// Throws std::invalid_argument for correctors < 1 or
// pressureMaxIter < 1.
//--------------------------------------------------

struct PISOCorrectResult
{
    int pressureIterations = 0;

    double massRaw = 0.0;
};

PISOCorrectResult pisoCorrect
(
    NavierStokes2D& ns,
    int correctors,
    double pressureTol,
    int pressureMaxIter,
    bool pressurePreconditioner
);

//--------------------------------------------------
// Options
//
// scheme                 : convection scheme for momentum
// dt                     : pseudo-time step; if <= 0 it is
//                          cfl min(dx, dy) / uRef
// cfl                    : used only when dt <= 0
// uRef                   : reference speed; if <= 0 the largest
//                          wall speed is used
// correctors             : pressure correctors per step (2 =
//                          PISO)
// momentumSweeps         : Gauss-Seidel sweeps of the predictor
// pressureTol            : relative tolerance of every p' solve
// pressureMaxIter        : iteration cap of every p' solve
// pressurePreconditioner : symmetric Gauss-Seidel on / off
// tol                    : stop when the scaled u, v and mass
//                          residuals are all below tol
// maxSteps               : time-step cap
// printEvery             : print a line every n steps (0 = silent)
//--------------------------------------------------

struct PISOOptions
{
    ConvectionScheme scheme = ConvectionScheme::Hybrid;

    double dt = 0.0;

    double cfl = 1.0;

    double uRef = 0.0;

    int correctors = 2;

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
// Scaled residuals (history has one entry per time step), the
// same definitions as the steady SIMPLE residuals so the two
// algorithms are comparable:
//
//   momentum : the time-derivative term of the step,
//              sum (rho V/dt) |u^(n+1) - u^n|, which equals the
//              steady momentum residual when the step is fully
//              solved, divided by sum |aP_steady u^(n+1)|
//   mass     : sum |b| of the first corrector divided by the
//              largest value seen in the first five steps
//--------------------------------------------------

struct PISOResult
{
    PISOStatus status = PISOStatus::MaxSteps;

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
// PISO
//
// Pressure Implicit with Splitting of Operators, marched in
// pseudo-time to the steady state on the staggered grid in
// NavierStokes2D. Starts from the current u, v and p stored in
// ns and leaves the final fields there. Per time step:
//
//   1. store u^n; assemble the momentum equations with the time
//      term (no under-relaxation) and solve them with the
//      current p -> u*, v*   (predictor)
//   2. pisoCorrect: the pressure correctors
//
// Throws std::invalid_argument for non-positive counts, for
// dt <= 0 with cfl <= 0, or when no reference speed is
// available (uRef <= 0 and every wall speed zero) while dt <= 0.
// Status Diverged is returned (fields may then hold NaN) when a
// residual is not finite or exceeds 1e6.
//--------------------------------------------------

PISOResult solvePISO
(
    NavierStokes2D& ns,
    const PISOOptions& opt
);

const char* pisoStatusName(PISOStatus s);

#endif