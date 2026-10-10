#ifndef SIMPLE_H
#define SIMPLE_H

#include <vector>

#include "NavierStokes2D.h"

enum class SIMPLEStatus
{
    Converged,
    MaxIterations,
    Diverged
};

//--------------------------------------------------
// Options
//
// scheme                 : convection scheme for momentum
// alphaU, alphaP         : under-relaxation of velocity (inside
//                          the momentum equation) and of the
//                          pressure update, both in (0, 1]
// momentumSweeps         : Gauss-Seidel sweeps per iteration
// pressureTol            : relative tolerance of the p' solve
// pressureMaxIter        : iteration cap of the p' solve
// pressurePreconditioner : symmetric Gauss-Seidel on / off
// tol                    : stop when the scaled u, v and mass
//                          residuals are all below tol
// maxIter                : outer iteration cap
// printEvery             : print a line every n iterations
//                          (0 = silent)
//--------------------------------------------------

struct SIMPLEOptions
{
    ConvectionScheme scheme = ConvectionScheme::Hybrid;

    double alphaU = 0.7;

    double alphaP = 0.3;

    int momentumSweeps = 3;

    double pressureTol = 1e-2;

    int pressureMaxIter = 1000;

    bool pressurePreconditioner = true;

    double tol = 1e-6;

    int maxIter = 20000;

    int printEvery = 0;
};

//--------------------------------------------------
// Result
//
// Scaled residuals (history has one entry per iteration):
//
//   momentum : sum |aP0 u_P - sum a_nb u_nb - b0 - pressure force|
//              / sum |aP0 u_P|, evaluated on the fields at the
//              start of the iteration (1 while u is still zero)
//   mass     : sum |b| of the pressure-correction equation,
//              evaluated after the momentum solve and before the
//              correction, divided by the largest value seen in
//              the first five iterations
//
// pressureIterations is the total number of CG iterations over
// all outer iterations.
//--------------------------------------------------

struct SIMPLEResult
{
    SIMPLEStatus status = SIMPLEStatus::MaxIterations;

    int iterations = 0;

    int pressureIterations = 0;

    double seconds = 0.0;

    double resU = 0.0;

    double resV = 0.0;

    double resMass = 0.0;

    std::vector<double> historyU;

    std::vector<double> historyV;

    std::vector<double> historyMass;
};

//--------------------------------------------------
// SIMPLE
//
// Semi-Implicit Method for Pressure-Linked Equations, steady,
// on the staggered grid in NavierStokes2D. Starts from the
// current u, v and p stored in ns and leaves the final fields
// there. Per iteration:
//
//   1. assemble and solve the momentum equations with the
//      current p -> u*, v*
//   2. assemble and solve the pressure-correction equation
//      with the mass imbalance of u*, v* as source -> p'
//   3. p = p* + alphaP p', u = u* + d (p'_W - p'_E),
//      v = v* + d (p'_S - p'_N)
//
// Throws std::invalid_argument for a relaxation factor outside
// (0, 1] or non-positive counts. Status Diverged is returned
// (fields may then hold NaN) when a residual is not finite or
// exceeds 1e6.
//--------------------------------------------------

SIMPLEResult solveSIMPLE
(
    NavierStokes2D& ns,
    const SIMPLEOptions& opt
);

const char* simpleStatusName(SIMPLEStatus s);

#endif