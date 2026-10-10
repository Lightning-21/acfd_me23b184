//------------------------------------------------------------//
// File: PIMPLE.cpp
//
// Description:
//
// PIMPLE (merged PISO-SIMPLE) for the 2D incompressible laminar
// equations, built on the staggered-grid pieces in NavierStokes2D
// and on the PISO corrector stage, marched in pseudo-time to the
// steady state.
//
// Every time step runs several outer iterations. Each one
// re-assembles the momentum equations from the latest iterate
// (the fluxes are therefore updated inside the step, unlike
// PISO), solves them with implicit under-relaxation, and then
// performs the PISO pressure correctors. The pressure change of
// an outer iteration is under-relaxed as in SIMPLE. The last
// outer iteration uses alphaUFinal (1 by default) and no pressure
// relaxation, so the converged step is unrelaxed and a single
// outer iteration is exactly PISO.
//
// Pressure relaxation is applied after pisoCorrect returns:
// the correctors only use p' (never p) to build the velocities,
// so relaxing the total pressure change at the end equals
// relaxing every p' by alphaP.
//
// The residuals are defined as in PISO.
//
//------------------------------------------------------------//

#include "PIMPLE.h"

#include "PISO.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace std;

//------------------------------------------------------------//
// File-Local Helpers
//------------------------------------------------------------//

// Scaled residual; 1 when the scale is zero (field still zero)
static double scaled
(
    double raw,
    double scale
)
{
    if(scale > 1e-300)
    {
        return raw/scale;
    }

    return 1.0;
}

static bool validRelaxation(double a)
{
    return a > 0.0 && a <= 1.0;
}

//------------------------------------------------------------//
// Names
//------------------------------------------------------------//

const char* pimpleStatusName(PIMPLEStatus s)
{
    switch(s)
    {
        case PIMPLEStatus::Converged: return "Converged";
        case PIMPLEStatus::MaxSteps:  return "MaxSteps";
        case PIMPLEStatus::Diverged:  return "Diverged";
    }

    return "Unknown";
}

//------------------------------------------------------------//
// PIMPLE
//------------------------------------------------------------//

PIMPLEResult solvePIMPLE
(
    NavierStokes2D& ns,
    const PIMPLEOptions& opt
)
{
    if(!validRelaxation(opt.alphaU) || !validRelaxation(opt.alphaP) || !validRelaxation(opt.alphaUFinal))
    {
        throw invalid_argument("solvePIMPLE: alphaU, alphaP and alphaUFinal must be in (0, 1]");
    }

    if(opt.outerIterations < 1 || opt.correctors < 1 || opt.momentumSweeps < 1 || opt.pressureMaxIter < 1 || opt.maxSteps < 1)
    {
        throw invalid_argument("solvePIMPLE: outer iterations, correctors, sweep and iteration counts must be positive");
    }

    double dt = opt.dt;

    if(dt <= 0.0)
    {
        if(opt.cfl <= 0.0)
        {
            throw invalid_argument("solvePIMPLE: dt <= 0 requires cfl > 0");
        }

        double uRef = opt.uRef;

        if(uRef <= 0.0)
        {
            uRef = max(max(fabs(ns.walls.uBottom), fabs(ns.walls.uTop)), max(fabs(ns.walls.vLeft), fabs(ns.walls.vRight)));
        }

        if(uRef <= 0.0)
        {
            throw invalid_argument("solvePIMPLE: no reference speed available for the CFL time step");
        }

        dt = opt.cfl*min(ns.dx, ns.dy)/uRef;
    }

    double timeCoeff = ns.rho*ns.dx*ns.dy/dt;

    PIMPLEResult res;

    res.dt = dt;

    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();

    double massScale = 0.0;

    vector<double> pStar;

    for(int step = 1; step <= opt.maxSteps; step++)
    {
        ns.storeOldTimeLevel();

        double massRaw = 0.0;
        int lastCG = 0;

        //--------------------------------------------------
        // Outer iterations
        //--------------------------------------------------

        for(int outer = 1; outer <= opt.outerIterations; outer++)
        {
            bool last = (outer == opt.outerIterations);

            double aU = last ? opt.alphaUFinal : opt.alphaU;
            double aP = last ? 1.0 : opt.alphaP;

            // Predictor
            ns.assembleMomentum(opt.scheme, aU, dt);

            ns.sweepMomentum(opt.momentumSweeps);

            pStar = ns.p;

            // Correctors
            PISOCorrectResult pr = pisoCorrect
            (
                ns,
                opt.correctors,
                opt.pressureTol,
                opt.pressureMaxIter,
                opt.pressurePreconditioner
            );

            res.pressureIterations += pr.pressureIterations;

            // Pressure relaxation of this outer iteration
            if(aP < 1.0)
            {
                for(size_t k = 0; k < ns.p.size(); k++)
                {
                    ns.p[k] = pStar[k] + aP*(ns.p[k]-pStar[k]);
                }
            }

            massRaw = pr.massRaw;
            lastCG = pr.pressureIterations;
        }

        //--------------------------------------------------
        // Residuals: time-derivative term over the steady
        // diagonal term (coefficients of the last outer
        // iteration)
        //--------------------------------------------------

        double numU = 0.0;
        double denU = 0.0;
        double numV = 0.0;
        double denV = 0.0;

        for(int J = 0; J < ns.Ny; J++)
        {
            for(int i = 1; i < ns.Nx; i++)
            {
                int k = ns.uIdx(i, J);

                numU += fabs(ns.u[k]-ns.uOld[k]);
                denU += fabs((ns.cu.aP0[k]-timeCoeff)*ns.u[k]);
            }
        }

        for(int j = 1; j < ns.Ny; j++)
        {
            for(int I = 0; I < ns.Nx; I++)
            {
                int k = ns.vIdx(I, j);

                numV += fabs(ns.v[k]-ns.vOld[k]);
                denV += fabs((ns.cv.aP0[k]-timeCoeff)*ns.v[k]);
            }
        }

        double ru = scaled(timeCoeff*numU, denU);
        double rv = scaled(timeCoeff*numV, denV);

        if(step <= 5)
        {
            massScale = max(massScale, massRaw);
        }

        double rm = scaled(massRaw, massScale);

        //--------------------------------------------------
        // Bookkeeping
        //--------------------------------------------------

        res.steps = step;
        res.time = step*dt;
        res.resU = ru;
        res.resV = rv;
        res.resMass = rm;

        res.historyU.push_back(ru);
        res.historyV.push_back(rv);
        res.historyMass.push_back(rm);

        if(opt.printEvery > 0 && (step == 1 || step%opt.printEvery == 0))
        {
            cout << "PIMPLE step " << setw(6) << step
                 << "  resU " << scientific << setprecision(3) << ru
                 << "  resV " << rv
                 << "  mass " << rm
                 << "  CG(last outer) " << lastCG << endl;
        }

        if(!isfinite(ru) || !isfinite(rv) || !isfinite(rm) || ru > 1e6 || rv > 1e6 || rm > 1e6)
        {
            res.status = PIMPLEStatus::Diverged;

            break;
        }

        if(ru < opt.tol && rv < opt.tol && rm < opt.tol)
        {
            res.status = PIMPLEStatus::Converged;

            break;
        }
    }

    res.seconds = chrono::duration<double>(chrono::steady_clock::now() - t0).count();

    return res;
}