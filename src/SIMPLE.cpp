//------------------------------------------------------------//
// File: SIMPLE.cpp
//
// Description:
//
// SIMPLE (Semi-Implicit Method for Pressure-Linked Equations)
// for the steady 2D incompressible laminar equations, built on
// the staggered-grid pieces in NavierStokes2D. One outer
// iteration is
//
//   1. momentum equations with the current pressure p*
//      (implicit under-relaxation alphaU, a few Gauss-Seidel
//      sweeps) -> u*, v*
//   2. pressure-correction equation, source = mass imbalance of
//      u*, v*, d = A/aP, solved by (preconditioned) CG -> p'
//   3. p = p* + alphaP p'; u and v corrected with the full p'
//
// No other transport equation is solved (laminar, constant
// properties). The residuals reported are the scaled residuals
// described in SIMPLE.h.
//
//------------------------------------------------------------//

#include "SIMPLE.h"

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

// sum |aP0 u| over the interior u faces
static double sumAbsAPu(const NavierStokes2D& ns)
{
    double s = 0.0;

    for(int J = 0; J < ns.Ny; J++)
    {
        for(int i = 1; i < ns.Nx; i++)
        {
            int k = ns.uIdx(i, J);

            s += fabs(ns.cu.aP0[k]*ns.u[k]);
        }
    }

    return s;
}

// sum |aP0 v| over the interior v faces
static double sumAbsAPv(const NavierStokes2D& ns)
{
    double s = 0.0;

    for(int j = 1; j < ns.Ny; j++)
    {
        for(int I = 0; I < ns.Nx; I++)
        {
            int k = ns.vIdx(I, j);

            s += fabs(ns.cv.aP0[k]*ns.v[k]);
        }
    }

    return s;
}

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

//------------------------------------------------------------//
// Names
//------------------------------------------------------------//

const char* simpleStatusName(SIMPLEStatus s)
{
    switch(s)
    {
        case SIMPLEStatus::Converged:     return "Converged";
        case SIMPLEStatus::MaxIterations: return "MaxIterations";
        case SIMPLEStatus::Diverged:      return "Diverged";
    }

    return "Unknown";
}

//------------------------------------------------------------//
// SIMPLE
//------------------------------------------------------------//

SIMPLEResult solveSIMPLE
(
    NavierStokes2D& ns,
    const SIMPLEOptions& opt
)
{
    if(!(opt.alphaU > 0.0 && opt.alphaU <= 1.0))
    {
        throw invalid_argument("solveSIMPLE: alphaU must be in (0, 1]");
    }

    if(!(opt.alphaP > 0.0 && opt.alphaP <= 1.0))
    {
        throw invalid_argument("solveSIMPLE: alphaP must be in (0, 1]");
    }

    if(opt.momentumSweeps < 1 || opt.pressureMaxIter < 1 || opt.maxIter < 1)
    {
        throw invalid_argument("solveSIMPLE: sweep and iteration counts must be positive");
    }

    SIMPLEResult res;

    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();

    vector<double> b;
    vector<double> pc;

    double massScale = 0.0;

    for(int it = 1; it <= opt.maxIter; it++)
    {
        //--------------------------------------------------
        // 1. Momentum
        //--------------------------------------------------

        ns.assembleMomentum(opt.scheme, opt.alphaU, 0.0);

        double ru = scaled(ns.residualU(), sumAbsAPu(ns));
        double rv = scaled(ns.residualV(), sumAbsAPv(ns));

        ns.sweepMomentum(opt.momentumSweeps);

        //--------------------------------------------------
        // 2. Pressure correction
        //--------------------------------------------------

        ns.computeD(false);

        ns.assemblePressureCorrection();

        double massRaw = ns.massSource(ns.u, ns.v, b);

        if(it <= 5)
        {
            massScale = max(massScale, massRaw);
        }

        double rm = scaled(massRaw, massScale);

        LinearSolveInfo info = ns.solvePressureCorrection
        (
            b,
            pc,
            opt.pressureTol,
            opt.pressureMaxIter,
            opt.pressurePreconditioner
        );

        res.pressureIterations += info.iterations;

        //--------------------------------------------------
        // 3. Correction
        //--------------------------------------------------

        ns.correctVelocities(pc, ns.u, ns.v);

        ns.updatePressure(pc, opt.alphaP);

        //--------------------------------------------------
        // Bookkeeping
        //--------------------------------------------------

        res.iterations = it;
        res.resU = ru;
        res.resV = rv;
        res.resMass = rm;

        res.historyU.push_back(ru);
        res.historyV.push_back(rv);
        res.historyMass.push_back(rm);

        if(opt.printEvery > 0 && (it == 1 || it%opt.printEvery == 0))
        {
            cout << "SIMPLE it " << setw(6) << it
                 << "  resU " << scientific << setprecision(3) << ru
                 << "  resV " << rv
                 << "  mass " << rm
                 << "  CG " << info.iterations << endl;
        }

        if(!isfinite(ru) || !isfinite(rv) || !isfinite(rm) || ru > 1e6 || rv > 1e6 || rm > 1e6)
        {
            res.status = SIMPLEStatus::Diverged;

            break;
        }

        if(ru < opt.tol && rv < opt.tol && rm < opt.tol)
        {
            res.status = SIMPLEStatus::Converged;

            break;
        }
    }

    res.seconds = chrono::duration<double>(chrono::steady_clock::now() - t0).count();

    return res;
}