//------------------------------------------------------------//
// File: PISO.cpp
//
// Description:
//
// PISO (Pressure Implicit with Splitting of Operators) for the
// 2D incompressible laminar equations, built on the staggered-
// grid pieces in NavierStokes2D and marched in pseudo-time to
// the steady state so that it can be compared with SIMPLE.
//
// One time step is a predictor (momentum equations with the
// time term, p* from the previous step) followed by pressure
// correctors. The corrector stage is a separate function,
// pisoCorrect, so that PIMPLE can reuse it inside its outer
// iterations.
//
// Corrector k >= 2 uses the velocity increment of the previous
// corrector,
//
//   u^(k) = u^(k-1) + sum a_nb (u^(k-1) - u^(k-2)) / aP
//                   + d (p'_W - p'_E)
//
// with u^(0) = u*. For k = 2 this is exactly u*** of the lecture
// notes. The right-hand side of that pressure-correction
// equation is evaluated as the mass imbalance of the
// intermediate velocity u^, which equals the lecture's source
// (the u**-field is divergence-free) and also absorbs the small
// imbalance left by an inexact CG solve.
//
//------------------------------------------------------------//

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

//------------------------------------------------------------//
// Names
//------------------------------------------------------------//

const char* pisoStatusName(PISOStatus s)
{
    switch(s)
    {
        case PISOStatus::Converged: return "Converged";
        case PISOStatus::MaxSteps:  return "MaxSteps";
        case PISOStatus::Diverged:  return "Diverged";
    }

    return "Unknown";
}

//------------------------------------------------------------//
// Corrector Stage
//------------------------------------------------------------//

PISOCorrectResult pisoCorrect
(
    NavierStokes2D& ns,
    int correctors,
    double pressureTol,
    int pressureMaxIter,
    bool pressurePreconditioner
)
{
    if(correctors < 1 || pressureMaxIter < 1)
    {
        throw invalid_argument("pisoCorrect: correctors and pressureMaxIter must be positive");
    }

    PISOCorrectResult res;

    ns.computeD(false);

    ns.assemblePressureCorrection();

    // u^(k-2) and u^(k-1); both start as the predictor u^(0) = u*
    vector<double> uPrev2 = ns.u;
    vector<double> vPrev2 = ns.v;
    vector<double> uPrev1 = ns.u;
    vector<double> vPrev1 = ns.v;

    vector<double> uh;
    vector<double> vh;
    vector<double> du;
    vector<double> dv;
    vector<double> su;
    vector<double> sv;
    vector<double> b;
    vector<double> pc;

    for(int k = 1; k <= correctors; k++)
    {
        uh = uPrev1;
        vh = vPrev1;

        if(k > 1)
        {
            du = uPrev1;
            dv = vPrev1;

            for(size_t m = 0; m < du.size(); m++)
            {
                du[m] -= uPrev2[m];
            }

            for(size_t m = 0; m < dv.size(); m++)
            {
                dv[m] -= vPrev2[m];
            }

            ns.neighbourSumU(du, su);
            ns.neighbourSumV(dv, sv);

            for(int J = 0; J < ns.Ny; J++)
            {
                for(int i = 1; i < ns.Nx; i++)
                {
                    int q = ns.uIdx(i, J);

                    uh[q] += su[q]/ns.cu.aP[q];
                }
            }

            for(int j = 1; j < ns.Ny; j++)
            {
                for(int I = 0; I < ns.Nx; I++)
                {
                    int q = ns.vIdx(I, j);

                    vh[q] += sv[q]/ns.cv.aP[q];
                }
            }
        }

        double massRaw = ns.massSource(uh, vh, b);

        if(k == 1)
        {
            res.massRaw = massRaw;
        }

        LinearSolveInfo info = ns.solvePressureCorrection
        (
            b,
            pc,
            pressureTol,
            pressureMaxIter,
            pressurePreconditioner
        );

        res.pressureIterations += info.iterations;

        ns.correctVelocities(pc, uh, vh);

        ns.updatePressure(pc, 1.0);

        uPrev2 = uPrev1;
        vPrev2 = vPrev1;

        uPrev1 = uh;
        vPrev1 = vh;
    }

    ns.u = uPrev1;
    ns.v = vPrev1;

    return res;
}

//------------------------------------------------------------//
// PISO
//------------------------------------------------------------//

PISOResult solvePISO
(
    NavierStokes2D& ns,
    const PISOOptions& opt
)
{
    if(opt.correctors < 1 || opt.momentumSweeps < 1 || opt.pressureMaxIter < 1 || opt.maxSteps < 1)
    {
        throw invalid_argument("solvePISO: correctors, sweep and iteration counts must be positive");
    }

    double dt = opt.dt;

    if(dt <= 0.0)
    {
        if(opt.cfl <= 0.0)
        {
            throw invalid_argument("solvePISO: dt <= 0 requires cfl > 0");
        }

        double uRef = opt.uRef;

        if(uRef <= 0.0)
        {
            uRef = max(max(fabs(ns.walls.uBottom), fabs(ns.walls.uTop)), max(fabs(ns.walls.vLeft), fabs(ns.walls.vRight)));
        }

        if(uRef <= 0.0)
        {
            throw invalid_argument("solvePISO: no reference speed available for the CFL time step");
        }

        dt = opt.cfl*min(ns.dx, ns.dy)/uRef;
    }

    double timeCoeff = ns.rho*ns.dx*ns.dy/dt;

    PISOResult res;

    res.dt = dt;

    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();

    double massScale = 0.0;

    for(int step = 1; step <= opt.maxSteps; step++)
    {
        //--------------------------------------------------
        // Predictor
        //--------------------------------------------------

        ns.storeOldTimeLevel();

        ns.assembleMomentum(opt.scheme, 1.0, dt);

        ns.sweepMomentum(opt.momentumSweeps);

        //--------------------------------------------------
        // Correctors
        //--------------------------------------------------

        PISOCorrectResult pr = pisoCorrect
        (
            ns,
            opt.correctors,
            opt.pressureTol,
            opt.pressureMaxIter,
            opt.pressurePreconditioner
        );

        res.pressureIterations += pr.pressureIterations;

        //--------------------------------------------------
        // Residuals: time-derivative term over the steady
        // diagonal term
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
            massScale = max(massScale, pr.massRaw);
        }

        double rm = scaled(pr.massRaw, massScale);

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
            cout << "PISO step " << setw(6) << step
                 << "  resU " << scientific << setprecision(3) << ru
                 << "  resV " << rv
                 << "  mass " << rm
                 << "  CG " << pr.pressureIterations << endl;
        }

        if(!isfinite(ru) || !isfinite(rv) || !isfinite(rm) || ru > 1e6 || rv > 1e6 || rm > 1e6)
        {
            res.status = PISOStatus::Diverged;

            break;
        }

        if(ru < opt.tol && rv < opt.tol && rm < opt.tol)
        {
            res.status = PISOStatus::Converged;

            break;
        }
    }

    res.seconds = chrono::duration<double>(chrono::steady_clock::now() - t0).count();

    return res;
}