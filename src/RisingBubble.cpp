//------------------------------------------------------------//
// File: RisingBubble.cpp
//
// Description:
// Standalone driver, NOT part of src/ — rising bubble benchmark,
// test case 1 of Hysing et al., "Quantitative benchmark
// computations of two-dimensional bubble dynamics", Int. J. Numer.
// Methods Fluids 60(11), 2009. A bubble of lighter fluid, at rest
// in a closed column of heavier liquid, rises under gravity while
// surface tension and viscosity control its shape.
//
//     domain      : 1 x 2 rectangle
//     bubble      : circle of radius 0.25 at (0.5, 0.5)
//     fluids      : rho1 = 1000, rho2 = 100, mu1 = 10, mu2 = 1
//     g = 0.98, sigma = 24.5   (Re = 35, Eo = 10)
//     walls       : no-slip on top and bottom, free-slip on the sides
//     final time  : t = 3
//
// The interface is a level set (upwind advection, RK3, PDE
// reinitialisation), the flow a projection method on a staggered
// grid, and the pressure equation is solved by the Krylov method
// and preconditioner named on the command line.
//
// Usage:
//     RisingBubble <N> [key=value ...]
//
// N is the number of cells across the width; the grid is N x 2N,
// so cells are square.
//
// Options:
//     tEnd=           final time                             (3)
//     cfl=            fraction of the stable time step       (0.5)
//     dt=             fixed time step (> 0 overrides cfl)    (0)
//     solver=         pressure solver, "<method>+<precond>"  (CG+IC0)
//     tol=            relative tolerance of the pressure solve (1e-10)
//     maxIter=        iteration cap of the pressure solve    (5000)
//     eps=            interface half-width in cells          (1.5)
//     reinit=         reinitialise every n steps (0 = never) (1)
//     reinitSteps=    pseudo-time steps per reinitialisation (15)
//     convection=     upwind | central                       (upwind)
//     record=         record the bubble every n steps        (1)
//     maxUnknowns=    limit on N*2N (dense matrices)         (16000)
//     field=          1 writes the final fields to a file    (0)
//     out=            output file prefix  (rising_N<N>)
//     summary=        summary file                           (summary_rising.csv)
//     print=          print every n steps (0 = silent)       (100)
//
// Writes, in the current directory:
//     <out>_bubble.csv    time, area, areaChange, xc, yc, vRise,
//                         perimeter, circularity, maxSpeed
//     <out>_steps.csv     step, time, dt, iterations, residual,
//                         max |div u|
//     <out>_field.csv     x, y, phi, p, u, v at the cell centres
//                         (only with field=1)
//     summary.csv         one row appended per run (header written
//                         when the file does not exist)
//
// Compared with the finest-grid (h = 1/320) reference of Hysing
// et al. for this test case, group TP2D:
//
//     minimum circularity   0.9013    at t = 1.9041
//     maximum rise velocity 0.2417    at t = 0.9213
//     centre of mass y(3)   1.0813
//
// Compilation Instruction:
// g++ RisingBubble.cpp src/LevelSet.cpp src/TwoPhaseFlow.cpp src/SolverSelector.cpp src/BubbleDiagnostics.cpp src/KrylovSolvers.cpp src/Preconditioner.cpp src/MatrixFunctions.cpp src/MatrixAnalysis.cpp -Iinclude -std=c++17 -O2 -o risingBubble && ./risingBubble 20
//------------------------------------------------------------//

#include "TwoPhaseFlow.h"
#include "SolverSelector.h"
#include "BubbleDiagnostics.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

//------------------------------------------------------------//
// Reference Values (Hysing et al. 2009, test case 1, TP2D)
//------------------------------------------------------------//

static const double refMinCircularity = 0.9013;

static const double refMinCircularityTime = 1.9041;

static const double refMaxRiseVelocity = 0.2417;

static const double refMaxRiseVelocityTime = 0.9213;

static const double refFinalCentreY = 1.0813;

//------------------------------------------------------------//
// Option Handling
//------------------------------------------------------------//

static void usage()
{
    cerr << "Usage: RisingBubble <N> [key=value ...]" << endl
         << "Options: tEnd cfl dt solver tol maxIter eps reinit reinitSteps"
         << " convection record maxUnknowns field out summary print" << endl;
}

static map<string, string> parseOptions
(
    int argc,
    char* argv[],
    int firstOption
)
{
    map<string, string> options;

    for(int k = firstOption; k < argc; k++)
    {
        string arg = argv[k];

        size_t eq = arg.find('=');

        if(eq == string::npos || eq == 0)
        {
            throw invalid_argument("option '"+arg+"' is not of the form key=value");
        }

        options[arg.substr(0, eq)] = arg.substr(eq+1);
    }

    return options;
}

static double getDouble
(
    const map<string, string>& options,
    const string& key,
    double fallback
)
{
    map<string, string>::const_iterator it = options.find(key);

    if(it == options.end())
    {
        return fallback;
    }

    return stod(it->second);
}

static int getInt
(
    const map<string, string>& options,
    const string& key,
    int fallback
)
{
    map<string, string>::const_iterator it = options.find(key);

    if(it == options.end())
    {
        return fallback;
    }

    return stoi(it->second);
}

static string getString
(
    const map<string, string>& options,
    const string& key,
    const string& fallback
)
{
    map<string, string>::const_iterator it = options.find(key);

    if(it == options.end())
    {
        return fallback;
    }

    return it->second;
}

static bool fileExists(const string& name)
{
    ifstream in(name.c_str());

    return in.good();
}

//------------------------------------------------------------//
// Field Output
//------------------------------------------------------------//

static void writeField
(
    const string& name,
    const TwoPhaseFlow& flow
)
{
    ofstream out(name.c_str());

    if(!out)
    {
        throw runtime_error("cannot open '"+name+"'");
    }

    out << "x,y,phi,p,u,v\n";

    out << setprecision(10);

    for(int J = 0; J < flow.Ny; J++)
    {
        for(int I = 0; I < flow.Nx; I++)
        {
            double uc = 0.5*(flow.u[flow.uIdx(I, J)] + flow.u[flow.uIdx(I+1, J)]);
            double vc = 0.5*(flow.v[flow.vIdx(I, J)] + flow.v[flow.vIdx(I, J+1)]);

            out << flow.level.xc(I) << ","
                << flow.level.yc(J) << ","
                << flow.level.phi[flow.level.idx(I, J)] << ","
                << flow.p[flow.pIdx(I, J)] << ","
                << uc << ","
                << vc << "\n";
        }
    }
}

//------------------------------------------------------------//
// Main
//------------------------------------------------------------//

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        usage();

        return 1;
    }

    try
    {
        int N = stoi(argv[1]);

        map<string, string> options = parseOptions(argc, argv, 2);

        double tEnd = getDouble(options, "tEnd", 3.0);
        double cfl = getDouble(options, "cfl", 0.5);
        double dtFixed = getDouble(options, "dt", 0.0);

        int printEvery = getInt(options, "print", 100);
        int recordEvery = getInt(options, "record", 1);
        int maxUnknowns = getInt(options, "maxUnknowns", 16000);

        double epsFactor = getDouble(options, "eps", 1.5);

        string solverName = getString(options, "solver", "CG+IC0");
        string convection = getString(options, "convection", "upwind");
        string out = getString(options, "out", "rising_N"+to_string(N));
        string summaryName = getString(options, "summary", "summary_rising.csv");

        bool writeFields = (getInt(options, "field", 0) != 0);

        if(N < 8)
        {
            throw invalid_argument("N must be at least 8");
        }

        if(tEnd <= 0.0)
        {
            throw invalid_argument("tEnd must be positive");
        }

        if(recordEvery < 1)
        {
            throw invalid_argument("record must be at least 1");
        }

        // Hysing et al. test case 1; the defaults of
        // TwoPhaseParameters already are these values
        TwoPhaseParameters params;

        params.rho1 = 1000.0;
        params.rho2 = 100.0;
        params.mu1 = 10.0;
        params.mu2 = 1.0;
        params.sigma = 24.5;
        params.gx = 0.0;
        params.gy = -0.98;

        params.walls.left = WallType::FreeSlip;
        params.walls.right = WallType::FreeSlip;
        params.walls.bottom = WallType::NoSlip;
        params.walls.top = WallType::NoSlip;

        params.reinitEvery = getInt(options, "reinit", 1);
        params.reinitSteps = getInt(options, "reinitSteps", 15);

        if(convection == "upwind")
        {
            params.convection = TPConvection::Upwind;
        }
        else if(convection == "central")
        {
            params.convection = TPConvection::Central;
        }
        else
        {
            throw invalid_argument("convection must be upwind or central");
        }

        KrylovOptions krylov;

        krylov.tol = getDouble(options, "tol", 1e-10);
        krylov.maxIter = getInt(options, "maxIter", 5000);

        SolverSelector solver(parseSolverList(solverName), krylov);

        TwoPhaseFlow flow(N, 2*N, 1.0, 2.0, params, epsFactor, maxUnknowns);

        flow.initBubble(0.5, 0.5, 0.25);

        BubbleHistory history;

        history.record(flow);

        string stepsName = out+"_steps.csv";

        ofstream stepsFile(stepsName.c_str());

        if(!stepsFile)
        {
            throw runtime_error("cannot open '"+stepsName+"'");
        }

        stepsFile << "step,time,dt,iterations,residual,maxDivergence\n";

        stepsFile << setprecision(10);

        int step = 0;

        int notConverged = 0;

        long totalIterations = 0;

        double maxDivergence = 0.0;

        while(flow.time < tEnd - 1e-12)
        {
            double dt = (dtFixed > 0.0) ? dtFixed : flow.stableTimeStep(cfl);

            // Land exactly on tEnd
            dt = min(dt, tEnd - flow.time);

            TwoPhaseStepInfo info;

            flow.step(dt, solver, info);

            step++;

            if(step%recordEvery == 0)
            {
                history.record(flow);
            }

            if(info.status != KrylovStatus::Converged)
            {
                notConverged++;
            }

            totalIterations += info.iterations;

            maxDivergence = max(maxDivergence, info.maxDivergence);

            stepsFile << step << ","
                      << flow.time << ","
                      << dt << ","
                      << info.iterations << ","
                      << info.trueRelResidual << ","
                      << info.maxDivergence << "\n";

            if(printEvery > 0 && step%printEvery == 0)
            {
                BubbleState s = measureBubble(flow);

                cout << "step " << step
                     << "  t = " << flow.time
                     << "  dt = " << dt
                     << "  yc = " << s.yc
                     << "  vRise = " << s.vRise
                     << "  circ = " << s.circularity
                     << "  iters = " << info.iterations
                     << endl;
            }
        }

        // Make sure the last state is in the history
        if(step%recordEvery != 0)
        {
            history.record(flow);
        }

        history.writeCSV(out+"_bubble.csv");

        if(writeFields)
        {
            writeField(out+"_field.csv", flow);
        }

        double tCirc = 0.0;
        double tRise = 0.0;

        double minCirc = history.minCircularity(tCirc);
        double maxRise = history.maxRiseVelocity(tRise);

        double finalY = history.last().yc;

        double areaChange = history.maxAbsAreaChange();

        string status = (notConverged == 0) ? "ok" : "pressure solve not converged";

        cout << "N = " << N << " x " << 2*N
             << "  solver = " << solverChoiceName(solver.current())
             << "  steps = " << step << endl
             << fixed << setprecision(4)
             << "min circularity   = " << minCirc << " at t = " << tCirc
             << "   (reference " << refMinCircularity << " at t = "
             << refMinCircularityTime << ")" << endl
             << "max rise velocity = " << maxRise << " at t = " << tRise
             << "   (reference " << refMaxRiseVelocity << " at t = "
             << refMaxRiseVelocityTime << ")" << endl
             << "centre of mass y  = " << finalY << " at t = " << flow.time
             << "   (reference " << refFinalCentreY << " at t = 3)" << endl
             << scientific << setprecision(3)
             << "max |area change| = " << areaChange << endl
             << "max |div u|       = " << maxDivergence << endl
             << "pressure solves   = " << totalIterations << " iterations in total"
             << endl
             << "status            = " << status << endl;

        bool writeHeader = !fileExists(summaryName);

        ofstream summary(summaryName.c_str(), ios::app);

        if(!summary)
        {
            throw runtime_error("cannot open '"+summaryName+"'");
        }

        summary << setprecision(10);

        if(writeHeader)
        {
            summary << "N,Nx,Ny,solver,steps,tEnd,minCirc,tMinCirc,maxRise,tMaxRise,"
                    << "finalY,maxAreaChange,maxDivergence,totalIterations,status\n";
        }

        summary << N << ","
                << N << ","
                << 2*N << ","
                << solverChoiceName(solver.current()) << ","
                << step << ","
                << flow.time << ","
                << minCirc << ","
                << tCirc << ","
                << maxRise << ","
                << tRise << ","
                << finalY << ","
                << areaChange << ","
                << maxDivergence << ","
                << totalIterations << ","
                << status << "\n";
    }
    catch(const exception& e)
    {
        cerr << "RisingBubble: " << e.what() << endl;

        usage();

        return 1;
    }

    return 0;
}