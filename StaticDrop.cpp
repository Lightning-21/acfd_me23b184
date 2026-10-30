//------------------------------------------------------------//
// File: StaticDrop.cpp
//
// Description:
// Standalone driver, NOT part of src/ — validation of the two-phase
// flow code on a static circular bubble. A circle of radius R sits
// at rest in a closed unit square, with no gravity. The exact
// solution is a fluid at rest with the Laplace pressure jump
//
//     p_inside - p_outside = sigma / R
//
// so any velocity that appears is a spurious current of the
// continuum surface force, and the measured jump is compared with
// sigma / R. The check exercises the level set (curvature, surface
// force, property fields), the projection and the pressure solve
// together.
//
// Usage:
//     StaticDrop <N> [key=value ...]
//
// N is the number of cells per side (the domain is the unit square,
// the bubble has radius 0.25 and is centred at (0.5, 0.5)).
//
// Options:
//     R=              bubble radius                          (0.25)
//     sigma=          surface tension                        (1)
//     rho1= rho2=     density of liquid / bubble             (1, 1)
//     mu1= mu2=       viscosity of liquid / bubble           (1, 1)
//     steps=          number of time steps                   (200)
//     cfl=            fraction of the stable time step       (0.5)
//     dt=             fixed time step (> 0 overrides cfl)    (0)
//     solver=         pressure solver, "<method>+<precond>"  (CG+IC0)
//     tol=            relative tolerance of the pressure solve (1e-10)
//     maxIter=        iteration cap of the pressure solve    (5000)
//     eps=            interface half-width in cells          (1.5)
//     reinit=         reinitialise every n steps (0 = never) (1)
//     maxUnknowns=    limit on N*N (dense matrices)          (16000)
//     out=            output file prefix  (staticDrop_N<N>)
//     summary=        summary file                           (summary_staticDrop.csv)
//     print=          print every n steps (0 = silent)       (20)
//
// Writes, in the current directory:
//     <out>_history.csv   step, time, pressureJump, relError,
//                         maxSpeed, areaChange, iterations
//     summary.csv         one row appended per run (header written
//                         when the file does not exist)
//
// Reported:
//     jump          measured pressure jump at the last step
//     exact         sigma / R
//     relError      |jump - exact| / exact
//     maxSpeed      largest velocity at the last step (spurious
//                   current; zero in the exact solution)
//     areaChange    relative change of the bubble area
//
// Compilation Instruction:
// g++ StaticDrop.cpp src/LevelSet.cpp src/TwoPhaseFlow.cpp src/SolverSelector.cpp src/BubbleDiagnostics.cpp src/KrylovSolvers.cpp src/Preconditioner.cpp src/MatrixFunctions.cpp src/MatrixAnalysis.cpp -Iinclude -std=c++17 -O2 -o staticDrop && ./staticDrop 32
//------------------------------------------------------------//

#include "TwoPhaseFlow.h"
#include "SolverSelector.h"
#include "BubbleDiagnostics.h"

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
// Option Handling
//------------------------------------------------------------//

static void usage()
{
    cerr << "Usage: StaticDrop <N> [key=value ...]" << endl
         << "Options: R sigma rho1 rho2 mu1 mu2 steps cfl dt solver tol maxIter"
         << " eps reinit maxUnknowns out summary print" << endl;
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

        double R = getDouble(options, "R", 0.25);
        double sigma = getDouble(options, "sigma", 1.0);

        int steps = getInt(options, "steps", 200);
        double cfl = getDouble(options, "cfl", 0.5);
        double dtFixed = getDouble(options, "dt", 0.0);

        int printEvery = getInt(options, "print", 20);
        int reinitEvery = getInt(options, "reinit", 1);
        int maxUnknowns = getInt(options, "maxUnknowns", 16000);

        double epsFactor = getDouble(options, "eps", 1.5);

        string solverName = getString(options, "solver", "CG+IC0");
        string out = getString(options, "out", "staticDrop_N"+to_string(N));
        string summaryName = getString(options, "summary", "summary_staticDrop.csv");

        if(N < 8)
        {
            throw invalid_argument("N must be at least 8");
        }

        if(R <= 0.0 || R >= 0.5)
        {
            throw invalid_argument("R must lie in (0, 0.5) so the bubble fits in the unit square");
        }

        if(steps < 1)
        {
            throw invalid_argument("steps must be at least 1");
        }

        TwoPhaseParameters params;

        params.rho1 = getDouble(options, "rho1", 1.0);
        params.rho2 = getDouble(options, "rho2", 1.0);
        params.mu1 = getDouble(options, "mu1", 1.0);
        params.mu2 = getDouble(options, "mu2", 1.0);
        params.sigma = sigma;
        params.gx = 0.0;
        params.gy = 0.0;
        params.reinitEvery = reinitEvery;

        // Closed box, no-slip on every wall
        params.walls.left = WallType::NoSlip;
        params.walls.right = WallType::NoSlip;
        params.walls.bottom = WallType::NoSlip;
        params.walls.top = WallType::NoSlip;

        KrylovOptions krylov;

        krylov.tol = getDouble(options, "tol", 1e-10);
        krylov.maxIter = getInt(options, "maxIter", 5000);

        SolverSelector solver(parseSolverList(solverName), krylov);

        TwoPhaseFlow flow(N, N, 1.0, 1.0, params, epsFactor, maxUnknowns);

        flow.initBubble(0.5, 0.5, R);

        BubbleHistory history;

        history.record(flow);

        double exact = sigma/R;

        string historyName = out+"_history.csv";

        ofstream histFile(historyName.c_str());

        if(!histFile)
        {
            throw runtime_error("cannot open '"+historyName+"'");
        }

        histFile << "step,time,pressureJump,relError,maxSpeed,areaChange,iterations\n";

        histFile << setprecision(10);

        int notConverged = 0;

        int totalIterations = 0;

        double lastJump = 0.0;

        double lastSpeed = 0.0;

        for(int step = 1; step <= steps; step++)
        {
            double dt = (dtFixed > 0.0) ? dtFixed : flow.stableTimeStep(cfl);

            TwoPhaseStepInfo info;

            flow.step(dt, solver, info);

            history.record(flow);

            if(info.status != KrylovStatus::Converged)
            {
                notConverged++;
            }

            totalIterations += info.iterations;

            lastJump = pressureJump(flow);

            lastSpeed = history.last().maxSpeed;

            double relError = fabs(lastJump - exact)/exact;

            histFile << step << ","
                     << flow.time << ","
                     << lastJump << ","
                     << relError << ","
                     << lastSpeed << ","
                     << history.relativeAreaChange(history.size()-1) << ","
                     << info.iterations << "\n";

            if(printEvery > 0 && step%printEvery == 0)
            {
                cout << "step " << step
                     << "  t = " << flow.time
                     << "  jump = " << lastJump
                     << "  relError = " << relError
                     << "  maxSpeed = " << lastSpeed
                     << "  iters = " << info.iterations
                     << endl;
            }
        }

        double relError = fabs(lastJump - exact)/exact;

        // Jump against sigma / R_eq, with R_eq from the measured area
        double exactEquivalent = sigma/equivalentRadius(history.last().area);

        double relErrorEquivalent = fabs(lastJump - exactEquivalent)/exactEquivalent;

        double areaChange = history.relativeAreaChange(history.size()-1);

        string status = (notConverged == 0) ? "ok" : "pressure solve not converged";

        cout << "N = " << N
             << "  solver = " << solverChoiceName(solver.current()) << endl
             << "steps = " << steps << "  t = " << flow.time << endl
             << "jump       = " << lastJump << endl
             << "exact      = " << exact << "  (sigma/R)" << endl
             << "relError   = " << relError << endl
             << "relError   = " << relErrorEquivalent << "  (against sigma/R_eq)" << endl
             << "maxSpeed   = " << lastSpeed << endl
             << "areaChange = " << areaChange << endl
             << "status     = " << status << endl;

        bool writeHeader = !fileExists(summaryName);

        ofstream summary(summaryName.c_str(), ios::app);

        if(!summary)
        {
            throw runtime_error("cannot open '"+summaryName+"'");
        }

        summary << setprecision(10);

        if(writeHeader)
        {
            summary << "N,solver,steps,time,sigma,R,jump,exact,relError,"
                    << "relErrorEq,maxSpeed,areaChange,totalIterations,status\n";
        }

        summary << N << ","
                << solverChoiceName(solver.current()) << ","
                << steps << ","
                << flow.time << ","
                << sigma << ","
                << R << ","
                << lastJump << ","
                << exact << ","
                << relError << ","
                << relErrorEquivalent << ","
                << lastSpeed << ","
                << areaChange << ","
                << totalIterations << ","
                << status << "\n";
    }
    catch(const exception& e)
    {
        cerr << "StaticDrop: " << e.what() << endl;

        usage();

        return 1;
    }

    return 0;
}