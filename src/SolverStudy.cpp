//------------------------------------------------------------//
// File: SolverStudy.cpp
//
// Description:
// Standalone driver, NOT part of src/ — solver study for the
// pressure Poisson equation of the rising bubble code. Every run
// marches the Hysing et al. test case 1 bubble (same geometry,
// walls, viscosities, gravity and surface tension as
// RisingBubble.cpp) for a fixed number of time steps and records
// what the pressure solves cost. The density ratio rho1/rho2 is a
// parameter: rho1 = 1000 and rho2 = 1000/ratio, so ratio 1, 10,
// 100, 1000 raises the contrast from none to the extreme of the
// proposal. No reference data exists for the ratios other than
// 10, so these runs only compare solvers.
//
// Two modes:
//
//     sweep     every solver in the list, for every grid size and
//               density ratio: iterations, matrix-vector products,
//               preconditioner applications, setup / iteration /
//               total time against grid size and density ratio
//
//     adaptive  the candidate list, ordered weakest first, is run
//               once as each fixed solver and once with the
//               iteration-count switching policy of SolverSelector.
//               The adaptive run is accepted only if its physical
//               results (final centre of mass and rise velocity)
//               agree with the best fixed run to within physTol;
//               if it brings no gain, that is the result reported
//
// Usage:
//     SolverStudy <sweep|adaptive> [key=value ...]
//
// Options:
//     N=              list of cells across the width; the grid
//                     is N x 2N                               (20)
//     ratios=         list of density ratios         (1,10,100,1000)
//     steps=          time steps per run                      (200)
//     cfl=            fraction of the stable time step at t = 0 (0.5)
//     dt=             fixed time step (> 0 overrides cfl)     (0)
//     solvers=        comma-separated "<method>+<precond>" list
//                     sweep:    CG+Jacobi,CG+SGS,CG+IC0,BiCGSTAB+Jacobi,
//                               BiCGSTAB+SGS,BiCGSTAB+ILU0,BiCGSTAB+IC0
//                     adaptive: CG+Jacobi,CG+SGS,CG+IC0
//     tol=            relative tolerance of the pressure solve (1e-10)
//     maxIter=        iteration cap of the pressure solve     (5000)
//     eps=            interface half-width in cells           (1.5)
//     mu1= mu2=       viscosity of liquid / bubble            (10, 1)
//     maxUnknowns=    limit on N*2N (dense matrices)          (16000)
//     up= down=       switching policy: move up above this many
//                     iterations, count a solve as easy below
//                     this many                               (60, 15)
//     patience=       easy solves in a row before moving down (3)
//     hold=           minimum solves between two switches     (5)
//     physTol=        adaptive mode: largest relative difference
//                     of final centre of mass and rise velocity
//                     from the best fixed run                 (1e-6)
//     summary=        run summary file     (summary_solverStudy.csv)
//     decision=       adaptive mode: decision file
//                                          (summary_switching.csv)
//     print=          1 prints one line per run, 0 is silent  (1)
//
// All runs of one grid size and density ratio use the SAME time
// step (from the stable time step at t = 0, or dt=), so the
// velocity fields differ only at the solver tolerance and the
// runs can be compared like for like. The largest advective CFL
// number reached is reported as maxCFL; a value above 1 means the
// fixed dt was too large for that case.
//
// A solver that cannot run (for example IC(0) on a matrix that is
// not SPD, or CG with a non-symmetric preconditioner) is recorded
// as failed and the study continues.
//
// Writes, in the current directory (rows are appended, the header
// is written when the file does not exist):
//     summary_solverStudy.csv   one row per run
//     summary_switching.csv     one row per adaptive comparison
//
// Compilation Instruction:
// g++ SolverStudy.cpp src/LevelSet.cpp src/TwoPhaseFlow.cpp src/SolverSelector.cpp src/BubbleDiagnostics.cpp src/KrylovSolvers.cpp src/Preconditioner.cpp src/MatrixFunctions.cpp src/MatrixAnalysis.cpp -Iinclude -std=c++17 -O2 -o solverStudy && ./solverStudy sweep N=20 steps=50
//------------------------------------------------------------//

#include "TwoPhaseFlow.h"
#include "SolverSelector.h"
#include "BubbleDiagnostics.h"

#include <algorithm>
#include <chrono>
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
    cerr << "Usage: SolverStudy <sweep|adaptive> [key=value ...]" << endl
         << "Options: N ratios steps cfl dt solvers tol maxIter eps mu1 mu2"
         << " maxUnknowns up down patience hold physTol summary decision print"
         << endl;
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

static vector<string> splitList(const string& text)
{
    vector<string> items;

    size_t start = 0;

    while(start <= text.size())
    {
        size_t comma = text.find(',', start);

        if(comma == string::npos)
        {
            comma = text.size();
        }

        string item = text.substr(start, comma-start);

        if(!item.empty())
        {
            items.push_back(item);
        }

        start = comma+1;
    }

    return items;
}

static bool fileExists(const string& name)
{
    ifstream in(name.c_str());

    return in.good();
}

//------------------------------------------------------------//
// One Run
//------------------------------------------------------------//

struct CaseSettings
{
    int N = 20;

    double ratio = 10.0;

    int steps = 200;

    double cfl = 0.5;

    // > 0: fixed time step; <= 0: from the stable time step at t = 0
    double dt = 0.0;

    double epsFactor = 1.5;

    double mu1 = 10.0;

    double mu2 = 1.0;

    double tol = 1e-10;

    int maxIter = 5000;

    int maxUnknowns = 16000;
};

struct CaseResult
{
    bool failed = false;

    // The time step the run used
    double dt = 0.0;

    long iterations = 0;

    long matVecs = 0;

    long precondApplies = 0;

    double setupSeconds = 0.0;

    double solveSeconds = 0.0;

    // Pressure solves only: setup + iteration
    double totalSeconds = 0.0;

    // The whole run, including assembly and the level set
    double wallSeconds = 0.0;

    // Largest true relative residual of the pressure solves
    double maxResidual = 0.0;

    double maxCFL = 0.0;

    int notConverged = 0;

    int switches = 0;

    int fallbacks = 0;

    // Final state and the smallest circularity of the run
    BubbleState state;

    double minCirc = 0.0;

    string status = "ok";
};

static CaseResult runCase
(
    const CaseSettings& cs,
    const vector<SolverChoice>& candidates,
    const SwitchPolicy& policy
)
{
    CaseResult res;

    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();

    try
    {
        // Hysing et al. test case 1, with rho2 = rho1/ratio
        TwoPhaseParameters params;

        params.rho1 = 1000.0;
        params.rho2 = 1000.0/cs.ratio;
        params.mu1 = cs.mu1;
        params.mu2 = cs.mu2;
        params.sigma = 24.5;
        params.gx = 0.0;
        params.gy = -0.98;

        params.walls.left = WallType::FreeSlip;
        params.walls.right = WallType::FreeSlip;
        params.walls.bottom = WallType::NoSlip;
        params.walls.top = WallType::NoSlip;

        KrylovOptions krylov;

        krylov.tol = cs.tol;
        krylov.maxIter = cs.maxIter;

        SolverSelector solver(candidates, krylov, policy);

        TwoPhaseFlow flow(cs.N, 2*cs.N, 1.0, 2.0, params, cs.epsFactor, cs.maxUnknowns);

        flow.initBubble(0.5, 0.5, 0.25);

        double dt = (cs.dt > 0.0) ? cs.dt : flow.stableTimeStep(cs.cfl);

        res.dt = dt;

        BubbleHistory history;

        history.record(flow);

        for(int step = 1; step <= cs.steps; step++)
        {
            TwoPhaseStepInfo info;

            flow.step(dt, solver, info);

            history.record(flow);

            if(info.status != KrylovStatus::Converged)
            {
                res.notConverged++;
            }

            res.maxResidual = max(res.maxResidual, info.trueRelResidual);

            double uMax = 0.0;
            double vMax = 0.0;

            for(size_t k = 0; k < flow.u.size(); k++)
            {
                uMax = max(uMax, fabs(flow.u[k]));
            }

            for(size_t k = 0; k < flow.v.size(); k++)
            {
                vMax = max(vMax, fabs(flow.v[k]));
            }

            res.maxCFL = max(res.maxCFL, dt*(uMax/flow.dx + vMax/flow.dy));
        }

        res.iterations = solver.totalIterations();
        res.matVecs = solver.totalMatVecs();
        res.precondApplies = solver.totalPrecondApplies();
        res.setupSeconds = solver.totalSetupSeconds();
        res.totalSeconds = solver.totalSeconds();
        res.solveSeconds = res.totalSeconds - res.setupSeconds;
        res.switches = solver.numSwitches();
        res.fallbacks = solver.numFallbacks();

        double timeOfMin = 0.0;

        res.minCirc = history.minCircularity(timeOfMin);
        res.state = history.last();

        if(res.notConverged > 0)
        {
            res.status = "pressure solve not converged";
        }
    }
    catch(const exception& e)
    {
        res.failed = true;
        res.status = string("failed: ")+e.what();
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    res.wallSeconds = chrono::duration<double>(t1-t0).count();

    return res;
}

//------------------------------------------------------------//
// Labels and Output
//------------------------------------------------------------//

static string adaptiveLabel(const vector<SolverChoice>& candidates)
{
    string label = "adaptive[";

    for(size_t k = 0; k < candidates.size(); k++)
    {
        if(k > 0)
        {
            label += ">";
        }

        label += solverChoiceName(candidates[k]);
    }

    return label+"]";
}

// Commas in a status message would break the CSV
static string csvSafe(const string& text)
{
    string out = text;

    for(size_t k = 0; k < out.size(); k++)
    {
        if(out[k] == ',' || out[k] == '\n')
        {
            out[k] = ';';
        }
    }

    return out;
}

static void writeRunRow
(
    ofstream& file,
    const string& mode,
    const CaseSettings& cs,
    const string& variant,
    const CaseResult& r
)
{
    file << mode << ","
         << cs.N << ","
         << cs.N << ","
         << 2*cs.N << ","
         << cs.ratio << ","
         << variant << ","
         << cs.steps << ","
         << r.dt << ","
         << r.iterations << ","
         << r.matVecs << ","
         << r.precondApplies << ","
         << r.setupSeconds << ","
         << r.solveSeconds << ","
         << r.totalSeconds << ","
         << r.wallSeconds << ","
         << r.maxResidual << ","
         << r.maxCFL << ","
         << r.notConverged << ","
         << r.switches << ","
         << r.fallbacks << ","
         << r.state.yc << ","
         << r.state.vRise << ","
         << r.minCirc << ","
         << csvSafe(r.status) << "\n";
}

static void printRun
(
    const CaseSettings& cs,
    const string& variant,
    const CaseResult& r
)
{
    cout << "N = " << cs.N << "x" << 2*cs.N
         << "  ratio = " << cs.ratio
         << "  " << variant << "  ";

    if(r.failed)
    {
        cout << r.status << endl;

        return;
    }

    cout << "iters = " << r.iterations
         << "  setup = " << r.setupSeconds << " s"
         << "  solve = " << r.solveSeconds << " s"
         << "  total = " << r.totalSeconds << " s"
         << "  maxCFL = " << r.maxCFL;

    if(r.switches > 0 || r.fallbacks > 0)
    {
        cout << "  switches = " << r.switches << "  fallbacks = " << r.fallbacks;
    }

    cout << "  " << r.status << endl;
}

static void writeHeaderIfNew
(
    const string& name,
    const string& header
)
{
    if(fileExists(name))
    {
        return;
    }

    ofstream file(name.c_str());

    if(!file)
    {
        throw runtime_error("cannot open '"+name+"'");
    }

    file << header << "\n";
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
        string mode = argv[1];

        if(mode != "sweep" && mode != "adaptive")
        {
            throw invalid_argument("mode must be sweep or adaptive");
        }

        map<string, string> options = parseOptions(argc, argv, 2);

        // Grid sizes and density ratios
        vector<int> gridSizes;

        vector<string> nItems = splitList(getString(options, "N", "20"));

        for(size_t k = 0; k < nItems.size(); k++)
        {
            int n = stoi(nItems[k]);

            if(n < 8)
            {
                throw invalid_argument("every N must be at least 8");
            }

            gridSizes.push_back(n);
        }

        vector<double> ratios;

        vector<string> ratioItems = splitList(getString(options, "ratios", "1,10,100,1000"));

        for(size_t k = 0; k < ratioItems.size(); k++)
        {
            double ratio = stod(ratioItems[k]);

            if(ratio <= 0.0)
            {
                throw invalid_argument("every density ratio must be positive");
            }

            ratios.push_back(ratio);
        }

        if(gridSizes.empty() || ratios.empty())
        {
            throw invalid_argument("N and ratios must not be empty");
        }

        // Settings shared by every run
        CaseSettings base;

        base.steps = getInt(options, "steps", 200);
        base.cfl = getDouble(options, "cfl", 0.5);
        base.dt = getDouble(options, "dt", 0.0);
        base.epsFactor = getDouble(options, "eps", 1.5);
        base.mu1 = getDouble(options, "mu1", 10.0);
        base.mu2 = getDouble(options, "mu2", 1.0);
        base.tol = getDouble(options, "tol", 1e-10);
        base.maxIter = getInt(options, "maxIter", 5000);
        base.maxUnknowns = getInt(options, "maxUnknowns", 16000);

        if(base.steps < 1)
        {
            throw invalid_argument("steps must be at least 1");
        }

        string defaultSolvers =
            (mode == "sweep")
          ? "CG+Jacobi,CG+SGS,CG+IC0,BiCGSTAB+Jacobi,BiCGSTAB+SGS,BiCGSTAB+ILU0,BiCGSTAB+IC0"
          : "CG+Jacobi,CG+SGS,CG+IC0";

        vector<SolverChoice> candidates = parseSolverList(getString(options, "solvers", defaultSolvers));

        SwitchPolicy policy;

        policy.adaptive = true;
        policy.upIters = getInt(options, "up", 60);
        policy.downIters = getInt(options, "down", 15);
        policy.patience = getInt(options, "patience", 3);
        policy.holdSteps = getInt(options, "hold", 5);

        double physTol = getDouble(options, "physTol", 1e-6);

        string summaryName = getString(options, "summary", "summary_solverStudy.csv");
        string decisionName = getString(options, "decision", "summary_switching.csv");

        bool printRuns = (getInt(options, "print", 1) != 0);

        writeHeaderIfNew
        (
            summaryName,
            "mode,N,Nx,Ny,ratio,variant,steps,dt,iterations,matVecs,precondApplies,"
            "setupSeconds,solveSeconds,totalSeconds,wallSeconds,maxResidual,maxCFL,"
            "notConverged,switches,fallbacks,yc,vRise,minCirc,status"
        );

        if(mode == "adaptive")
        {
            writeHeaderIfNew
            (
                decisionName,
                "N,Nx,Ny,ratio,bestFixed,bestFixedSeconds,adaptiveSeconds,"
                "bestFixedIterations,adaptiveIterations,secondsRatio,"
                "relDiffYc,relDiffVRise,physicalOK,switches,fallbacks,verdict"
            );
        }

        ofstream summary(summaryName.c_str(), ios::app);

        if(!summary)
        {
            throw runtime_error("cannot open '"+summaryName+"'");
        }

        summary << setprecision(10);

        ofstream decision;

        if(mode == "adaptive")
        {
            decision.open(decisionName.c_str(), ios::app);

            if(!decision)
            {
                throw runtime_error("cannot open '"+decisionName+"'");
            }

            decision << setprecision(10);
        }

        cout << setprecision(4);

        for(size_t gi = 0; gi < gridSizes.size(); gi++)
        {
            for(size_t ri = 0; ri < ratios.size(); ri++)
            {
                CaseSettings cs = base;

                cs.N = gridSizes[gi];
                cs.ratio = ratios[ri];

                // The first run of a case fixes the time step for all
                // the others, so that they can be compared like for like
                bool dtKnown = (cs.dt > 0.0);

                if(mode == "sweep")
                {
                    for(size_t k = 0; k < candidates.size(); k++)
                    {
                        vector<SolverChoice> one(1, candidates[k]);

                        SwitchPolicy fixedPolicy;

                        CaseResult r = runCase(cs, one, fixedPolicy);

                        if(!dtKnown && r.dt > 0.0)
                        {
                            cs.dt = r.dt;

                            dtKnown = true;
                        }

                        string variant = solverChoiceName(candidates[k]);

                        writeRunRow(summary, mode, cs, variant, r);

                        summary.flush();

                        if(printRuns)
                        {
                            printRun(cs, variant, r);
                        }
                    }

                    continue;
                }

                // Adaptive mode: every candidate fixed, then adaptive
                vector<CaseResult> fixedResults;

                for(size_t k = 0; k < candidates.size(); k++)
                {
                    vector<SolverChoice> one(1, candidates[k]);

                    SwitchPolicy fixedPolicy;

                    CaseResult r = runCase(cs, one, fixedPolicy);

                    if(!dtKnown && r.dt > 0.0)
                    {
                        cs.dt = r.dt;

                        dtKnown = true;
                    }

                    string variant = "fixed:"+solverChoiceName(candidates[k]);

                    writeRunRow(summary, mode, cs, variant, r);

                    summary.flush();

                    if(printRuns)
                    {
                        printRun(cs, variant, r);
                    }

                    fixedResults.push_back(r);
                }

                CaseResult adaptive = runCase(cs, candidates, policy);

                string adaptiveName = adaptiveLabel(candidates);

                writeRunRow(summary, mode, cs, adaptiveName, adaptive);

                summary.flush();

                if(printRuns)
                {
                    printRun(cs, adaptiveName, adaptive);
                }

                // Best fixed run: the smallest total time among the
                // runs that finished and converged
                int best = -1;

                for(size_t k = 0; k < fixedResults.size(); k++)
                {
                    const CaseResult& r = fixedResults[k];

                    if(r.failed || r.notConverged > 0)
                    {
                        continue;
                    }

                    if(best < 0 || r.totalSeconds < fixedResults[best].totalSeconds)
                    {
                        best = (int)k;
                    }
                }

                string bestName = "none";
                string verdict;

                double bestSeconds = 0.0;
                double bestIterations = 0.0;
                double secondsRatio = 0.0;
                double diffYc = 0.0;
                double diffV = 0.0;

                bool physicalOK = false;

                if(adaptive.failed || adaptive.notConverged > 0)
                {
                    verdict = "REJECTED: adaptive run failed or did not converge";
                }
                else if(best < 0)
                {
                    verdict = "NO COMPARISON: no fixed run converged";
                }
                else
                {
                    const CaseResult& ref = fixedResults[best];

                    bestName = solverChoiceName(candidates[best]);
                    bestSeconds = ref.totalSeconds;
                    bestIterations = (double)ref.iterations;

                    secondsRatio = adaptive.totalSeconds/ref.totalSeconds;

                    diffYc = fabs(adaptive.state.yc - ref.state.yc)/fabs(ref.state.yc);

                    // The rise velocity is zero at rest, so scale the
                    // difference with the larger of the two values
                    double vScale = max(fabs(ref.state.vRise), fabs(adaptive.state.vRise));

                    diffV = (vScale > 0.0) ? fabs(adaptive.state.vRise - ref.state.vRise)/vScale : 0.0;

                    physicalOK = (diffYc <= physTol && diffV <= physTol);

                    if(!physicalOK)
                    {
                        verdict = "REJECTED: physical results differ from the best fixed run";
                    }
                    else if(adaptive.totalSeconds < ref.totalSeconds)
                    {
                        verdict = "ACCEPTED: faster than the best fixed run";
                    }
                    else
                    {
                        verdict = "NO GAIN: not faster than the best fixed run";
                    }
                }

                decision << cs.N << ","
                         << cs.N << ","
                         << 2*cs.N << ","
                         << cs.ratio << ","
                         << bestName << ","
                         << bestSeconds << ","
                         << adaptive.totalSeconds << ","
                         << bestIterations << ","
                         << adaptive.iterations << ","
                         << secondsRatio << ","
                         << diffYc << ","
                         << diffV << ","
                         << (physicalOK ? 1 : 0) << ","
                         << adaptive.switches << ","
                         << adaptive.fallbacks << ","
                         << csvSafe(verdict) << "\n";

                decision.flush();

                cout << "=> N = " << cs.N << "x" << 2*cs.N
                     << "  ratio = " << cs.ratio
                     << "  best fixed = " << bestName
                     << " (" << bestSeconds << " s)"
                     << "  adaptive = " << adaptive.totalSeconds << " s"
                     << "  relDiff(yc) = " << diffYc
                     << "  relDiff(vRise) = " << diffV
                     << "  " << verdict << endl;
            }
        }
    }
    catch(const exception& e)
    {
        cerr << "SolverStudy: " << e.what() << endl;

        usage();

        return 1;
    }

    return 0;
}