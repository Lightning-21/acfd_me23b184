//------------------------------------------------------------//
// File: PVCouplingCavity.cpp
//
// Description:
// Standalone driver, NOT part of src/ — steady 2D laminar
// lid-driven square cavity (unit square, lid u = 1 on top,
// rho = 1, nu = 1/Re) solved on the staggered grid with one of
// the pressure-velocity coupling algorithms:
//
//     simple : SIMPLE, steady iterations
//     piso   : PISO, marched in pseudo-time to the steady state
//     pimple : PIMPLE, marched in pseudo-time to the steady state
//
// Usage:
//     PVCouplingCavity <algorithm> <Re> <N> [key=value ...]
//
// N is the number of cells per side (use an even N: the vertical
// and horizontal centrelines then pass exactly through velocity
// nodes).
//
// Options (defaults depend on the algorithm):
//     scheme=hybrid|upwind     convection scheme          (hybrid)
//     tol=                     scaled-residual tolerance  (1e-6)
//     maxIter=                 iteration / time-step cap  (simple
//                              40000, piso and pimple 60000)
//     ptol=                    p' CG relative tolerance   (1e-2)
//     pmax=                    p' CG iteration cap        (1000)
//     precond=1|0              SGS preconditioner on / off (1)
//     sweeps=                  momentum GS sweeps         (3)
//     alphaU= alphaP=          relaxation  (simple 0.7 / 0.3,
//                              pimple 0.8 / 0.5, unused by piso)
//     alphaUFinal=             last outer iteration, pimple (1)
//     cfl= dt=                 piso / pimple time step: dt if > 0,
//                              else cfl min(dx,dy)/1       (cfl 16)
//     correctors=              pressure correctors per step (2)
//     outer=                   pimple outer iterations      (2)
//     out=                     output file prefix (default
//                              cavity_<algorithm>_Re<Re>_N<N>)
//     summary=                 summary file               (summary.csv)
//     print=                   print every n iterations   (0)
//
// Writes, in the current directory:
//     <out>_history.csv   iteration, resU, resV, resMass
//     <out>_profiles.csv  s, u(0.5, s), v(s, 0.5), 257 points on [0,1]
//     summary.csv         one row appended per run (header written
//                         when the file does not exist)
//
// Residuals are the scaled residuals defined in SIMPLE.h / PISO.h,
// so the three algorithms are comparable. A run that does not
// converge is still written out and flagged in the status column.
//
// Compilation Instruction:
// g++ PVCouplingCavity.cpp src/NavierStokes2D.cpp src/SIMPLE.cpp src/PISO.cpp src/PIMPLE.cpp -Iinclude -std=c++17 -O2 -o pvCouplingCavity && ./pvCouplingCavity simple 100 32
//------------------------------------------------------------//

#include "NavierStokes2D.h"
#include "SIMPLE.h"
#include "PISO.h"
#include "PIMPLE.h"

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
    cerr << "Usage: PVCouplingCavity <simple|piso|pimple> <Re> <N> [key=value ...]" << endl
         << "Keys: scheme tol maxIter ptol pmax precond sweeps alphaU alphaP alphaUFinal"
         << " cfl dt correctors outer out summary print" << endl
         << "See the header of PVCouplingCavity.cpp for the meaning and defaults." << endl;
}

static bool parseOptions
(
    int argc,
    char** argv,
    int first,
    map<string, string>& opts
)
{
    static const char* keys[] =
    {
        "scheme", "tol", "maxIter", "ptol", "pmax", "precond", "sweeps",
        "alphaU", "alphaP", "alphaUFinal", "cfl", "dt", "correctors",
        "outer", "out", "summary", "print"
    };

    for(int a = first; a < argc; a++)
    {
        string arg = argv[a];

        size_t eq = arg.find('=');

        if(eq == string::npos || eq == 0)
        {
            cerr << "Bad option '" << arg << "' (expected key=value)" << endl;

            return false;
        }

        string key = arg.substr(0, eq);

        bool known = false;

        for(size_t k = 0; k < sizeof(keys)/sizeof(keys[0]); k++)
        {
            if(key == keys[k])
            {
                known = true;
            }
        }

        if(!known)
        {
            cerr << "Unknown option '" << key << "'" << endl;

            return false;
        }

        opts[key] = arg.substr(eq + 1);
    }

    return true;
}

static double getDouble
(
    const map<string, string>& o,
    const string& key,
    double def
)
{
    map<string, string>::const_iterator it = o.find(key);

    if(it == o.end())
    {
        return def;
    }

    char* end = 0;

    double v = strtod(it->second.c_str(), &end);

    if(end == it->second.c_str() || *end != '\0')
    {
        throw runtime_error("option " + key + ": '" + it->second + "' is not a number");
    }

    return v;
}

static int getInt
(
    const map<string, string>& o,
    const string& key,
    int def
)
{
    double v = getDouble(o, key, def);

    if(v != floor(v) || fabs(v) > 2e9)
    {
        throw runtime_error("option " + key + " must be an integer");
    }

    return (int)v;
}

static string getString
(
    const map<string, string>& o,
    const string& key,
    const string& def
)
{
    map<string, string>::const_iterator it = o.find(key);

    if(it == o.end())
    {
        return def;
    }

    return it->second;
}

static string fmtNumber(double x)
{
    ostringstream s;

    s << x;

    return s.str();
}

//------------------------------------------------------------//
// Run Record: what every algorithm reports in common
//------------------------------------------------------------//

struct RunInfo
{
    string status;

    bool converged = false;

    int count = 0;

    int pressureIterations = 0;

    double seconds = 0.0;

    // Empty entries (NaN) are written as blank CSV fields
    double dt = NAN;

    double outer = NAN;

    double correctors = NAN;

    double alphaU = NAN;

    double alphaP = NAN;

    double cfl = NAN;

    double resU = 0.0;

    double resV = 0.0;

    double resMass = 0.0;

    vector<double> historyU;

    vector<double> historyV;

    vector<double> historyMass;
};

static void writeField
(
    ostream& os,
    double x
)
{
    if(!std::isnan(x))
    {
        os << x;
    }
}

//------------------------------------------------------------//
// Main
//------------------------------------------------------------//

int main(int argc, char** argv)
{
    if(argc < 4)
    {
        usage();

        return 1;
    }

    string alg = argv[1];

    if(alg != "simple" && alg != "piso" && alg != "pimple")
    {
        cerr << "Unknown algorithm '" << alg << "'" << endl;

        usage();

        return 1;
    }

    map<string, string> opts;

    if(!parseOptions(argc, argv, 4, opts))
    {
        usage();

        return 1;
    }

    try
    {
        char* end = 0;

        double Re = strtod(argv[2], &end);

        if(end == argv[2] || *end != '\0' || !(Re > 0.0))
        {
            throw runtime_error("Re must be a positive number");
        }

        double Nd = strtod(argv[3], &end);

        if(end == argv[3] || *end != '\0' || Nd != floor(Nd) || Nd < 2.0 || Nd > 4096.0)
        {
            throw runtime_error("N must be an integer between 2 and 4096");
        }

        int N = (int)Nd;

        if(N%2 != 0)
        {
            cerr << "Warning: N is odd, so the centrelines are interpolated" << endl;
        }

        string schemeName = getString(opts, "scheme", "hybrid");

        ConvectionScheme scheme = ConvectionScheme::Hybrid;

        if(schemeName == "upwind")
        {
            scheme = ConvectionScheme::Upwind;
        }
        else if(schemeName != "hybrid")
        {
            throw runtime_error("scheme must be hybrid or upwind");
        }

        double tol = getDouble(opts, "tol", 1e-6);
        double ptol = getDouble(opts, "ptol", 1e-2);
        int pmax = getInt(opts, "pmax", 1000);
        int precond = getInt(opts, "precond", 1);
        int sweeps = getInt(opts, "sweeps", 3);
        int printEvery = getInt(opts, "print", 0);

        WallVelocities walls;

        walls.uTop = 1.0;

        NavierStokes2D ns(N, N, 1.0, 1.0, 1.0, 1.0/Re, walls);

        RunInfo run;

        //--------------------------------------------------
        // Solve
        //--------------------------------------------------

        if(alg == "simple")
        {
            SIMPLEOptions o;

            o.scheme = scheme;
            o.alphaU = getDouble(opts, "alphaU", 0.7);
            o.alphaP = getDouble(opts, "alphaP", 0.3);
            o.momentumSweeps = sweeps;
            o.pressureTol = ptol;
            o.pressureMaxIter = pmax;
            o.pressurePreconditioner = (precond != 0);
            o.tol = tol;
            o.maxIter = getInt(opts, "maxIter", 40000);
            o.printEvery = printEvery;

            SIMPLEResult r = solveSIMPLE(ns, o);

            run.status = simpleStatusName(r.status);
            run.converged = (r.status == SIMPLEStatus::Converged);
            run.count = r.iterations;
            run.pressureIterations = r.pressureIterations;
            run.seconds = r.seconds;
            run.alphaU = o.alphaU;
            run.alphaP = o.alphaP;
            run.resU = r.resU;
            run.resV = r.resV;
            run.resMass = r.resMass;
            run.historyU = r.historyU;
            run.historyV = r.historyV;
            run.historyMass = r.historyMass;
        }
        else if(alg == "piso")
        {
            PISOOptions o;

            o.scheme = scheme;
            o.dt = getDouble(opts, "dt", 0.0);
            o.cfl = getDouble(opts, "cfl", 16.0);
            o.correctors = getInt(opts, "correctors", 2);
            o.momentumSweeps = sweeps;
            o.pressureTol = ptol;
            o.pressureMaxIter = pmax;
            o.pressurePreconditioner = (precond != 0);
            o.tol = tol;
            o.maxSteps = getInt(opts, "maxIter", 60000);
            o.printEvery = printEvery;

            PISOResult r = solvePISO(ns, o);

            run.status = pisoStatusName(r.status);
            run.converged = (r.status == PISOStatus::Converged);
            run.count = r.steps;
            run.pressureIterations = r.pressureIterations;
            run.seconds = r.seconds;
            run.dt = r.dt;
            run.correctors = o.correctors;
            run.cfl = r.dt/min(ns.dx, ns.dy);
            run.resU = r.resU;
            run.resV = r.resV;
            run.resMass = r.resMass;
            run.historyU = r.historyU;
            run.historyV = r.historyV;
            run.historyMass = r.historyMass;
        }
        else
        {
            PIMPLEOptions o;

            o.scheme = scheme;
            o.dt = getDouble(opts, "dt", 0.0);
            o.cfl = getDouble(opts, "cfl", 16.0);
            o.outerIterations = getInt(opts, "outer", 2);
            o.correctors = getInt(opts, "correctors", 2);
            o.alphaU = getDouble(opts, "alphaU", 0.8);
            o.alphaP = getDouble(opts, "alphaP", 0.5);
            o.alphaUFinal = getDouble(opts, "alphaUFinal", 1.0);
            o.momentumSweeps = sweeps;
            o.pressureTol = ptol;
            o.pressureMaxIter = pmax;
            o.pressurePreconditioner = (precond != 0);
            o.tol = tol;
            o.maxSteps = getInt(opts, "maxIter", 60000);
            o.printEvery = printEvery;

            PIMPLEResult r = solvePIMPLE(ns, o);

            run.status = pimpleStatusName(r.status);
            run.converged = (r.status == PIMPLEStatus::Converged);
            run.count = r.steps;
            run.pressureIterations = r.pressureIterations;
            run.seconds = r.seconds;
            run.dt = r.dt;
            run.outer = o.outerIterations;
            run.correctors = o.correctors;
            run.alphaU = o.alphaU;
            run.alphaP = o.alphaP;
            run.cfl = r.dt/min(ns.dx, ns.dy);
            run.resU = r.resU;
            run.resV = r.resV;
            run.resMass = r.resMass;
            run.historyU = r.historyU;
            run.historyV = r.historyV;
            run.historyMass = r.historyMass;
        }

        //--------------------------------------------------
        // Final-field quantities
        //--------------------------------------------------

        double finalMass = ns.massResidual();

        double uMid = ns.sampleU(0.5, 0.5);

        string prefix = getString(opts, "out", "cavity_" + alg + "_Re" + fmtNumber(Re) + "_N" + fmtNumber(N));

        string summaryName = getString(opts, "summary", "summary.csv");

        //--------------------------------------------------
        // History
        //--------------------------------------------------

        {
            ofstream hist((prefix + "_history.csv").c_str());

            if(!hist)
            {
                throw runtime_error("cannot write " + prefix + "_history.csv");
            }

            hist << setprecision(10);

            hist << "iteration,resU,resV,resMass" << endl;

            for(size_t k = 0; k < run.historyU.size(); k++)
            {
                hist << k + 1 << "," << run.historyU[k] << "," << run.historyV[k] << "," << run.historyMass[k] << endl;
            }
        }

        //--------------------------------------------------
        // Centreline profiles
        //--------------------------------------------------

        {
            ofstream prof((prefix + "_profiles.csv").c_str());

            if(!prof)
            {
                throw runtime_error("cannot write " + prefix + "_profiles.csv");
            }

            prof << setprecision(10);

            prof << "s,u_x0.5,v_y0.5" << endl;

            int M = 256;

            for(int m = 0; m <= M; m++)
            {
                double s = (double)m/M;

                prof << s << "," << ns.sampleU(0.5, s) << "," << ns.sampleV(s, 0.5) << endl;
            }
        }

        //--------------------------------------------------
        // Summary row
        //--------------------------------------------------

        bool needHeader = true;

        {
            ifstream probe(summaryName.c_str());

            if(probe.good() && probe.peek() != ifstream::traits_type::eof())
            {
                needHeader = false;
            }
        }

        {
            ofstream sum(summaryName.c_str(), ios::app);

            if(!sum)
            {
                throw runtime_error("cannot write " + summaryName);
            }

            sum << setprecision(10);

            if(needHeader)
            {
                sum << "algorithm,Re,N,status,iterations_or_steps,pressureIterations,seconds,"
                    << "dt,cfl,outer,correctors,alphaU,alphaP,pressureTol,tol,"
                    << "resU,resV,resMass,finalContinuity,uMid" << endl;
            }

            sum << alg << "," << Re << "," << N << "," << run.status << ","
                << run.count << "," << run.pressureIterations << "," << run.seconds << ",";

            writeField(sum, run.dt);
            sum << ",";
            writeField(sum, run.cfl);
            sum << ",";
            writeField(sum, run.outer);
            sum << ",";
            writeField(sum, run.correctors);
            sum << ",";
            writeField(sum, run.alphaU);
            sum << ",";
            writeField(sum, run.alphaP);
            sum << "," << ptol << "," << tol << ","
                << run.resU << "," << run.resV << "," << run.resMass << ","
                << finalMass << "," << uMid << endl;
        }

        //--------------------------------------------------
        // Console report
        //--------------------------------------------------

        cout << setprecision(6);

        cout << alg << "  Re=" << Re << "  N=" << N << "  " << run.status
             << "  " << (alg == "simple" ? "iterations=" : "steps=") << run.count
             << "  CG=" << run.pressureIterations
             << "  time=" << run.seconds << "s"
             << "  u(0.5,0.5)=" << uMid
             << "  continuity=" << finalMass << endl;

        if(!run.converged)
        {
            cerr << "WARNING: " << alg << " Re=" << Re << " N=" << N << " ended with status " << run.status
                 << " (final scaled residuals u " << run.resU << ", v " << run.resV << ", mass " << run.resMass << ")" << endl;
        }

        cout << "History  -> " << prefix << "_history.csv (" << run.historyU.size() << " rows)" << endl
             << "Profiles -> " << prefix << "_profiles.csv" << endl
             << "Summary  -> " << summaryName << endl;
    }
    catch(const exception& e)
    {
        cerr << "Error: " << e.what() << endl;

        return 1;
    }

    return 0;
}