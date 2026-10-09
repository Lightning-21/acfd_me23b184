//------------------------------------------------------------//
// File: KrylovComparison.cpp
//
// Description:
// Standalone example, NOT part of src/ — compares Steepest
// Descent, CG and BiCGSTAB (with and without preconditioners)
// on dense systems A x = b built from the 2D convection-
// diffusion operator on the unit square (5-point stencil,
// scaled by h^2, m x m interior points):
//
//     4 u_P - (1 - c) u_E - (1 + c) u_W - u_N - u_S,
//     c = beta*h/2
//
//     beta = 0  -> 2D Poisson, symmetric positive definite
//     beta > 0  -> non-symmetric
//
// b is built as A*xExact with xExact = 16 x(1-x) y(1-y) at the
// nodes, so the solution error ||x - xExact|| / ||xExact||
// checks the solver independently of the discretisation.
//
// (sin(pi x) sin(pi y) is deliberately NOT used: it is an exact
// eigenvector of the Poisson matrix, so b = lambda*x and every
// Krylov method converges in one iteration. x(1-x)y(1-y) mixes
// many modes and gives a meaningful iteration count.)
//
// Sections:
//   1. SPD system        : SD, CG, BiCGSTAB x preconditioners
//   2. Non-symmetric     : SD, CG (symmetric-only methods) vs
//                          BiCGSTAB x preconditioners / modes
//   3. Auto selection    : solveKrylov(Auto) on both systems
//   4. Scaling           : iterations and time against grid size
//
// Note: the Poisson matrix has a constant diagonal, so Jacobi is
// a scalar multiple of the identity and changes nothing there
// (its rows match the unpreconditioned ones).
//
// Writes krylov_history.csv (residual history, SPD case, no
// preconditioner) and krylov_scaling.csv.
//
// Compilation Instruction:
// g++ KrylovComparison.cpp src/KrylovSolvers.cpp src/MatrixAnalysis.cpp src/MatrixFunctions.cpp src/Preconditioner.cpp -Iinclude -std=c++17 -O2 -o krylovComparison && ./krylovComparison
//------------------------------------------------------------//

#include "KrylovSolvers.h"
#include "MatrixAnalysis.h"
#include "MatrixFunctions.h"
#include "Preconditioner.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <exception>
#include <string>
#include <vector>

using namespace std;

typedef vector<vector<double>> Matrix;

const double PI = 3.14159265358979323846;


//------------------------------------------------------------//
// Problem Setup
//------------------------------------------------------------//

Matrix buildMatrix(int m, double beta)
{
    int n = m*m;

    double h = 1.0/(m + 1);
    double c = 0.5*beta*h;

    Matrix A(n, vector<double>(n, 0.0));

    for(int j = 0; j < m; j++)
    {
        for(int i = 0; i < m; i++)
        {
            int p = j*m + i;

            A[p][p] = 4.0;

            if(i > 0)   A[p][p-1] = -(1.0 + c);
            if(i < m-1) A[p][p+1] = -(1.0 - c);
            if(j > 0)   A[p][p-m] = -1.0;
            if(j < m-1) A[p][p+m] = -1.0;
        }
    }

    return A;
}

vector<double> exactSolution(int m)
{
    double h = 1.0/(m + 1);

    vector<double> x(m*m);

    for(int j = 0; j < m; j++)
    {
        for(int i = 0; i < m; i++)
        {
            double xi = (i+1)*h;
            double yj = (j+1)*h;

            x[j*m+i] = 16.0*xi*(1.0-xi)*yj*(1.0-yj);
        }
    }

    return x;
}

// 2-norm condition number of the m x m Poisson matrix:
// cot^2(pi h / 2)
double poissonCondition(int m)
{
    double h = 1.0/(m + 1);
    double t = tan(0.5*PI*h);

    return 1.0/(t*t);
}

string precondName(PrecondMethod method)
{
    switch(method)
    {
        case PrecondMethod::Jacobi:               return "Jacobi";
        case PrecondMethod::GaussSeidel:          return "GS";
        case PrecondMethod::SymmetricGaussSeidel: return "SGS";
        case PrecondMethod::SOR:                  return "SOR";
        case PrecondMethod::LU:                   return "LU";
        case PrecondMethod::Cholesky:             return "Cholesky";
        case PrecondMethod::ILU0:                 return "ILU0";
        case PrecondMethod::IC0:                  return "IC0";
    }

    return "unknown";
}

string modeName(PrecondMode mode)
{
    switch(mode)
    {
        case PrecondMode::Left:  return "Left";
        case PrecondMode::Right: return "Right";
        case PrecondMode::Split: return "Split";
    }

    return "unknown";
}


//------------------------------------------------------------//
// Running and Reporting
//------------------------------------------------------------//

struct Problem
{
    Matrix A;

    vector<double> b;

    vector<double> xExact;
};

Problem makeProblem(int m, double beta)
{
    Problem p;

    p.A = buildMatrix(m, beta);
    p.xExact = exactSolution(m);
    p.b = matVec(p.A, p.xExact);

    return p;
}

double solutionError(const vector<double>& x, const vector<double>& xExact)
{
    return norm2(axpy(-1.0, xExact, x))/norm2(xExact);
}

void printHeader()
{
    cout << left
         << setw(16) << "Method"
         << setw(14) << "Precond"
         << setw(14) << "Status"
         << right
         << setw(7)  << "Iter"
         << setw(8)  << "MatVec"
         << setw(8)  << "PrecAp"
         << setw(12) << "TrueRes"
         << setw(12) << "SolErr"
         << setw(11) << "Time[ms]"
         << endl;
}

// Runs one solver and prints one table row. Exceptions (for
// example an unsuitable preconditioner) are reported in the row
// instead of stopping the program. ok is false if it threw.
KrylovResult runSolver
(
    KrylovMethod method,
    const string& label,
    const Preconditioner* P,
    const Problem& prob,
    const KrylovOptions& opt,
    bool& ok
)
{
    KrylovResult r;

    ok = true;

    try
    {
        r = solveKrylov(method, prob.A, prob.b, vector<double>(), P, opt);
    }
    catch(const exception& e)
    {
        ok = false;

        cout << left << setw(16) << krylovMethodName(method)
             << setw(14) << label
             << "not run: " << e.what() << endl;

        return r;
    }

    cout << left
         << setw(16) << krylovMethodName(r.method)
         << setw(14) << label
         << setw(14) << krylovStatusName(r.status)
         << right
         << setw(7)  << r.iterations
         << setw(8)  << r.matVecs
         << setw(8)  << r.precondApplies
         << setw(12) << scientific << setprecision(2) << r.trueRelResidual
         << setw(12) << solutionError(r.x, prob.xExact)
         << setw(11) << fixed << setprecision(2) << r.seconds*1000.0
         << endl;

    return r;
}

// Every method with no preconditioner, then with each
// preconditioner in Left mode
void compare
(
    const vector<KrylovMethod>& methods,
    const vector<PrecondMethod>& precs,
    const Problem& prob,
    const KrylovOptions& opt
)
{
    printHeader();

    bool ok;

    for(KrylovMethod method : methods)
    {
        runSolver(method, "none", nullptr, prob, opt, ok);

        for(PrecondMethod pm : precs)
        {
            try
            {
                Preconditioner P = buildPreconditioner(pm, PrecondMode::Left, prob.A);

                runSolver(method, precondName(pm), &P, prob, opt, ok);
            }
            catch(const exception& e)
            {
                cout << left << setw(16) << krylovMethodName(method)
                     << setw(14) << precondName(pm)
                     << "setup failed: " << e.what() << endl;
            }
        }

        cout << endl;
    }
}

void printAnalysis(const char* name, const Matrix& A)
{
    MatrixInfo info = analyzeMatrix(A, true);

    cout << name << ": n = " << info.n
         << ", symmetric = " << (info.symmetric ? "yes" : "no")
         << ", SPD = " << (info.spd ? "yes" : "no")
         << ", condition estimate = " << scientific << setprecision(3)
         << info.conditionEstimate << fixed
         << ", Auto would pick " << krylovMethodName(chooseMethod(info))
         << endl << endl;
}


//------------------------------------------------------------//
// Main
//------------------------------------------------------------//

int main()
{
    int m = 16;
    double beta = 20.0;

    KrylovOptions opt;

    opt.tol = 1.0e-10;
    opt.maxIter = 20000;

    bool ok;

    //--------------------------------------------------------//
    // 1. SPD system (Poisson)
    //--------------------------------------------------------//

    cout << "==========================================" << endl;
    cout << " 1. SPD system: 2D Poisson, " << m << " x " << m << " points" << endl;
    cout << "==========================================" << endl << endl;

    Problem spd = makeProblem(m, 0.0);

    printAnalysis("Poisson", spd.A);

    compare
    (
        { KrylovMethod::SteepestDescent, KrylovMethod::CG, KrylovMethod::BiCGSTAB },
        { PrecondMethod::Jacobi, PrecondMethod::SymmetricGaussSeidel, PrecondMethod::IC0 },
        spd, opt
    );

    // Residual history of the three unpreconditioned solvers
    {
        KrylovResult sd = runSolver(KrylovMethod::SteepestDescent, "", nullptr, spd, opt, ok);
        KrylovResult cg = runSolver(KrylovMethod::CG, "", nullptr, spd, opt, ok);
        KrylovResult bi = runSolver(KrylovMethod::BiCGSTAB, "", nullptr, spd, opt, ok);

        ofstream hist("krylov_history.csv");

        hist << "iteration,SteepestDescent,CG,BiCGSTAB\n";

        size_t rows = max(sd.history.size(), max(cg.history.size(), bi.history.size()));

        hist << scientific << setprecision(10);

        for(size_t k = 0; k < rows; k++)
        {
            hist << k;

            const vector<double>* h[3] = { &sd.history, &cg.history, &bi.history };

            for(int s = 0; s < 3; s++)
            {
                hist << ",";

                if(k < h[s]->size())
                {
                    hist << (*h[s])[k];
                }
            }

            hist << "\n";
        }

        cout << "\nResidual history -> krylov_history.csv (" << rows << " rows)" << endl << endl;
    }

    //--------------------------------------------------------//
    // 2. Non-symmetric system (convection-diffusion)
    //--------------------------------------------------------//

    cout << "==========================================" << endl;
    cout << " 2. Non-symmetric: convection-diffusion, beta = " << beta << endl;
    cout << "==========================================" << endl << endl;

    Problem nonsym = makeProblem(m, beta);

    printAnalysis("ConvDiff", nonsym.A);

    // CG relies on symmetry and does not converge here. SD can
    // still converge for mild non-symmetry like this, but it is
    // not guaranteed. (Fewer iterations allowed so a failure is
    // quick.)
    KrylovOptions optShort = opt;

    optShort.maxIter = 2000;

    compare({ KrylovMethod::SteepestDescent, KrylovMethod::CG }, {}, nonsym, optShort);

    compare
    (
        { KrylovMethod::BiCGSTAB },
        { PrecondMethod::Jacobi, PrecondMethod::SymmetricGaussSeidel, PrecondMethod::ILU0 },
        nonsym, opt
    );

    // BiCGSTAB honours the preconditioning mode
    cout << "BiCGSTAB with ILU0, by mode:" << endl;

    printHeader();

    for(PrecondMode mode : { PrecondMode::Left, PrecondMode::Right, PrecondMode::Split })
    {
        Preconditioner P = buildPreconditioner(PrecondMethod::ILU0, mode, nonsym.A);

        runSolver(KrylovMethod::BiCGSTAB, "ILU0-" + modeName(mode), &P, nonsym, opt, ok);
    }

    cout << endl;

    //--------------------------------------------------------//
    // 3. Auto selection
    //--------------------------------------------------------//

    cout << "==========================================" << endl;
    cout << " 3. Auto selection" << endl;
    cout << "==========================================" << endl << endl;

    printHeader();

    runSolver(KrylovMethod::Auto, "Poisson", nullptr, spd, opt, ok);
    runSolver(KrylovMethod::Auto, "ConvDiff", nullptr, nonsym, opt, ok);

    cout << endl;

    //--------------------------------------------------------//
    // 4. Scaling with grid size (Poisson)
    //--------------------------------------------------------//

    cout << "==========================================" << endl;
    cout << " 4. Scaling: iterations and time vs grid size" << endl;
    cout << "==========================================" << endl << endl;

    vector<int> sizes = { 8, 16, 24, 32 };

    ofstream scal("krylov_scaling.csv");

    scal << "m,n,kappa,SD_iter,CG_iter,PCG_IC0_iter,BiCGSTAB_iter,"
            "SD_ms,CG_ms,PCG_IC0_ms,BiCGSTAB_ms\n";

    cout << right
         << setw(5)  << "m"
         << setw(7)  << "n"
         << setw(10) << "kappa"
         << setw(9)  << "SD"
         << setw(9)  << "CG"
         << setw(10) << "CG+IC0"
         << setw(10) << "BiCGSTAB"
         << "     (iterations)"
         << endl;

    for(int ms : sizes)
    {
        Problem p = makeProblem(ms, 0.0);

        Preconditioner ic = buildPreconditioner(PrecondMethod::IC0, PrecondMode::Left, p.A);

        KrylovResult sd = solveKrylov(KrylovMethod::SteepestDescent, p.A, p.b, vector<double>(), nullptr, opt);
        KrylovResult cg = solveKrylov(KrylovMethod::CG, p.A, p.b, vector<double>(), nullptr, opt);
        KrylovResult pc = solveKrylov(KrylovMethod::CG, p.A, p.b, vector<double>(), &ic, opt);
        KrylovResult bi = solveKrylov(KrylovMethod::BiCGSTAB, p.A, p.b, vector<double>(), nullptr, opt);

        cout << setw(5)  << ms
             << setw(7)  << ms*ms
             << setw(10) << fixed << setprecision(1) << poissonCondition(ms)
             << setw(9)  << sd.iterations
             << setw(9)  << cg.iterations
             << setw(10) << pc.iterations
             << setw(10) << bi.iterations
             << endl;

        scal << ms << "," << ms*ms << "," << poissonCondition(ms) << ","
             << sd.iterations << "," << cg.iterations << ","
             << pc.iterations << "," << bi.iterations << ","
             << sd.seconds*1000.0 << "," << cg.seconds*1000.0 << ","
             << pc.seconds*1000.0 << "," << bi.seconds*1000.0 << "\n";
    }

    cout << endl << "Scaling table -> krylov_scaling.csv" << endl;
    cout << "(SD grows ~ kappa, CG ~ sqrt(kappa); kappa ~ 1/h^2)" << endl;

    return 0;
}