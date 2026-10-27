//------------------------------------------------------------//
// File: SolverSelector.cpp
//
// Description:
//
// Run-time choice of Krylov method and preconditioner for the
// pressure Poisson solve of the two-phase flow code, built on
// solveKrylov (KrylovSolvers) and buildPreconditioner
// (Preconditioner), both dense.
//
//   Names     : "CG+IC0", "BiCGSTAB+ILU0", "CG", ... parsed
//               into a SolverChoice
//   Selection : SolverSelector holds an ordered list of
//               choices, weakest first, and solves one system
//               per call
//   Switching : an optional policy moves up or down the list
//               between solves from the observed iteration
//               count
//
// The preconditioner is rebuilt in every solve because the
// matrix changes with the interface, and its setup time is
// recorded separately from the iteration time so the cost of
// every candidate can be compared fairly.
//
//------------------------------------------------------------//

#include "SolverSelector.h"

#include <cctype>
#include <chrono>
#include <stdexcept>

using namespace std;

//------------------------------------------------------------//
// Name Helpers
//------------------------------------------------------------//

namespace
{
    // Lower case, with spaces, '-' and '_' removed
    string normalise(const string& s)
    {
        string out;

        for(size_t k = 0; k < s.size(); k++)
        {
            char c = s[k];

            if(c == ' ' || c == '-' || c == '_' || c == '\t')
            {
                continue;
            }

            out += (char)tolower((unsigned char)c);
        }

        return out;
    }

    const char* precondName(PrecondMethod m)
    {
        switch(m)
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

        return "?";
    }

    KrylovMethod parseMethod(const string& original)
    {
        string n = normalise(original);

        if(n == "cg")
        {
            return KrylovMethod::CG;
        }

        if(n == "bicgstab")
        {
            return KrylovMethod::BiCGSTAB;
        }

        if(n == "steepestdescent" || n == "sd")
        {
            return KrylovMethod::SteepestDescent;
        }

        if(n == "auto")
        {
            return KrylovMethod::Auto;
        }

        throw invalid_argument("unknown Krylov method '"+original+"'");
    }

    // Sets c.usePreconditioner and c.precond from a name
    void parsePrecond
    (
        const string& original,
        SolverChoice& c
    )
    {
        string n = normalise(original);

        c.usePreconditioner = true;

        if(n.empty() || n == "none")
        {
            c.usePreconditioner = false;
        }
        else if(n == "jacobi")
        {
            c.precond = PrecondMethod::Jacobi;
        }
        else if(n == "gs" || n == "gaussseidel")
        {
            c.precond = PrecondMethod::GaussSeidel;
        }
        else if(n == "sgs" || n == "symmetricgaussseidel")
        {
            c.precond = PrecondMethod::SymmetricGaussSeidel;
        }
        else if(n == "sor")
        {
            c.precond = PrecondMethod::SOR;
        }
        else if(n == "lu")
        {
            c.precond = PrecondMethod::LU;
        }
        else if(n == "cholesky")
        {
            c.precond = PrecondMethod::Cholesky;
        }
        else if(n == "ilu0" || n == "ilu")
        {
            c.precond = PrecondMethod::ILU0;
        }
        else if(n == "ic0" || n == "ic")
        {
            c.precond = PrecondMethod::IC0;
        }
        else
        {
            throw invalid_argument("unknown preconditioner '"+original+"'");
        }
    }
}

//------------------------------------------------------------//
// Names
//------------------------------------------------------------//

SolverChoice parseSolverChoice
(
    const string& name,
    PrecondMode mode,
    double omega
)
{
    SolverChoice c;

    c.mode = mode;
    c.omega = omega;

    size_t plus = name.find('+');

    if(plus == string::npos)
    {
        c.method = parseMethod(name);
        c.usePreconditioner = false;
    }
    else
    {
        c.method = parseMethod(name.substr(0, plus));
        parsePrecond(name.substr(plus+1), c);
    }

    return c;
}

vector<SolverChoice> parseSolverList
(
    const string& names,
    PrecondMode mode,
    double omega
)
{
    vector<SolverChoice> list;

    size_t start = 0;

    while(start <= names.size())
    {
        size_t comma = names.find(',', start);

        if(comma == string::npos)
        {
            comma = names.size();
        }

        string item = names.substr(start, comma-start);

        if(!normalise(item).empty())
        {
            list.push_back(parseSolverChoice(item, mode, omega));
        }

        start = comma+1;
    }

    if(list.empty())
    {
        throw invalid_argument("parseSolverList: no solver names given");
    }

    return list;
}

string solverChoiceName(const SolverChoice& c)
{
    string name = krylovMethodName(c.method);

    if(c.usePreconditioner)
    {
        name += string("+")+precondName(c.precond);
    }

    return name;
}

//------------------------------------------------------------//
// Constructor
//------------------------------------------------------------//

SolverSelector::SolverSelector
(
    const vector<SolverChoice>& candidates_,
    const KrylovOptions& opt_,
    const SwitchPolicy& policy_
)
:
    candidates(candidates_),
    opt(opt_),
    policy(policy_),
    index(0),
    step(0),
    sinceSwitch(0),
    easyCount(0)
{
    if(candidates.empty())
    {
        throw invalid_argument("SolverSelector: no candidate solvers given.");
    }

    if
    (
        policy.adaptive
     && (policy.upIters < 1 || policy.downIters < 1
      || policy.patience < 1 || policy.holdSteps < 1)
    )
    {
        throw invalid_argument
        (
            "SolverSelector: upIters, downIters, patience and holdSteps "
            "must all be at least 1 for an adaptive policy."
        );
    }
}

//------------------------------------------------------------//
// Running One Choice
//------------------------------------------------------------//

KrylovResult SolverSelector::runChoice
(
    const SolverChoice& c,
    const vector<vector<double>>& A,
    const vector<double>& b,
    const vector<double>& x0,
    double& setupSeconds
) const
{
    if(!c.usePreconditioner)
    {
        setupSeconds = 0.0;

        return solveKrylov(c.method, A, b, x0, nullptr, opt);
    }

    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();

    Preconditioner P = buildPreconditioner(c.precond, c.mode, A, c.omega);

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    setupSeconds = chrono::duration<double>(t1-t0).count();

    return solveKrylov(c.method, A, b, x0, &P, opt);
}

//------------------------------------------------------------//
// Switching Policy
//------------------------------------------------------------//

bool SolverSelector::updatePolicy(const KrylovResult& r)
{
    if(!policy.adaptive || candidates.size() < 2)
    {
        return false;
    }

    sinceSwitch++;

    bool failed = (r.status != KrylovStatus::Converged);

    bool last = (index == (int)candidates.size()-1);

    // Too slow or not converged: one candidate up. A failure
    // does not wait for holdSteps.
    if
    (
        !last
     && (failed || (r.iterations > policy.upIters && sinceSwitch >= policy.holdSteps))
    )
    {
        index++;

        sinceSwitch = 0;
        easyCount = 0;

        return true;
    }

    if(r.iterations < policy.downIters && !failed)
    {
        easyCount++;
    }
    else
    {
        easyCount = 0;
    }

    // Easy for long enough: one candidate down
    if(index > 0 && easyCount >= policy.patience && sinceSwitch >= policy.holdSteps)
    {
        index--;

        sinceSwitch = 0;
        easyCount = 0;

        return true;
    }

    return false;
}

//------------------------------------------------------------//
// Solve
//------------------------------------------------------------//

KrylovResult SolverSelector::solve
(
    const vector<vector<double>>& A,
    const vector<double>& b,
    const vector<double>& x0
)
{
    bool fellBack = false;

    int attempts = 0;

    while(true)
    {
        const SolverChoice& c = candidates[index];

        double setupSeconds = 0.0;

        KrylovResult r;

        try
        {
            r = runChoice(c, A, b, x0, setupSeconds);
        }
        catch(const runtime_error&)
        {
            // Preconditioner setup failed (not SPD, zero pivot).
            // A fixed run reports it; an adaptive run tries the
            // next candidate in the same step.
            attempts++;

            if(!policy.adaptive || attempts >= (int)candidates.size())
            {
                throw;
            }

            index = (index+1)%(int)candidates.size();

            sinceSwitch = 0;
            easyCount = 0;

            fellBack = true;

            continue;
        }

        SolveRecord rec;

        rec.step = step;
        rec.candidate = index;
        rec.name = solverChoiceName(c);
        rec.status = r.status;
        rec.iterations = r.iterations;
        rec.matVecs = r.matVecs;
        rec.precondApplies = r.precondApplies;
        rec.setupSeconds = setupSeconds;
        rec.solveSeconds = r.seconds;
        rec.trueRelResidual = r.trueRelResidual;
        rec.fellBack = fellBack;

        rec.switched = updatePolicy(r);

        records.push_back(rec);

        step++;

        return r;
    }
}

//------------------------------------------------------------//
// State
//------------------------------------------------------------//

const SolverChoice& SolverSelector::current() const
{
    return candidates[index];
}

int SolverSelector::currentIndex() const
{
    return index;
}

const vector<SolveRecord>& SolverSelector::history() const
{
    return records;
}

void SolverSelector::reset()
{
    index = 0;
    step = 0;
    sinceSwitch = 0;
    easyCount = 0;

    records.clear();
}

//------------------------------------------------------------//
// Totals
//------------------------------------------------------------//

double SolverSelector::totalSeconds() const
{
    double sum = 0.0;

    for(size_t k = 0; k < records.size(); k++)
    {
        sum += records[k].setupSeconds + records[k].solveSeconds;
    }

    return sum;
}

double SolverSelector::totalSetupSeconds() const
{
    double sum = 0.0;

    for(size_t k = 0; k < records.size(); k++)
    {
        sum += records[k].setupSeconds;
    }

    return sum;
}

long SolverSelector::totalIterations() const
{
    long sum = 0;

    for(size_t k = 0; k < records.size(); k++)
    {
        sum += records[k].iterations;
    }

    return sum;
}

long SolverSelector::totalMatVecs() const
{
    long sum = 0;

    for(size_t k = 0; k < records.size(); k++)
    {
        sum += records[k].matVecs;
    }

    return sum;
}

long SolverSelector::totalPrecondApplies() const
{
    long sum = 0;

    for(size_t k = 0; k < records.size(); k++)
    {
        sum += records[k].precondApplies;
    }

    return sum;
}

int SolverSelector::numSwitches() const
{
    int count = 0;

    for(size_t k = 0; k < records.size(); k++)
    {
        if(records[k].switched)
        {
            count++;
        }
    }

    return count;
}

int SolverSelector::numFallbacks() const
{
    int count = 0;

    for(size_t k = 0; k < records.size(); k++)
    {
        if(records[k].fellBack)
        {
            count++;
        }
    }

    return count;
}