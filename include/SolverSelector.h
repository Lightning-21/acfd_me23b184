#ifndef SOLVERSELECTOR_H
#define SOLVERSELECTOR_H

#include <string>
#include <vector>

#include "KrylovSolvers.h"
#include "Preconditioner.h"

//--------------------------------------------------
// Solver Choice
//
// One Krylov method together with one preconditioner (or
// none). This is the unit the selector picks, names and
// switches between.
//
// method            : CG, BiCGSTAB, SteepestDescent or Auto
// usePreconditioner : false = no preconditioning
// precond, mode     : preconditioner and its Left / Right /
//                     Split mode (ignored when
//                     usePreconditioner = false)
// omega             : SOR relaxation parameter (SOR only)
//--------------------------------------------------

struct SolverChoice
{
    KrylovMethod method = KrylovMethod::CG;

    bool usePreconditioner = true;

    PrecondMethod precond = PrecondMethod::Jacobi;

    PrecondMode mode = PrecondMode::Left;

    double omega = 1.0;
};

//--------------------------------------------------
// Names
//
// A choice is written "<method>+<preconditioner>", for
// example "CG+IC0" or "BiCGSTAB+ILU0". "CG" alone, or
// "CG+none", means no preconditioner. Matching is not case
// sensitive and ignores spaces, '-' and '_'.
//
//   method         : CG, SteepestDescent (or SD), BiCGSTAB, Auto
//   preconditioner : none, Jacobi, GS, SGS, SOR, LU,
//                    Cholesky, ILU0 (or ILU), IC0 (or IC)
//
// parseSolverChoice throws std::invalid_argument for an
// unknown name. parseSolverList reads a comma-separated list
// such as "CG+Jacobi,CG+SGS,CG+IC0" and throws if it is empty.
// mode and omega are applied to every choice parsed.
//
// solverChoiceName returns the same text a choice is parsed
// from ("CG+IC0", "CG").
//--------------------------------------------------

SolverChoice parseSolverChoice
(
    const std::string& name,
    PrecondMode mode = PrecondMode::Left,
    double omega = 1.0
);

std::vector<SolverChoice> parseSolverList
(
    const std::string& names,
    PrecondMode mode = PrecondMode::Left,
    double omega = 1.0
);

std::string solverChoiceName(const SolverChoice& c);

//--------------------------------------------------
// Switching Policy
//
// Decides, after every solve and from the iteration count
// alone, which candidate to use for the next one. The
// candidates are an ordered list, weakest / cheapest first
// and strongest last, so "up" is a stronger preconditioner
// and "down" a cheaper one.
//
// adaptive   : false = stay on the first candidate (fixed run)
// upIters    : a solve that needs more than this many
//              iterations moves one candidate up
// downIters  : a solve that needs fewer than this many
//              iterations counts as "easy"
// patience   : this many easy solves in a row move one
//              candidate down
// holdSteps  : minimum number of solves between two switches
//              (a solve that does not converge moves up at
//              once and ignores this)
//
// downIters should be well below the iteration count the next
// weaker candidate would need, otherwise the policy flips
// back and forth between two candidates.
//--------------------------------------------------

struct SwitchPolicy
{
    bool adaptive = false;

    int upIters = 60;

    int downIters = 15;

    int patience = 3;

    int holdSteps = 5;
};

//--------------------------------------------------
// Solve Record
//
// One entry per solve, kept in order in the selector's
// history.
//
// candidate    : index of the choice that produced the result
// name         : solverChoiceName of that choice
// setupSeconds : time to build the preconditioner (the
//                matrix changes every step, so it is rebuilt
//                every solve); solveSeconds is the Krylov
//                iteration time (KrylovResult::seconds)
// switched     : the policy moved to another candidate after
//                this solve
// fellBack     : the preconditioner setup of an earlier
//                candidate failed in this step, so this
//                result comes from the next one
//--------------------------------------------------

struct SolveRecord
{
    int step = 0;

    int candidate = 0;

    std::string name;

    KrylovStatus status = KrylovStatus::MaxIterations;

    int iterations = 0;

    int matVecs = 0;

    int precondApplies = 0;

    double setupSeconds = 0.0;

    double solveSeconds = 0.0;

    double trueRelResidual = 0.0;

    bool switched = false;

    bool fellBack = false;
};

//--------------------------------------------------
// SolverSelector
//
// Solves a sequence of dense linear systems A x = b (the
// pressure Poisson equation, once per time step) with a
// Krylov method and preconditioner chosen by name at run
// time and, optionally, switched between steps by a
// SwitchPolicy.
//
//   SolverSelector selector
//   (
//       parseSolverList("CG+Jacobi,CG+SGS,CG+IC0"),
//       options,
//       policy
//   );
//
//   KrylovResult r = selector.solve(A, b, x0);
//
// With policy.adaptive = false only the first candidate is
// used, which gives the fixed-pair runs the adaptive run is
// compared against.
//
// If the preconditioner setup of the current candidate fails
// (std::runtime_error, for example IC(0) on a matrix that is
// not SPD or a zero pivot) and the policy is adaptive, the
// selector moves to the next candidate and solves again in
// the same step; a fixed run rethrows.
//--------------------------------------------------

class SolverSelector
{
public:

    //--------------------------------------------------
    // Constructor
    //
    // Throws std::invalid_argument if candidates is empty or
    // if any threshold of an adaptive policy is not positive.
    //--------------------------------------------------

    SolverSelector
    (
        const std::vector<SolverChoice>& candidates_,
        const KrylovOptions& opt_,
        const SwitchPolicy& policy_ = SwitchPolicy()
    );

    //--------------------------------------------------
    // Solve
    //
    // x0 may be empty (zero initial guess). The result is the
    // KrylovResult of the choice that was used, and a record
    // of it is appended to the history.
    //--------------------------------------------------

    KrylovResult solve
    (
        const std::vector<std::vector<double>>& A,
        const std::vector<double>& b,
        const std::vector<double>& x0
    );

    //--------------------------------------------------
    // State
    //
    // current() is the choice the next solve will use.
    // reset goes back to the first candidate and clears the
    // history and the policy counters.
    //--------------------------------------------------

    const SolverChoice& current() const;

    int currentIndex() const;

    const std::vector<SolveRecord>& history() const;

    void reset();

    //--------------------------------------------------
    // Totals over the history
    //
    // totalSeconds = setup + solve time. Work is counted in
    // products with A and applications of M^-1, the same
    // quantities KrylovResult reports.
    //--------------------------------------------------

    double totalSeconds() const;

    double totalSetupSeconds() const;

    long totalIterations() const;

    long totalMatVecs() const;

    long totalPrecondApplies() const;

    int numSwitches() const;

    int numFallbacks() const;

    //--------------------------------------------------
    // Data
    //--------------------------------------------------

    std::vector<SolverChoice> candidates;

    KrylovOptions opt;

    SwitchPolicy policy;

private:

    int index;

    int step;

    // Solves since the last switch
    int sinceSwitch;

    // Consecutive easy solves
    int easyCount;

    std::vector<SolveRecord> records;

    KrylovResult runChoice
    (
        const SolverChoice& c,
        const std::vector<std::vector<double>>& A,
        const std::vector<double>& b,
        const std::vector<double>& x0,
        double& setupSeconds
    ) const;

    // Updates index from the solve just made; true if it moved
    bool updatePolicy(const KrylovResult& r);
};

#endif