#ifndef MULTIGRID_H
#define MULTIGRID_H

#include <vector>
#include <string>

enum class CycleType { V, W, F };

struct GridLevel
{
    int Nx;
    int Ny;
    double dx;
    double dy;
    std::vector<double> phi;
    std::vector<double> f;
    std::vector<double> r;
};

class MultiGrid
{
public:

    //--------------------------------------------------
    // Solve
    //
    // Throws std::invalid_argument if numLevels < 1, or if
    // Nx-1/Ny-1 is not evenly divisible by 2^(numLevels-1)
    // (coarsening would not land on the same physical nodes
    // restrictField/prolongField assume). Throws
    // FiniteDifference::InsufficientNodesError if any level
    // in the resulting hierarchy is too small for the given
    // stencil's halfWidth.
    //--------------------------------------------------

    static int solve
    (
        CycleType cycleType,
        int numLevels,
        std::vector<double>& phi,
        const std::vector<double>& f,
        int Nx,
        int Ny,
        double dx,
        double dy,
        const std::vector<int>& offsets,
        int maxCycles,
        double tol,
        int nu1 = 2,
        int nu2 = 2
    );

private:

    //--------------------------------------------------
    // Throws std::invalid_argument if N-1 is not evenly
    // divisible by 2^(numLevels-1) — i.e. this many levels
    // cannot coarsen this grid down cleanly.
    //--------------------------------------------------

    static void validateCoarsening
    (
        int N,
        int numLevels,
        const std::string& label
    );

    static std::vector<GridLevel> buildHierarchy
    (
        int numLevels,
        const std::vector<double>& phi,
        const std::vector<double>& f,
        int Nx,
        int Ny,
        double dx,
        double dy
    );

    //--------------------------------------------------
    // Restriction / Prolongation
    //
    // Boundary detection here now checks only the true
    // physical edge (index 0 or N-1 on the coarse/fine
    // grid), not a halfWidth-wide margin. That margin
    // assumption is what LaplacianOperator/GaussSeidel
    // used to (incorrectly) rely on; once those two treat
    // every non-edge node as ordinary interior, restriction
    // and prolongation can do the same.
    //--------------------------------------------------

    static std::vector<double> restrictField
    (
        const std::vector<double>& fine,
        int NxFine,
        int NyFine,
        int NxCoarse,
        int NyCoarse
    );

    static std::vector<double> prolongField
    (
        const std::vector<double>& coarse,
        int NxCoarse,
        int NyCoarse,
        int NxFine,
        int NyFine
    );

    static void solveCoarsest
    (
        GridLevel& level,
        const std::vector<int>& offsets
    );

    static void cycle
    (
        std::vector<GridLevel>& levels,
        int level,
        int gamma,
        const std::vector<int>& offsets,
        int nu1,
        int nu2
    );

    static void fmg
    (
        std::vector<GridLevel>& levels,
        const std::vector<int>& offsets,
        int nu1,
        int nu2
    );
};

#endif