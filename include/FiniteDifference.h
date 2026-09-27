#ifndef FINITEDIFFERENCE_H
#define FINITEDIFFERENCE_H

#include <vector>
#include <stdexcept>
#include <string>

class FiniteDifference
{
public:

    //--------------------------------------------------
    // Errors
    //
    // Thrown whenever a grid does not have enough nodes
    // along a direction to support a stencil of the given
    // halfWidth — e.g. a multigrid level that has coarsened
    // past the point where the stencil (even with ghost-
    // node extrapolation) can be evaluated at all. Callers
    // are expected to let this propagate rather than guess
    // at a degraded stencil.
    //--------------------------------------------------

    class InsufficientNodesError : public std::runtime_error
    {
    public:
        explicit InsufficientNodesError(const std::string& message);
    };

    //--------------------------------------------------
    // Stencil Result
    //--------------------------------------------------

    struct FDStencil
    {
        std::vector<int> offsets;
        int achievedOrder;
    };

    //--------------------------------------------------
    // Stencil Construction (first derivative only)
    //--------------------------------------------------

    static FDStencil buildStencil(int orderAcc);

    static std::vector<double> computeWeights
    (
        const std::vector<int>& offsets,
        double h
    );

    static std::vector<double> fdWeights
    (
        const std::vector<int>& offsets
    );

    //--------------------------------------------------
    // Shared Helpers (reused by LaplacianOperator)
    //--------------------------------------------------

    static std::vector<int> centralOffsets(int halfWidth);

    static std::vector<double> solveLinearSystem
    (
        std::vector<std::vector<double>> A,
        std::vector<double> b,
        int n
    );

    static double factorial(int k);

    //--------------------------------------------------
    // Ghost-Node Extrapolation
    //
    // Fits the unique polynomial through the given
    // (position, value) pairs and evaluates it at
    // targetPosition. Used by LaplacianOperator and
    // GaussSeidel to fill a value just outside the domain
    // when a stencil node falls past a Dirichlet boundary.
    //--------------------------------------------------

    static double extrapolate
    (
        const std::vector<double>& knownValues,
        const std::vector<double>& knownPositions,
        double targetPosition
    );

    //--------------------------------------------------
    // Grid-Size Validation
    //
    // Throws InsufficientNodesError if N (the node count
    // along one direction) is too small to evaluate a
    // stencil of the given halfWidth — i.e. N < halfWidth+1,
    // the minimum needed for ghostAwareX/Y's extrapolation
    // window to exist at all. label identifies the caller
    // and direction in the error message.
    //--------------------------------------------------

    static void checkGridSize
    (
        int N,
        int halfWidth,
        const std::string& label
    );

private:

    static int verifyOrder(const std::vector<int>& offsets);
};

#endif