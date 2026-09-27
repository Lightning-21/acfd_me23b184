#ifndef FINITEDIFFERENCE_H
#define FINITEDIFFERENCE_H

#include <vector>

class FiniteDifference
{
public:

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

private:

    static int verifyOrder(const std::vector<int>& offsets);
};

#endif