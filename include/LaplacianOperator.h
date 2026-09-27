#ifndef LAPLACIANOPERATOR_H
#define LAPLACIANOPERATOR_H

#include <vector>

class LaplacianOperator
{
public:

    //--------------------------------------------------
    // Stencil Result
    //--------------------------------------------------

    struct LaplacianStencil
    {
        std::vector<int> offsets;
        int achievedOrder;
    };

    //--------------------------------------------------
    // Stencil Construction
    //--------------------------------------------------

    static LaplacianStencil buildStencil(int orderAcc);

    static std::vector<double> computeWeights
    (
        const std::vector<int>& offsets,
        double h
    );

    //--------------------------------------------------
    // Operator Application
    //
    // Both act over every physical interior node (all
    // nodes except i = 0, Nx-1 and j = 0, Ny-1, which hold
    // the Dirichlet boundary values), not just the ones a
    // plain halfWidth margin would allow.
    //--------------------------------------------------

    static std::vector<double> apply
    (
        const std::vector<double>& phi,
        int Nx,
        int Ny,
        double dx,
        double dy,
        const std::vector<int>& offsets
    );

    static std::vector<double> residual
    (
        const std::vector<double>& phi,
        const std::vector<double>& f,
        int Nx,
        int Ny,
        double dx,
        double dy,
        const std::vector<int>& offsets
    );

    //--------------------------------------------------
    // Ghost-Aware Neighbour Access
    //
    // Value of phi at (i+offset, j) [ghostAwareX] or
    // (i, j+offset) [ghostAwareY]. If that node lies
    // inside [0, N-1] it is just read directly. If it
    // falls past the boundary (which happens whenever a
    // node is closer than halfWidth to the edge, for any
    // stencil wider than the 3-point 2nd-order one), the
    // value is extrapolated instead: a polynomial is fit
    // through the boundary node and the halfWidth nearest
    // interior nodes on that side, and evaluated at the
    // ghost location. This is what apply()/residual() and
    // GaussSeidel::sweep use so 4th/6th-order stencils are
    // handled correctly right up to the boundary, instead
    // of silently leaving near-boundary nodes unupdated.
    //--------------------------------------------------

    static double ghostAwareX
    (
        const std::vector<double>& phi,
        int i,
        int j,
        int Nx,
        int Ny,
        int offset,
        int halfWidth
    );

    static double ghostAwareY
    (
        const std::vector<double>& phi,
        int i,
        int j,
        int Nx,
        int Ny,
        int offset,
        int halfWidth
    );

private:

    static std::vector<double> fdWeights(const std::vector<int>& offsets);

    static int verifyOrder(const std::vector<int>& offsets);
};

#endif