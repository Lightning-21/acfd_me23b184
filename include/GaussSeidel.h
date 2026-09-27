#ifndef GAUSSSEIDEL_H
#define GAUSSSEIDEL_H

#include <vector>

class GaussSeidel
{
public:

    //--------------------------------------------------
    // Smoothing / Solving
    //
    // Generic for any central Laplacian stencil order.
    // Reuses LaplacianOperator::computeWeights, residual
    // and ghostAwareX/Y, so sweep() updates every physical
    // interior node correctly even one or more nodes in
    // from a Dirichlet boundary. Used both as the baseline
    // standalone solver and as the smoother inside
    // MultiGrid.
    //--------------------------------------------------

    static void sweep
    (
        std::vector<double>& phi,
        const std::vector<double>& f,
        int Nx,
        int Ny,
        double dx,
        double dy,
        const std::vector<int>& offsets
    );

    static double residualNorm
    (
        const std::vector<double>& phi,
        const std::vector<double>& f,
        int Nx,
        int Ny,
        double dx,
        double dy,
        const std::vector<int>& offsets
    );

    static int solve
    (
        std::vector<double>& phi,
        const std::vector<double>& f,
        int Nx,
        int Ny,
        double dx,
        double dy,
        const std::vector<int>& offsets,
        int maxIter,
        double tol
    );
};

#endif