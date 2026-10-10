#ifndef NAVIERSTOKES2D_H
#define NAVIERSTOKES2D_H

#include <vector>

//--------------------------------------------------
// Convection Scheme
//
// Upwind : a = D + max(-F, 0)
// Hybrid : a = max(-F, D - F/2, 0)
//
// F is the mass flux leaving the control volume through the
// face and D the diffusion conductance of that face.
//--------------------------------------------------

enum class ConvectionScheme
{
    Upwind,
    Hybrid
};

//--------------------------------------------------
// Wall Velocities
//
// Tangential velocity of each wall. Wall-normal velocity is
// zero on every wall. The lid-driven cavity is uTop = 1 and
// everything else 0.
//--------------------------------------------------

struct WallVelocities
{
    double uBottom = 0.0;

    double uTop = 0.0;

    double vLeft = 0.0;

    double vRight = 0.0;
};

//--------------------------------------------------
// Momentum Coefficients
//
// One set per velocity component, stored at every face of
// that component (only interior faces are used).
//
//   aP u_P = aE u_E + aW u_W + aN u_N + aS u_S + b
//            + pressure force
//
// aP0, b0 : steady / time-step form before under-relaxation
// aP , b  : after under-relaxation (used by the solve)
// d       : velocity-correction coefficient, u' = d (p'_W - p'_E)
//           (zero on boundary faces)
//--------------------------------------------------

struct MomentumCoeffs
{
    std::vector<double> aE;

    std::vector<double> aW;

    std::vector<double> aN;

    std::vector<double> aS;

    std::vector<double> aP;

    std::vector<double> aP0;

    std::vector<double> b;

    std::vector<double> b0;

    std::vector<double> d;
};

//--------------------------------------------------
// Pressure-Correction Coefficients
//
//   aP p'_P = aE p'_E + aW p'_W + aN p'_N + aS p'_S + b
//
// One entry per pressure cell. Boundary faces have d = 0, so
// the cells next to a wall carry no coupling across it
// (Neumann condition for free).
//--------------------------------------------------

struct PressureCoeffs
{
    std::vector<double> aE;

    std::vector<double> aW;

    std::vector<double> aN;

    std::vector<double> aS;

    std::vector<double> aP;
};

//--------------------------------------------------
// Linear Solve Information
//--------------------------------------------------

struct LinearSolveInfo
{
    int iterations = 0;

    // 2-norm of the right-hand side residual at the start
    double initialResidual = 0.0;

    // Final residual relative to the initial one
    double relResidual = 0.0;

    bool converged = false;
};

//--------------------------------------------------
// NavierStokes2D
//
// Shared building blocks for the pressure-velocity coupling
// algorithms (SIMPLE, SIMPLEC, PISO, PIMPLE). Laminar, 2D,
// incompressible, uniform Cartesian grid, staggered (MAC)
// layout:
//
//   p(I,J)  at cell centres     I = 0..Nx-1, J = 0..Ny-1
//   u(i,J)  at vertical faces   i = 0..Nx,   J = 0..Ny-1
//   v(I,j)  at horizontal faces I = 0..Nx-1, j = 0..Ny
//
// u(0,J), u(Nx,J), v(I,0) and v(I,Ny) lie on the walls and
// stay zero (impermeable walls). Only the interior faces
// are unknowns. The algorithms themselves are NOT here; this
// class only supplies the pieces they are built from.
//--------------------------------------------------

class NavierStokes2D
{
public:

    //--------------------------------------------------
    // Data
    //--------------------------------------------------

    int Nx;

    int Ny;

    double Lx;

    double Ly;

    double dx;

    double dy;

    double rho;

    double nu;

    WallVelocities walls;

    // Current fields
    std::vector<double> u;

    std::vector<double> v;

    std::vector<double> p;

    // Previous time level (used only when dt > 0)
    std::vector<double> uOld;

    std::vector<double> vOld;

    MomentumCoeffs cu;

    MomentumCoeffs cv;

    PressureCoeffs cp;

    // Pressure cell pinned to p' = 0 in the correction solve
    int refCell;

    //--------------------------------------------------
    // Constructor
    //--------------------------------------------------

    NavierStokes2D
    (
        int Nx_,
        int Ny_,
        double Lx_,
        double Ly_,
        double rho_,
        double nu_,
        const WallVelocities& walls_
    );

    //--------------------------------------------------
    // Indexing
    //--------------------------------------------------

    int uIdx
    (
        int i,
        int J
    ) const
    {
        return J*(Nx + 1) + i;
    }

    int vIdx
    (
        int I,
        int j
    ) const
    {
        return j*Nx + I;
    }

    int pIdx
    (
        int I,
        int J
    ) const
    {
        return J*Nx + I;
    }

    //--------------------------------------------------
    // Time Level
    //
    // Copies u, v into uOld, vOld. Call once at the start of
    // every time step of PISO / PIMPLE.
    //--------------------------------------------------

    void storeOldTimeLevel();

    //--------------------------------------------------
    // Momentum Equations
    //
    // assembleMomentum builds cu and cv from the current u, v
    // (mass fluxes) with the chosen convection scheme.
    //
    //   alphaU : implicit under-relaxation factor (1 = none)
    //   dt     : time step; dt <= 0 gives the steady equation,
    //            dt > 0 adds rho V/dt (u - uOld)
    //
    // The pressure force is not stored in b; it is read from
    // p whenever it is needed. sweepMomentum does Gauss-Seidel
    // sweeps on both components with the current p.
    //--------------------------------------------------

    void assembleMomentum
    (
        ConvectionScheme scheme,
        double alphaU,
        double dt
    );

    void sweepMomentum(int sweeps);

    // Sum over |residual| of the un-relaxed momentum equation
    // (raw, not normalised)
    double residualU() const;

    double residualV() const;

    // out = sum over neighbours of a_nb f_nb at every interior
    // face (the momentum "H" operator without the source)
    void neighbourSumU
    (
        const std::vector<double>& f,
        std::vector<double>& out
    ) const;

    void neighbourSumV
    (
        const std::vector<double>& f,
        std::vector<double>& out
    ) const;

    //--------------------------------------------------
    // Velocity-Correction Coefficients d
    //
    // consistent = false : d = A / aP              (SIMPLE, PISO)
    // consistent = true  : d = A / (aP - sum a_nb) (SIMPLEC)
    //
    // Falls back to A / aP at any face where aP - sum a_nb is
    // not safely positive.
    //--------------------------------------------------

    void computeD(bool consistent);

    //--------------------------------------------------
    // Pressure-Correction Equation
    //
    // assemblePressureCorrection fills cp from cu.d and cv.d.
    // The matrix is symmetric and singular (pure Neumann); the
    // solve pins refCell, which makes it SPD.
    //
    // massSource writes b(I,J) = (uf_w - uf_e) dy + (vf_s - vf_n) dx
    // for any velocity pair and returns sum |b| (the mass
    // residual) before the pinned cell is zeroed.
    //
    // solvePressureCorrection is a matrix-free (preconditioned)
    // CG: no dense matrix. pc is the correction p'. Optional
    // preconditioner is symmetric Gauss-Seidel on the 5-point
    // stencil. relTol is relative to the initial residual.
    //--------------------------------------------------

    void assemblePressureCorrection();

    double massSource
    (
        const std::vector<double>& uf,
        const std::vector<double>& vf,
        std::vector<double>& b
    ) const;

    void applyPressureOperator
    (
        const std::vector<double>& x,
        std::vector<double>& y
    ) const;

    LinearSolveInfo solvePressureCorrection
    (
        const std::vector<double>& rhs,
        std::vector<double>& pc,
        double relTol,
        int maxIter,
        bool usePreconditioner
    ) const;

    //--------------------------------------------------
    // Applying Corrections
    //
    // correctVelocities adds d (p'_W - p'_E) to the interior
    // faces of any velocity pair (not only the stored u, v).
    // updatePressure does p += alphaP p'.
    //--------------------------------------------------

    void correctVelocities
    (
        const std::vector<double>& pc,
        std::vector<double>& uf,
        std::vector<double>& vf
    ) const;

    void updatePressure
    (
        const std::vector<double>& pc,
        double alphaP
    );

    // Sum |b| of the current u, v
    double massResidual() const;

    //--------------------------------------------------
    // Sampling
    //
    // Bilinear interpolation of u or v at any (x, y) inside the
    // domain, using the tangential wall velocities between the
    // first interior node and the wall. The wall-normal
    // component is zero on its own walls (u on x = 0 and
    // x = Lx, v on y = 0 and y = Ly). For centreline profiles
    // and comparison with benchmark data.
    //--------------------------------------------------

    double sampleU
    (
        double x,
        double y
    ) const;

    double sampleV
    (
        double x,
        double y
    ) const;

private:

    double sumNbU
    (
        const std::vector<double>& f,
        int i,
        int J
    ) const;

    double sumNbV
    (
        const std::vector<double>& f,
        int I,
        int j
    ) const;

    double columnU
    (
        int i,
        double y
    ) const;

    double rowV
    (
        int j,
        double x
    ) const;

    void applyPreconditioner
    (
        const std::vector<double>& r,
        std::vector<double>& z
    ) const;
};

#endif