#ifndef TWOPHASEFLOW_H
#define TWOPHASEFLOW_H

#include <vector>

#include "KrylovSolvers.h"
#include "LevelSet.h"
#include "SolverSelector.h"

//--------------------------------------------------
// Wall Conditions
//
// Every wall is impermeable. The tangential condition is
// either
//
//   NoSlip   : tangential velocity zero at the wall
//   FreeSlip : zero tangential shear stress at the wall
//
// Moving walls are not supported.
//
// The defaults are those of the rising bubble benchmark of
// Hysing et al. (2009): no-slip on the top and bottom walls,
// free-slip on the two vertical walls.
//--------------------------------------------------

enum class WallType
{
    NoSlip,
    FreeSlip
};

struct WallConditions
{
    WallType left = WallType::FreeSlip;

    WallType right = WallType::FreeSlip;

    WallType bottom = WallType::NoSlip;

    WallType top = WallType::NoSlip;
};

//--------------------------------------------------
// Convection Scheme
//
// Applied to the advective term u . grad(u), evaluated in
// non-conservative form with explicit time stepping.
//
//   Upwind  : first-order upwind, robust
//   Central : second-order central; with explicit Euler it
//             needs a small cell Reynolds number
//--------------------------------------------------

enum class TPConvection
{
    Upwind,
    Central
};

//--------------------------------------------------
// Parameters
//
// rho1, mu1 : liquid (level set phi > 0)
// rho2, mu2 : bubble (level set phi < 0)
// sigma     : surface tension coefficient
// gx, gy    : gravitational acceleration (a body force
//             per unit mass, so gy = -g points down)
//
// The defaults are the values of test case 1 of Hysing et al.
// (2009), Table 1: rho1 = 1000, rho2 = 100, mu1 = 10, mu2 = 1,
// g = 0.98, sigma = 24.5 (Re = 35, Eo = 10), on a 1 x 2 domain
// with a bubble of radius 0.25 at (0.5, 0.5), run to t = 3.
//
// reinitEvery : reinitialise phi every this many steps
//               (0 = never)
// reinitSteps : pseudo-time steps per reinitialisation, each of
//               dtau = min(dx,dy)/10. The equation is marched
//               until |grad(phi)| = 1 inside the interface
//               thickness, so about eps/dtau = 15 steps are
//               needed for eps = 1.5 dx.
//--------------------------------------------------

struct TwoPhaseParameters
{
    double rho1 = 1000.0;

    double rho2 = 100.0;

    double mu1 = 10.0;

    double mu2 = 1.0;

    double sigma = 24.5;

    double gx = 0.0;

    double gy = -0.98;

    WallConditions walls;

    TPConvection convection = TPConvection::Upwind;

    int reinitEvery = 1;

    int reinitSteps = 15;
};

//--------------------------------------------------
// Step Information
//
// Summary of the pressure solve of one time step. The full
// per-solve record (setup time, solve time, candidate name)
// is in the selector's history.
//--------------------------------------------------

struct TwoPhaseStepInfo
{
    double dt = 0.0;

    KrylovStatus status = KrylovStatus::MaxIterations;

    int iterations = 0;

    int matVecs = 0;

    int precondApplies = 0;

    double trueRelResidual = 0.0;

    // max |div u| after the projection
    double maxDivergence = 0.0;
};

//--------------------------------------------------
// TwoPhaseFlow
//
// 2D incompressible two-phase flow in a closed rectangular
// column: level set interface, variable density and
// viscosity, continuum surface force, projection method.
// Uniform Cartesian grid, staggered (MAC) layout, the same
// as NavierStokes2D and LevelSet:
//
//   p(I,J)  at cell centres     I = 0..Nx-1, J = 0..Ny-1
//   u(i,J)  at vertical faces   i = 0..Nx,   J = 0..Ny-1
//   v(I,j)  at horizontal faces I = 0..Nx-1, j = 0..Ny
//
// u(0,J), u(Nx,J), v(I,0) and v(I,Ny) lie on the walls and
// stay zero. Only the interior faces are unknowns.
//
// One time step (step), in the order of the solution
// algorithm of Sharma (2015), Sec. 2.3:
//
//   1. density and viscosity from the current phi
//   2. predictor, explicit Euler:
//
//        u* = u^n + dt [ - u.grad(u) + (1/rho) div(mu (grad u
//             + grad u^T)) + g + F_sigma/rho ]
//
//      The viscous term is div(2 mu D), D = (grad u +
//      grad u^T)/2, and F_sigma = sigma kappa delta(phi) n.
//   3. pressure Poisson equation
//
//        div( (1/rho) grad(p) ) = div(u*) / dt
//
//      with the face value 1/rho_f = 2/(rho_L + rho_R), the
//      harmonic mean of the two cell values of 1/rho. The
//      same face value is used in the correction, so the
//      projection is consistent. The matrix changes every
//      step and is DENSE (std::vector<std::vector<double>>),
//      symmetric positive definite once the pressure of
//      refCell is pinned to zero.
//   4. correction  u^{n+1} = u* - dt (1/rho_f) grad(p)
//   5. advect phi with u^{n+1} (LevelSet), then reinitialise it
//
// The pressure solve goes through a SolverSelector, so the
// Krylov method and preconditioner are chosen (and may be
// switched) outside this class.
//
// The viscous term uses the viscosity at cell centres for the
// normal stresses and the average of the four surrounding
// cells at the cell corners for the shear stress.
//--------------------------------------------------

class TwoPhaseFlow
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

    TwoPhaseParameters params;

    LevelSet level;

    // Current fields
    std::vector<double> u;

    std::vector<double> v;

    std::vector<double> p;

    // Density and viscosity at cell centres
    std::vector<double> rho;

    std::vector<double> mu;

    // Pressure cell pinned to p = 0
    int refCell;

    double time;

    int stepCount;

    //--------------------------------------------------
    // Constructor
    //
    // Throws std::invalid_argument for Nx < 3, Ny < 3,
    // non-positive lengths or densities, negative viscosity or
    // surface tension, and when Nx*Ny exceeds maxUnknowns.
    //
    // Memory: the dense pressure matrix needs 8 (Nx*Ny)^2 bytes
    // (about 2 GB for 16000 unknowns). A preconditioner built by
    // buildPreconditioner stores four more dense matrices of the
    // same size (F1, F2, M1, M2), also for Jacobi, so the peak
    // while solving is about five times that: roughly 10 GB for
    // 16000 unknowns and 6.6 GB for the 80 x 160 grid. Lower
    // maxUnknowns to fit the machine.
    //--------------------------------------------------

    TwoPhaseFlow
    (
        int Nx_,
        int Ny_,
        double Lx_,
        double Ly_,
        const TwoPhaseParameters& params_,
        double epsFactor = 1.5,
        int maxUnknowns = 16000
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
    // Initial State
    //
    // A circular bubble of radius R centred at (xCentre,
    // yCentre), fluid at rest, p = 0, time = 0. Density and
    // viscosity are filled from the level set.
    //--------------------------------------------------

    void initBubble
    (
        double xCentre,
        double yCentre,
        double R
    );

    //--------------------------------------------------
    // Time Step Limit
    //
    // cfl times the smallest of
    //
    //   convective : 1 / (max|u|/dx + max|v|/dy)
    //   viscous    : (1/4) min(dx,dy)^2 / max(mu/rho)
    //   capillary  : sqrt((rho1 + rho2) h^3 / (4 pi sigma)),
    //                h = min(dx,dy)  (Brackbill et al.)
    //   gravity    : sqrt(h / |g|)
    //
    // Terms that do not apply (zero velocity, sigma = 0,
    // g = 0) are left out.
    //--------------------------------------------------

    double stableTimeStep(double cfl) const;

    //--------------------------------------------------
    // Properties
    //
    // rho and mu from the current level set.
    //--------------------------------------------------

    void computeProperties();

    //--------------------------------------------------
    // Time Step
    //
    // Advances u, v, p and phi by dt (the steps listed above;
    // rho and mu stay those of the old interface) and fills
    // info. p is used as the initial guess of the
    // pressure solve. Throws std::invalid_argument for
    // dt <= 0.
    //--------------------------------------------------

    void step
    (
        double dt,
        SolverSelector& solver,
        TwoPhaseStepInfo& info
    );

    //--------------------------------------------------
    // The Pieces of a Step
    //
    // Public so they can be tested one at a time.
    // predictVelocity needs rho and mu to be current and
    // returns u*, v* (the stored u, v are the old level).
    // assemblePressureSystem builds the dense matrix and the
    // right-hand side from u*, v* and dt. correctVelocity
    // overwrites u, v with u* - dt (1/rho_f) grad(p), using the
    // stored p.
    //--------------------------------------------------

    void predictVelocity
    (
        double dt,
        std::vector<double>& us,
        std::vector<double>& vs
    ) const;

    void assemblePressureSystem
    (
        const std::vector<double>& us,
        const std::vector<double>& vs,
        double dt
    );

    void correctVelocity
    (
        double dt,
        const std::vector<double>& us,
        const std::vector<double>& vs
    );

    //--------------------------------------------------
    // Pressure System
    //
    // The matrix and right-hand side of the last
    // assemblePressureSystem call.
    //--------------------------------------------------

    const std::vector<std::vector<double>>& pressureMatrix() const
    {
        return A;
    }

    const std::vector<double>& pressureRhs() const
    {
        return b;
    }

    //--------------------------------------------------
    // Diagnostics
    //
    // max |div u| over the cells for the stored u, v.
    //--------------------------------------------------

    double maxDivergence() const;

private:

    std::vector<std::vector<double>> A;

    std::vector<double> b;

    //--------------------------------------------------
    // Face values of 1/rho: harmonic mean of the two
    // neighbouring cell values, 2/(rho_L + rho_R). Interior
    // faces only.
    //--------------------------------------------------

    double invRhoU
    (
        int i,
        int J
    ) const;

    double invRhoV
    (
        int I,
        int j
    ) const;

    //--------------------------------------------------
    // Velocity with the wall ghost value
    //
    // uAt accepts J = -1 and J = Ny, vAt accepts I = -1 and
    // I = Nx. The ghost is -u (no-slip) or +u (free-slip) of
    // the first cell inside the wall, which puts the wall
    // value at zero or the wall gradient at zero.
    //--------------------------------------------------

    double uAt
    (
        const std::vector<double>& f,
        int i,
        int J
    ) const;

    double vAt
    (
        const std::vector<double>& f,
        int I,
        int j
    ) const;

    //--------------------------------------------------
    // Shear stress mu (du/dy + dv/dx) at the cell corner
    // (inode, jnode), inode = 0..Nx, jnode = 0..Ny, with mu
    // averaged from the surrounding cells.
    //--------------------------------------------------

    double shearStress
    (
        const std::vector<double>& uf,
        const std::vector<double>& vf,
        int inode,
        int jnode
    ) const;
};

#endif