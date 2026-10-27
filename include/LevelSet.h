#ifndef LEVELSET_H
#define LEVELSET_H

#include <vector>

//--------------------------------------------------
// LevelSet
//
// Level set function on a uniform Cartesian grid for the 2D
// two-phase flow problem. Sign convention:
//
//   phi > 0 : fluid 1 (the liquid column)
//   phi < 0 : fluid 2 (the bubble)
//   phi = 0 : the interface
//
// phi is stored at cell centres, with the same layout as the
// pressure in NavierStokes2D:
//
//   phi(I,J) at cell centres   I = 0..Nx-1, J = 0..Ny-1
//   index  = J*Nx + I
//   x(I)   = (I + 1/2) dx,   y(J) = (J + 1/2) dy
//
// The velocity that advects phi is on the staggered (MAC)
// faces, again as in NavierStokes2D:
//
//   u(i,J) at vertical faces    i = 0..Nx,   J = 0..Ny-1
//          index = J*(Nx+1) + i,   size (Nx+1)*Ny
//   v(I,j) at horizontal faces  I = 0..Nx-1, j = 0..Ny
//          index = j*Nx + I,       size Nx*(Ny+1)
//
// Every wall of the closed column is impermeable, so phi is
// advected and reinitialised with a zero-gradient (clamped)
// condition at the walls.
//
// This class only supplies the interface pieces: advection,
// reinitialisation, smoothed Heaviside / delta, density and
// viscosity fields, curvature and the continuum surface force.
// The flow solve is NOT here.
//--------------------------------------------------

class LevelSet
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

    // Half-width of the smoothed interface, eps = epsFactor*max(dx,dy)
    double eps;

    std::vector<double> phi;

    //--------------------------------------------------
    // Constructor
    //
    // Throws std::invalid_argument if Nx < 3, Ny < 3, Lx <= 0,
    // Ly <= 0 or epsFactor <= 0. phi is initialised to zero;
    // call initCircle (or fill phi directly) afterwards.
    //--------------------------------------------------

    LevelSet
    (
        int Nx_,
        int Ny_,
        double Lx_,
        double Ly_,
        double epsFactor = 1.5
    );

    //--------------------------------------------------
    // Indexing and Cell Centres
    //--------------------------------------------------

    int idx
    (
        int I,
        int J
    ) const
    {
        return J*Nx + I;
    }

    double xc(int I) const
    {
        return (I + 0.5)*dx;
    }

    double yc(int J) const
    {
        return (J + 0.5)*dy;
    }

    //--------------------------------------------------
    // Initialisation
    //
    // Signed distance to a circle of radius R centred at
    // (xCentre, yCentre): negative inside (the bubble),
    // positive outside (the liquid).
    //--------------------------------------------------

    void initCircle
    (
        double xCentre,
        double yCentre,
        double R
    );

    //--------------------------------------------------
    // Advection
    //
    //   phi_t + u . grad(phi) = 0
    //
    // First-order upwind in space, third-order TVD Runge-Kutta
    // (Shu-Osher) in time. u and v are the face velocities
    // described above; they are averaged to the cell centres
    // for the upwind decision and held fixed over the three
    // stages.
    //
    // Throws std::invalid_argument if u or v has the wrong size.
    //
    // advectiveTimeStep returns the largest dt for which the
    // advective CFL number does not exceed cfl:
    //
    //   dt = cfl / (max|u|/dx + max|v|/dy)
    //
    // and a very large value if the velocity is zero.
    //--------------------------------------------------

    double advectiveTimeStep
    (
        const std::vector<double>& u,
        const std::vector<double>& v,
        double cfl
    ) const;

    void advect
    (
        const std::vector<double>& u,
        const std::vector<double>& v,
        double dt
    );

    //--------------------------------------------------
    // Reinitialisation
    //
    //   phi_tau + S(phi0) (|grad(phi)| - 1) = 0
    //
    // with the smoothed sign S = phi0/sqrt(phi0^2 + h^2),
    // h = min(dx,dy), and the Godunov upwind approximation of
    // |grad(phi)|. Marched nSteps pseudo-time steps of
    // dtau = min(dx,dy)/2 starting from the current phi (phi0).
    //
    // Returns the largest change of phi in the last step, as a
    // convergence monitor.
    //--------------------------------------------------

    double reinitialise(int nSteps);

    //--------------------------------------------------
    // Smoothed Heaviside and Delta
    //
    //   H(p) = 0                                     p <= -eps
    //        = 1/2 (1 + p/eps + sin(pi p/eps)/pi)    |p| < eps
    //        = 1                                     p >= eps
    //
    //   delta(p) = dH/dp = 1/(2 eps) (1 + cos(pi p/eps))   |p| < eps
    //                    = 0                               otherwise
    //--------------------------------------------------

    double smoothedHeaviside(double p) const;

    double smoothedDelta(double p) const;

    //--------------------------------------------------
    // Material Property Field
    //
    //   out = value2 + (value1 - value2) H(phi)
    //
    // value1 is the liquid (phi > 0) and value2 the bubble
    // (phi < 0). Used for both density and viscosity. out is
    // resized to Nx*Ny.
    //--------------------------------------------------

    void fillProperty
    (
        double value1,
        double value2,
        std::vector<double>& out
    ) const;

    //--------------------------------------------------
    // Curvature
    //
    //   kappa = div( grad(phi) / |grad(phi)| )
    //
    // at the cell centres (central differences, one-sided at
    // the walls). With phi < 0 inside, kappa = +1/R on a circle
    // of radius R. kappa is resized to Nx*Ny.
    //--------------------------------------------------

    void curvature(std::vector<double>& kappa) const;

    //--------------------------------------------------
    // Continuum Surface Force
    //
    //   F = - sigma kappa delta(phi) grad(phi)
    //
    // per unit volume, evaluated at the interior faces of the
    // staggered grid: fu is the x-component on the u faces and
    // fv the y-component on the v faces. Faces on the walls are
    // set to zero. F points towards the centre of the bubble,
    // so a static circular bubble has p_inside - p_outside =
    // sigma/R when the momentum equation carries + F.
    //
    // fu is resized to (Nx+1)*Ny and fv to Nx*(Ny+1).
    //--------------------------------------------------

    void surfaceForce
    (
        double sigma,
        std::vector<double>& fu,
        std::vector<double>& fv
    ) const;

private:

    //--------------------------------------------------
    // Index clamping (zero-gradient condition at the walls)
    //--------------------------------------------------

    int clampI(int I) const;

    int clampJ(int J) const;

    //--------------------------------------------------
    // Cell-centre derivative of a cell-centred field: central
    // in the interior, one-sided at the walls
    //--------------------------------------------------

    double ddx
    (
        const std::vector<double>& f,
        int I,
        int J
    ) const;

    double ddy
    (
        const std::vector<double>& f,
        int I,
        int J
    ) const;

    //--------------------------------------------------
    // Face velocities averaged to the cell centres
    //--------------------------------------------------

    void centreVelocities
    (
        const std::vector<double>& u,
        const std::vector<double>& v,
        std::vector<double>& ucc,
        std::vector<double>& vcc
    ) const;

    //--------------------------------------------------
    // out = -(u phi_x + v phi_y), first-order upwind
    //--------------------------------------------------

    void advectionRHS
    (
        const std::vector<double>& phiIn,
        const std::vector<double>& ucc,
        const std::vector<double>& vcc,
        std::vector<double>& out
    ) const;
};

#endif