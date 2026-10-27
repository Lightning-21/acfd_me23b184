//------------------------------------------------------------//
// File: LevelSet.cpp
//
// Description:
//
// Level set function for the 2D two-phase (rising bubble)
// problem on a uniform Cartesian grid. phi is stored at the cell
// centres; the advecting velocity lives on the staggered (MAC)
// faces, matching the layout of NavierStokes2D.
//
//   Advection        : first-order upwind, 3rd-order TVD RK
//   Reinitialisation : Sussman-Smereka-Osher pseudo-time PDE,
//                      Godunov upwind |grad(phi)|, dtau = dx/10
//   Interface        : smoothed Heaviside / delta of half-width
//                      eps, density and viscosity fields,
//                      curvature and continuum surface force
//
// Formulas follow A. Sharma, Sadhana 40(3), 2015, Eqs. (1)-(4),
// (7), (9) and (11): n = grad(phi)/|grad(phi)|, kappa = -div(n),
// H_eps and delta_eps with 2 eps = 3 dx, rho_m = H + chi (1 - H),
// phi_tau = S_eps(phi0) (1 - |grad(phi)|), F = sigma kappa n delta.
//
// Sign convention: phi > 0 in the liquid, phi < 0 in the bubble.
//
//------------------------------------------------------------//

#include "LevelSet.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

using namespace std;

//------------------------------------------------------------//
// Constructor
//------------------------------------------------------------//

LevelSet::LevelSet
(
    int Nx_,
    int Ny_,
    double Lx_,
    double Ly_,
    double epsFactor
)
:
    Nx(Nx_),
    Ny(Ny_),
    Lx(Lx_),
    Ly(Ly_),
    dx(0.0),
    dy(0.0),
    eps(0.0)
{
    if(Nx < 3 || Ny < 3)
    {
        throw invalid_argument
        (
            "LevelSet: Nx and Ny must be at least 3 (got "+
            to_string(Nx)+" and "+to_string(Ny)+")."
        );
    }

    if(Lx <= 0.0 || Ly <= 0.0)
    {
        throw invalid_argument("LevelSet: Lx and Ly must be positive.");
    }

    if(epsFactor <= 0.0)
    {
        throw invalid_argument("LevelSet: epsFactor must be positive.");
    }

    dx = Lx/Nx;
    dy = Ly/Ny;

    eps = epsFactor*max(dx, dy);

    phi.assign(Nx*Ny, 0.0);
}

//------------------------------------------------------------//
// Initialisation
//------------------------------------------------------------//

void LevelSet::initCircle
(
    double xCentre,
    double yCentre,
    double R
)
{
    for(int J = 0; J < Ny; J++)
    {
        for(int I = 0; I < Nx; I++)
        {
            double ex = xc(I) - xCentre;
            double ey = yc(J) - yCentre;

            phi[idx(I, J)] = sqrt(ex*ex + ey*ey) - R;
        }
    }
}

//------------------------------------------------------------//
// Index Clamping and Derivatives
//------------------------------------------------------------//

int LevelSet::clampI(int I) const
{
    return min(max(I, 0), Nx-1);
}

int LevelSet::clampJ(int J) const
{
    return min(max(J, 0), Ny-1);
}

double LevelSet::ddx
(
    const vector<double>& f,
    int I,
    int J
) const
{
    if(I == 0)
    {
        return (f[idx(1, J)] - f[idx(0, J)])/dx;
    }

    if(I == Nx-1)
    {
        return (f[idx(Nx-1, J)] - f[idx(Nx-2, J)])/dx;
    }

    return (f[idx(I+1, J)] - f[idx(I-1, J)])/(2.0*dx);
}

double LevelSet::ddy
(
    const vector<double>& f,
    int I,
    int J
) const
{
    if(J == 0)
    {
        return (f[idx(I, 1)] - f[idx(I, 0)])/dy;
    }

    if(J == Ny-1)
    {
        return (f[idx(I, Ny-1)] - f[idx(I, Ny-2)])/dy;
    }

    return (f[idx(I, J+1)] - f[idx(I, J-1)])/(2.0*dy);
}

//------------------------------------------------------------//
// Advection
//------------------------------------------------------------//

void LevelSet::centreVelocities
(
    const vector<double>& u,
    const vector<double>& v,
    vector<double>& ucc,
    vector<double>& vcc
) const
{
    ucc.assign(Nx*Ny, 0.0);
    vcc.assign(Nx*Ny, 0.0);

    for(int J = 0; J < Ny; J++)
    {
        for(int I = 0; I < Nx; I++)
        {
            ucc[idx(I, J)] = 0.5*(u[J*(Nx+1) + I] + u[J*(Nx+1) + I+1]);

            vcc[idx(I, J)] = 0.5*(v[J*Nx + I] + v[(J+1)*Nx + I]);
        }
    }
}

void LevelSet::advectionRHS
(
    const vector<double>& phiIn,
    const vector<double>& ucc,
    const vector<double>& vcc,
    vector<double>& out
) const
{
    out.assign(Nx*Ny, 0.0);

    for(int J = 0; J < Ny; J++)
    {
        for(int I = 0; I < Nx; I++)
        {
            int k = idx(I, J);

            double p = phiIn[k];

            double pW = phiIn[idx(clampI(I-1), J)];
            double pE = phiIn[idx(clampI(I+1), J)];
            double pS = phiIn[idx(I, clampJ(J-1))];
            double pN = phiIn[idx(I, clampJ(J+1))];

            double dpdx = (ucc[k] > 0.0) ? (p - pW)/dx : (pE - p)/dx;
            double dpdy = (vcc[k] > 0.0) ? (p - pS)/dy : (pN - p)/dy;

            out[k] = -(ucc[k]*dpdx + vcc[k]*dpdy);
        }
    }
}

double LevelSet::advectiveTimeStep
(
    const vector<double>& u,
    const vector<double>& v,
    double cfl
) const
{
    double uMax = 0.0;
    double vMax = 0.0;

    for(size_t k = 0; k < u.size(); k++)
    {
        uMax = max(uMax, fabs(u[k]));
    }

    for(size_t k = 0; k < v.size(); k++)
    {
        vMax = max(vMax, fabs(v[k]));
    }

    double rate = uMax/dx + vMax/dy;

    if(rate <= 0.0)
    {
        return numeric_limits<double>::max();
    }

    return cfl/rate;
}

void LevelSet::advect
(
    const vector<double>& u,
    const vector<double>& v,
    double dt
)
{
    if((int)u.size() != (Nx+1)*Ny)
    {
        throw invalid_argument
        (
            "LevelSet::advect: u has size "+to_string(u.size())+
            ", expected (Nx+1)*Ny = "+to_string((Nx+1)*Ny)+"."
        );
    }

    if((int)v.size() != Nx*(Ny+1))
    {
        throw invalid_argument
        (
            "LevelSet::advect: v has size "+to_string(v.size())+
            ", expected Nx*(Ny+1) = "+to_string(Nx*(Ny+1))+"."
        );
    }

    int n = Nx*Ny;

    vector<double> ucc;
    vector<double> vcc;

    centreVelocities(u, v, ucc, vcc);

    vector<double> L;
    vector<double> phi1(n);
    vector<double> phi2(n);

    // Stage 1: phi1 = phi + dt L(phi)
    advectionRHS(phi, ucc, vcc, L);

    for(int k = 0; k < n; k++)
    {
        phi1[k] = phi[k] + dt*L[k];
    }

    // Stage 2: phi2 = 3/4 phi + 1/4 (phi1 + dt L(phi1))
    advectionRHS(phi1, ucc, vcc, L);

    for(int k = 0; k < n; k++)
    {
        phi2[k] = 0.75*phi[k] + 0.25*(phi1[k] + dt*L[k]);
    }

    // Stage 3: phi = 1/3 phi + 2/3 (phi2 + dt L(phi2))
    advectionRHS(phi2, ucc, vcc, L);

    for(int k = 0; k < n; k++)
    {
        phi[k] = (phi[k] + 2.0*(phi2[k] + dt*L[k]))/3.0;
    }
}

//------------------------------------------------------------//
// Reinitialisation
//------------------------------------------------------------//

double LevelSet::reinitialise
(
    int nSteps,
    double dtauFactor
)
{
    if(nSteps < 0)
    {
        throw invalid_argument("LevelSet::reinitialise: nSteps must not be negative.");
    }

    if(dtauFactor <= 0.0)
    {
        throw invalid_argument("LevelSet::reinitialise: dtauFactor must be positive.");
    }

    int n = Nx*Ny;

    double h = min(dx, dy);

    double dtau = dtauFactor*h;

    // Smoothed sign of the initial field, held fixed
    vector<double> S(n);

    for(int k = 0; k < n; k++)
    {
        S[k] = phi[k]/sqrt(phi[k]*phi[k] + h*h);
    }

    vector<double> phiNew(n);

    double maxChange = 0.0;

    for(int step = 0; step < nSteps; step++)
    {
        maxChange = 0.0;

        for(int J = 0; J < Ny; J++)
        {
            for(int I = 0; I < Nx; I++)
            {
                int k = idx(I, J);

                double p = phi[k];

                double pW = phi[idx(clampI(I-1), J)];
                double pE = phi[idx(clampI(I+1), J)];
                double pS = phi[idx(I, clampJ(J-1))];
                double pN = phi[idx(I, clampJ(J+1))];

                // Backward (a) and forward (b) differences
                double ax = (p - pW)/dx;
                double bx = (pE - p)/dx;
                double ay = (p - pS)/dy;
                double by = (pN - p)/dy;

                double gx2;
                double gy2;

                if(S[k] > 0.0)
                {
                    double t1 = max(ax, 0.0);
                    double t2 = min(bx, 0.0);

                    gx2 = max(t1*t1, t2*t2);

                    t1 = max(ay, 0.0);
                    t2 = min(by, 0.0);

                    gy2 = max(t1*t1, t2*t2);
                }
                else
                {
                    double t1 = min(ax, 0.0);
                    double t2 = max(bx, 0.0);

                    gx2 = max(t1*t1, t2*t2);

                    t1 = min(ay, 0.0);
                    t2 = max(by, 0.0);

                    gy2 = max(t1*t1, t2*t2);
                }

                double gradMag = sqrt(gx2 + gy2);

                phiNew[k] = p - dtau*S[k]*(gradMag - 1.0);

                maxChange = max(maxChange, fabs(phiNew[k] - p));
            }
        }

        phi = phiNew;
    }

    return maxChange;
}

//------------------------------------------------------------//
// Smoothed Heaviside and Delta
//------------------------------------------------------------//

double LevelSet::smoothedHeaviside(double p) const
{
    if(p <= -eps)
    {
        return 0.0;
    }

    if(p >= eps)
    {
        return 1.0;
    }

    const double pi = acos(-1.0);

    return 0.5*(1.0 + p/eps + sin(pi*p/eps)/pi);
}

double LevelSet::smoothedDelta(double p) const
{
    if(fabs(p) >= eps)
    {
        return 0.0;
    }

    const double pi = acos(-1.0);

    return 0.5/eps*(1.0 + cos(pi*p/eps));
}

//------------------------------------------------------------//
// Material Property Field
//------------------------------------------------------------//

void LevelSet::fillProperty
(
    double value1,
    double value2,
    vector<double>& out
) const
{
    out.assign(Nx*Ny, 0.0);

    for(int k = 0; k < Nx*Ny; k++)
    {
        out[k] = value2 + (value1 - value2)*smoothedHeaviside(phi[k]);
    }
}

//------------------------------------------------------------//
// Curvature
//------------------------------------------------------------//

void LevelSet::curvature(vector<double>& kappa) const
{
    int n = Nx*Ny;

    vector<double> nx(n);
    vector<double> ny(n);

    // Unit normal grad(phi)/|grad(phi)| at the cell centres
    for(int J = 0; J < Ny; J++)
    {
        for(int I = 0; I < Nx; I++)
        {
            double gx = ddx(phi, I, J);
            double gy = ddy(phi, I, J);

            double mag = max(sqrt(gx*gx + gy*gy), 1.0e-12);

            nx[idx(I, J)] = gx/mag;
            ny[idx(I, J)] = gy/mag;
        }
    }

    kappa.assign(n, 0.0);

    for(int J = 0; J < Ny; J++)
    {
        for(int I = 0; I < Nx; I++)
        {
            kappa[idx(I, J)] = -(ddx(nx, I, J) + ddy(ny, I, J));
        }
    }
}

//------------------------------------------------------------//
// Continuum Surface Force
//------------------------------------------------------------//

void LevelSet::surfaceForce
(
    double sigma,
    vector<double>& fu,
    vector<double>& fv
) const
{
    vector<double> kappa;

    curvature(kappa);

    fu.assign((Nx+1)*Ny, 0.0);
    fv.assign(Nx*(Ny+1), 0.0);

    // u faces between cell (i-1,J) and cell (i,J); walls stay zero
    for(int J = 0; J < Ny; J++)
    {
        for(int i = 1; i < Nx; i++)
        {
            int kL = idx(i-1, J);
            int kR = idx(i, J);

            double dphidx = (phi[kR] - phi[kL])/dx;

            double dphidy = 0.5*(ddy(phi, i-1, J) + ddy(phi, i, J));

            double mag = max(sqrt(dphidx*dphidx + dphidy*dphidy), 1.0e-12);

            double kappaF = 0.5*(kappa[kL] + kappa[kR]);

            double phiF = 0.5*(phi[kL] + phi[kR]);

            fu[J*(Nx+1) + i] = sigma*kappaF*smoothedDelta(phiF)*dphidx/mag;
        }
    }

    // v faces between cell (I,j-1) and cell (I,j); walls stay zero
    for(int j = 1; j < Ny; j++)
    {
        for(int I = 0; I < Nx; I++)
        {
            int kS = idx(I, j-1);
            int kN = idx(I, j);

            double dphidy = (phi[kN] - phi[kS])/dy;

            double dphidx = 0.5*(ddx(phi, I, j-1) + ddx(phi, I, j));

            double mag = max(sqrt(dphidx*dphidx + dphidy*dphidy), 1.0e-12);

            double kappaF = 0.5*(kappa[kS] + kappa[kN]);

            double phiF = 0.5*(phi[kS] + phi[kN]);

            fv[j*Nx + I] = sigma*kappaF*smoothedDelta(phiF)*dphidy/mag;
        }
    }
}