//------------------------------------------------------------//
// File: TwoPhaseFlow.cpp
//
// Description:
//
// 2D incompressible two-phase flow (rising bubble) on a uniform
// staggered grid. The interface is a level set (LevelSet), the
// density and viscosity follow from a smoothed Heaviside
// function, surface tension is the continuum surface force, and
// velocity and pressure are coupled by a projection method.
//
// The pressure equation
//
//   div( (1/rho) grad(p) ) = div(u*) / dt
//
// has a dense matrix that changes every step with the interface.
// It is symmetric, and positive definite once the pressure of one
// cell is pinned to zero (the pure-Neumann problem is otherwise
// singular). It is solved through a SolverSelector, so the Krylov
// method and preconditioner are chosen outside this file.
//
// The same face value of 1/rho, 2/(rho_L + rho_R), is used in the
// Poisson matrix and in the velocity correction, which keeps the
// projection consistent.
//
//------------------------------------------------------------//

#include "TwoPhaseFlow.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

using namespace std;

//------------------------------------------------------------//
// Constructor
//------------------------------------------------------------//

TwoPhaseFlow::TwoPhaseFlow
(
    int Nx_,
    int Ny_,
    double Lx_,
    double Ly_,
    const TwoPhaseParameters& params_,
    double epsFactor,
    int maxUnknowns
)
:
    Nx(Nx_),
    Ny(Ny_),
    Lx(Lx_),
    Ly(Ly_),
    dx(Lx_/Nx_),
    dy(Ly_/Ny_),
    params(params_),
    level(Nx_, Ny_, Lx_, Ly_, epsFactor),
    refCell(0),
    time(0.0),
    stepCount(0)
{
    if(params.rho1 <= 0.0 || params.rho2 <= 0.0)
    {
        throw invalid_argument("TwoPhaseFlow: densities must be positive.");
    }

    if(params.mu1 < 0.0 || params.mu2 < 0.0)
    {
        throw invalid_argument("TwoPhaseFlow: viscosities must not be negative.");
    }

    if(params.sigma < 0.0)
    {
        throw invalid_argument("TwoPhaseFlow: sigma must not be negative.");
    }

    int n = Nx*Ny;

    if(n > maxUnknowns)
    {
        throw invalid_argument
        (
            "TwoPhaseFlow: "+to_string(n)+" unknowns exceed the limit of "+
            to_string(maxUnknowns)+"; the dense pressure matrix would need "+
            to_string((double)n*(double)n*8.0/1.0e9)+" GB, and about five times "+
            "that with a dense preconditioner."
        );
    }

    u.assign((Nx+1)*Ny, 0.0);
    v.assign(Nx*(Ny+1), 0.0);
    p.assign(n, 0.0);

    rho.assign(n, params.rho1);
    mu.assign(n, params.mu1);

    A.assign(n, vector<double>(n, 0.0));
    b.assign(n, 0.0);
}

//------------------------------------------------------------//
// Initial State
//------------------------------------------------------------//

void TwoPhaseFlow::initBubble
(
    double xCentre,
    double yCentre,
    double R
)
{
    level.initCircle(xCentre, yCentre, R);

    u.assign((Nx+1)*Ny, 0.0);
    v.assign(Nx*(Ny+1), 0.0);
    p.assign(Nx*Ny, 0.0);

    time = 0.0;
    stepCount = 0;

    computeProperties();
}

//------------------------------------------------------------//
// Properties
//------------------------------------------------------------//

void TwoPhaseFlow::computeProperties()
{
    level.fillProperty(params.rho1, params.rho2, rho);

    level.fillProperty(params.mu1, params.mu2, mu);
}

//------------------------------------------------------------//
// Time Step Limit
//------------------------------------------------------------//

double TwoPhaseFlow::stableTimeStep(double cfl) const
{
    const double pi = acos(-1.0);

    double h = min(dx, dy);

    double dt = numeric_limits<double>::max();

    // Convective
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

    if(rate > 0.0)
    {
        dt = min(dt, 1.0/rate);
    }

    // Viscous
    double nuMax = max(params.mu1/params.rho1, params.mu2/params.rho2);

    if(nuMax > 0.0)
    {
        dt = min(dt, 0.25*h*h/nuMax);
    }

    // Capillary
    if(params.sigma > 0.0)
    {
        dt = min(dt, sqrt((params.rho1 + params.rho2)*h*h*h/(4.0*pi*params.sigma)));
    }

    // Gravity
    double g = sqrt(params.gx*params.gx + params.gy*params.gy);

    if(g > 0.0)
    {
        dt = min(dt, sqrt(h/g));
    }

    return cfl*dt;
}

//------------------------------------------------------------//
// Face Values and Ghost Values
//------------------------------------------------------------//

double TwoPhaseFlow::invRhoU
(
    int i,
    int J
) const
{
    return 2.0/(rho[pIdx(i-1, J)] + rho[pIdx(i, J)]);
}

double TwoPhaseFlow::invRhoV
(
    int I,
    int j
) const
{
    return 2.0/(rho[pIdx(I, j-1)] + rho[pIdx(I, j)]);
}

double TwoPhaseFlow::uAt
(
    const vector<double>& f,
    int i,
    int J
) const
{
    if(J < 0)
    {
        double w = (params.walls.bottom == WallType::NoSlip) ? -1.0 : 1.0;

        return w*f[uIdx(i, 0)];
    }

    if(J >= Ny)
    {
        double w = (params.walls.top == WallType::NoSlip) ? -1.0 : 1.0;

        return w*f[uIdx(i, Ny-1)];
    }

    return f[uIdx(i, J)];
}

double TwoPhaseFlow::vAt
(
    const vector<double>& f,
    int I,
    int j
) const
{
    if(I < 0)
    {
        double w = (params.walls.left == WallType::NoSlip) ? -1.0 : 1.0;

        return w*f[vIdx(0, j)];
    }

    if(I >= Nx)
    {
        double w = (params.walls.right == WallType::NoSlip) ? -1.0 : 1.0;

        return w*f[vIdx(Nx-1, j)];
    }

    return f[vIdx(I, j)];
}

//------------------------------------------------------------//
// Shear Stress at a Cell Corner
//------------------------------------------------------------//

double TwoPhaseFlow::shearStress
(
    const vector<double>& uf,
    const vector<double>& vf,
    int inode,
    int jnode
) const
{
    int iL = min(max(inode-1, 0), Nx-1);
    int iR = min(max(inode, 0), Nx-1);
    int jB = min(max(jnode-1, 0), Ny-1);
    int jT = min(max(jnode, 0), Ny-1);

    double muNode = 0.25*
    (
        mu[pIdx(iL, jB)] + mu[pIdx(iR, jB)]
      + mu[pIdx(iL, jT)] + mu[pIdx(iR, jT)]
    );

    double dudy = (uAt(uf, inode, jnode) - uAt(uf, inode, jnode-1))/dy;

    double dvdx = (vAt(vf, inode, jnode) - vAt(vf, inode-1, jnode))/dx;

    return muNode*(dudy + dvdx);
}

//------------------------------------------------------------//
// Predictor
//------------------------------------------------------------//

void TwoPhaseFlow::predictVelocity
(
    double dt,
    vector<double>& us,
    vector<double>& vs
) const
{
    // Surface tension force per unit volume on the faces
    vector<double> fu;
    vector<double> fv;

    level.surfaceForce(params.sigma, fu, fv);

    us = u;
    vs = v;

    // x-momentum on the interior u faces
    for(int J = 0; J < Ny; J++)
    {
        for(int i = 1; i < Nx; i++)
        {
            double uc = u[uIdx(i, J)];

            double vc = 0.25*
            (
                v[vIdx(i-1, J)] + v[vIdx(i, J)]
              + v[vIdx(i-1, J+1)] + v[vIdx(i, J+1)]
            );

            double uW = u[uIdx(i-1, J)];
            double uE = u[uIdx(i+1, J)];
            double uS = uAt(u, i, J-1);
            double uN = uAt(u, i, J+1);

            double dudx;
            double dudy;

            if(params.convection == TPConvection::Upwind)
            {
                dudx = (uc > 0.0) ? (uc - uW)/dx : (uE - uc)/dx;
                dudy = (vc > 0.0) ? (uc - uS)/dy : (uN - uc)/dy;
            }
            else
            {
                dudx = (uE - uW)/(2.0*dx);
                dudy = (uN - uS)/(2.0*dy);
            }

            double conv = uc*dudx + vc*dudy;

            double muR = mu[pIdx(i, J)];
            double muL = mu[pIdx(i-1, J)];

            double visc =
                (2.0*muR*(uE - uc) - 2.0*muL*(uc - uW))/(dx*dx)
              + (shearStress(u, v, i, J+1) - shearStress(u, v, i, J))/dy;

            double invRho = invRhoU(i, J);

            us[uIdx(i, J)] = uc + dt*
            (
               -conv
              + invRho*visc
              + params.gx
              + invRho*fu[uIdx(i, J)]
            );
        }
    }

    // y-momentum on the interior v faces
    for(int j = 1; j < Ny; j++)
    {
        for(int I = 0; I < Nx; I++)
        {
            double vc = v[vIdx(I, j)];

            double uc = 0.25*
            (
                u[uIdx(I, j-1)] + u[uIdx(I+1, j-1)]
              + u[uIdx(I, j)] + u[uIdx(I+1, j)]
            );

            double vW = vAt(v, I-1, j);
            double vE = vAt(v, I+1, j);
            double vS = v[vIdx(I, j-1)];
            double vN = v[vIdx(I, j+1)];

            double dvdx;
            double dvdy;

            if(params.convection == TPConvection::Upwind)
            {
                dvdx = (uc > 0.0) ? (vc - vW)/dx : (vE - vc)/dx;
                dvdy = (vc > 0.0) ? (vc - vS)/dy : (vN - vc)/dy;
            }
            else
            {
                dvdx = (vE - vW)/(2.0*dx);
                dvdy = (vN - vS)/(2.0*dy);
            }

            double conv = uc*dvdx + vc*dvdy;

            double muT = mu[pIdx(I, j)];
            double muB = mu[pIdx(I, j-1)];

            double visc =
                (shearStress(u, v, I+1, j) - shearStress(u, v, I, j))/dx
              + (2.0*muT*(vN - vc) - 2.0*muB*(vc - vS))/(dy*dy);

            double invRho = invRhoV(I, j);

            vs[vIdx(I, j)] = vc + dt*
            (
               -conv
              + invRho*visc
              + params.gy
              + invRho*fv[vIdx(I, j)]
            );
        }
    }
}

//------------------------------------------------------------//
// Pressure System
//------------------------------------------------------------//

void TwoPhaseFlow::assemblePressureSystem
(
    const vector<double>& us,
    const vector<double>& vs,
    double dt
)
{
    int n = Nx*Ny;

    for(int k = 0; k < n; k++)
    {
        fill(A[k].begin(), A[k].end(), 0.0);
    }

    b.assign(n, 0.0);

    for(int J = 0; J < Ny; J++)
    {
        for(int I = 0; I < Nx; I++)
        {
            int k = pIdx(I, J);

            // Face coefficients (1/rho)/h^2; a wall face carries
            // no flux (Neumann)
            double cE = (I < Nx-1) ? invRhoU(I+1, J)/(dx*dx) : 0.0;
            double cW = (I > 0) ? invRhoU(I, J)/(dx*dx) : 0.0;
            double cN = (J < Ny-1) ? invRhoV(I, J+1)/(dy*dy) : 0.0;
            double cS = (J > 0) ? invRhoV(I, J)/(dy*dy) : 0.0;

            A[k][k] = cE + cW + cN + cS;

            if(I < Nx-1)
            {
                A[k][pIdx(I+1, J)] = -cE;
            }

            if(I > 0)
            {
                A[k][pIdx(I-1, J)] = -cW;
            }

            if(J < Ny-1)
            {
                A[k][pIdx(I, J+1)] = -cN;
            }

            if(J > 0)
            {
                A[k][pIdx(I, J-1)] = -cS;
            }

            double div =
                (us[uIdx(I+1, J)] - us[uIdx(I, J)])/dx
              + (vs[vIdx(I, J+1)] - vs[vIdx(I, J)])/dy;

            b[k] = -div/dt;
        }
    }

    // Pin p(refCell) = 0: identity row and zero column, which
    // keeps the matrix symmetric and makes it positive definite
    for(int k = 0; k < n; k++)
    {
        A[k][refCell] = 0.0;
    }

    fill(A[refCell].begin(), A[refCell].end(), 0.0);

    A[refCell][refCell] = 1.0;

    b[refCell] = 0.0;
}

//------------------------------------------------------------//
// Correction
//------------------------------------------------------------//

void TwoPhaseFlow::correctVelocity
(
    double dt,
    const vector<double>& us,
    const vector<double>& vs
)
{
    u = us;
    v = vs;

    for(int J = 0; J < Ny; J++)
    {
        for(int i = 1; i < Nx; i++)
        {
            double dpdx = (p[pIdx(i, J)] - p[pIdx(i-1, J)])/dx;

            u[uIdx(i, J)] = us[uIdx(i, J)] - dt*invRhoU(i, J)*dpdx;
        }
    }

    for(int j = 1; j < Ny; j++)
    {
        for(int I = 0; I < Nx; I++)
        {
            double dpdy = (p[pIdx(I, j)] - p[pIdx(I, j-1)])/dy;

            v[vIdx(I, j)] = vs[vIdx(I, j)] - dt*invRhoV(I, j)*dpdy;
        }
    }
}

//------------------------------------------------------------//
// Time Step
//------------------------------------------------------------//

void TwoPhaseFlow::step
(
    double dt,
    SolverSelector& solver,
    TwoPhaseStepInfo& info
)
{
    if(dt <= 0.0)
    {
        throw invalid_argument("TwoPhaseFlow::step: dt must be positive.");
    }

    // Properties from the current interface
    computeProperties();

    // Predictor, pressure solve, correction
    vector<double> us;
    vector<double> vs;

    predictVelocity(dt, us, vs);

    assemblePressureSystem(us, vs, dt);

    KrylovResult r = solver.solve(A, b, p);

    if((int)r.x.size() != Nx*Ny)
    {
        throw runtime_error("TwoPhaseFlow::step: pressure solve returned a vector of the wrong size.");
    }

    p = r.x;

    correctVelocity(dt, us, vs);

    // Interface: advected with the new, divergence-free velocity
    // and then reinitialised
    level.advect(u, v, dt);

    stepCount++;

    if(params.reinitEvery > 0 && stepCount%params.reinitEvery == 0)
    {
        level.reinitialise(params.reinitSteps);
    }

    time += dt;

    info.dt = dt;
    info.status = r.status;
    info.iterations = r.iterations;
    info.matVecs = r.matVecs;
    info.precondApplies = r.precondApplies;
    info.trueRelResidual = r.trueRelResidual;
    info.maxDivergence = maxDivergence();
}

//------------------------------------------------------------//
// Diagnostics
//------------------------------------------------------------//

double TwoPhaseFlow::maxDivergence() const
{
    double worst = 0.0;

    for(int J = 0; J < Ny; J++)
    {
        for(int I = 0; I < Nx; I++)
        {
            double div =
                (u[uIdx(I+1, J)] - u[uIdx(I, J)])/dx
              + (v[vIdx(I, J+1)] - v[vIdx(I, J)])/dy;

            worst = max(worst, fabs(div));
        }
    }

    return worst;
}