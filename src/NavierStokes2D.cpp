//------------------------------------------------------------//
// File: NavierStokes2D.cpp
//
// Description:
//
// Shared building blocks for the pressure-velocity coupling
// algorithms (SIMPLE, SIMPLEC, PISO, PIMPLE): staggered-grid
// storage, momentum-equation assembly (upwind / hybrid
// convection, central diffusion, optional time term and
// implicit under-relaxation), Gauss-Seidel momentum sweeps,
// the pressure-correction equation, and a matrix-free
// (optionally symmetric Gauss-Seidel preconditioned) CG solver
// for that equation. The coupling algorithms themselves live
// in PVCoupling.cpp.
//
// Conventions
//
//   Control volume of u(i,J): from the centre of pressure cell
//   (i-1,J) to the centre of pressure cell (i,J). Control
//   volume of v(I,j): from the centre of cell (I,j-1) to the
//   centre of cell (I,j).
//
//   Pressure force on u(i,J): (p(i-1,J) - p(i,J)) dy
//   Pressure force on v(I,j): (p(I,j-1) - p(I,j)) dx
//
//   Wall next to a velocity node (tangential component): the
//   wall is half a cell away, so its diffusion conductance is
//   2 mu A / d and it enters aP and b only (no neighbour).
//
//   aP does not include the (F_e - F_w + F_n - F_s) term. It
//   is zero for a converged flow, and dropping it keeps the
//   diagonal dominant while the mass imbalance is still large.
//
//------------------------------------------------------------//

#include "NavierStokes2D.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace std;

//------------------------------------------------------------//
// File-Local Helpers
//------------------------------------------------------------//

// Neighbour coefficient for a face with outward mass flux Fout
// and diffusion conductance D
static double faceCoeff
(
    ConvectionScheme scheme,
    double Fout,
    double D
)
{
    if(scheme == ConvectionScheme::Hybrid)
    {
        return max(max(-Fout, D - 0.5*Fout), 0.0);
    }

    return D + max(-Fout, 0.0);
}

static double dotProd
(
    const vector<double>& a,
    const vector<double>& b
)
{
    double s = 0.0;

    for(size_t k = 0; k < a.size(); k++)
    {
        s += a[k]*b[k];
    }

    return s;
}

//------------------------------------------------------------//
// Constructor
//------------------------------------------------------------//

NavierStokes2D::NavierStokes2D
(
    int Nx_,
    int Ny_,
    double Lx_,
    double Ly_,
    double rho_,
    double nu_,
    const WallVelocities& walls_
)
:
    Nx(Nx_),
    Ny(Ny_),
    Lx(Lx_),
    Ly(Ly_),
    dx(Lx_/Nx_),
    dy(Ly_/Ny_),
    rho(rho_),
    nu(nu_),
    walls(walls_),
    refCell(0)
{
    if(Nx < 2 || Ny < 2)
    {
        throw invalid_argument("NavierStokes2D: Nx and Ny must be at least 2");
    }

    int nu_f = (Nx + 1)*Ny;
    int nv_f = Nx*(Ny + 1);
    int np   = Nx*Ny;

    u.assign(nu_f, 0.0);
    v.assign(nv_f, 0.0);
    p.assign(np, 0.0);

    uOld = u;
    vOld = v;

    cu.aE.assign(nu_f, 0.0);
    cu.aW.assign(nu_f, 0.0);
    cu.aN.assign(nu_f, 0.0);
    cu.aS.assign(nu_f, 0.0);
    cu.aP.assign(nu_f, 1.0);
    cu.aP0.assign(nu_f, 1.0);
    cu.b.assign(nu_f, 0.0);
    cu.b0.assign(nu_f, 0.0);
    cu.d.assign(nu_f, 0.0);

    cv.aE.assign(nv_f, 0.0);
    cv.aW.assign(nv_f, 0.0);
    cv.aN.assign(nv_f, 0.0);
    cv.aS.assign(nv_f, 0.0);
    cv.aP.assign(nv_f, 1.0);
    cv.aP0.assign(nv_f, 1.0);
    cv.b.assign(nv_f, 0.0);
    cv.b0.assign(nv_f, 0.0);
    cv.d.assign(nv_f, 0.0);

    cp.aE.assign(np, 0.0);
    cp.aW.assign(np, 0.0);
    cp.aN.assign(np, 0.0);
    cp.aS.assign(np, 0.0);
    cp.aP.assign(np, 1.0);
}

//------------------------------------------------------------//
// Time Level
//------------------------------------------------------------//

void NavierStokes2D::storeOldTimeLevel()
{
    uOld = u;
    vOld = v;
}

//------------------------------------------------------------//
// Neighbour Sums
//
// Faces on the boundary are never asked for. Rows next to a
// wall have a zero coefficient on the wall side, so the
// out-of-range neighbour is simply skipped.
//------------------------------------------------------------//

double NavierStokes2D::sumNbU
(
    const vector<double>& f,
    int i,
    int J
) const
{
    int k = uIdx(i, J);

    double s = cu.aE[k]*f[uIdx(i + 1, J)] + cu.aW[k]*f[uIdx(i - 1, J)];

    if(J < Ny - 1)
    {
        s += cu.aN[k]*f[uIdx(i, J + 1)];
    }

    if(J > 0)
    {
        s += cu.aS[k]*f[uIdx(i, J - 1)];
    }

    return s;
}

double NavierStokes2D::sumNbV
(
    const vector<double>& f,
    int I,
    int j
) const
{
    int k = vIdx(I, j);

    double s = cv.aN[k]*f[vIdx(I, j + 1)] + cv.aS[k]*f[vIdx(I, j - 1)];

    if(I < Nx - 1)
    {
        s += cv.aE[k]*f[vIdx(I + 1, j)];
    }

    if(I > 0)
    {
        s += cv.aW[k]*f[vIdx(I - 1, j)];
    }

    return s;
}

void NavierStokes2D::neighbourSumU
(
    const vector<double>& f,
    vector<double>& out
) const
{
    out.assign((Nx + 1)*Ny, 0.0);

    for(int J = 0; J < Ny; J++)
    {
        for(int i = 1; i < Nx; i++)
        {
            out[uIdx(i, J)] = sumNbU(f, i, J);
        }
    }
}

void NavierStokes2D::neighbourSumV
(
    const vector<double>& f,
    vector<double>& out
) const
{
    out.assign(Nx*(Ny + 1), 0.0);

    for(int j = 1; j < Ny; j++)
    {
        for(int I = 0; I < Nx; I++)
        {
            out[vIdx(I, j)] = sumNbV(f, I, j);
        }
    }
}

//------------------------------------------------------------//
// Momentum Assembly
//------------------------------------------------------------//

void NavierStokes2D::assembleMomentum
(
    ConvectionScheme scheme,
    double alphaU,
    double dt
)
{
    double mu = rho*nu;

    // Diffusion conductances of an interior face
    double Dx = mu*dy/dx;
    double Dy = mu*dx/dy;

    double timeCoeff = (dt > 0.0) ? rho*dx*dy/dt : 0.0;

    //--------------------------------------------------
    // u momentum: interior vertical faces i = 1..Nx-1
    //--------------------------------------------------

    for(int J = 0; J < Ny; J++)
    {
        for(int i = 1; i < Nx; i++)
        {
            int k = uIdx(i, J);

            double Fe = rho*0.5*(u[uIdx(i, J)] + u[uIdx(i + 1, J)])*dy;
            double Fw = rho*0.5*(u[uIdx(i - 1, J)] + u[uIdx(i, J)])*dy;
            double Fn = rho*0.5*(v[vIdx(i - 1, J + 1)] + v[vIdx(i, J + 1)])*dx;
            double Fs = rho*0.5*(v[vIdx(i - 1, J)] + v[vIdx(i, J)])*dx;

            double aE = faceCoeff(scheme, Fe, Dx);
            double aW = faceCoeff(scheme, -Fw, Dx);
            double aN = faceCoeff(scheme, Fn, Dy);
            double aS = faceCoeff(scheme, -Fs, Dy);

            double wallCoeff = 0.0;
            double wallSrc = 0.0;

            if(J == 0)
            {
                aS = 0.0;
                wallCoeff += 2.0*Dy;
                wallSrc += 2.0*Dy*walls.uBottom;
            }

            if(J == Ny - 1)
            {
                aN = 0.0;
                wallCoeff += 2.0*Dy;
                wallSrc += 2.0*Dy*walls.uTop;
            }

            cu.aE[k] = aE;
            cu.aW[k] = aW;
            cu.aN[k] = aN;
            cu.aS[k] = aS;

            double aP0 = aE + aW + aN + aS + wallCoeff + timeCoeff;
            double b0 = wallSrc + timeCoeff*uOld[k];

            cu.aP0[k] = aP0;
            cu.b0[k] = b0;
            cu.aP[k] = aP0/alphaU;
            cu.b[k] = b0 + (1.0 - alphaU)/alphaU*aP0*u[k];
        }
    }

    //--------------------------------------------------
    // v momentum: interior horizontal faces j = 1..Ny-1
    //--------------------------------------------------

    for(int j = 1; j < Ny; j++)
    {
        for(int I = 0; I < Nx; I++)
        {
            int k = vIdx(I, j);

            double Fe = rho*0.5*(u[uIdx(I + 1, j - 1)] + u[uIdx(I + 1, j)])*dy;
            double Fw = rho*0.5*(u[uIdx(I, j - 1)] + u[uIdx(I, j)])*dy;
            double Fn = rho*0.5*(v[vIdx(I, j)] + v[vIdx(I, j + 1)])*dx;
            double Fs = rho*0.5*(v[vIdx(I, j - 1)] + v[vIdx(I, j)])*dx;

            double aE = faceCoeff(scheme, Fe, Dx);
            double aW = faceCoeff(scheme, -Fw, Dx);
            double aN = faceCoeff(scheme, Fn, Dy);
            double aS = faceCoeff(scheme, -Fs, Dy);

            double wallCoeff = 0.0;
            double wallSrc = 0.0;

            if(I == 0)
            {
                aW = 0.0;
                wallCoeff += 2.0*Dx;
                wallSrc += 2.0*Dx*walls.vLeft;
            }

            if(I == Nx - 1)
            {
                aE = 0.0;
                wallCoeff += 2.0*Dx;
                wallSrc += 2.0*Dx*walls.vRight;
            }

            cv.aE[k] = aE;
            cv.aW[k] = aW;
            cv.aN[k] = aN;
            cv.aS[k] = aS;

            double aP0 = aE + aW + aN + aS + wallCoeff + timeCoeff;
            double b0 = wallSrc + timeCoeff*vOld[k];

            cv.aP0[k] = aP0;
            cv.b0[k] = b0;
            cv.aP[k] = aP0/alphaU;
            cv.b[k] = b0 + (1.0 - alphaU)/alphaU*aP0*v[k];
        }
    }
}

//------------------------------------------------------------//
// Momentum Sweeps and Residuals
//------------------------------------------------------------//

void NavierStokes2D::sweepMomentum(int sweeps)
{
    for(int s = 0; s < sweeps; s++)
    {
        for(int J = 0; J < Ny; J++)
        {
            for(int i = 1; i < Nx; i++)
            {
                int k = uIdx(i, J);

                double rhs = cu.b[k]
                           + (p[pIdx(i - 1, J)] - p[pIdx(i, J)])*dy
                           + sumNbU(u, i, J);

                u[k] = rhs/cu.aP[k];
            }
        }

        for(int j = 1; j < Ny; j++)
        {
            for(int I = 0; I < Nx; I++)
            {
                int k = vIdx(I, j);

                double rhs = cv.b[k]
                           + (p[pIdx(I, j - 1)] - p[pIdx(I, j)])*dx
                           + sumNbV(v, I, j);

                v[k] = rhs/cv.aP[k];
            }
        }
    }
}

double NavierStokes2D::residualU() const
{
    double sum = 0.0;

    for(int J = 0; J < Ny; J++)
    {
        for(int i = 1; i < Nx; i++)
        {
            int k = uIdx(i, J);

            double r = cu.b0[k]
                     + (p[pIdx(i - 1, J)] - p[pIdx(i, J)])*dy
                     + sumNbU(u, i, J)
                     - cu.aP0[k]*u[k];

            sum += fabs(r);
        }
    }

    return sum;
}

double NavierStokes2D::residualV() const
{
    double sum = 0.0;

    for(int j = 1; j < Ny; j++)
    {
        for(int I = 0; I < Nx; I++)
        {
            int k = vIdx(I, j);

            double r = cv.b0[k]
                     + (p[pIdx(I, j - 1)] - p[pIdx(I, j)])*dx
                     + sumNbV(v, I, j)
                     - cv.aP0[k]*v[k];

            sum += fabs(r);
        }
    }

    return sum;
}

//------------------------------------------------------------//
// Velocity-Correction Coefficients
//------------------------------------------------------------//

void NavierStokes2D::computeD(bool consistent)
{
    for(int J = 0; J < Ny; J++)
    {
        for(int i = 1; i < Nx; i++)
        {
            int k = uIdx(i, J);

            double den = cu.aP[k];

            if(consistent)
            {
                double sumNb = cu.aE[k] + cu.aW[k] + cu.aN[k] + cu.aS[k];

                double dc = cu.aP[k] - sumNb;

                if(dc > 1e-6*cu.aP[k])
                {
                    den = dc;
                }
            }

            cu.d[k] = dy/den;
        }
    }

    for(int j = 1; j < Ny; j++)
    {
        for(int I = 0; I < Nx; I++)
        {
            int k = vIdx(I, j);

            double den = cv.aP[k];

            if(consistent)
            {
                double sumNb = cv.aE[k] + cv.aW[k] + cv.aN[k] + cv.aS[k];

                double dc = cv.aP[k] - sumNb;

                if(dc > 1e-6*cv.aP[k])
                {
                    den = dc;
                }
            }

            cv.d[k] = dx/den;
        }
    }
}

//------------------------------------------------------------//
// Pressure-Correction Equation
//------------------------------------------------------------//

void NavierStokes2D::assemblePressureCorrection()
{
    for(int J = 0; J < Ny; J++)
    {
        for(int I = 0; I < Nx; I++)
        {
            int k = pIdx(I, J);

            // d is zero on boundary faces, so wall-side
            // coefficients vanish automatically
            double aE = cu.d[uIdx(I + 1, J)]*dy;
            double aW = cu.d[uIdx(I, J)]*dy;
            double aN = cv.d[vIdx(I, J + 1)]*dx;
            double aS = cv.d[vIdx(I, J)]*dx;

            cp.aE[k] = aE;
            cp.aW[k] = aW;
            cp.aN[k] = aN;
            cp.aS[k] = aS;
            cp.aP[k] = aE + aW + aN + aS;
        }
    }
}

double NavierStokes2D::massSource
(
    const vector<double>& uf,
    const vector<double>& vf,
    vector<double>& b
) const
{
    b.assign(Nx*Ny, 0.0);

    double sumAbs = 0.0;

    for(int J = 0; J < Ny; J++)
    {
        for(int I = 0; I < Nx; I++)
        {
            double val = (uf[uIdx(I, J)] - uf[uIdx(I + 1, J)])*dy
                       + (vf[vIdx(I, J)] - vf[vIdx(I, J + 1)])*dx;

            b[pIdx(I, J)] = val;

            sumAbs += fabs(val);
        }
    }

    b[refCell] = 0.0;

    return sumAbs;
}

// y = A x with the pinned cell treated as a removed unknown:
// row and column of refCell are replaced by the identity, so
// the operator is symmetric positive definite
void NavierStokes2D::applyPressureOperator
(
    const vector<double>& x,
    vector<double>& y
) const
{
    y.assign(Nx*Ny, 0.0);

    for(int J = 0; J < Ny; J++)
    {
        for(int I = 0; I < Nx; I++)
        {
            int k = pIdx(I, J);

            if(k == refCell)
            {
                y[k] = x[k];

                continue;
            }

            double s = cp.aP[k]*x[k];

            if(I < Nx - 1 && k + 1 != refCell)
            {
                s -= cp.aE[k]*x[k + 1];
            }

            if(I > 0 && k - 1 != refCell)
            {
                s -= cp.aW[k]*x[k - 1];
            }

            if(J < Ny - 1 && k + Nx != refCell)
            {
                s -= cp.aN[k]*x[k + Nx];
            }

            if(J > 0 && k - Nx != refCell)
            {
                s -= cp.aS[k]*x[k - Nx];
            }

            y[k] = s;
        }
    }
}

// z = M^-1 r with M = (D + L) D^-1 (D + U), the symmetric
// Gauss-Seidel preconditioner on the 5-point stencil
void NavierStokes2D::applyPreconditioner
(
    const vector<double>& r,
    vector<double>& z
) const
{
    int n = Nx*Ny;

    vector<double> y(n, 0.0);

    for(int J = 0; J < Ny; J++)
    {
        for(int I = 0; I < Nx; I++)
        {
            int k = pIdx(I, J);

            if(k == refCell)
            {
                continue;
            }

            double s = r[k];

            if(I > 0)
            {
                s += cp.aW[k]*y[k - 1];
            }

            if(J > 0)
            {
                s += cp.aS[k]*y[k - Nx];
            }

            y[k] = s/cp.aP[k];
        }
    }

    for(int k = 0; k < n; k++)
    {
        y[k] *= cp.aP[k];
    }

    z.assign(n, 0.0);

    for(int J = Ny - 1; J >= 0; J--)
    {
        for(int I = Nx - 1; I >= 0; I--)
        {
            int k = pIdx(I, J);

            if(k == refCell)
            {
                continue;
            }

            double s = y[k];

            if(I < Nx - 1)
            {
                s += cp.aE[k]*z[k + 1];
            }

            if(J < Ny - 1)
            {
                s += cp.aN[k]*z[k + Nx];
            }

            z[k] = s/cp.aP[k];
        }
    }
}

LinearSolveInfo NavierStokes2D::solvePressureCorrection
(
    const vector<double>& rhs,
    vector<double>& pc,
    double relTol,
    int maxIter,
    bool usePreconditioner
) const
{
    int n = Nx*Ny;

    LinearSolveInfo info;

    pc.assign(n, 0.0);

    // Zero initial guess, so the first residual is the rhs
    vector<double> r(rhs);

    r[refCell] = 0.0;

    double r0 = sqrt(dotProd(r, r));

    info.initialResidual = r0;

    if(r0 < 1e-300)
    {
        info.converged = true;

        return info;
    }

    vector<double> z(n, 0.0);
    vector<double> dir(n, 0.0);
    vector<double> Ad(n, 0.0);

    if(usePreconditioner)
    {
        applyPreconditioner(r, z);
    }
    else
    {
        z = r;
    }

    dir = z;

    double rz = dotProd(r, z);

    for(int it = 1; it <= maxIter; it++)
    {
        applyPressureOperator(dir, Ad);

        double alpha = rz/dotProd(dir, Ad);

        for(int k = 0; k < n; k++)
        {
            pc[k] += alpha*dir[k];
            r[k] -= alpha*Ad[k];
        }

        double rn = sqrt(dotProd(r, r));

        info.iterations = it;
        info.relResidual = rn/r0;

        if(rn <= relTol*r0)
        {
            info.converged = true;

            break;
        }

        if(usePreconditioner)
        {
            applyPreconditioner(r, z);
        }
        else
        {
            z = r;
        }

        double rzNew = dotProd(r, z);

        double beta = rzNew/rz;

        rz = rzNew;

        for(int k = 0; k < n; k++)
        {
            dir[k] = z[k] + beta*dir[k];
        }
    }

    return info;
}

//------------------------------------------------------------//
// Applying Corrections
//------------------------------------------------------------//

void NavierStokes2D::correctVelocities
(
    const vector<double>& pc,
    vector<double>& uf,
    vector<double>& vf
) const
{
    for(int J = 0; J < Ny; J++)
    {
        for(int i = 1; i < Nx; i++)
        {
            int k = uIdx(i, J);

            uf[k] += cu.d[k]*(pc[pIdx(i - 1, J)] - pc[pIdx(i, J)]);
        }
    }

    for(int j = 1; j < Ny; j++)
    {
        for(int I = 0; I < Nx; I++)
        {
            int k = vIdx(I, j);

            vf[k] += cv.d[k]*(pc[pIdx(I, j - 1)] - pc[pIdx(I, j)]);
        }
    }
}

void NavierStokes2D::updatePressure
(
    const vector<double>& pc,
    double alphaP
)
{
    for(size_t k = 0; k < p.size(); k++)
    {
        p[k] += alphaP*pc[k];
    }
}

double NavierStokes2D::massResidual() const
{
    vector<double> b;

    return massSource(u, v, b);
}

//------------------------------------------------------------//
// Sampling
//------------------------------------------------------------//

// u along column i at height y: nodes at y = (J + 0.5) dy, wall
// value at y = 0 and y = Ly
double NavierStokes2D::columnU
(
    int i,
    double y
) const
{
    // On the vertical walls u is the wall-normal velocity: zero
    if(i == 0 || i == Nx)
    {
        return 0.0;
    }

    if(y <= 0.5*dy)
    {
        double t = max(y, 0.0)/(0.5*dy);

        return (1.0 - t)*walls.uBottom + t*u[uIdx(i, 0)];
    }

    if(y >= Ly - 0.5*dy)
    {
        double t = (Ly - min(y, Ly))/(0.5*dy);

        return (1.0 - t)*walls.uTop + t*u[uIdx(i, Ny - 1)];
    }

    double s = y/dy - 0.5;

    int J = min(static_cast<int>(floor(s)), Ny - 2);

    double t = s - J;

    return (1.0 - t)*u[uIdx(i, J)] + t*u[uIdx(i, J + 1)];
}

// v along row j at position x: nodes at x = (I + 0.5) dx, wall
// value at x = 0 and x = Lx
double NavierStokes2D::rowV
(
    int j,
    double x
) const
{
    // On the horizontal walls v is the wall-normal velocity: zero
    if(j == 0 || j == Ny)
    {
        return 0.0;
    }

    if(x <= 0.5*dx)
    {
        double t = max(x, 0.0)/(0.5*dx);

        return (1.0 - t)*walls.vLeft + t*v[vIdx(0, j)];
    }

    if(x >= Lx - 0.5*dx)
    {
        double t = (Lx - min(x, Lx))/(0.5*dx);

        return (1.0 - t)*walls.vRight + t*v[vIdx(Nx - 1, j)];
    }

    double s = x/dx - 0.5;

    int I = min(static_cast<int>(floor(s)), Nx - 2);

    double t = s - I;

    return (1.0 - t)*v[vIdx(I, j)] + t*v[vIdx(I + 1, j)];
}

double NavierStokes2D::sampleU
(
    double x,
    double y
) const
{
    double fx = x/dx;

    int i = min(max(static_cast<int>(floor(fx)), 0), Nx - 1);

    double t = fx - i;

    return (1.0 - t)*columnU(i, y) + t*columnU(i + 1, y);
}

double NavierStokes2D::sampleV
(
    double x,
    double y
) const
{
    double fy = y/dy;

    int j = min(max(static_cast<int>(floor(fy)), 0), Ny - 1);

    double t = fy - j;

    return (1.0 - t)*rowV(j, x) + t*rowV(j + 1, x);
}