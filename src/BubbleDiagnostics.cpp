//------------------------------------------------------------//
// File: BubbleDiagnostics.cpp
//
// Description:
//
// Measurements on a rising bubble from a TwoPhaseFlow: area,
// centre of mass, mean rise velocity, interface length and
// circularity (the quantities of the Hysing et al. benchmark),
// the spurious-current speed and the pressure jump of a static
// drop, plus a history that keeps them over time.
//
// Area, centre of mass and rise velocity are weighted integrals
// over the cells with the bubble indicator 1 - H(phi). The
// interface length is the length of the phi = 0 contour found by
// marching squares on the cell-centre grid with linear
// interpolation, which is more accurate than integrating the
// smoothed delta function.
//
//------------------------------------------------------------//

#include "BubbleDiagnostics.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

using namespace std;

//------------------------------------------------------------//
// Interface Length (marching squares)
//------------------------------------------------------------//

namespace
{
    double distance
    (
        double x1,
        double y1,
        double x2,
        double y2
    )
    {
        return sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
    }

    // Length of the phi = 0 contour inside the square whose
    // corners are the cell centres (I,J), (I+1,J), (I,J+1),
    // (I+1,J+1)
    double squareContourLength
    (
        const LevelSet& level,
        int I,
        int J
    )
    {
        double f00 = level.phi[level.idx(I, J)];
        double f10 = level.phi[level.idx(I+1, J)];
        double f01 = level.phi[level.idx(I, J+1)];
        double f11 = level.phi[level.idx(I+1, J+1)];

        double x0 = level.xc(I);
        double y0 = level.yc(J);
        double x1 = x0 + level.dx;
        double y1 = y0 + level.dy;

        // Crossing points on the four edges, in the order
        // bottom, right, top, left
        bool cross[4];

        double px[4];
        double py[4];

        for(int e = 0; e < 4; e++)
        {
            cross[e] = false;
            px[e] = 0.0;
            py[e] = 0.0;
        }

        if((f00 < 0.0) != (f10 < 0.0))
        {
            double t = f00/(f00 - f10);

            cross[0] = true;
            px[0] = x0 + t*level.dx;
            py[0] = y0;
        }

        if((f10 < 0.0) != (f11 < 0.0))
        {
            double t = f10/(f10 - f11);

            cross[1] = true;
            px[1] = x1;
            py[1] = y0 + t*level.dy;
        }

        if((f01 < 0.0) != (f11 < 0.0))
        {
            double t = f01/(f01 - f11);

            cross[2] = true;
            px[2] = x0 + t*level.dx;
            py[2] = y1;
        }

        if((f00 < 0.0) != (f01 < 0.0))
        {
            double t = f00/(f00 - f01);

            cross[3] = true;
            px[3] = x0;
            py[3] = y0 + t*level.dy;
        }

        int count = 0;

        for(int e = 0; e < 4; e++)
        {
            if(cross[e])
            {
                count++;
            }
        }

        if(count == 2)
        {
            int first = -1;
            int second = -1;

            for(int e = 0; e < 4; e++)
            {
                if(cross[e])
                {
                    if(first < 0)
                    {
                        first = e;
                    }
                    else
                    {
                        second = e;
                    }
                }
            }

            return distance(px[first], py[first], px[second], py[second]);
        }

        if(count == 4)
        {
            // Saddle: the sign at the centre decides which two
            // corners are cut off
            double centre = 0.25*(f00 + f10 + f01 + f11);

            if((centre < 0.0) == (f00 < 0.0))
            {
                // corners (I+1,J) and (I,J+1) cut off
                return distance(px[0], py[0], px[1], py[1])
                     + distance(px[2], py[2], px[3], py[3]);
            }

            // corners (I,J) and (I+1,J+1) cut off
            return distance(px[3], py[3], px[0], py[0])
                 + distance(px[1], py[1], px[2], py[2]);
        }

        return 0.0;
    }

    double interfaceLength(const LevelSet& level)
    {
        double length = 0.0;

        for(int J = 0; J < level.Ny-1; J++)
        {
            for(int I = 0; I < level.Nx-1; I++)
            {
                length += squareContourLength(level, I, J);
            }
        }

        return length;
    }
}

//------------------------------------------------------------//
// Measurements
//------------------------------------------------------------//

double equivalentRadius(double area)
{
    const double pi = acos(-1.0);

    return sqrt(max(area, 0.0)/pi);
}

BubbleState measureBubble(const TwoPhaseFlow& flow)
{
    const double pi = acos(-1.0);

    const LevelSet& level = flow.level;

    int Nx = flow.Nx;
    int Ny = flow.Ny;

    double cellArea = flow.dx*flow.dy;

    double area = 0.0;
    double sumX = 0.0;
    double sumY = 0.0;
    double sumV = 0.0;

    double maxSpeed = 0.0;

    for(int J = 0; J < Ny; J++)
    {
        for(int I = 0; I < Nx; I++)
        {
            double w = (1.0 - level.smoothedHeaviside(level.phi[level.idx(I, J)]))*cellArea;

            double uc = 0.5*(flow.u[flow.uIdx(I, J)] + flow.u[flow.uIdx(I+1, J)]);
            double vc = 0.5*(flow.v[flow.vIdx(I, J)] + flow.v[flow.vIdx(I, J+1)]);

            area += w;
            sumX += w*level.xc(I);
            sumY += w*level.yc(J);
            sumV += w*vc;

            maxSpeed = max(maxSpeed, sqrt(uc*uc + vc*vc));
        }
    }

    BubbleState s;

    s.time = flow.time;
    s.area = area;
    s.maxSpeed = maxSpeed;
    s.perimeter = interfaceLength(level);

    if(area > 0.0)
    {
        s.xc = sumX/area;
        s.yc = sumY/area;
        s.vRise = sumV/area;

        double dEquivalent = sqrt(4.0*area/pi);

        if(s.perimeter > 0.0)
        {
            s.circularity = pi*dEquivalent/s.perimeter;
        }
    }

    return s;
}

double pressureJump
(
    const TwoPhaseFlow& flow,
    double marginFactor
)
{
    const LevelSet& level = flow.level;

    double margin = marginFactor*level.eps;

    double sumIn = 0.0;
    double sumOut = 0.0;

    int nIn = 0;
    int nOut = 0;

    for(int k = 0; k < flow.Nx*flow.Ny; k++)
    {
        if(level.phi[k] < -margin)
        {
            sumIn += flow.p[k];
            nIn++;
        }
        else if(level.phi[k] > margin)
        {
            sumOut += flow.p[k];
            nOut++;
        }
    }

    if(nIn == 0 || nOut == 0)
    {
        throw runtime_error
        (
            "pressureJump: no cell lies on one side of the interface "
            "beyond the margin (inside: "+to_string(nIn)+
            ", outside: "+to_string(nOut)+")."
        );
    }

    return sumIn/nIn - sumOut/nOut;
}

//------------------------------------------------------------//
// Bubble History
//------------------------------------------------------------//

void BubbleHistory::record(const TwoPhaseFlow& flow)
{
    records.push_back(measureBubble(flow));
}

const vector<BubbleState>& BubbleHistory::states() const
{
    return records;
}

int BubbleHistory::size() const
{
    return (int)records.size();
}

void BubbleHistory::clear()
{
    records.clear();
}

double BubbleHistory::maxRiseVelocity(double& timeAtMax) const
{
    if(records.empty())
    {
        throw runtime_error("BubbleHistory::maxRiseVelocity: nothing recorded.");
    }

    int best = 0;

    for(int k = 1; k < (int)records.size(); k++)
    {
        if(records[k].vRise > records[best].vRise)
        {
            best = k;
        }
    }

    timeAtMax = records[best].time;

    return records[best].vRise;
}

double BubbleHistory::minCircularity(double& timeAtMin) const
{
    if(records.empty())
    {
        throw runtime_error("BubbleHistory::minCircularity: nothing recorded.");
    }

    int best = 0;

    for(int k = 1; k < (int)records.size(); k++)
    {
        if(records[k].circularity < records[best].circularity)
        {
            best = k;
        }
    }

    timeAtMin = records[best].time;

    return records[best].circularity;
}

const BubbleState& BubbleHistory::last() const
{
    if(records.empty())
    {
        throw runtime_error("BubbleHistory::last: nothing recorded.");
    }

    return records.back();
}

double BubbleHistory::relativeAreaChange(int k) const
{
    if(records.empty())
    {
        throw runtime_error("BubbleHistory::relativeAreaChange: nothing recorded.");
    }

    if(k < 0 || k >= (int)records.size())
    {
        throw out_of_range("BubbleHistory::relativeAreaChange: record index out of range.");
    }

    double area0 = records[0].area;

    if(area0 <= 0.0)
    {
        return 0.0;
    }

    return (records[k].area - area0)/area0;
}

double BubbleHistory::maxAbsAreaChange() const
{
    double worst = 0.0;

    for(int k = 0; k < (int)records.size(); k++)
    {
        worst = max(worst, fabs(relativeAreaChange(k)));
    }

    return worst;
}

//------------------------------------------------------------//
// Output
//------------------------------------------------------------//

void BubbleHistory::writeCSV(const string& filename) const
{
    ofstream out(filename.c_str());

    if(!out)
    {
        throw runtime_error("BubbleHistory::writeCSV: cannot open '"+filename+"'.");
    }

    out << "time,area,areaChange,xc,yc,vRise,perimeter,circularity,maxSpeed\n";

    out << setprecision(10);

    for(int k = 0; k < (int)records.size(); k++)
    {
        const BubbleState& s = records[k];

        out << s.time << ","
            << s.area << ","
            << relativeAreaChange(k) << ","
            << s.xc << ","
            << s.yc << ","
            << s.vRise << ","
            << s.perimeter << ","
            << s.circularity << ","
            << s.maxSpeed << "\n";
    }
}