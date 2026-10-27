#ifndef BUBBLEDIAGNOSTICS_H
#define BUBBLEDIAGNOSTICS_H

#include <string>
#include <vector>

#include "TwoPhaseFlow.h"

//--------------------------------------------------
// Bubble State
//
// The quantities of the rising bubble benchmark, measured
// from a TwoPhaseFlow at one instant. The bubble is the
// region phi < 0; its indicator is 1 - H(phi) with the
// smoothed Heaviside of the level set.
//
//   area       : integral of (1 - H(phi)) over the domain
//   xc, yc     : centre of mass of the bubble
//                  yc = int y (1 - H) dA / int (1 - H) dA
//   vRise      : mean vertical velocity of the bubble
//                  vRise = int v (1 - H) dA / int (1 - H) dA
//                (v averaged from the two faces of a cell)
//   perimeter  : length of the phi = 0 contour, from linear
//                interpolation of phi between cell centres
//                (marching squares)
//   circularity: perimeter of the circle with the same area
//                divided by the perimeter of the bubble,
//                pi d_a / perimeter, d_a = sqrt(4 area/pi).
//                1 for a circle, smaller for a deformed bubble.
//   maxSpeed   : max over the cells of |velocity|, the
//                spurious current measure for a static drop
//
// Only the contour between the first and last cell centres
// is seen, so the bubble must stay away from the walls.
//--------------------------------------------------

struct BubbleState
{
    double time = 0.0;

    double area = 0.0;

    double xc = 0.0;

    double yc = 0.0;

    double vRise = 0.0;

    double perimeter = 0.0;

    double circularity = 0.0;

    double maxSpeed = 0.0;
};

//--------------------------------------------------
// Measurements
//
// measureBubble returns the state above. It never throws;
// with no bubble in the domain (area = 0) the centre of
// mass, rise velocity and circularity are 0.
//
// equivalentRadius is sqrt(area/pi).
//
// pressureJump is the static drop measure
//
//   mean p over the cells with phi < -margin
//   - mean p over the cells with phi > +margin
//
// with margin = marginFactor*eps, so that the smeared
// interface region is left out. For a static circular drop
// of radius R and no gravity it should be sigma/R.
// Throws std::runtime_error if there is no cell on one of
// the two sides.
//--------------------------------------------------

BubbleState measureBubble(const TwoPhaseFlow& flow);

double equivalentRadius(double area);

double pressureJump
(
    const TwoPhaseFlow& flow,
    double marginFactor = 2.0
);

//--------------------------------------------------
// Bubble History
//
// Collects a BubbleState at chosen times, for the
// comparison with the benchmark reference values:
//
//   max rise velocity and the time it occurs
//   minimum circularity and the time it occurs
//   final centre of mass
//   area (mass) change relative to the first record
//
// writeCSV writes one row per record. The relative area
// change is a column of the file.
//--------------------------------------------------

class BubbleHistory
{
public:

    //--------------------------------------------------
    // Recording
    //--------------------------------------------------

    void record(const TwoPhaseFlow& flow);

    const std::vector<BubbleState>& states() const;

    int size() const;

    void clear();

    //--------------------------------------------------
    // Reductions
    //
    // All throw std::runtime_error when nothing has been
    // recorded. The time of the extremum is returned through
    // the argument.
    //--------------------------------------------------

    double maxRiseVelocity(double& timeAtMax) const;

    double minCircularity(double& timeAtMin) const;

    const BubbleState& last() const;

    // (area_k - area_0)/area_0 for record k
    double relativeAreaChange(int k) const;

    // Largest |relativeAreaChange| over the history
    double maxAbsAreaChange() const;

    //--------------------------------------------------
    // Output
    //
    // Columns: time, area, areaChange, xc, yc, vRise,
    // perimeter, circularity, maxSpeed. Throws
    // std::runtime_error if the file cannot be opened.
    //--------------------------------------------------

    void writeCSV(const std::string& filename) const;

private:

    std::vector<BubbleState> records;
};

#endif