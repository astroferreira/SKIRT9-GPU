/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#ifndef GPUDETECTORINFO_HPP
#define GPUDETECTORINFO_HPP

#include "Basics.hpp"

////////////////////////////////////////////////////////////////////

/** Describes a total-only distant frame detector for a fused GPU observer/detector pass. */
struct GpuFrameDetectorInfo
{
    const void* accumulatorKey{nullptr};
    size_t numAccumulatorValues{0};
    double costheta{0.};
    double sintheta{0.};
    double cosphi{0.};
    double sinphi{0.};
    double cosomega{0.};
    double sinomega{0.};
    int numPixelsX{0};
    int numPixelsY{0};
    double xpmin{0.};
    double xpsiz{0.};
    double ypmin{0.};
    double ypsiz{0.};
    double redshift{0.};
    size_t numPixelsInFrame{0};
    vector<int> bandOffsetv;
    vector<double> bandWavelengthv;
    vector<double> bandTransmissionv;
    vector<double> bandWidthv;
};

////////////////////////////////////////////////////////////////////

/** Describes a total-only distant SED detector for a fused GPU observer/detector pass. */
struct GpuSedDetectorInfo
{
    const void* accumulatorKey{nullptr};
    size_t numAccumulatorValues{0};
    double costheta{0.};
    double sintheta{0.};
    double cosphi{0.};
    double sinphi{0.};
    double radiusSquared{0.};
    double redshift{0.};
    vector<double> leftBorderv;
    vector<double> rightBorderv;
};

////////////////////////////////////////////////////////////////////

#endif
