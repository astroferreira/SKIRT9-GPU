#ifndef GPUSTOCHASTICDUSTDATA_HPP
#define GPUSTOCHASTICDUSTDATA_HPP

#include "Basics.hpp"

/** Immutable grain tables for batches of stochastic dust emissivities. Temperature and triangular
    transition tables are concatenated in grain-major, then grid-major order; each triangle is stored
    column-major (for column i, the rows f > i are contiguous). The wavelength table holds the emission
    wavelengths followed by the corresponding Planck prefactors 2hc^2/lambda^5. \c maxCount bounds the
    number of temperature points in any single probability calculation. */
struct GpuStochasticDustData
{
    unsigned long long cacheToken{0};
    int numInput{0}, numOutput{0}, numGrains{0}, numTypes{0}, maxCount{0};
    vector<int> gridSizev, grainTypev;
    vector<double> temperaturev, blackbodyv, heatingv, coolingv, planckAbsv;
    vector<short> heatingBinv;
    vector<double> rfCrossSectionv, emCrossSectionv, rfWidthv, cmbv, wavelengthv;
    vector<double> grainMassv, maxEnthalpyTempv;
    double planckFactor1{0.}, planckFactor2{0.};
    double maxTemperature{0.}, equilibriumWidth{0.}, mediumWidth{0.};
};

#endif
