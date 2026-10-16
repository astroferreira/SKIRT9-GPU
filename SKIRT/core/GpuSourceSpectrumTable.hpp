/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       Copyright 2026 Astronomical Observatory, Ghent University ////
///////////////////////////////////////////////////////////////// */

#ifndef GPUSOURCESPECTRUMTABLE_HPP
#define GPUSOURCESPECTRUMTABLE_HPP

#include "Basics.hpp"

////////////////////////////////////////////////////////////////////

/** Describes a stored SED table and the non-wavelength interpolation parameters for a batch of
    source entities. The axis and quantity pointers remain owned by the corresponding SEDFamily
    instance and stay valid throughout the simulation. The parameters vector contains one
    consecutive parameter tuple per entity and excludes the wavelength coordinate and any overall
    luminosity scale. */
struct GpuSourceSpectrumTable
{
    vector<const double*> axes;
    vector<size_t> axisSizes;
    vector<bool> axisLog;
    const double* values{nullptr};
    size_t quantityStride{0};
    bool quantityLog{false};
    bool clampWavelength{false};
    vector<double> parameters;
};

////////////////////////////////////////////////////////////////////

#endif
