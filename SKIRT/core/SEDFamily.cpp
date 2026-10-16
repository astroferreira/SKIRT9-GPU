/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       Copyright 2026 Astronomical Observatory, Ghent University ////
///////////////////////////////////////////////////////////////// */

#include "SEDFamily.hpp"

////////////////////////////////////////////////////////////////////

bool SEDFamily::gpuSpectrumTable(GpuSourceSpectrumTable& table, const vector<double>& flattenedParameters,
                                 size_t numEntities) const
{
    table = GpuSourceSpectrumTable();
    (void)flattenedParameters;
    (void)numEntities;
    return false;
}

////////////////////////////////////////////////////////////////////
