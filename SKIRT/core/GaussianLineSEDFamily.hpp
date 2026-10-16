/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#ifndef GAUSSIANLINESEDFAMILY_HPP
#define GAUSSIANLINESEDFAMILY_HPP

#include "SEDFamily.hpp"

//////////////////////////////////////////////////////////////////////

/** GaussianLineSEDFamily represents a configurable isolated emission line. Each imported entity
    supplies the integrated line luminosity and the intrinsic velocity dispersion. Bulk and
    unresolved velocity shifts can be added independently through the velocity columns supported
    by an ImportedSource. */
class GaussianLineSEDFamily : public SEDFamily
{
    ITEM_CONCRETE(GaussianLineSEDFamily, SEDFamily, "a family of Gaussian spectra around a configurable line")

        PROPERTY_DOUBLE(wavelength, "the rest wavelength of the line")
        ATTRIBUTE_QUANTITY(wavelength, "wavelength")
        ATTRIBUTE_MIN_VALUE(wavelength, "]0")

    ITEM_END()

public:
    /** The imported parameters are integrated line luminosity and intrinsic velocity dispersion. */
    vector<SnapshotParameter> parameterInfo() const override;

    /** Returns a conservative intrinsic range of plus or minus four percent around the line. */
    Range intrinsicWavelengthRange() const override;

    /** Returns the Gaussian specific luminosity at the specified wavelength. */
    double specificLuminosity(double wavelength, const Array& parameters) const override;

    /** Builds the normalized PDF and CDF over the requested wavelength range. */
    double cdf(Array& lambdav, Array& pv, Array& Pv, const Range& wavelengthRange,
               const Array& parameters) const override;
};

////////////////////////////////////////////////////////////////////

#endif
