/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#include "GaussianLineSEDFamily.hpp"
#include "Constants.hpp"
#include "NR.hpp"

//////////////////////////////////////////////////////////////////////

namespace
{
    double unitGaussian(double x)
    {
        return (0.5 * M_SQRT1_2 * M_2_SQRTPI) * exp(-0.5 * x * x);
    }
}

////////////////////////////////////////////////////////////////////

vector<SnapshotParameter> GaussianLineSEDFamily::parameterInfo() const
{
    return {SnapshotParameter::custom("line luminosity", "bolluminosity", "W"),
            SnapshotParameter::custom("intrinsic dispersion", "velocity", "km/s")};
}

////////////////////////////////////////////////////////////////////

Range GaussianLineSEDFamily::intrinsicWavelengthRange() const
{
    return Range(_wavelength * 0.96, _wavelength * 1.04);
}

////////////////////////////////////////////////////////////////////

double GaussianLineSEDFamily::specificLuminosity(double wavelength, const Array& parameters) const
{
    double wavelengthDispersion = parameters[1] * _wavelength / Constants::c();
    return parameters[0] * unitGaussian((wavelength - _wavelength) / wavelengthDispersion) /
           wavelengthDispersion;
}

////////////////////////////////////////////////////////////////////

double GaussianLineSEDFamily::cdf(Array& lambdav, Array& pv, Array& Pv, const Range& wavelengthRange,
                                  const Array& parameters) const
{
    double L = parameters[0];
    double wavelengthDispersion = parameters[1] * _wavelength / Constants::c();

    // Restrict the table to the part of the Gaussian carrying all numerically relevant power.
    // This keeps the cost independent of a simulation's potentially broad wavelength range.
    Range range(_wavelength - 10. * wavelengthDispersion, _wavelength + 10. * wavelengthDispersion);
    range.intersect(wavelengthRange);
    if (range.empty())
    {
        lambdav.resize(0);
        pv.resize(0);
        Pv.resize(0);
        return 0.;
    }

    constexpr size_t numIntervals = 160;
    NR::buildLinearGrid(lambdav, range.min(), range.max(), numIntervals);
    pv.resize(numIntervals + 1);
    for (size_t i = 0; i <= numIntervals; ++i)
        pv[i] = L * unitGaussian((lambdav[i] - _wavelength) / wavelengthDispersion) /
                wavelengthDispersion;
    return NR::cdf2(false, lambdav, pv, Pv);
}

////////////////////////////////////////////////////////////////////
