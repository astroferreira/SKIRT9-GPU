/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#ifndef SEDFAMILY_HPP
#define SEDFAMILY_HPP

#include "Array.hpp"
#include "Range.hpp"
#include "SimulationItem.hpp"
#include "SnapshotParameter.hpp"

//////////////////////////////////////////////////////////////////////

/** SEDFamily is an abstract class for representing some family of SEDs. Each subclass implements
    an %SED family, where the exact form of the %SED depends on one or more parameters. This base
    class offers a generic interface for obtaining information on a particlar #SED from the family,
    given the appropriate number and type of parameter values. */
class SEDFamily : public SimulationItem
{
    ITEM_ABSTRACT(SEDFamily, SimulationItem, "an SED family")
    ITEM_END()

public:
    /** This function returns the number and type of parameters used by this particular %SED family
        as a list of SnapshotParameter objects. Each of these objects specifies unit information
        and a human-readable descripton for the parameter. */
    virtual vector<SnapshotParameter> parameterInfo() const = 0;

    /** This function returns the intrinsic wavelength range of the %SED family. Outside this
        range, all luminosities are zero. */
    virtual Range intrinsicWavelengthRange() const = 0;

    /** This function returns the specific luminosity \f$L_\lambda\f$ (i.e. radiative power per
        unit of wavelength) for the %SED with the specified parameters at the specified wavelength,
        or zero if the wavelength is outside of the %SED's intrinsic wavelength range. The number
        and type of parameters must match the information returned by the parameterInfo() function;
        if not the behavior is undefined. */
    virtual double specificLuminosity(double wavelength, const Array& parameters) const = 0;

    /** This function constructs both the normalized probability density function (pdf) and the
        corresponding normalized cumulative distribution function (cdf) for the %SED with the
        specified parameters over the specified wavelength range. If the wavelength range or any of
        the parameter values are out of range of the internal grid, the corresponding luminosities
        are considered to be zero. The number and type of parameters must match the information
        returned by the parameterInfo() function; if not the behavior is undefined.

        The resulting wavelength grid is constructed into \em lambdav, the corresponding pdf into
        \em pv, and the corresponding cdf into \em Yv. The function returns the normalization
        factor, i.e. the value of Pv[n] before normalization. */
    virtual double cdf(Array& lambdav, Array& pv, Array& Pv, const Range& wavelengthRange,
                       const Array& parameters) const = 0;

    /** This function optionally calculates the normalization factor returned by cdf() for a batch
        of parameter vectors. The \em flattenedParameters vector contains \em numEntities
        consecutive parameter vectors, each with the number and type of entries returned by
        parameterInfo(). Implementations return true only if the batch calculation succeeded and
        stored one luminosity per entity in \em luminosities; otherwise callers must fall back to
        the scalar cdf() function. */
    virtual bool cdfBatch(vector<double>& luminosities, const Range& wavelengthRange,
                          const vector<double>& flattenedParameters, size_t numEntities) const
    {
        (void)luminosities;
        (void)wavelengthRange;
        (void)flattenedParameters;
        (void)numEntities;
        return false;
    }

    /** This function optionally samples wavelengths and evaluates the corresponding normalized
        specific luminosities for a batch of parameter vectors. The \em flattenedParameters vector
        contains \em numSamples consecutive parameter vectors. For each sample, if the optional
        \em forcedWavelengths vector has a positive value, that wavelength is used; otherwise the
        corresponding value in \em intrinsicRandoms is used to sample from the intrinsic SED CDF.
        Implementations return true only if the batch calculation succeeded. */
    virtual bool sampleWavelengthBatch(vector<double>& wavelengths, vector<double>& specificLuminosities,
                                       const Range& wavelengthRange, const vector<double>& flattenedParameters,
                                       const vector<double>& intrinsicRandoms,
                                       const vector<double>& forcedWavelengths, size_t numSamples) const
    {
        (void)wavelengths;
        (void)specificLuminosities;
        (void)wavelengthRange;
        (void)flattenedParameters;
        (void)intrinsicRandoms;
        (void)forcedWavelengths;
        (void)numSamples;
        return false;
    }
};

////////////////////////////////////////////////////////////////////

#endif
