/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#include "MappingsSEDFamily.hpp"
#include "Constants.hpp"
#include "GpuAcceleration.hpp"

////////////////////////////////////////////////////////////////////

MappingsSEDFamily::MappingsSEDFamily(SimulationItem* parent)
{
    parent->addChild(this);
    setup();
}

////////////////////////////////////////////////////////////////////

void MappingsSEDFamily::setupSelfBefore()
{
    SEDFamily::setupSelfBefore();

    _table.open(this, "MappingsSEDFamily", "lambda(m),Z(1),logC(1),P(Pa),fPDR(1)", "Llambda(W/m)", false);
}

////////////////////////////////////////////////////////////////////

vector<SnapshotParameter> MappingsSEDFamily::parameterInfo() const
{
    return {
        SnapshotParameter::custom("star formation rate", "massrate", "Msun/yr"),
        SnapshotParameter::metallicity(),
        SnapshotParameter::custom("compactness"),
        SnapshotParameter::custom("pressure", "pressure", "Pa"),
        SnapshotParameter::custom("covering factor"),
    };
}

////////////////////////////////////////////////////////////////////

Range MappingsSEDFamily::intrinsicWavelengthRange() const
{
    return _table.axisRange<0>();
}

////////////////////////////////////////////////////////////////////

double MappingsSEDFamily::specificLuminosity(double wavelength, const Array& parameters) const
{
    double SFR = parameters[0] / Constants::Msun() * Constants::year();
    double Z = parameters[1];
    double logC = parameters[2];
    double p = parameters[3];
    double fPDR = parameters[4];

    return SFR * _table(wavelength, Z, logC, p, fPDR);
}

////////////////////////////////////////////////////////////////////

double MappingsSEDFamily::cdf(Array& lambdav, Array& pv, Array& Pv, const Range& wavelengthRange,
                              const Array& parameters) const
{
    double SFR = parameters[0] / Constants::Msun() * Constants::year();
    double Z = parameters[1];
    double logC = parameters[2];
    double p = parameters[3];
    double fPDR = parameters[4];

    return SFR * _table.cdf(lambdav, pv, Pv, wavelengthRange, Z, logC, p, fPDR);
}

////////////////////////////////////////////////////////////////////

bool MappingsSEDFamily::cdfBatch(vector<double>& luminosities, const Range& wavelengthRange,
                                 const vector<double>& flattenedParameters, size_t numEntities) const
{
    if (flattenedParameters.size() != 5 * numEntities) return false;

    vector<double> parameterValues(4 * numEntities);
    vector<double> scaleValues(numEntities);
    for (size_t m = 0; m != numEntities; ++m)
    {
        scaleValues[m] = flattenedParameters[5 * m] / Constants::Msun() * Constants::year();
        parameterValues[4 * m] = flattenedParameters[5 * m + 1];
        parameterValues[4 * m + 1] = flattenedParameters[5 * m + 2];
        parameterValues[4 * m + 2] = flattenedParameters[5 * m + 3];
        parameterValues[4 * m + 3] = flattenedParameters[5 * m + 4];
    }

    vector<const double*> axisData{_table.axisData<0>(), _table.axisData<1>(), _table.axisData<2>(),
                                   _table.axisData<3>(), _table.axisData<4>()};
    vector<size_t> axisSizes{_table.axisSize<0>(), _table.axisSize<1>(), _table.axisSize<2>(),
                             _table.axisSize<3>(), _table.axisSize<4>()};
    vector<bool> axisLog{_table.axisIsLog<0>(), _table.axisIsLog<1>(), _table.axisIsLog<2>(),
                         _table.axisIsLog<3>(), _table.axisIsLog<4>()};
    return GpuAcceleration::storedTableCdf(luminosities, 5, axisData, axisSizes, axisLog, _table.quantityDataRaw(),
                                           _table.quantityStep(), _table.quantityIsLog(), _table.clampsFirstAxis(),
                                           wavelengthRange.min(), wavelengthRange.max(), parameterValues, scaleValues,
                                           numEntities);
}

////////////////////////////////////////////////////////////////////

bool MappingsSEDFamily::sampleWavelengthBatch(vector<double>& wavelengths, vector<double>& specificLuminosities,
                                              const Range& wavelengthRange,
                                              const vector<double>& flattenedParameters,
                                              const vector<double>& intrinsicRandoms,
                                              const vector<double>& forcedWavelengths, size_t numSamples) const
{
    if (flattenedParameters.size() != 5 * numSamples) return false;

    vector<double> parameterValues(4 * numSamples);
    for (size_t m = 0; m != numSamples; ++m)
    {
        parameterValues[4 * m] = flattenedParameters[5 * m + 1];
        parameterValues[4 * m + 1] = flattenedParameters[5 * m + 2];
        parameterValues[4 * m + 2] = flattenedParameters[5 * m + 3];
        parameterValues[4 * m + 3] = flattenedParameters[5 * m + 4];
    }

    vector<const double*> axisData{_table.axisData<0>(), _table.axisData<1>(), _table.axisData<2>(),
                                   _table.axisData<3>(), _table.axisData<4>()};
    vector<size_t> axisSizes{_table.axisSize<0>(), _table.axisSize<1>(), _table.axisSize<2>(),
                             _table.axisSize<3>(), _table.axisSize<4>()};
    vector<bool> axisLog{_table.axisIsLog<0>(), _table.axisIsLog<1>(), _table.axisIsLog<2>(),
                         _table.axisIsLog<3>(), _table.axisIsLog<4>()};
    return GpuAcceleration::storedTableSampleWavelengths(wavelengths, specificLuminosities, 5, axisData, axisSizes,
                                                         axisLog, _table.quantityDataRaw(), _table.quantityStep(),
                                                         _table.quantityIsLog(), _table.clampsFirstAxis(),
                                                         wavelengthRange.min(), wavelengthRange.max(),
                                                         parameterValues, intrinsicRandoms, forcedWavelengths,
                                                         numSamples);
}

////////////////////////////////////////////////////////////////////
