/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#include "BruzualCharlotSEDFamily.hpp"
#include "Constants.hpp"
#include "GpuAcceleration.hpp"

////////////////////////////////////////////////////////////////////

BruzualCharlotSEDFamily::BruzualCharlotSEDFamily(SimulationItem* parent, IMF imf, Resolution resolution)
{
    parent->addChild(this);
    _imf = imf;
    _resolution = resolution;
    setup();
}

////////////////////////////////////////////////////////////////////

void BruzualCharlotSEDFamily::setupSelfBefore()
{
    SEDFamily::setupSelfBefore();

    string name = "BruzualCharlotSEDFamily_";
    name += _imf == IMF::Chabrier ? "Chabrier" : "Salpeter";
    name += "_";
    name += _resolution == Resolution::Low ? "lr" : "hr";

    _table.open(this, name, "lambda(m),Z(1),t(yr)", "Llambda(W/m)", false);
}

////////////////////////////////////////////////////////////////////

vector<SnapshotParameter> BruzualCharlotSEDFamily::parameterInfo() const
{
    return {SnapshotParameter::initialMass(), SnapshotParameter::metallicity(), SnapshotParameter::age()};
}

////////////////////////////////////////////////////////////////////

Range BruzualCharlotSEDFamily::intrinsicWavelengthRange() const
{
    return _table.axisRange<0>();
}

////////////////////////////////////////////////////////////////////

double BruzualCharlotSEDFamily::specificLuminosity(double wavelength, const Array& parameters) const
{
    double M = parameters[0] / Constants::Msun();
    double Z = parameters[1];
    double t = parameters[2] / Constants::year();

    return M * _table(wavelength, Z, t);
}

////////////////////////////////////////////////////////////////////

double BruzualCharlotSEDFamily::cdf(Array& lambdav, Array& pv, Array& Pv, const Range& wavelengthRange,
                                    const Array& parameters) const
{
    double M = parameters[0] / Constants::Msun();
    double Z = parameters[1];
    double t = parameters[2] / Constants::year();

    return M * _table.cdf(lambdav, pv, Pv, wavelengthRange, Z, t);
}

////////////////////////////////////////////////////////////////////

bool BruzualCharlotSEDFamily::cdfBatch(vector<double>& luminosities, const Range& wavelengthRange,
                                       const vector<double>& flattenedParameters, size_t numEntities) const
{
    if (flattenedParameters.size() != 3 * numEntities) return false;

    vector<double> parameterValues(2 * numEntities);
    vector<double> scaleValues(numEntities);
    for (size_t m = 0; m != numEntities; ++m)
    {
        scaleValues[m] = flattenedParameters[3 * m] / Constants::Msun();
        parameterValues[2 * m] = flattenedParameters[3 * m + 1];
        parameterValues[2 * m + 1] = flattenedParameters[3 * m + 2] / Constants::year();
    }

    vector<const double*> axisData{_table.axisData<0>(), _table.axisData<1>(), _table.axisData<2>()};
    vector<size_t> axisSizes{_table.axisSize<0>(), _table.axisSize<1>(), _table.axisSize<2>()};
    vector<bool> axisLog{_table.axisIsLog<0>(), _table.axisIsLog<1>(), _table.axisIsLog<2>()};
    return GpuAcceleration::storedTableCdf(luminosities, 3, axisData, axisSizes, axisLog, _table.quantityDataRaw(),
                                           _table.quantityStep(), _table.quantityIsLog(), _table.clampsFirstAxis(),
                                           wavelengthRange.min(), wavelengthRange.max(), parameterValues, scaleValues,
                                           numEntities);
}

////////////////////////////////////////////////////////////////////

bool BruzualCharlotSEDFamily::sampleWavelengthBatch(vector<double>& wavelengths,
                                                    vector<double>& specificLuminosities,
                                                    const Range& wavelengthRange,
                                                    const vector<double>& flattenedParameters,
                                                    const vector<double>& intrinsicRandoms,
                                                    const vector<double>& forcedWavelengths,
                                                    size_t numSamples) const
{
    if (flattenedParameters.size() != 3 * numSamples) return false;

    vector<double> parameterValues(2 * numSamples);
    for (size_t m = 0; m != numSamples; ++m)
    {
        parameterValues[2 * m] = flattenedParameters[3 * m + 1];
        parameterValues[2 * m + 1] = flattenedParameters[3 * m + 2] / Constants::year();
    }

    vector<const double*> axisData{_table.axisData<0>(), _table.axisData<1>(), _table.axisData<2>()};
    vector<size_t> axisSizes{_table.axisSize<0>(), _table.axisSize<1>(), _table.axisSize<2>()};
    vector<bool> axisLog{_table.axisIsLog<0>(), _table.axisIsLog<1>(), _table.axisIsLog<2>()};
    return GpuAcceleration::storedTableSampleWavelengths(wavelengths, specificLuminosities, 3, axisData, axisSizes,
                                                         axisLog, _table.quantityDataRaw(), _table.quantityStep(),
                                                         _table.quantityIsLog(), _table.clampsFirstAxis(),
                                                         wavelengthRange.min(), wavelengthRange.max(),
                                                         parameterValues, intrinsicRandoms, forcedWavelengths,
                                                         numSamples);
}

////////////////////////////////////////////////////////////////////
