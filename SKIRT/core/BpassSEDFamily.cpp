/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#include "BpassSEDFamily.hpp"
#include "Constants.hpp"
#include "GpuAcceleration.hpp"

////////////////////////////////////////////////////////////////////

BpassSEDFamily::BpassSEDFamily(SimulationItem* parent, IMF imf, Resolution resolution)
{
    parent->addChild(this);
    _imf = imf;
    _resolution = resolution;
    setup();
}

////////////////////////////////////////////////////////////////////

void BpassSEDFamily::setupSelfBefore()
{
    SEDFamily::setupSelfBefore();

    string name = "BpassSEDFamily";
    switch (_imf)
    {
        case IMF::Chabrier100: name += "_Chabrier100"; break;
        case IMF::Chabrier300: name += "_Chabrier300"; break;
        case IMF::Kroupa100: name += "_Kroupa100"; break;
        case IMF::Kroupa300: name += "_Kroupa300"; break;
    }
    if (_resolution == Resolution::Downsampled) name += "_downsampled";

    _table.open(this, name, "lambda(m),Z(1),t(yr)", "Llambda(W/m)", false);
}

////////////////////////////////////////////////////////////////////

vector<SnapshotParameter> BpassSEDFamily::parameterInfo() const
{
    return {SnapshotParameter::initialMass(), SnapshotParameter::metallicity(), SnapshotParameter::age()};
}

////////////////////////////////////////////////////////////////////

Range BpassSEDFamily::intrinsicWavelengthRange() const
{
    return _table.axisRange<0>();
}

////////////////////////////////////////////////////////////////////

double BpassSEDFamily::specificLuminosity(double wavelength, const Array& parameters) const
{
    double M = parameters[0] / Constants::Msun();
    double Z = parameters[1];
    double t = parameters[2] / Constants::year();

    return M * _table(wavelength, Z, t);
}

////////////////////////////////////////////////////////////////////

double BpassSEDFamily::cdf(Array& lambdav, Array& pv, Array& Pv, const Range& wavelengthRange,
                           const Array& parameters) const
{
    double M = parameters[0] / Constants::Msun();
    double Z = parameters[1];
    double t = parameters[2] / Constants::year();

    return M * _table.cdf(lambdav, pv, Pv, wavelengthRange, Z, t);
}

////////////////////////////////////////////////////////////////////

bool BpassSEDFamily::cdfBatch(vector<double>& luminosities, const Range& wavelengthRange,
                            const vector<double>& flattenedParameters, size_t numEntities) const
{
    if (!numEntities || flattenedParameters.size() / 3 != numEntities
        || flattenedParameters.size() % 3) return false;

    vector<double> parameters(2 * numEntities), scales(numEntities);
    for (size_t m = 0; m != numEntities; ++m)
    {
        scales[m] = flattenedParameters[3 * m] / Constants::Msun();
        parameters[2 * m] = flattenedParameters[3 * m + 1];
        parameters[2 * m + 1] = flattenedParameters[3 * m + 2] / Constants::year();
    }
    vector<const double*> axes{_table.axisData<0>(), _table.axisData<1>(), _table.axisData<2>()};
    vector<size_t> sizes{_table.axisSize<0>(), _table.axisSize<1>(), _table.axisSize<2>()};
    vector<bool> logarithmic{_table.axisIsLog<0>(), _table.axisIsLog<1>(), _table.axisIsLog<2>()};
    return GpuAcceleration::storedTableCdf(luminosities, 3, axes, sizes, logarithmic, _table.quantityDataRaw(),
                                          _table.quantityStep(), _table.quantityIsLog(), _table.clampsFirstAxis(),
                                          wavelengthRange.min(), wavelengthRange.max(), parameters, scales, numEntities);
}

////////////////////////////////////////////////////////////////////

bool BpassSEDFamily::gpuSpectrumTable(GpuSourceSpectrumTable& table,
                                      const vector<double>& flattenedParameters, size_t numEntities) const
{
    table = GpuSourceSpectrumTable();
    if (!numEntities || flattenedParameters.size() / 3 != numEntities || flattenedParameters.size() % 3)
        return false;

    table.axes = {_table.axisData<0>(), _table.axisData<1>(), _table.axisData<2>()};
    table.axisSizes = {_table.axisSize<0>(), _table.axisSize<1>(), _table.axisSize<2>()};
    table.axisLog = {_table.axisIsLog<0>(), _table.axisIsLog<1>(), _table.axisIsLog<2>()};
    table.values = _table.quantityDataRaw();
    table.quantityStride = _table.quantityStep();
    table.quantityLog = _table.quantityIsLog();
    table.clampWavelength = _table.clampsFirstAxis();
    table.parameters.resize(2 * numEntities);
    for (size_t m = 0; m != numEntities; ++m)
    {
        table.parameters[2 * m] = flattenedParameters[3 * m + 1];
        table.parameters[2 * m + 1] = flattenedParameters[3 * m + 2] / Constants::year();
    }
    return true;
}

////////////////////////////////////////////////////////////////////
