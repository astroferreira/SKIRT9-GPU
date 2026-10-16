/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#include "ToddlersSEDFamily.hpp"
#include "Constants.hpp"
#include "GpuAcceleration.hpp"

////////////////////////////////////////////////////////////////////

ToddlersSEDFamily::ToddlersSEDFamily(SimulationItem* parent, SedMode sedMode, StellarTemplate stellarTemplate,
                                     bool includeDust, Resolution resolution, SFRPeriod sfrPeriod)
{
    parent->addChild(this);
    _sedMode = sedMode;
    _stellarTemplate = stellarTemplate;
    _includeDust = includeDust;
    _resolution = resolution;
    _sfrPeriod = sfrPeriod;
    setup();
}

////////////////////////////////////////////////////////////////////

void ToddlersSEDFamily::setupSelfBefore()
{
    SEDFamily::setupSelfBefore();

    // --- construct the name of the resource file corresponding to the property settings ---

    // base name
    string name = "ToddlersSEDFamily";

    // SED mode
    name += _sedMode == SedMode::Cloud ? "_Cloud" : "_SFRNormalized";

    // stellar population
    switch (_stellarTemplate)
    {
        case StellarTemplate::SB99Kroupa100Sin: name += "_SB99_kroupa100_sin"; break;
        case StellarTemplate::BPASSChab100Bin: name += "_BPASS_chab100_bin"; break;
        case StellarTemplate::BPASSChab300Bin: name += "_BPASS_chab300_bin"; break;
    }

    // dust option
    name += _includeDust ? "_Dust" : "_noDust";

    // resolution
    name += _resolution == Resolution::Low ? "_lr" : "_hr";

    // integration period (only for SFRNormalized mode)
    if (_sedMode == SedMode::SFRNormalized)
    {
        name += _sfrPeriod == SFRPeriod::Period30Myr ? "_30Myr" : "_10Myr";
    }

    // --- open the resource file ---

    if (_sedMode == SedMode::Cloud)
    {
        _cloudTable.open(this, name, "lambda(m),time(Myr),Z(1),SFE(1),n_cl(1/cm3),M_cl(Msun)", "Llambda(W/m)", false);
    }
    else  // SFRNormalized
    {
        _sfrNormalizedTable.open(this, name, "lambda(m),Z(1),SFE(1),n_cl(1/cm3)", "Llambda(W/m)", false);
    }
}

////////////////////////////////////////////////////////////////////

vector<SnapshotParameter> ToddlersSEDFamily::parameterInfo() const
{
    if (_sedMode == SedMode::Cloud)
    {
        return {
            SnapshotParameter::age(),
            SnapshotParameter::metallicity(),
            SnapshotParameter::custom("star formation efficiency"),
            SnapshotParameter::custom("cloud number density", "numbervolumedensity", "1/cm3"),
            SnapshotParameter::custom("cloud mass", "mass", "Msun"),
            SnapshotParameter::custom("scaling factor"),
        };
    }
    else  // SFRNormalized
    {
        return {
            SnapshotParameter::metallicity(),
            SnapshotParameter::custom("star formation efficiency"),
            SnapshotParameter::custom("cloud number density", "numbervolumedensity", "1/cm3"),
            SnapshotParameter::custom("star formation rate", "massrate", "Msun/yr"),
        };
    }
}

////////////////////////////////////////////////////////////////////

Range ToddlersSEDFamily::intrinsicWavelengthRange() const
{
    if (_sedMode == SedMode::Cloud)
    {
        return _cloudTable.axisRange<0>();
    }
    else  // SFRNormalized
    {
        return _sfrNormalizedTable.axisRange<0>();
    }
}

////////////////////////////////////////////////////////////////////

double ToddlersSEDFamily::specificLuminosity(double wavelength, const Array& parameters) const
{
    if (_sedMode == SedMode::Cloud)
    {
        // Extract parameters for Cloud mode
        double age = parameters[0] / (1e6 * Constants::year());  // Convert from s to Myr
        double Z = parameters[1];                                // Metallicity
        double SFE = parameters[2];                              // Star formation efficiency
        double n_cl = parameters[3] / 1e6;                       // Convert from 1/m³ to 1/cm³
        double M_cl = parameters[4] / Constants::Msun();         // Convert from kg to Msun
        double scaling = parameters[5];                          // Optional scaling factor

        // Use 6D table
        return scaling * _cloudTable(wavelength, age, Z, SFE, n_cl, M_cl);
    }
    else  // SFRNormalized
    {
        // extract parameters for SFRNormalized mode
        double Z = parameters[0];
        double SFE = parameters[1];
        double n_cl = parameters[2] / 1e6;                                   // Convert from 1/m³ to 1/cm³
        double sfr = parameters[3] / Constants::Msun() * Constants::year();  // Convert from Msun/yr to kg/s

        // use 4D table
        return sfr * _sfrNormalizedTable(wavelength, Z, SFE, n_cl);
    }
}

////////////////////////////////////////////////////////////////////

double ToddlersSEDFamily::cdf(Array& lambdav, Array& pv, Array& Pv, const Range& wavelengthRange,
                              const Array& parameters) const
{
    if (_sedMode == SedMode::Cloud)
    {
        // Extract parameters for Cloud mode
        double age = parameters[0] / (1e6 * Constants::year());  // Convert from s to Myr
        double Z = parameters[1];                                // Metallicity
        double SFE = parameters[2];                              // Star formation efficiency
        double n_cl = parameters[3] / 1e6;                       // Convert from 1/m³ to 1/cm³
        double M_cl = parameters[4] / Constants::Msun();         // Convert from kg to Msun
        double scaling = parameters[5];                          // Optional scaling factor

        // Use 6D table
        return scaling * _cloudTable.cdf(lambdav, pv, Pv, wavelengthRange, age, Z, SFE, n_cl, M_cl);
    }
    else  // SFRNormalized
    {
        // Extract parameters for SFRNormalized mode
        double Z = parameters[0];
        double SFE = parameters[1];
        double n_cl = parameters[2] / 1e6;                                   // Convert from 1/m³ to 1/cm³
        double sfr = parameters[3] / Constants::Msun() * Constants::year();  // Convert from Msun/yr to kg/s

        // Use 4D table
        return sfr * _sfrNormalizedTable.cdf(lambdav, pv, Pv, wavelengthRange, Z, SFE, n_cl);
    }
}

////////////////////////////////////////////////////////////////////

bool ToddlersSEDFamily::cdfBatch(vector<double>& luminosities, const Range& wavelengthRange,
                               const vector<double>& flattenedParameters, size_t numEntities) const
{
    if (_sedMode != SedMode::SFRNormalized || !numEntities
        || flattenedParameters.size() / 4 != numEntities || flattenedParameters.size() % 4) return false;

    vector<double> parameters(3 * numEntities), scales(numEntities);
    for (size_t m = 0; m != numEntities; ++m)
    {
        parameters[3 * m] = flattenedParameters[4 * m];
        parameters[3 * m + 1] = flattenedParameters[4 * m + 1];
        parameters[3 * m + 2] = flattenedParameters[4 * m + 2] / 1e6;
        scales[m] = flattenedParameters[4 * m + 3] / Constants::Msun() * Constants::year();
    }
    const auto& table = _sfrNormalizedTable;
    vector<const double*> axes{table.axisData<0>(), table.axisData<1>(), table.axisData<2>(), table.axisData<3>()};
    vector<size_t> sizes{table.axisSize<0>(), table.axisSize<1>(), table.axisSize<2>(), table.axisSize<3>()};
    vector<bool> logarithmic{table.axisIsLog<0>(), table.axisIsLog<1>(), table.axisIsLog<2>(), table.axisIsLog<3>()};
    return GpuAcceleration::storedTableCdf(luminosities, 4, axes, sizes, logarithmic, table.quantityDataRaw(),
                                          table.quantityStep(), table.quantityIsLog(), table.clampsFirstAxis(),
                                          wavelengthRange.min(), wavelengthRange.max(), parameters, scales, numEntities);
}

////////////////////////////////////////////////////////////////////

bool ToddlersSEDFamily::gpuSpectrumTable(GpuSourceSpectrumTable& table,
                                         const vector<double>& flattenedParameters, size_t numEntities) const
{
    table = GpuSourceSpectrumTable();
    if (_sedMode != SedMode::SFRNormalized || !numEntities
        || flattenedParameters.size() / 4 != numEntities || flattenedParameters.size() % 4)
        return false;

    const auto& sourceTable = _sfrNormalizedTable;
    table.axes = {sourceTable.axisData<0>(), sourceTable.axisData<1>(), sourceTable.axisData<2>(),
                  sourceTable.axisData<3>()};
    table.axisSizes = {sourceTable.axisSize<0>(), sourceTable.axisSize<1>(), sourceTable.axisSize<2>(),
                       sourceTable.axisSize<3>()};
    table.axisLog = {sourceTable.axisIsLog<0>(), sourceTable.axisIsLog<1>(), sourceTable.axisIsLog<2>(),
                     sourceTable.axisIsLog<3>()};
    table.values = sourceTable.quantityDataRaw();
    table.quantityStride = sourceTable.quantityStep();
    table.quantityLog = sourceTable.quantityIsLog();
    table.clampWavelength = sourceTable.clampsFirstAxis();
    table.parameters.resize(3 * numEntities);
    for (size_t m = 0; m != numEntities; ++m)
    {
        table.parameters[3 * m] = flattenedParameters[4 * m];
        table.parameters[3 * m + 1] = flattenedParameters[4 * m + 1];
        table.parameters[3 * m + 2] = flattenedParameters[4 * m + 2] / 1e6;
    }
    return true;
}

////////////////////////////////////////////////////////////////////
