/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#include "SEDInstrument.hpp"
#include "FluxRecorder.hpp"
#include "GpuDetectorInfo.hpp"
#include "PhotonPacket.hpp"

////////////////////////////////////////////////////////////////////

void SEDInstrument::setupSelfBefore()
{
    ApertureInstrument::setupSelfBefore();

    // configure flux recorder
    instrumentFluxRecorder()->includeFluxDensityForDistant();
}

////////////////////////////////////////////////////////////////////

void SEDInstrument::detect(PhotonPacket* pp)
{
    if (isInsideAperture(pp)) instrumentFluxRecorder()->detect(pp, 0);
}

////////////////////////////////////////////////////////////////////

bool SEDInstrument::detectTotalBatch(const vector<PhotonPacket*>& ppv)
{
    vector<PhotonPacket*> insidev;
    insidev.reserve(ppv.size());
    for (PhotonPacket* pp : ppv)
        if (isInsideAperture(pp)) insidev.push_back(pp);
    return instrumentFluxRecorder()->detectTotalBatch(insidev);
}

////////////////////////////////////////////////////////////////////

bool SEDInstrument::detectTotalBatch(const vector<Position>& positionv, const vector<double>& wavelengthv,
                                     const vector<double>& luminosityv, const vector<double>& tauv)
{
    if (positionv.size() != wavelengthv.size() || positionv.size() != luminosityv.size()) return false;
    if (!tauv.empty() && tauv.size() != positionv.size()) return false;

    vector<double> insideWavelengthv;
    vector<double> insideLuminosityv;
    vector<double> insideTauv;
    insideWavelengthv.reserve(positionv.size());
    insideLuminosityv.reserve(positionv.size());
    insideTauv.reserve(positionv.size());
    for (size_t i = 0; i != positionv.size(); ++i)
    {
        if (!isInsideAperture(positionv[i])) continue;
        insideWavelengthv.push_back(wavelengthv[i]);
        insideLuminosityv.push_back(luminosityv[i]);
        if (!tauv.empty()) insideTauv.push_back(tauv[i]);
    }
    return instrumentFluxRecorder()->detectTotalBatch(insideWavelengthv, insideLuminosityv, insideTauv);
}

////////////////////////////////////////////////////////////////////

bool SEDInstrument::getGpuDetectorInfo(GpuSedDetectorInfo& info)
{
    if (!instrumentFluxRecorder()->getGpuSedDetectorInfo(info)) return false;
    info.costheta = cos(inclination());
    info.sintheta = sin(inclination());
    info.cosphi = cos(azimuth());
    info.sinphi = sin(azimuth());
    info.radiusSquared = radius() * radius();
    return true;
}

////////////////////////////////////////////////////////////////////
