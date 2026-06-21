/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#include "FrameInstrument.hpp"
#include "FluxRecorder.hpp"
#include "PhotonPacket.hpp"

////////////////////////////////////////////////////////////////////

void FrameInstrument::setupSelfBefore()
{
    DistantInstrument::setupSelfBefore();

    // configure flux recorder
    instrumentFluxRecorder()->includeSurfaceBrightnessForDistant(
        numPixelsX(), numPixelsY(), fieldOfViewX() / numPixelsX(), fieldOfViewY() / numPixelsY(), centerX(), centerY());

    // precalculate information needed by pixelOnDetector() function
    _costheta = cos(inclination());
    _sintheta = sin(inclination());
    _cosphi = cos(azimuth());
    _sinphi = sin(azimuth());
    _cosomega = cos(roll());
    _sinomega = sin(roll());
    _Nxp = numPixelsX();
    _Nyp = numPixelsY();
    _xpmin = centerX() - 0.5 * fieldOfViewX();
    _xpsiz = fieldOfViewX() / numPixelsX();
    _ypmin = centerY() - 0.5 * fieldOfViewY();
    _ypsiz = fieldOfViewY() / numPixelsY();
}

////////////////////////////////////////////////////////////////////

void FrameInstrument::detect(PhotonPacket* pp)
{
    int l = pixelOnDetector(pp);
    instrumentFluxRecorder()->detect(pp, l);
}

////////////////////////////////////////////////////////////////////

bool FrameInstrument::detectTotalBatch(const vector<PhotonPacket*>& ppv)
{
    vector<int> pixelv(ppv.size(), -1);
    for (size_t i = 0; i != ppv.size(); ++i)
        if (ppv[i]) pixelv[i] = pixelOnDetector(ppv[i]);
    return instrumentFluxRecorder()->detectTotalBatch(ppv, pixelv);
}

////////////////////////////////////////////////////////////////////

bool FrameInstrument::detectTotalBatch(const vector<Position>& positionv, const vector<double>& wavelengthv,
                                       const vector<double>& luminosityv, const vector<double>& tauv)
{
    if (instrumentFluxRecorder()->detectTotalFrameBandBatch(
            positionv, wavelengthv, luminosityv, tauv, _costheta, _sintheta, _cosphi, _sinphi, _cosomega,
            _sinomega, _Nxp, _Nyp, _xpmin, _xpsiz, _ypmin, _ypsiz))
        return true;

    vector<int> pixelv(positionv.size(), -1);
    for (size_t i = 0; i != positionv.size(); ++i) pixelv[i] = pixelOnDetector(positionv[i]);
    return instrumentFluxRecorder()->detectTotalBatch(pixelv, wavelengthv, luminosityv, tauv);
}

////////////////////////////////////////////////////////////////////

bool FrameInstrument::supportsHenyeyGreensteinScatteringFrameBandBatch()
{
    return instrumentFluxRecorder()->supportsHenyeyGreensteinScatteringFrameBandBatch();
}

////////////////////////////////////////////////////////////////////

bool FrameInstrument::supportsObservedFrameBandBatch()
{
    return instrumentFluxRecorder()->supportsObservedFrameBandBatch();
}

////////////////////////////////////////////////////////////////////

bool FrameInstrument::detectObservedFrameBandBatch(const vector<Position>& positionv,
                                                  const vector<double>& wavelengthv,
                                                  const vector<double>& luminosityv)
{
    if (positionv.empty()) return false;
    Direction bfkobs = this->bfkobs(positionv.front());
    return instrumentFluxRecorder()->detectObservedFrameBandBatch(
        positionv, wavelengthv, luminosityv, bfkobs, _costheta, _sintheta, _cosphi, _sinphi,
        _cosomega, _sinomega, _Nxp, _Nyp, _xpmin, _xpsiz, _ypmin, _ypsiz);
}

////////////////////////////////////////////////////////////////////

bool FrameInstrument::detectHenyeyGreensteinScatteringFrameBandBatch(const vector<PhotonPacket*>& ppv,
                                                                     const vector<Position>& positionv,
                                                                     const vector<double>& wavelengthv)
{
    if (positionv.empty()) return false;
    Direction bfkobs = this->bfkobs(positionv.front());
    return instrumentFluxRecorder()->detectHenyeyGreensteinScatteringFrameBandBatch(
        ppv, positionv, wavelengthv, bfkobs, _costheta, _sintheta, _cosphi, _sinphi, _cosomega,
        _sinomega, _Nxp, _Nyp, _xpmin, _xpsiz, _ypmin, _ypsiz);
}

////////////////////////////////////////////////////////////////////

int FrameInstrument::pixelOnDetector(const PhotonPacket* pp) const
{
    return pixelOnDetector(pp->position());
}

////////////////////////////////////////////////////////////////////

int FrameInstrument::pixelOnDetector(const Position& bfr) const
{
    // get the position
    double x, y, z;
    bfr.cartesian(x, y, z);

    // transform to detector coordinates using inclination, azimuth, and roll angle
    double xpp = -_sinphi * x + _cosphi * y;
    double ypp = -_cosphi * _costheta * x - _sinphi * _costheta * y + _sintheta * z;
    double xp = _cosomega * xpp - _sinomega * ypp;
    double yp = _sinomega * xpp + _cosomega * ypp;

    // scale and round to pixel index
    int i = static_cast<int>(floor((xp - _xpmin) / _xpsiz));
    int j = static_cast<int>(floor((yp - _ypmin) / _ypsiz));
    if (i < 0 || i >= _Nxp || j < 0 || j >= _Nyp)
        return -1;
    else
        return i + _Nxp * j;
}

////////////////////////////////////////////////////////////////////
