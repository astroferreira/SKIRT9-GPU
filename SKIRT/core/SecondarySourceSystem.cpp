/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#include "SecondarySourceSystem.hpp"
#include "GpuAcceleration.hpp"
#include "Configuration.hpp"
#include "ContGasSecondarySource.hpp"
#include "DustSecondarySource.hpp"
#include "EmittingGasMix.hpp"
#include "FatalError.hpp"
#include "LineGasSecondarySource.hpp"
#include "MediumSystem.hpp"
#include "NR.hpp"
#include "PhotonPacket.hpp"
#include "ProbePhotonPacketInterface.hpp"
#include <cmath>
#include <limits>

////////////////////////////////////////////////////////////////////

SecondarySourceSystem::SecondarySourceSystem(SimulationItem* parent)
{
    parent->addChild(this);
    setup();
}

////////////////////////////////////////////////////////////////////

void SecondarySourceSystem::setupSelfBefore()
{
    SimulationItem::setupSelfBefore();

    auto config = find<Configuration>();
    auto ms = find<MediumSystem>();

    // remember the source bias
    _xi = config->secondarySourceBias();

    // add aggregate dust source if applicable
    if (config->hasDustEmission() && ms->hasDust())
    {
        _sources.push_back(new DustSecondarySource(this));
        _wv.push_back(config->dustEmissionSourceWeight());
    }

    // add gas sources if applicable
    if (config->hasGasEmission())
    {
        for (int h : ms->gasMediumIndices())
        {
            auto emittingMix = dynamic_cast<const EmittingGasMix*>(ms->mix(0, h));
            if (emittingMix)
            {
                if (emittingMix->hasContinuumEmission())
                {
                    _sources.push_back(new ContGasSecondarySource(this, h));
                    _wv.push_back(emittingMix->sourceWeight());
                }
                if (emittingMix->hasLineEmission())
                {
                    _sources.push_back(new LineGasSecondarySource(this, h));
                    _wv.push_back(emittingMix->sourceWeight());
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////

void SecondarySourceSystem::installLaunchCallBack(ProbePhotonPacketInterface* callback)
{
    _callbackv.push_back(callback);
}

////////////////////////////////////////////////////////////////////

int SecondarySourceSystem::numSources() const
{
    return _sources.size();
}

////////////////////////////////////////////////////////////////////

bool SecondarySourceSystem::prepareForLaunch(size_t numPackets)
{
    // Primary source buffers are no longer needed while preparing dust and gas emission.
    GpuAcceleration::clearSourceLaunchCaches();
    // obtain the luminosity for each source
    int Ns = _sources.size();
    _Lv.resize(Ns);
    for (int s = 0; s != Ns; ++s) _Lv[s] = _sources[s]->prepareLuminosities();

    // calculate the total luminosity and report failure if it is zero
    _L = _Lv.sum();
    if (!_L) return false;

    // normalize the individual luminosities to unity
    _Lv /= _L;

    // calculate the launch weight for each source, normalized to unity
    Array wv = NR::array(_wv);
    Array wLv = wv * _Lv;
    _Wv = (1 - _xi) * wLv / wLv.sum() + _xi * wv / wv.sum();

    // resize the history index mapping vector
    _Iv.resize(Ns + 1);

    // determine the first history index for each source
    _Iv[0] = 0;
    double W = 0.;
    for (int s = 1; s != Ns; ++s)
    {
        // track the cumulative normalized weight as a floating point number
        // and limit the index to numPackets to avoid issues with rounding errors
        W += _Wv[s - 1];
        _Iv[s] = min(numPackets, static_cast<size_t>(std::round(W * numPackets)));
    }
    _Iv[Ns] = numPackets;

    // calculate the average luminosity contribution for each packet
    _Lpp = _L / numPackets;

    //  pass the mapping on to each source
    for (int s = 0; s != Ns; ++s) _sources[s]->preparePacketMap(_Iv[s], _Iv[s + 1] - _Iv[s]);

    // report success
    return true;
}

////////////////////////////////////////////////////////////////////

void SecondarySourceSystem::launch(PhotonPacket* pp, size_t historyIndex) const
{
    // ask the appropriate source to prepare the photon packet for launch
    auto s = std::upper_bound(_Iv.cbegin(), _Iv.cend(), historyIndex) - _Iv.cbegin() - 1;
    double weight = _Lv[s] / _Wv[s];
    _sources[s]->launch(pp, historyIndex, _Lpp * weight);

    // add origin info (the index reflects the secondary source, not the medium component)
    pp->setSecondaryOrigin(s);

    // invoke any installed launch call-backs
    for (auto callback : _callbackv) callback->probePhotonPacket(pp);
}

////////////////////////////////////////////////////////////////////

void SecondarySourceSystem::launchBatch(PhotonPacket* ppv, size_t firstHistoryIndex, size_t numPackets) const
{
    if (!numPackets) return;

    size_t endHistoryIndex = firstHistoryIndex + numPackets;
    size_t offset = 0;
    while (offset != numPackets)
    {
        size_t historyIndex = firstHistoryIndex + offset;
        auto s = std::upper_bound(_Iv.cbegin(), _Iv.cend(), historyIndex) - _Iv.cbegin() - 1;
        size_t sourceEnd = min(endHistoryIndex, _Iv[s + 1]);
        size_t count = sourceEnd - historyIndex;
        double weight = _Lv[s] / _Wv[s];

        _sources[s]->launchBatch(ppv + offset, historyIndex, count, _Lpp * weight);
        for (size_t i = 0; i != count; ++i)
        {
            PhotonPacket* pp = ppv + offset + i;
            pp->setSecondaryOrigin(s);
            for (auto callback : _callbackv) callback->probePhotonPacket(pp);
        }
        offset += count;
    }
}

////////////////////////////////////////////////////////////////////

bool SecondarySourceSystem::launchBatchData(vector<double>& packetData, size_t firstHistoryIndex,
                                            size_t numPackets) const
{
    packetData.clear();
    if (!numPackets || !_callbackv.empty() || numPackets > std::numeric_limits<size_t>::max() / 8) return false;

    packetData.reserve(8 * numPackets);
    size_t endHistoryIndex = firstHistoryIndex + numPackets;
    size_t offset = 0;
    while (offset != numPackets)
    {
        size_t historyIndex = firstHistoryIndex + offset;
        auto s = std::upper_bound(_Iv.cbegin(), _Iv.cend(), historyIndex) - _Iv.cbegin() - 1;
        size_t sourceEnd = min(endHistoryIndex, _Iv[s + 1]);
        size_t count = sourceEnd - historyIndex;
        double weight = _Lv[s] / _Wv[s];

        vector<double> sourceData;
        if (!_sources[s]->launchBatchData(sourceData, historyIndex, count, _Lpp * weight)
            || sourceData.size() != 8 * count)
        {
            packetData.clear();
            return false;
        }
        packetData.insert(packetData.end(), sourceData.begin(), sourceData.end());
        offset += count;
    }
    return packetData.size() == 8 * numPackets;
}

////////////////////////////////////////////////////////////////////

bool SecondarySourceSystem::gpuSourceLaunchData(const GpuSourceLaunchData*& sourceData,
                                                double& luminosityPerPacket, size_t firstHistoryIndex,
                                                size_t numPackets) const
{
    sourceData = nullptr;
    luminosityPerPacket = 0.;
    if (!numPackets || !_callbackv.empty() || _Iv.size() < 2
        || firstHistoryIndex > std::numeric_limits<size_t>::max() - numPackets)
        return false;

    size_t s = static_cast<size_t>(std::upper_bound(_Iv.cbegin(), _Iv.cend(), firstHistoryIndex) - _Iv.cbegin() - 1);
    if (s >= _sources.size() || firstHistoryIndex > std::numeric_limits<size_t>::max() - numPackets
        || firstHistoryIndex + numPackets > _Iv[s + 1])
        return false;
    sourceData = _sources[s]->gpuSourceLaunchData();
    if (!sourceData) return false;

    luminosityPerPacket = _Lpp * _Lv[s] / _Wv[s];
    return std::isfinite(luminosityPerPacket) && luminosityPerPacket > 0.;
}

////////////////////////////////////////////////////////////////////
