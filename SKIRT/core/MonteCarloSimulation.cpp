/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#include "MonteCarloSimulation.hpp"
#include "DistantInstrument.hpp"
#include "FrameInstrument.hpp"
#include "GpuAcceleration.hpp"
#include "Log.hpp"
#include "Parallel.hpp"
#include "ParallelFactory.hpp"
#include "PhotonPacket.hpp"
#include "ProcessManager.hpp"
#include "SecondarySourceSystem.hpp"
#include "ShortArray.hpp"
#include "SpecialFunctions.hpp"
#include "StringUtils.hpp"
#include "TimeLogger.hpp"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>

////////////////////////////////////////////////////////////////////

namespace
{
    bool isTruthyEnvironmentFlag(const char* name)
    {
        const char* value = std::getenv(name);
        if (!value) return false;
        string text(value);
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
        return !(text.empty() || text == "0" || text == "false" || text == "off" || text == "no");
    }

    bool environmentFlag(const char* name, bool fallback)
    {
        const char* value = std::getenv(name);
        if (!value) return fallback;
        string text(value);
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
        return !(text.empty() || text == "0" || text == "false" || text == "off" || text == "no");
    }

    size_t environmentSizeValue(const char* name, size_t fallback)
    {
        const char* text = std::getenv(name);
        if (!text) return fallback;
        char* end = nullptr;
        unsigned long value = std::strtoul(text, &end, 10);
        return end != text && value > 0 ? static_cast<size_t>(value) : fallback;
    }

    int environmentThreadValue(const char* name, int fallback)
    {
        const char* text = std::getenv(name);
        if (!text) return fallback;
        char* end = nullptr;
        long value = std::strtol(text, &end, 10);
        return end != text && value >= 0 ? static_cast<int>(value) : fallback;
    }

    int lifeCycleThreadLimit(const Configuration* config)
    {
        if (!config || !environmentFlag("SKIRTGPU_BATCH_FIRST_FORCED", true) || !config->hasMedium()
            || !config->forceScattering() || !GpuAcceleration::isProcessEnabled()
            || !environmentFlag("SKIRTGPU_LIMIT_LIFECYCLE_THREADS", true))
            return 0;

        return environmentThreadValue("SKIRTGPU_LIFECYCLE_THREADS", 0);
    }

    double exponCutoffFromUniform(double xmax, double u)
    {
        if (xmax == 0.0) return 0.0;
        if (xmax < 1e-10) return u * xmax;
        double x = -log(1.0 - u * (1.0 - exp(-xmax)));
        return x > xmax ? xmax : x;
    }
}

////////////////////////////////////////////////////////////////////

void MonteCarloSimulation::setupSimulation()
{
    // perform regular setup for the hierarchy and wait for all processes to finish
    {
        TimeLogger logger(log(), "setup");
        _config->setup();  // first of all perform setup for the configuration object
        SimulationItem::setup();
        wait("setup");
    }

    // write setup output
    {
        TimeLogger logger(log(), "setup output");

        // notify the probe system
        probeSystem()->probeSetup();
    }
}

////////////////////////////////////////////////////////////////////

void MonteCarloSimulation::setupSelfAfter()
{
    Simulation::setupSelfAfter();

    // construct a secondary source system to help launch secondary photon packets if required
    if (_config->hasSecondaryEmission()) _secondarySourceSystem = new SecondarySourceSystem(this);
}

////////////////////////////////////////////////////////////////////

Configuration* MonteCarloSimulation::config() const
{
    return _config;
}

////////////////////////////////////////////////////////////////////

void MonteCarloSimulation::runSimulation()
{
    // run the simulation
    {
        TimeLogger logger(log(), "the run");

        bool hasPrimaryLuminosity = sourceSystem()->luminosity() > 0.;

        // special case of merged primary and secondary iterations
        if (_config->hasMergedIterations() && hasPrimaryLuminosity)
        {
            if (_config->hasPrimaryIterations()) runPrimaryEmissionIterations();
            runMergedEmissionIterations();
            runPrimaryEmission();
            runSecondaryEmission();
        }
        else
        {
            // primary emission phase, possibly with dynamic medium state iterations
            if (_config->hasPrimaryIterations() && hasPrimaryLuminosity) runPrimaryEmissionIterations();
            runPrimaryEmission();

            // optional secondary emission phase, possibly with dynamic secondary emission iterations
            if (_config->hasSecondaryEmission())
            {
                if (_config->hasSecondaryIterations()) runSecondaryEmissionIterations();
                runSecondaryEmission();
            }
        }
    }

    // write final output
    {
        TimeLogger logger(log(), "final output");

        // notify the probe system
        probeSystem()->probeRun();

        // write instrument output
        instrumentSystem()->flush();
        instrumentSystem()->write();
    }
}

////////////////////////////////////////////////////////////////////

void MonteCarloSimulation::runPrimaryEmission()
{
    string segment = "primary emission";
    TimeLogger logger(log(), segment);

    // clear the radiation field
    if (_config->hasRadiationField()) mediumSystem()->clearRadiationField(true);

    // shoot photons from primary sources, if needed
    size_t Npp = _config->numPrimaryPackets();
    if (!Npp)
    {
        log()->warning("Skipping primary emission because no photon packets were requested");
    }
    else if (!sourceSystem()->luminosity())
    {
        log()->warning("Skipping primary emission because the total luminosity of primary sources is zero");
    }
    else
    {
        initProgress(segment, Npp);
        sourceSystem()->prepareForLaunch(Npp);
	        auto parallel = find<ParallelFactory>()->parallelDistributed(lifeCycleThreadLimit(_config));
        parallel->call(
            Npp, [this](size_t i, size_t n) { performLifeCycle(i, n, true, true, _config->hasRadiationField()); });
        instrumentSystem()->flush();
    }

    // wait for all processes to finish and synchronize the radiation field
    wait(segment);
    if (_config->hasRadiationField()) mediumSystem()->communicateRadiationField(true);

    // update secondary dynamic medium state if applicable (in which case we have a medium system)
    if (_config->hasSecondaryDynamicState()) mediumSystem()->updateSecondaryDynamicMediumState();
}

////////////////////////////////////////////////////////////////////

void MonteCarloSimulation::runSecondaryEmission()
{
    string segment = "secondary emission";
    TimeLogger logger(log(), segment);

    // determine whether we need to store the radiation field during secondary emission
    // if so, clear the secondary radiation field
    bool storeRF = _config->storeEmissionRadiationField();
    if (storeRF) mediumSystem()->clearRadiationField(false);

    // shoot photons from secondary sources, if needed
    size_t Npp = _config->numSecondaryPackets();
    if (!Npp)
    {
        log()->warning("Skipping secondary emission because no photon packets were requested");
    }
    else if (!_secondarySourceSystem->prepareForLaunch(Npp))
    {
        log()->warning("Skipping secondary emission because the total luminosity of secondary sources is zero");
    }
    else
    {
        initProgress(segment, Npp);
	        auto parallel = find<ParallelFactory>()->parallelDistributed(lifeCycleThreadLimit(_config));
        parallel->call(Npp, [this, storeRF](size_t i, size_t n) { performLifeCycle(i, n, false, true, storeRF); });
        instrumentSystem()->flush();
    }

    // wait for all processes to finish and synchronize the radiation field if needed
    wait(segment);
    if (storeRF) mediumSystem()->communicateRadiationField(false);
}

////////////////////////////////////////////////////////////////////

namespace
{
    // this helper class checks secondary emission convergence based on the dust-absorbed luminosity
    class DustAbsorptionConvergence
    {
        double _prevLabsseco{0.};  // remembers the absorbed luminosity in the previous iteration

    public:
        // this function determines and logs the total absorbed luminosity and related percentages
        // it returns true if secondary emission can be considered to be converged, false otherwise
        bool logConvergenceInfo(Log* log, Units* units, MediumSystem* mediumSystem, int iter, double fractionOfPrimary,
                                double fractionOfPrevious)
        {
            // determine and log the total absorbed luminosity
            double Labsprim, Labsseco;
            std::tie(Labsprim, Labsseco) = mediumSystem->totalDustAbsorbedLuminosity();
            log->info("The total dust-absorbed primary luminosity is "
                      + StringUtils::toString(units->obolluminosity(Labsprim), 'g') + " " + units->ubolluminosity());
            log->info("The total dust-absorbed secondary luminosity in iteration " + std::to_string(iter) + " is "
                      + StringUtils::toString(units->obolluminosity(Labsseco), 'g') + " " + units->ubolluminosity());

            // log the current performance and corresponding convergence criteria
            if (Labsprim > 0. && Labsseco > 0.)
            {
                if (iter == 1)
                {
                    log->info("--> absorbed secondary luminosity is "
                              + StringUtils::toString(Labsseco / Labsprim * 100., 'f', 2)
                              + "% of absorbed primary luminosity (convergence criterion is "
                              + StringUtils::toString(fractionOfPrimary * 100., 'f', 2) + "%)");
                }
                else
                {
                    log->info("--> absorbed secondary luminosity changed by "
                              + StringUtils::toString(abs((Labsseco - _prevLabsseco) / Labsseco) * 100., 'f', 2)
                              + "% compared to previous iteration (convergence criterion is "
                              + StringUtils::toString(fractionOfPrevious * 100., 'f', 2) + "%)");
                }
            }

            // secondary emission has reached convergence if one or more of the following conditions holds:
            // - the absorbed primary luminosity is zero
            // - the absorbed secondary luminosity is zero
            // - the absorbed secondary luminosity is less than a given fraction of the absorbed primary luminosity
            // - the absorbed secondary luminosity has changed by less than a given fraction compared to previous iter
            bool converged = Labsprim <= 0. || Labsseco <= 0. || Labsseco / Labsprim < fractionOfPrimary
                             || abs((Labsseco - _prevLabsseco) / Labsseco) < fractionOfPrevious;
            _prevLabsseco = Labsseco;
            return converged;
        }
    };

    // this function logs the convergence status and returns true if the loop should exit, false if it should continue
    // specifically, the loop exits
    //   - if convergence is reached after the minimum number of iterations, or
    //   - if the maximum number of iterations has completed, even if there is no convergence
    bool logLoopConvergence(Log* log, bool converged, int iter, int minIters, int maxIters)
    {
        // force at least the minimum number of iterations
        if (converged && iter < minIters)
        {
            log->info("Convergence reached but continuing until " + std::to_string(minIters)
                      + " iterations have been performed");
            return false;
        }
        // exit the loop if convergence has been reached after at least the minimum number of iterations
        if (converged && iter >= minIters)
        {
            log->info("Convergence reached after " + std::to_string(iter) + " iterations");
            return true;
        }
        // continue if convergence has not been reached after fewer than the maximum number of iterations
        if (!converged && iter < maxIters)
        {
            log->info("Convergence not yet reached after " + std::to_string(iter) + " iterations");
            return false;
        }
        // exit the loop if convergence has not been reached after the maximum number of iterations
        if (!converged && iter >= maxIters)
        {
            log->error("Convergence not yet reached after " + std::to_string(iter) + " iterations");
            return true;
        }
        return true;  // the logic can never get here
    }
}

////////////////////////////////////////////////////////////////////

void MonteCarloSimulation::runPrimaryEmissionIterations()
{
    // when this function is called
    //  - the number of photon packets and the primary source luminosity are guaranteed to be nonzero
    //  - the data structures to store the radiation field are guaranteed to exist

    // get the parallel engine
	    auto parallel = find<ParallelFactory>()->parallelDistributed(lifeCycleThreadLimit(_config));

    // get the parameters controlling the dynamic state iteration
    size_t Npp = _config->numPrimaryIterationPackets();
    int minIters = _config->minPrimaryIterations();
    int maxIters = _config->maxPrimaryIterations();

    // prepare the source system for the appropriate number of packets
    sourceSystem()->prepareForLaunch(Npp);

    // loop over the dynamic state iterations
    int iter = 0;
    while (true)
    {
        ++iter;

        bool converged = true;
        {
            string segment = "primary emission iteration " + std::to_string(iter);
            TimeLogger logger(log(), segment);

            mediumSystem()->beginDynamicMediumStateIteration();

            // clear the radiation field
            mediumSystem()->clearRadiationField(true);

            // launch photon packets
            initProgress(segment, Npp);
            parallel->call(Npp, [this](size_t i, size_t n) { performLifeCycle(i, n, true, false, true); });
            instrumentSystem()->flush();

            // wait for all processes to finish and synchronize the radiation field
            wait(segment);
            mediumSystem()->communicateRadiationField(true);

            // update the primary dynamic medium state and log convergence info
            converged = mediumSystem()->updatePrimaryDynamicMediumState();
        }

        // notify the probe system
        probeSystem()->probePrimary(iter);

        // verify and log loop convergence
        if (logLoopConvergence(log(), converged, iter, minIters, maxIters)) break;
    }
}

////////////////////////////////////////////////////////////////////

void MonteCarloSimulation::runSecondaryEmissionIterations()
{
    // when this function is called
    //  - the number of photon packets is guaranteed to be nonzero
    //  - the total luminosity of secondary sources may still be zero
    //  - the data structures to store the radiation field are guaranteed to exist

    // get the parallel engine
	    auto parallel = find<ParallelFactory>()->parallelDistributed(lifeCycleThreadLimit(_config));

    // get the parameters controlling the dynamic secondary emission iteration
    size_t Npp = _config->numSecondaryIterationPackets();
    int minIters = _config->minSecondaryIterations();
    int maxIters = _config->maxSecondaryIterations();
    double fractionOfPrimary = _config->maxFractionOfPrimary();
    double fractionOfPrevious = _config->maxFractionOfPrevious();

    // helper object to verify convergence of secondary emission
    DustAbsorptionConvergence dustConvergence;

    // loop over the secondary emission iterations
    int iter = 0;
    while (true)
    {
        ++iter;

        bool converged = true;
        {
            string segment = "secondary emission iteration " + std::to_string(iter);
            TimeLogger logger(log(), segment);

            mediumSystem()->beginDynamicMediumStateIteration();

            // clear the secondary radiation field
            mediumSystem()->clearRadiationField(false);

            // prepare the source system; terminate if secondary luminosity is zero (which would be very unusual)
            if (!_secondarySourceSystem->prepareForLaunch(Npp))
            {
                log()->warning(
                    "Skipping secondary emission iterations because the total luminosity of secondary sources is zero");
                return;
            }

            // launch photon packets
            initProgress(segment, Npp);
            parallel->call(Npp, [this](size_t i, size_t n) { performLifeCycle(i, n, false, false, true); });
            instrumentSystem()->flush();

            // wait for all processes to finish and synchronize the radiation field
            wait(segment);
            mediumSystem()->communicateRadiationField(false);

            // update secondary dynamic medium state and log convergence info
            converged &= mediumSystem()->updateSecondaryDynamicMediumState();

            // log dust emission convergence info
            if (mediumSystem()->hasDust())
                converged &= dustConvergence.logConvergenceInfo(log(), units(), mediumSystem(), iter, fractionOfPrimary,
                                                                fractionOfPrevious);
        }

        // notify the probe system
        probeSystem()->probeSecondary(iter);

        // verify and log loop convergence
        if (logLoopConvergence(log(), converged, iter, minIters, maxIters)) break;
    }
}

////////////////////////////////////////////////////////////////////

void MonteCarloSimulation::runMergedEmissionIterations()
{
    // when this function is called
    //  - the number of photon packets and the primary source luminosity are guaranteed to be nonzero
    //  - the total luminosity of secondary sources may still be zero
    //  - the data structures to store the radiation field are guaranteed to exist

    // get the parallel engine
	    auto parallel = find<ParallelFactory>()->parallelDistributed(lifeCycleThreadLimit(_config));

    // get the parameters controlling the merged iteration
    size_t Npp1 = _config->numPrimaryIterationPackets();
    size_t Npp2 = _config->numSecondaryIterationPackets();
    int minIters = _config->minSecondaryIterations();
    int maxIters = _config->maxSecondaryIterations();
    double fractionOfPrimary = _config->maxFractionOfPrimary();
    double fractionOfPrevious = _config->maxFractionOfPrevious();

    // prepare the primary source system for the appropriate number of packets
    sourceSystem()->prepareForLaunch(Npp1);

    // helper object to verify convergence of secondary emission
    DustAbsorptionConvergence dustConvergence;

    // loop over the merged iterations
    int iter = 0;
    while (true)
    {
        ++iter;

        bool converged = true;
        {
            string segment = "merged primary and secondary emission iteration " + std::to_string(iter);
            string segment1 = "merged primary emission iteration " + std::to_string(iter);
            string segment2 = "merged secondary emission iteration " + std::to_string(iter);
            TimeLogger logger(log(), segment);

            mediumSystem()->beginDynamicMediumStateIteration();

            // clear the radiation field
            mediumSystem()->clearRadiationField(true);

            // launch photon packets
            initProgress(segment1, Npp1);
            parallel->call(Npp1, [this](size_t i, size_t n) { performLifeCycle(i, n, true, false, true); });
            instrumentSystem()->flush();

            // wait for all processes to finish and synchronize the radiation field
            wait(segment1);
            mediumSystem()->communicateRadiationField(true);

            // update secondary dynamic medium state and log convergence info
            converged &= mediumSystem()->updateSecondaryDynamicMediumState();

            // clear the secondary radiation field
            mediumSystem()->clearRadiationField(false);

            // prepare the source system; terminate if secondary luminosity is zero (which would be very unusual)
            if (!_secondarySourceSystem->prepareForLaunch(Npp2))
            {
                log()->warning(
                    "Skipping merged emission iterations because the total luminosity of secondary sources is zero");
                return;
            }

            // launch photon packets
            initProgress(segment2, Npp2);
            parallel->call(Npp2, [this](size_t i, size_t n) { performLifeCycle(i, n, false, false, true); });
            instrumentSystem()->flush();

            // wait for all processes to finish and synchronize the radiation field
            wait(segment2);
            mediumSystem()->communicateRadiationField(false);

            // update the primary dynamic medium state and log convergence info
            converged &= mediumSystem()->updatePrimaryDynamicMediumState();

            // log dust emission convergence info
            if (mediumSystem()->hasDust())
                converged &= dustConvergence.logConvergenceInfo(log(), units(), mediumSystem(), iter, fractionOfPrimary,
                                                                fractionOfPrevious);
        }

        // notify the probe system
        probeSystem()->probeSecondary(iter);

        // verify and log loop convergence
        if (logLoopConvergence(log(), converged, iter, minIters, maxIters)) break;
    }
}

////////////////////////////////////////////////////////////////////

void MonteCarloSimulation::wait(std::string scope)
{
    if (ProcessManager::isMultiProc())
    {
        log()->info("Waiting for other processes to finish " + scope + "...");
        ProcessManager::wait();
    }
}

////////////////////////////////////////////////////////////////////

void MonteCarloSimulation::initProgress(string segment, size_t numTotal)
{
    _segment = segment;

    log()->info("Launching " + StringUtils::toString(static_cast<double>(numTotal)) + " " + _segment
                + " photon packets");
    log()->infoSetElapsed(numTotal);
}

////////////////////////////////////////////////////////////////////

void MonteCarloSimulation::logProgress(size_t numDone)
{
    // log message if the minimum time has elapsed
    log()->infoIfElapsed("Launched " + _segment + " photon packets: ", numDone);
}

////////////////////////////////////////////////////////////////////

namespace
{
    // maximum number of photon packets processed between two invocations of infoIfElapsed()
    const size_t logProgressChunkSize = 1000;

    size_t lifeCycleChunkSize(const Configuration* config)
    {
        if (config && environmentFlag("SKIRTGPU_BATCH_FIRST_FORCED", true) && config->hasMedium()
            && config->forceScattering() && GpuAcceleration::isProcessEnabled())
            return environmentSizeValue("SKIRTGPU_BATCH_FIRST_FORCED_CHUNK", 262144);

        return logProgressChunkSize;
    }

    double elapsedMilliseconds(std::chrono::steady_clock::time_point begin,
                               std::chrono::steady_clock::time_point end)
    {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    }

    string profileMilliseconds(double value)
    {
        return StringUtils::toString(value, 'f', 2);
    }
}

////////////////////////////////////////////////////////////////////

void MonteCarloSimulation::performLifeCycle(size_t firstIndex, size_t numIndices, bool primary, bool peel, bool store)
{
    PhotonPacket pp, ppp;

    // loop over the history indices, with interruptions for progress logging
    while (numIndices)
    {
        size_t currentChunkSize = min(lifeCycleChunkSize(_config), numIndices);
        bool processedChunk = performLifeCycleWithBatchedFirstForcedStep(firstIndex, currentChunkSize, primary, peel,
                                                                         store);
        if (!processedChunk)
        {
            for (size_t historyIndex = firstIndex; historyIndex != firstIndex + currentChunkSize; ++historyIndex)
            {
                // launch a photon packet from the requested source
                if (primary)
                    sourceSystem()->launch(&pp, historyIndex);
                else
                    _secondarySourceSystem->launch(&pp, historyIndex);
                if (pp.luminosity() > 0)
                {
                    if (peel) peelOffEmission(&pp, &ppp);

                    // trace the packet through the media, if any
                    if (_config->hasMedium())
                    {
                        // --- forced scattering ---
                        if (_config->forceScattering())
                        {
                            double Lthreshold = pp.luminosity() / _config->minWeightReduction();
                            int minScattEvents = _config->minScattEvents();
                            while (true)
                            {
                                // calculate segments and optical depths for the complete path
                                if (_config->explicitAbsorption())
                                    mediumSystem()->setScatteringAndAbsorptionOpticalDepths(&pp);
                                else
                                    mediumSystem()->setExtinctionOpticalDepths(&pp);

                                // advance the packet
                                if (store) storeRadiationField(primary, &pp);
                                simulateForcedPropagation(&pp);

                                // if the packet's weight drops below the threshold, terminate it
                                if (pp.luminosity() <= 0
                                    || (pp.luminosity() <= Lthreshold && pp.numScatt() >= minScattEvents))
                                    break;

                                // process the scattering event
                                if (peel) peelOffScattering(&pp, &ppp);
                                mediumSystem()->simulateScattering(random(), &pp);
                            }
                        }
                        // --- non-forced scattering ---
                        else
                        {
                            while (true)
                            {
                                // advance the packet (without storing the radiation field)
                                // if the interaction point is outside of the path, terminate the packet
                                if (!simulateNonForcedPropagation(&pp)) break;

                                // if the packet's weight drops to zero, terminate it
                                if (pp.luminosity() <= 0) break;

                                // process the scattering event
                                if (peel) peelOffScattering(&pp, &ppp);
                                mediumSystem()->simulateScattering(random(), &pp);
                            }
                        }
                    }
                }
            }
        }

        // log progress
        logProgress(currentChunkSize);
        firstIndex += currentChunkSize;
        numIndices -= currentChunkSize;
    }
}

////////////////////////////////////////////////////////////////////

void MonteCarloSimulation::peelOffEmission(const PhotonPacket* pp, PhotonPacket* ppp)
{
    for (Instrument* instrument : _instrumentSystem->instruments())
    {
        if (!instrument->isSameObserverAsPreceding())
        {
            const Direction bfkobs = instrument->bfkobs(pp->position());
            ppp->launchEmissionPeelOff(pp, bfkobs);

            // if the photon packet is polarised, we have to rotate the Stokes vector into the frame of the instrument
            if (ppp->isPolarized())
            {
                ppp->rotateIntoPlane(bfkobs, instrument->bfky(pp->position()));
            }
        }
        instrument->detect(ppp);
    }
}

////////////////////////////////////////////////////////////////////

bool MonteCarloSimulation::peelOffEmissionBatch(const vector<PhotonPacket*>& ppv)
{
    if (peelOffEmissionBatchDirect(ppv)) return true;

    if (ppv.empty() || !_config->hasMedium() || !GpuAcceleration::isProcessEnabled()) return false;

    const auto& instruments = _instrumentSystem->instruments();
    for (Instrument* instrument : instruments)
        if (!dynamic_cast<DistantInstrument*>(instrument)) return false;

    vector<size_t> observerIndexForInstrument;
    vector<vector<PhotonPacket>> observerPeelPackets;
    observerIndexForInstrument.reserve(instruments.size());
    observerPeelPackets.reserve(instruments.size());

    size_t lastObserverIndex = 0;
    for (Instrument* instrument : instruments)
    {
        if (instrument->isSameObserverAsPreceding())
        {
            if (observerPeelPackets.empty()) return false;
            observerIndexForInstrument.push_back(lastObserverIndex);
            continue;
        }

        size_t observerIndex = observerPeelPackets.size();
        observerPeelPackets.emplace_back(ppv.size());
        vector<PhotonPacket>& peelPackets = observerPeelPackets.back();
        vector<PhotonPacket*> peelPacketPointers;
        vector<double> distancev(ppv.size(), std::numeric_limits<double>::infinity());
        vector<double> tauv;
        peelPacketPointers.reserve(ppv.size());

        for (size_t i = 0; i != ppv.size(); ++i)
        {
            PhotonPacket* pp = ppv[i];
            if (!pp) return false;

            const Direction bfkobs = instrument->bfkobs(pp->position());
            peelPackets[i].launchEmissionPeelOff(pp, bfkobs);

            if (peelPackets[i].isPolarized())
            {
                peelPackets[i].rotateIntoPlane(bfkobs, instrument->bfky(pp->position()));
            }
            peelPacketPointers.push_back(&peelPackets[i]);
        }

        if (!mediumSystem()->observedExtinctionOpticalDepths(peelPacketPointers, distancev, tauv)
            || tauv.size() != ppv.size())
            return false;

        for (size_t i = 0; i != ppv.size(); ++i) peelPackets[i].setObservedOpticalDepth(tauv[i]);
        observerIndexForInstrument.push_back(observerIndex);
        lastObserverIndex = observerIndex;
    }

    for (size_t i = 0; i != ppv.size(); ++i)
    {
        for (size_t instrumentIndex = 0; instrumentIndex != instruments.size(); ++instrumentIndex)
        {
            size_t observerIndex = observerIndexForInstrument[instrumentIndex];
            instruments[instrumentIndex]->detect(&observerPeelPackets[observerIndex][i]);
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool MonteCarloSimulation::peelOffEmissionBatchDirect(const vector<PhotonPacket*>& ppv)
{
    if (ppv.empty() || !_config->hasMedium() || !GpuAcceleration::isProcessEnabled() || _config->hasPolarization())
        return false;

    const auto& instruments = _instrumentSystem->instruments();
    vector<Position> emptyPositionv;
    vector<double> emptyDoublev;
    for (Instrument* instrument : instruments)
    {
        auto frameInstrument = dynamic_cast<FrameInstrument*>(instrument);
        if (!frameInstrument) return false;
        if (!frameInstrument->detectTotalBatch(emptyPositionv, emptyDoublev, emptyDoublev, emptyDoublev))
            return false;
    }

    vector<Position> positionv;
    vector<double> lambdav;
    vector<double> luminosityv;
    positionv.reserve(ppv.size());
    lambdav.reserve(ppv.size());
    luminosityv.reserve(ppv.size());
    for (PhotonPacket* pp : ppv)
    {
        if (!pp) return false;
        if (pp->hasEmissionPeelOffAdjustments()) return false;
        positionv.push_back(pp->position());
        lambdav.push_back(pp->wavelength());
        luminosityv.push_back(pp->luminosity());
    }

    if (isTruthyEnvironmentFlag("SKIRTGPU_EMISSION_PEEL_FRAME_FUSED"))
    {
        bool fusedSupported = true;
        for (Instrument* instrument : instruments)
        {
            auto frameInstrument = dynamic_cast<FrameInstrument*>(instrument);
            if (!frameInstrument || !frameInstrument->supportsObservedFrameBandBatch())
            {
                fusedSupported = false;
                break;
            }
        }
        if (fusedSupported)
        {
            bool fusedDetected = true;
            for (Instrument* instrument : instruments)
            {
                auto frameInstrument = dynamic_cast<FrameInstrument*>(instrument);
                if (!frameInstrument->detectObservedFrameBandBatch(positionv, lambdav, luminosityv))
                {
                    fusedDetected = false;
                    break;
                }
            }
            if (fusedDetected) return true;
        }
    }

    vector<size_t> observerIndexForInstrument;
    vector<vector<double>> observerTauv;
    observerIndexForInstrument.reserve(instruments.size());
    observerTauv.reserve(instruments.size());

    vector<double> distancev(ppv.size(), std::numeric_limits<double>::infinity());
    size_t lastObserverIndex = 0;
    for (Instrument* instrument : instruments)
    {
        if (instrument->isSameObserverAsPreceding())
        {
            if (observerTauv.empty()) return false;
            observerIndexForInstrument.push_back(lastObserverIndex);
            continue;
        }

        Direction bfkobs = instrument->bfkobs(positionv.front());
        vector<double> tauv;
        if (!mediumSystem()->observedExtinctionOpticalDepths(positionv, bfkobs, lambdav, distancev, tauv)
            || tauv.size() != ppv.size())
            return false;

        size_t observerIndex = observerTauv.size();
        observerTauv.push_back(std::move(tauv));
        observerIndexForInstrument.push_back(observerIndex);
        lastObserverIndex = observerIndex;
    }

    for (size_t instrumentIndex = 0; instrumentIndex != instruments.size(); ++instrumentIndex)
    {
        size_t observerIndex = observerIndexForInstrument[instrumentIndex];
        auto frameInstrument = dynamic_cast<FrameInstrument*>(instruments[instrumentIndex]);
        if (!frameInstrument->detectTotalBatch(positionv, lambdav, luminosityv, observerTauv[observerIndex]))
            return false;
    }
    return true;
}

////////////////////////////////////////////////////////////////////

void MonteCarloSimulation::storeRadiationField(bool primary, const PhotonPacket* pp)
{
    // use a faster version in case there are no kinematics
    if (_config->hasConstantPerceivedWavelength())
    {
        int ell = _config->radiationFieldWLG()->bin(pp->wavelength());
        if (ell >= 0)
        {
            double luminosity = pp->luminosity();
            vector<double> Ldsv;
            if (GpuAcceleration::isSynchronousPhotonCycleEnabled()
                && GpuAcceleration::radiationFieldContributions(pp, luminosity, Ldsv))
            {
                const auto& segments = pp->segments();
                for (size_t i = 0; i != segments.size(); ++i)
                {
                    int m = segments[i].m();
                    if (m >= 0) mediumSystem()->storeRadiationField(primary, m, ell, Ldsv[i]);
                }
                return;
            }

            double lnExtBeg = 0.;  // extinction factor and its logarithm at begin of current segment
            double extBeg = 1.;
            for (const auto& segment : pp->segments())
            {
                double lnExtEnd = -segment.tauExt();  // extinction factor and its logarithm at end of current segment
                double extEnd = exp(lnExtEnd);
                int m = segment.m();
                if (m >= 0)
                {
                    // use this flavor of the lnmean function to avoid recalculating the logarithm of the extinction
                    double extMean = SpecialFunctions::lnmean(extEnd, extBeg, lnExtEnd, lnExtBeg);
                    double Lds = luminosity * extMean * segment.ds();
                    mediumSystem()->storeRadiationField(primary, m, ell, Lds);
                }
                lnExtBeg = lnExtEnd;
                extBeg = extEnd;
            }
        }
    }
    else
    {
        double lnExtBeg = 0.;  // extinction factor and its logarithm at begin of current segment
        double extBeg = 1.;
        for (const auto& segment : pp->segments())
        {
            double lnExtEnd = -segment.tauExt();  // extinction factor and its logarithm at end of current segment
            double extEnd = exp(lnExtEnd);
            int m = segment.m();
            if (m >= 0)
            {
                double lambda = pp->perceivedWavelength(mediumSystem()->bulkVelocity(m),
                                                        _config->hubbleExpansionRate() * segment.s());
                int ell = _config->radiationFieldWLG()->bin(lambda);
                if (ell >= 0)
                {
                    // use this flavor of the lnmean function to avoid recalculating the logarithm of the extinction
                    double extMean = SpecialFunctions::lnmean(extEnd, extBeg, lnExtEnd, lnExtBeg);
                    double Lds = pp->perceivedLuminosity(lambda) * extMean * segment.ds();
                    mediumSystem()->storeRadiationField(primary, m, ell, Lds);
                }
            }
            lnExtBeg = lnExtEnd;
            extBeg = extEnd;
        }
    }
}

////////////////////////////////////////////////////////////////////

bool MonteCarloSimulation::storeRadiationFieldBatch(bool primary, const vector<PhotonPacket*>& ppv)
{
    if (!_config->hasConstantPerceivedWavelength() || ppv.empty()) return false;

    vector<const SpatialGridPath*> pathv;
    vector<double> luminosityv;
    vector<int> wavelengthBinv;
    pathv.reserve(ppv.size());
    luminosityv.reserve(ppv.size());
    wavelengthBinv.reserve(ppv.size());
    for (const PhotonPacket* pp : ppv)
    {
        int ell = _config->radiationFieldWLG()->bin(pp->wavelength());
        if (ell >= 0)
        {
            pathv.push_back(pp);
            luminosityv.push_back(pp->luminosity());
            wavelengthBinv.push_back(ell);
        }
    }
    if (pathv.empty()) return true;

    vector<int> binIndexv;
    vector<double> summedLdsv;
    if (GpuAcceleration::radiationFieldContributionSums(pathv, luminosityv, wavelengthBinv,
                                                        _config->radiationFieldWLG()->numBins(), binIndexv,
                                                        summedLdsv))
    {
        mediumSystem()->storeRadiationField(primary, binIndexv, summedLdsv);
        return true;
    }

    vector<int> pathOffsetv;
    vector<double> Ldsv;
    if (!GpuAcceleration::radiationFieldContributions(pathv, luminosityv, pathOffsetv, Ldsv)) return false;

    for (size_t p = 0; p != pathv.size(); ++p)
    {
        const auto& segments = pathv[p]->segments();
        size_t begin = static_cast<size_t>(pathOffsetv[p]);
        int ell = wavelengthBinv[p];
        for (size_t i = 0; i != segments.size(); ++i)
        {
            int m = segments[i].m();
            if (m >= 0) mediumSystem()->storeRadiationField(primary, m, ell, Ldsv[begin + i]);
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool MonteCarloSimulation::storeRadiationFieldAndSimulateForcedPropagationBatch(
    bool primary, const vector<PhotonPacket*>& ppv)
{
    if (!_config->hasConstantPerceivedWavelength() || _config->explicitAbsorption() || ppv.empty()
        || !GpuAcceleration::isProcessEnabled())
        return false;

    vector<double> luminosityv;
    vector<int> wavelengthBinv;
    vector<const SpatialGridPath*> pathv;
    vector<double> tauinteractv;
    vector<double> taupathv;
    vector<double> pathBiasWeightv(ppv.size(), 1.);
    luminosityv.reserve(ppv.size());
    wavelengthBinv.reserve(ppv.size());
    pathv.reserve(ppv.size());
    tauinteractv.reserve(ppv.size());
    taupathv.reserve(ppv.size());

    double xi = _config->pathLengthBias();
    for (PhotonPacket* pp : ppv)
    {
        if (!pp) return false;

        luminosityv.push_back(pp->luminosity());
        wavelengthBinv.push_back(_config->radiationFieldWLG()->bin(pp->wavelength()));
        pathv.push_back(pp);

        double taupath = pp->totalOpticalDepth();
        taupathv.push_back(taupath);
        if (taupath <= 0.)
        {
            tauinteractv.push_back(0.);
            continue;
        }

        double tau = 0.;
        if (xi == 0.)
        {
            tau = random()->exponCutoff(taupath);
        }
        else
        {
            tau = random()->uniform() < xi ? random()->uniform() * taupath : random()->exponCutoff(taupath);
            double p = -exp(-tau) / expm1(-taupath);
            double q = (1.0 - xi) * p + xi / taupath;
            pp->applyBias(p / q);
        }
        tauinteractv.push_back(tau);
    }

    vector<int> binIndexv;
    vector<double> summedLdsv;
    vector<int> cellv;
    vector<double> distancev;
    vector<double> tauAbsv;
    vector<double> weightv;
    bool haveCombined = mediumSystem()->radiationFieldAndForcedPropagationResults(
        ppv, luminosityv, wavelengthBinv, _config->radiationFieldWLG()->numBins(), tauinteractv,
        pathBiasWeightv, binIndexv, summedLdsv, cellv, distancev, tauAbsv, weightv, primary)
                        && cellv.size() == ppv.size() && distancev.size() == ppv.size()
                        && tauAbsv.size() == ppv.size() && weightv.size() == ppv.size();

    if (haveCombined)
    {
        mediumSystem()->storeRadiationField(primary, binIndexv, summedLdsv);
    }
    else
    {
        if (!storeRadiationFieldBatch(primary, ppv))
            for (PhotonPacket* pp : ppv) storeRadiationField(primary, pp);

        haveCombined = mediumSystem()->forcedPropagationResults(ppv, tauinteractv, pathBiasWeightv, cellv,
                                                                distancev, tauAbsv, weightv)
                       && cellv.size() == ppv.size() && distancev.size() == ppv.size()
                       && tauAbsv.size() == ppv.size() && weightv.size() == ppv.size();
        if (!haveCombined)
        {
            bool foundOnGpu = GpuAcceleration::findInteractionPointsInCumulativePaths(
                pathv, tauinteractv, false, cellv, distancev, tauAbsv);
            vector<double> albedov;
            bool haveBatchAlbedos = mediumSystem()->albedosForScattering(ppv, albedov)
                                    && albedov.size() == ppv.size();

            for (size_t i = 0; i != ppv.size(); ++i)
            {
                PhotonPacket* pp = ppv[i];
                if (taupathv[i] <= 0.)
                {
                    pp->applyBias(0.);
                    continue;
                }

                if (foundOnGpu)
                    pp->setInteractionPoint(cellv[i], distancev[i], tauAbsv[i]);
                else
                    pp->findInteractionPoint(tauinteractv[i]);

                double albedo = haveBatchAlbedos ? albedov[i] : mediumSystem()->albedoForScattering(pp);
                pp->applyBias(-expm1(-taupathv[i]) * albedo);
                pp->propagate(pp->interactionDistance());
            }
            return true;
        }
    }

    for (size_t i = 0; i != ppv.size(); ++i)
    {
        PhotonPacket* pp = ppv[i];
        if (cellv[i] < 0)
        {
            pp->applyBias(0.);
            continue;
        }
        pp->setInteractionPoint(cellv[i], distancev[i], tauAbsv[i]);
        pp->applyBias(weightv[i]);
        pp->propagate(pp->interactionDistance());
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool MonteCarloSimulation::storeRadiationFieldAndSimulateForcedPropagationWithoutPreparedPaths(
    bool primary, const vector<PhotonPacket*>& ppv, vector<Direction>* scatterDirectionv)
{
    if (!environmentFlag("SKIRTGPU_RESIDENT_FORCED", true) || !_config->hasConstantPerceivedWavelength()
        || _config->explicitAbsorption() || ppv.empty() || !GpuAcceleration::isProcessEnabled())
        return false;
    if (scatterDirectionv) scatterDirectionv->clear();

    static std::atomic<size_t> profileStoreForcedCounter{0};
    bool profileStoreForced = isTruthyEnvironmentFlag("SKIRTGPU_PROFILE_STORE_FORCED");
    size_t profileCall = 0;
    bool emitProfile = false;
    if (profileStoreForced)
    {
        size_t limit = environmentSizeValue("SKIRTGPU_PROFILE_STORE_FORCED_LIMIT", 0);
        profileCall = profileStoreForcedCounter.fetch_add(1, std::memory_order_relaxed);
        emitProfile = limit == 0 || profileCall < limit;
    }
    auto profileStart = std::chrono::steady_clock::now();
    double profileSetupMs = 0.;
    double profileRandomMs = 0.;
    double profileResidentGpuMs = 0.;
    double profileFallbackPrepareMs = 0.;
    double profileFallbackGpuMs = 0.;
    double profileRfStoreMs = 0.;
    double profilePacketApplyMs = 0.;

    vector<double> luminosityv;
    vector<int> wavelengthBinv;
    vector<const SpatialGridPath*> pathv;
    luminosityv.reserve(ppv.size());
    wavelengthBinv.reserve(ppv.size());
    pathv.reserve(ppv.size());
    for (PhotonPacket* pp : ppv)
    {
        if (!pp) return false;
        luminosityv.push_back(pp->luminosity());
        wavelengthBinv.push_back(_config->radiationFieldWLG()->bin(pp->wavelength()));
        pathv.push_back(pp);
    }
    if (emitProfile) profileSetupMs = elapsedMilliseconds(profileStart, std::chrono::steady_clock::now());

    double xi = _config->pathLengthBias();
    vector<double> taupathv;
    vector<double> tauinteractv;
    vector<double> pathBiasWeightv(ppv.size(), 1.);
    vector<int> binIndexv;
    vector<double> summedLdsv;
    vector<int> cellv;
    vector<double> distancev;
    vector<double> tauAbsv;
    vector<double> weightv;

    auto validateCombinedResults = [&]() {
        if (binIndexv.size() != summedLdsv.size() || cellv.size() != ppv.size()
            || distancev.size() != ppv.size() || tauAbsv.size() != ppv.size()
            || weightv.size() != ppv.size())
            return false;
        for (size_t k = 0; k != binIndexv.size(); ++k)
            if (binIndexv[k] < 0 || !std::isfinite(summedLdsv[k])) return false;
        for (size_t i = 0; i != ppv.size(); ++i)
        {
            if (cellv[i] < 0) continue;
            if (!std::isfinite(distancev[i]) || distancev[i] < 0. || !std::isfinite(tauAbsv[i])
                || tauAbsv[i] < 0. || !std::isfinite(weightv[i]))
                return false;
        }
        return true;
    };

    vector<double> randomSelectv;
    vector<double> randomSamplev;
    vector<double> scatterRandomCosthetav;
    vector<double> scatterRandomPhiv;
    bool sampleOnGpu = environmentFlag("SKIRTGPU_RESIDENT_SAMPLE_ON_GPU", true);
    bool scatterOnGpu = scatterDirectionv && environmentFlag("SKIRTGPU_RESIDENT_HG_SCATTER", false);
    bool haveCombined = false;
    if (sampleOnGpu)
    {
        randomSelectv.reserve(ppv.size());
        randomSamplev.reserve(ppv.size());
        if (scatterOnGpu)
        {
            scatterRandomCosthetav.reserve(ppv.size());
            scatterRandomPhiv.reserve(ppv.size());
        }
        for (size_t i = 0; i != ppv.size(); ++i)
        {
            randomSelectv.push_back(random()->uniform());
            randomSamplev.push_back(xi == 0. ? 0. : random()->uniform());
            if (scatterOnGpu)
            {
                scatterRandomCosthetav.push_back(random()->uniform());
                scatterRandomPhiv.push_back(random()->uniform());
            }
        }
        if (emitProfile)
            profileRandomMs = elapsedMilliseconds(profileStart, std::chrono::steady_clock::now()) - profileSetupMs;

        if (scatterOnGpu)
        {
            auto stageStart = std::chrono::steady_clock::now();
            haveCombined =
                mediumSystem()
                    ->sampledRadiationFieldForcedPropagationAndHenyeyGreensteinScatteringResultsWithoutPreparedPaths(
                        ppv, luminosityv, wavelengthBinv, _config->radiationFieldWLG()->numBins(), randomSelectv,
                        randomSamplev, xi, scatterRandomCosthetav, scatterRandomPhiv, binIndexv, summedLdsv, cellv,
                        distancev, tauAbsv, weightv, *scatterDirectionv, primary)
                && validateCombinedResults() && scatterDirectionv->size() == ppv.size();
            if (emitProfile)
                profileResidentGpuMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
            if (!haveCombined) scatterDirectionv->clear();
        }
        if (!haveCombined)
        {
            auto stageStart = std::chrono::steady_clock::now();
            haveCombined = mediumSystem()->sampledRadiationFieldAndForcedPropagationResultsWithoutPreparedPaths(
                               ppv, luminosityv, wavelengthBinv, _config->radiationFieldWLG()->numBins(), randomSelectv,
                               randomSamplev, xi, binIndexv, summedLdsv, cellv, distancev, tauAbsv, weightv, primary)
                           && validateCombinedResults();
            if (emitProfile)
                profileResidentGpuMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
        }
    }
    else
    {
        vector<double> distanceLimitv(ppv.size(), std::numeric_limits<double>::infinity());
        auto stageStart = std::chrono::steady_clock::now();
        if (mediumSystem()->observedExtinctionOpticalDepths(ppv, distanceLimitv, taupathv)
            && taupathv.size() == ppv.size())
        {
            if (emitProfile)
            {
                profileFallbackPrepareMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
                stageStart = std::chrono::steady_clock::now();
            }
            tauinteractv.reserve(ppv.size());
            for (size_t i = 0; i != ppv.size(); ++i)
            {
                PhotonPacket* pp = ppv[i];
                double taupath = taupathv[i];
                if (taupath <= 0.)
                {
                    tauinteractv.push_back(0.);
                    continue;
                }

                double tau = 0.;
                if (xi == 0.)
                {
                    tau = random()->exponCutoff(taupath);
                }
                else
                {
                    tau = random()->uniform() < xi ? random()->uniform() * taupath
                                                    : random()->exponCutoff(taupath);
                    double p = -exp(-tau) / expm1(-taupath);
                    double q = (1.0 - xi) * p + xi / taupath;
                    pp->applyBias(p / q);
                }
                tauinteractv.push_back(tau);
            }
            if (emitProfile)
            {
                profileRandomMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
                stageStart = std::chrono::steady_clock::now();
            }

            haveCombined = mediumSystem()->radiationFieldAndForcedPropagationResultsWithoutPreparedPaths(
                               ppv, luminosityv, wavelengthBinv, _config->radiationFieldWLG()->numBins(),
                               tauinteractv, pathBiasWeightv, binIndexv, summedLdsv, cellv, distancev, tauAbsv,
                               weightv, primary)
                           && validateCombinedResults();
            if (emitProfile)
                profileFallbackGpuMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
        }
    }

    auto logStoreForcedProfile = [&](const string& result) {
        if (!emitProfile) return;
        double totalMs = elapsedMilliseconds(profileStart, std::chrono::steady_clock::now());
        log()->info("SKIRTGPU store-forced profile call=" + std::to_string(profileCall)
                    + " packets=" + std::to_string(ppv.size())
                    + " bins=" + std::to_string(binIndexv.size())
                    + " combined=" + string(haveCombined ? "yes" : "no")
                    + " sample_gpu=" + string(sampleOnGpu ? "yes" : "no")
                    + " scatter_gpu=" + string(scatterOnGpu ? "yes" : "no")
                    + " setup_ms=" + profileMilliseconds(profileSetupMs)
                    + " random_ms=" + profileMilliseconds(profileRandomMs)
                    + " resident_gpu_ms=" + profileMilliseconds(profileResidentGpuMs)
                    + " fallback_prepare_ms=" + profileMilliseconds(profileFallbackPrepareMs)
                    + " fallback_gpu_ms=" + profileMilliseconds(profileFallbackGpuMs)
                    + " rf_store_ms=" + profileMilliseconds(profileRfStoreMs)
                    + " packet_apply_ms=" + profileMilliseconds(profilePacketApplyMs)
                    + " total_ms=" + profileMilliseconds(totalMs)
                    + " result=" + result);
    };

    if (haveCombined)
    {
        auto stageStart = std::chrono::steady_clock::now();
        mediumSystem()->storeRadiationField(primary, binIndexv, summedLdsv);
        if (emitProfile)
            profileRfStoreMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
    }
    else
    {
        auto stageStart = std::chrono::steady_clock::now();
        if (!mediumSystem()->setExtinctionOpticalDepths(ppv))
            for (PhotonPacket* pp : ppv) mediumSystem()->setExtinctionOpticalDepths(pp);
        if (emitProfile)
        {
            profileFallbackPrepareMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
            stageStart = std::chrono::steady_clock::now();
        }

        if (tauinteractv.size() != ppv.size() || taupathv.size() != ppv.size())
        {
            bool usePredrawnRandoms = randomSelectv.size() == ppv.size() && randomSamplev.size() == ppv.size();
            taupathv.clear();
            tauinteractv.clear();
            taupathv.reserve(ppv.size());
            tauinteractv.reserve(ppv.size());
            for (size_t i = 0; i != ppv.size(); ++i)
            {
                PhotonPacket* pp = ppv[i];
                double taupath = pp->totalOpticalDepth();
                taupathv.push_back(taupath);
                if (taupath <= 0.)
                {
                    tauinteractv.push_back(0.);
                    continue;
                }

                double tau = 0.;
                if (xi == 0.)
                {
                    tau = usePredrawnRandoms ? exponCutoffFromUniform(taupath, randomSelectv[i])
                                             : random()->exponCutoff(taupath);
                }
                else
                {
                    double select = usePredrawnRandoms ? randomSelectv[i] : random()->uniform();
                    if (select < xi)
                    {
                        double sample = usePredrawnRandoms ? randomSamplev[i] : random()->uniform();
                        tau = sample * taupath;
                    }
                    else
                    {
                        tau = usePredrawnRandoms ? exponCutoffFromUniform(taupath, randomSamplev[i])
                                                 : random()->exponCutoff(taupath);
                    }
                    double p = -exp(-tau) / expm1(-taupath);
                    double q = (1.0 - xi) * p + xi / taupath;
                    pp->applyBias(p / q);
                }
                tauinteractv.push_back(tau);
            }
        }
        if (emitProfile)
        {
            profileRandomMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
            stageStart = std::chrono::steady_clock::now();
        }

        haveCombined = mediumSystem()->radiationFieldAndForcedPropagationResults(
            ppv, luminosityv, wavelengthBinv, _config->radiationFieldWLG()->numBins(), tauinteractv,
            pathBiasWeightv, binIndexv, summedLdsv, cellv, distancev, tauAbsv, weightv, primary)
                        && validateCombinedResults();
        if (emitProfile)
        {
            profileFallbackGpuMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
            stageStart = std::chrono::steady_clock::now();
        }
        if (haveCombined)
        {
            mediumSystem()->storeRadiationField(primary, binIndexv, summedLdsv);
            if (emitProfile)
            {
                profileRfStoreMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
                stageStart = std::chrono::steady_clock::now();
            }
        }
        else
        {
            if (!storeRadiationFieldBatch(primary, ppv))
                for (PhotonPacket* pp : ppv) storeRadiationField(primary, pp);
            if (emitProfile)
            {
                profileRfStoreMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
                stageStart = std::chrono::steady_clock::now();
            }

            haveCombined = mediumSystem()->forcedPropagationResults(ppv, tauinteractv, pathBiasWeightv, cellv,
                                                                    distancev, tauAbsv, weightv)
                           && cellv.size() == ppv.size() && distancev.size() == ppv.size()
                           && tauAbsv.size() == ppv.size() && weightv.size() == ppv.size();
            if (emitProfile)
            {
                profileFallbackGpuMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
                stageStart = std::chrono::steady_clock::now();
            }
            if (!haveCombined)
            {
                bool foundOnGpu = GpuAcceleration::findInteractionPointsInCumulativePaths(
                    pathv, tauinteractv, false, cellv, distancev, tauAbsv);
                vector<double> albedov;
                bool haveBatchAlbedos = mediumSystem()->albedosForScattering(ppv, albedov)
                                        && albedov.size() == ppv.size();

                for (size_t i = 0; i != ppv.size(); ++i)
                {
                    PhotonPacket* pp = ppv[i];
                    if (taupathv[i] <= 0.)
                    {
                        pp->applyBias(0.);
                        continue;
                    }

                    if (foundOnGpu)
                        pp->setInteractionPoint(cellv[i], distancev[i], tauAbsv[i]);
                    else
                        pp->findInteractionPoint(tauinteractv[i]);

                    double albedo = haveBatchAlbedos ? albedov[i] : mediumSystem()->albedoForScattering(pp);
                    pp->applyBias(-expm1(-taupathv[i]) * albedo);
                    pp->propagate(pp->interactionDistance());
                }
                if (emitProfile)
                    profilePacketApplyMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
                logStoreForcedProfile("fallback_manual");
                return true;
            }
        }
    }

    auto stageStart = std::chrono::steady_clock::now();
    for (size_t i = 0; i != ppv.size(); ++i)
    {
        PhotonPacket* pp = ppv[i];
        if (cellv[i] < 0)
        {
            pp->applyBias(0.);
            continue;
        }
        pp->setInteractionPoint(cellv[i], distancev[i], tauAbsv[i]);
        pp->applyBias(weightv[i]);
        pp->propagate(pp->interactionDistance());
    }
    if (emitProfile)
        profilePacketApplyMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
    logStoreForcedProfile("combined");
    return true;
}

////////////////////////////////////////////////////////////////////

void MonteCarloSimulation::simulateForcedPropagation(PhotonPacket* pp)
{
    // get the total optical depth
    double taupath = pp->totalOpticalDepth();

    // if there is no extinction along the path, this photon packet cannot scatter, so terminate it right away
    if (taupath <= 0.)
    {
        pp->applyBias(0.);
        return;
    }

    // generate a random optical depth
    double xi = _config->pathLengthBias();
    double tau = 0.;
    if (xi == 0.)
    {
        tau = random()->exponCutoff(taupath);
    }
    else
    {
        tau = random()->uniform() < xi ? random()->uniform() * taupath : random()->exponCutoff(taupath);
        double p = -exp(-tau) / expm1(-taupath);
        double q = (1.0 - xi) * p + xi / taupath;
        double weight = p / q;
        pp->applyBias(weight);
    }

    // determine the physical position of the interaction point
    int m = -1;
    double s = 0.;
    double tauAbs = 0.;
    if (GpuAcceleration::findInteractionPointInCumulativePath(pp, tau, _config->explicitAbsorption(), m, s, tauAbs))
        pp->setInteractionPoint(m, s, tauAbs);
    else
        pp->findInteractionPoint(tau);

    // adjust the photon packet weight with the escape fraction and, depending on the type of photon cycle,
    // with either the scattered fraction or the cumulative absorption optical depth at the interaction point
    if (_config->explicitAbsorption())
    {
        tauAbs = pp->interactionOpticalDepth();
        pp->applyBias(-expm1(-taupath) * exp(-tauAbs));
    }
    else
    {
        double albedo = mediumSystem()->albedoForScattering(pp);
        pp->applyBias(-expm1(-taupath) * albedo);
    }

    // advance the photon packet position
    pp->propagate(pp->interactionDistance());
}

////////////////////////////////////////////////////////////////////

void MonteCarloSimulation::simulateForcedPropagationBatch(const vector<PhotonPacket*>& ppv)
{
    vector<const SpatialGridPath*> pathv;
    vector<double> tauinteractv;
    vector<double> taupathv;
    vector<PhotonPacket*> activev;
    pathv.reserve(ppv.size());
    tauinteractv.reserve(ppv.size());
    taupathv.reserve(ppv.size());
    activev.reserve(ppv.size());

    double xi = _config->pathLengthBias();
    for (PhotonPacket* pp : ppv)
    {
        if (!pp) continue;
        double taupath = pp->totalOpticalDepth();
        if (taupath <= 0.)
        {
            pp->applyBias(0.);
            continue;
        }

        double tau = 0.;
        if (xi == 0.)
        {
            tau = random()->exponCutoff(taupath);
        }
        else
        {
            tau = random()->uniform() < xi ? random()->uniform() * taupath : random()->exponCutoff(taupath);
            double p = -exp(-tau) / expm1(-taupath);
            double q = (1.0 - xi) * p + xi / taupath;
            pp->applyBias(p / q);
        }

        activev.push_back(pp);
        pathv.push_back(pp);
        tauinteractv.push_back(tau);
        taupathv.push_back(taupath);
    }

    if (activev.empty()) return;

    vector<int> cellv;
    vector<double> distancev;
    vector<double> tauAbsv;
    vector<double> weightv;
    vector<double> pathBiasWeightv(activev.size(), 1.);
    bool haveForcedBatch = mediumSystem()->forcedPropagationResults(activev, tauinteractv, pathBiasWeightv, cellv,
                                                                    distancev, tauAbsv, weightv)
                            && cellv.size() == activev.size() && distancev.size() == activev.size()
                            && tauAbsv.size() == activev.size() && weightv.size() == activev.size();
    if (haveForcedBatch)
    {
        for (size_t i = 0; i != activev.size(); ++i)
        {
            PhotonPacket* pp = activev[i];
            pp->setInteractionPoint(cellv[i], distancev[i], tauAbsv[i]);
            pp->applyBias(weightv[i]);
            pp->propagate(pp->interactionDistance());
        }
        return;
    }

    bool foundOnGpu = GpuAcceleration::findInteractionPointsInCumulativePaths(
        pathv, tauinteractv, _config->explicitAbsorption(), cellv, distancev, tauAbsv);

    for (size_t i = 0; i != activev.size(); ++i)
    {
        PhotonPacket* pp = activev[i];
        if (foundOnGpu)
            pp->setInteractionPoint(cellv[i], distancev[i], tauAbsv[i]);
        else
            pp->findInteractionPoint(tauinteractv[i]);
    }

    vector<double> albedov;
    bool haveBatchAlbedos = !_config->explicitAbsorption()
                            && mediumSystem()->albedosForScattering(activev, albedov)
                            && albedov.size() == activev.size();

    for (size_t i = 0; i != activev.size(); ++i)
    {
        PhotonPacket* pp = activev[i];
        if (_config->explicitAbsorption())
        {
            double tauAbs = pp->interactionOpticalDepth();
            pp->applyBias(-expm1(-taupathv[i]) * exp(-tauAbs));
        }
        else
        {
            double albedo = haveBatchAlbedos ? albedov[i] : mediumSystem()->albedoForScattering(pp);
            pp->applyBias(-expm1(-taupathv[i]) * albedo);
        }

        pp->propagate(pp->interactionDistance());
    }
}

////////////////////////////////////////////////////////////////////

bool MonteCarloSimulation::performLifeCycleWithBatchedFirstForcedStep(size_t firstIndex, size_t numIndices,
                                                                      bool primary, bool peel, bool store)
{
    if (!environmentFlag("SKIRTGPU_BATCH_FIRST_FORCED", true) || !_config->hasMedium()
        || !_config->forceScattering() || !GpuAcceleration::isProcessEnabled() || numIndices < 2)
        return false;

    static std::atomic<size_t> profileBatchCounter{0};
    bool profileBatch = isTruthyEnvironmentFlag("SKIRTGPU_PROFILE_BATCH");
    size_t profileBatchCall = 0;
    if (profileBatch)
    {
        size_t limit = environmentSizeValue("SKIRTGPU_PROFILE_BATCH_LIMIT", 0);
        profileBatchCall = profileBatchCounter.fetch_add(1, std::memory_order_relaxed);
        if (limit != 0 && profileBatchCall >= limit) profileBatch = false;
    }
    auto profileStart = std::chrono::steady_clock::now();
    auto profileMark = profileStart;
    double launchMs = 0.;
    double peelMs = 0.;
    double emissionPeelBatchMs = 0.;
    double emissionPeelScalarMs = 0.;
    double prepareMs = 0.;
    double storeMs = 0.;
    double forcedMs = 0.;
    double combinedStoreForcedMs = 0.;
    double scatterMs = 0.;
    double laterPeelMs = 0.;
    double laterPeelBatchAttemptMs = 0.;
    double laterPeelScalarMs = 0.;
    double scalarMs = 0.;
    auto captureProfile = [&](double& bucket) {
        if (!profileBatch) return;
        auto now = std::chrono::steady_clock::now();
        bucket += elapsedMilliseconds(profileMark, now);
        profileMark = now;
    };

    vector<PhotonPacket> ppv(numIndices);
    PhotonPacket ppp;
    vector<PhotonPacket*> activev;
    vector<double> Lthresholdv(numIndices, 0.);
    activev.reserve(numIndices);
    bool residentHgScatter = environmentFlag("SKIRTGPU_RESIDENT_HG_SCATTER", false);
    vector<Direction> pendingScatterDirectionv(numIndices);
    vector<char> pendingScatterValidv(numIndices, false);
    auto rememberScatterDirections = [&](const vector<PhotonPacket*>& packets, const vector<Direction>& directions) {
        if (directions.size() != packets.size()) return;
        for (size_t i = 0; i != packets.size(); ++i)
        {
            PhotonPacket* pp = packets[i];
            if (!pp) continue;
            auto index = static_cast<size_t>(pp - ppv.data());
            if (index >= numIndices) continue;
            if (pp->luminosity() > 0.)
            {
                pendingScatterDirectionv[index] = directions[i];
                pendingScatterValidv[index] = true;
            }
            else
            {
                pendingScatterValidv[index] = false;
            }
        }
    };

    if (primary)
        sourceSystem()->launchBatch(ppv.data(), firstIndex, numIndices);
    else
        _secondarySourceSystem->launchBatch(ppv.data(), firstIndex, numIndices);
    for (size_t i = 0; i != numIndices; ++i)
    {
        PhotonPacket* pp = &ppv[i];
        if (pp->luminosity() > 0)
        {
            Lthresholdv[i] = pp->luminosity() / _config->minWeightReduction();
            activev.push_back(pp);
        }
    }
    captureProfile(launchMs);

    if (peel && !activev.empty())
    {
        auto stageStart = std::chrono::steady_clock::now();
        bool peeledBatch = peelOffEmissionBatch(activev);
        if (profileBatch)
            emissionPeelBatchMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
        if (!peeledBatch)
        {
            stageStart = std::chrono::steady_clock::now();
            for (PhotonPacket* pp : activev) peelOffEmission(pp, &ppp);
            if (profileBatch)
                emissionPeelScalarMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
        }
    }
    captureProfile(peelMs);

    bool preparedBatch = false;
    bool residentBatch = false;
    vector<Direction> residentScatterDirections;
    if (!activev.empty() && store && !_config->explicitAbsorption()
        && storeRadiationFieldAndSimulateForcedPropagationWithoutPreparedPaths(
            primary, activev, residentHgScatter ? &residentScatterDirections : nullptr))
    {
        residentBatch = true;
        rememberScatterDirections(activev, residentScatterDirections);
        captureProfile(combinedStoreForcedMs);
    }

    if (!residentBatch)
    {
        if (!activev.empty())
        {
            preparedBatch = _config->explicitAbsorption()
                                ? mediumSystem()->setScatteringAndAbsorptionOpticalDepths(activev)
                                : mediumSystem()->setExtinctionOpticalDepths(activev);
        }
        captureProfile(prepareMs);

        if (preparedBatch)
        {
            if (store && storeRadiationFieldAndSimulateForcedPropagationBatch(primary, activev))
            {
                captureProfile(combinedStoreForcedMs);
            }
            else
            {
                if (store && !storeRadiationFieldBatch(primary, activev))
                    for (PhotonPacket* pp : activev) storeRadiationField(primary, pp);
                captureProfile(storeMs);
                simulateForcedPropagationBatch(activev);
                captureProfile(forcedMs);
            }
        }
    }

    int minScattEvents = _config->minScattEvents();
    size_t laterScatteringEvents = 0;
    size_t batchedRounds = 0;
    bool batchForcedRounds = (preparedBatch || residentBatch) && environmentFlag("SKIRTGPU_BATCH_FORCED_ROUNDS", true);
    if (batchForcedRounds)
    {
        bool compactRoundQueue = environmentFlag("SKIRTGPU_COMPACT_ROUND_QUEUE", false);
        vector<PhotonPacket*> roundv;
        vector<PhotonPacket*> nextRoundv;
        roundv.reserve(activev.size());
        nextRoundv.reserve(activev.size());
        if (compactRoundQueue)
        {
            for (PhotonPacket* pp : activev)
            {
                auto index = static_cast<size_t>(pp - ppv.data());
                if (index < numIndices && pp->luminosity() > 0
                    && !(pp->luminosity() <= Lthresholdv[index] && pp->numScatt() >= minScattEvents))
                    roundv.push_back(pp);
            }
        }

        while (true)
        {
            if (!compactRoundQueue)
            {
                roundv.clear();
                for (size_t i = 0; i != numIndices; ++i)
                {
                    PhotonPacket* pp = &ppv[i];
                    if (pp->luminosity() > 0
                        && !(pp->luminosity() <= Lthresholdv[i] && pp->numScatt() >= minScattEvents))
                        roundv.push_back(pp);
                }
            }
            if (roundv.empty()) break;

            ++batchedRounds;
            if (peel)
            {
                auto stageStart = std::chrono::steady_clock::now();
                bool peeledBatch = peelOffScatteringBatch(roundv);
                if (profileBatch)
                    laterPeelBatchAttemptMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
                if (!peeledBatch)
                {
                    stageStart = std::chrono::steady_clock::now();
                    for (PhotonPacket* pp : roundv) peelOffScattering(pp, &ppp);
                    if (profileBatch)
                        laterPeelScalarMs += elapsedMilliseconds(stageStart, std::chrono::steady_clock::now());
                }
            }
            captureProfile(laterPeelMs);

            bool scatterPrecomputed = residentHgScatter;
            if (scatterPrecomputed)
            {
                for (PhotonPacket* pp : roundv)
                {
                    auto index = static_cast<size_t>(pp - ppv.data());
                    if (index >= numIndices || !pendingScatterValidv[index])
                    {
                        scatterPrecomputed = false;
                        break;
                    }
                }
            }
            if (scatterPrecomputed)
            {
                for (PhotonPacket* pp : roundv)
                {
                    auto index = static_cast<size_t>(pp - ppv.data());
                    pp->setScatteringComponent(0);
                    pp->scatter(pendingScatterDirectionv[index], Vec(), pp->wavelength());
                    pendingScatterValidv[index] = false;
                }
            }
            bool scatterBatched = !scatterPrecomputed && environmentFlag("SKIRTGPU_BATCH_HG_SCATTER", false)
                                  && mediumSystem()->simulateScatteringBatch(random(), roundv);
            if (!scatterPrecomputed && !scatterBatched)
            {
                for (PhotonPacket* pp : roundv) mediumSystem()->simulateScattering(random(), pp);
            }
            laterScatteringEvents += roundv.size();
            captureProfile(scatterMs);

            residentScatterDirections.clear();
            bool roundResident = store && !_config->explicitAbsorption()
                                  && storeRadiationFieldAndSimulateForcedPropagationWithoutPreparedPaths(
                                      primary, roundv, residentHgScatter ? &residentScatterDirections : nullptr);
            if (roundResident)
            {
                rememberScatterDirections(roundv, residentScatterDirections);
                captureProfile(combinedStoreForcedMs);
            }
            else
            {
                bool roundPrepared = _config->explicitAbsorption()
                                         ? mediumSystem()->setScatteringAndAbsorptionOpticalDepths(roundv)
                                         : mediumSystem()->setExtinctionOpticalDepths(roundv);
                captureProfile(prepareMs);

                if (roundPrepared)
                {
                    if (store && storeRadiationFieldAndSimulateForcedPropagationBatch(primary, roundv))
                    {
                        captureProfile(combinedStoreForcedMs);
                    }
                    else
                    {
                        if (store && !storeRadiationFieldBatch(primary, roundv))
                            for (PhotonPacket* pp : roundv) storeRadiationField(primary, pp);
                        captureProfile(storeMs);
                        simulateForcedPropagationBatch(roundv);
                        captureProfile(forcedMs);
                    }
                }
                else
                {
                    for (PhotonPacket* pp : roundv)
                    {
                        if (_config->explicitAbsorption())
                            mediumSystem()->setScatteringAndAbsorptionOpticalDepths(pp);
                        else
                            mediumSystem()->setExtinctionOpticalDepths(pp);
                        if (store) storeRadiationField(primary, pp);
                        simulateForcedPropagation(pp);
                    }
                    captureProfile(scalarMs);
                }
            }

            if (compactRoundQueue)
            {
                nextRoundv.clear();
                for (PhotonPacket* pp : roundv)
                {
                    auto index = static_cast<size_t>(pp - ppv.data());
                    if (index < numIndices && pp->luminosity() > 0
                        && !(pp->luminosity() <= Lthresholdv[index] && pp->numScatt() >= minScattEvents))
                        nextRoundv.push_back(pp);
                }
                roundv.swap(nextRoundv);
            }
        }
    }
    else
    {
        for (size_t i = 0; i != numIndices; ++i)
        {
            PhotonPacket* pp = &ppv[i];
            if (pp->luminosity() <= 0) continue;

            if (!preparedBatch)
            {
                if (_config->explicitAbsorption())
                    mediumSystem()->setScatteringAndAbsorptionOpticalDepths(pp);
                else
                    mediumSystem()->setExtinctionOpticalDepths(pp);
                if (store) storeRadiationField(primary, pp);
                simulateForcedPropagation(pp);
            }

            while (pp->luminosity() > 0
                   && !(pp->luminosity() <= Lthresholdv[i] && pp->numScatt() >= minScattEvents))
            {
                ++laterScatteringEvents;
                if (peel) peelOffScattering(pp, &ppp);
                mediumSystem()->simulateScattering(random(), pp);

                if (_config->explicitAbsorption())
                    mediumSystem()->setScatteringAndAbsorptionOpticalDepths(pp);
                else
                    mediumSystem()->setExtinctionOpticalDepths(pp);
                if (store) storeRadiationField(primary, pp);
                simulateForcedPropagation(pp);
            }
        }
        captureProfile(scalarMs);
    }

    if (profileBatch)
    {
        double totalMs = elapsedMilliseconds(profileStart, std::chrono::steady_clock::now());
        log()->info("SKIRTGPU batch profile call=" + std::to_string(profileBatchCall)
                    + " index=" + std::to_string(firstIndex) + " count=" + std::to_string(numIndices)
                    + " active=" + std::to_string(activev.size())
                    + " prepared=" + string(preparedBatch ? "yes" : "no")
                    + " launch_ms=" + profileMilliseconds(launchMs)
                    + " peel_ms=" + profileMilliseconds(peelMs)
                    + " emission_peel_batch_ms=" + profileMilliseconds(emissionPeelBatchMs)
                    + " emission_peel_scalar_ms=" + profileMilliseconds(emissionPeelScalarMs)
                    + " prepare_ms=" + profileMilliseconds(prepareMs)
                    + " store_ms=" + profileMilliseconds(storeMs)
                    + " forced_ms=" + profileMilliseconds(forcedMs)
                    + " store_forced_ms=" + profileMilliseconds(combinedStoreForcedMs)
                    + " later_peel_ms=" + profileMilliseconds(laterPeelMs)
                    + " later_peel_batch_ms=" + profileMilliseconds(laterPeelBatchAttemptMs)
                    + " later_peel_scalar_ms=" + profileMilliseconds(laterPeelScalarMs)
                    + " scatter_ms=" + profileMilliseconds(scatterMs)
                    + " scalar_ms=" + profileMilliseconds(scalarMs)
                    + " total_ms=" + profileMilliseconds(totalMs)
                    + " later_scatter=" + std::to_string(laterScatteringEvents)
                    + " batched_rounds=" + std::to_string(batchedRounds));
    }

    return true;
}

////////////////////////////////////////////////////////////////////

bool MonteCarloSimulation::simulateNonForcedPropagation(PhotonPacket* pp)
{
    // generate a random interaction optical depth
    double tauinteract = random()->expon();

    if (_config->explicitAbsorption())
    {
        // find the physical interaction point corresponding to this scattering optical depth
        // and calculate the absorption optical depth at the interaction point;
        // if the interaction point is outside of the path, terminate the photon packet
        if (!mediumSystem()->setInteractionPointUsingScatteringAndAbsorption(pp, tauinteract)) return false;

        // get the cumulative absorption optical depth at the interaction point
        double tauAbs = pp->interactionOpticalDepth();

        // adjust the photon packet weight by the corresponding extinction (or stimulation, for negative optical depth)
        pp->applyBias(exp(-tauAbs));
    }
    else
    {
        // find the physical interaction point corresponding to this optical depth
        // if the interaction point is outside of the path, terminate the photon packet
        if (!mediumSystem()->setInteractionPointUsingExtinction(pp, tauinteract)) return false;

        // calculate the albedo for the cell containing the interaction point
        double albedo = mediumSystem()->albedoForScattering(pp);

        // adjust the photon packet weight by the albedo
        pp->applyBias(albedo);
    }

    // advance the photon packet position
    pp->propagate(pp->interactionDistance());
    return true;
}

////////////////////////////////////////////////////////////////////

void MonteCarloSimulation::peelOffScattering(PhotonPacket* pp, PhotonPacket* ppp)
{
    // determine the perceived wavelength at the scattering location
    double lambda = mediumSystem()->perceivedWavelengthForScattering(pp);

    // determine the scattering opacity weight for each medium component;
    // abort if none of the media scatter this photon packet
    ShortArray wv;
    if (!mediumSystem()->weightsForScattering(wv, lambda, pp)) return;

    // now do the actual peel-off
    if (_config->needIndividualPeelOff())
    {
        // if wavelengths may change, send a peel-off photon packet per medium component to each instrument
        int numMedia = wv.size();
        for (int h = 0; h != numMedia; ++h)
        {
            // skip media that don't scatter this photon packet
            if (wv[h] > 0.)
            {
                for (Instrument* instr : _instrumentSystem->instruments())
                {
                    if (!instr->isSameObserverAsPreceding())
                    {
                        // get the direction towards the instrument and (for polarization only) its Y-axis orientation
                        Direction bfkobs = instr->bfkobs(pp->position());
                        Direction bfky = _config->hasPolarization() ? instr->bfky(pp->position()) : Direction();

                        // calculate peel-off for the current component and launch the peel-off photon packet
                        mediumSystem()->peelOffScattering(h, wv[h], lambda, bfkobs, bfky, pp, ppp);
                    }

                    // have the peel-off photon packet detected
                    instr->detect(ppp);
                }
            }
        }
    }
    else
    {
        // if wavelengths cannot change, send a consolidated peel-off photon packet to each instrument
        for (Instrument* instr : _instrumentSystem->instruments())
        {
            if (!instr->isSameObserverAsPreceding())
            {
                // get the direction towards the instrument and (for polarization only) its Y-axis orientation
                Direction bfkobs = instr->bfkobs(pp->position());
                Direction bfky = _config->hasPolarization() ? instr->bfky(pp->position()) : Direction();

                // calculate peel-off for all medium components and launch the peel-off photon packet
                // (all media must either support polarization or not; combining these support levels is not allowed)
                mediumSystem()->peelOffScattering(wv, lambda, bfkobs, bfky, pp, ppp);
            }

            // have the peel-off photon packet detected
            instr->detect(ppp);
        }
    }
}

////////////////////////////////////////////////////////////////////

bool MonteCarloSimulation::peelOffScatteringBatch(const vector<PhotonPacket*>& ppv)
{
    if (isTruthyEnvironmentFlag("SKIRTGPU_BATCH_SCATTER_PEEL_DIRECT") && peelOffScatteringBatchDirect(ppv))
        return true;

    if (!isTruthyEnvironmentFlag("SKIRTGPU_BATCH_SCATTER_PEEL") || ppv.empty() || _config->needIndividualPeelOff()
        || !GpuAcceleration::isProcessEnabled())
        return false;

    const auto& instruments = _instrumentSystem->instruments();
    for (Instrument* instrument : instruments)
        if (!dynamic_cast<DistantInstrument*>(instrument)) return false;

    vector<double> lambdav(ppv.size(), 0.);
    vector<ShortArray> weightv(ppv.size());
    vector<char> activev(ppv.size(), false);
    size_t activeCount = 0;
    bool hasActive = false;
    for (size_t i = 0; i != ppv.size(); ++i)
    {
        PhotonPacket* pp = ppv[i];
        if (!pp || pp->luminosity() <= 0.) continue;

        double lambda = mediumSystem()->perceivedWavelengthForScattering(pp);
        if (!mediumSystem()->weightsForScattering(weightv[i], lambda, pp)) continue;

        lambdav[i] = lambda;
        activev[i] = true;
        ++activeCount;
        hasActive = true;
    }
    if (!hasActive) return true;

    vector<size_t> observerIndexForInstrument;
    vector<vector<PhotonPacket>> observerPeelPackets;
    observerIndexForInstrument.reserve(instruments.size());
    observerPeelPackets.reserve(instruments.size());

    size_t lastObserverIndex = 0;
    for (Instrument* instrument : instruments)
    {
        if (instrument->isSameObserverAsPreceding())
        {
            if (observerPeelPackets.empty()) return false;
            observerIndexForInstrument.push_back(lastObserverIndex);
            continue;
        }

        size_t observerIndex = observerPeelPackets.size();
        observerPeelPackets.emplace_back();
        vector<PhotonPacket>& peelPackets = observerPeelPackets.back();
        peelPackets.reserve(activeCount);
        vector<PhotonPacket*> peelPacketPointers;
        vector<Position> positionv;
        vector<double> peelLambdav;
        vector<double> distancev;
        vector<double> tauv;
        peelPacketPointers.reserve(activeCount);
        positionv.reserve(activeCount);
        peelLambdav.reserve(activeCount);
        distancev.reserve(activeCount);
        Direction observerDirection;
        bool haveObserverDirection = false;

        for (size_t i = 0; i != ppv.size(); ++i)
        {
            if (!activev[i]) continue;

            PhotonPacket* pp = ppv[i];
            const Direction bfkobs = instrument->bfkobs(pp->position());
            const Direction bfky = _config->hasPolarization() ? instrument->bfky(pp->position()) : Direction();
            peelPackets.emplace_back();
            PhotonPacket* peelPacket = &peelPackets.back();
            mediumSystem()->peelOffScattering(weightv[i], lambdav[i], bfkobs, bfky, pp, peelPacket);
            if (!haveObserverDirection)
            {
                observerDirection = bfkobs;
                haveObserverDirection = true;
            }
            peelPacketPointers.push_back(peelPacket);
            positionv.push_back(peelPacket->position());
            peelLambdav.push_back(peelPacket->wavelength());
            distancev.push_back(std::numeric_limits<double>::infinity());
        }

        if (!peelPacketPointers.empty())
        {
            if (!(haveObserverDirection
                  && mediumSystem()->observedExtinctionOpticalDepths(positionv, observerDirection, peelLambdav,
                                                                     distancev, tauv)
                  && tauv.size() == peelPacketPointers.size())
                && (!mediumSystem()->observedExtinctionOpticalDepths(peelPacketPointers, distancev, tauv)
                    || tauv.size() != peelPacketPointers.size()))
                return false;

            if (tauv.size() != peelPacketPointers.size())
                return false;

            for (size_t p = 0; p != peelPacketPointers.size(); ++p)
                peelPacketPointers[p]->setObservedOpticalDepth(tauv[p]);
        }

        observerIndexForInstrument.push_back(observerIndex);
        lastObserverIndex = observerIndex;
    }

    for (size_t instrumentIndex = 0; instrumentIndex != instruments.size(); ++instrumentIndex)
    {
        size_t observerIndex = observerIndexForInstrument[instrumentIndex];
        vector<PhotonPacket*> detectv;
        vector<PhotonPacket>& peelPackets = observerPeelPackets[observerIndex];
        detectv.reserve(peelPackets.size());
        for (PhotonPacket& peelPacket : peelPackets) detectv.push_back(&peelPacket);
        if (detectv.empty()) continue;

        if (auto frameInstrument = dynamic_cast<FrameInstrument*>(instruments[instrumentIndex]))
            if (frameInstrument->detectTotalBatch(detectv)) continue;

        for (PhotonPacket* pp : detectv) instruments[instrumentIndex]->detect(pp);
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool MonteCarloSimulation::peelOffScatteringBatchDirect(const vector<PhotonPacket*>& ppv)
{
    if (ppv.empty() || _config->needIndividualPeelOff() || !GpuAcceleration::isProcessEnabled()
        || !mediumSystem()->supportsSingleHenyeyGreensteinScatteringPeelOff())
        return false;

    static std::atomic<size_t> profileScatterPeelCounter{0};
    bool profileScatterPeel = isTruthyEnvironmentFlag("SKIRTGPU_PROFILE_SCATTER_PEEL");
    size_t profileScatterPeelCall = 0;
    if (profileScatterPeel)
    {
        size_t limit = environmentSizeValue("SKIRTGPU_PROFILE_SCATTER_PEEL_LIMIT", 0);
        profileScatterPeelCall = profileScatterPeelCounter.fetch_add(1, std::memory_order_relaxed);
        if (limit != 0 && profileScatterPeelCall >= limit) profileScatterPeel = false;
    }
    auto profileStart = std::chrono::steady_clock::now();
    auto profileMark = profileStart;
    double prepMs = 0.;
    double observerMs = 0.;
    double detectMs = 0.;
    auto captureProfile = [&](double& bucket) {
        if (!profileScatterPeel) return;
        auto now = std::chrono::steady_clock::now();
        bucket += elapsedMilliseconds(profileMark, now);
        profileMark = now;
    };

    const auto& instruments = _instrumentSystem->instruments();
    vector<Position> emptyPositionv;
    vector<double> emptyDoublev;
    for (Instrument* instrument : instruments)
    {
        auto frameInstrument = dynamic_cast<FrameInstrument*>(instrument);
        if (!frameInstrument) return false;
        if (!frameInstrument->detectTotalBatch(emptyPositionv, emptyDoublev, emptyDoublev, emptyDoublev))
            return false;
    }

    vector<PhotonPacket*> activePacketv;
    vector<Position> positionv;
    vector<double> lambdav;
    activePacketv.reserve(ppv.size());
    positionv.reserve(ppv.size());
    lambdav.reserve(ppv.size());
    for (PhotonPacket* pp : ppv)
    {
        if (!pp || pp->luminosity() <= 0.) continue;
        if (pp->interactionCellIndex() < 0) return false;
        activePacketv.push_back(pp);
        positionv.push_back(pp->position());
        lambdav.push_back(mediumSystem()->perceivedWavelengthForScattering(pp));
    }
    if (activePacketv.empty()) return true;
    captureProfile(prepMs);

    if (isTruthyEnvironmentFlag("SKIRTGPU_BATCH_SCATTER_PEEL_FRAME_FUSED"))
    {
        bool fusedSupported = true;
        for (Instrument* instrument : instruments)
        {
            auto frameInstrument = dynamic_cast<FrameInstrument*>(instrument);
            if (!frameInstrument || !frameInstrument->supportsHenyeyGreensteinScatteringFrameBandBatch())
            {
                fusedSupported = false;
                break;
            }
        }
        if (fusedSupported)
        {
            bool fusedDetected = true;
            for (Instrument* instrument : instruments)
            {
                auto frameInstrument = dynamic_cast<FrameInstrument*>(instrument);
                if (!frameInstrument->detectHenyeyGreensteinScatteringFrameBandBatch(
                        activePacketv, positionv, lambdav))
                {
                    fusedDetected = false;
                    break;
                }
            }
            if (fusedDetected)
            {
                captureProfile(observerMs);
                if (profileScatterPeel)
                {
                    double totalMs = elapsedMilliseconds(profileStart, std::chrono::steady_clock::now());
                    log()->info("SKIRTGPU scatter-peel direct profile call="
                                + std::to_string(profileScatterPeelCall)
                                + " packets=" + std::to_string(ppv.size())
                                + " active=" + std::to_string(activePacketv.size())
                                + " instruments=" + std::to_string(instruments.size())
                                + " observers=" + std::to_string(instruments.size())
                                + " fused_frame=1"
                                + " prep_ms=" + profileMilliseconds(prepMs)
                                + " observer_ms=" + profileMilliseconds(observerMs)
                                + " detect_ms=" + profileMilliseconds(detectMs)
                                + " total_ms=" + profileMilliseconds(totalMs));
                }
                return true;
            }
        }
    }

    vector<size_t> observerIndexForInstrument;
    vector<vector<double>> observerLuminosityv;
    vector<vector<double>> observerTauv;
    observerIndexForInstrument.reserve(instruments.size());
    observerLuminosityv.reserve(instruments.size());
    observerTauv.reserve(instruments.size());

    vector<double> distancev(activePacketv.size(), std::numeric_limits<double>::infinity());
    size_t lastObserverIndex = 0;
    for (Instrument* instrument : instruments)
    {
        if (instrument->isSameObserverAsPreceding())
        {
            if (observerLuminosityv.empty()) return false;
            observerIndexForInstrument.push_back(lastObserverIndex);
            continue;
        }

        Direction bfkobs = instrument->bfkobs(positionv.front());
        vector<double> luminosityv;
        vector<double> tauv;
        if (isTruthyEnvironmentFlag("SKIRTGPU_BATCH_SCATTER_PEEL_FUSED")
            && mediumSystem()->henyeyGreensteinScatteringObservedLuminosities(activePacketv, positionv, lambdav,
                                                                              bfkobs, distancev, luminosityv)
            && luminosityv.size() == activePacketv.size())
        {
            tauv.assign(activePacketv.size(), 0.);
        }
        else
        {
            if (!mediumSystem()->henyeyGreensteinScatteringLuminosities(activePacketv, lambdav, bfkobs, luminosityv)
                || luminosityv.size() != activePacketv.size())
                return false;

            if (!mediumSystem()->observedExtinctionOpticalDepths(positionv, bfkobs, lambdav, distancev, tauv)
                || tauv.size() != activePacketv.size())
                return false;
        }

        size_t observerIndex = observerLuminosityv.size();
        observerLuminosityv.push_back(std::move(luminosityv));
        observerTauv.push_back(std::move(tauv));
        observerIndexForInstrument.push_back(observerIndex);
        lastObserverIndex = observerIndex;
    }
    captureProfile(observerMs);

    for (size_t instrumentIndex = 0; instrumentIndex != instruments.size(); ++instrumentIndex)
    {
        size_t observerIndex = observerIndexForInstrument[instrumentIndex];
        auto frameInstrument = dynamic_cast<FrameInstrument*>(instruments[instrumentIndex]);
        if (!frameInstrument->detectTotalBatch(positionv, lambdav, observerLuminosityv[observerIndex],
                                               observerTauv[observerIndex]))
            return false;
    }
    captureProfile(detectMs);

    if (profileScatterPeel)
    {
        double totalMs = elapsedMilliseconds(profileStart, std::chrono::steady_clock::now());
        log()->info("SKIRTGPU scatter-peel direct profile call=" + std::to_string(profileScatterPeelCall)
                    + " packets=" + std::to_string(ppv.size())
                    + " active=" + std::to_string(activePacketv.size())
                    + " instruments=" + std::to_string(instruments.size())
                    + " observers=" + std::to_string(observerLuminosityv.size())
                    + " prep_ms=" + profileMilliseconds(prepMs)
                    + " observer_ms=" + profileMilliseconds(observerMs)
                    + " detect_ms=" + profileMilliseconds(detectMs)
                    + " total_ms=" + profileMilliseconds(totalMs));
    }
    return true;
}

////////////////////////////////////////////////////////////////////
