/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#include "DustSecondarySource.hpp"
#include "AngularDistributionInterface.hpp"
#include "Configuration.hpp"
#include "DisjointWavelengthGrid.hpp"
#include "FatalError.hpp"
#include "GpuAcceleration.hpp"
#include "Log.hpp"
#include "LogWavelengthDistribution.hpp"
#include "MediumSystem.hpp"
#include "MultiGrainDustMix.hpp"
#include "NR.hpp"
#include "Parallel.hpp"
#include "ParallelFactory.hpp"
#include "PhotonPacket.hpp"
#include "PolarizationProfileInterface.hpp"
#include "ProcessManager.hpp"
#include "Random.hpp"
#include "StringUtils.hpp"
#include "Units.hpp"
#include "VelocityInterface.hpp"
#include "VoronoiMeshSpatialGrid.hpp"
#include "WavelengthDistribution.hpp"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>

////////////////////////////////////////////////////////////////////

namespace
{
    std::atomic<unsigned long long> _gpuDustSourceCacheToken{1ULL << 63};

    bool deviceDustSourceLifecycleEnabled()
    {
        const char* value = std::getenv("SKIRTGPU_DEVICE_DUST_SOURCE_LIFECYCLE");
        if (!value) return true;
        string text(value);
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
        return !(text.empty() || text == "0" || text == "false" || text == "off" || text == "no");
    }

    bool dustOptionEnabled(const char* name, bool fallback = false)
    {
        const char* value = std::getenv(name);
        if (!value) return fallback;
        string text(value);
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
        return !(text.empty() || text == "0" || text == "false" || text == "off" || text == "no");
    }

    double checkDustEmissivity(const MultiGrainDustMix* mix, const double* fieldValues,
                              const double* spectrum, size_t numInput, size_t numOutput, string context)
    {
        Array field(numInput);
        std::copy_n(fieldValues, numInput, begin(field));
        Array expected = mix->emissivity(field);
        double peak = expected.max(), largest = 0.;
        for (size_t ell = 0; ell != numOutput; ++ell)
        {
            double reference = expected[ell];
            double denominator = max(std::abs(reference), peak * 1e-12);
            double error = denominator > 0. ? std::abs(spectrum[ell] - reference) / denominator
                                            : std::abs(spectrum[ell] - reference);
            if (!std::isfinite(reference) || !std::isfinite(error) || error > 1e-6)
                throw FATALERROR("GPU stochastic dust emissivity mismatch: " + context
                                 + " wavelength=" + std::to_string(ell)
                                 + " relative_error=" + StringUtils::toString(error));
            largest = max(largest, error);
        }
        return largest;
    }

    int secondaryLaunchThreadCount()
    {
        const char* text = std::getenv("SKIRTGPU_SECONDARY_LAUNCH_THREADS");
        if (!text) return 4;

        char* end = nullptr;
        unsigned long value = std::strtoul(text, &end, 10);
        if (end == text || value < 1) return 1;
        return static_cast<int>(min(value, 8UL));
    }

    Parallel* secondaryLaunchParallel(int threadCount)
    {
        thread_local std::unique_ptr<ParallelFactory> factory;
        thread_local int factoryThreadCount = 0;
        if (!factory || factoryThreadCount != threadCount)
        {
            factory = std::make_unique<ParallelFactory>();
            factory->setMaxThreadCount(threadCount);
            factoryThreadCount = threadCount;
        }
        return factory->parallelRootOnly();
    }
}

////////////////////////////////////////////////////////////////////

double DustSecondarySource::prepareLuminosities()
{
    // cache some pointers for later use
    _config = find<Configuration>();
    _ms = find<MediumSystem>();
    _random = find<Random>();

    int numCells = _ms->numCells();

    // --------- luminosities 1 ---------

    // calculate the absorbed (and thus to be emitted) dust luminosity for each spatial cell
    // this can be somewhat time-consuming, so we do this in parallel
    _Lv.resize(numCells);
    Array Labs1v;
    Array Labs2v;
    if (_ms->dustAbsorbedLuminosities(Labs1v, Labs2v))
    {
        _Lv = Labs1v + Labs2v;
    }
    else
    {
        find<ParallelFactory>()->parallelDistributed()->call(numCells, [this](size_t firstIndex, size_t numIndices) {
            for (size_t m = firstIndex; m != firstIndex + numIndices; ++m)
            {
                _Lv[m] = _ms->dustLuminosity(m);
            }
        });
        ProcessManager::sumToAll(_Lv);
    }

    // --------- library mapping ---------

    // obtain the spatial cell library mapping (from cell indices to library entry indices);
    // pass the cell luminosities to the library so it can avoid mapping zero-luminosity cells
    _nv = _config->cellLibrary()->mapping(_Lv);

    // construct a list of spatial cell indices sorted so that cells belonging to the same entry are consecutive
    _mv.resize(numCells);
    for (int m = 0; m != numCells; ++m) _mv[m] = m;
    std::sort(begin(_mv), end(_mv), [this](int m1, int m2) { return _nv[m1] < _nv[m2]; });

    // --------- luminosities 2 ---------

    // force cells that have not been mapped by the library to zero luminosity
    // so that we won't launch photon packets for these cells
    for (int m = 0; m != numCells; ++m)
        if (_nv[m] < 0) _Lv[m] = 0.;

    // calculate  the total luminosity, and normalize the individual luminosities to unity
    double L = _Lv.sum();
    if (!GpuAcceleration::divideValues(&_Lv[0], numCells, L)) _Lv /= L;

    // --------- logging ---------

    auto log = find<Log>();
    auto units = find<Units>();

    // luminosity and spatial cells
    int emittingCells = 0;  // number of nonzero luminosity cells
    for (int m = 0; m != numCells; ++m)
        if (_Lv[m] > 0.) emittingCells++;
    log->info("Dust luminosity: " + StringUtils::toString(units->obolluminosity(L), 'g') + " "
              + units->ubolluminosity());
    log->info("  Emitting from " + std::to_string(emittingCells) + " out of " + std::to_string(numCells)
              + " spatial cells");

    // library entries
    int numEntries = _config->cellLibrary()->numEntries();
    vector<int> mapped(numEntries);  // number of cells mapped to each library entry
    for (int m = 0; m != numCells; ++m)
        if (_nv[m] >= 0) mapped[_nv[m]]++;

    int usedEntries = 0;     // number of library entries that have at least one mapped cell
    int maxMappedCells = 0;  // largest number of cells mapped to a library entry
    int totMappedCells = 0;  // total number of cells mapped to a library entry
    for (int n = 0; n != numEntries; ++n)
    {
        if (mapped[n] > 0)
        {
            usedEntries++;
            maxMappedCells = max(mapped[n], maxMappedCells);
            totMappedCells += mapped[n];
        }
    }

    if (numEntries != numCells || totMappedCells != numCells || numEntries != usedEntries || maxMappedCells != 1)
    {
        log->info("  Using " + std::to_string(usedEntries) + " out of " + std::to_string(numEntries)
                  + " library entries");
        log->info("  Largest number of cells per library entry: " + std::to_string(maxMappedCells));
        log->info("  Average number of cells per (used) library entry: "
                  + StringUtils::toString(static_cast<double>(totMappedCells) / usedEntries, 'f', 1));
    }

    // return the total luminosity
    return L;
}

////////////////////////////////////////////////////////////////////

void DustSecondarySource::preparePacketMap(size_t firstIndex, size_t numIndices)
{
    int numCells = _ms->numCells();

    // --------- weights ---------

    // calculate the final, composite-biased launch weight for each cell, normalized to unity
    double xi = _config->secondarySpatialBias();
    _Wv.resize(numCells);
    if (!GpuAcceleration::compositeLaunchWeights(&_Lv[0], numCells, xi, &_Wv[0]))
    {
        // determine a uniform weight for each cell with non-negligable emission, and normalize to unity
        Array wv(numCells);
        for (int m = 0; m != numCells; ++m) wv[m] = _Lv[m] > 0 ? 1. : 0.;
        wv /= wv.sum();
        _Wv = (1 - xi) * _Lv + xi * wv;
    }

    // determine the first history index for each cell, using the adjusted cell ordering so that
    // all photon packets for a given library entry are launched consecutively
    _Iv.resize(numCells + 1);
    _Iv[0] = firstIndex;
    double W = 0.;
    for (int p = 1; p != numCells; ++p)
    {
        // track the cumulative normalized weight as a floating point number
        // and limit the index to firstIndex+numIndices to avoid issues with rounding errors
        W += _Wv[_mv[p - 1]];
        _Iv[p] = firstIndex + min(numIndices, static_cast<size_t>(std::round(W * numIndices)));
    }
    _Iv[numCells] = firstIndex + numIndices;
    _hasGpuSourceLaunchData = prepareGpuSourceLaunchData();
}

////////////////////////////////////////////////////////////////////

namespace
{
    // An instance of this class obtains and/or calculates the information needed to launch photon packets
    // from the dust in a given cell in the spatial grid, and remembers the information for fast retrieval.
    // This information includes the normalized regular and cumulative dust emission spectra, calculated from
    // the stored radiation field and the dust properties using the configured emission calculator, and
    // the average bulk velocity of the material in the cell, obtained from the medium system.
    class DustCellEmission : public VelocityInterface
    {
    private:
        // information initialized once, during the first call to calculateIfNeeded()
        MediumSystem* _ms{nullptr};  // the medium system
        Array _wavelengthGrid;       // the dust emission wavelength grid
        Range _wavelengthRange;      // the range of the dust emission wavelength grid
        int _numWavelengths{0};      // the number of wavelengths in the dust emission wavelength grid
        vector<int> _hv;             // a list of the media indices for the media containing dust
        int _numMedia{0};            // the number of dust media in the system (and thus the size of hv)
        int _numCells{0};            // the number of cells in the spatial grid (and thus the size of mv and nv)

        // information on a particular spatial cell, initialized by calculateIfNeeded()
        int _p{-1};                // spatial cell launch-order index
        int _n{-1};                // library entry index
        vector<Array> _evv;        // emissivity spectrum for each medium component, if applicable
        Array _lambdav, _pv, _Pv;  // normalized emission spectrum
        Vec _bfv;                  // bulk velocity
        unsigned long long _preparedToken{0};
        size_t _preparedGeneration{0};
        int _preparedP{-1};

    public:
        // instances of this class are allocated in thread-local storage, which means that the constructor is
        // invoked under the hood for each new execution thread; hence it does trivial initialization only
        DustCellEmission() {}

        // Reuse the already validated batch spectrum during host photon launch (notably
        // secondary iterations without peel-off). Recomputing stochastic heating per cell
        // here defeats the batch preparation. Cache identity includes the iteration generation.
        bool usePreparedSpectrum(int p, const GpuSourceLaunchData& data)
        {
            size_t count = data.wavelengthv.size();
            int row = data.spectrumRowIndexv.empty() ? p : data.spectrumRowIndexv[p];
            if (row < 0 || count < 2 || data.pdfv.size() < (static_cast<size_t>(row) + 1) * count)
                return false;
            if (_preparedToken == data.cacheToken && _preparedGeneration == data.generation && _preparedP == p)
                return true;
            Array wavelengths(count), emission(count);
            for (size_t ell = 0; ell != count; ++ell)
            {
                wavelengths[ell] = data.wavelengthv[ell];
                emission[ell] = data.pdfv[static_cast<size_t>(row) * count + ell];
            }
            NR::cdf<NR::interpolateLogLog>(_lambdav, _pv, _Pv, wavelengths, emission,
                                          Range(wavelengths[0], wavelengths[count - 1]));
            _bfv = Vec();  // prepared dust launch is restricted to stationary media
            _ms = nullptr; // invalidate scalar/library state if a later launch needs fallback
            _p = _n = -1;
            _preparedToken = data.cacheToken;
            _preparedGeneration = data.generation;
            _preparedP = p;
            return true;
        }

        // calculates the emission information for the given cell if it is different from what's already stored
        //   p:  launch-order cell index (cells mapped to a given library entry have consecutive p indices)
        //   mv: map from launch-order cell index p to regular cell index m
        //   nv: map from regular cell index m to library entry index n
        //   ms: medium system
        //   config: configuration object
        void calculateIfNeeded(int p, const vector<int>& mv, const vector<int>& nv, MediumSystem* ms,
                               Configuration* config)
        {
            _preparedP = -1;  // scalar fallback replaces any previously prepared spectrum
            // when called for the first time for a given simulation, cache some info
            if (_ms != ms)
            {
                _p = -1;
                _n = -1;
                _ms = ms;
                auto wavelengthGrid = config->dustEmissionWLG();
                _wavelengthGrid = wavelengthGrid->extlambdav();
                _wavelengthRange = wavelengthGrid->wavelengthRange();
                _numWavelengths = _wavelengthGrid.size();
                _hv = ms->dustMediumIndices();
                _numMedia = _hv.size();
                _numCells = ms->numCells();
                _evv.resize(ms->numMedia());
            }

            // if this photon packet is launched from the same cell as the previous one, we don't need to do anything
            if (p == _p) return;

            // remember the new cell index and map to the other indices
            _p = p;
            int m = mv[p];
            int n = nv[m];

            // if this new cell maps to a new library entry, we need to process the library entry
            if (n != _n)
            {
                // remember the new library entry index
                _n = n;

                // determine the number of cells mapped to this library entry (they are consecutive in p)
                int pp = p + 1;
                for (; pp != _numCells; ++pp)
                    if (nv[mv[pp]] != n) break;
                int numMappedCells = pp - p;

                // if only a single cell maps to the library entry, we can simply calculate its emission
                if (numMappedCells == 1)
                {
                    calculateSingleSpectrum(m);
                }

                // if multiple cells map to the library entry, we use the average radiation field for these cells
                else
                {
                    Array Jv = _ms->meanIntensity(m);
                    for (int i = 1; i != numMappedCells; ++i) Jv += _ms->meanIntensity(mv[p + i]);
                    Jv /= numMappedCells;

                    // if there is a single dust medium (and assuming that there are no variable dust mixes),
                    // we can use a single emission spectrum for all cells mapped to the library entry, calculated
                    // using the average radiation field, because the cells differ only in dust density, which is
                    // irrelevant because the emission spectrum is normalized anyway
                    if (_numMedia == 1)
                    {
                        calculateSingleSpectrum(Jv, m);
                    }

                    // otherwise, we need to calculate and remember the emission spectrum for each medium component
                    // (still using the average radiation field and assuming that there are no variable dust mixes)
                    // so that we can apply the relative density weights for each cell later on
                    else
                    {
                        calculateEmissivityPerMedium(Jv, m);
                        calculateWeightedSpectrum(m);
                    }
                }
            }

            // if this new cell maps to the same library entry, we can use its cached info
            else
            {
                // if there is a single dust medium we can use the already calculated emission spectrum;
                // otherwise, we need to apply the relative density weights for this cell to the calculated
                // emission spectra for each dust medium, and renormalize the resulting spectrum
                if (_numMedia != 1)
                {
                    calculateWeightedSpectrum(m);
                }
            }

            // remember the average bulk velocity for this cell
            _bfv = ms->bulkVelocity(m);
        }

    private:
        // calculate the emission spectrum for the dust mixes of the specified cell,
        // and store the result in the data members _lambdav, _pv, _Pv
        void calculateSingleSpectrum(int m)
        {
            // get the emmissivity spectrum for all dust medium components in the cell
            const Array& ev = _ms->dustEmissionSpectrum(m);

            // calculate the normalized plain and cumulative distributions
            NR::cdf<NR::interpolateLogLog>(_lambdav, _pv, _Pv, _wavelengthGrid, ev, _wavelengthRange);
        }

        // calculate the emission spectrum for the specified radiation field and the dust mixes of the specified cell,
        // and store the result in the data members _lambdav, _pv, _Pv
        void calculateSingleSpectrum(const Array& Jv, int m)
        {
            // accumulate the emmissivity spectrum for all dust medium components in the cell, weighted by density
            Array ev(_numWavelengths);
            for (int h : _hv) ev += _ms->numberDensity(m, h) * _ms->mix(m, h)->emissivity(Jv);

            // calculate the normalized plain and cumulative distributions
            NR::cdf<NR::interpolateLogLog>(_lambdav, _pv, _Pv, _wavelengthGrid, ev, _wavelengthRange);
        }

        // calculate the emmissivity spectra for the specified radiation field and the dust mixes of the specified cell,
        // and store the individual spectra in the data members _evv
        void calculateEmissivityPerMedium(const Array& Jv, int m)
        {
            for (int h : _hv) _evv[h] = _ms->mix(m, h)->emissivity(Jv);
        }

        // calculate the emission spectrum for the specified cell, weighted across multiple media by density,
        // given the precalculated emissivity spectra _evv for each medium,
        // and store the result in the data members _lambdav, _pv, _Pv
        void calculateWeightedSpectrum(int m)
        {
            // accumulate the emmissivity spectrum for all dust medium components in the cell, weighed by density
            Array ev(_numWavelengths);
            for (int h : _hv) ev += _ms->numberDensity(m, h) * _evv[h];

            // calculate the normalized plain and cumulative distributions
            NR::cdf<NR::interpolateLogLog>(_lambdav, _pv, _Pv, _wavelengthGrid, ev, _wavelengthRange);
        }

    public:
        // returns a random wavelength generated from the spectral distribution
        double generateWavelength(Random* random) const { return random->cdfLogLog(_lambdav, _pv, _Pv); }

        // returns the normalized specific luminosity for the given wavelength
        double specificLuminosity(double lambda) const
        {
            return NR::value<NR::interpolateLogLog>(lambda, _lambdav, _pv);
        }

        const Array& wavelengths() const { return _lambdav; }
        const Array& pdf() const { return _pv; }
        const Array& cdf() const { return _Pv; }

        Vec velocity() const override { return _bfv; }
    };

    // An instance of this class obtains the information needed to determine angular distribution probabilities
    // and polarization components for photon packets emitted from the dust in a given cell in the spatial grid,
    // and remembers some information for fast retrieval. It is actually used only when there are spheroidal
    // dust grains in the simulation.
    class DustCellPolarisedEmission : public AngularDistributionInterface, public PolarizationProfileInterface
    {
    private:
        // information initialized once, during the first call to calculateIfNeeded()
        MediumSystem* _ms{nullptr};  // the medium system
        vector<int> _hv;             // a list of the media indices for the media containing dust

        // information on a particular spatial cell, initialized by calculateIfNeeded()
        int _m{-1};              // spatial cell index
        Direction _B_direction;  // direction of the magnetic field

        // information on a particular photon packet, initialized by calculateIfNeeded()
        double _lambda{-1.};  // wavelength of the photon packet

    public:
        // instances of this class are allocated in thread-local storage, which means that the constructor is
        // invoked under the hood for each new execution thread; hence it does trivial initialization only
        DustCellPolarisedEmission() {}

        // calculates the emission information for the given cell and photon packet wavelength
        // if it is different from what's already stored
        void calculateIfNeeded(int m, MediumSystem* ms, double lambda)
        {
            // always remember the photon packet wavelength
            _lambda = lambda;

            // if this packet is launched from the same cell as the previous one, we don't need to do anything else
            if (m == _m) return;

            // when called for the first time, cache some info
            if (_m == -1)
            {
                _ms = ms;
                _hv = ms->dustMediumIndices();
            }

            // remember the new cell index and map to the other indices
            _m = m;

            // normalise the magnetic field direction
            _B_direction.set(_ms->magneticField(_m), true);
        }

        // this AngularDistributionInterface implementation returns the probability for the given direction
        double probabilityForDirection(Direction bfk) const override
        {
            // get the zenith angle of the direction
            // this is the angle between the magnetic field direction (z axis of the reference frame)
            // and the direction
            const double theta = std::acos(Vec::dot(_B_direction, bfk));

            // compute
            //   - Qabstheta: the absorption coefficient for the direction
            //   - Qabstot: the angular averaged absorption coefficient
            double Qabstheta = 0.;
            double Qabstot = 0.;
            // we need to sum over all dust species
            for (int h : _hv)
            {
                const Array& thetas = _ms->mix(_m, h)->thetaGrid();
                const Array& Qabs = _ms->mix(_m, h)->sectionsAbs(_lambda);
                Qabstheta += _ms->numberDensity(_m, h) * NR::value<NR::interpolateLinLin>(theta, thetas, Qabs);
                Qabstot += _ms->numberDensity(_m, h) * _ms->mix(_m, h)->sectionAbs(_lambda);
            }
            // the probability for this direction is the ratio of the two values
            return Qabstheta / Qabstot;
        }

        // generate a random direction based on the anisotropic emission profile
        Direction generateDirection(Random* random) const
        {
            // first generate a random azimuth angle
            const double phi = 2. * M_PI * random->uniform();

            // now generate a random zenith angle
            // first, compute the cumulative distribution for a photon at the given wavelength
            // create empty PDF and CDF arrays with the same size as the zenith angle grid
            const Array& thetas = _ms->mix(_m, _hv[0])->thetaGrid();
            Array pdf(thetas.size()), cdf(thetas.size());
            // add the distributions for the different components
            for (int h : _hv)
            {
                const Array& Qabs = _ms->mix(_m, h)->sectionsAbs(_lambda);
                pdf += _ms->numberDensity(_m, h) * Qabs;
            }
            // the CDF is computed from the PDF by integration
            // since this is an integration is spherical coordinates, we need to
            // multiply the PDF with the volume element sin(theta)
            for (size_t i = 0; i < thetas.size(); ++i)
            {
                pdf[i] *= std::sin(thetas[i]);
            }
            // convert to the CDF
            // note that we use the midpoint of each bin as integration point
            // and omit the common factors from the integration
            for (size_t i = 1; i < thetas.size(); ++i)
            {
                cdf[i] = cdf[i - 1] + pdf[i - 1] + pdf[i];
            }
            // normalise the CDF
            for (size_t i = 0; i < thetas.size(); ++i)
            {
                cdf[i] /= cdf[thetas.size() - 1];
            }
            // draw a random zenith angle from the CDF
            const double theta = random->cdfLinLin(thetas, cdf);

            // generate a random direction
            const Direction krand(theta, phi);

            // The random direction is defined in a frame that has the B direction
            // as z-axis (the x and y axis can be chosen arbitrarily because of the
            // symmetry in phi)
            // We need to (passively) rotate the random direction to the lab frame
            // where the z-axis is the actual z-axis
            // This corresponds to a rotation over -beta degrees along an axis
            // that is perpendicular to the plane through the B direction and the
            // z-axis, where beta is the angle between the B direction and the
            // z-axis
            // Note that in this case, it does not matter whether we rotate over
            // beta or -beta degrees, since the difference between these two is
            // simply a rotation over 180 degrees in phi, and we still sample a
            // uniform distribution in phi.
            // If the angle is small enough, we simply assume that B already lies
            // along the z-axis and return immediately.
            const double cosbeta = _B_direction.z();
            if (fabs(cosbeta) < 0.99999)
            {
                // B is not the z-axis. Compute beta and sin(beta)
                const double beta = std::acos(cosbeta);
                const double sinbeta = std::sin(beta);
                // compute the direction of the rotation axis, z x B
                const Direction n(Vec::cross(Vec(0., 0., 1.), _B_direction), true);
                // now use Rodrigues' rotation formula to carry out the rotation
                return Direction(
                    krand * cosbeta + Vec::cross(n, krand) * sinbeta + n * Vec::dot(n, krand) * (1. - cosbeta), false);
            }
            else
            {
                return krand;
            }
        }

        // this PolarizationProfileInterface implementation returns the Stokes vector for the given emission direction
        StokesVector polarizationForDirection(Direction bfk) const override
        {
            // get the zenith angle of the direction
            // this is the angle between the magnetic field direction (z axis of the reference frame)
            // and the direction
            const double theta = std::acos(Vec::dot(_B_direction, bfk));

            // we sum over all dust species
            double Qabsval = 0.;
            double Qabspolval = 0.;
            for (int h : _hv)
            {
                const Array& thetas = _ms->mix(_m, h)->thetaGrid();
                const Array& Qabs = _ms->mix(_m, h)->sectionsAbs(_lambda);
                const Array& Qabspol = _ms->mix(_m, h)->sectionsAbspol(_lambda);
                Qabsval += _ms->numberDensity(_m, h) * NR::value<NR::interpolateLinLin>(theta, thetas, Qabs);
                Qabspolval += _ms->numberDensity(_m, h) * NR::value<NR::interpolateLinLin>(theta, thetas, Qabspol);
            }
            // the reference direction is any direction perpendicular to the
            // propagation direction and the magnetic field direction
            return StokesVector(Qabsval, Qabspolval, 0., 0., Direction(Vec::cross(bfk, _B_direction), true));
        }
    };

    // setup instances of the above classes to cache dust emission information for each parallel execution thread
    thread_local DustCellEmission t_dustcell;
    thread_local DustCellPolarisedEmission t_dustcellpol;
}

////////////////////////////////////////////////////////////////////

bool DustSecondarySource::prepareGpuSourceLaunchData()
{
    if (!deviceDustSourceLifecycleEnabled() || !GpuAcceleration::isProcessEnabled() || ProcessManager::size() != 1
        || _config->hasMovingMedia()
        || _config->hasSpheroidalPolarization() || _ms->dustMediumIndices().size() != 1)
        return false;

    auto grid = dynamic_cast<const VoronoiMeshSpatialGrid*>(_ms->grid());
    auto logBias = _config->dustEmissionWavelengthBias()
                       ? dynamic_cast<const LogWavelengthDistribution*>(
                             _config->dustEmissionWavelengthBiasDistribution())
                       : nullptr;
    int numCells = _ms->numCells();
    if (!grid || numCells <= 0 || (_config->dustEmissionWavelengthBias() && !logBias)
        || _config->cellLibrary()->numEntries() != numCells || _Iv.size() != static_cast<size_t>(numCells + 1))
        return false;
    vector<bool> usedEntries(numCells, false);
    for (int m = 0; m != numCells; ++m)
    {
        int n = _nv[m];
        if (n < 0 || n >= numCells || usedEntries[n]) return false;
        usedEntries[n] = true;
    }

    vector<size_t> active;
    for (size_t p = 0; p != static_cast<size_t>(numCells); ++p)
        if (_Iv[p] != _Iv[p + 1]) active.push_back(p);
    if (active.empty()) return false;
    size_t firstActive = active.front();
    bool compact = dustOptionEnabled("SKIRTGPU_COMPACT_DUST_SPECTRA");
    size_t numSpectrumRows = compact ? active.size() : static_cast<size_t>(numCells);

    DustCellEmission firstEmission;
    firstEmission.calculateIfNeeded(static_cast<int>(firstActive), _mv, _nv, _ms, _config);
    size_t numWavelengths = firstEmission.wavelengths().size();
    if (numWavelengths < 2 || firstEmission.pdf().size() != numWavelengths
        || firstEmission.cdf().size() != numWavelengths
        || numSpectrumRows > std::numeric_limits<size_t>::max() / numWavelengths)
        return false;

    GpuSourceLaunchData data;
    data.cacheToken = _gpuSourceLaunchData.cacheToken
                          ? _gpuSourceLaunchData.cacheToken
                          : _gpuDustSourceCacheToken.fetch_add(1, std::memory_order_relaxed);
    data.generation = _gpuSourceLaunchData.generation + 1;
    data.historyBeginv.resize(_Iv.size());
    for (size_t p = 0; p != _Iv.size(); ++p)
        data.historyBeginv[p] = static_cast<unsigned long long>(_Iv[p]);
    data.luminosityWeightv.resize(numCells);
    data.cellIndexv.resize(numCells);
    data.cellExtentv.resize(6 * static_cast<size_t>(numCells));
    data.wavelengthv.resize(numWavelengths);
    for (size_t ell = 0; ell != numWavelengths; ++ell)
        data.wavelengthv[ell] = firstEmission.wavelengths()[ell];
    if (compact)
    {
        data.spectrumRowIndexv.assign(numCells, -1);
        for (size_t row = 0; row != active.size(); ++row)
            data.spectrumRowIndexv[active[row]] = static_cast<int>(row);
    }
    data.pdfv.assign(numSpectrumRows * numWavelengths, 0.f);
    data.cdfv.assign(data.pdfv.size(), 0.f);

    // Keep every cell: transport column-density bounds also use non-emitting absorbers.
    vector<double> regularCellExtents = grid->gpuCellExtentCoordinates();
    if (regularCellExtents.size() != 6 * static_cast<size_t>(numCells)) return false;
    for (int p = 0; p != numCells; ++p)
    {
        int m = _mv[p];
        data.cellIndexv[p] = m;
        data.luminosityWeightv[p] = _Wv[m] > 0. ? _Lv[m] / _Wv[m] : 0.;
        std::copy_n(regularCellExtents.begin() + 6 * static_cast<size_t>(m), 6,
                    data.cellExtentv.begin() + 6 * static_cast<size_t>(p));
    }

    std::atomic<bool> valid{true};
    int medium = _ms->dustMediumIndices().front();
    auto gpuMix = dynamic_cast<const MultiGrainDustMix*>(_ms->mix(_mv[firstActive], medium));
    if (gpuMix && _config->hasStochasticDustEmission() && dustOptionEnabled("SKIRTGPU_STOCHASTIC_DUST", true))
    {
        // The warp-per-cell kernel needs no per-cell scratch matrix, so device memory is only the fields and
        // spectra (about 11 kB per cell); thousands of cells per launch are needed to fill a GPU. About sixteen
        // batches per pass keep the GPUs evenly loaded (7 batches of 16384 left one GPU a batch behind).
        size_t batchSize = min(size_t(16384), max(size_t(2048), (active.size() + 15) / 16));
        if (const char* text = std::getenv("SKIRTGPU_STOCHASTIC_DUST_BATCH_CELLS"))
        {
            char* end = nullptr;
            unsigned long requested = std::strtoul(text, &end, 10);
            if (end == text || *end || requested < 1 || requested > 1048576)
                throw FATALERROR("SKIRTGPU_STOCHASTIC_DUST_BATCH_CELLS must be between 1 and 1048576");
            batchSize = requested;
        }
        size_t numBatches = (active.size() + batchSize - 1) / batchSize;
        bool verify = dustOptionEnabled("SKIRTGPU_VALIDATE_DUST_EMISSION");
        std::atomic<size_t> gpuCells{0}, fallbackCells{0}, checkedCells{0};
        double largestError = 0.;
        std::mutex errorMutex;
        auto wavelengths = _config->dustEmissionWLG()->extlambdav();
        auto range = _config->dustEmissionWLG()->wavelengthRange();
        size_t numInput = _config->radiationFieldWLG()->numBins();
        size_t numOutput = wavelengths.size();
        if (verify)
        {
            const vector<double> scales{0., 1e-8, 1e-4, 1., 1e4, 1e8};
            Array reference = _ms->meanIntensity(_mv[firstActive]);
            vector<double> fields(scales.size() * numInput), spectra;
            for (size_t i = 0; i != scales.size(); ++i)
                for (size_t k = 0; k != numInput; ++k) fields[i * numInput + k] = scales[i] * reference[k];
            if (!gpuMix->emissivityBatch(fields, scales.size(), spectra))
                throw FATALERROR("GPU dust stress validation failed: " + GpuAcceleration::lastError());
            double error = 0.;
            for (size_t i = 0; i != scales.size(); ++i)
                error = max(error, checkDustEmissivity(gpuMix, fields.data() + i * numInput,
                                                      spectra.data() + i * numOutput, numInput, numOutput,
                                                      "radiation scale=" + StringUtils::toString(scales[i])));
            find<Log>()->info("GPU dust zero-field and heating-scale checks passed; max normalized error="
                              + StringUtils::toString(error));
        }
        find<Log>()->info("Preparing stochastic dust spectra in GPU batches of at most "
                          + std::to_string(batchSize) + " cells");
        find<Log>()->infoSetElapsed(active.size());
        find<ParallelFactory>()->parallelRootOnly()->call(
            numBatches, [&](size_t firstBatch, size_t count) {
                DustCellEmission scalar;
                for (size_t batch = firstBatch; batch != firstBatch + count; ++batch)
                {
                    size_t beginCell = batch * batchSize;
                    size_t countCells = min(batchSize, active.size() - beginCell);
                    vector<double> fields(countCells * numInput), spectra;
                    bool sameMix = true;
                    for (size_t i = 0; i != countCells; ++i)
                    {
                        int m = _mv[active[beginCell + i]];
                        sameMix = sameMix && _ms->mix(m, medium) == gpuMix;
                        Array field = _ms->meanIntensity(m);
                        std::copy(begin(field), end(field), fields.begin() + i * numInput);
                    }
                    bool accelerated = sameMix && gpuMix->emissivityBatch(fields, countCells, spectra);
                    if (!accelerated && verify)
                        throw FATALERROR("Stochastic dust GPU validation could not run: " + GpuAcceleration::lastError());
                    if (accelerated && verify)
                    {
                        // Cover the full launch ordering without repeating every expensive scalar calculation.
                        for (size_t i = 0; i < countCells; i += max(size_t(1), countCells - 1))
                        {
                            double error = checkDustEmissivity(gpuMix, fields.data() + i * numInput,
                                                               spectra.data() + i * numOutput, numInput, numOutput,
                                                               "cell=" + std::to_string(_mv[active[beginCell + i]]));
                            std::lock_guard<std::mutex> lock(errorMutex);
                            largestError = max(largestError, error);
                            ++checkedCells;
                        }
                    }
                    if (accelerated) gpuCells += countCells;
                    else fallbackCells += countCells;
                    for (size_t i = 0; i != countCells; ++i)
                    {
                        size_t p = active[beginCell + i];
                        size_t row = compact ? beginCell + i : p;
                        Array lambdav, pdf, cdf;
                        if (accelerated)
                        {
                            Array emission(numOutput);
                            double density = _ms->numberDensity(_mv[p], medium);
                            for (size_t ell = 0; ell != numOutput; ++ell)
                                emission[ell] = density * spectra[i * numOutput + ell];
                            NR::cdf<NR::interpolateLogLog>(lambdav, pdf, cdf, wavelengths, emission, range);
                        }
                        else
                        {
                            scalar.calculateIfNeeded(static_cast<int>(p), _mv, _nv, _ms, _config);
                            lambdav = scalar.wavelengths();
                            pdf = scalar.pdf();
                            cdf = scalar.cdf();
                        }
                        if (lambdav.size() != numWavelengths || pdf.size() != numWavelengths
                            || cdf.size() != numWavelengths)
                        {
                            valid.store(false);
                            continue;
                        }
                        for (size_t ell = 0; ell != numWavelengths; ++ell)
                        {
                            data.pdfv[row * numWavelengths + ell] = static_cast<float>(pdf[ell]);
                            data.cdfv[row * numWavelengths + ell] = static_cast<float>(cdf[ell]);
                        }
                    }
                    find<Log>()->infoIfElapsed("Prepared stochastic dust spectra: ", countCells);
                }
            });
        find<Log>()->info("Stochastic dust spectra: GPU cells=" + std::to_string(gpuCells.load())
                          + ", CPU fallback cells=" + std::to_string(fallbackCells.load())
                          + ", CPU-checked cells=" + std::to_string(checkedCells.load())
                          + ", max normalized error=" + StringUtils::toString(largestError));
    }
    else find<ParallelFactory>()->parallelRootOnly()->call(
        active.size(),
        [this, numWavelengths, compact, &active, &data, &valid](size_t firstActiveIndex, size_t count) {
            DustCellEmission emission;
            for (size_t a = firstActiveIndex; a != firstActiveIndex + count; ++a)
            {
                size_t p = active[a];
                emission.calculateIfNeeded(static_cast<int>(p), _mv, _nv, _ms, _config);
                if (emission.wavelengths().size() != numWavelengths || emission.pdf().size() != numWavelengths
                    || emission.cdf().size() != numWavelengths)
                {
                    valid.store(false, std::memory_order_relaxed);
                    continue;
                }
                size_t offset = (compact ? a : p) * numWavelengths;
                for (size_t ell = 0; ell != numWavelengths; ++ell)
                {
                    data.pdfv[offset + ell] = static_cast<float>(emission.pdf()[ell]);
                    data.cdfv[offset + ell] = static_cast<float>(emission.cdf()[ell]);
                }
            }
        });
    if (!valid.load(std::memory_order_relaxed) || !data.hasValidSpectrumLayout()) return false;

    data.wavelengthBias = _config->dustEmissionWavelengthBias();
    if (logBias)
    {
        double biasMin = max(logBias->minWavelength(), data.wavelengthv.front());
        double biasMax = min(logBias->maxWavelength(), data.wavelengthv.back());
        if (!(biasMin > 0.) || !(biasMax > biasMin)) return false;
        data.biasLogMin = std::log(biasMin);
        data.biasLogWidth = std::log(biasMax) - std::log(biasMin);
    }

    _gpuSourceLaunchData = std::move(data);
    find<Log>()->info("GPU dust spectral storage: rows=" + std::to_string(numSpectrumRows) + "/"
                      + std::to_string(numCells) + ", PDF/CDF bytes="
                      + std::to_string(2 * _gpuSourceLaunchData.pdfv.size() * sizeof(float))
                      + ", row-index bytes="
                      + std::to_string(_gpuSourceLaunchData.spectrumRowIndexv.size() * sizeof(int)));
    find<Log>()->info("Prepared direct GPU dust-source spectra with " + std::to_string(numWavelengths)
                      + " wavelength points per cell");
    return true;
}

////////////////////////////////////////////////////////////////////

const GpuSourceLaunchData* DustSecondarySource::gpuSourceLaunchData() const
{
    return _hasGpuSourceLaunchData ? &_gpuSourceLaunchData : nullptr;
}

////////////////////////////////////////////////////////////////////

void DustSecondarySource::launch(PhotonPacket* pp, size_t historyIndex, double L) const
{
    // select the spatial cell from which to launch based on the history index of this photon packet
    auto p = std::upper_bound(_Iv.cbegin(), _Iv.cend(), historyIndex) - _Iv.cbegin() - 1;
    auto m = _mv[p];

    launchFromCell(pp, historyIndex, L, static_cast<int>(p), m);
}

////////////////////////////////////////////////////////////////////

void DustSecondarySource::launchBatch(PhotonPacket* ppv, size_t firstHistoryIndex, size_t numPackets, double L) const
{
    if (!numPackets) return;

    size_t endHistoryIndex = firstHistoryIndex + numPackets;

    int launchThreads = secondaryLaunchThreadCount();
    if (launchThreads > 1 && GpuAcceleration::isProcessEnabled() && numPackets >= 65536)
    {
        size_t firstCellOrderIndex =
            std::upper_bound(_Iv.cbegin(), _Iv.cend(), firstHistoryIndex) - _Iv.cbegin() - 1;
        size_t lastCellOrderIndex =
            std::upper_bound(_Iv.cbegin(), _Iv.cend(), endHistoryIndex - 1) - _Iv.cbegin() - 1;
        size_t numCellOrderIndices = lastCellOrderIndex - firstCellOrderIndex + 1;

        secondaryLaunchParallel(launchThreads)->call(
            numCellOrderIndices,
            [this, ppv, firstHistoryIndex, endHistoryIndex, firstCellOrderIndex, L](size_t firstCellOffset,
                                                                                 size_t numCellOffsets) {
                size_t endCellOffset = firstCellOffset + numCellOffsets;
                for (size_t cellOffset = firstCellOffset; cellOffset != endCellOffset; ++cellOffset)
                {
                    size_t p = firstCellOrderIndex + cellOffset;
                    size_t cellBegin = max(firstHistoryIndex, _Iv[p]);
                    size_t cellEnd = min(endHistoryIndex, _Iv[p + 1]);
                    int cellOrderIndex = static_cast<int>(p);
                    int m = _mv[p];
                    for (size_t historyIndex = cellBegin; historyIndex != cellEnd; ++historyIndex)
                    {
                        launchFromCell(&ppv[historyIndex - firstHistoryIndex], historyIndex, L, cellOrderIndex, m);
                    }
                }
            });
        return;
    }

    size_t offset = 0;
    while (offset != numPackets)
    {
        size_t historyIndex = firstHistoryIndex + offset;
        auto p = std::upper_bound(_Iv.cbegin(), _Iv.cend(), historyIndex) - _Iv.cbegin() - 1;
        size_t cellEnd = min(endHistoryIndex, _Iv[p + 1]);
        size_t count = cellEnd - historyIndex;
        int cellOrderIndex = static_cast<int>(p);
        int m = _mv[p];

        for (size_t i = 0; i != count; ++i)
            launchFromCell(&ppv[offset + i], historyIndex + i, L, cellOrderIndex, m);
        offset += count;
    }
}

////////////////////////////////////////////////////////////////////

bool DustSecondarySource::launchBatchData(vector<double>& packetData, size_t firstHistoryIndex, size_t numPackets,
                                          double L) const
{
    packetData.clear();
    if (!numPackets || _config->hasMovingMedia() || _config->hasSpheroidalPolarization()
        || numPackets > std::numeric_limits<size_t>::max() / 8)
        return false;

    packetData.resize(8 * numPackets);
    size_t endHistoryIndex = firstHistoryIndex + numPackets;
    auto launchCellRange = [this, &packetData, firstHistoryIndex, endHistoryIndex, L](size_t firstCellOrderIndex,
                                                                                    size_t endCellOrderIndex) {
        PhotonPacket packet;
        for (size_t p = firstCellOrderIndex; p != endCellOrderIndex; ++p)
        {
            size_t cellBegin = max(firstHistoryIndex, _Iv[p]);
            size_t cellEnd = min(endHistoryIndex, _Iv[p + 1]);
            int cellOrderIndex = static_cast<int>(p);
            int m = _mv[p];
            for (size_t historyIndex = cellBegin; historyIndex != cellEnd; ++historyIndex)
            {
                launchFromCell(&packet, historyIndex, L, cellOrderIndex, m);
                size_t offset = 8 * (historyIndex - firstHistoryIndex);
                packet.position().cartesian(packetData[offset], packetData[offset + 1], packetData[offset + 2]);
                packet.direction().cartesian(packetData[offset + 3], packetData[offset + 4],
                                             packetData[offset + 5]);
                packetData[offset + 6] = packet.wavelength();
                packetData[offset + 7] = packet.luminosity();
            }
        }
    };

    size_t firstCellOrderIndex =
        std::upper_bound(_Iv.cbegin(), _Iv.cend(), firstHistoryIndex) - _Iv.cbegin() - 1;
    size_t lastCellOrderIndex =
        std::upper_bound(_Iv.cbegin(), _Iv.cend(), endHistoryIndex - 1) - _Iv.cbegin() - 1;
    size_t numCellOrderIndices = lastCellOrderIndex - firstCellOrderIndex + 1;
    int launchThreads = secondaryLaunchThreadCount();
    if (launchThreads > 1 && GpuAcceleration::isProcessEnabled() && numPackets >= 65536)
    {
        secondaryLaunchParallel(launchThreads)->call(
            numCellOrderIndices,
            [firstCellOrderIndex, &launchCellRange](size_t firstCellOffset, size_t numCellOffsets) {
                launchCellRange(firstCellOrderIndex + firstCellOffset,
                                firstCellOrderIndex + firstCellOffset + numCellOffsets);
            });
    }
    else
    {
        launchCellRange(firstCellOrderIndex, lastCellOrderIndex + 1);
    }
    return true;
}

////////////////////////////////////////////////////////////////////

void DustSecondarySource::launchFromCell(PhotonPacket* pp, size_t historyIndex, double L, int p, int m) const
{
    // calculate the weight related to biased source selection
    double ws = _Lv[m] / _Wv[m];

    // calculate the emission spectrum and bulk velocity for this cell, if not already available
    if (!_hasGpuSourceLaunchData || !dustOptionEnabled("SKIRTGPU_HOST_PREPARED_DUST", true)
        || !t_dustcell.usePreparedSpectrum(p, _gpuSourceLaunchData))
        t_dustcell.calculateIfNeeded(p, _mv, _nv, _ms, _config);

    // generate a random wavelength from the emission spectrum for the cell and/or from the bias distribution
    double lambda, w;
    {
        double xi = _config->dustEmissionWavelengthBias();
        if (!xi)
        {
            // no biasing -- simply use the intrinsic spectral distribution
            lambda = t_dustcell.generateWavelength(_random);
            w = 1.;
        }
        else
        {
            // biasing -- use one or the other distribution
            if (_random->uniform() > xi)
                lambda = t_dustcell.generateWavelength(_random);
            else
                lambda = _config->dustEmissionWavelengthBiasDistribution()->generateWavelength();

            // calculate the compensating weight factor
            double s = t_dustcell.specificLuminosity(lambda);
            if (!s)
            {
                // if the wavelength can't occur in the intrinsic distribution,
                // the weight factor is zero regardless of the probability in the bias distribution
                // (handling this separately also avoids NaNs in the pathological case s=b=0)
                w = 0.;
            }
            else
            {
                // regular composite bias weight
                double b = _config->dustEmissionWavelengthBiasDistribution()->probability(lambda);
                w = s / ((1 - xi) * s + xi * b);
            }
        }
    }

    // generate a random position in this spatial cell
    Position bfr = _ms->grid()->randomPositionInCell(m);

    // provide a redshift interface for the appropriate velocity, if it is nonzero
    VelocityInterface* bvi = t_dustcell.velocity().isNull() ? nullptr : &t_dustcell;

    // provide a polarisation interface for polarised emission, if applicable
    // generate a random emission direction
    DustCellPolarisedEmission* dpe = nullptr;
    Direction bfk;
    if (_config->hasSpheroidalPolarization())
    {
        t_dustcellpol.calculateIfNeeded(m, _ms, lambda);
        dpe = &t_dustcellpol;
        bfk = t_dustcellpol.generateDirection(_random);
    }
    else
    {
        bfk = _random->direction();
    }

    pp->launch(historyIndex, lambda, L * ws * w, bfr, bfk, bvi, dpe, dpe);
}

////////////////////////////////////////////////////////////////////
