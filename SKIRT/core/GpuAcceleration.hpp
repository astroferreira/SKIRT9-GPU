/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#ifndef GPUACCELERATION_HPP
#define GPUACCELERATION_HPP

#include "Basics.hpp"
class AdaptiveMeshSpatialGrid;
class CartesianSpatialGrid;
class Cylinder2DSpatialGrid;
class Cylinder3DSpatialGrid;
class Direction;
class MediumState;
class Position;
class Sphere1DSpatialGrid;
class Sphere2DSpatialGrid;
class Sphere3DSpatialGrid;
class SpatialGridPath;
class TetraMeshSpatialGrid;
class TreeSpatialGrid;
class VoronoiMeshSpatialGrid;

//////////////////////////////////////////////////////////////////////

/** GpuAcceleration offers optional runtime-loaded GPU kernels for numerically isolated hot
    sections of the photon life cycle. If the GPU runtime is unavailable or disabled, all functions
    return false and callers must use their existing CPU implementation. */
class GpuAcceleration final
{
public:
    /** Enables or disables GPU acceleration for the current process. The default can also be set
        with the SKIRTGPU or SKIRT_GPU environment variables. */
    static void setProcessEnabled(bool enabled);

    /** Returns true if GPU acceleration has been requested for the current process. */
    static bool isProcessEnabled();

    /** Returns a short status string describing the current GPU backend state. */
    static string status();

    /** Returns true if unbatched single-photon path generation should be attempted on the GPU.
        This is disabled by default because current path kernels launch one synchronized CUDA
        thread per photon packet. Set SKIRTGPU_TRACE_PATHS=1 to opt into these diagnostic kernels. */
    static bool isPathGenerationEnabled();

    /** Returns true if synchronous per-photon photon-cycle kernels should be used in production
        transport. This is disabled by default until those kernels are batched across photon
        packets. Set SKIRTGPU_SYNC_PHOTON_CYCLE=1 to opt in for diagnostics and profiling. */
    static bool isSynchronousPhotonCycleEnabled();

    /** Runs a deterministic GPU kernel parity check. Returns false if the GPU backend is not
        available or if the result differs from the CPU reference. */
    static bool selfTest(string& message);

    /** Marks the cached GPU copy of the specified medium state as stale. */
    static void invalidateMediumState(const MediumState& state);

    /** Calculates the path segments through a Cartesian spatial grid on the GPU. The path is
        replaced with the generated segments. If maxDistance is finite, the first segment whose
        exit distance exceeds this value is included, matching the existing peel-off traversal. */
    static bool setCartesianSpatialGridPath(const CartesianSpatialGrid& grid, SpatialGridPath* path,
                                            double maxDistance);

    /** Calculates the path segments through a tree spatial grid on the GPU. The path is replaced
        with the generated segments. If maxDistance is finite, the first segment whose exit
        distance exceeds this value is included, matching the existing peel-off traversal. */
    static bool setTreeSpatialGridPath(const TreeSpatialGrid& grid, SpatialGridPath* path, double maxDistance);

    /** Calculates the path segments through an adaptive mesh spatial grid on the GPU. The path is
        replaced with the generated segments. If maxDistance is finite, the first segment whose
        exit distance exceeds this value is included, matching the existing peel-off traversal. */
    static bool setAdaptiveMeshSpatialGridPath(const AdaptiveMeshSpatialGrid& grid, SpatialGridPath* path,
                                               double maxDistance);

    /** Calculates the path segments through a 1D spherical spatial grid on the GPU. The path is
        replaced with the generated segments. If maxDistance is finite, the first segment whose
        exit distance exceeds this value is included, matching the existing peel-off traversal. */
    static bool setSphere1DSpatialGridPath(const Sphere1DSpatialGrid& grid, SpatialGridPath* path,
                                           double maxDistance);

    /** Calculates the path segments through a 2D spherical spatial grid on the GPU. The path is
        replaced with the generated segments. If maxDistance is finite, the first segment whose
        exit distance exceeds this value is included, matching the existing peel-off traversal. */
    static bool setSphere2DSpatialGridPath(const Sphere2DSpatialGrid& grid, SpatialGridPath* path,
                                           double maxDistance);

    /** Calculates the path segments through a 3D spherical spatial grid on the GPU. The path is
        replaced with the generated segments. If maxDistance is finite, the first segment whose
        exit distance exceeds this value is included, matching the existing peel-off traversal. */
    static bool setSphere3DSpatialGridPath(const Sphere3DSpatialGrid& grid, SpatialGridPath* path,
                                           double maxDistance);

    /** Calculates the path segments through a 2D cylindrical spatial grid on the GPU. The path is
        replaced with the generated segments. If maxDistance is finite, the first segment whose
        exit distance exceeds this value is included, matching the existing peel-off traversal. */
    static bool setCylinder2DSpatialGridPath(const Cylinder2DSpatialGrid& grid, SpatialGridPath* path,
                                             double maxDistance);

    /** Calculates the path segments through a 3D cylindrical spatial grid on the GPU. The path is
        replaced with the generated segments. If maxDistance is finite, the first segment whose
        exit distance exceeds this value is included, matching the existing peel-off traversal. */
    static bool setCylinder3DSpatialGridPath(const Cylinder3DSpatialGrid& grid, SpatialGridPath* path,
                                             double maxDistance);

    /** Calculates the path segments through a tetrahedral spatial grid on the GPU. The path is
        replaced with the generated segments. If maxDistance is finite, the first segment whose
        exit distance exceeds this value is included, matching the existing peel-off traversal. */
    static bool setTetraMeshSpatialGridPath(const TetraMeshSpatialGrid& grid, SpatialGridPath* path,
                                            double maxDistance);

    /** Calculates the path segments through a Voronoi mesh spatial grid on the GPU. The path is
        replaced with the generated segments. If maxDistance is finite, the first segment whose
        exit distance exceeds this value is included, matching the existing peel-off traversal. */
    static bool setVoronoiMeshSpatialGridPath(const VoronoiMeshSpatialGrid& grid, SpatialGridPath* path,
                                              double maxDistance);

    /** Calculates the path segments through a Voronoi mesh spatial grid for a batch of paths on
        the GPU. Each path is replaced with the generated segments. If maxDistance is finite, the
        first segment whose exit distance exceeds this value is included for each path. */
    static bool setVoronoiMeshSpatialGridPaths(const VoronoiMeshSpatialGrid& grid,
                                               const vector<SpatialGridPath*>& paths, double maxDistance);

    /** Calculates cumulative extinction optical depths for constant-section media on the GPU. */
    static bool setExtinctionOpticalDepths(SpatialGridPath* path, const MediumState& state,
                                           const vector<double>& sectionv);

    /** Calculates cumulative extinction optical depths for a batch of constant-section paths on
        the GPU. */
    static bool setExtinctionOpticalDepths(const vector<SpatialGridPath*>& paths, const MediumState& state,
                                           const vector<double>& sectionv);

    /** Calculates cumulative extinction optical depths on the GPU, evaluating per-medium section
        values from flattened dust lookup tables at the specified wavelength. */
    static bool setExtinctionOpticalDepthsFromTables(SpatialGridPath* path, const MediumState& state,
                                                     const vector<int>& mediaIndexv,
                                                     const vector<int>& lookupBeginv,
                                                     const vector<int>& lookupCountv,
                                                     const vector<double>& lookupWavelengthv,
                                                     const vector<double>& sectionExtv, double lambda);

    /** Calculates cumulative extinction optical depths for a batch of paths on the GPU,
        evaluating per-medium section values from flattened dust lookup tables at the specified
        wavelength for each path. */
    static bool setExtinctionOpticalDepthsFromTables(const vector<SpatialGridPath*>& paths,
                                                     const MediumState& state, const vector<int>& mediaIndexv,
                                                     const vector<int>& lookupBeginv,
                                                     const vector<int>& lookupCountv,
                                                     const vector<double>& lookupWavelengthv,
                                                     const vector<double>& sectionExtv,
                                                     const vector<double>& lambdav);

    /** Calculates Voronoi mesh paths and cumulative extinction optical depths in a single GPU
        compact-output pass, evaluating per-medium section values from flattened dust lookup
        tables at the specified wavelength for each path. */
    static bool setVoronoiMeshSpatialGridExtinctionOpticalDepthsFromTables(
        const VoronoiMeshSpatialGrid& grid, const vector<SpatialGridPath*>& paths, const MediumState& state,
        const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
        const vector<double>& lookupWavelengthv, const vector<double>& sectionExtv, const vector<double>& lambdav);

    /** Calculates total extinction optical depths for Voronoi mesh paths in a single GPU pass
        without materializing the path segments on the host, evaluating per-medium section values
        from flattened dust lookup tables at the specified wavelength for each path. */
    static bool getVoronoiMeshSpatialGridExtinctionOpticalDepthsFromTables(
        const VoronoiMeshSpatialGrid& grid, const vector<const SpatialGridPath*>& paths, const MediumState& state,
        const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
        const vector<double>& lookupWavelengthv, const vector<double>& sectionExtv,
        const vector<double>& lambdav, const vector<double>& maxDistancev, vector<double>& tauv);

    /** Calculates total extinction optical depths for Voronoi mesh rays in a single GPU pass
        without materializing host path objects, evaluating per-medium section values from
        flattened dust lookup tables at the specified wavelength for each ray. */
    static bool getVoronoiMeshSpatialGridExtinctionOpticalDepthsFromTables(
        const VoronoiMeshSpatialGrid& grid, const vector<Position>& positions, const vector<Direction>& directions,
        const MediumState& state, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
        const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
        const vector<double>& sectionExtv, const vector<double>& lambdav,
        const vector<double>& maxDistancev, vector<double>& tauv);

    /** Calculates total extinction optical depths for Voronoi mesh rays sharing a single
        direction in a single GPU pass, without materializing host path objects. */
    static bool getVoronoiMeshSpatialGridExtinctionOpticalDepthsFromTables(
        const VoronoiMeshSpatialGrid& grid, const vector<Position>& positions, const Direction& direction,
        const MediumState& state, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
        const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
        const vector<double>& sectionExtv, const vector<double>& lambdav,
        const vector<double>& maxDistancev, vector<double>& tauv);

    /** Calculates direct Henyey-Greenstein scattering peel-off luminosities with observer
        extinction for Voronoi mesh paths in a single GPU pass. The input direction vector is
        flattened as x,y,z triples, one per packet. */
	    static bool getVoronoiMeshSpatialGridHenyeyGreensteinScatteringObservedLuminositiesFromTables(
	        const VoronoiMeshSpatialGrid& grid, const vector<const SpatialGridPath*>& paths,
	        const MediumState& state, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
	        const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
	        const vector<double>& sectionExtv, const vector<double>& lambdav,
	        const vector<double>& maxDistancev, const vector<double>& inputDirectionv,
	        const vector<double>& packetLuminosityv, Direction bfkobs, int hgLookupBegin,
	        int hgLookupCount, const vector<double>& asymmparv, vector<double>& luminosityv);

	    /** Calculates direct Henyey-Greenstein scattering peel-off luminosities with observer
	        extinction for Voronoi mesh rays and accumulates the redshifted frame-band detector
	        contributions into a persistent GPU accumulator. This avoids materializing observer
	        luminosities or detector key/value pairs on the host. */
	    static bool getVoronoiMeshSpatialGridHenyeyGreensteinScatteringFrameBandAccumulateFromTables(
	        const VoronoiMeshSpatialGrid& grid, const vector<Position>& positionv, const MediumState& state,
	        const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
	        const vector<double>& lookupWavelengthv, const vector<double>& sectionExtv,
	        const vector<double>& lambdav, const vector<double>& maxDistancev,
	        const vector<double>& inputDirectionv, const vector<double>& packetLuminosityv, Direction bfkobs,
	        int hgLookupBegin, int hgLookupCount, const vector<double>& asymmparv, const void* accumulatorKey,
	        size_t numAccumulatorValues, double costheta, double sintheta, double cosphi, double sinphi,
	        double cosomega, double sinomega, int numPixelsX, int numPixelsY, double xpmin, double xpsiz,
	        double ypmin, double ypsiz, double redshift, size_t numPixelsInFrame,
	        const vector<int>& bandOffsetv, const vector<double>& bandWavelengthv,
	        const vector<double>& bandTransmissionv, const vector<double>& bandWidthv);

	    /** Calculates the extinction optical depth for a constant-section path on the GPU. If the
	        returned optical depth reaches the specified maximum, the output is positive infinity. */
    static bool getExtinctionOpticalDepth(const SpatialGridPath* path, const MediumState& state,
                                          const vector<double>& sectionv, double taumax, double& tau);

    /** Calculates the extinction optical depth on the GPU, evaluating per-medium section values
        from flattened dust lookup tables at the specified wavelength. If the returned optical
        depth reaches the specified maximum, the output is positive infinity. */
    static bool getExtinctionOpticalDepthFromTables(const SpatialGridPath* path, const MediumState& state,
                                                    const vector<int>& mediaIndexv,
                                                    const vector<int>& lookupBeginv,
                                                    const vector<int>& lookupCountv,
                                                    const vector<double>& lookupWavelengthv,
                                                    const vector<double>& sectionExtv, double lambda, double taumax,
                                                    double& tau);

    /** Finds the interaction point for a constant-section extinction path using GPU-calculated
        segment optical-depth contributions. The function returns true if the GPU calculation
        succeeded; in that case, the found flag indicates whether the interaction lies inside the
        path. */
    static bool findInteractionPointUsingExtinction(const SpatialGridPath* path, const MediumState& state,
                                                   const vector<double>& sectionv, double tauinteract, bool& found,
                                                   int& m, double& s);

    /** Finds the interaction point for an extinction path using GPU table-evaluated section
        values. The function returns true if the GPU calculation succeeded; in that case, the
        found flag indicates whether the interaction lies inside the path. */
    static bool findInteractionPointUsingExtinctionFromTables(const SpatialGridPath* path, const MediumState& state,
                                                             const vector<int>& mediaIndexv,
                                                             const vector<int>& lookupBeginv,
                                                             const vector<int>& lookupCountv,
                                                             const vector<double>& lookupWavelengthv,
                                                             const vector<double>& sectionExtv, double lambda,
                                                             double tauinteract, bool& found, int& m, double& s);

    /** Calculates cumulative scattering and absorption optical depths for constant-section media
        on the GPU. */
    static bool setScatteringAndAbsorptionOpticalDepths(SpatialGridPath* path, const MediumState& state,
                                                        const vector<double>& sectionScav,
                                                        const vector<double>& sectionAbsv);

    /** Calculates cumulative scattering and absorption optical depths for a batch of
        constant-section paths on the GPU. */
    static bool setScatteringAndAbsorptionOpticalDepths(const vector<SpatialGridPath*>& paths,
                                                        const MediumState& state,
                                                        const vector<double>& sectionScav,
                                                        const vector<double>& sectionAbsv);

    /** Calculates cumulative scattering and absorption optical depths on the GPU, evaluating
        per-medium section values from flattened dust lookup tables at the specified wavelength. */
    static bool setScatteringAndAbsorptionOpticalDepthsFromTables(SpatialGridPath* path, const MediumState& state,
                                                                  const vector<int>& mediaIndexv,
                                                                  const vector<int>& lookupBeginv,
                                                                  const vector<int>& lookupCountv,
                                                                  const vector<double>& lookupWavelengthv,
                                                                  const vector<double>& sectionScav,
                                                                  const vector<double>& sectionAbsv, double lambda);

    /** Calculates cumulative scattering and absorption optical depths for a batch of paths on the
        GPU, evaluating per-medium section values from flattened dust lookup tables at the
        specified wavelength for each path. */
    static bool setScatteringAndAbsorptionOpticalDepthsFromTables(
        const vector<SpatialGridPath*>& paths, const MediumState& state, const vector<int>& mediaIndexv,
        const vector<int>& lookupBeginv, const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
        const vector<double>& sectionScav, const vector<double>& sectionAbsv, const vector<double>& lambdav);

    /** Calculates Voronoi mesh paths and cumulative scattering plus absorption optical depths in a
        single GPU compact-output pass, evaluating per-medium section values from flattened dust
        lookup tables at the specified wavelength for each path. */
    static bool setVoronoiMeshSpatialGridScatteringAndAbsorptionOpticalDepthsFromTables(
        const VoronoiMeshSpatialGrid& grid, const vector<SpatialGridPath*>& paths, const MediumState& state,
        const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
        const vector<double>& lookupWavelengthv, const vector<double>& sectionScav,
        const vector<double>& sectionAbsv, const vector<double>& lambdav);

    /** Finds the interaction point for a constant-section scattering path using GPU-calculated
        segment scattering and absorption optical-depth contributions. The function returns true
        if the GPU calculation succeeded; in that case, the found flag indicates whether the
        interaction lies inside the path. */
    static bool findInteractionPointUsingScatteringAndAbsorption(const SpatialGridPath* path, const MediumState& state,
                                                                 const vector<double>& sectionScav,
                                                                 const vector<double>& sectionAbsv,
                                                                 double tauinteract, bool& found, int& m, double& s,
                                                                 double& tauAbs);

    /** Finds the interaction point for a scattering path using GPU table-evaluated section values.
        The function returns true if the GPU calculation succeeded; in that case, the found flag
        indicates whether the interaction lies inside the path. */
    static bool findInteractionPointUsingScatteringAndAbsorptionFromTables(
        const SpatialGridPath* path, const MediumState& state, const vector<int>& mediaIndexv,
        const vector<int>& lookupBeginv, const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
        const vector<double>& sectionScav, const vector<double>& sectionAbsv, double lambda, double tauinteract,
        bool& found, int& m, double& s, double& tauAbs);

    /** Calculates the scattering albedo and normalized per-medium scattering weights for a
        constant-section scattering event in the specified cell. */
    static bool scatteringProperties(const MediumState& state, int cellIndex, const vector<double>& sectionScav,
                                     const vector<double>& sectionExtv, double& albedo, vector<double>& weights);

    /** Calculates the scattering albedo and normalized per-medium scattering weights using GPU
        table-evaluated dust section values at the specified wavelength. */
    static bool scatteringPropertiesFromTables(const MediumState& state, int cellIndex,
                                               const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
                                               const vector<int>& lookupCountv,
                                               const vector<double>& lookupWavelengthv,
                                               const vector<double>& sectionScav,
                                               const vector<double>& sectionExtv, double lambda, double& albedo,
                                               vector<double>& weights);

    /** Calculates scattering albedos for a batch of constant-section scattering events. */
    static bool scatteringAlbedos(const MediumState& state, const vector<int>& cellv,
                                  const vector<double>& sectionScav, const vector<double>& sectionExtv,
                                  vector<double>& albedov);

    /** Calculates scattering albedos for a batch of table-evaluated dust scattering events. */
    static bool scatteringAlbedosFromTables(const MediumState& state, const vector<int>& cellv,
                                            const vector<double>& lambdav, const vector<int>& mediaIndexv,
                                            const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
                                            const vector<double>& lookupWavelengthv,
                                            const vector<double>& sectionScav,
                                            const vector<double>& sectionExtv, vector<double>& albedov);

    /** Finds the interaction point in a path that already stores cumulative extinction or
        scattering/absorption optical depths. This is used by forced-scattering propagation after
        the cumulative optical-depth path has been prepared. */
    static bool findInteractionPointInCumulativePath(const SpatialGridPath* path, double tauinteract,
                                                     bool hasAbsorption, int& m, double& s, double& tauAbs);

    /** Finds interaction points in a batch of paths that already store cumulative extinction or
        scattering/absorption optical depths. The output vectors receive one cell index, distance,
        and cumulative absorption optical depth per input path. */
    static bool findInteractionPointsInCumulativePaths(const vector<const SpatialGridPath*>& paths,
                                                       const vector<double>& tauinteractv, bool hasAbsorption,
                                                       vector<int>& cellv, vector<double>& distancev,
                                                       vector<double>& tauAbsv);

    /** Calculates the forced-propagation interaction points and luminosity bias factors for a
        batch of cumulative optical-depth paths. The \em pathBiasWeightv input contains the
        path-length-bias compensation for each path. For non-explicit-absorption paths,
        \em albedov must contain the scattering albedo at each interaction cell; for
        explicit-absorption paths it is ignored. */
    static bool forcedPropagationResults(const vector<const SpatialGridPath*>& paths,
                                         const vector<double>& tauinteractv,
                                         const vector<double>& pathBiasWeightv, bool hasAbsorption,
                                         const vector<double>& albedov, vector<int>& cellv,
                                         vector<double>& distancev, vector<double>& tauAbsv,
                                         vector<double>& weightv);

    /** Calculates forced-propagation interaction points and luminosity bias factors for
        non-explicit-absorption paths, using GPU table-evaluated dust scattering albedos at each
        interaction cell. */
    static bool forcedPropagationResultsFromTables(
        const vector<const SpatialGridPath*>& paths, const MediumState& state, const vector<double>& tauinteractv,
        const vector<double>& pathBiasWeightv, const vector<double>& lambdav, const vector<int>& mediaIndexv,
        const vector<int>& lookupBeginv, const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
        const vector<double>& sectionScav, const vector<double>& sectionExtv, vector<int>& cellv,
        vector<double>& distancev, vector<double>& tauAbsv, vector<double>& weightv);

    /** Calculates reduced radiation-field luminosity-distance contributions and forced-propagation
        interaction points/bias factors in one GPU runtime pass over the same flattened paths. This
        is intended for non-explicit-absorption, constant-perceived-wavelength forced-scattering
        batches using table-evaluated dust sections. */
    static bool radiationFieldSumsAndForcedPropagationResultsFromTables(
        const vector<const SpatialGridPath*>& paths, const MediumState& state,
        const vector<double>& luminosityv, const vector<int>& wavelengthBinv, int numWavelengths,
        const vector<double>& tauinteractv, const vector<double>& pathBiasWeightv,
        const vector<double>& lambdav, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
        const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
        const vector<double>& sectionScav, const vector<double>& sectionExtv,
        vector<int>& binIndexv, vector<double>& Ldsv, vector<int>& cellv,
        vector<double>& distancev, vector<double>& tauAbsv, vector<double>& weightv);

    /** Calculates Voronoi mesh paths, reduced radiation-field luminosity-distance contributions,
        and forced-propagation interaction points/bias factors in one GPU runtime pass without
        materializing the path segments on the host. This is intended for non-explicit-absorption,
        constant-perceived-wavelength forced-scattering batches using table-evaluated dust
        sections. */
    static bool voronoiMeshSpatialGridRadiationFieldAndForcedPropagationResultsFromTables(
        const VoronoiMeshSpatialGrid& grid, const vector<Position>& positionv,
        const vector<Direction>& directionv, const MediumState& state,
        const vector<double>& luminosityv, const vector<int>& wavelengthBinv, int numWavelengths,
        const vector<double>& tauinteractv, const vector<double>& pathBiasWeightv,
        const vector<double>& lambdav, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
        const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
        const vector<double>& sectionScav, const vector<double>& sectionExtv,
        vector<int>& binIndexv, vector<double>& Ldsv, vector<int>& cellv,
        vector<double>& distancev, vector<double>& tauAbsv, vector<double>& weightv,
        const void* accumulatorKey = nullptr, size_t numAccumulatorValues = 0);

    /** Calculates Voronoi mesh paths, reduced radiation-field luminosity-distance contributions,
        and forced-propagation interaction points/bias factors in one GPU runtime pass without
        materializing the path segments or precomputing total path optical depths on the host. The
        randomSelectv/randomSamplev inputs contain CPU-generated uniform deviates used by the GPU
        kernel to sample the forced interaction optical depth after it has calculated the path's
        total optical depth. */
    static bool voronoiMeshSpatialGridSampledRadiationFieldAndForcedPropagationResultsFromTables(
        const VoronoiMeshSpatialGrid& grid, const vector<Position>& positionv,
        const vector<Direction>& directionv, const MediumState& state,
        const vector<double>& luminosityv, const vector<int>& wavelengthBinv, int numWavelengths,
        const vector<double>& randomSelectv, const vector<double>& randomSamplev, double pathLengthBias,
        const vector<double>& lambdav, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
        const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
        const vector<double>& sectionScav, const vector<double>& sectionExtv,
        vector<int>& binIndexv, vector<double>& Ldsv, vector<int>& cellv,
        vector<double>& distancev, vector<double>& tauAbsv, vector<double>& weightv,
        const void* accumulatorKey = nullptr, size_t numAccumulatorValues = 0);

    /** Calculates the sampled resident Voronoi/table radiation-field plus forced-propagation pass
        and also samples Henyey-Greenstein outgoing directions for the same incoming packet
        directions. The extra scatter random vectors contain CPU-generated uniform deviates and the
        output direction vector is flattened as x,y,z triples, one per packet. */
    static bool
    voronoiMeshSpatialGridSampledRadiationFieldForcedPropagationAndHenyeyGreensteinScatteringResultsFromTables(
        const VoronoiMeshSpatialGrid& grid, const vector<Position>& positionv,
        const vector<Direction>& directionv, const MediumState& state,
        const vector<double>& luminosityv, const vector<int>& wavelengthBinv, int numWavelengths,
        const vector<double>& randomSelectv, const vector<double>& randomSamplev, double pathLengthBias,
        const vector<double>& scatterRandomCosthetav, const vector<double>& scatterRandomPhiv,
        int hgLookupBegin, int hgLookupCount, const vector<double>& asymmparv,
        const vector<double>& lambdav, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
        const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
        const vector<double>& sectionScav, const vector<double>& sectionExtv,
        vector<int>& binIndexv, vector<double>& Ldsv, vector<int>& cellv,
        vector<double>& distancev, vector<double>& tauAbsv, vector<double>& weightv,
        vector<double>& scatterDirectionv, const void* accumulatorKey = nullptr,
        size_t numAccumulatorValues = 0);

    /** Calculates per-cell dust absorbed luminosities from the radiation field for
        constant-section dust media. The absorption section vector is flattened with dust medium
        index as the slow dimension and wavelength bin as the fast dimension. */
    static bool dustAbsorbedLuminosities(const MediumState& state, int numCells, int numWavelengths,
                                         const vector<int>& dustMedia, const vector<double>& sectionAbsv,
                                         const double* primaryRadiationField,
                                         const double* secondaryRadiationField,
                                         vector<double>& primaryLuminosities,
                                         vector<double>& secondaryLuminosities);

    /** Calculates the total dust absorbed luminosities from the radiation field for
        constant-section dust media, respectively for the primary and stable secondary radiation
        fields. */
    static bool totalDustAbsorbedLuminosity(const MediumState& state, int numCells, int numWavelengths,
                                            const vector<int>& dustMedia, const vector<double>& sectionAbsv,
                                            const double* primaryRadiationField,
                                            const double* secondaryRadiationField, double& primaryLuminosity,
                                            double& secondaryLuminosity);

    /** Calculates the radiation-field luminosity-distance contribution for each path segment on
        the GPU, assuming the photon packet has constant perceived wavelength along the path. */
    static bool radiationFieldContributions(const SpatialGridPath* path, double luminosity, vector<double>& Ldsv);

    /** Calculates radiation-field luminosity-distance contributions for a batch of cumulative
        optical-depth paths on the GPU, assuming each photon packet has constant perceived
        wavelength along its path. The pathOffsetv output maps each path to its flattened segment
        range in Ldsv. */
    static bool radiationFieldContributions(const vector<const SpatialGridPath*>& paths,
                                            const vector<double>& luminosityv, vector<int>& pathOffsetv,
                                            vector<double>& Ldsv);

    /** Calculates and reduces radiation-field luminosity-distance contributions for a batch of
        cumulative optical-depth paths on the GPU, assuming each photon packet has constant
        perceived wavelength along its path. The output vectors contain flattened radiation-field
        table indices and their summed values. */
    static bool radiationFieldContributionSums(const vector<const SpatialGridPath*>& paths,
                                               const vector<double>& luminosityv,
                                               const vector<int>& wavelengthBinv, int numWavelengths,
                                               vector<int>& binIndexv, vector<double>& Ldsv);

    /** Multiplies one spectral value per wavelength by the corresponding factor on the GPU. The
        host array is updated only if the GPU operation succeeds. */
    static bool scaleWavelengthValues(double* values, size_t numWavelengths, const vector<double>& factorv);

    /** Multiplies each wavelength frame in a flattened IFU datacube by the corresponding
        wavelength factor on the GPU. The host array is updated only if the GPU operation
        succeeds. */
    static bool scaleFrameWavelengthValues(double* values, size_t numWavelengths, size_t numPixelsInFrame,
                                           const vector<double>& factorv);

    /** Divides all values in the specified host array by a scalar on the GPU. The host array is
        updated only if the GPU operation succeeds. */
    static bool divideValues(double* values, size_t numValues, double divisor);

    /** Multiplies all values in the specified host array by a scalar on the GPU. The host array is
        updated only if the GPU operation succeeds. */
    static bool multiplyValues(double* values, size_t numValues, double factor);

    /** Sums two to four equally-sized arrays on the GPU and stores the result in the host output
        array. The host output array is updated only if the GPU operation succeeds. The third and
        fourth inputs may be null. */
    static bool sumValues(double* output, size_t numValues, const double* value1, const double* value2,
                          const double* value3 = nullptr, const double* value4 = nullptr);

    /** Sums values by nonnegative integer key on the GPU. The output vectors contain compact keys
        and their summed values. Negative input keys are ignored. */
    static bool sumValuesByKey(const vector<int>& keyv, const vector<double>& valuev,
                               vector<int>& compactKeyv, vector<double>& compactValuev);

    /** Adds values by nonnegative integer key into a persistent GPU accumulator identified by the
        specified host key. Values with keys outside the accumulator range are ignored. */
    static bool accumulateValuesByKey(const void* accumulatorKey, size_t numAccumulatorValues,
                                      const vector<int>& keyv, const vector<double>& valuev);

    /** Adds the persistent GPU accumulator identified by the specified host key into the host
        array and clears the GPU-side accumulator on all devices. */
    static bool retrieveAndClearAccumulatedValues(const void* accumulatorKey, double* values, size_t numValues);

    /** Clears the persistent GPU accumulator identified by the specified host key on all devices. */
    static void clearAccumulatedValues(const void* accumulatorKey);

    /** Projects positions onto a distant frame detector, evaluates band transmissions, applies
        observer extinction, and returns compact total-flux contributions keyed by flattened
        pixel/wavelength-bin index. This is intended for total-only distant frame/IFU detector
        batches. */
    static bool frameBandTotalFluxSums(
        const vector<Position>& positionv, const vector<double>& wavelengthv, const vector<double>& luminosityv,
        const vector<double>& tauv, bool hasMedium, double costheta, double sintheta, double cosphi,
        double sinphi, double cosomega, double sinomega, int numPixelsX, int numPixelsY, double xpmin,
        double xpsiz, double ypmin, double ypsiz, double redshift, size_t numPixelsInFrame,
        const vector<int>& bandOffsetv, const vector<double>& bandWavelengthv,
        const vector<double>& bandTransmissionv, const vector<double>& bandWidthv,
        vector<int>& compactKeyv, vector<double>& compactValuev);

    /** Projects positions onto a distant frame detector, evaluates band transmissions, applies
        observer extinction, and accumulates total-flux contributions into a persistent GPU
        accumulator keyed by the specified host pointer. */
    static bool frameBandTotalFluxAccumulate(
        const void* accumulatorKey, size_t numAccumulatorValues,
        const vector<Position>& positionv, const vector<double>& wavelengthv, const vector<double>& luminosityv,
        const vector<double>& tauv, bool hasMedium, double costheta, double sintheta, double cosphi,
        double sinphi, double cosomega, double sinomega, int numPixelsX, int numPixelsY, double xpmin,
        double xpsiz, double ypmin, double ypsiz, double redshift, size_t numPixelsInFrame,
        const vector<int>& bandOffsetv, const vector<double>& bandWavelengthv,
        const vector<double>& bandTransmissionv, const vector<double>& bandWidthv);

    /** Calculates direct Henyey-Greenstein scattering peel-off luminosities on the GPU. The input
        direction vector is flattened as x,y,z triples, one per packet. */
    static bool henyeyGreensteinScatteringLuminosities(const vector<double>& inputDirectionv,
                                                       const vector<double>& packetLuminosityv,
                                                       const vector<double>& lambdav, Direction bfkobs,
                                                       int lookupBegin, int lookupCount,
                                                       const vector<double>& lookupWavelengthv,
                                                       const vector<double>& asymmparv,
                                                       vector<double>& luminosityv);

    /** Samples Henyey-Greenstein scattering directions on the GPU. The input and output direction
        vectors are flattened as x,y,z triples, one per packet. The two random vectors contain
        pre-drawn uniform deviates so the caller keeps ownership of the random stream. */
    static bool henyeyGreensteinScatteringDirections(const vector<double>& inputDirectionv,
                                                     const vector<double>& lambdav,
                                                     const vector<double>& randomCosthetav,
                                                     const vector<double>& randomPhiv, int lookupBegin,
                                                     int lookupCount, const vector<double>& lookupWavelengthv,
                                                     const vector<double>& asymmparv,
                                                     vector<double>& outputDirectionv);

    /** Samples isotropic launch directions on the GPU. The output direction vector is flattened
        as x,y,z triples, one per packet. The two random vectors contain pre-drawn uniform deviates
        so the caller keeps ownership of the random stream. */
    static bool isotropicDirections(const vector<double>& randomCosthetav, const vector<double>& randomPhiv,
                                    vector<double>& outputDirectionv);

    /** Calculates stored-table CDF normalization factors for many SED parameter vectors on the
        GPU. The first table axis is integrated from xmin to xmax; the remaining axes are supplied
        in parameterValues with entity index as the slow dimension. The scaleValues vector is
        multiplied into the output normalization for each entity. */
    static bool storedTableCdf(vector<double>& luminosities, int numAxes, const vector<const double*>& axisData,
                               const vector<size_t>& axisSizes, const vector<bool>& axisLog, const double* quantity,
                               size_t quantityStep, bool quantityLog, bool clampFirstAxis, double xmin, double xmax,
                               const vector<double>& parameterValues, const vector<double>& scaleValues,
                               size_t numEntities);

    /** Samples wavelengths from stored-table SED CDFs and evaluates the corresponding normalized
        specific luminosities on the GPU. The first table axis is wavelength; the remaining axes
        are supplied in parameterValues with sample index as the slow dimension. If
        forcedWavelengths is non-empty, positive values in that vector are evaluated directly
        instead of sampling from the intrinsic SED using intrinsicRandoms. */
    static bool storedTableSampleWavelengths(vector<double>& wavelengths, vector<double>& specificLuminosities,
                                             int numAxes, const vector<const double*>& axisData,
                                             const vector<size_t>& axisSizes, const vector<bool>& axisLog,
                                             const double* quantity, size_t quantityStep, bool quantityLog,
                                             bool clampFirstAxis, double xmin, double xmax,
                                             const vector<double>& parameterValues,
                                             const vector<double>& intrinsicRandoms,
                                             const vector<double>& forcedWavelengths, size_t numSamples);

    /** Constructs secondary-source composite launch weights on the GPU from normalized
        per-cell luminosities and the configured uniform spatial-bias fraction. */
    static bool compositeLaunchWeights(const double* luminosityv, size_t numValues, double spatialBias,
                                       double* weightv);
};

//////////////////////////////////////////////////////////////////////

#endif
