/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#include "GpuAcceleration.hpp"
#include "AdaptiveMeshSpatialGrid.hpp"
#include "CartesianSpatialGrid.hpp"
#include "Cylinder2DSpatialGrid.hpp"
#include "Cylinder3DSpatialGrid.hpp"
#include "MediumState.hpp"
#include "ProcessManager.hpp"
#include "Sphere1DSpatialGrid.hpp"
#include "Sphere2DSpatialGrid.hpp"
#include "Sphere3DSpatialGrid.hpp"
#include "SpecialFunctions.hpp"
#include "SpatialGridPath.hpp"
#include "StateVariable.hpp"
#include "TetraMeshSpatialGrid.hpp"
#include "TreeSpatialGrid.hpp"
#include "VoronoiMeshSpatialGrid.hpp"
#include <array>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

////////////////////////////////////////////////////////////////////

namespace
{
    bool _processEnabled = false;

    double interpolateLinLin(double x, double x1, double x2, double y1, double y2)
    {
        return x2 != x1 ? y1 + (x - x1) * (y2 - y1) / (x2 - x1) : y1;
    }

    bool isTruthy(const char* value)
    {
        if (!value) return false;
        string text(value);
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
        return !(text.empty() || text == "0" || text == "false" || text == "off" || text == "no");
    }

    size_t minGpuSegments()
    {
        const char* text = std::getenv("SKIRTGPU_MIN_SEGMENTS");
        if (!text) return 64;
        char* end = nullptr;
        long value = std::strtol(text, &end, 10);
        return end != text && value > 0 ? static_cast<size_t>(value) : 0;
    }

    size_t nextPowerOfTwo(size_t value)
    {
        if (value <= 1) return 1;
        --value;
        for (size_t shift = 1; shift < sizeof(size_t) * 8; shift <<= 1) value |= value >> shift;
        return value + 1;
    }

    bool pathGenerationEnabled()
    {
        return isTruthy(std::getenv("SKIRTGPU_TRACE_PATHS"));
    }

    bool synchronousPhotonCycleEnabled()
    {
        return isTruthy(std::getenv("SKIRTGPU_SYNC_PHOTON_CYCLE"));
    }

    bool residentProfileEnabled()
    {
        return isTruthy(std::getenv("SKIRTGPU_PROFILE_RESIDENT"));
    }

    size_t residentProfileLimit()
    {
        const char* text = std::getenv("SKIRTGPU_PROFILE_RESIDENT_LIMIT");
        if (!text) return 64;
        char* end = nullptr;
        unsigned long value = std::strtoul(text, &end, 10);
        return end != text ? static_cast<size_t>(value) : 64;
    }

    double millisecondsBetween(std::chrono::steady_clock::time_point begin,
                               std::chrono::steady_clock::time_point end)
    {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    }

    std::atomic<size_t> _residentProfileCount{0};
    std::mutex _residentProfileMutex;
}

////////////////////////////////////////////////////////////////////

#if defined(__unix__) || defined(__APPLE__)

namespace
{
    using CUdevice = int;
    using CUresult = int;
    using CUdeviceptr = unsigned long long;
    using CUcontext = struct CUctx_st*;
    using CUmodule = struct CUmod_st*;
    using CUfunction = struct CUfunc_st*;
    using CUstream = struct CUstream_st*;
    using nvrtcProgram = struct _nvrtcProgram*;
    using nvrtcResult = int;

    const CUresult CUDA_SUCCESS_VALUE = 0;
    const nvrtcResult NVRTC_SUCCESS_VALUE = 0;

    template<typename T> bool loadSymbol(void* handle, const char* name, T& function)
    {
        function = reinterpret_cast<T>(dlsym(handle, name));
        return function != nullptr;
    }

    template<typename T> bool loadVersionedSymbol(void* handle, const char* name, const char* versionedName, T& function)
    {
        return loadSymbol(handle, name, function) || loadSymbol(handle, versionedName, function);
    }

    void* openLibrary(const vector<string>& names)
    {
        for (const auto& name : names)
        {
            if (name.empty()) continue;
            void* handle = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (handle) return handle;
        }
        return nullptr;
    }

    void* openLibraryGlobal(const vector<string>& names)
    {
        for (const auto& name : names)
        {
            if (name.empty()) continue;
            void* handle = dlopen(name.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (handle) return handle;
        }
        return nullptr;
    }

    const char* kernelSource()
    {
        return R"cuda(
static __device__ double skirt_lnmean(double x1, double x2, double lnx1, double lnx2)
{
    if (x1 > x2)
    {
        double tmp = x1;
        x1 = x2;
        x2 = tmp;
        tmp = lnx1;
        lnx1 = lnx2;
        lnx2 = tmp;
    }
    if (x1 <= 0.0) return 0.0;

    double x = x2 / x1 - 1.0;
    if (x < 1e-3)
    {
        return x1
               / (1.0 - 1.0 / 2.0 * x + 1.0 / 3.0 * x * x - 1.0 / 4.0 * x * x * x
                  + 1.0 / 5.0 * x * x * x * x - 1.0 / 6.0 * x * x * x * x * x);
    }
    return (x2 - x1) / (lnx2 - lnx1);
}

static __device__ unsigned int skirt_hash_uint(unsigned int value)
{
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

static __device__ void skirt_atomic_add_double(double* address, double value)
{
#if __CUDA_ARCH__ >= 600
    atomicAdd(address, value);
#else
    unsigned long long int* addressAsUll = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *addressAsUll;
    unsigned long long int assumed;
    do
    {
        assumed = old;
        old = atomicCAS(addressAsUll, assumed,
                        __double_as_longlong(value + __longlong_as_double(static_cast<long long>(assumed))));
    } while (assumed != old);
#endif
}

static __device__ int skirt_locate_basic(const double* xv, double x, int n)
{
    int jl = -1;
    int ju = n;
    while (ju - jl > 1)
    {
        int jm = (ju + jl) >> 1;
        if (x < xv[jm])
            ju = jm;
        else
            jl = jm;
    }
    return jl;
}

static __device__ int skirt_locate_array(const double* xv, double x, int numBorders)
{
    if (x == xv[numBorders - 1]) return numBorders - 2;
    return skirt_locate_basic(xv, x, numBorders);
}

static __device__ int skirt_locate_clip(const double* xv, double x, int numBorders)
{
    if (x < xv[0]) return 0;
    return skirt_locate_basic(xv, x, numBorders - 1);
}

static __device__ int skirt_locate_fail(const double* xv, double x, int numBorders)
{
    if (x > xv[numBorders - 1]) return -1;
    return skirt_locate_basic(xv, x, numBorders - 1);
}

static __device__ double skirt_quadratic_smallest_positive(double b, double c)
{
    if (b * b > c)
    {
        if (b > 0.0)
        {
            if (c < 0.0)
            {
                double x1 = -b - sqrt(b * b - c);
                return c / x1;
            }
        }
        else
        {
            double x2 = -b + sqrt(b * b - c);
            if (c > 0.0)
            {
                double x1 = c / x2;
                if (x1 < x2) return x1;
            }
            return x2;
        }
    }
    return 0.0;
}

static __device__ double skirt_smallest_positive_scaled(double a, double b, double c)
{
    if (fabs(a) > 1e-9) return skirt_quadratic_smallest_positive(b / a, c / a);
    double x = -0.5 * c / b;
    return x > 0.0 ? x : 0.0;
}

extern "C" __global__ void cartesian_grid_path(const double* xv, const double* yv, const double* zv,
                                               int nx, int ny, int nz,
                                               double rx, double ry, double rz,
                                               double kx, double ky, double kz,
                                               double xmin, double ymin, double zmin,
                                               double xmax, double ymax, double zmax,
                                               double maxDistance, int maxSegments,
                                               int* cellOutv, double* dsOutv,
                                               int* countOutv, int* statusOutv)
{
    countOutv[0] = 0;
    statusOutv[0] = 1;
    if (nx <= 0 || ny <= 0 || nz <= 0 || maxSegments <= 0) return;

    double dx = xmax - xmin;
    double dy = ymax - ymin;
    double dz = zmax - zmin;
    double eps = 1e-12 * sqrt(dx * dx + dy * dy + dz * dz);
    double cumds = 0.0;

    if (rx <= xmin)
    {
        if (kx <= 0.0) return;
        double ds = (xmin - rx) / kx;
        rx = xmin + eps;
        ry += ky * ds;
        rz += kz * ds;
        cumds += ds;
    }
    else if (rx >= xmax)
    {
        if (kx >= 0.0) return;
        double ds = (xmax - rx) / kx;
        rx = xmax - eps;
        ry += ky * ds;
        rz += kz * ds;
        cumds += ds;
    }

    if (ry <= ymin)
    {
        if (ky <= 0.0) return;
        double ds = (ymin - ry) / ky;
        rx += kx * ds;
        ry = ymin + eps;
        rz += kz * ds;
        cumds += ds;
    }
    else if (ry >= ymax)
    {
        if (ky >= 0.0) return;
        double ds = (ymax - ry) / ky;
        rx += kx * ds;
        ry = ymax - eps;
        rz += kz * ds;
        cumds += ds;
    }

    if (rz <= zmin)
    {
        if (kz <= 0.0) return;
        double ds = (zmin - rz) / kz;
        rx += kx * ds;
        ry += ky * ds;
        rz = zmin + eps;
        cumds += ds;
    }
    else if (rz >= zmax)
    {
        if (kz >= 0.0) return;
        double ds = (zmax - rz) / kz;
        rx += kx * ds;
        ry += ky * ds;
        rz = zmax - eps;
        cumds += ds;
    }

    if (!(rx >= xmin && rx <= xmax && ry >= ymin && ry <= ymax && rz >= zmin && rz <= zmax)) return;

    int count = 0;
    double distance = 0.0;
    if (cumds > 0.0)
    {
        if (count >= maxSegments)
        {
            statusOutv[0] = -1;
            return;
        }
        cellOutv[count] = -1;
        dsOutv[count] = cumds;
        ++count;
        distance += cumds;
        if (distance > maxDistance)
        {
            countOutv[0] = count;
            return;
        }
    }

    int i = skirt_locate_clip(xv, rx, nx + 1);
    int j = skirt_locate_clip(yv, ry, ny + 1);
    int k = skirt_locate_clip(zv, rz, nz + 1);
    const double big = 1.7976931348623157e+308;

    while (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
    {
        int m = k + nz * j + nz * ny * i;
        double xE = (kx < 0.0) ? xv[i] : xv[i + 1];
        double yE = (ky < 0.0) ? yv[j] : yv[j + 1];
        double zE = (kz < 0.0) ? zv[k] : zv[k + 1];
        double dsx = (fabs(kx) > 1e-15) ? (xE - rx) / kx : big;
        double dsy = (fabs(ky) > 1e-15) ? (yE - ry) / ky : big;
        double dsz = (fabs(kz) > 1e-15) ? (zE - rz) / kz : big;

        int axis = 2;
        double ds = dsz;
        if (dsx <= dsy && dsx <= dsz)
        {
            axis = 0;
            ds = dsx;
        }
        else if (dsy < dsx && dsy <= dsz)
        {
            axis = 1;
            ds = dsy;
        }

        if (ds > 0.0)
        {
            if (count >= maxSegments)
            {
                statusOutv[0] = -1;
                return;
            }
            cellOutv[count] = m;
            dsOutv[count] = ds;
            ++count;
            distance += ds;
            if (distance > maxDistance)
            {
                countOutv[0] = count;
                return;
            }
        }

        if (axis == 0)
        {
            rx = xE;
            ry += ky * ds;
            rz += kz * ds;
            i += (kx < 0.0) ? -1 : 1;
        }
        else if (axis == 1)
        {
            rx += kx * ds;
            ry = yE;
            rz += kz * ds;
            j += (ky < 0.0) ? -1 : 1;
        }
        else
        {
            rx += kx * ds;
            ry += ky * ds;
            rz = zE;
            k += (kz < 0.0) ? -1 : 1;
        }
    }

    countOutv[0] = count;
}

static __device__ bool skirt_tree_contains(const double* boundsv, int node, double x, double y, double z)
{
    int base = 6 * node;
    return x >= boundsv[base] && x <= boundsv[base + 3] && y >= boundsv[base + 1] && y <= boundsv[base + 4]
           && z >= boundsv[base + 2] && z <= boundsv[base + 5];
}

static __device__ int skirt_tree_find_leaf(const double* boundsv, const int* childBeginv,
                                           const int* childCountv, const int* childIndexv, int numNodes,
                                           double x, double y, double z)
{
    if (numNodes <= 0 || !skirt_tree_contains(boundsv, 0, x, y, z)) return -1;

    int node = 0;
    for (int guard = 0; guard != numNodes; ++guard)
    {
        int count = childCountv[node];
        if (count <= 0) return node;

        int next = -1;
        int begin = childBeginv[node];
        for (int i = count - 1; i >= 0; --i)
        {
            int child = childIndexv[begin + i];
            if (child >= 0 && child < numNodes && skirt_tree_contains(boundsv, child, x, y, z))
            {
                next = child;
                break;
            }
        }
        if (next < 0) return -1;
        node = next;
    }
    return -1;
}

extern "C" __global__ void tree_grid_path(const double* boundsv, const int* childBeginv,
                                          const int* childCountv, const int* childIndexv,
                                          const int* cellIndexv, int numNodes,
                                          double rx, double ry, double rz,
                                          double kx, double ky, double kz,
                                          double xmin, double ymin, double zmin,
                                          double xmax, double ymax, double zmax,
                                          double eps, double maxDistance, int maxSegments,
                                          int* cellOutv, double* dsOutv, int* countOutv,
                                          int* statusOutv)
{
    countOutv[0] = 0;
    statusOutv[0] = 1;
    if (numNodes <= 0 || maxSegments <= 0) return;

    double cumds = 0.0;

    if (rx <= xmin)
    {
        if (kx <= 0.0) return;
        double ds = (xmin - rx) / kx;
        rx = xmin + eps;
        ry += ky * ds;
        rz += kz * ds;
        cumds += ds;
    }
    else if (rx >= xmax)
    {
        if (kx >= 0.0) return;
        double ds = (xmax - rx) / kx;
        rx = xmax - eps;
        ry += ky * ds;
        rz += kz * ds;
        cumds += ds;
    }

    if (ry <= ymin)
    {
        if (ky <= 0.0) return;
        double ds = (ymin - ry) / ky;
        rx += kx * ds;
        ry = ymin + eps;
        rz += kz * ds;
        cumds += ds;
    }
    else if (ry >= ymax)
    {
        if (ky >= 0.0) return;
        double ds = (ymax - ry) / ky;
        rx += kx * ds;
        ry = ymax - eps;
        rz += kz * ds;
        cumds += ds;
    }

    if (rz <= zmin)
    {
        if (kz <= 0.0) return;
        double ds = (zmin - rz) / kz;
        rx += kx * ds;
        ry += ky * ds;
        rz = zmin + eps;
        cumds += ds;
    }
    else if (rz >= zmax)
    {
        if (kz >= 0.0) return;
        double ds = (zmax - rz) / kz;
        rx += kx * ds;
        ry += ky * ds;
        rz = zmax - eps;
        cumds += ds;
    }

    if (!skirt_tree_contains(boundsv, 0, rx, ry, rz)) return;

    int count = 0;
    double distance = 0.0;
    int node = skirt_tree_find_leaf(boundsv, childBeginv, childCountv, childIndexv, numNodes, rx, ry, rz);
    if (node < 0) return;

    if (cumds > 0.0)
    {
        if (count >= maxSegments)
        {
            statusOutv[0] = -1;
            return;
        }
        cellOutv[count] = -1;
        dsOutv[count] = cumds;
        ++count;
        distance += cumds;
        if (distance > maxDistance)
        {
            countOutv[0] = count;
            return;
        }
    }

    const double big = 1.7976931348623157e+308;
    while (node >= 0)
    {
        int base = 6 * node;
        double xnext = (kx < 0.0) ? boundsv[base] : boundsv[base + 3];
        double ynext = (ky < 0.0) ? boundsv[base + 1] : boundsv[base + 4];
        double znext = (kz < 0.0) ? boundsv[base + 2] : boundsv[base + 5];
        double dsx = (fabs(kx) > 1e-15) ? (xnext - rx) / kx : big;
        double dsy = (fabs(ky) > 1e-15) ? (ynext - ry) / ky : big;
        double dsz = (fabs(kz) > 1e-15) ? (znext - rz) / kz : big;

        double ds = dsz;
        if (dsx <= dsy && dsx <= dsz)
            ds = dsx;
        else if (dsy <= dsx && dsy <= dsz)
            ds = dsy;

        if (ds > 0.0)
        {
            if (count >= maxSegments)
            {
                statusOutv[0] = -1;
                return;
            }
            cellOutv[count] = cellIndexv[node];
            dsOutv[count] = ds;
            ++count;
            distance += ds;
            if (distance > maxDistance)
            {
                countOutv[0] = count;
                return;
            }
        }

        int oldNode = node;
        double step = ds + eps;
        rx += kx * step;
        ry += ky * step;
        rz += kz * step;
        node = skirt_tree_find_leaf(boundsv, childBeginv, childCountv, childIndexv, numNodes, rx, ry, rz);
        if (node == oldNode)
        {
            rx = nextafter(rx, (kx < 0.0) ? -big : big);
            ry = nextafter(ry, (ky < 0.0) ? -big : big);
            rz = nextafter(rz, (kz < 0.0) ? -big : big);
            node = skirt_tree_find_leaf(boundsv, childBeginv, childCountv, childIndexv, numNodes, rx, ry, rz);
        }
        if (node == oldNode) node = -1;
    }

    countOutv[0] = count;
}

extern "C" __global__ void sphere1d_grid_path(const double* rv, int nr,
                                              double rx, double ry, double rz,
                                              double kx, double ky, double kz,
                                              double maxDistance, int maxSegments,
                                              int* cellOutv, double* dsOutv,
                                              int* countOutv, int* statusOutv)
{
    countOutv[0] = 0;
    statusOutv[0] = 1;
    if (nr <= 0 || maxSegments <= 0) return;

    double rmin = rv[0];
    double rmax = rv[nr];
    double r = sqrt(rx * rx + ry * ry + rz * rz);
    double q = rx * kx + ry * ky + rz * kz;
    double p = sqrt((r - q) * (r + q));
    int i = -1;
    int imin = -1;
    int phase = 1;  // 0 = inwards, 1 = outwards
    int count = 0;
    double distance = 0.0;

    if (r > rmax)
    {
        if (q > 0.0 || p > rmax) return;

        double qmax = -sqrt((rmax - p) * (rmax + p));
        double ds = qmax - q;
        i = nr - 1;
        q = qmax;
        imin = skirt_locate_array(rv, p, nr + 1);
        phase = (i > imin) ? 0 : 1;
        if (ds > 0.0)
        {
            if (count >= maxSegments)
            {
                statusOutv[0] = -1;
                return;
            }
            cellOutv[count] = -1;
            dsOutv[count] = ds;
            ++count;
            distance += ds;
            if (distance > maxDistance)
            {
                countOutv[0] = count;
                return;
            }
        }
    }
    else if (r < rmin)
    {
        double qmin = sqrt((rmin - p) * (rmin + p));
        double ds = qmin - q;
        i = 0;
        q = qmin;
        phase = 1;
        if (ds > 0.0)
        {
            if (count >= maxSegments)
            {
                statusOutv[0] = -1;
                return;
            }
            cellOutv[count] = -1;
            dsOutv[count] = ds;
            ++count;
            distance += ds;
            if (distance > maxDistance)
            {
                countOutv[0] = count;
                return;
            }
        }
    }
    else
    {
        i = skirt_locate_clip(rv, r, nr + 1);
        phase = 1;
        if (q < 0.0)
        {
            imin = skirt_locate_array(rv, p, nr + 1);
            if (i > imin) phase = 0;
        }
    }

    while (i < nr)
    {
        double ds = 0.0;
        int m = i;
        if (phase == 0)
        {
            double rN = rv[i];
            double qN = -sqrt((rN - p) * (rN + p));
            ds = qN - q;
            --i;
            q = qN;
            if (i <= imin) phase = 1;
        }
        else
        {
            double rN = rv[i + 1];
            double qN = sqrt((rN - p) * (rN + p));
            ds = qN - q;
            ++i;
            q = qN;
        }

        if (ds > 0.0)
        {
            if (count >= maxSegments)
            {
                statusOutv[0] = -1;
                return;
            }
            cellOutv[count] = m;
            dsOutv[count] = ds;
            ++count;
            distance += ds;
            if (distance > maxDistance)
            {
                countOutv[0] = count;
                return;
            }
        }
    }

    countOutv[0] = count;
}

static __device__ bool skirt_sphere2d_set_cell_indices(const double* rv, const double* thetav, int nr, int ntheta,
                                                       double rx, double ry, double rz, int* ip, int* jp)
{
    double radius = sqrt(rx * rx + ry * ry + rz * rz);
    double theta = radius == 0.0 ? 0.0 : acos(rz / radius);
    *ip = skirt_locate_array(rv, radius, nr + 1);
    *jp = skirt_locate_clip(thetav, theta, ntheta + 1);
    return *ip < nr;
}

extern "C" __global__ void sphere2d_grid_path(const double* rv, const double* thetav, const double* cv,
                                              int nr, int ntheta,
                                              double rx, double ry, double rz,
                                              double kx, double ky, double kz,
                                              double maxDistance, int maxSegments,
                                              int* cellOutv, double* dsOutv,
                                              int* countOutv, int* statusOutv)
{
    countOutv[0] = 0;
    statusOutv[0] = 1;
    if (nr <= 0 || ntheta <= 0 || maxSegments <= 0) return;

    double rmax = rv[nr];
    double eps = 1e-11 * rmax;
    double r = sqrt(rx * rx + ry * ry + rz * rz);
    int i = -1;
    int j = -1;
    int count = 0;
    double distance = 0.0;

    if (r > rmax)
    {
        double s = skirt_quadratic_smallest_positive(rx * kx + ry * ky + rz * kz, r * r - rmax * rmax);
        if (s <= 0.0) return;
        rx += kx * (s + eps);
        ry += ky * (s + eps);
        rz += kz * (s + eps);
        if (!skirt_sphere2d_set_cell_indices(rv, thetav, nr, ntheta, rx, ry, rz, &i, &j)) return;
        if (s > 0.0)
        {
            if (count >= maxSegments)
            {
                statusOutv[0] = -1;
                return;
            }
            cellOutv[count] = -1;
            dsOutv[count] = s;
            ++count;
            distance += s;
            if (distance > maxDistance)
            {
                countOutv[0] = count;
                return;
            }
        }
    }
    else
    {
        if (r < eps)
        {
            rx += kx * eps;
            ry += ky * eps;
            rz += kz * eps;
        }
        if (!skirt_sphere2d_set_cell_indices(rv, thetav, nr, ntheta, rx, ry, rz, &i, &j)) return;
    }

    int guard = 0;
    int maxGuard = 16 * (nr + ntheta + 1);
    while (guard++ < maxGuard)
    {
        int icur = i;
        int jcur = j;
        double ds = 1.7976931348623157e+308;

        if (icur > 0 || (icur == 0 && rv[0] > 0.0))
        {
            double rB = rv[icur];
            double s =
                skirt_quadratic_smallest_positive(rx * kx + ry * ky + rz * kz, rx * rx + ry * ry + rz * rz - rB * rB);
            if (s > 0.0 && s < ds)
            {
                ds = s;
                i = icur - 1;
                j = jcur;
            }
        }

        {
            double rB = rv[icur + 1];
            double s =
                skirt_quadratic_smallest_positive(rx * kx + ry * ky + rz * kz, rx * rx + ry * ry + rz * rz - rB * rB);
            if (s > 0.0 && s < ds)
            {
                ds = s;
                i = icur + 1;
                j = jcur;
            }
        }

        if (jcur > 0)
        {
            double c = cv[jcur];
            double s = c ? skirt_smallest_positive_scaled(c * c - kz * kz,
                                                          c * c * (rx * kx + ry * ky + rz * kz) - rz * kz,
                                                          c * c * (rx * rx + ry * ry + rz * rz) - rz * rz)
                         : -rz / kz;
            if (s > 0.0 && s < ds)
            {
                ds = s;
                i = icur;
                j = jcur - 1;
            }
        }

        if (jcur < ntheta - 1)
        {
            double c = cv[jcur + 1];
            double s = c ? skirt_smallest_positive_scaled(c * c - kz * kz,
                                                          c * c * (rx * kx + ry * ky + rz * kz) - rz * kz,
                                                          c * c * (rx * rx + ry * ry + rz * rz) - rz * rz)
                         : -rz / kz;
            if (s > 0.0 && s < ds)
            {
                ds = s;
                i = icur;
                j = jcur + 1;
            }
        }

        if (i != icur || j != jcur)
        {
            if (ds > 0.0)
            {
                if (count >= maxSegments)
                {
                    statusOutv[0] = -1;
                    return;
                }
                cellOutv[count] = jcur + ntheta * icur;
                dsOutv[count] = ds;
                ++count;
                distance += ds;
                if (distance > maxDistance)
                {
                    countOutv[0] = count;
                    return;
                }
            }
            rx += kx * (ds + eps);
            ry += ky * (ds + eps);
            rz += kz * (ds + eps);
            if (i >= nr) break;
        }
        else
        {
            rx += kx * eps;
            ry += ky * eps;
            rz += kz * eps;
            if (!skirt_sphere2d_set_cell_indices(rv, thetav, nr, ntheta, rx, ry, rz, &i, &j)) break;
        }
    }

    countOutv[0] = count;
}

static __device__ bool skirt_sphere3d_set_cell_indices(const double* rv, const double* thetav, const double* phiv,
                                                       int nr, int ntheta, int nphi, double rx, double ry,
                                                       double rz, int* ip, int* jp, int* kp)
{
    double radius = sqrt(rx * rx + ry * ry + rz * rz);
    double theta = radius == 0.0 ? 0.0 : acos(rz / radius);
    double phi = radius == 0.0 ? 0.0 : atan2(ry, rx);
    *ip = skirt_locate_array(rv, radius, nr + 1);
    *jp = skirt_locate_clip(thetav, theta, ntheta + 1);
    *kp = skirt_locate_clip(phiv, phi, nphi + 1);
    return *ip < nr;
}

static __device__ double skirt_sphere3d_intersection_meridional_plane(const double* sinv, const double* cosv, int k,
                                                                      double rx, double ry, double kx, double ky)
{
    double q = kx * sinv[k] - ky * cosv[k];
    if (fabs(q) < 1e-12) return 0.0;
    return -(rx * sinv[k] - ry * cosv[k]) / q;
}

extern "C" __global__ void sphere3d_grid_path(const double* rv, const double* thetav, const double* phiv,
                                              const double* cv, const double* sinv, const double* cosv,
                                              int nr, int ntheta, int nphi, double eps,
                                              double rx, double ry, double rz,
                                              double kx, double ky, double kz,
                                              double maxDistance, int maxSegments,
                                              int* cellOutv, double* dsOutv,
                                              int* countOutv, int* statusOutv)
{
    countOutv[0] = 0;
    statusOutv[0] = 1;
    if (nr <= 0 || ntheta <= 0 || nphi <= 0 || maxSegments <= 0) return;

    int i = -1;
    int j = -1;
    int k = -1;
    int count = 0;
    double distance = 0.0;
    double radius = sqrt(rx * rx + ry * ry + rz * rz);

    if (radius > rv[nr])
    {
        double ds = skirt_quadratic_smallest_positive(rx * kx + ry * ky + rz * kz,
                                                      radius * radius - rv[nr] * rv[nr]);
        if (ds <= 0.0) return;

        rx += kx * (ds + eps);
        ry += ky * (ds + eps);
        rz += kz * (ds + eps);
        if (!skirt_sphere3d_set_cell_indices(rv, thetav, phiv, nr, ntheta, nphi, rx, ry, rz, &i, &j, &k)) return;

        if (count >= maxSegments)
        {
            statusOutv[0] = -1;
            return;
        }
        cellOutv[count] = -1;
        dsOutv[count] = ds;
        ++count;
        distance += ds;
        if (distance > maxDistance)
        {
            countOutv[0] = count;
            return;
        }
    }
    else
    {
        if (!skirt_sphere3d_set_cell_indices(rv, thetav, phiv, nr, ntheta, nphi, rx, ry, rz, &i, &j, &k)) return;
    }

    int guard = 0;
    int maxGuard = 24 * (nr + ntheta + nphi + 1);
    while (guard++ < maxGuard)
    {
        if (i >= 0)
        {
            int icur = i;
            int jcur = j;
            int kcur = k;
            double ds = 1.7976931348623157e+308;
            double r2 = rx * rx + ry * ry + rz * rz;
            double rk = rx * kx + ry * ky + rz * kz;

            {
                double rB = rv[icur];
                double s = skirt_quadratic_smallest_positive(rk, r2 - rB * rB);
                if (s > 0.0 && s < ds)
                {
                    ds = s;
                    i = icur - 1;
                    j = jcur;
                    k = kcur;
                }
            }

            {
                double rB = rv[icur + 1];
                double s = skirt_quadratic_smallest_positive(rk, r2 - rB * rB);
                if (s > 0.0 && s < ds)
                {
                    ds = s;
                    i = icur + 1;
                    j = jcur;
                    k = kcur;
                }
            }

            if (jcur > 0)
            {
                double c = cv[jcur];
                double s = c ? skirt_smallest_positive_scaled(c * c - kz * kz, c * c * rk - rz * kz,
                                                              c * c * r2 - rz * rz)
                             : -rz / kz;
                if (s > 0.0 && s < ds)
                {
                    ds = s;
                    i = icur;
                    j = jcur - 1;
                    k = kcur;
                }
            }

            if (jcur < ntheta - 1)
            {
                double c = cv[jcur + 1];
                double s = c ? skirt_smallest_positive_scaled(c * c - kz * kz, c * c * rk - rz * kz,
                                                              c * c * r2 - rz * rz)
                             : -rz / kz;
                if (s > 0.0 && s < ds)
                {
                    ds = s;
                    i = icur;
                    j = jcur + 1;
                    k = kcur;
                }
            }

            {
                double s = skirt_sphere3d_intersection_meridional_plane(sinv, cosv, kcur, rx, ry, kx, ky);
                if (s > 0.0 && s < ds)
                {
                    ds = s;
                    i = icur;
                    j = jcur;
                    k = kcur > 0 ? kcur - 1 : nphi - 1;
                }
            }

            {
                double s = skirt_sphere3d_intersection_meridional_plane(sinv, cosv, kcur + 1, rx, ry, kx, ky);
                if (s > 0.0 && s < ds)
                {
                    ds = s;
                    i = icur;
                    j = jcur;
                    k = (kcur + 1) % nphi;
                }
            }

            if (ds == 1.7976931348623157e+308) break;

            if (count >= maxSegments)
            {
                statusOutv[0] = -1;
                return;
            }
            cellOutv[count] = kcur + (jcur + icur * ntheta) * nphi;
            dsOutv[count] = ds;
            ++count;
            distance += ds;
            if (distance > maxDistance)
            {
                countOutv[0] = count;
                return;
            }

            rx += kx * (ds + eps);
            ry += ky * (ds + eps);
            rz += kz * (ds + eps);
            if (i >= nr) break;
        }
        else
        {
            double r2 = rx * rx + ry * ry + rz * rz;
            double ds = skirt_quadratic_smallest_positive(rx * kx + ry * ky + rz * kz, r2 - rv[0] * rv[0]);
            if (ds <= 0.0) break;

            if (count >= maxSegments)
            {
                statusOutv[0] = -1;
                return;
            }
            cellOutv[count] = -1;
            dsOutv[count] = ds;
            ++count;
            distance += ds;
            if (distance > maxDistance)
            {
                countOutv[0] = count;
                return;
            }

            rx += kx * (ds + eps);
            ry += ky * (ds + eps);
            rz += kz * (ds + eps);
            if (!skirt_sphere3d_set_cell_indices(rv, thetav, phiv, nr, ntheta, nphi, rx, ry, rz, &i, &j, &k)) break;
        }
    }

    countOutv[0] = count;
}

extern "C" __global__ void cylinder2d_grid_path(const double* Rv, const double* zv, int nR, int nz,
                                                double rx, double ry, double rz,
                                                double kx, double ky, double kzInput,
                                                double maxDistance, int maxSegments,
                                                int* cellOutv, double* dsOutv,
                                                int* countOutv, int* statusOutv)
{
    countOutv[0] = 0;
    statusOutv[0] = 1;
    if (nR <= 0 || nz <= 0 || maxSegments <= 0) return;

    double R = sqrt(rx * rx + ry * ry);
    double z = rz;
    double kq = sqrt(kx * kx + ky * ky);
    double kz = kzInput;
    if (kq == 0.0) kq = 1e-20;
    if (kz == 0.0) kz = 1e-20;
    double q = (rx * kx + ry * ky) / kq;
    double p = sqrt(fmax(0.0, (R - q) * (R + q)));
    double Rmax = Rv[nR];
    double zmin = zv[0];
    double zmax = zv[nz];
    double cumds = 0.0;
    int count = 0;
    double distance = 0.0;

    if (R >= Rmax)
    {
        if (q > 0.0 || p > Rmax) return;
        double qmin = -sqrt((Rmax - p) * (Rmax + p));
        double ds = (qmin - q) / kq;
        q = qmin;
        R = Rmax - 1e-8 * (Rv[nR] - Rv[nR - 1]);
        z += kz * ds;
        cumds += ds;
    }

    if (z < zmin)
    {
        if (kz <= 0.0) return;
        double ds = (zmin - z) / kz;
        q += kq * ds;
        R = sqrt(p * p + q * q);
        z = zmin + 1e-8 * (zv[1] - zv[0]);
        cumds += ds;
    }
    else if (z > zmax)
    {
        if (kz >= 0.0) return;
        double ds = (zmax - z) / kz;
        q += kq * ds;
        R = sqrt(p * p + q * q);
        z = zmax - 1e-8 * (zv[nz] - zv[nz - 1]);
        cumds += ds;
    }

    if (isinf(R) || isnan(R) || isinf(z) || isnan(z)) return;
    if (R >= Rmax || z <= zmin || z >= zmax) return;

    if (cumds > 0.0)
    {
        if (count >= maxSegments)
        {
            statusOutv[0] = -1;
            return;
        }
        cellOutv[count] = -1;
        dsOutv[count] = cumds;
        ++count;
        distance += cumds;
        if (distance > maxDistance)
        {
            countOutv[0] = count;
            return;
        }
    }

    int i = skirt_locate_array(Rv, R, nR + 1);
    int k = skirt_locate_clip(zv, z, nz + 1);
    int imin = -1;
    int phase = 1;  // 0 UpInwards, 1 UpOutwards, 2 DownInwards, 3 DownOutwards
    if (kz >= 0.0)
    {
        phase = 1;
        if (q < 0.0)
        {
            imin = skirt_locate_array(Rv, p, nR + 1);
            if (i > imin) phase = 0;
        }
    }
    else
    {
        phase = 3;
        if (q < 0.0)
        {
            imin = skirt_locate_array(Rv, p, nR + 1);
            if (i > imin) phase = 2;
        }
    }

    int guard = 0;
    int maxGuard = 8 * (nR + nz + 1);
    while (guard++ < maxGuard)
    {
        int m = i < 0 ? -1 : k + nz * i;
        double ds = 0.0;
        if (phase == 0)
        {
            double RN = Rv[i];
            double qN = -sqrt((RN - p) * (RN + p));
            double zN = zv[k + 1];
            double dsq = (qN - q) / kq;
            double dsz = (zN - z) / kz;
            if (dsq < dsz)
            {
                ds = dsq;
                --i;
                q = qN;
                z += kz * ds;
                if (i <= imin) phase = 1;
            }
            else
            {
                ds = dsz;
                ++k;
                if (k < nz)
                {
                    q += kq * ds;
                    z = zN;
                }
            }
        }
        else if (phase == 1)
        {
            double RN = Rv[i + 1];
            double qN = sqrt((RN - p) * (RN + p));
            double zN = zv[k + 1];
            double dsq = (qN - q) / kq;
            double dsz = (zN - z) / kz;
            if (dsq < dsz)
            {
                ds = dsq;
                ++i;
                if (i < nR)
                {
                    q = qN;
                    z += kz * ds;
                }
            }
            else
            {
                ds = dsz;
                ++k;
                if (k < nz)
                {
                    q += kq * ds;
                    z = zN;
                }
            }
        }
        else if (phase == 2)
        {
            double RN = Rv[i];
            double qN = -sqrt((RN - p) * (RN + p));
            double zN = zv[k];
            double dsq = (qN - q) / kq;
            double dsz = (zN - z) / kz;
            if (dsq < dsz)
            {
                ds = dsq;
                --i;
                q = qN;
                z += kz * ds;
                if (i <= imin) phase = 3;
            }
            else
            {
                ds = dsz;
                --k;
                if (k >= 0)
                {
                    q += kq * ds;
                    z = zN;
                }
            }
        }
        else
        {
            double RN = Rv[i + 1];
            double qN = sqrt((RN - p) * (RN + p));
            double zN = zv[k];
            double dsq = (qN - q) / kq;
            double dsz = (zN - z) / kz;
            if (dsq < dsz)
            {
                ds = dsq;
                ++i;
                if (i < nR)
                {
                    q = qN;
                    z += kz * ds;
                }
            }
            else
            {
                ds = dsz;
                --k;
                if (k >= 0)
                {
                    q += kq * ds;
                    z = zN;
                }
            }
        }

        if (ds > 0.0)
        {
            if (count >= maxSegments)
            {
                statusOutv[0] = -1;
                return;
            }
            cellOutv[count] = m;
            dsOutv[count] = ds;
            ++count;
            distance += ds;
            if (distance > maxDistance)
            {
                countOutv[0] = count;
                return;
            }
        }

        if (i >= nR || k < 0 || k >= nz) break;
    }

    countOutv[0] = count;
}

static __device__ bool skirt_cylinder3d_set_cell_indices(const double* Rv, const double* phiv, const double* zv,
                                                         int nR, int nphi, int nz, double rx, double ry, double rz,
                                                         int* ip, int* jp, int* kp)
{
    double R = sqrt(rx * rx + ry * ry);
    double phi = atan2(ry, rx);
    *ip = skirt_locate_array(Rv, R, nR + 1);
    *jp = skirt_locate_clip(phiv, phi, nphi + 1);
    *kp = skirt_locate_fail(zv, rz, nz + 1);
    return *ip < nR && *kp >= 0;
}

static __device__ double skirt_cylinder3d_first_intersection_cylinder(const double* Rv, int i, double kq2,
                                                                      double rx, double ry, double kx, double ky)
{
    if (fabs(kq2) < 1e-12) return 0.0;
    double b = rx * kx + ry * ky;
    double c = rx * rx + ry * ry - Rv[i] * Rv[i];
    return skirt_quadratic_smallest_positive(b / kq2, c / kq2);
}

static __device__ double skirt_cylinder3d_intersection_meridional_plane(const double* sinv, const double* cosv,
                                                                        int j, double rx, double ry, double kx,
                                                                        double ky)
{
    double q = kx * sinv[j] - ky * cosv[j];
    if (fabs(q) < 1e-12) return 0.0;
    return -(rx * sinv[j] - ry * cosv[j]) / q;
}

static __device__ double skirt_cylinder3d_intersection_horizontal_plane(const double* zv, int k, double rz, double kz)
{
    if (fabs(kz) < 1e-12) return 0.0;
    return (zv[k] - rz) / kz;
}

extern "C" __global__ void cylinder3d_grid_path(const double* Rv, const double* phiv, const double* zv,
                                                const double* sinv, const double* cosv,
                                                int nR, int nphi, int nz, double eps, int hasHole,
                                                double rx, double ry, double rz,
                                                double kx, double ky, double kz,
                                                double maxDistance, int maxSegments,
                                                int* cellOutv, double* dsOutv,
                                                int* countOutv, int* statusOutv)
{
    countOutv[0] = 0;
    statusOutv[0] = 1;
    if (nR <= 0 || nphi <= 0 || nz <= 0 || maxSegments <= 0) return;

    double kq2 = kx * kx + ky * ky;
    int i = -1;
    int j = -1;
    int k = -1;
    int count = 0;
    double distance = 0.0;
    double cumds = 0.0;

    double R = sqrt(rx * rx + ry * ry);
    if (R > Rv[nR])
    {
        double ds = skirt_cylinder3d_first_intersection_cylinder(Rv, nR, kq2, rx, ry, kx, ky);
        if (ds <= 0.0) return;
        rx += kx * (ds + eps);
        ry += ky * (ds + eps);
        rz += kz * (ds + eps);
        cumds += ds;
    }

    if (rz < zv[0])
    {
        double ds = skirt_cylinder3d_intersection_horizontal_plane(zv, 0, rz, kz);
        if (ds <= 0.0) return;
        rx += kx * (ds + eps);
        ry += ky * (ds + eps);
        rz += kz * (ds + eps);
        cumds += ds;
    }
    else if (rz > zv[nz])
    {
        double ds = skirt_cylinder3d_intersection_horizontal_plane(zv, nz, rz, kz);
        if (ds <= 0.0) return;
        rx += kx * (ds + eps);
        ry += ky * (ds + eps);
        rz += kz * (ds + eps);
        cumds += ds;
    }

    if (!skirt_cylinder3d_set_cell_indices(Rv, phiv, zv, nR, nphi, nz, rx, ry, rz, &i, &j, &k)) return;

    if (cumds > 0.0)
    {
        if (count >= maxSegments)
        {
            statusOutv[0] = -1;
            return;
        }
        cellOutv[count] = -1;
        dsOutv[count] = cumds;
        ++count;
        distance += cumds;
        if (distance > maxDistance)
        {
            countOutv[0] = count;
            return;
        }
    }

    int guard = 0;
    int maxGuard = 12 * (nR + nphi + nz + 1);
    while (guard++ < maxGuard)
    {
        if (i >= 0)
        {
            int icur = i;
            int jcur = j;
            int kcur = k;
            double ds = 1.7976931348623157e+308;

            {
                double s = skirt_cylinder3d_first_intersection_cylinder(Rv, icur, kq2, rx, ry, kx, ky);
                if (s > 0.0 && s < ds)
                {
                    ds = s;
                    i = icur - 1;
                    j = jcur;
                    k = kcur;
                }
            }

            {
                double s = skirt_cylinder3d_first_intersection_cylinder(Rv, icur + 1, kq2, rx, ry, kx, ky);
                if (s > 0.0 && s < ds)
                {
                    ds = s;
                    i = icur + 1;
                    j = jcur;
                    k = kcur;
                }
            }

            {
                double s = skirt_cylinder3d_intersection_meridional_plane(sinv, cosv, jcur, rx, ry, kx, ky);
                if (s > 0.0 && s < ds)
                {
                    ds = s;
                    i = icur;
                    j = jcur > 0 ? jcur - 1 : nphi - 1;
                    k = kcur;
                }
            }

            {
                double s = skirt_cylinder3d_intersection_meridional_plane(sinv, cosv, jcur + 1, rx, ry, kx, ky);
                if (s > 0.0 && s < ds)
                {
                    ds = s;
                    i = icur;
                    j = (jcur + 1) % nphi;
                    k = kcur;
                }
            }

            {
                double s = skirt_cylinder3d_intersection_horizontal_plane(zv, kcur, rz, kz);
                if (s > 0.0 && s < ds)
                {
                    ds = s;
                    i = icur;
                    j = jcur;
                    k = kcur - 1;
                }
            }

            {
                double s = skirt_cylinder3d_intersection_horizontal_plane(zv, kcur + 1, rz, kz);
                if (s > 0.0 && s < ds)
                {
                    ds = s;
                    i = icur;
                    j = jcur;
                    k = kcur + 1;
                }
            }

            if (ds == 1.7976931348623157e+308) break;

            if (count >= maxSegments)
            {
                statusOutv[0] = -1;
                return;
            }
            cellOutv[count] = kcur + (jcur + icur * nphi) * nz;
            dsOutv[count] = ds;
            ++count;
            distance += ds;
            if (distance > maxDistance)
            {
                countOutv[0] = count;
                return;
            }

            rx += kx * (ds + eps);
            ry += ky * (ds + eps);
            rz += kz * (ds + eps);
            if (i >= nR || k < 0 || k >= nz) break;
        }
        else if (!hasHole)
        {
            int kcur = k;
            double ds = 1.7976931348623157e+308;

            {
                double s = skirt_cylinder3d_first_intersection_cylinder(Rv, 0, kq2, rx, ry, kx, ky);
                if (s > 0.0 && s < ds) ds = s;
            }

            {
                double s = skirt_cylinder3d_intersection_horizontal_plane(zv, kcur, rz, kz);
                if (s > 0.0 && s < ds)
                {
                    ds = s;
                    k = kcur - 1;
                }
            }

            {
                double s = skirt_cylinder3d_intersection_horizontal_plane(zv, kcur + 1, rz, kz);
                if (s > 0.0 && s < ds)
                {
                    ds = s;
                    k = kcur + 1;
                }
            }

            if (ds == 1.7976931348623157e+308) break;

            if (count >= maxSegments)
            {
                statusOutv[0] = -1;
                return;
            }
            cellOutv[count] = kcur + j * nz;
            dsOutv[count] = ds;
            ++count;
            distance += ds;
            if (distance > maxDistance)
            {
                countOutv[0] = count;
                return;
            }

            rx += kx * (ds + eps);
            ry += ky * (ds + eps);
            rz += kz * (ds + eps);
            if (k == kcur)
            {
                if (!skirt_cylinder3d_set_cell_indices(Rv, phiv, zv, nR, nphi, nz, rx, ry, rz, &i, &j, &k)) break;
            }
            else if (k < 0 || k >= nz)
            {
                break;
            }
        }
        else
        {
            double ds = skirt_cylinder3d_first_intersection_cylinder(Rv, 0, kq2, rx, ry, kx, ky);
            if (ds <= 0.0) break;

            if (count >= maxSegments)
            {
                statusOutv[0] = -1;
                return;
            }
            cellOutv[count] = -1;
            dsOutv[count] = ds;
            ++count;
            distance += ds;
            if (distance > maxDistance)
            {
                countOutv[0] = count;
                return;
            }

            rx += kx * (ds + eps);
            ry += ky * (ds + eps);
            rz += kz * (ds + eps);
            if (!skirt_cylinder3d_set_cell_indices(Rv, phiv, zv, nR, nphi, nz, rx, ry, rz, &i, &j, &k)) break;
        }
    }

    countOutv[0] = count;
}

static __device__ bool skirt_tetra_contains(const double* vertexv, const int* faceAnchorv, const double* faceNormalv,
                                            int cell, double rx, double ry, double rz)
{
    for (int f = 0; f != 4; ++f)
    {
        int anchor = faceAnchorv[4 * cell + f];
        int vertexBase = 3 * anchor;
        double vx = vertexv[vertexBase];
        double vy = vertexv[vertexBase + 1];
        double vz = vertexv[vertexBase + 2];
        int normalBase = 12 * cell + 3 * f;
        double nx = faceNormalv[normalBase];
        double ny = faceNormalv[normalBase + 1];
        double nz = faceNormalv[normalBase + 2];
        double side = (vx - rx) * nx + (vy - ry) * ny + (vz - rz) * nz;
        if (side < 0.0) return false;
    }
    return true;
}

static __device__ int skirt_tetra_find_cell(const double* vertexv, const int* faceAnchorv, const double* faceNormalv,
                                            int numCells, double rx, double ry, double rz)
{
    for (int m = 0; m != numCells; ++m)
        if (skirt_tetra_contains(vertexv, faceAnchorv, faceNormalv, m, rx, ry, rz)) return m;
    return -1;
}

extern "C" __global__ void tetra_mesh_grid_path(const double* vertexv, const int* faceAnchorv,
                                                const int* faceNeighborv, const double* faceNormalv,
                                                const double* centroidv, int numCells, double eps,
                                                double rx, double ry, double rz,
                                                double kx, double ky, double kz,
                                                double xmin, double ymin, double zmin,
                                                double xmax, double ymax, double zmax,
                                                double maxDistance, int maxSegments,
                                                int* cellOutv, double* dsOutv,
                                                int* countOutv, int* statusOutv)
{
    countOutv[0] = 0;
    statusOutv[0] = 1;
    if (numCells <= 0 || maxSegments <= 0) return;

    double cumds = 0.0;
    int count = 0;
    double distance = 0.0;

    if (rx <= xmin)
    {
        if (kx <= 0.0) return;
        double ds = (xmin - rx) / kx;
        rx = xmin + eps;
        ry += ky * ds;
        rz += kz * ds;
        cumds += ds;
    }
    else if (rx >= xmax)
    {
        if (kx >= 0.0) return;
        double ds = (xmax - rx) / kx;
        rx = xmax - eps;
        ry += ky * ds;
        rz += kz * ds;
        cumds += ds;
    }

    if (ry <= ymin)
    {
        if (ky <= 0.0) return;
        double ds = (ymin - ry) / ky;
        rx += kx * ds;
        ry = ymin + eps;
        rz += kz * ds;
        cumds += ds;
    }
    else if (ry >= ymax)
    {
        if (ky >= 0.0) return;
        double ds = (ymax - ry) / ky;
        rx += kx * ds;
        ry = ymax - eps;
        rz += kz * ds;
        cumds += ds;
    }

    if (rz <= zmin)
    {
        if (kz <= 0.0) return;
        double ds = (zmin - rz) / kz;
        rx += kx * ds;
        ry += ky * ds;
        rz = zmin + eps;
        cumds += ds;
    }
    else if (rz >= zmax)
    {
        if (kz >= 0.0) return;
        double ds = (zmax - rz) / kz;
        rx += kx * ds;
        ry += ky * ds;
        rz = zmax - eps;
        cumds += ds;
    }

    if (rx < xmin || rx > xmax || ry < ymin || ry > ymax || rz < zmin || rz > zmax) return;

    if (cumds > 0.0)
    {
        if (count >= maxSegments)
        {
            statusOutv[0] = -1;
            return;
        }
        cellOutv[count] = -1;
        dsOutv[count] = cumds;
        ++count;
        distance += cumds;
        if (distance > maxDistance)
        {
            countOutv[0] = count;
            return;
        }
    }

    int cell = skirt_tetra_find_cell(vertexv, faceAnchorv, faceNormalv, numCells, rx, ry, rz);
    if (cell < 0) return;

    int guard = 0;
    int maxGuard = 8 * (numCells + 1);
    while (guard++ < maxGuard)
    {
        int exitFace = -1;
        double ds = 1.7976931348623157e+308;
        for (int f = 0; f != 4; ++f)
        {
            int normalBase = 12 * cell + 3 * f;
            double nx = faceNormalv[normalBase];
            double ny = faceNormalv[normalBase + 1];
            double nz = faceNormalv[normalBase + 2];
            double ndotk = nx * kx + ny * ky + nz * kz;
            if (ndotk > 0.0)
            {
                int anchor = faceAnchorv[4 * cell + f];
                int vertexBase = 3 * anchor;
                double vx = vertexv[vertexBase];
                double vy = vertexv[vertexBase + 1];
                double vz = vertexv[vertexBase + 2];
                double s = (nx * (vx - rx) + ny * (vy - ry) + nz * (vz - rz)) / ndotk;
                if (s > 0.0 && s < ds)
                {
                    ds = s;
                    exitFace = f;
                }
            }
        }

        if (exitFace < 0 || ds < eps)
        {
            rx += kx * eps;
            ry += ky * eps;
            rz += kz * eps;
            cell = skirt_tetra_find_cell(vertexv, faceAnchorv, faceNormalv, numCells, rx, ry, rz);
            if (cell < 0) break;
            continue;
        }

        int nextCell = faceNeighborv[4 * cell + exitFace];
        if (nextCell < 0) break;

        if (count >= maxSegments)
        {
            statusOutv[0] = -1;
            return;
        }
        cellOutv[count] = cell;
        dsOutv[count] = ds;
        ++count;
        distance += ds;
        if (distance > maxDistance)
        {
            countOutv[0] = count;
            return;
        }

        rx += kx * ds;
        ry += ky * ds;
        rz += kz * ds;
        cell = nextCell;
    }

    countOutv[0] = count;
}

static __device__ int skirt_voronoi_find_cell(const double* sitev, int numCells, double rx, double ry, double rz,
                                              double xmin, double ymin, double zmin,
                                              double xmax, double ymax, double zmax)
{
    if (rx < xmin || rx > xmax || ry < ymin || ry > ymax || rz < zmin || rz > zmax) return -1;
    int best = -1;
    double bestDistance = 1.7976931348623157e+308;
    for (int m = 0; m != numCells; ++m)
    {
        int base = 3 * m;
        double dx = rx - sitev[base];
        double dy = ry - sitev[base + 1];
        double dz = rz - sitev[base + 2];
        double distance = dx * dx + dy * dy + dz * dz;
        if (distance < bestDistance)
        {
            best = m;
            bestDistance = distance;
        }
    }
    return best;
}

static __device__ int skirt_voronoi_find_cell_in_block(const double* sitev, int numCells,
                                                       const int* blockBeginv, const int* blockCountv,
                                                       const int* blockIndexv, int blockN,
                                                       double rx, double ry, double rz,
                                                       double xmin, double ymin, double zmin,
                                                       double xmax, double ymax, double zmax)
{
    if (blockN <= 0 || !blockBeginv || !blockCountv || !blockIndexv)
        return skirt_voronoi_find_cell(sitev, numCells, rx, ry, rz, xmin, ymin, zmin, xmax, ymax, zmax);
    if (rx < xmin || rx > xmax || ry < ymin || ry > ymax || rz < zmin || rz > zmax) return -1;

    int i = static_cast<int>(blockN * (rx - xmin) / (xmax - xmin));
    int j = static_cast<int>(blockN * (ry - ymin) / (ymax - ymin));
    int k = static_cast<int>(blockN * (rz - zmin) / (zmax - zmin));
    if (i < 0) i = 0;
    if (j < 0) j = 0;
    if (k < 0) k = 0;
    if (i >= blockN) i = blockN - 1;
    if (j >= blockN) j = blockN - 1;
    if (k >= blockN) k = blockN - 1;

    int block = i * blockN * blockN + j * blockN + k;
    int begin = blockBeginv[block];
    int count = blockCountv[block];
    if (begin < 0 || count <= 0)
        return skirt_voronoi_find_cell(sitev, numCells, rx, ry, rz, xmin, ymin, zmin, xmax, ymax, zmax);

    int best = -1;
    double bestDistance = 1.7976931348623157e+308;
    for (int p = 0; p != count; ++p)
    {
        int m = blockIndexv[begin + p];
        if (m < 0 || m >= numCells) continue;
        int base = 3 * m;
        double dx = rx - sitev[base];
        double dy = ry - sitev[base + 1];
        double dz = rz - sitev[base + 2];
        double distance = dx * dx + dy * dy + dz * dz;
        if (distance < bestDistance)
        {
            best = m;
            bestDistance = distance;
        }
    }
    return best >= 0 ? best : skirt_voronoi_find_cell(sitev, numCells, rx, ry, rz, xmin, ymin, zmin, xmax, ymax, zmax);
}

extern "C" __global__ void voronoi_mesh_grid_path(const double* sitev, const int* neighborBeginv,
                                                  const int* neighborCountv, const int* neighborIndexv,
                                                  int numCells, double eps,
                                                  double rx, double ry, double rz,
                                                  double kx, double ky, double kz,
                                                  double xmin, double ymin, double zmin,
                                                  double xmax, double ymax, double zmax,
                                                  double maxDistance, int maxSegments,
                                                  int* cellOutv, double* dsOutv,
                                                  int* countOutv, int* statusOutv)
{
    countOutv[0] = 0;
    statusOutv[0] = 1;
    if (numCells <= 0 || maxSegments <= 0) return;

    double cumds = 0.0;
    int count = 0;
    double distance = 0.0;

    if (rx <= xmin)
    {
        if (kx <= 0.0) return;
        double ds = (xmin - rx) / kx;
        rx = xmin + eps;
        ry += ky * ds;
        rz += kz * ds;
        cumds += ds;
    }
    else if (rx >= xmax)
    {
        if (kx >= 0.0) return;
        double ds = (xmax - rx) / kx;
        rx = xmax - eps;
        ry += ky * ds;
        rz += kz * ds;
        cumds += ds;
    }

    if (ry <= ymin)
    {
        if (ky <= 0.0) return;
        double ds = (ymin - ry) / ky;
        rx += kx * ds;
        ry = ymin + eps;
        rz += kz * ds;
        cumds += ds;
    }
    else if (ry >= ymax)
    {
        if (ky >= 0.0) return;
        double ds = (ymax - ry) / ky;
        rx += kx * ds;
        ry = ymax - eps;
        rz += kz * ds;
        cumds += ds;
    }

    if (rz <= zmin)
    {
        if (kz <= 0.0) return;
        double ds = (zmin - rz) / kz;
        rx += kx * ds;
        ry += ky * ds;
        rz = zmin + eps;
        cumds += ds;
    }
    else if (rz >= zmax)
    {
        if (kz >= 0.0) return;
        double ds = (zmax - rz) / kz;
        rx += kx * ds;
        ry += ky * ds;
        rz = zmax - eps;
        cumds += ds;
    }

    int cell = skirt_voronoi_find_cell(sitev, numCells, rx, ry, rz, xmin, ymin, zmin, xmax, ymax, zmax);
    if (cell < 0) return;

    if (cumds > 0.0)
    {
        if (count >= maxSegments)
        {
            statusOutv[0] = -1;
            return;
        }
        cellOutv[count] = -1;
        dsOutv[count] = cumds;
        ++count;
        distance += cumds;
        if (distance > maxDistance)
        {
            countOutv[0] = count;
            return;
        }
    }

    int guard = 0;
    int maxGuard = 8 * (numCells + 1);
    while (guard++ < maxGuard)
    {
        int begin = neighborBeginv[cell];
        int n = neighborCountv[cell];
        int nextCell = -99;
        double ds = 1.7976931348623157e+308;
        int siteBase = 3 * cell;
        double prx = sitev[siteBase];
        double pry = sitev[siteBase + 1];
        double prz = sitev[siteBase + 2];
        for (int i = 0; i != n; ++i)
        {
            int neighbor = neighborIndexv[begin + i];
            double si = 0.0;
            if (neighbor >= 0)
            {
                int neighborBase = 3 * neighbor;
                double pix = sitev[neighborBase];
                double piy = sitev[neighborBase + 1];
                double piz = sitev[neighborBase + 2];
                double nx = pix - prx;
                double ny = piy - pry;
                double nz = piz - prz;
                double ndotk = nx * kx + ny * ky + nz * kz;
                if (ndotk > 0.0)
                {
                    double px = 0.5 * (pix + prx);
                    double py = 0.5 * (piy + pry);
                    double pz = 0.5 * (piz + prz);
                    si = (nx * (px - rx) + ny * (py - ry) + nz * (pz - rz)) / ndotk;
                }
            }
            else
            {
                if (neighbor == -1)
                    si = (xmin - rx) / kx;
                else if (neighbor == -2)
                    si = (xmax - rx) / kx;
                else if (neighbor == -3)
                    si = (ymin - ry) / ky;
                else if (neighbor == -4)
                    si = (ymax - ry) / ky;
                else if (neighbor == -5)
                    si = (zmin - rz) / kz;
                else if (neighbor == -6)
                    si = (zmax - rz) / kz;
            }

            if (si > 0.0 && si < ds)
            {
                ds = si;
                nextCell = neighbor;
            }
        }

        if (nextCell == -99)
        {
            rx += kx * eps;
            ry += ky * eps;
            rz += kz * eps;
            cell = skirt_voronoi_find_cell(sitev, numCells, rx, ry, rz, xmin, ymin, zmin, xmax, ymax, zmax);
            if (cell < 0) break;
            continue;
        }

        if (count >= maxSegments)
        {
            statusOutv[0] = -1;
            return;
        }
        cellOutv[count] = cell;
        dsOutv[count] = ds;
        ++count;
        distance += ds;
        if (distance > maxDistance)
        {
            countOutv[0] = count;
            return;
        }

        rx += kx * (ds + eps);
        ry += ky * (ds + eps);
        rz += kz * (ds + eps);
        if (nextCell < 0) break;
        cell = nextCell;
    }

    countOutv[0] = count;
}

extern "C" __global__ void voronoi_mesh_grid_paths(const double* sitev, const int* neighborBeginv,
                                                   const int* neighborCountv, const int* neighborIndexv,
                                                   int numCells, double eps,
                                                   const double* positionv, const double* directionv, int numPaths,
                                                   double xmin, double ymin, double zmin,
                                                   double xmax, double ymax, double zmax,
                                                   double maxDistance, int maxSegments,
                                                   int* cellOutv, double* dsOutv,
                                                   int* countOutv, int* statusOutv)
{
    int pathIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (pathIndex >= numPaths) return;

    int* pathCellOutv = cellOutv + pathIndex * maxSegments;
    double* pathDsOutv = dsOutv + pathIndex * maxSegments;
    int* pathCountOutv = countOutv + pathIndex;
    int* pathStatusOutv = statusOutv + pathIndex;

    pathCountOutv[0] = 0;
    pathStatusOutv[0] = 1;
    if (numCells <= 0 || maxSegments <= 0) return;

    int pathBase = 3 * pathIndex;
    double rx = positionv[pathBase];
    double ry = positionv[pathBase + 1];
    double rz = positionv[pathBase + 2];
    double kx = directionv[pathBase];
    double ky = directionv[pathBase + 1];
    double kz = directionv[pathBase + 2];

    double cumds = 0.0;
    int count = 0;
    double distance = 0.0;

    if (rx <= xmin)
    {
        if (kx <= 0.0) return;
        double ds = (xmin - rx) / kx;
        rx = xmin + eps;
        ry += ky * ds;
        rz += kz * ds;
        cumds += ds;
    }
    else if (rx >= xmax)
    {
        if (kx >= 0.0) return;
        double ds = (xmax - rx) / kx;
        rx = xmax - eps;
        ry += ky * ds;
        rz += kz * ds;
        cumds += ds;
    }

    if (ry <= ymin)
    {
        if (ky <= 0.0) return;
        double ds = (ymin - ry) / ky;
        rx += kx * ds;
        ry = ymin + eps;
        rz += kz * ds;
        cumds += ds;
    }
    else if (ry >= ymax)
    {
        if (ky >= 0.0) return;
        double ds = (ymax - ry) / ky;
        rx += kx * ds;
        ry = ymax - eps;
        rz += kz * ds;
        cumds += ds;
    }

    if (rz <= zmin)
    {
        if (kz <= 0.0) return;
        double ds = (zmin - rz) / kz;
        rx += kx * ds;
        ry += ky * ds;
        rz = zmin + eps;
        cumds += ds;
    }
    else if (rz >= zmax)
    {
        if (kz >= 0.0) return;
        double ds = (zmax - rz) / kz;
        rx += kx * ds;
        ry += ky * ds;
        rz = zmax - eps;
        cumds += ds;
    }

    int cell = skirt_voronoi_find_cell(sitev, numCells, rx, ry, rz, xmin, ymin, zmin, xmax, ymax, zmax);
    if (cell < 0) return;

    if (cumds > 0.0)
    {
        if (count >= maxSegments)
        {
            pathStatusOutv[0] = -1;
            return;
        }
        pathCellOutv[count] = -1;
        pathDsOutv[count] = cumds;
        ++count;
        distance += cumds;
        if (distance > maxDistance)
        {
            pathCountOutv[0] = count;
            return;
        }
    }

    int guard = 0;
    int maxGuard = 8 * (numCells + 1);
    while (guard++ < maxGuard)
    {
        int begin = neighborBeginv[cell];
        int n = neighborCountv[cell];
        int nextCell = -99;
        double ds = 1.7976931348623157e+308;
        int siteBase = 3 * cell;
        double prx = sitev[siteBase];
        double pry = sitev[siteBase + 1];
        double prz = sitev[siteBase + 2];
        for (int i = 0; i != n; ++i)
        {
            int neighbor = neighborIndexv[begin + i];
            double si = 0.0;
            if (neighbor >= 0)
            {
                int neighborBase = 3 * neighbor;
                double pix = sitev[neighborBase];
                double piy = sitev[neighborBase + 1];
                double piz = sitev[neighborBase + 2];
                double nx = pix - prx;
                double ny = piy - pry;
                double nz = piz - prz;
                double ndotk = nx * kx + ny * ky + nz * kz;
                if (ndotk > 0.0)
                {
                    double px = 0.5 * (pix + prx);
                    double py = 0.5 * (piy + pry);
                    double pz = 0.5 * (piz + prz);
                    si = (nx * (px - rx) + ny * (py - ry) + nz * (pz - rz)) / ndotk;
                }
            }
            else
            {
                if (neighbor == -1)
                    si = (xmin - rx) / kx;
                else if (neighbor == -2)
                    si = (xmax - rx) / kx;
                else if (neighbor == -3)
                    si = (ymin - ry) / ky;
                else if (neighbor == -4)
                    si = (ymax - ry) / ky;
                else if (neighbor == -5)
                    si = (zmin - rz) / kz;
                else if (neighbor == -6)
                    si = (zmax - rz) / kz;
            }

            if (si > 0.0 && si < ds)
            {
                ds = si;
                nextCell = neighbor;
            }
        }

        if (nextCell == -99)
        {
            rx += kx * eps;
            ry += ky * eps;
            rz += kz * eps;
            cell = skirt_voronoi_find_cell(sitev, numCells, rx, ry, rz, xmin, ymin, zmin, xmax, ymax, zmax);
            if (cell < 0) break;
            continue;
        }

        if (count >= maxSegments)
        {
            pathStatusOutv[0] = -1;
            return;
        }
        pathCellOutv[count] = cell;
        pathDsOutv[count] = ds;
        ++count;
        distance += ds;
        if (distance > maxDistance)
        {
            pathCountOutv[0] = count;
            return;
        }

        rx += kx * (ds + eps);
        ry += ky * (ds + eps);
        rz += kz * (ds + eps);
        if (nextCell < 0) break;
        cell = nextCell;
    }

    pathCountOutv[0] = count;
}

static __device__ void skirt_voronoi_trace_compact(const double* sitev, const int* neighborBeginv,
                                                   const int* neighborCountv, const int* neighborIndexv,
                                                   const int* blockBeginv, const int* blockCountv,
                                                   const int* blockIndexv, int blockN,
                                                   int numCells, double eps,
                                                   double rx, double ry, double rz,
                                                   double kx, double ky, double kz,
                                                   double xmin, double ymin, double zmin,
                                                   double xmax, double ymax, double zmax,
                                                   double maxDistance, int segmentCapacity, int outputOffset,
                                                   int* cellOutv, double* dsOutv,
                                                   int* countOutv, int* statusOutv, bool writeSegments)
{
    countOutv[0] = 0;
    statusOutv[0] = 1;
    if (numCells <= 0 || segmentCapacity < 0) return;

    double cumds = 0.0;
    int count = 0;
    double distance = 0.0;

    if (rx <= xmin)
    {
        if (kx <= 0.0) return;
        double ds = (xmin - rx) / kx;
        rx = xmin + eps;
        ry += ky * ds;
        rz += kz * ds;
        cumds += ds;
    }
    else if (rx >= xmax)
    {
        if (kx >= 0.0) return;
        double ds = (xmax - rx) / kx;
        rx = xmax - eps;
        ry += ky * ds;
        rz += kz * ds;
        cumds += ds;
    }

    if (ry <= ymin)
    {
        if (ky <= 0.0) return;
        double ds = (ymin - ry) / ky;
        rx += kx * ds;
        ry = ymin + eps;
        rz += kz * ds;
        cumds += ds;
    }
    else if (ry >= ymax)
    {
        if (ky >= 0.0) return;
        double ds = (ymax - ry) / ky;
        rx += kx * ds;
        ry = ymax - eps;
        rz += kz * ds;
        cumds += ds;
    }

    if (rz <= zmin)
    {
        if (kz <= 0.0) return;
        double ds = (zmin - rz) / kz;
        rx += kx * ds;
        ry += ky * ds;
        rz = zmin + eps;
        cumds += ds;
    }
    else if (rz >= zmax)
    {
        if (kz >= 0.0) return;
        double ds = (zmax - rz) / kz;
        rx += kx * ds;
        ry += ky * ds;
        rz = zmax - eps;
        cumds += ds;
    }

    int cell = skirt_voronoi_find_cell_in_block(sitev, numCells, blockBeginv, blockCountv, blockIndexv, blockN,
                                                rx, ry, rz, xmin, ymin, zmin, xmax, ymax, zmax);
    if (cell < 0) return;

    if (cumds > 0.0)
    {
        if (count >= segmentCapacity)
        {
            statusOutv[0] = -1;
            return;
        }
        if (writeSegments)
        {
            cellOutv[outputOffset + count] = -1;
            dsOutv[outputOffset + count] = cumds;
        }
        ++count;
        distance += cumds;
        if (distance > maxDistance)
        {
            countOutv[0] = count;
            return;
        }
    }

    int guard = 0;
    int maxGuard = 8 * (numCells + 1);
    while (guard++ < maxGuard)
    {
        int begin = neighborBeginv[cell];
        int n = neighborCountv[cell];
        int nextCell = -99;
        double ds = 1.7976931348623157e+308;
        int siteBase = 3 * cell;
        double prx = sitev[siteBase];
        double pry = sitev[siteBase + 1];
        double prz = sitev[siteBase + 2];
        for (int i = 0; i != n; ++i)
        {
            int neighbor = neighborIndexv[begin + i];
            double si = 0.0;
            if (neighbor >= 0)
            {
                int neighborBase = 3 * neighbor;
                double pix = sitev[neighborBase];
                double piy = sitev[neighborBase + 1];
                double piz = sitev[neighborBase + 2];
                double nx = pix - prx;
                double ny = piy - pry;
                double nz = piz - prz;
                double ndotk = nx * kx + ny * ky + nz * kz;
                if (ndotk > 0.0)
                {
                    double px = 0.5 * (pix + prx);
                    double py = 0.5 * (piy + pry);
                    double pz = 0.5 * (piz + prz);
                    si = (nx * (px - rx) + ny * (py - ry) + nz * (pz - rz)) / ndotk;
                }
            }
            else
            {
                if (neighbor == -1)
                    si = (xmin - rx) / kx;
                else if (neighbor == -2)
                    si = (xmax - rx) / kx;
                else if (neighbor == -3)
                    si = (ymin - ry) / ky;
                else if (neighbor == -4)
                    si = (ymax - ry) / ky;
                else if (neighbor == -5)
                    si = (zmin - rz) / kz;
                else if (neighbor == -6)
                    si = (zmax - rz) / kz;
            }

            if (si > 0.0 && si < ds)
            {
                ds = si;
                nextCell = neighbor;
            }
        }

        if (nextCell == -99)
        {
            rx += kx * eps;
            ry += ky * eps;
            rz += kz * eps;
            cell = skirt_voronoi_find_cell_in_block(sitev, numCells, blockBeginv, blockCountv, blockIndexv, blockN,
                                                    rx, ry, rz, xmin, ymin, zmin, xmax, ymax, zmax);
            if (cell < 0) break;
            continue;
        }

        if (count >= segmentCapacity)
        {
            statusOutv[0] = -1;
            return;
        }
        if (writeSegments)
        {
            cellOutv[outputOffset + count] = cell;
            dsOutv[outputOffset + count] = ds;
        }
        ++count;
        distance += ds;
        if (distance > maxDistance)
        {
            countOutv[0] = count;
            return;
        }

        rx += kx * (ds + eps);
        ry += ky * (ds + eps);
        rz += kz * (ds + eps);
        if (nextCell < 0) break;
        cell = nextCell;
    }

    countOutv[0] = count;
}

extern "C" __global__ void voronoi_mesh_grid_paths_count(const double* sitev, const int* neighborBeginv,
                                                         const int* neighborCountv, const int* neighborIndexv,
                                                         const int* blockBeginv, const int* blockCountv,
                                                         const int* blockIndexv, int blockN,
                                                         int numCells, double eps,
                                                         const double* positionv, const double* directionv,
                                                         int numPaths, double xmin, double ymin, double zmin,
                                                         double xmax, double ymax, double zmax,
                                                         double maxDistance, int maxSegments,
                                                         int* countOutv, int* statusOutv)
{
    int pathIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (pathIndex >= numPaths) return;

    int pathBase = 3 * pathIndex;
    skirt_voronoi_trace_compact(sitev, neighborBeginv, neighborCountv, neighborIndexv,
                                blockBeginv, blockCountv, blockIndexv, blockN, numCells, eps,
                                positionv[pathBase], positionv[pathBase + 1], positionv[pathBase + 2],
                                directionv[pathBase], directionv[pathBase + 1], directionv[pathBase + 2],
                                xmin, ymin, zmin, xmax, ymax, zmax, maxDistance, maxSegments, 0,
                                0, 0, countOutv + pathIndex, statusOutv + pathIndex, false);
}

extern "C" __global__ void path_count_offsets(const int* countv, const int* statusv, int numPaths,
                                              int maxSegments, int* pathOffsetv, int* summaryv)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    long long total = 0;
    int ok = 1;
    const long long maxResidentSegments = 1073741823LL;
    for (int pathIndex = 0; pathIndex != numPaths; ++pathIndex)
    {
        int count = countv[pathIndex];
        if (statusv[pathIndex] != 1 || count < 0 || count > maxSegments || total > maxResidentSegments) ok = 0;
        pathOffsetv[pathIndex] = total > 2147483647LL ? 2147483647 : static_cast<int>(total);
        if (count > 0) total += count;
        if (total > maxResidentSegments) ok = 0;
    }
    pathOffsetv[numPaths] = total > 2147483647LL ? 2147483647 : static_cast<int>(total);
    summaryv[0] = total > 2147483647LL ? 2147483647 : static_cast<int>(total);
    summaryv[1] = ok;
}

extern "C" __global__ void validate_path_offsets(const int* countv, const int* statusv, const int* pathOffsetv,
                                                 int numPaths, int* okv)
{
    int pathIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (pathIndex >= numPaths) return;

    int expectedCount = pathOffsetv[pathIndex + 1] - pathOffsetv[pathIndex];
    int count = countv[pathIndex];
    if (statusv[pathIndex] != 1 || count != expectedCount || count < 0) atomicExch(okv, 0);
}

extern "C" __global__ void voronoi_mesh_grid_paths_compact(const double* sitev, const int* neighborBeginv,
                                                           const int* neighborCountv, const int* neighborIndexv,
                                                           const int* blockBeginv, const int* blockCountv,
                                                           const int* blockIndexv, int blockN,
                                                           int numCells, double eps,
                                                           const double* positionv, const double* directionv,
                                                           int numPaths, const int* pathOffsetv,
                                                           double xmin, double ymin, double zmin,
                                                           double xmax, double ymax, double zmax,
                                                           double maxDistance, int* cellOutv, double* dsOutv,
                                                           int* countOutv, int* statusOutv)
{
    int pathIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (pathIndex >= numPaths) return;

    int pathBase = 3 * pathIndex;
    int outputOffset = pathOffsetv[pathIndex];
    int segmentCapacity = pathOffsetv[pathIndex + 1] - outputOffset;
    skirt_voronoi_trace_compact(sitev, neighborBeginv, neighborCountv, neighborIndexv,
                                blockBeginv, blockCountv, blockIndexv, blockN, numCells, eps,
                                positionv[pathBase], positionv[pathBase + 1], positionv[pathBase + 2],
                                directionv[pathBase], directionv[pathBase + 1], directionv[pathBase + 2],
                                xmin, ymin, zmin, xmax, ymax, zmax, maxDistance, segmentCapacity, outputOffset,
                                cellOutv, dsOutv, countOutv + pathIndex, statusOutv + pathIndex, true);
}

extern "C" __global__ void constant_section_contribution(const int* cellv, const double* dsv,
                                                         const double* statev, int numSegments,
                                                         int numVars, int numMedia,
                                                         const int* densityOffsetv,
                                                         const double* section1v,
                                                         const double* section2v,
                                                         int hasSecond,
                                                         double* out1v, double* out2v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numSegments) return;

    int m = cellv[i];
    double out1 = 0.0;
    double out2 = 0.0;
    if (m >= 0)
    {
        double ds = dsv[i];
        int base = m * numVars;
        for (int h = 0; h != numMedia; ++h)
        {
            double ns = statev[base + densityOffsetv[h]] * ds;
            out1 += section1v[h] * ns;
            if (hasSecond) out2 += section2v[h] * ns;
        }
    }
    out1v[i] = out1;
    if (hasSecond) out2v[i] = out2;
}

static __device__ int skirt_section_lookup_index(const double* lookupWavelengthv, int begin, int count, double lambda)
{
    if (count <= 1) return begin;
    if (lambda < lookupWavelengthv[begin]) return begin;

    int jl = -1;
    int ju = count - 1;
    while (ju - jl > 1)
    {
        int jm = (ju + jl) >> 1;
        if (lambda < lookupWavelengthv[begin + jm])
            ju = jm;
        else
            jl = jm;
    }
    return begin + (jl < 0 ? 0 : jl);
}

extern "C" __global__ void table_section_contribution(const int* cellv, const double* dsv,
                                                      const double* statev, int numSegments,
                                                      int numVars, int numTables,
                                                      const int* densityOffsetv,
                                                      const int* lookupBeginv,
                                                      const int* lookupCountv,
                                                      const double* lookupWavelengthv,
                                                      const double* section1Tablev,
                                                      const double* section2Tablev,
                                                      int hasSecond, double lambda,
                                                      double* out1v, double* out2v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numSegments) return;

    int m = cellv[i];
    double out1 = 0.0;
    double out2 = 0.0;
    if (m >= 0)
    {
        double ds = dsv[i];
        int base = m * numVars;
        for (int h = 0; h != numTables; ++h)
        {
            int tableIndex = skirt_section_lookup_index(lookupWavelengthv, lookupBeginv[h], lookupCountv[h], lambda);
            double ns = statev[base + densityOffsetv[h]] * ds;
            out1 += section1Tablev[tableIndex] * ns;
            if (hasSecond) out2 += section2Tablev[tableIndex] * ns;
        }
    }
    out1v[i] = out1;
    if (hasSecond) out2v[i] = out2;
}

extern "C" __global__ void radiation_field_contribution(const int* cellv, const double* dsv,
                                                        const double* tauExtv, int numSegments,
                                                        double luminosity, double* outv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numSegments) return;

    if (cellv[i] < 0)
    {
        outv[i] = 0.0;
        return;
    }

    double lnExtBeg = i == 0 ? 0.0 : -tauExtv[i - 1];
    double lnExtEnd = -tauExtv[i];
    double extBeg = exp(lnExtBeg);
    double extEnd = exp(lnExtEnd);
    double extMean = skirt_lnmean(extEnd, extBeg, lnExtEnd, lnExtBeg);
    outv[i] = luminosity * extMean * dsv[i];
}

extern "C" __global__ void radiation_field_contributions_batch(const int* pathOffsetv, const int* pathIndexv,
                                                               const int* cellv, const double* dsv,
                                                               const double* tauExtv,
                                                               const double* luminosityv,
                                                               int numSegments, double* outv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numSegments) return;

    if (cellv[i] < 0)
    {
        outv[i] = 0.0;
        return;
    }

    int path = pathIndexv[i];
    int begin = pathOffsetv[path];
    double lnExtBeg = i == begin ? 0.0 : -tauExtv[i - 1];
    double lnExtEnd = -tauExtv[i];
    double extBeg = exp(lnExtBeg);
    double extEnd = exp(lnExtEnd);
    double extMean = skirt_lnmean(extEnd, extBeg, lnExtEnd, lnExtBeg);
    outv[i] = luminosityv[path] * extMean * dsv[i];
}

extern "C" __global__ void radiation_field_contribution_sums_batch(
    const int* pathOffsetv, const int* pathIndexv, const int* cellv, const double* dsv,
    const double* tauExtv, const double* luminosityv, const int* wavelengthBinv,
    int numWavelengths, int numSegments, int hashCapacity, int* keyOutv, double* sumOutv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numSegments) return;

    int cell = cellv[i];
    if (cell < 0) return;

    int path = pathIndexv[i];
    int ell = wavelengthBinv[path];
    if (ell < 0 || ell >= numWavelengths) return;

    int begin = pathOffsetv[path];
    double lnExtBeg = i == begin ? 0.0 : -tauExtv[i - 1];
    double lnExtEnd = -tauExtv[i];
    double extBeg = exp(lnExtBeg);
    double extEnd = exp(lnExtEnd);
    double extMean = skirt_lnmean(extEnd, extBeg, lnExtEnd, lnExtBeg);
    double contribution = luminosityv[path] * extMean * dsv[i];
    if (contribution == 0.0) return;

    int key = cell * numWavelengths + ell;
    unsigned int mask = static_cast<unsigned int>(hashCapacity - 1);
    unsigned int slot = skirt_hash_uint(static_cast<unsigned int>(key)) & mask;
    for (int probe = 0; probe != hashCapacity; ++probe)
    {
        int old = atomicCAS(&keyOutv[slot], -1, key);
        if (old == -1 || old == key)
        {
            skirt_atomic_add_double(&sumOutv[slot], contribution);
            return;
        }
        slot = (slot + 1U) & mask;
    }
}

extern "C" __global__ void radiation_field_compact_sums(const int* keyv, const double* sumv, int hashCapacity,
                                                        int* countOutv, int* compactKeyv, double* compactSumv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hashCapacity) return;

    int key = keyv[i];
    double sum = sumv[i];
    if (key >= 0 && sum != 0.0)
    {
        int out = atomicAdd(countOutv, 1);
        compactKeyv[out] = key;
        compactSumv[out] = sum;
    }
}

extern "C" __global__ void path_segment_metadata(const int* pathOffsetv, const double* dsv, int numPaths,
                                                 int* pathIndexOutv, double* svOutv)
{
    int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= numPaths) return;

    double s = 0.0;
    int begin = pathOffsetv[path];
    int end = pathOffsetv[path + 1];
    for (int i = begin; i != end; ++i)
    {
        pathIndexOutv[i] = path;
        s += dsv[i];
        svOutv[i] = s;
    }
}

extern "C" __global__ void sum_key_values(const int* inputKeyv, const double* inputValuev, int numValues,
                                          int hashCapacity, int* keyOutv, double* sumOutv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numValues) return;

    int key = inputKeyv[i];
    double value = inputValuev[i];
    if (key < 0 || value == 0.0) return;

    unsigned int mask = static_cast<unsigned int>(hashCapacity - 1);
    unsigned int slot = skirt_hash_uint(static_cast<unsigned int>(key)) & mask;
    for (int probe = 0; probe != hashCapacity; ++probe)
    {
        int old = atomicCAS(&keyOutv[slot], -1, key);
        if (old == -1 || old == key)
        {
            skirt_atomic_add_double(&sumOutv[slot], value);
            return;
        }
        slot = (slot + 1U) & mask;
    }
}

extern "C" __global__ void accumulate_values_by_key(const int* inputKeyv, const double* inputValuev, int numValues,
                                                    double* accumulatorv, int numAccumulatorValues)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numValues) return;

    int key = inputKeyv[i];
    double value = inputValuev[i];
    if (key < 0 || key >= numAccumulatorValues || value == 0.0) return;
    skirt_atomic_add_double(&accumulatorv[key], value);
}

static __device__ double skirt_band_relative_transmission(const double* wavelengthv, const double* transv,
                                                          int begin, int end, double width, double lambda)
{
    if (end - begin < 2) return 0.0;
    if (!(lambda > wavelengthv[begin] && lambda < wavelengthv[end - 1])) return 0.0;

    int lo = begin + 1;
    int hi = end;
    while (lo < hi)
    {
        int mid = lo + (hi - lo) / 2;
        if (lambda < wavelengthv[mid])
            hi = mid;
        else
            lo = mid + 1;
    }

    int i = lo;
    if (i <= begin || i >= end) return 0.0;
    double x0 = wavelengthv[i - 1];
    double x1 = wavelengthv[i];
    double y0 = transv[i - 1];
    double y1 = transv[i];
    double t = x1 != x0 ? (lambda - x0) / (x1 - x0) : 0.0;
    return (y0 + t * (y1 - y0)) * width;
}

extern "C" __global__ void frame_band_total_flux_values(
    const double* positionv, const double* wavelengthv, const double* luminosityv, const double* tauv,
    int hasMedium, int numPackets, int numBands, double costheta, double sintheta, double cosphi,
    double sinphi, double cosomega, double sinomega, int numPixelsX, int numPixelsY, double xpmin,
    double xpsiz, double ypmin, double ypsiz, double redshift, int numPixelsInFrame,
    const int* bandOffsetv, const double* bandWavelengthv, const double* bandTransmissionv,
    const double* bandWidthv, int* keyOutv, double* valueOutv)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int numValues = numPackets * numBands;
    if (index >= numValues) return;

    int packet = index / numBands;
    int ell = index - packet * numBands;
    keyOutv[index] = -1;
    valueOutv[index] = 0.0;

    double luminosity = luminosityv[packet];
    if (luminosity <= 0.0) return;

    int base = 3 * packet;
    double x = positionv[base];
    double y = positionv[base + 1];
    double z = positionv[base + 2];
    double xpp = -sinphi * x + cosphi * y;
    double ypp = -cosphi * costheta * x - sinphi * costheta * y + sintheta * z;
    double xp = cosomega * xpp - sinomega * ypp;
    double yp = sinomega * xpp + cosomega * ypp;
    int i = static_cast<int>(floor((xp - xpmin) / xpsiz));
    int j = static_cast<int>(floor((yp - ypmin) / ypsiz));
    if (i < 0 || i >= numPixelsX || j < 0 || j >= numPixelsY) return;

    double lambda = wavelengthv[packet] * (1.0 + redshift);
    int begin = bandOffsetv[ell];
    int end = bandOffsetv[ell + 1];
    double transmission = skirt_band_relative_transmission(bandWavelengthv, bandTransmissionv, begin, end,
                                                           bandWidthv[ell], lambda);
    if (transmission == 0.0) return;

    double value = luminosity * transmission;
    if (hasMedium) value *= exp(-tauv[packet]);
    if (value == 0.0) return;

    int pixel = i + numPixelsX * j;
    keyOutv[index] = pixel + ell * numPixelsInFrame;
    valueOutv[index] = value;
}

static __device__ double skirt_hg_value(double g, double costheta)
{
    double t = 1.0 + g * g - 2.0 * g * costheta;
    return (1.0 - g) * (1.0 + g) / sqrt(t * t * t);
}

static __device__ double skirt_hg_integral(double g, double cosalpha, double cosbeta)
{
    double ta = sqrt(1.0 + g * g - 2.0 * g * cosalpha);
    double tb = sqrt(1.0 + g * g - 2.0 * g * cosbeta);
    double f1 = (1.0 - g) * (1.0 + g) / g;
    double f2 = (tb - ta) / (tb * ta);
    return f1 * f2;
}

static __device__ double skirt_hg_mean(double g, double costheta)
{
    constexpr double pi = 3.14159265358979323846264338327950288;
    constexpr double delta = 4.0 * pi / 180.0;
    double theta = acos(costheta);
    double cosalpha = cos(theta - delta);
    double cosbeta = cos(theta + delta);
    if (theta < delta)
        return (skirt_hg_integral(g, 1.0, cosalpha) + skirt_hg_integral(g, 1.0, cosbeta))
               / (2.0 - cosalpha - cosbeta);
    if (theta > pi - delta)
        return (skirt_hg_integral(g, cosalpha, -1.0) + skirt_hg_integral(g, cosbeta, -1.0))
               / (2.0 + cosalpha + cosbeta);
    return skirt_hg_integral(g, cosalpha, cosbeta) / (cosalpha - cosbeta);
}

extern "C" __global__ void henyey_greenstein_scattering_luminosities(
    const double* inputDirectionv, const double* packetLuminosityv, const double* lambdav, int numPackets,
    double obsx, double obsy, double obsz, int lookupBegin, int lookupCount, const double* lookupWavelengthv,
    const double* asymmparv, double* luminosityv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPackets) return;

    double packetLuminosity = packetLuminosityv[i];
    if (packetLuminosity <= 0.0)
    {
        luminosityv[i] = 0.0;
        return;
    }

    int tableIndex = skirt_section_lookup_index(lookupWavelengthv, lookupBegin, lookupCount, lambdav[i]);
    double g = asymmparv[tableIndex];
    double costheta = inputDirectionv[3 * i] * obsx + inputDirectionv[3 * i + 1] * obsy
                      + inputDirectionv[3 * i + 2] * obsz;
    double phase = fabs(g) > 0.95 ? skirt_hg_mean(g, costheta) : skirt_hg_value(g, costheta);
    luminosityv[i] = packetLuminosity * phase;
}

extern "C" __global__ void henyey_greenstein_scattering_directions(
    const double* inputDirectionv, const double* lambdav, const double* randomCosthetav, const double* randomPhiv,
    int numPackets, int lookupBegin, int lookupCount, const double* lookupWavelengthv, const double* asymmparv,
    double* outputDirectionv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPackets) return;

    constexpr double pi = 3.14159265358979323846264338327950288;
    int tableIndex = skirt_section_lookup_index(lookupWavelengthv, lookupBegin, lookupCount, lambdav[i]);
    double g = asymmparv[tableIndex];
    double u = randomCosthetav[i];
    double phi = 2.0 * pi * randomPhiv[i];
    double cosphi = cos(phi);
    double sinphi = sin(phi);

    double costheta = 0.0;
    if (fabs(g) < 1e-6)
    {
        costheta = 2.0 * u - 1.0;
        double sintheta = sqrt(fabs((1.0 - costheta) * (1.0 + costheta)));
        outputDirectionv[3 * i] = cosphi * sintheta;
        outputDirectionv[3 * i + 1] = sinphi * sintheta;
        outputDirectionv[3 * i + 2] = costheta;
        return;
    }

    double f = ((1.0 - g) * (1.0 + g)) / (1.0 - g + 2.0 * g * u);
    costheta = (1.0 + g * g - f * f) / (2.0 * g);
    double sintheta = sqrt(fabs((1.0 - costheta) * (1.0 + costheta)));

    double kx = inputDirectionv[3 * i];
    double ky = inputDirectionv[3 * i + 1];
    double kz = inputDirectionv[3 * i + 2];
    double kxnew = 0.0;
    double kynew = 0.0;
    double kznew = 0.0;
    if (kz > 0.99999)
    {
        kxnew = cosphi * sintheta;
        kynew = sinphi * sintheta;
        kznew = costheta;
    }
    else if (kz < -0.99999)
    {
        kxnew = cosphi * sintheta;
        kynew = sinphi * sintheta;
        kznew = -costheta;
    }
    else
    {
        double root = sqrt((1.0 - kz) * (1.0 + kz));
        kxnew = sintheta / root * (-kx * kz * cosphi + ky * sinphi) + kx * costheta;
        kynew = -sintheta / root * (ky * kz * cosphi + kx * sinphi) + ky * costheta;
        kznew = root * sintheta * cosphi + kz * costheta;
    }
    outputDirectionv[3 * i] = kxnew;
    outputDirectionv[3 * i + 1] = kynew;
    outputDirectionv[3 * i + 2] = kznew;
}

extern "C" __global__ void isotropic_directions(const double* randomCosthetav, const double* randomPhiv,
                                                int numPackets, double* outputDirectionv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPackets) return;

    constexpr double pi = 3.14159265358979323846264338327950288;
    double costheta = 2.0 * randomCosthetav[i] - 1.0;
    double sintheta = sqrt(fabs((1.0 - costheta) * (1.0 + costheta)));
    double phi = 2.0 * pi * randomPhiv[i];
    outputDirectionv[3 * i] = sintheta * cos(phi);
    outputDirectionv[3 * i + 1] = sintheta * sin(phi);
    outputDirectionv[3 * i + 2] = costheta;
}

extern "C" __global__ void optical_depth_sum(const double* contributionv, int numSegments, double taumax, double* outv)
{
    double tau = 0.0;
    for (int i = 0; i != numSegments; ++i)
    {
        tau += contributionv[i];
        if (tau >= taumax)
        {
            outv[0] = 1.0 / 0.0;
            return;
        }
    }
    outv[0] = tau;
}

extern "C" __global__ void cumulative_optical_depths(const double* contribution1v, const double* contribution2v,
                                                     int hasSecond, int numSegments, double* out1v, double* out2v)
{
    double tau1 = 0.0;
    double tau2 = 0.0;
    for (int i = 0; i != numSegments; ++i)
    {
        tau1 += contribution1v[i];
        out1v[i] = tau1;
        if (hasSecond)
        {
            tau2 += contribution2v[i];
            out2v[i] = tau2;
        }
    }
}

extern "C" __global__ void cumulative_constant_section_optical_depths_batch(
    const int* pathOffsetv, const int* cellv, const double* dsv, const double* statev, int numVars, int numMedia,
    const int* densityOffsetv, const double* section1v, const double* section2v, int hasSecond, int numPaths,
    double* out1v, double* out2v)
{
    int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= numPaths) return;

    int begin = pathOffsetv[path];
    int end = pathOffsetv[path + 1];
    double tau1 = 0.0;
    double tau2 = 0.0;
    for (int i = begin; i != end; ++i)
    {
        int m = cellv[i];
        if (m >= 0)
        {
            double ds = dsv[i];
            int base = m * numVars;
            for (int h = 0; h != numMedia; ++h)
            {
                double ns = statev[base + densityOffsetv[h]] * ds;
                tau1 += section1v[h] * ns;
                if (hasSecond) tau2 += section2v[h] * ns;
            }
        }
        out1v[i] = tau1;
        if (hasSecond) out2v[i] = tau2;
    }
}

extern "C" __global__ void cumulative_table_section_optical_depths_batch(
    const int* pathOffsetv, const int* cellv, const double* dsv, const double* statev, int numVars, int numTables,
    const int* densityOffsetv, const int* lookupBeginv, const int* lookupCountv, const double* lookupWavelengthv,
    const double* section1Tablev, const double* section2Tablev, int hasSecond, const double* lambdav, int numPaths,
    double* out1v, double* out2v)
{
    int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= numPaths) return;

    int begin = pathOffsetv[path];
    int end = pathOffsetv[path + 1];
    double lambda = lambdav[path];
    double tau1 = 0.0;
    double tau2 = 0.0;
    for (int i = begin; i != end; ++i)
    {
        int m = cellv[i];
        if (m >= 0)
        {
            double ds = dsv[i];
            int base = m * numVars;
            for (int h = 0; h != numTables; ++h)
            {
                int tableIndex = skirt_section_lookup_index(lookupWavelengthv, lookupBeginv[h], lookupCountv[h],
                                                            lambda);
                double ns = statev[base + densityOffsetv[h]] * ds;
                tau1 += section1Tablev[tableIndex] * ns;
                if (hasSecond) tau2 += section2Tablev[tableIndex] * ns;
            }
        }
        out1v[i] = tau1;
        if (hasSecond) out2v[i] = tau2;
    }
}

static __device__ void skirt_voronoi_trace_table_cumulative(
    const double* sitev, const int* neighborBeginv, const int* neighborCountv, const int* neighborIndexv,
    const int* blockBeginv, const int* blockCountv, const int* blockIndexv, int blockN, int numCells, double eps,
    double rx, double ry, double rz, double kx, double ky, double kz, double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax, double maxDistance, int segmentCapacity, int outputOffset,
    const double* statev, int numVars, int numTables, const int* densityOffsetv, const int* lookupBeginv,
    const int* lookupCountv, const double* lookupWavelengthv, const double* section1Tablev,
    const double* section2Tablev, int hasSecond, double lambda, int* cellOutv, double* dsOutv, double* tau1Outv,
    double* tau2Outv, int* countOutv, int* statusOutv)
{
    countOutv[0] = 0;
    statusOutv[0] = 1;
    if (numCells <= 0 || segmentCapacity < 0) return;

    double cumds = 0.0;
    int count = 0;
    double distance = 0.0;
    double tau1 = 0.0;
    double tau2 = 0.0;

    if (rx <= xmin)
    {
        if (kx <= 0.0) return;
        double ds = (xmin - rx) / kx;
        rx = xmin + eps;
        ry += ky * ds;
        rz += kz * ds;
        cumds += ds;
    }
    else if (rx >= xmax)
    {
        if (kx >= 0.0) return;
        double ds = (xmax - rx) / kx;
        rx = xmax - eps;
        ry += ky * ds;
        rz += kz * ds;
        cumds += ds;
    }

    if (ry <= ymin)
    {
        if (ky <= 0.0) return;
        double ds = (ymin - ry) / ky;
        rx += kx * ds;
        ry = ymin + eps;
        rz += kz * ds;
        cumds += ds;
    }
    else if (ry >= ymax)
    {
        if (ky >= 0.0) return;
        double ds = (ymax - ry) / ky;
        rx += kx * ds;
        ry = ymax - eps;
        rz += kz * ds;
        cumds += ds;
    }

    if (rz <= zmin)
    {
        if (kz <= 0.0) return;
        double ds = (zmin - rz) / kz;
        rx += kx * ds;
        ry += ky * ds;
        rz = zmin + eps;
        cumds += ds;
    }
    else if (rz >= zmax)
    {
        if (kz >= 0.0) return;
        double ds = (zmax - rz) / kz;
        rx += kx * ds;
        ry += ky * ds;
        rz = zmax - eps;
        cumds += ds;
    }

    int cell = skirt_voronoi_find_cell_in_block(sitev, numCells, blockBeginv, blockCountv, blockIndexv, blockN,
                                                rx, ry, rz, xmin, ymin, zmin, xmax, ymax, zmax);
    if (cell < 0) return;

    if (cumds > 0.0)
    {
        if (count >= segmentCapacity)
        {
            statusOutv[0] = -1;
            return;
        }
        int out = outputOffset + count;
        cellOutv[out] = -1;
        dsOutv[out] = cumds;
        tau1Outv[out] = tau1;
        if (hasSecond) tau2Outv[out] = tau2;
        ++count;
        distance += cumds;
        if (distance > maxDistance)
        {
            countOutv[0] = count;
            return;
        }
    }

    int guard = 0;
    int maxGuard = 8 * (numCells + 1);
    while (guard++ < maxGuard)
    {
        int begin = neighborBeginv[cell];
        int n = neighborCountv[cell];
        int nextCell = -99;
        double ds = 1.7976931348623157e+308;
        int siteBase = 3 * cell;
        double prx = sitev[siteBase];
        double pry = sitev[siteBase + 1];
        double prz = sitev[siteBase + 2];
        for (int i = 0; i != n; ++i)
        {
            int neighbor = neighborIndexv[begin + i];
            double si = 0.0;
            if (neighbor >= 0)
            {
                int neighborBase = 3 * neighbor;
                double pix = sitev[neighborBase];
                double piy = sitev[neighborBase + 1];
                double piz = sitev[neighborBase + 2];
                double nx = pix - prx;
                double ny = piy - pry;
                double nz = piz - prz;
                double ndotk = nx * kx + ny * ky + nz * kz;
                if (ndotk > 0.0)
                {
                    double px = 0.5 * (pix + prx);
                    double py = 0.5 * (piy + pry);
                    double pz = 0.5 * (piz + prz);
                    si = (nx * (px - rx) + ny * (py - ry) + nz * (pz - rz)) / ndotk;
                }
            }
            else
            {
                if (neighbor == -1)
                    si = (xmin - rx) / kx;
                else if (neighbor == -2)
                    si = (xmax - rx) / kx;
                else if (neighbor == -3)
                    si = (ymin - ry) / ky;
                else if (neighbor == -4)
                    si = (ymax - ry) / ky;
                else if (neighbor == -5)
                    si = (zmin - rz) / kz;
                else if (neighbor == -6)
                    si = (zmax - rz) / kz;
            }

            if (si > 0.0 && si < ds)
            {
                ds = si;
                nextCell = neighbor;
            }
        }

        if (nextCell == -99)
        {
            rx += kx * eps;
            ry += ky * eps;
            rz += kz * eps;
            cell = skirt_voronoi_find_cell_in_block(sitev, numCells, blockBeginv, blockCountv, blockIndexv, blockN,
                                                    rx, ry, rz, xmin, ymin, zmin, xmax, ymax, zmax);
            if (cell < 0) break;
            continue;
        }

        if (count >= segmentCapacity)
        {
            statusOutv[0] = -1;
            return;
        }
        for (int h = 0; h != numTables; ++h)
        {
            int tableIndex = skirt_section_lookup_index(lookupWavelengthv, lookupBeginv[h], lookupCountv[h],
                                                        lambda);
            double ns = statev[cell * numVars + densityOffsetv[h]] * ds;
            tau1 += section1Tablev[tableIndex] * ns;
            if (hasSecond) tau2 += section2Tablev[tableIndex] * ns;
        }

        int out = outputOffset + count;
        cellOutv[out] = cell;
        dsOutv[out] = ds;
        tau1Outv[out] = tau1;
        if (hasSecond) tau2Outv[out] = tau2;
        ++count;
        distance += ds;
        if (distance > maxDistance)
        {
            countOutv[0] = count;
            return;
        }

        rx += kx * (ds + eps);
        ry += ky * (ds + eps);
        rz += kz * (ds + eps);
        if (nextCell < 0) break;
        cell = nextCell;
    }

    countOutv[0] = count;
}

extern "C" __global__ void voronoi_table_section_optical_depths_compact(
    const double* sitev, const int* neighborBeginv, const int* neighborCountv, const int* neighborIndexv,
    const int* blockBeginv, const int* blockCountv, const int* blockIndexv, int blockN, int numCells, double eps,
    const double* positionv, const double* directionv, int numPaths, const int* pathOffsetv, double xmin,
    double ymin, double zmin, double xmax, double ymax, double zmax, double maxDistance, const double* statev,
    int numVars, int numTables, const int* densityOffsetv, const int* lookupBeginv, const int* lookupCountv,
    const double* lookupWavelengthv, const double* section1Tablev, const double* section2Tablev, int hasSecond,
    const double* lambdav, int* cellOutv, double* dsOutv, double* tau1Outv, double* tau2Outv, int* countOutv,
    int* statusOutv)
{
    int pathIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (pathIndex >= numPaths) return;

    int pathBase = 3 * pathIndex;
    int outputOffset = pathOffsetv[pathIndex];
    int segmentCapacity = pathOffsetv[pathIndex + 1] - outputOffset;
    skirt_voronoi_trace_table_cumulative(
        sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv, blockCountv, blockIndexv, blockN,
        numCells, eps, positionv[pathBase], positionv[pathBase + 1], positionv[pathBase + 2],
        directionv[pathBase], directionv[pathBase + 1], directionv[pathBase + 2], xmin, ymin, zmin, xmax, ymax, zmax,
        maxDistance, segmentCapacity, outputOffset, statev, numVars, numTables, densityOffsetv, lookupBeginv,
        lookupCountv, lookupWavelengthv, section1Tablev, section2Tablev, hasSecond, lambdav[pathIndex], cellOutv,
        dsOutv, tau1Outv, tau2Outv, countOutv + pathIndex, statusOutv + pathIndex);
}

static __device__ double skirt_voronoi_trace_table_total(
    const double* sitev, const int* neighborBeginv, const int* neighborCountv, const int* neighborIndexv,
    const int* blockBeginv, const int* blockCountv, const int* blockIndexv, int blockN, int numCells, double eps,
    double rx, double ry, double rz, double kx, double ky, double kz, double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax, double maxDistance, const double* statev, int numVars, int numTables,
    const int* densityOffsetv, const int* lookupBeginv, const int* lookupCountv, const double* lookupWavelengthv,
    const double* sectionTablev, double lambda, int* statusOut)
{
    statusOut[0] = 1;
    if (numCells <= 0) return 0.0;

    double cumds = 0.0;
    double distance = 0.0;
    double tau = 0.0;

    if (rx <= xmin)
    {
        if (kx <= 0.0) return 0.0;
        double ds = (xmin - rx) / kx;
        rx = xmin + eps;
        ry += ky * ds;
        rz += kz * ds;
        cumds += ds;
    }
    else if (rx >= xmax)
    {
        if (kx >= 0.0) return 0.0;
        double ds = (xmax - rx) / kx;
        rx = xmax - eps;
        ry += ky * ds;
        rz += kz * ds;
        cumds += ds;
    }

    if (ry <= ymin)
    {
        if (ky <= 0.0) return 0.0;
        double ds = (ymin - ry) / ky;
        rx += kx * ds;
        ry = ymin + eps;
        rz += kz * ds;
        cumds += ds;
    }
    else if (ry >= ymax)
    {
        if (ky >= 0.0) return 0.0;
        double ds = (ymax - ry) / ky;
        rx += kx * ds;
        ry = ymax - eps;
        rz += kz * ds;
        cumds += ds;
    }

    if (rz <= zmin)
    {
        if (kz <= 0.0) return 0.0;
        double ds = (zmin - rz) / kz;
        rx += kx * ds;
        ry += ky * ds;
        rz = zmin + eps;
        cumds += ds;
    }
    else if (rz >= zmax)
    {
        if (kz >= 0.0) return 0.0;
        double ds = (zmax - rz) / kz;
        rx += kx * ds;
        ry += ky * ds;
        rz = zmax - eps;
        cumds += ds;
    }

    distance += cumds;
    if (distance > maxDistance) return 0.0;

    int cell = skirt_voronoi_find_cell_in_block(sitev, numCells, blockBeginv, blockCountv, blockIndexv, blockN,
                                                rx, ry, rz, xmin, ymin, zmin, xmax, ymax, zmax);
    if (cell < 0) return 0.0;

    int guard = 0;
    int maxGuard = 8 * (numCells + 1);
    while (guard++ < maxGuard)
    {
        int begin = neighborBeginv[cell];
        int n = neighborCountv[cell];
        int nextCell = -99;
        double ds = 1.7976931348623157e+308;
        int siteBase = 3 * cell;
        double prx = sitev[siteBase];
        double pry = sitev[siteBase + 1];
        double prz = sitev[siteBase + 2];
        for (int i = 0; i != n; ++i)
        {
            int neighbor = neighborIndexv[begin + i];
            double si = 0.0;
            if (neighbor >= 0)
            {
                int neighborBase = 3 * neighbor;
                double pix = sitev[neighborBase];
                double piy = sitev[neighborBase + 1];
                double piz = sitev[neighborBase + 2];
                double nx = pix - prx;
                double ny = piy - pry;
                double nz = piz - prz;
                double ndotk = nx * kx + ny * ky + nz * kz;
                if (ndotk > 0.0)
                {
                    double px = 0.5 * (pix + prx);
                    double py = 0.5 * (piy + pry);
                    double pz = 0.5 * (piz + prz);
                    si = (nx * (px - rx) + ny * (py - ry) + nz * (pz - rz)) / ndotk;
                }
            }
            else
            {
                if (neighbor == -1)
                    si = (xmin - rx) / kx;
                else if (neighbor == -2)
                    si = (xmax - rx) / kx;
                else if (neighbor == -3)
                    si = (ymin - ry) / ky;
                else if (neighbor == -4)
                    si = (ymax - ry) / ky;
                else if (neighbor == -5)
                    si = (zmin - rz) / kz;
                else if (neighbor == -6)
                    si = (zmax - rz) / kz;
            }

            if (si > 0.0 && si < ds)
            {
                ds = si;
                nextCell = neighbor;
            }
        }

        if (nextCell == -99)
        {
            rx += kx * eps;
            ry += ky * eps;
            rz += kz * eps;
            cell = skirt_voronoi_find_cell_in_block(sitev, numCells, blockBeginv, blockCountv, blockIndexv, blockN,
                                                    rx, ry, rz, xmin, ymin, zmin, xmax, ymax, zmax);
            if (cell < 0) return tau;
            continue;
        }

        double clippedDs = ds;
        if (distance + clippedDs > maxDistance) clippedDs = maxDistance - distance;
        if (clippedDs > 0.0)
        {
            for (int h = 0; h != numTables; ++h)
            {
                int tableIndex = skirt_section_lookup_index(lookupWavelengthv, lookupBeginv[h], lookupCountv[h],
                                                            lambda);
                double ns = statev[cell * numVars + densityOffsetv[h]] * clippedDs;
                tau += sectionTablev[tableIndex] * ns;
            }
        }
        distance += ds;
        if (distance > maxDistance) return tau;

        rx += kx * (ds + eps);
        ry += ky * (ds + eps);
        rz += kz * (ds + eps);
        if (nextCell < 0) return tau;
        cell = nextCell;
    }

    statusOut[0] = -1;
    return tau;
}

extern "C" __global__ void voronoi_table_extinction_optical_depth_totals(
    const double* sitev, const int* neighborBeginv, const int* neighborCountv, const int* neighborIndexv,
    const int* blockBeginv, const int* blockCountv, const int* blockIndexv, int blockN, int numCells, double eps,
    const double* positionv, const double* directionv, int numPaths, double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax, const double* maxDistancev, const double* statev, int numVars,
    int numTables, const int* densityOffsetv, const int* lookupBeginv, const int* lookupCountv,
    const double* lookupWavelengthv, const double* sectionTablev, const double* lambdav, double* tauv,
    int* statusv)
{
    int pathIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (pathIndex >= numPaths) return;

    int pathBase = 3 * pathIndex;
    tauv[pathIndex] = skirt_voronoi_trace_table_total(
        sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv, blockCountv, blockIndexv, blockN,
        numCells, eps, positionv[pathBase], positionv[pathBase + 1], positionv[pathBase + 2],
        directionv[pathBase], directionv[pathBase + 1], directionv[pathBase + 2], xmin, ymin, zmin, xmax, ymax, zmax,
        maxDistancev[pathIndex], statev, numVars, numTables, densityOffsetv, lookupBeginv, lookupCountv,
        lookupWavelengthv, sectionTablev, lambdav[pathIndex], statusv + pathIndex);
}

extern "C" __global__ void voronoi_table_hg_scattering_observed_luminosities(
    const double* sitev, const int* neighborBeginv, const int* neighborCountv, const int* neighborIndexv,
    const int* blockBeginv, const int* blockCountv, const int* blockIndexv, int blockN, int numCells, double eps,
    const double* positionv, const double* directionv, const double* inputDirectionv,
    const double* packetLuminosityv, int numPaths, double obsx, double obsy, double obsz, double xmin, double ymin,
    double zmin, double xmax, double ymax, double zmax, const double* maxDistancev, const double* statev,
    int numVars, int numTables, const int* densityOffsetv, const int* lookupBeginv, const int* lookupCountv,
    const double* lookupWavelengthv, const double* sectionTablev, const double* lambdav, int hgLookupBegin,
    int hgLookupCount, const double* asymmparv, double* luminosityv, int* statusv)
{
    int pathIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (pathIndex >= numPaths) return;

    statusv[pathIndex] = 1;
    double packetLuminosity = packetLuminosityv[pathIndex];
    if (packetLuminosity <= 0.0)
    {
        luminosityv[pathIndex] = 0.0;
        return;
    }

    int pathBase = 3 * pathIndex;
    double tau = skirt_voronoi_trace_table_total(
        sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv, blockCountv, blockIndexv, blockN,
        numCells, eps, positionv[pathBase], positionv[pathBase + 1], positionv[pathBase + 2],
        directionv[pathBase], directionv[pathBase + 1], directionv[pathBase + 2], xmin, ymin, zmin, xmax, ymax, zmax,
        maxDistancev[pathIndex], statev, numVars, numTables, densityOffsetv, lookupBeginv, lookupCountv,
        lookupWavelengthv, sectionTablev, lambdav[pathIndex], statusv + pathIndex);

    int tableIndex = skirt_section_lookup_index(lookupWavelengthv, hgLookupBegin, hgLookupCount, lambdav[pathIndex]);
    double g = asymmparv[tableIndex];
    double costheta = inputDirectionv[pathBase] * obsx + inputDirectionv[pathBase + 1] * obsy
                      + inputDirectionv[pathBase + 2] * obsz;
    double phase = fabs(g) > 0.95 ? skirt_hg_mean(g, costheta) : skirt_hg_value(g, costheta);
    luminosityv[pathIndex] = packetLuminosity * phase * exp(-tau);
}

extern "C" __global__ void voronoi_table_hg_scattering_frame_band_accumulate(
    const double* sitev, const int* neighborBeginv, const int* neighborCountv, const int* neighborIndexv,
    const int* blockBeginv, const int* blockCountv, const int* blockIndexv, int blockN, int numCells, double eps,
    const double* positionv, const double* inputDirectionv, const double* packetLuminosityv, int numPaths,
    double obsx, double obsy, double obsz, double xmin, double ymin, double zmin, double xmax, double ymax,
    double zmax, const double* maxDistancev, const double* statev, int numVars, int numTables,
    const int* densityOffsetv, const int* lookupBeginv, const int* lookupCountv, const double* lookupWavelengthv,
    const double* sectionTablev, const double* lambdav, int hgLookupBegin, int hgLookupCount,
    const double* asymmparv, double costhetaFrame, double sinthetaFrame, double cosphi, double sinphi,
    double cosomega, double sinomega, int numPixelsX, int numPixelsY, double xpmin, double xpsiz,
    double ypmin, double ypsiz, double redshift, int numPixelsInFrame, int numBands,
    const int* bandOffsetv, const double* bandWavelengthv, const double* bandTransmissionv,
    const double* bandWidthv, double* accumulator, int numAccumulatorValues, int* statusv)
{
    int pathIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (pathIndex >= numPaths) return;

    statusv[pathIndex] = 1;
    double packetLuminosity = packetLuminosityv[pathIndex];
    if (packetLuminosity <= 0.0) return;

    int pathBase = 3 * pathIndex;
    double x = positionv[pathBase];
    double y = positionv[pathBase + 1];
    double z = positionv[pathBase + 2];
    double xpp = -sinphi * x + cosphi * y;
    double ypp = -cosphi * costhetaFrame * x - sinphi * costhetaFrame * y + sinthetaFrame * z;
    double xp = cosomega * xpp - sinomega * ypp;
    double yp = sinomega * xpp + cosomega * ypp;
    int i = static_cast<int>(floor((xp - xpmin) / xpsiz));
    int j = static_cast<int>(floor((yp - ypmin) / ypsiz));
    if (i < 0 || i >= numPixelsX || j < 0 || j >= numPixelsY) return;

    double lambda = lambdav[pathIndex];
    double tau = skirt_voronoi_trace_table_total(
        sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv, blockCountv, blockIndexv, blockN,
        numCells, eps, x, y, z, obsx, obsy, obsz, xmin, ymin, zmin, xmax, ymax, zmax,
        maxDistancev[pathIndex], statev, numVars, numTables, densityOffsetv, lookupBeginv, lookupCountv,
        lookupWavelengthv, sectionTablev, lambda, statusv + pathIndex);
    if (statusv[pathIndex] != 1) return;

    int tableIndex = skirt_section_lookup_index(lookupWavelengthv, hgLookupBegin, hgLookupCount, lambda);
    double g = asymmparv[tableIndex];
    double scatteringCostheta = inputDirectionv[pathBase] * obsx + inputDirectionv[pathBase + 1] * obsy
                                + inputDirectionv[pathBase + 2] * obsz;
    double phase = fabs(g) > 0.95 ? skirt_hg_mean(g, scatteringCostheta) : skirt_hg_value(g, scatteringCostheta);
    double luminosity = packetLuminosity * phase * exp(-tau);
    if (luminosity == 0.0) return;

    int pixel = i + numPixelsX * j;
    double lambdaObserved = lambda * (1.0 + redshift);
    for (int ell = 0; ell != numBands; ++ell)
    {
        int begin = bandOffsetv[ell];
        int end = bandOffsetv[ell + 1];
        double transmission = skirt_band_relative_transmission(bandWavelengthv, bandTransmissionv, begin, end,
                                                               bandWidthv[ell], lambdaObserved);
        if (transmission == 0.0) continue;

        int key = pixel + ell * numPixelsInFrame;
        if (key >= 0 && key < numAccumulatorValues)
            skirt_atomic_add_double(accumulator + key, luminosity * transmission);
    }
}

extern "C" __global__ void interaction_point_extinction(const int* cellv, const double* dsv,
                                                        const double* contributionv, int numSegments,
                                                        double tauinteract, int* foundv, int* cellOutv,
                                                        double* distanceOutv)
{
    double tau = 0.0;
    double distance = 0.0;
    for (int i = 0; i != numSegments; ++i)
    {
        double tau0 = tau;
        double s0 = distance;
        tau += contributionv[i];
        distance += dsv[i];
        if (tauinteract < tau)
        {
            foundv[0] = 1;
            cellOutv[0] = cellv[i];
            distanceOutv[0] = tau != tau0 ? s0 + (tauinteract - tau0) * (distance - s0) / (tau - tau0) : s0;
            return;
        }
    }
    foundv[0] = 0;
    cellOutv[0] = -1;
    distanceOutv[0] = 0.0;
}

extern "C" __global__ void interaction_point_sca_abs(const int* cellv, const double* dsv,
                                                     const double* scaContributionv,
                                                     const double* absContributionv, int numSegments,
                                                     double tauinteract, int* foundv, int* cellOutv,
                                                     double* distanceOutv, double* tauAbsOutv)
{
    double tauSca = 0.0;
    double tauAbs = 0.0;
    double distance = 0.0;
    for (int i = 0; i != numSegments; ++i)
    {
        double tauSca0 = tauSca;
        double tauAbs0 = tauAbs;
        double s0 = distance;
        tauSca += scaContributionv[i];
        tauAbs += absContributionv[i];
        distance += dsv[i];
        if (tauinteract < tauSca)
        {
            double t = tauSca != tauSca0 ? (tauinteract - tauSca0) / (tauSca - tauSca0) : 0.0;
            foundv[0] = 1;
            cellOutv[0] = cellv[i];
            distanceOutv[0] = s0 + t * (distance - s0);
            tauAbsOutv[0] = tauAbs0 + t * (tauAbs - tauAbs0);
            return;
        }
    }
    foundv[0] = 0;
    cellOutv[0] = -1;
    distanceOutv[0] = 0.0;
    tauAbsOutv[0] = 0.0;
}

extern "C" __global__ void cumulative_path_interaction_point(const int* cellv, const double* sv,
                                                             const double* tauv, const double* tauAbsv,
                                                             int hasAbsorption, int numSegments,
                                                             double tauinteract, int* cellOutv,
                                                             double* distanceOutv, double* tauAbsOutv)
{
    if (numSegments <= 0)
    {
        cellOutv[0] = -1;
        distanceOutv[0] = 0.0;
        tauAbsOutv[0] = 0.0;
        return;
    }

    for (int i = 0; i != numSegments; ++i)
    {
        if (tauinteract < tauv[i])
        {
            double tau0 = i == 0 ? 0.0 : tauv[i - 1];
            double s0 = i == 0 ? 0.0 : sv[i - 1];
            double tauAbs0 = (!hasAbsorption || i == 0) ? 0.0 : tauAbsv[i - 1];
            double tauAbs1 = hasAbsorption ? tauAbsv[i] : 0.0;
            double t = tauv[i] != tau0 ? (tauinteract - tau0) / (tauv[i] - tau0) : 0.0;
            cellOutv[0] = cellv[i];
            distanceOutv[0] = s0 + t * (sv[i] - s0);
            tauAbsOutv[0] = tauAbs0 + t * (tauAbs1 - tauAbs0);
            return;
        }
    }

    cellOutv[0] = cellv[numSegments - 1];
    distanceOutv[0] = sv[numSegments - 1];
    tauAbsOutv[0] = hasAbsorption ? tauAbsv[numSegments - 1] : 0.0;
}

extern "C" __global__ void cumulative_path_interaction_points(const int* pathOffsetv, const int* cellv,
                                                              const double* sv, const double* tauv,
                                                              const double* tauAbsv, int hasAbsorption,
                                                              const double* tauinteractv, int numPaths,
                                                              int* cellOutv, double* distanceOutv,
                                                              double* tauAbsOutv)
{
    int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= numPaths) return;

    int begin = pathOffsetv[path];
    int end = pathOffsetv[path + 1];
    if (begin >= end)
    {
        cellOutv[path] = -1;
        distanceOutv[path] = 0.0;
        tauAbsOutv[path] = 0.0;
        return;
    }

    double tauinteract = tauinteractv[path];
    for (int i = begin; i != end; ++i)
    {
        if (tauinteract < tauv[i])
        {
            int previous = i - 1;
            double tau0 = i == begin ? 0.0 : tauv[previous];
            double s0 = i == begin ? 0.0 : sv[previous];
            double tauAbs0 = (!hasAbsorption || i == begin) ? 0.0 : tauAbsv[previous];
            double tauAbs1 = hasAbsorption ? tauAbsv[i] : 0.0;
            double t = tauv[i] != tau0 ? (tauinteract - tau0) / (tauv[i] - tau0) : 0.0;
            cellOutv[path] = cellv[i];
            distanceOutv[path] = s0 + t * (sv[i] - s0);
            tauAbsOutv[path] = tauAbs0 + t * (tauAbs1 - tauAbs0);
            return;
        }
    }

    int last = end - 1;
    cellOutv[path] = cellv[last];
    distanceOutv[path] = sv[last];
    tauAbsOutv[path] = hasAbsorption ? tauAbsv[last] : 0.0;
}

extern "C" __global__ void forced_propagation_results(const int* pathOffsetv, const int* cellv,
                                                       const double* sv, const double* tauv,
                                                       const double* tauAbsv, int hasAbsorption,
                                                       const double* tauinteractv,
                                                       const double* pathBiasWeightv,
                                                       const double* albedov, int numPaths, int* cellOutv,
                                                       double* distanceOutv, double* tauAbsOutv,
                                                       double* weightOutv)
{
    int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= numPaths) return;

    int begin = pathOffsetv[path];
    int end = pathOffsetv[path + 1];
    if (begin >= end)
    {
        cellOutv[path] = -1;
        distanceOutv[path] = 0.0;
        tauAbsOutv[path] = 0.0;
        weightOutv[path] = 0.0;
        return;
    }

    double taupath = tauv[end - 1];
    if (taupath <= 0.0)
    {
        cellOutv[path] = -1;
        distanceOutv[path] = 0.0;
        tauAbsOutv[path] = 0.0;
        weightOutv[path] = 0.0;
        return;
    }

    double tauinteract = tauinteractv[path];
    int interactionCell = cellv[end - 1];
    double interactionDistance = sv[end - 1];
    double interactionTauAbs = hasAbsorption ? tauAbsv[end - 1] : 0.0;
    for (int i = begin; i != end; ++i)
    {
        if (tauinteract < tauv[i])
        {
            int previous = i - 1;
            double tau0 = i == begin ? 0.0 : tauv[previous];
            double s0 = i == begin ? 0.0 : sv[previous];
            double tauAbs0 = (!hasAbsorption || i == begin) ? 0.0 : tauAbsv[previous];
            double tauAbs1 = hasAbsorption ? tauAbsv[i] : 0.0;
            double t = tauv[i] != tau0 ? (tauinteract - tau0) / (tauv[i] - tau0) : 0.0;
            interactionCell = cellv[i];
            interactionDistance = s0 + t * (sv[i] - s0);
            interactionTauAbs = tauAbs0 + t * (tauAbs1 - tauAbs0);
            break;
        }
    }

    double escapeWeight = -expm1(-taupath);
    double mediumWeight = hasAbsorption ? exp(-interactionTauAbs) : albedov[path];
    cellOutv[path] = interactionCell;
    distanceOutv[path] = interactionDistance;
    tauAbsOutv[path] = interactionTauAbs;
    weightOutv[path] = pathBiasWeightv[path] * escapeWeight * mediumWeight;
}

extern "C" __global__ void forced_propagation_table_albedo_results(
    const int* pathOffsetv, const int* cellv, const double* sv, const double* tauv,
    const double* tauinteractv, const double* pathBiasWeightv, const double* lambdav,
    const double* statev, int numVars, int numTables, const int* densityOffsetv,
    const int* lookupBeginv, const int* lookupCountv, const double* lookupWavelengthv,
    const double* sectionScaTablev, const double* sectionExtTablev, int numPaths,
    int* cellOutv, double* distanceOutv, double* tauAbsOutv, double* weightOutv)
{
    int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= numPaths) return;

    int begin = pathOffsetv[path];
    int end = pathOffsetv[path + 1];
    if (begin >= end)
    {
        cellOutv[path] = -1;
        distanceOutv[path] = 0.0;
        tauAbsOutv[path] = 0.0;
        weightOutv[path] = 0.0;
        return;
    }

    double taupath = tauv[end - 1];
    if (taupath <= 0.0)
    {
        cellOutv[path] = -1;
        distanceOutv[path] = 0.0;
        tauAbsOutv[path] = 0.0;
        weightOutv[path] = 0.0;
        return;
    }

    double tauinteract = tauinteractv[path];
    int interactionCell = cellv[end - 1];
    double interactionDistance = sv[end - 1];
    for (int i = begin; i != end; ++i)
    {
        if (tauinteract < tauv[i])
        {
            int previous = i - 1;
            double tau0 = i == begin ? 0.0 : tauv[previous];
            double s0 = i == begin ? 0.0 : sv[previous];
            double t = tauv[i] != tau0 ? (tauinteract - tau0) / (tauv[i] - tau0) : 0.0;
            interactionCell = cellv[i];
            interactionDistance = s0 + t * (sv[i] - s0);
            break;
        }
    }

    double albedo = 0.0;
    if (interactionCell >= 0)
    {
        double ksca = 0.0;
        double kext = 0.0;
        int base = interactionCell * numVars;
        double lambda = lambdav[path];
        for (int h = 0; h != numTables; ++h)
        {
            int tableIndex = skirt_section_lookup_index(lookupWavelengthv, lookupBeginv[h],
                                                        lookupCountv[h], lambda);
            double n = statev[base + densityOffsetv[h]];
            ksca += sectionScaTablev[tableIndex] * n;
            kext += sectionExtTablev[tableIndex] * n;
        }
        albedo = kext > 0.0 ? ksca / kext : 0.0;
    }

    cellOutv[path] = interactionCell;
    distanceOutv[path] = interactionDistance;
    tauAbsOutv[path] = 0.0;
    weightOutv[path] = pathBiasWeightv[path] * (-expm1(-taupath)) * albedo;
}

static __device__ double skirt_expon_cutoff_from_uniform(double xmax, double u)
{
    if (xmax == 0.0) return 0.0;
    if (xmax < 1.0e-10) return u * xmax;
    double x = -log(1.0 - u * (1.0 - exp(-xmax)));
    return x > xmax ? xmax : x;
}

extern "C" __global__ void forced_propagation_table_albedo_sampled_results(
    const int* pathOffsetv, const int* cellv, const double* sv, const double* tauv,
    const double* randomSelectv, const double* randomSamplev, double pathLengthBias,
    const double* lambdav, const double* statev, int numVars, int numTables,
    const int* densityOffsetv, const int* lookupBeginv, const int* lookupCountv,
    const double* lookupWavelengthv, const double* sectionScaTablev,
    const double* sectionExtTablev, int numPaths, int* cellOutv, double* distanceOutv,
    double* tauAbsOutv, double* weightOutv)
{
    int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= numPaths) return;

    int begin = pathOffsetv[path];
    int end = pathOffsetv[path + 1];
    if (begin >= end)
    {
        cellOutv[path] = -1;
        distanceOutv[path] = 0.0;
        tauAbsOutv[path] = 0.0;
        weightOutv[path] = 0.0;
        return;
    }

    double taupath = tauv[end - 1];
    if (taupath <= 0.0)
    {
        cellOutv[path] = -1;
        distanceOutv[path] = 0.0;
        tauAbsOutv[path] = 0.0;
        weightOutv[path] = 0.0;
        return;
    }

    double u0 = randomSelectv[path];
    double u1 = randomSamplev[path];
    double tauinteract = 0.0;
    double pathBiasWeight = 1.0;
    if (pathLengthBias == 0.0)
    {
        tauinteract = skirt_expon_cutoff_from_uniform(taupath, u0);
    }
    else
    {
        if (u0 < pathLengthBias)
            tauinteract = u1 * taupath;
        else
            tauinteract = skirt_expon_cutoff_from_uniform(taupath, u1);
        double p = -exp(-tauinteract) / expm1(-taupath);
        double q = (1.0 - pathLengthBias) * p + pathLengthBias / taupath;
        pathBiasWeight = p / q;
    }

    int interactionCell = cellv[end - 1];
    double interactionDistance = sv[end - 1];
    for (int i = begin; i != end; ++i)
    {
        if (tauinteract < tauv[i])
        {
            int previous = i - 1;
            double tau0 = i == begin ? 0.0 : tauv[previous];
            double s0 = i == begin ? 0.0 : sv[previous];
            double t = tauv[i] != tau0 ? (tauinteract - tau0) / (tauv[i] - tau0) : 0.0;
            interactionCell = cellv[i];
            interactionDistance = s0 + t * (sv[i] - s0);
            break;
        }
    }

    double albedo = 0.0;
    if (interactionCell >= 0)
    {
        double ksca = 0.0;
        double kext = 0.0;
        int base = interactionCell * numVars;
        double lambda = lambdav[path];
        for (int h = 0; h != numTables; ++h)
        {
            int tableIndex = skirt_section_lookup_index(lookupWavelengthv, lookupBeginv[h],
                                                        lookupCountv[h], lambda);
            double n = statev[base + densityOffsetv[h]];
            ksca += sectionScaTablev[tableIndex] * n;
            kext += sectionExtTablev[tableIndex] * n;
        }
        albedo = kext > 0.0 ? ksca / kext : 0.0;
    }

    cellOutv[path] = interactionCell;
    distanceOutv[path] = interactionDistance;
    tauAbsOutv[path] = 0.0;
    weightOutv[path] = pathBiasWeight * (-expm1(-taupath)) * albedo;
}

extern "C" __global__ void dust_absorbed_luminosity(const double* statev, int numCells, int numWavelengths,
                                                    int numVars, int numDustMedia,
                                                    const int* densityOffsetv,
                                                    const double* sectionAbsv,
                                                    const double* rf1v, const double* rf2v, int hasRf2,
                                                    double* out1v, double* out2v)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= numCells) return;

    int stateBase = m * numVars;
    int rfBase = m * numWavelengths;
    double labs1 = 0.0;
    double labs2 = 0.0;
    for (int ell = 0; ell != numWavelengths; ++ell)
    {
        double opacity = 0.0;
        for (int j = 0; j != numDustMedia; ++j)
        {
            opacity += sectionAbsv[j * numWavelengths + ell] * statev[stateBase + densityOffsetv[j]];
        }
        int index = rfBase + ell;
        labs1 += opacity * rf1v[index];
        if (hasRf2) labs2 += opacity * rf2v[index];
    }
    out1v[m] = labs1;
    out2v[m] = labs2;
}

extern "C" __global__ void dust_absorbed_luminosity_sum(const double* labs1v, const double* labs2v, int numCells,
                                                        double* totalv)
{
    double total1 = 0.0;
    double total2 = 0.0;
    for (int m = 0; m != numCells; ++m)
    {
        total1 += labs1v[m];
        total2 += labs2v[m];
    }
    totalv[0] = total1;
    totalv[1] = total2;
}

extern "C" __global__ void scattering_properties(const double* statev, int cellIndex, int numVars, int numMedia,
                                                 const int* densityOffsetv,
                                                 const double* sectionScav,
                                                 const double* sectionExtv,
                                                 double* albedov, double* weightv)
{
    double ksca = 0.0;
    double kext = 0.0;
    int base = cellIndex * numVars;
    for (int h = 0; h != numMedia; ++h)
    {
        double n = statev[base + densityOffsetv[h]];
        double sca = sectionScav[h] * n;
        weightv[h] = sca;
        ksca += sca;
        kext += sectionExtv[h] * n;
    }
    if (ksca > 0.0)
    {
        for (int h = 0; h != numMedia; ++h) weightv[h] /= ksca;
    }
    else
    {
        for (int h = 0; h != numMedia; ++h) weightv[h] = 0.0;
    }
    albedov[0] = kext > 0.0 ? ksca / kext : 0.0;
}

extern "C" __global__ void scattering_albedos(const double* statev, const int* cellv, int numVars, int numMedia,
                                              const int* densityOffsetv, const double* sectionScav,
                                              const double* sectionExtv, int numPaths, double* albedov)
{
    int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= numPaths) return;

    int cellIndex = cellv[path];
    if (cellIndex < 0)
    {
        albedov[path] = 0.0;
        return;
    }

    double ksca = 0.0;
    double kext = 0.0;
    int base = cellIndex * numVars;
    for (int h = 0; h != numMedia; ++h)
    {
        double n = statev[base + densityOffsetv[h]];
        ksca += sectionScav[h] * n;
        kext += sectionExtv[h] * n;
    }
    albedov[path] = kext > 0.0 ? ksca / kext : 0.0;
}

extern "C" __global__ void table_scattering_properties(const double* statev, int cellIndex, int numVars,
                                                       int numTables, const int* densityOffsetv,
                                                       const int* lookupBeginv,
                                                       const int* lookupCountv,
                                                       const double* lookupWavelengthv,
                                                       const double* sectionScaTablev,
                                                       const double* sectionExtTablev,
                                                       double lambda, double* albedov, double* weightv)
{
    double ksca = 0.0;
    double kext = 0.0;
    int base = cellIndex * numVars;
    for (int h = 0; h != numTables; ++h)
    {
        int tableIndex = skirt_section_lookup_index(lookupWavelengthv, lookupBeginv[h], lookupCountv[h], lambda);
        double n = statev[base + densityOffsetv[h]];
        double sca = sectionScaTablev[tableIndex] * n;
        weightv[h] = sca;
        ksca += sca;
        kext += sectionExtTablev[tableIndex] * n;
    }
    if (ksca > 0.0)
    {
        for (int h = 0; h != numTables; ++h) weightv[h] /= ksca;
    }
    else
    {
        for (int h = 0; h != numTables; ++h) weightv[h] = 0.0;
    }
    albedov[0] = kext > 0.0 ? ksca / kext : 0.0;
}

extern "C" __global__ void table_scattering_albedos(const double* statev, const int* cellv,
                                                    const double* lambdav, int numVars, int numTables,
                                                    const int* densityOffsetv, const int* lookupBeginv,
                                                    const int* lookupCountv, const double* lookupWavelengthv,
                                                    const double* sectionScaTablev,
                                                    const double* sectionExtTablev, int numPaths,
                                                    double* albedov)
{
    int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= numPaths) return;

    int cellIndex = cellv[path];
    if (cellIndex < 0)
    {
        albedov[path] = 0.0;
        return;
    }

    double ksca = 0.0;
    double kext = 0.0;
    int base = cellIndex * numVars;
    double lambda = lambdav[path];
    for (int h = 0; h != numTables; ++h)
    {
        int tableIndex = skirt_section_lookup_index(lookupWavelengthv, lookupBeginv[h], lookupCountv[h], lambda);
        double n = statev[base + densityOffsetv[h]];
        ksca += sectionScaTablev[tableIndex] * n;
        kext += sectionExtTablev[tableIndex] * n;
    }
    albedov[path] = kext > 0.0 ? ksca / kext : 0.0;
}

extern "C" __global__ void scale_wavelength_values(double* values, unsigned long long numValues,
                                                   const double* factorv)
{
    unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numValues) return;
    values[i] *= factorv[i];
}

extern "C" __global__ void scale_frame_wavelength_values(double* values, unsigned long long numValues,
                                                         unsigned long long numPixelsInFrame,
                                                         const double* factorv)
{
    unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numValues || numPixelsInFrame == 0) return;
    unsigned long long ell = i / numPixelsInFrame;
    values[i] *= factorv[ell];
}

extern "C" __global__ void divide_values(double* values, unsigned long long numValues, double divisor)
{
    unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numValues) return;
    values[i] /= divisor;
}

extern "C" __global__ void multiply_values(double* values, unsigned long long numValues, double factor)
{
    unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numValues) return;
    values[i] *= factor;
}

extern "C" __global__ void sum_values(double* output, unsigned long long numValues, const double* value1v,
                                      const double* value2v, const double* value3v, const double* value4v,
                                      int numInputs)
{
    unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numValues) return;

    double value = value1v[i] + value2v[i];
    if (numInputs > 2) value += value3v[i];
    if (numInputs > 3) value += value4v[i];
    output[i] = value;
}

extern "C" __global__ void emitting_cell_counts(const double* luminosityv, unsigned long long numValues,
                                                unsigned int* partialCountv)
{
    __shared__ unsigned int counts[256];
    unsigned int tid = threadIdx.x;
    unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + tid;
    counts[tid] = i < numValues && luminosityv[i] > 0.0 ? 1u : 0u;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride) counts[tid] += counts[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partialCountv[blockIdx.x] = counts[0];
}

extern "C" __global__ void composite_launch_weights(const double* luminosityv, unsigned long long numValues,
                                                    unsigned long long emittingCells, double spatialBias,
                                                    double* weightv)
{
    unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numValues) return;

    double uniformWeight = luminosityv[i] > 0.0 ? 1.0 / static_cast<double>(emittingCells) : 0.0;
    weightv[i] = (1.0 - spatialBias) * luminosityv[i] + spatialBias * uniformWeight;
}

static __device__ int skirt_lower_bound_double(const double* values, int count, double x)
{
    int first = 0;
    int len = count;
    while (len > 0)
    {
        int half = len >> 1;
        int mid = first + half;
        if (values[mid] < x)
        {
            first = mid + 1;
            len -= half + 1;
        }
        else
            len = half;
    }
    return first;
}

static __device__ int skirt_upper_bound_double(const double* values, int count, double x)
{
    int first = 0;
    int len = count;
    while (len > 0)
    {
        int half = len >> 1;
        int mid = first + half;
        if (!(x < values[mid]))
        {
            first = mid + 1;
            len -= half + 1;
        }
        else
            len = half;
    }
    return first;
}

static __device__ double skirt_gln(double p, double x)
{
    double q = 1.0 - p;
    if (q == 0.0) return log(x);
    if (fabs(q) < 1e-3)
    {
        double lnx = log(x);
        double s = q * lnx;
        return lnx * (1.0 + 0.5 * s + 1.0 / 6.0 * s * s + 1.0 / 24.0 * s * s * s);
    }
    return (pow(x, q) - 1.0) / q;
}

static __device__ double skirt_gexp(double p, double x)
{
    double q = 1.0 - p;
    if (q == 0.0) return exp(x);
    if (fabs(q) < 1e-3)
    {
        double x2 = x * x;
        return exp(x)
               * (1.0 - 0.5 * x2 * q + 1.0 / 24.0 * x * x2 * (8.0 + 3.0 * x) * q * q
                  - 1.0 / 48.0 * x2 * x2 * (12.0 + 8.0 * x + x2) * q * q * q);
    }
    return pow(1.0 + q * x, 1.0 / q);
}

static __device__ unsigned long long skirt_table_flattened_index(const int* indices, const int* axisLengths,
                                                                 int numAxes)
{
    unsigned long long result = static_cast<unsigned long long>(indices[numAxes - 1]);
    for (int k = numAxes - 2; k >= 0; --k)
    {
        result = result * static_cast<unsigned long long>(axisLengths[k])
                 + static_cast<unsigned long long>(indices[k]);
    }
    return result;
}

static __device__ double skirt_stored_table_pdf(const double* axisValues, const int* axisOffsets,
                                                const int* axisLengths, const int* axisLog, const double* quantity,
                                                int numAxes, int quantityLog, int clampFirstAxis,
                                                const double* parameters, int entity, double x, int pointIndex,
                                                int minRight, int lastPoint)
{
    int i2[5] = {0, 0, 0, 0, 0};
    double f[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    for (int k = 1; k != numAxes; ++k)
    {
        const double* axis = axisValues + axisOffsets[k];
        int length = axisLengths[k];
        double value = parameters[static_cast<unsigned long long>(entity) * (numAxes - 1) + (k - 1)];
        int right = skirt_lower_bound_double(axis, length, value);
        if (right == 0)
        {
            right++;
            value = axis[0];
        }
        else if (right == length)
        {
            right--;
            value = axis[right];
        }
        i2[k] = right;

        double x1 = axis[right - 1];
        double x2 = axis[right];
        if (axisLog[k])
        {
            value = log(value);
            x1 = log(x1);
            x2 = log(x2);
        }
        f[k] = (value - x1) / (x2 - x1);
    }

    const double* firstAxis = axisValues + axisOffsets[0];
    if (pointIndex == 0 || pointIndex == lastPoint)
    {
        int length = axisLengths[0];
        int right = skirt_lower_bound_double(firstAxis, length, x);
        if (right == 0)
        {
            if (!clampFirstAxis && x != firstAxis[0]) return 0.0;
            right++;
            x = firstAxis[0];
        }
        else if (right == length)
        {
            if (!clampFirstAxis) return 0.0;
            right--;
            x = firstAxis[right];
        }
        i2[0] = right;

        double x1 = firstAxis[right - 1];
        double x2 = firstAxis[right];
        if (axisLog[0])
        {
            x = log(x);
            x1 = log(x1);
            x2 = log(x2);
        }
        f[0] = (x - x1) / (x2 - x1);
    }
    else
    {
        i2[0] = pointIndex + minRight - 1;
        f[0] = 1.0;
    }

    int numTerms = 1 << numAxes;
    double y = 0.0;
    for (int t = 0; t != numTerms; ++t)
    {
        int term = t;
        double front = 1.0;
        int indices[5] = {0, 0, 0, 0, 0};
        for (int k = 0; k != numAxes; ++k)
        {
            int left = term & 1;
            indices[k] = i2[k] - left;
            front *= left ? (1.0 - f[k]) : f[k];
            term >>= 1;
        }
        if (front)
        {
            double value = quantity[skirt_table_flattened_index(indices, axisLengths, numAxes)];
            if (quantityLog)
            {
                if (value <= 0.0) return 0.0;
                y += front * log(value);
            }
            else
                y += front * value;
        }
    }
    return quantityLog ? exp(y) : y;
}

extern "C" __global__ void stored_table_cdf_batch(const double* axisValues, const int* axisOffsets,
                                                  const int* axisLengths, const int* axisLog, const double* quantity,
                                                  int numAxes, int quantityLog, int clampFirstAxis, double xmin,
                                                  double xmax, const double* parameters, const double* scaleValues,
                                                  unsigned long long numEntities, double* luminosities)
{
    unsigned long long entity = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (entity >= numEntities || numAxes < 2 || numAxes > 5)
        return;

    const double* firstAxis = axisValues + axisOffsets[0];
    int firstLength = axisLengths[0];
    int minRight = skirt_upper_bound_double(firstAxis, firstLength, xmin);
    int maxRight = skirt_lower_bound_double(firstAxis, firstLength, xmax);
    int numPoints = 2 + maxRight - minRight;
    if (numPoints < 2)
    {
        luminosities[entity] = 0.0;
        return;
    }

    double prevX = xmin;
    double prevPdf = skirt_stored_table_pdf(axisValues, axisOffsets, axisLengths, axisLog, quantity, numAxes,
                                            quantityLog, clampFirstAxis, parameters, static_cast<int>(entity), prevX,
                                            0, minRight, numPoints - 1);
    double norm = 0.0;
    int loglog = axisLog[0] && quantityLog;
    for (int i = 1; i != numPoints; ++i)
    {
        double x = (i == numPoints - 1) ? xmax : firstAxis[i + minRight - 1];
        double pdf = skirt_stored_table_pdf(axisValues, axisOffsets, axisLengths, axisLog, quantity, numAxes,
                                            quantityLog, clampFirstAxis, parameters, static_cast<int>(entity), x, i,
                                            minRight, numPoints - 1);
        double area = 0.0;
        if (!loglog)
            area = 0.5 * (prevPdf + pdf) * (x - prevX);
        else if (prevPdf > 0.0 && pdf > 0.0)
        {
            double alpha = log(pdf / prevPdf) / log(x / prevX);
            area = prevPdf * prevX * skirt_gln(-alpha, x / prevX);
        }
        norm += area;
        prevX = x;
        prevPdf = pdf;
    }

    luminosities[entity] = scaleValues[entity] * norm;
}

extern "C" __global__ void stored_table_sample_wavelength_batch(
    const double* axisValues, const int* axisOffsets, const int* axisLengths, const int* axisLog,
    const double* quantity, int numAxes, int quantityLog, int clampFirstAxis, double xmin, double xmax,
    const double* parameters, const double* intrinsicRandoms, const double* forcedWavelengths,
    int hasForcedWavelengths, unsigned long long numSamples, double* wavelengths, double* specificLuminosities)
{
    unsigned long long sample = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sample >= numSamples || numAxes < 2 || numAxes > 5)
        return;

    const double* firstAxis = axisValues + axisOffsets[0];
    int firstLength = axisLengths[0];
    int minRight = skirt_upper_bound_double(firstAxis, firstLength, xmin);
    int maxRight = skirt_lower_bound_double(firstAxis, firstLength, xmax);
    int numPoints = 2 + maxRight - minRight;
    if (numPoints < 2)
    {
        wavelengths[sample] = xmin;
        specificLuminosities[sample] = 0.0;
        return;
    }

    int sampleIndex = static_cast<int>(sample);
    double prevX = xmin;
    double prevPdf = skirt_stored_table_pdf(axisValues, axisOffsets, axisLengths, axisLog, quantity, numAxes,
                                            quantityLog, clampFirstAxis, parameters, sampleIndex, prevX, 0, minRight,
                                            numPoints - 1);
    double norm = 0.0;
    int loglog = axisLog[0] && quantityLog;
    for (int i = 1; i != numPoints; ++i)
    {
        double x = (i == numPoints - 1) ? xmax : firstAxis[i + minRight - 1];
        double pdf = skirt_stored_table_pdf(axisValues, axisOffsets, axisLengths, axisLog, quantity, numAxes,
                                            quantityLog, clampFirstAxis, parameters, sampleIndex, x, i, minRight,
                                            numPoints - 1);
        double area = 0.0;
        if (!loglog)
            area = 0.5 * (prevPdf + pdf) * (x - prevX);
        else if (prevPdf > 0.0 && pdf > 0.0)
        {
            double alpha = log(pdf / prevPdf) / log(x / prevX);
            area = prevPdf * prevX * skirt_gln(-alpha, x / prevX);
        }
        norm += area;
        prevX = x;
        prevPdf = pdf;
    }

    double lambda = hasForcedWavelengths && forcedWavelengths[sample] > 0.0 ? forcedWavelengths[sample] : xmin;
    if (norm > 0.0 && !(hasForcedWavelengths && forcedWavelengths[sample] > 0.0))
    {
        double target = intrinsicRandoms[sample] * norm;
        double cumulative = 0.0;
        prevX = xmin;
        prevPdf = skirt_stored_table_pdf(axisValues, axisOffsets, axisLengths, axisLog, quantity, numAxes,
                                         quantityLog, clampFirstAxis, parameters, sampleIndex, prevX, 0, minRight,
                                         numPoints - 1);
        for (int i = 1; i != numPoints; ++i)
        {
            double x = (i == numPoints - 1) ? xmax : firstAxis[i + minRight - 1];
            double pdf = skirt_stored_table_pdf(axisValues, axisOffsets, axisLengths, axisLog, quantity, numAxes,
                                                quantityLog, clampFirstAxis, parameters, sampleIndex, x, i, minRight,
                                                numPoints - 1);
            double area = 0.0;
            if (!loglog)
                area = 0.5 * (prevPdf + pdf) * (x - prevX);
            else if (prevPdf > 0.0 && pdf > 0.0)
            {
                double alpha = log(pdf / prevPdf) / log(x / prevX);
                area = prevPdf * prevX * skirt_gln(-alpha, x / prevX);
            }

            if (target <= cumulative + area || i == numPoints - 1)
            {
                double local = target - cumulative;
                if (prevPdf > 0.0 && pdf > 0.0 && prevX > 0.0 && x > 0.0 && x != prevX)
                {
                    double alpha = log(pdf / prevPdf) / log(x / prevX);
                    lambda = prevX * skirt_gexp(-alpha, local / (prevPdf * prevX));
                }
                else
                    lambda = area > 0.0 ? prevX + (x - prevX) * local / area : prevX;
                break;
            }
            cumulative += area;
            prevX = x;
            prevPdf = pdf;
        }
    }

    double pdfAtLambda = norm > 0.0 ? skirt_stored_table_pdf(axisValues, axisOffsets, axisLengths, axisLog, quantity,
                                                             numAxes, quantityLog, clampFirstAxis, parameters,
                                                             sampleIndex, lambda, 0, 0, 0)
                                    : 0.0;
    wavelengths[sample] = lambda;
    specificLuminosities[sample] = norm > 0.0 ? pdfAtLambda / norm : 0.0;
}
)cuda";
    }

    class CudaRuntime final
    {
    public:
        explicit CudaRuntime(int deviceSlot = -1) : _deviceSlot(deviceSlot) {}

        bool available()
        {
            std::lock_guard<std::mutex> lock(_mutex);
            return ensureReady();
        }

        int availableDeviceCount()
        {
            std::lock_guard<std::mutex> lock(_mutex);
            return ensureReady() ? _deviceCount : 0;
        }

        string status()
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (ensureReady())
            {
                std::ostringstream out;
                out << "enabled on " << _deviceName << " (device " << _deviceIndex << "/" << _deviceCount
                    << ", compute " << _major << "." << _minor << "; path generation "
                    << (pathGenerationEnabled() ? "GPU" : "CPU") << "; sync photon cycle "
                    << (synchronousPhotonCycleEnabled() ? "GPU" : "CPU") << ")";
                return out.str();
            }
            return "unavailable: " + _error;
        }

        string lastError()
        {
            std::lock_guard<std::mutex> lock(_mutex);
            return _error;
        }

        void invalidateMediumState(const MediumState& state)
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_stateHost == state.data()) _stateDirty = true;
        }

        bool traceCartesianGridPath(const double* xv, const double* yv, const double* zv, int nx, int ny, int nz,
                                    const Position& position, const Direction& direction, double xmin, double ymin,
                                    double zmin, double xmax, double ymax, double zmax, double maxDistance,
                                    SpatialGridPath* path)
        {
            if (!path || !xv || !yv || !zv || nx <= 0 || ny <= 0 || nz <= 0) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            double rx, ry, rz;
            position.cartesian(rx, ry, rz);
            double kx, ky, kz;
            direction.cartesian(kx, ky, kz);

            size_t capacity = static_cast<size_t>(nx) + static_cast<size_t>(ny) + static_cast<size_t>(nz) + 2;
            if (capacity > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            int maxSegments = static_cast<int>(capacity);

            vector<int> cellv(capacity, -1);
            vector<double> dsv(capacity, 0.);
            int count = 0;
            int status = 0;

            CUdeviceptr dXv = 0;
            CUdeviceptr dYv = 0;
            CUdeviceptr dZv = 0;
            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dCountv = 0;
            CUdeviceptr dStatusv = 0;

            auto freeAll = [&]() {
                if (dXv) _cuMemFree(dXv);
                if (dYv) _cuMemFree(dYv);
                if (dZv) _cuMemFree(dZv);
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dCountv) _cuMemFree(dCountv);
                if (dStatusv) _cuMemFree(dStatusv);
            };

            bool ok = allocateAndCopy(dXv, xv, static_cast<size_t>(nx + 1) * sizeof(double))
                      && allocateAndCopy(dYv, yv, static_cast<size_t>(ny + 1) * sizeof(double))
                      && allocateAndCopy(dZv, zv, static_cast<size_t>(nz + 1) * sizeof(double))
                      && allocate(dCellv, cellv.size() * sizeof(int))
                      && allocate(dDsv, dsv.size() * sizeof(double)) && allocate(dCountv, sizeof(int))
                      && allocate(dStatusv, sizeof(int));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dXv,      &dYv,      &dZv,       &nx,        &ny,        &nz,
                            &rx,       &ry,       &rz,        &kx,        &ky,        &kz,
                            &xmin,     &ymin,     &zmin,      &xmax,      &ymax,      &zmax,
                            &maxDistance, &maxSegments, &dCellv,   &dDsv,      &dCountv,  &dStatusv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelCartesianGridPath, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                          "cuLaunchKernel(cartesian_grid_path)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(&count, dCountv, sizeof(int)), "cuMemcpyDtoH(cartesian path count)")
                 && check(_cuMemcpyDtoH(&status, dStatusv, sizeof(int)), "cuMemcpyDtoH(cartesian path status)");
            if (ok && status == 1 && count > 0)
            {
                ok = check(_cuMemcpyDtoH(cellv.data(), dCellv, static_cast<size_t>(count) * sizeof(int)),
                           "cuMemcpyDtoH(cartesian path cells)")
                     && check(_cuMemcpyDtoH(dsv.data(), dDsv, static_cast<size_t>(count) * sizeof(double)),
                              "cuMemcpyDtoH(cartesian path lengths)");
            }

            freeAll();
            if (!ok || status != 1 || count < 0 || count > maxSegments) return false;

            path->clear();
            for (int i = 0; i != count; ++i) path->addSegment(cellv[i], dsv[i]);
            return true;
        }

        bool traceTreeGridPath(const void* gridKey, const vector<double>& nodeBoundsv,
                               const vector<int>& childBeginv, const vector<int>& childCountv,
                               const vector<int>& childIndexv, const vector<int>& cellIndexv,
                               const Position& position, const Direction& direction, double xmin, double ymin,
                               double zmin, double xmax, double ymax, double zmax, double maxDistance,
                               SpatialGridPath* path)
        {
            if (!path || !gridKey || cellIndexv.empty() || nodeBoundsv.size() != 6 * cellIndexv.size()
                || childBeginv.size() != cellIndexv.size() || childCountv.size() != cellIndexv.size())
                return false;

            size_t leafCount = 0;
            for (int cellIndex : cellIndexv)
                if (cellIndex >= 0) ++leafCount;
            if (!leafCount) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;
            if (!ensureTreeOnDevice(gridKey, nodeBoundsv, childBeginv, childCountv, childIndexv, cellIndexv))
                return false;

            double rx, ry, rz;
            position.cartesian(rx, ry, rz);
            double kx, ky, kz;
            direction.cartesian(kx, ky, kz);
            double dx = xmax - xmin;
            double dy = ymax - ymin;
            double dz = zmax - zmin;
            double eps = 1e-12 * sqrt(dx * dx + dy * dy + dz * dz);

            size_t capacity = leafCount + 1;
            if (capacity > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            int maxSegments = static_cast<int>(capacity);
            int numNodes = static_cast<int>(cellIndexv.size());

            vector<int> cellv(capacity, -1);
            vector<double> dsv(capacity, 0.);
            int count = 0;
            int status = 0;

            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dCountv = 0;
            CUdeviceptr dStatusv = 0;

            auto freeAll = [&]() {
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dCountv) _cuMemFree(dCountv);
                if (dStatusv) _cuMemFree(dStatusv);
            };

            bool ok = allocate(dCellv, cellv.size() * sizeof(int)) && allocate(dDsv, dsv.size() * sizeof(double))
                      && allocate(dCountv, sizeof(int)) && allocate(dStatusv, sizeof(int));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&_treeBoundsDevice,     &_treeChildBeginDevice, &_treeChildCountDevice,
                            &_treeChildIndexDevice, &_treeCellIndexDevice,  &numNodes,
                            &rx,                    &ry,                    &rz,
                            &kx,                    &ky,                    &kz,
                            &xmin,                  &ymin,                  &zmin,
                            &xmax,                  &ymax,                  &zmax,
                            &eps,                   &maxDistance,           &maxSegments,
                            &dCellv,                &dDsv,                  &dCountv,
                            &dStatusv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelTreeGridPath, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                          "cuLaunchKernel(tree_grid_path)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(&count, dCountv, sizeof(int)), "cuMemcpyDtoH(tree path count)")
                 && check(_cuMemcpyDtoH(&status, dStatusv, sizeof(int)), "cuMemcpyDtoH(tree path status)");
            if (ok && status == 1 && count > 0)
            {
                ok = check(_cuMemcpyDtoH(cellv.data(), dCellv, static_cast<size_t>(count) * sizeof(int)),
                           "cuMemcpyDtoH(tree path cells)")
                     && check(_cuMemcpyDtoH(dsv.data(), dDsv, static_cast<size_t>(count) * sizeof(double)),
                              "cuMemcpyDtoH(tree path lengths)");
            }

            freeAll();
            if (!ok || status != 1 || count < 0 || count > maxSegments) return false;

            path->clear();
            for (int i = 0; i != count; ++i) path->addSegment(cellv[i], dsv[i]);
            return true;
        }

        bool traceSphere1DGridPath(const double* rv, int nr, const Position& position, const Direction& direction,
                                   double maxDistance, SpatialGridPath* path)
        {
            if (!path || !rv || nr <= 0) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            double rx, ry, rz;
            position.cartesian(rx, ry, rz);
            double kx, ky, kz;
            direction.cartesian(kx, ky, kz);

            size_t capacity = 2 * static_cast<size_t>(nr) + 2;
            if (capacity > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            int maxSegments = static_cast<int>(capacity);

            vector<int> cellv(capacity, -1);
            vector<double> dsv(capacity, 0.);
            int count = 0;
            int status = 0;

            CUdeviceptr dRv = 0;
            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dCountv = 0;
            CUdeviceptr dStatusv = 0;

            auto freeAll = [&]() {
                if (dRv) _cuMemFree(dRv);
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dCountv) _cuMemFree(dCountv);
                if (dStatusv) _cuMemFree(dStatusv);
            };

            bool ok = allocateAndCopy(dRv, rv, static_cast<size_t>(nr + 1) * sizeof(double))
                      && allocate(dCellv, cellv.size() * sizeof(int)) && allocate(dDsv, dsv.size() * sizeof(double))
                      && allocate(dCountv, sizeof(int)) && allocate(dStatusv, sizeof(int));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dRv, &nr, &rx, &ry, &rz, &kx, &ky, &kz, &maxDistance, &maxSegments,
                            &dCellv, &dDsv, &dCountv, &dStatusv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelSphere1DGridPath, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                          "cuLaunchKernel(sphere1d_grid_path)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(&count, dCountv, sizeof(int)), "cuMemcpyDtoH(sphere1d path count)")
                 && check(_cuMemcpyDtoH(&status, dStatusv, sizeof(int)), "cuMemcpyDtoH(sphere1d path status)");
            if (ok && status == 1 && count > 0)
            {
                ok = check(_cuMemcpyDtoH(cellv.data(), dCellv, static_cast<size_t>(count) * sizeof(int)),
                           "cuMemcpyDtoH(sphere1d path cells)")
                     && check(_cuMemcpyDtoH(dsv.data(), dDsv, static_cast<size_t>(count) * sizeof(double)),
                              "cuMemcpyDtoH(sphere1d path lengths)");
            }

            freeAll();
            if (!ok || status != 1 || count < 0 || count > maxSegments) return false;

            path->clear();
            for (int i = 0; i != count; ++i) path->addSegment(cellv[i], dsv[i]);
            return true;
        }

        bool traceSphere2DGridPath(const double* rv, const double* thetav, const double* cv, int nr, int ntheta,
                                   const Position& position, const Direction& direction, double maxDistance,
                                   SpatialGridPath* path)
        {
            if (!path || !rv || !thetav || !cv || nr <= 0 || ntheta <= 0) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            double rx, ry, rz;
            position.cartesian(rx, ry, rz);
            double kx, ky, kz;
            direction.cartesian(kx, ky, kz);

            size_t capacity = 4 * (static_cast<size_t>(nr) + static_cast<size_t>(ntheta)) + 8;
            if (capacity > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            int maxSegments = static_cast<int>(capacity);

            vector<int> cellv(capacity, -1);
            vector<double> dsv(capacity, 0.);
            int count = 0;
            int status = 0;

            CUdeviceptr dRv = 0;
            CUdeviceptr dThetav = 0;
            CUdeviceptr dCv = 0;
            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dCountv = 0;
            CUdeviceptr dStatusv = 0;

            auto freeAll = [&]() {
                if (dRv) _cuMemFree(dRv);
                if (dThetav) _cuMemFree(dThetav);
                if (dCv) _cuMemFree(dCv);
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dCountv) _cuMemFree(dCountv);
                if (dStatusv) _cuMemFree(dStatusv);
            };

            bool ok = allocateAndCopy(dRv, rv, static_cast<size_t>(nr + 1) * sizeof(double))
                      && allocateAndCopy(dThetav, thetav, static_cast<size_t>(ntheta + 1) * sizeof(double))
                      && allocateAndCopy(dCv, cv, static_cast<size_t>(ntheta + 1) * sizeof(double))
                      && allocate(dCellv, cellv.size() * sizeof(int)) && allocate(dDsv, dsv.size() * sizeof(double))
                      && allocate(dCountv, sizeof(int)) && allocate(dStatusv, sizeof(int));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dRv,   &dThetav, &dCv,       &nr,        &ntheta,   &rx,       &ry,
                            &rz,    &kx,      &ky,        &kz,        &maxDistance, &maxSegments,
                            &dCellv, &dDsv,   &dCountv,   &dStatusv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelSphere2DGridPath, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                          "cuLaunchKernel(sphere2d_grid_path)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(&count, dCountv, sizeof(int)), "cuMemcpyDtoH(sphere2d path count)")
                 && check(_cuMemcpyDtoH(&status, dStatusv, sizeof(int)), "cuMemcpyDtoH(sphere2d path status)");
            if (ok && status == 1 && count > 0)
            {
                ok = check(_cuMemcpyDtoH(cellv.data(), dCellv, static_cast<size_t>(count) * sizeof(int)),
                           "cuMemcpyDtoH(sphere2d path cells)")
                     && check(_cuMemcpyDtoH(dsv.data(), dDsv, static_cast<size_t>(count) * sizeof(double)),
                              "cuMemcpyDtoH(sphere2d path lengths)");
            }

            freeAll();
            if (!ok || status != 1 || count < 0 || count > maxSegments) return false;

            path->clear();
            for (int i = 0; i != count; ++i) path->addSegment(cellv[i], dsv[i]);
            return true;
        }

        bool traceSphere3DGridPath(const double* rv, const double* thetav, const double* phiv, const double* cv,
                                   const double* sinv, const double* cosv, int nr, int ntheta, int nphi, double eps,
                                   const Position& position, const Direction& direction, double maxDistance,
                                   SpatialGridPath* path)
        {
            if (!path || !rv || !thetav || !phiv || !cv || !sinv || !cosv || nr <= 0 || ntheta <= 0 || nphi <= 0)
                return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            double rx, ry, rz;
            position.cartesian(rx, ry, rz);
            double kx, ky, kz;
            direction.cartesian(kx, ky, kz);

            size_t capacity =
                6 * (static_cast<size_t>(nr) + static_cast<size_t>(ntheta) + static_cast<size_t>(nphi)) + 16;
            if (capacity > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            int maxSegments = static_cast<int>(capacity);

            vector<int> cellv(capacity, -1);
            vector<double> dsv(capacity, 0.);
            int count = 0;
            int status = 0;

            CUdeviceptr dRv = 0;
            CUdeviceptr dThetav = 0;
            CUdeviceptr dPhiv = 0;
            CUdeviceptr dCv = 0;
            CUdeviceptr dSinv = 0;
            CUdeviceptr dCosv = 0;
            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dCountv = 0;
            CUdeviceptr dStatusv = 0;

            auto freeAll = [&]() {
                if (dRv) _cuMemFree(dRv);
                if (dThetav) _cuMemFree(dThetav);
                if (dPhiv) _cuMemFree(dPhiv);
                if (dCv) _cuMemFree(dCv);
                if (dSinv) _cuMemFree(dSinv);
                if (dCosv) _cuMemFree(dCosv);
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dCountv) _cuMemFree(dCountv);
                if (dStatusv) _cuMemFree(dStatusv);
            };

            bool ok = allocateAndCopy(dRv, rv, static_cast<size_t>(nr + 1) * sizeof(double))
                      && allocateAndCopy(dThetav, thetav, static_cast<size_t>(ntheta + 1) * sizeof(double))
                      && allocateAndCopy(dPhiv, phiv, static_cast<size_t>(nphi + 1) * sizeof(double))
                      && allocateAndCopy(dCv, cv, static_cast<size_t>(ntheta + 1) * sizeof(double))
                      && allocateAndCopy(dSinv, sinv, static_cast<size_t>(nphi + 1) * sizeof(double))
                      && allocateAndCopy(dCosv, cosv, static_cast<size_t>(nphi + 1) * sizeof(double))
                      && allocate(dCellv, cellv.size() * sizeof(int)) && allocate(dDsv, dsv.size() * sizeof(double))
                      && allocate(dCountv, sizeof(int)) && allocate(dStatusv, sizeof(int));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dRv,    &dThetav, &dPhiv,     &dCv,      &dSinv,       &dCosv,
                            &nr,     &ntheta,  &nphi,      &eps,      &rx,          &ry,
                            &rz,     &kx,      &ky,        &kz,       &maxDistance, &maxSegments,
                            &dCellv, &dDsv,    &dCountv,   &dStatusv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelSphere3DGridPath, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                          "cuLaunchKernel(sphere3d_grid_path)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(&count, dCountv, sizeof(int)), "cuMemcpyDtoH(sphere3d path count)")
                 && check(_cuMemcpyDtoH(&status, dStatusv, sizeof(int)), "cuMemcpyDtoH(sphere3d path status)");
            if (ok && status == 1 && count > 0)
            {
                ok = check(_cuMemcpyDtoH(cellv.data(), dCellv, static_cast<size_t>(count) * sizeof(int)),
                           "cuMemcpyDtoH(sphere3d path cells)")
                     && check(_cuMemcpyDtoH(dsv.data(), dDsv, static_cast<size_t>(count) * sizeof(double)),
                              "cuMemcpyDtoH(sphere3d path lengths)");
            }

            freeAll();
            if (!ok || status != 1 || count < 0 || count > maxSegments) return false;

            path->clear();
            for (int i = 0; i != count; ++i) path->addSegment(cellv[i], dsv[i]);
            return true;
        }

        bool traceCylinder2DGridPath(const double* Rv, const double* zv, int nR, int nz, const Position& position,
                                     const Direction& direction, double maxDistance, SpatialGridPath* path)
        {
            if (!path || !Rv || !zv || nR <= 0 || nz <= 0) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            double rx, ry, rz;
            position.cartesian(rx, ry, rz);
            double kx, ky, kz;
            direction.cartesian(kx, ky, kz);

            size_t capacity = 4 * (static_cast<size_t>(nR) + static_cast<size_t>(nz)) + 8;
            if (capacity > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            int maxSegments = static_cast<int>(capacity);

            vector<int> cellv(capacity, -1);
            vector<double> dsv(capacity, 0.);
            int count = 0;
            int status = 0;

            CUdeviceptr dRv = 0;
            CUdeviceptr dZv = 0;
            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dCountv = 0;
            CUdeviceptr dStatusv = 0;

            auto freeAll = [&]() {
                if (dRv) _cuMemFree(dRv);
                if (dZv) _cuMemFree(dZv);
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dCountv) _cuMemFree(dCountv);
                if (dStatusv) _cuMemFree(dStatusv);
            };

            bool ok = allocateAndCopy(dRv, Rv, static_cast<size_t>(nR + 1) * sizeof(double))
                      && allocateAndCopy(dZv, zv, static_cast<size_t>(nz + 1) * sizeof(double))
                      && allocate(dCellv, cellv.size() * sizeof(int)) && allocate(dDsv, dsv.size() * sizeof(double))
                      && allocate(dCountv, sizeof(int)) && allocate(dStatusv, sizeof(int));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dRv,        &dZv,   &nR,       &nz,       &rx,       &ry,      &rz,
                            &kx,         &ky,    &kz,       &maxDistance, &maxSegments,
                            &dCellv,     &dDsv,  &dCountv,  &dStatusv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelCylinder2DGridPath, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                          "cuLaunchKernel(cylinder2d_grid_path)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(&count, dCountv, sizeof(int)), "cuMemcpyDtoH(cylinder2d path count)")
                 && check(_cuMemcpyDtoH(&status, dStatusv, sizeof(int)), "cuMemcpyDtoH(cylinder2d path status)");
            if (ok && status == 1 && count > 0)
            {
                ok = check(_cuMemcpyDtoH(cellv.data(), dCellv, static_cast<size_t>(count) * sizeof(int)),
                           "cuMemcpyDtoH(cylinder2d path cells)")
                     && check(_cuMemcpyDtoH(dsv.data(), dDsv, static_cast<size_t>(count) * sizeof(double)),
                              "cuMemcpyDtoH(cylinder2d path lengths)");
            }

            freeAll();
            if (!ok || status != 1 || count < 0 || count > maxSegments) return false;

            path->clear();
            for (int i = 0; i != count; ++i) path->addSegment(cellv[i], dsv[i]);
            return true;
        }

        bool traceCylinder3DGridPath(const double* Rv, const double* phiv, const double* zv, const double* sinv,
                                     const double* cosv, int nR, int nphi, int nz, double eps, bool hasHole,
                                     const Position& position, const Direction& direction, double maxDistance,
                                     SpatialGridPath* path)
        {
            if (!path || !Rv || !phiv || !zv || !sinv || !cosv || nR <= 0 || nphi <= 0 || nz <= 0) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            double rx, ry, rz;
            position.cartesian(rx, ry, rz);
            double kx, ky, kz;
            direction.cartesian(kx, ky, kz);
            int hasHoleValue = hasHole ? 1 : 0;

            size_t capacity =
                8 * (static_cast<size_t>(nR) + static_cast<size_t>(nphi) + static_cast<size_t>(nz)) + 16;
            if (capacity > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            int maxSegments = static_cast<int>(capacity);

            vector<int> cellv(capacity, -1);
            vector<double> dsv(capacity, 0.);
            int count = 0;
            int status = 0;

            CUdeviceptr dRv = 0;
            CUdeviceptr dPhiv = 0;
            CUdeviceptr dZv = 0;
            CUdeviceptr dSinv = 0;
            CUdeviceptr dCosv = 0;
            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dCountv = 0;
            CUdeviceptr dStatusv = 0;

            auto freeAll = [&]() {
                if (dRv) _cuMemFree(dRv);
                if (dPhiv) _cuMemFree(dPhiv);
                if (dZv) _cuMemFree(dZv);
                if (dSinv) _cuMemFree(dSinv);
                if (dCosv) _cuMemFree(dCosv);
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dCountv) _cuMemFree(dCountv);
                if (dStatusv) _cuMemFree(dStatusv);
            };

            bool ok = allocateAndCopy(dRv, Rv, static_cast<size_t>(nR + 1) * sizeof(double))
                      && allocateAndCopy(dPhiv, phiv, static_cast<size_t>(nphi + 1) * sizeof(double))
                      && allocateAndCopy(dZv, zv, static_cast<size_t>(nz + 1) * sizeof(double))
                      && allocateAndCopy(dSinv, sinv, static_cast<size_t>(nphi + 1) * sizeof(double))
                      && allocateAndCopy(dCosv, cosv, static_cast<size_t>(nphi + 1) * sizeof(double))
                      && allocate(dCellv, cellv.size() * sizeof(int)) && allocate(dDsv, dsv.size() * sizeof(double))
                      && allocate(dCountv, sizeof(int)) && allocate(dStatusv, sizeof(int));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dRv,          &dPhiv,       &dZv,      &dSinv,    &dCosv,   &nR,
                            &nphi,         &nz,          &eps,      &hasHoleValue,
                            &rx,           &ry,          &rz,       &kx,       &ky,      &kz,
                            &maxDistance,  &maxSegments, &dCellv,   &dDsv,     &dCountv, &dStatusv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelCylinder3DGridPath, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                          "cuLaunchKernel(cylinder3d_grid_path)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(&count, dCountv, sizeof(int)), "cuMemcpyDtoH(cylinder3d path count)")
                 && check(_cuMemcpyDtoH(&status, dStatusv, sizeof(int)), "cuMemcpyDtoH(cylinder3d path status)");
            if (ok && status == 1 && count > 0)
            {
                ok = check(_cuMemcpyDtoH(cellv.data(), dCellv, static_cast<size_t>(count) * sizeof(int)),
                           "cuMemcpyDtoH(cylinder3d path cells)")
                     && check(_cuMemcpyDtoH(dsv.data(), dDsv, static_cast<size_t>(count) * sizeof(double)),
                              "cuMemcpyDtoH(cylinder3d path lengths)");
            }

            freeAll();
            if (!ok || status != 1 || count < 0 || count > maxSegments) return false;

            path->clear();
            for (int i = 0; i != count; ++i) path->addSegment(cellv[i], dsv[i]);
            return true;
        }

        bool traceTetraMeshGridPath(const void* gridKey, const vector<double>& vertexv,
                                    const vector<int>& tetraVertexv, const vector<int>& faceAnchorv,
                                    const vector<int>& faceNeighborv, const vector<double>& faceNormalv,
                                    const vector<double>& centroidv, int numCells, double eps,
                                    const Position& position, const Direction& direction, double xmin, double ymin,
                                    double zmin, double xmax, double ymax, double zmax, double maxDistance,
                                    SpatialGridPath* path)
        {
            if (!path || !gridKey || numCells <= 0 || vertexv.empty()
                || tetraVertexv.size() != 4 * static_cast<size_t>(numCells)
                || faceAnchorv.size() != 4 * static_cast<size_t>(numCells)
                || faceNeighborv.size() != 4 * static_cast<size_t>(numCells)
                || faceNormalv.size() != 12 * static_cast<size_t>(numCells)
                || centroidv.size() != 3 * static_cast<size_t>(numCells))
                return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;
            if (!ensureTetraOnDevice(gridKey, vertexv, tetraVertexv, faceAnchorv, faceNeighborv, faceNormalv, centroidv))
                return false;

            double rx, ry, rz;
            position.cartesian(rx, ry, rz);
            double kx, ky, kz;
            direction.cartesian(kx, ky, kz);

            size_t capacity = static_cast<size_t>(numCells) + 1;
            if (capacity > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            int maxSegments = static_cast<int>(capacity);

            vector<int> cellv(capacity, -1);
            vector<double> dsv(capacity, 0.);
            int count = 0;
            int status = 0;

            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dCountv = 0;
            CUdeviceptr dStatusv = 0;

            auto freeAll = [&]() {
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dCountv) _cuMemFree(dCountv);
                if (dStatusv) _cuMemFree(dStatusv);
            };

            bool ok = allocate(dCellv, cellv.size() * sizeof(int)) && allocate(dDsv, dsv.size() * sizeof(double))
                      && allocate(dCountv, sizeof(int)) && allocate(dStatusv, sizeof(int));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&_tetraVertexDevice, &_tetraFaceAnchorDevice, &_tetraFaceNeighborDevice,
                            &_tetraFaceNormalDevice, &_tetraCentroidDevice, &numCells, &eps,
                            &rx, &ry, &rz, &kx, &ky, &kz, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax,
                            &maxDistance, &maxSegments, &dCellv, &dDsv, &dCountv, &dStatusv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelTetraMeshGridPath, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                          "cuLaunchKernel(tetra_mesh_grid_path)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(&count, dCountv, sizeof(int)), "cuMemcpyDtoH(tetra path count)")
                 && check(_cuMemcpyDtoH(&status, dStatusv, sizeof(int)), "cuMemcpyDtoH(tetra path status)");
            if (ok && status == 1 && count > 0)
            {
                ok = check(_cuMemcpyDtoH(cellv.data(), dCellv, static_cast<size_t>(count) * sizeof(int)),
                           "cuMemcpyDtoH(tetra path cells)")
                     && check(_cuMemcpyDtoH(dsv.data(), dDsv, static_cast<size_t>(count) * sizeof(double)),
                              "cuMemcpyDtoH(tetra path lengths)");
            }

            freeAll();
            if (!ok || status != 1 || count < 0 || count > maxSegments) return false;

            path->clear();
            for (int i = 0; i != count; ++i) path->addSegment(cellv[i], dsv[i]);
            return true;
        }

        bool traceVoronoiMeshGridPath(const void* gridKey, const vector<double>& sitev,
                                      const vector<int>& neighborBeginv, const vector<int>& neighborCountv,
                                      const vector<int>& neighborIndexv, int numCells, double eps,
                                      const Position& position, const Direction& direction, double xmin, double ymin,
                                      double zmin, double xmax, double ymax, double zmax, double maxDistance,
                                      SpatialGridPath* path)
        {
            if (!path || !gridKey || numCells <= 0 || sitev.size() != 3 * static_cast<size_t>(numCells)
                || neighborBeginv.size() != static_cast<size_t>(numCells)
                || neighborCountv.size() != static_cast<size_t>(numCells))
                return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;
            static const vector<int> emptyBlockv;
            if (!ensureVoronoiOnDevice(gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv,
                                       emptyBlockv, emptyBlockv, emptyBlockv))
                return false;

            double rx, ry, rz;
            position.cartesian(rx, ry, rz);
            double kx, ky, kz;
            direction.cartesian(kx, ky, kz);

            size_t capacity = static_cast<size_t>(numCells) + 1;
            if (capacity > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            int maxSegments = static_cast<int>(capacity);

            vector<int> cellv(capacity, -1);
            vector<double> dsv(capacity, 0.);
            int count = 0;
            int status = 0;

            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dCountv = 0;
            CUdeviceptr dStatusv = 0;

            auto freeAll = [&]() {
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dCountv) _cuMemFree(dCountv);
                if (dStatusv) _cuMemFree(dStatusv);
            };

            bool ok = allocate(dCellv, cellv.size() * sizeof(int)) && allocate(dDsv, dsv.size() * sizeof(double))
                      && allocate(dCountv, sizeof(int)) && allocate(dStatusv, sizeof(int));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&_voronoiSiteDevice,          &_voronoiNeighborBeginDevice,
                            &_voronoiNeighborCountDevice, &_voronoiNeighborIndexDevice,
                            &numCells,                    &eps,
                            &rx,                          &ry,
                            &rz,                          &kx,
                            &ky,                          &kz,
                            &xmin,                        &ymin,
                            &zmin,                        &xmax,
                            &ymax,                        &zmax,
                            &maxDistance,                 &maxSegments,
                            &dCellv,                      &dDsv,
                            &dCountv,                     &dStatusv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelVoronoiMeshGridPath, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                          "cuLaunchKernel(voronoi_mesh_grid_path)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(&count, dCountv, sizeof(int)), "cuMemcpyDtoH(voronoi path count)")
                 && check(_cuMemcpyDtoH(&status, dStatusv, sizeof(int)), "cuMemcpyDtoH(voronoi path status)");
            if (ok && status == 1 && count > 0)
            {
                ok = check(_cuMemcpyDtoH(cellv.data(), dCellv, static_cast<size_t>(count) * sizeof(int)),
                           "cuMemcpyDtoH(voronoi path cells)")
                     && check(_cuMemcpyDtoH(dsv.data(), dDsv, static_cast<size_t>(count) * sizeof(double)),
                              "cuMemcpyDtoH(voronoi path lengths)");
            }

            freeAll();
            if (!ok || status != 1 || count < 0 || count > maxSegments) return false;

            path->clear();
            for (int i = 0; i != count; ++i) path->addSegment(cellv[i], dsv[i]);
            return true;
        }

	        bool traceVoronoiMeshGridPaths(const void* gridKey, const vector<double>& sitev,
	                                       const vector<int>& neighborBeginv, const vector<int>& neighborCountv,
	                                       const vector<int>& neighborIndexv, const vector<int>& blockBeginv,
	                                       const vector<int>& blockCountv, const vector<int>& blockIndexv,
	                                       int blockN, int numCells, double eps, double xmin, double ymin,
	                                       double zmin, double xmax, double ymax, double zmax, double maxDistance,
	                                       const vector<SpatialGridPath*>& paths)
	        {
	            if (!gridKey || paths.empty() || numCells <= 0 || sitev.size() != 3 * static_cast<size_t>(numCells)
	                || neighborBeginv.size() != static_cast<size_t>(numCells)
	                || neighborCountv.size() != static_cast<size_t>(numCells))
	                return false;
	            if (blockN > 0
	                && (blockBeginv.size() != static_cast<size_t>(blockN) * blockN * blockN
	                    || blockCountv.size() != blockBeginv.size()))
	                return false;
	            for (auto path : paths)
	                if (!path) return false;

	            size_t numPathsHost = paths.size();
	            if (numPathsHost > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
	            int numPaths = static_cast<int>(numPathsHost);

	            size_t capacity = static_cast<size_t>(numCells) + 1;
	            if (capacity > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
	            int maxSegments = static_cast<int>(capacity);

	            vector<double> positionv(3 * numPathsHost);
	            vector<double> directionv(3 * numPathsHost);
	            for (size_t p = 0; p != numPathsHost; ++p)
	            {
                double rx, ry, rz;
                paths[p]->position().cartesian(rx, ry, rz);
                double kx, ky, kz;
                paths[p]->direction().cartesian(kx, ky, kz);
                size_t base = 3 * p;
                positionv[base] = rx;
                positionv[base + 1] = ry;
                positionv[base + 2] = rz;
                directionv[base] = kx;
                directionv[base + 1] = ky;
                directionv[base + 2] = kz;
            }

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
	            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
	            if (!ensureModule()) return false;
	            if (!ensureVoronoiOnDevice(gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv,
	                                       blockBeginv, blockCountv, blockIndexv))
	                return false;

	            vector<int> countv(numPathsHost, 0);
	            vector<int> statusv(numPathsHost, 0);

	            CUdeviceptr dPositionv = 0;
	            CUdeviceptr dDirectionv = 0;
	            CUdeviceptr dPathOffsetv = 0;
	            CUdeviceptr dCellv = 0;
	            CUdeviceptr dDsv = 0;
	            CUdeviceptr dCountv = 0;
	            CUdeviceptr dStatusv = 0;

	            auto freeAll = [&]() {
	                if (dPositionv) _cuMemFree(dPositionv);
	                if (dDirectionv) _cuMemFree(dDirectionv);
	                if (dPathOffsetv) _cuMemFree(dPathOffsetv);
	                if (dCellv) _cuMemFree(dCellv);
	                if (dDsv) _cuMemFree(dDsv);
	                if (dCountv) _cuMemFree(dCountv);
	                if (dStatusv) _cuMemFree(dStatusv);
	            };

	            bool ok = allocateAndCopy(dPositionv, positionv.data(), positionv.size() * sizeof(double))
	                      && allocateAndCopy(dDirectionv, directionv.data(), directionv.size() * sizeof(double))
	                      && allocate(dCountv, countv.size() * sizeof(int))
	                      && allocate(dStatusv, statusv.size() * sizeof(int));
	            if (!ok)
	            {
	                freeAll();
                return false;
            }

	            unsigned int blockSize = 128;
	            unsigned int gridSize = static_cast<unsigned int>((numPathsHost + blockSize - 1) / blockSize);
	            void* args[] = {&_voronoiSiteDevice,          &_voronoiNeighborBeginDevice,
	                            &_voronoiNeighborCountDevice, &_voronoiNeighborIndexDevice,
	                            &_voronoiBlockBeginDevice,    &_voronoiBlockCountDevice,
	                            &_voronoiBlockIndexDevice,    &blockN,
	                            &numCells,                    &eps,
	                            &dPositionv,                  &dDirectionv,
	                            &numPaths,                    &xmin,
	                            &ymin,                        &zmin,
	                            &xmax,                        &ymax,
	                            &zmax,                        &maxDistance,
	                            &maxSegments,                 &dCountv,
	                            &dStatusv};
	            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
	                 && check(_cuLaunchKernel(_kernelVoronoiMeshGridPathsCount, gridSize, 1, 1, blockSize, 1, 1, 0,
	                                          nullptr, args, nullptr),
	                          "cuLaunchKernel(voronoi_mesh_grid_paths_count)")
	                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
	                 && check(_cuMemcpyDtoH(countv.data(), dCountv, countv.size() * sizeof(int)),
	                          "cuMemcpyDtoH(voronoi compact path counts)")
	                 && check(_cuMemcpyDtoH(statusv.data(), dStatusv, statusv.size() * sizeof(int)),
	                          "cuMemcpyDtoH(voronoi compact path status)");
	            if (!ok)
	            {
	                freeAll();
	                return false;
	            }
	            for (size_t p = 0; p != numPathsHost; ++p)
	            {
	                if (statusv[p] != 1 || countv[p] < 0 || countv[p] > maxSegments)
	                {
	                    _error = "voronoi compact path count kernel failed for path " + std::to_string(p);
	                    freeAll();
	                    return false;
	                }
	            }

	            vector<int> pathOffsetv(numPathsHost + 1, 0);
	            size_t totalSegments = 0;
	            for (size_t p = 0; p != numPathsHost; ++p)
	            {
	                if (totalSegments > static_cast<size_t>(std::numeric_limits<int>::max()))
	                {
	                    freeAll();
	                    return false;
	                }
	                pathOffsetv[p] = static_cast<int>(totalSegments);
	                totalSegments += static_cast<size_t>(countv[p]);
	            }
	            if (totalSegments > static_cast<size_t>(std::numeric_limits<int>::max()))
	            {
	                freeAll();
	                return false;
	            }
	            pathOffsetv[numPathsHost] = static_cast<int>(totalSegments);

	            if (totalSegments == 0)
	            {
	                for (SpatialGridPath* path : paths) path->clear();
	                freeAll();
	                return true;
	            }

	            vector<int> cellv(totalSegments, -1);
	            vector<double> dsv(totalSegments, 0.);
	            ok = allocateAndCopy(dPathOffsetv, pathOffsetv.data(), pathOffsetv.size() * sizeof(int))
	                 && allocate(dCellv, cellv.size() * sizeof(int)) && allocate(dDsv, dsv.size() * sizeof(double));
	            if (!ok)
	            {
	                freeAll();
	                return false;
	            }

	            void* compactArgs[] = {&_voronoiSiteDevice,          &_voronoiNeighborBeginDevice,
	                                   &_voronoiNeighborCountDevice, &_voronoiNeighborIndexDevice,
	                                   &_voronoiBlockBeginDevice,    &_voronoiBlockCountDevice,
	                                   &_voronoiBlockIndexDevice,    &blockN,
	                                   &numCells,                    &eps,
	                                   &dPositionv,                  &dDirectionv,
	                                   &numPaths,                    &dPathOffsetv,
	                                   &xmin,                        &ymin,
	                                   &zmin,                        &xmax,
	                                   &ymax,                        &zmax,
	                                   &maxDistance,                 &dCellv,
	                                   &dDsv,                        &dCountv,
	                                   &dStatusv};
	            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
	                 && check(_cuLaunchKernel(_kernelVoronoiMeshGridPathsCompact, gridSize, 1, 1, blockSize, 1, 1, 0,
	                                          nullptr, compactArgs, nullptr),
	                          "cuLaunchKernel(voronoi_mesh_grid_paths_compact)")
	                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
	                 && check(_cuMemcpyDtoH(countv.data(), dCountv, countv.size() * sizeof(int)),
	                          "cuMemcpyDtoH(voronoi compact output counts)")
	                 && check(_cuMemcpyDtoH(statusv.data(), dStatusv, statusv.size() * sizeof(int)),
	                          "cuMemcpyDtoH(voronoi compact output status)")
	                 && check(_cuMemcpyDtoH(cellv.data(), dCellv, cellv.size() * sizeof(int)),
	                          "cuMemcpyDtoH(voronoi compact path cells)")
	                 && check(_cuMemcpyDtoH(dsv.data(), dDsv, dsv.size() * sizeof(double)),
	                          "cuMemcpyDtoH(voronoi compact path lengths)");
	            if (ok)
	            {
	                for (size_t p = 0; p != numPathsHost; ++p)
	                {
	                    int expectedCount = pathOffsetv[p + 1] - pathOffsetv[p];
	                    if (statusv[p] != 1 || countv[p] != expectedCount)
	                    {
	                        ok = false;
	                        _error = "voronoi compact path output kernel failed for path " + std::to_string(p);
	                        break;
	                    }
	                }
	            }

	            freeAll();
	            if (!ok) return false;

	            for (size_t p = 0; p != numPathsHost; ++p)
	            {
	                SpatialGridPath* path = paths[p];
	                path->clear();
	                int begin = pathOffsetv[p];
	                int end = pathOffsetv[p + 1];
	                for (int i = begin; i != end; ++i) path->addSegment(cellv[i], dsv[i]);
	            }
	            return true;
	        }

        bool traceVoronoiMeshGridPathsTableOpticalDepths(
            const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
            const vector<int>& neighborCountv, const vector<int>& neighborIndexv, const vector<int>& blockBeginv,
            const vector<int>& blockCountv, const vector<int>& blockIndexv, int blockN, int numCells, double eps,
            double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, double maxDistance,
            const vector<SpatialGridPath*>& paths, const MediumState& state, const vector<int>& mediaIndexv,
            const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
            const vector<double>& lookupWavelengthv, const vector<double>& section1Tablev,
            const vector<double>* section2Tablev, const vector<double>& lambdav)
        {
            if (!gridKey || paths.empty() || lambdav.size() != paths.size() || numCells <= 0
                || sitev.size() != 3 * static_cast<size_t>(numCells)
                || neighborBeginv.size() != static_cast<size_t>(numCells)
                || neighborCountv.size() != static_cast<size_t>(numCells))
                return false;
            if (mediaIndexv.empty() || lookupBeginv.size() != mediaIndexv.size()
                || lookupCountv.size() != mediaIndexv.size() || lookupWavelengthv.empty()
                || section1Tablev.size() != lookupWavelengthv.size()
                || (section2Tablev && section2Tablev->size() != lookupWavelengthv.size()))
                return false;
            if (blockN > 0
                && (blockBeginv.size() != static_cast<size_t>(blockN) * blockN * blockN
                    || blockCountv.size() != blockBeginv.size()))
                return false;
            if (paths.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            for (auto path : paths)
                if (!path) return false;

            for (int m = 0; m != numCells; ++m)
            {
                int begin = neighborBeginv[m];
                int count = neighborCountv[m];
                if (begin < 0 || count < 0) return false;
                if (static_cast<size_t>(begin) + static_cast<size_t>(count) > neighborIndexv.size()) return false;
                for (int i = 0; i != count; ++i)
                {
                    int neighbor = neighborIndexv[begin + i];
                    if (neighbor >= numCells || neighbor < -6) return false;
                }
            }
            for (size_t b = 0; b != blockBeginv.size(); ++b)
            {
                int begin = blockBeginv[b];
                int count = blockCountv[b];
                if (begin < 0 || count < 0) return false;
                if (static_cast<size_t>(begin) + static_cast<size_t>(count) > blockIndexv.size()) return false;
                for (int i = 0; i != count; ++i)
                {
                    int cell = blockIndexv[begin + i];
                    if (cell < 0 || cell >= numCells) return false;
                }
            }
            for (size_t h = 0; h != mediaIndexv.size(); ++h)
            {
                if (mediaIndexv[h] < 0 || lookupBeginv[h] < 0 || lookupCountv[h] < 2) return false;
                size_t begin = static_cast<size_t>(lookupBeginv[h]);
                size_t count = static_cast<size_t>(lookupCountv[h]);
                if (begin + count > lookupWavelengthv.size()) return false;
            }

            size_t numPathsHost = paths.size();
            int numPaths = static_cast<int>(numPathsHost);

            size_t capacity = static_cast<size_t>(numCells) + 1;
            if (capacity > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            int maxSegments = static_cast<int>(capacity);

            vector<double> positionv(3 * numPathsHost);
            vector<double> directionv(3 * numPathsHost);
            for (size_t p = 0; p != numPathsHost; ++p)
            {
                double rx, ry, rz;
                paths[p]->position().cartesian(rx, ry, rz);
                double kx, ky, kz;
                paths[p]->direction().cartesian(kx, ky, kz);
                size_t base = 3 * p;
                positionv[base] = rx;
                positionv[base + 1] = ry;
                positionv[base + 2] = rz;
                directionv[base] = kx;
                directionv[base + 1] = ky;
                directionv[base + 2] = kz;
            }

            vector<int> densityOffsetv(mediaIndexv.size());
            for (size_t h = 0; h != mediaIndexv.size(); ++h)
            {
                int offset = state.numberDensityOffset(mediaIndexv[h]);
                if (offset < 0) return false;
                densityOffsetv[h] = offset;
            }

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;
            if (!ensureVoronoiOnDevice(gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv,
                                       blockCountv, blockIndexv))
                return false;

            vector<int> countv(numPathsHost, 0);
            vector<int> statusv(numPathsHost, 0);

            CUdeviceptr dPositionv = 0;
            CUdeviceptr dDirectionv = 0;
            CUdeviceptr dPathOffsetv = 0;
            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dCountv = 0;
            CUdeviceptr dStatusv = 0;
            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dLookupBeginv = 0;
            CUdeviceptr dLookupCountv = 0;
            CUdeviceptr dLookupWavelengthv = 0;
            CUdeviceptr dSection1Tablev = 0;
            CUdeviceptr dSection2Tablev = 0;
            CUdeviceptr dLambdav = 0;
            CUdeviceptr dOut1v = 0;
            CUdeviceptr dOut2v = 0;

            auto freeAll = [&]() {
                if (dPositionv) _cuMemFree(dPositionv);
                if (dDirectionv) _cuMemFree(dDirectionv);
                if (dPathOffsetv) _cuMemFree(dPathOffsetv);
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dCountv) _cuMemFree(dCountv);
                if (dStatusv) _cuMemFree(dStatusv);
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dLookupBeginv) _cuMemFree(dLookupBeginv);
                if (dLookupCountv) _cuMemFree(dLookupCountv);
                if (dLookupWavelengthv) _cuMemFree(dLookupWavelengthv);
                if (dSection1Tablev) _cuMemFree(dSection1Tablev);
                if (dSection2Tablev) _cuMemFree(dSection2Tablev);
                if (dLambdav) _cuMemFree(dLambdav);
                if (dOut1v) _cuMemFree(dOut1v);
                if (dOut2v) _cuMemFree(dOut2v);
            };

            bool ok = allocateAndCopy(dPositionv, positionv.data(), positionv.size() * sizeof(double))
                      && allocateAndCopy(dDirectionv, directionv.data(), directionv.size() * sizeof(double))
                      && allocate(dCountv, countv.size() * sizeof(int))
                      && allocate(dStatusv, statusv.size() * sizeof(int));
            if (!ok)
            {
                freeAll();
                return false;
            }

            unsigned int blockSize = 128;
            unsigned int gridSize = static_cast<unsigned int>((numPathsHost + blockSize - 1) / blockSize);
            void* countArgs[] = {&_voronoiSiteDevice,          &_voronoiNeighborBeginDevice,
                                 &_voronoiNeighborCountDevice, &_voronoiNeighborIndexDevice,
                                 &_voronoiBlockBeginDevice,    &_voronoiBlockCountDevice,
                                 &_voronoiBlockIndexDevice,    &blockN,
                                 &numCells,                    &eps,
                                 &dPositionv,                  &dDirectionv,
                                 &numPaths,                    &xmin,
                                 &ymin,                        &zmin,
                                 &xmax,                        &ymax,
                                 &zmax,                        &maxDistance,
                                 &maxSegments,                 &dCountv,
                                 &dStatusv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelVoronoiMeshGridPathsCount, gridSize, 1, 1, blockSize, 1, 1, 0,
                                          nullptr, countArgs, nullptr),
                          "cuLaunchKernel(voronoi_mesh_grid_paths_count)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(countv.data(), dCountv, countv.size() * sizeof(int)),
                          "cuMemcpyDtoH(voronoi table path counts)")
                 && check(_cuMemcpyDtoH(statusv.data(), dStatusv, statusv.size() * sizeof(int)),
                          "cuMemcpyDtoH(voronoi table path status)");
            if (!ok)
            {
                freeAll();
                return false;
            }
            for (size_t p = 0; p != numPathsHost; ++p)
            {
                if (statusv[p] != 1 || countv[p] < 0 || countv[p] > maxSegments)
                {
                    _error = "voronoi table path count kernel failed for path " + std::to_string(p);
                    freeAll();
                    return false;
                }
            }

            vector<int> pathOffsetv(numPathsHost + 1, 0);
            size_t totalSegments = 0;
            for (size_t p = 0; p != numPathsHost; ++p)
            {
                if (totalSegments > static_cast<size_t>(std::numeric_limits<int>::max()))
                {
                    freeAll();
                    return false;
                }
                pathOffsetv[p] = static_cast<int>(totalSegments);
                totalSegments += static_cast<size_t>(countv[p]);
            }
            if (totalSegments > static_cast<size_t>(std::numeric_limits<int>::max()))
            {
                freeAll();
                return false;
            }
            pathOffsetv[numPathsHost] = static_cast<int>(totalSegments);

            if (totalSegments == 0)
            {
                for (SpatialGridPath* path : paths) path->clear();
                freeAll();
                return true;
            }

            vector<int> cellv(totalSegments, -1);
            vector<double> dsv(totalSegments, 0.);
            vector<double> out1v(totalSegments, 0.);
            vector<double> out2v(section2Tablev ? totalSegments : 0, 0.);
            ok = allocateAndCopy(dPathOffsetv, pathOffsetv.data(), pathOffsetv.size() * sizeof(int))
                 && allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                 && allocateAndCopy(dLookupBeginv, lookupBeginv.data(), lookupBeginv.size() * sizeof(int))
                 && allocateAndCopy(dLookupCountv, lookupCountv.data(), lookupCountv.size() * sizeof(int))
                 && allocateAndCopy(dLookupWavelengthv, lookupWavelengthv.data(),
                                    lookupWavelengthv.size() * sizeof(double))
                 && allocateAndCopy(dSection1Tablev, section1Tablev.data(), section1Tablev.size() * sizeof(double))
                 && allocateAndCopy(dLambdav, lambdav.data(), lambdav.size() * sizeof(double))
                 && allocate(dCellv, cellv.size() * sizeof(int)) && allocate(dDsv, dsv.size() * sizeof(double))
                 && allocate(dOut1v, out1v.size() * sizeof(double));
            if (section2Tablev)
                ok = ok
                     && allocateAndCopy(dSection2Tablev, section2Tablev->data(),
                                        section2Tablev->size() * sizeof(double))
                     && allocate(dOut2v, out2v.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            int numVars = state.numVars();
            int numTables = static_cast<int>(mediaIndexv.size());
            int hasSecond = section2Tablev ? 1 : 0;
            void* compactArgs[] = {&_voronoiSiteDevice,          &_voronoiNeighborBeginDevice,
                                   &_voronoiNeighborCountDevice, &_voronoiNeighborIndexDevice,
                                   &_voronoiBlockBeginDevice,    &_voronoiBlockCountDevice,
                                   &_voronoiBlockIndexDevice,    &blockN,
                                   &numCells,                    &eps,
                                   &dPositionv,                  &dDirectionv,
                                   &numPaths,                    &dPathOffsetv,
                                   &xmin,                        &ymin,
                                   &zmin,                        &xmax,
                                   &ymax,                        &zmax,
                                   &maxDistance,                 &_stateDevice,
                                   &numVars,                     &numTables,
                                   &dDensityOffsetv,             &dLookupBeginv,
                                   &dLookupCountv,               &dLookupWavelengthv,
                                   &dSection1Tablev,             &dSection2Tablev,
                                   &hasSecond,                   &dLambdav,
                                   &dCellv,                      &dDsv,
                                   &dOut1v,                      &dOut2v,
                                   &dCountv,                     &dStatusv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelVoronoiTableSectionOpticalDepthsCompact, gridSize, 1, 1,
                                          blockSize, 1, 1, 0, nullptr, compactArgs, nullptr),
                          "cuLaunchKernel(voronoi_table_section_optical_depths_compact)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(countv.data(), dCountv, countv.size() * sizeof(int)),
                          "cuMemcpyDtoH(voronoi table output counts)")
                 && check(_cuMemcpyDtoH(statusv.data(), dStatusv, statusv.size() * sizeof(int)),
                          "cuMemcpyDtoH(voronoi table output status)")
                 && check(_cuMemcpyDtoH(cellv.data(), dCellv, cellv.size() * sizeof(int)),
                          "cuMemcpyDtoH(voronoi table path cells)")
                 && check(_cuMemcpyDtoH(dsv.data(), dDsv, dsv.size() * sizeof(double)),
                          "cuMemcpyDtoH(voronoi table path lengths)")
                 && check(_cuMemcpyDtoH(out1v.data(), dOut1v, out1v.size() * sizeof(double)),
                          "cuMemcpyDtoH(voronoi table optical depth)");
            if (ok && section2Tablev)
                ok = check(_cuMemcpyDtoH(out2v.data(), dOut2v, out2v.size() * sizeof(double)),
                           "cuMemcpyDtoH(voronoi table absorption optical depth)");
            if (ok)
            {
                for (size_t p = 0; p != numPathsHost; ++p)
                {
                    int expectedCount = pathOffsetv[p + 1] - pathOffsetv[p];
                    if (statusv[p] != 1 || countv[p] != expectedCount)
                    {
                        ok = false;
                        _error = "voronoi table output kernel failed for path " + std::to_string(p);
                        break;
                    }
                }
            }

            freeAll();
            if (!ok) return false;

            for (size_t p = 0; p != numPathsHost; ++p)
            {
                SpatialGridPath* path = paths[p];
                path->clear();
                int begin = pathOffsetv[p];
                int end = pathOffsetv[p + 1];
                for (int i = begin; i != end; ++i)
                {
                    path->addSegment(cellv[i], dsv[i]);
                    if (section2Tablev)
                        path->segments().back().setOpticalDepth(out1v[i], out2v[i]);
                    else
                        path->segments().back().setOpticalDepth(out1v[i]);
                }
            }
            return true;
        }

        bool voronoiTableRadiationFieldSumsAndForcedPropagationResults(
            const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
            const vector<int>& neighborCountv, const vector<int>& neighborIndexv, const vector<int>& blockBeginv,
            const vector<int>& blockCountv, const vector<int>& blockIndexv, int blockN, int numCells, double eps,
            double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, double maxDistance,
            const vector<Position>& positions, const vector<Direction>& directions, const MediumState& state,
            const vector<double>& luminosityv, const vector<int>& wavelengthBinv, int numWavelengths,
            const vector<double>& tauinteractv, const vector<double>& pathBiasWeightv, const vector<double>& lambdav,
            const vector<double>* randomSelectv, const vector<double>* randomSamplev, double pathLengthBias,
            const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
            const vector<double>& lookupWavelengthv, const vector<double>& sectionScaTablev,
            const vector<double>& sectionExtTablev, vector<int>& binIndexv, vector<double>& Ldsv,
            vector<int>& cellOutv, vector<double>& distanceOutv, vector<double>& tauAbsOutv,
            vector<double>& weightOutv, const vector<double>* scatterRandomCosthetav,
            const vector<double>* scatterRandomPhiv, int hgLookupBegin, int hgLookupCount,
            const vector<double>* hgAsymmparv, vector<double>* scatterDirectionOutv,
            const void* accumulatorKey = nullptr, size_t numAccumulatorValues = 0)
        {
            bool sampleOnGpu = randomSelectv || randomSamplev;
            bool scatterOnGpu = scatterRandomCosthetav || scatterRandomPhiv || hgAsymmparv || scatterDirectionOutv;
            bool accumulateOnGpu = accumulatorKey && numAccumulatorValues > 0;
            if (!gridKey || positions.empty() || directions.size() != positions.size()
                || luminosityv.size() != positions.size() || wavelengthBinv.size() != positions.size()
                || lambdav.size() != positions.size() || numWavelengths <= 0 || numCells <= 0
                || sitev.size() != 3 * static_cast<size_t>(numCells)
                || neighborBeginv.size() != static_cast<size_t>(numCells)
                || neighborCountv.size() != static_cast<size_t>(numCells))
                return false;
            if (sampleOnGpu)
            {
                if (!randomSelectv || !randomSamplev || randomSelectv->size() != positions.size()
                    || randomSamplev->size() != positions.size() || pathLengthBias < 0.0 || pathLengthBias > 1.0)
                    return false;
            }
            else if (tauinteractv.size() != positions.size() || pathBiasWeightv.size() != positions.size())
                return false;
            if (scatterOnGpu)
            {
                if (!scatterRandomCosthetav || !scatterRandomPhiv || !hgAsymmparv || !scatterDirectionOutv
                    || scatterRandomCosthetav->size() != positions.size()
                    || scatterRandomPhiv->size() != positions.size() || hgLookupBegin < 0 || hgLookupCount < 2)
                    return false;
            }
            if (mediaIndexv.empty() || lookupBeginv.size() != mediaIndexv.size()
                || lookupCountv.size() != mediaIndexv.size() || lookupWavelengthv.empty()
                || sectionScaTablev.size() != lookupWavelengthv.size()
                || sectionExtTablev.size() != lookupWavelengthv.size())
                return false;
            if (blockN > 0
                && (blockBeginv.size() != static_cast<size_t>(blockN) * blockN * blockN
                    || blockCountv.size() != blockBeginv.size()))
                return false;
            if (positions.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            if (accumulateOnGpu && numAccumulatorValues > static_cast<size_t>(std::numeric_limits<int>::max()))
                return false;

            int maxCellForKey = (std::numeric_limits<int>::max() - (numWavelengths - 1)) / numWavelengths;
            if (numCells - 1 > maxCellForKey) return false;
            for (int m = 0; m != numCells; ++m)
            {
                int begin = neighborBeginv[m];
                int count = neighborCountv[m];
                if (begin < 0 || count < 0) return false;
                if (static_cast<size_t>(begin) + static_cast<size_t>(count) > neighborIndexv.size()) return false;
                for (int i = 0; i != count; ++i)
                {
                    int neighbor = neighborIndexv[begin + i];
                    if (neighbor >= numCells || neighbor < -6) return false;
                }
            }
            for (size_t b = 0; b != blockBeginv.size(); ++b)
            {
                int begin = blockBeginv[b];
                int count = blockCountv[b];
                if (begin < 0 || count < 0) return false;
                if (static_cast<size_t>(begin) + static_cast<size_t>(count) > blockIndexv.size()) return false;
                for (int i = 0; i != count; ++i)
                {
                    int cell = blockIndexv[begin + i];
                    if (cell < 0 || cell >= numCells) return false;
                }
            }
            for (size_t h = 0; h != mediaIndexv.size(); ++h)
            {
                if (mediaIndexv[h] < 0 || lookupBeginv[h] < 0 || lookupCountv[h] < 2) return false;
                size_t begin = static_cast<size_t>(lookupBeginv[h]);
                size_t count = static_cast<size_t>(lookupCountv[h]);
                if (begin + count > lookupWavelengthv.size()) return false;
            }
            if (scatterOnGpu
                && (static_cast<size_t>(hgLookupBegin) + static_cast<size_t>(hgLookupCount) > lookupWavelengthv.size()
                    || hgAsymmparv->size() != lookupWavelengthv.size()))
                return false;
            for (int ell : wavelengthBinv)
                if (ell < -1 || ell >= numWavelengths) return false;

            size_t numPathsHost = positions.size();
            int numPaths = static_cast<int>(numPathsHost);
            size_t capacity = static_cast<size_t>(numCells) + 1;
            if (capacity > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            int maxSegments = static_cast<int>(capacity);

            vector<double> positionv(3 * numPathsHost);
            vector<double> directionv(3 * numPathsHost);
            for (size_t p = 0; p != numPathsHost; ++p)
            {
                double rx, ry, rz;
                positions[p].cartesian(rx, ry, rz);
                double kx, ky, kz;
                directions[p].cartesian(kx, ky, kz);
                size_t base = 3 * p;
                positionv[base] = rx;
                positionv[base + 1] = ry;
                positionv[base + 2] = rz;
                directionv[base] = kx;
                directionv[base + 1] = ky;
                directionv[base + 2] = kz;
            }

            vector<int> densityOffsetv(mediaIndexv.size());
            for (size_t h = 0; h != mediaIndexv.size(); ++h)
            {
                int offset = state.numberDensityOffset(mediaIndexv[h]);
                if (offset < 0) return false;
                densityOffsetv[h] = offset;
            }

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;
            if (!ensureVoronoiOnDevice(gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv,
                                       blockCountv, blockIndexv))
                return false;

	            CUdeviceptr dPositionv = 0;
	            CUdeviceptr dDirectionv = 0;
	            CUdeviceptr dPathOffsetv = 0;
	            CUdeviceptr dPathScanv = 0;
	            CUdeviceptr dPathIndexv = 0;
            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dSv = 0;
            CUdeviceptr dTauv = 0;
            CUdeviceptr dPathCountv = 0;
            CUdeviceptr dPathStatusv = 0;
            CUdeviceptr dLuminosityv = 0;
            CUdeviceptr dWavelengthBinv = 0;
            CUdeviceptr dTauinteractv = 0;
            CUdeviceptr dPathBiasWeightv = 0;
            CUdeviceptr dRandomSelectv = 0;
            CUdeviceptr dRandomSamplev = 0;
            CUdeviceptr dScatterRandomCosthetav = 0;
            CUdeviceptr dScatterRandomPhiv = 0;
            CUdeviceptr dScatterAsymmparv = 0;
            CUdeviceptr dScatterDirectionOutv = 0;
            CUdeviceptr dLambdav = 0;
            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dLookupBeginv = 0;
            CUdeviceptr dLookupCountv = 0;
            CUdeviceptr dLookupWavelengthv = 0;
            CUdeviceptr dSectionScaTablev = 0;
            CUdeviceptr dSectionExtTablev = 0;
            CUdeviceptr dNullTablev = 0;
            CUdeviceptr dKeyv = 0;
            CUdeviceptr dSumv = 0;
            CUdeviceptr dCompactCountv = 0;
            CUdeviceptr dCompactKeyv = 0;
            CUdeviceptr dCompactSumv = 0;
            CUdeviceptr dAccumulator = 0;
            CUdeviceptr dCellOutv = 0;
            CUdeviceptr dDistanceOutv = 0;
            CUdeviceptr dTauAbsOutv = 0;
            CUdeviceptr dWeightOutv = 0;

	            auto freeAll = [&]() {
	                // Resident forced/store buffers are per-device scratch buffers reused across photon waves.
	            };

	            size_t totalSegments = 0;
	            size_t hashCapacity = 0;
	            int compactCount = 0;
	            int pathOutputOk = 0;
	            size_t residentProfileIndex = 0;
	            bool emitResidentProfile = false;
	            if (residentProfileEnabled())
	            {
	                residentProfileIndex = _residentProfileCount.fetch_add(1, std::memory_order_relaxed);
	                emitResidentProfile = residentProfileIndex < residentProfileLimit();
	            }
	            auto residentProfileStart = std::chrono::steady_clock::now();
	            auto residentProfileMark = residentProfileStart;
	            double residentInputMs = 0.;
	            double residentCountMs = 0.;
	            double residentUploadMs = 0.;
	            double residentKernelMs = 0.;
	            double residentOutputMs = 0.;
	            auto captureResidentProfile = [&](double& bucket) {
	                if (!emitResidentProfile) return;
	                auto now = std::chrono::steady_clock::now();
	                bucket += millisecondsBetween(residentProfileMark, now);
	                residentProfileMark = now;
	            };
	            auto printResidentProfile = [&]() {
	                if (!emitResidentProfile) return;
	                double totalMs = millisecondsBetween(residentProfileStart, std::chrono::steady_clock::now());
	                std::lock_guard<std::mutex> profileLock(_residentProfileMutex);
	                std::cerr << "SKIRTGPU resident profile"
	                          << " call=" << residentProfileIndex
	                          << " device=" << _deviceSlot
	                          << " paths=" << numPathsHost
	                          << " segments=" << totalSegments
	                          << " compact=" << compactCount
	                          << " hash_capacity=" << hashCapacity
	                          << " sample=" << (sampleOnGpu ? "yes" : "no")
	                          << " input_ms=" << residentInputMs
	                          << " count_ms=" << residentCountMs
	                          << " upload_ms=" << residentUploadMs
	                          << " kernel_ms=" << residentKernelMs
	                          << " output_ms=" << residentOutputMs
	                          << " total_ms=" << totalMs << "\n";
	            };

	            bool ok = copyResidentScratch(ResidentPosition, dPositionv, positionv.data(),
	                                          positionv.size() * sizeof(double), "cuMemcpyHtoD(resident positions)")
	                      && copyResidentScratch(ResidentDirection, dDirectionv, directionv.data(),
	                                             directionv.size() * sizeof(double),
	                                             "cuMemcpyHtoD(resident directions)")
	                      && ensureResidentScratch(ResidentPathOffset, dPathOffsetv,
	                                               (numPathsHost + 1) * sizeof(int),
	                                               "cuMemAlloc(resident path offsets)")
	                      && ensureResidentScratch(ResidentPathScan, dPathScanv, 2 * sizeof(int),
	                                               "cuMemAlloc(resident path scan)")
	                      && ensureResidentScratch(ResidentPathCount, dPathCountv, numPathsHost * sizeof(int),
	                                               "cuMemAlloc(resident path counts)")
	                      && ensureResidentScratch(ResidentPathStatus, dPathStatusv, numPathsHost * sizeof(int),
	                                               "cuMemAlloc(resident path status)");
	            captureResidentProfile(residentInputMs);
	            if (!ok)
	            {
	                freeAll();
	                return false;
            }

            unsigned int pathBlockSize = 128;
            unsigned int pathGridSize = static_cast<unsigned int>((numPathsHost + pathBlockSize - 1) / pathBlockSize);
            void* countArgs[] = {&_voronoiSiteDevice,          &_voronoiNeighborBeginDevice,
                                 &_voronoiNeighborCountDevice, &_voronoiNeighborIndexDevice,
                                 &_voronoiBlockBeginDevice,    &_voronoiBlockCountDevice,
                                 &_voronoiBlockIndexDevice,    &blockN,
                                 &numCells,                    &eps,
                                 &dPositionv,                  &dDirectionv,
	                                 &numPaths,                    &xmin,
	                                 &ymin,                        &zmin,
	                                 &xmax,                        &ymax,
	                                 &zmax,                        &maxDistance,
	                                 &maxSegments,                 &dPathCountv,
	                                 &dPathStatusv};
	            int scanSummary[2] = {0, 0};
	            void* scanArgs[] = {&dPathCountv, &dPathStatusv, &numPaths, &maxSegments, &dPathOffsetv, &dPathScanv};
	            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
	                 && check(_cuLaunchKernel(_kernelVoronoiMeshGridPathsCount, pathGridSize, 1, 1, pathBlockSize, 1, 1,
	                                          0, nullptr, countArgs, nullptr),
	                          "cuLaunchKernel(voronoi resident path counts)")
	                 && check(_cuLaunchKernel(_kernelPathCountOffsets, 1, 1, 1, 1, 1, 1, 0, nullptr, scanArgs,
	                                          nullptr),
	                          "cuLaunchKernel(path_count_offsets)")
	                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
	                 && check(_cuMemcpyDtoH(scanSummary, dPathScanv, sizeof(scanSummary)),
	                          "cuMemcpyDtoH(voronoi resident path scan)");
	            captureResidentProfile(residentCountMs);
	            totalSegments = scanSummary[0] > 0 ? static_cast<size_t>(scanSummary[0]) : 0;
	            if (!ok || scanSummary[1] != 1
	                || totalSegments > static_cast<size_t>(std::numeric_limits<int>::max()) / 2)
	            {
	                if (ok && scanSummary[1] != 1) _error = "voronoi resident path count kernel failed";
	                freeAll();
	                return false;
	            }

	            binIndexv.clear();
            Ldsv.clear();
            cellOutv.assign(numPathsHost, -1);
            distanceOutv.assign(numPathsHost, 0.);
            tauAbsOutv.assign(numPathsHost, 0.);
            weightOutv.assign(numPathsHost, 0.);
            if (scatterOnGpu) scatterDirectionOutv->assign(3 * numPathsHost, 0.);
		    if (totalSegments == 0)
		    {
	                printResidentProfile();
	                freeAll();
	                return true;
	            }

	            hashCapacity = nextPowerOfTwo(totalSegments * 2);
            if (hashCapacity > static_cast<size_t>(std::numeric_limits<int>::max()))
            {
                freeAll();
                return false;
            }
            int hashCapacityInt = static_cast<int>(hashCapacity);
            int numSegments = static_cast<int>(totalSegments);
            int numTables = static_cast<int>(mediaIndexv.size());
            int numVars = state.numVars();
            int hasSecond = 0;

	            ok = ensureResidentScratch(ResidentPathIndex, dPathIndexv, totalSegments * sizeof(int),
	                                          "cuMemAlloc(resident path indices)")
                 && ensureResidentScratch(ResidentCell, dCellv, totalSegments * sizeof(int),
                                          "cuMemAlloc(resident cells)")
                 && ensureResidentScratch(ResidentDs, dDsv, totalSegments * sizeof(double),
                                          "cuMemAlloc(resident segment lengths)")
                 && ensureResidentScratch(ResidentDistance, dSv, totalSegments * sizeof(double),
                                          "cuMemAlloc(resident segment distances)")
                 && ensureResidentScratch(ResidentTau, dTauv, totalSegments * sizeof(double),
                                          "cuMemAlloc(resident segment optical depths)")
                 && copyResidentScratch(ResidentLuminosity, dLuminosityv, luminosityv.data(),
                                        luminosityv.size() * sizeof(double),
                                        "cuMemcpyHtoD(resident luminosities)")
                 && copyResidentScratch(ResidentWavelengthBin, dWavelengthBinv, wavelengthBinv.data(),
                                        wavelengthBinv.size() * sizeof(int),
                                        "cuMemcpyHtoD(resident wavelength bins)")
                 && copyResidentScratch(ResidentLambda, dLambdav, lambdav.data(), lambdav.size() * sizeof(double),
                                        "cuMemcpyHtoD(resident wavelengths)")
                 && copyResidentScratch(ResidentDensityOffset, dDensityOffsetv, densityOffsetv.data(),
                                        densityOffsetv.size() * sizeof(int),
                                        "cuMemcpyHtoD(resident density offsets)")
                 && copyResidentScratch(ResidentLookupBegin, dLookupBeginv, lookupBeginv.data(),
                                        lookupBeginv.size() * sizeof(int),
                                        "cuMemcpyHtoD(resident lookup begins)")
                 && copyResidentScratch(ResidentLookupCount, dLookupCountv, lookupCountv.data(),
                                        lookupCountv.size() * sizeof(int),
                                        "cuMemcpyHtoD(resident lookup counts)")
                 && copyResidentScratch(ResidentLookupWavelength, dLookupWavelengthv, lookupWavelengthv.data(),
                                        lookupWavelengthv.size() * sizeof(double),
                                        "cuMemcpyHtoD(resident lookup wavelengths)")
                 && copyResidentScratch(ResidentSectionSca, dSectionScaTablev, sectionScaTablev.data(),
                                        sectionScaTablev.size() * sizeof(double),
                                        "cuMemcpyHtoD(resident scattering sections)")
                 && copyResidentScratch(ResidentSectionExt, dSectionExtTablev, sectionExtTablev.data(),
                                        sectionExtTablev.size() * sizeof(double),
                                        "cuMemcpyHtoD(resident extinction sections)")
                 && ensureResidentScratch(ResidentKey, dKeyv, hashCapacity * sizeof(int),
                                          "cuMemAlloc(resident radiation keys)")
                 && ensureResidentScratch(ResidentSum, dSumv, hashCapacity * sizeof(double),
                                          "cuMemAlloc(resident radiation sums)")
                 && ensureResidentScratch(ResidentCompactCount, dCompactCountv, sizeof(int),
                                          "cuMemAlloc(resident compact count)")
                 && ensureResidentScratch(ResidentCompactKey, dCompactKeyv, totalSegments * sizeof(int),
                                          "cuMemAlloc(resident compact keys)")
                 && ensureResidentScratch(ResidentCompactSum, dCompactSumv, totalSegments * sizeof(double),
                                          "cuMemAlloc(resident compact sums)")
                 && ensureResidentScratch(ResidentCellOut, dCellOutv, cellOutv.size() * sizeof(int),
                                          "cuMemAlloc(resident output cells)")
                 && ensureResidentScratch(ResidentDistanceOut, dDistanceOutv, distanceOutv.size() * sizeof(double),
                                          "cuMemAlloc(resident output distances)")
                 && ensureResidentScratch(ResidentTauAbsOut, dTauAbsOutv, tauAbsOutv.size() * sizeof(double),
                                          "cuMemAlloc(resident output absorption depths)")
                 && ensureResidentScratch(ResidentWeightOut, dWeightOutv, weightOutv.size() * sizeof(double),
                                          "cuMemAlloc(resident output weights)");
            if (sampleOnGpu)
                ok = ok
                     && copyResidentScratch(ResidentRandomSelect, dRandomSelectv, randomSelectv->data(),
                                            randomSelectv->size() * sizeof(double),
                                            "cuMemcpyHtoD(resident random selects)")
                     && copyResidentScratch(ResidentRandomSample, dRandomSamplev, randomSamplev->data(),
                                            randomSamplev->size() * sizeof(double),
                                            "cuMemcpyHtoD(resident random samples)");
            else
                ok = ok
                     && copyResidentScratch(ResidentTauInteract, dTauinteractv, tauinteractv.data(),
                                            tauinteractv.size() * sizeof(double),
                                            "cuMemcpyHtoD(resident interaction depths)")
                     && copyResidentScratch(ResidentPathBiasWeight, dPathBiasWeightv, pathBiasWeightv.data(),
                                            pathBiasWeightv.size() * sizeof(double),
		                                            "cuMemcpyHtoD(resident path bias weights)");
            if (scatterOnGpu)
                ok = ok
                     && copyResidentScratch(ResidentScatterRandomCostheta, dScatterRandomCosthetav,
                                            scatterRandomCosthetav->data(),
                                            scatterRandomCosthetav->size() * sizeof(double),
                                            "cuMemcpyHtoD(resident scatter costheta randoms)")
                     && copyResidentScratch(ResidentScatterRandomPhi, dScatterRandomPhiv,
                                            scatterRandomPhiv->data(), scatterRandomPhiv->size() * sizeof(double),
                                            "cuMemcpyHtoD(resident scatter phi randoms)")
                     && copyResidentScratch(ResidentScatterAsymmpar, dScatterAsymmparv, hgAsymmparv->data(),
                                            hgAsymmparv->size() * sizeof(double),
                                            "cuMemcpyHtoD(resident scatter asymmetry)")
                     && ensureResidentScratch(ResidentScatterDirectionOut, dScatterDirectionOutv,
                                              scatterDirectionOutv->size() * sizeof(double),
                                              "cuMemAlloc(resident scatter directions)");
		    captureResidentProfile(residentUploadMs);
	            if (!ok)
	            {
	                freeAll();
                return false;
            }
            if (accumulateOnGpu && !ensureValueAccumulator(accumulatorKey, numAccumulatorValues, dAccumulator))
            {
                freeAll();
                return false;
            }

            void* compactPathArgs[] = {&_voronoiSiteDevice,          &_voronoiNeighborBeginDevice,
                                       &_voronoiNeighborCountDevice, &_voronoiNeighborIndexDevice,
                                       &_voronoiBlockBeginDevice,    &_voronoiBlockCountDevice,
                                       &_voronoiBlockIndexDevice,    &blockN,
                                       &numCells,                    &eps,
                                       &dPositionv,                  &dDirectionv,
                                       &numPaths,                    &dPathOffsetv,
                                       &xmin,                        &ymin,
                                       &zmin,                        &xmax,
                                       &ymax,                        &zmax,
                                       &maxDistance,                 &_stateDevice,
                                       &numVars,                     &numTables,
                                       &dDensityOffsetv,             &dLookupBeginv,
                                       &dLookupCountv,               &dLookupWavelengthv,
                                       &dSectionExtTablev,           &dNullTablev,
                                       &hasSecond,                   &dLambdav,
                                       &dCellv,                      &dDsv,
                                       &dTauv,                       &dNullTablev,
                                       &dPathCountv,                 &dPathStatusv};
            void* metadataArgs[] = {&dPathOffsetv, &dDsv, &numPaths, &dPathIndexv, &dSv};
            unsigned int segmentBlockSize = 256;
            unsigned int segmentGridSize = static_cast<unsigned int>((totalSegments + segmentBlockSize - 1)
                                                                    / segmentBlockSize);
            void* rfArgs[] = {&dPathOffsetv, &dPathIndexv, &dCellv,        &dDsv,
                              &dTauv,        &dLuminosityv, &dWavelengthBinv, &numWavelengths,
                              &numSegments,  &hashCapacityInt, &dKeyv,     &dSumv};
            void* forcedArgs[] = {&dPathOffsetv,      &dCellv,          &dSv,             &dTauv,
                                  &dTauinteractv,     &dPathBiasWeightv, &dLambdav,       &_stateDevice,
            &numVars,           &numTables,       &dDensityOffsetv, &dLookupBeginv,
            &dLookupCountv,     &dLookupWavelengthv, &dSectionScaTablev,
            &dSectionExtTablev, &numPaths,        &dCellOutv,      &dDistanceOutv,
            &dTauAbsOutv,       &dWeightOutv};
            void* sampledForcedArgs[] = {&dPathOffsetv,      &dCellv,        &dSv,             &dTauv,
                                         &dRandomSelectv,    &dRandomSamplev, &pathLengthBias, &dLambdav,
                                         &_stateDevice,      &numVars,       &numTables,      &dDensityOffsetv,
                                         &dLookupBeginv,     &dLookupCountv, &dLookupWavelengthv,
                                         &dSectionScaTablev, &dSectionExtTablev, &numPaths,   &dCellOutv,
                                         &dDistanceOutv,     &dTauAbsOutv,  &dWeightOutv};
            CUfunction forcedKernel = sampleOnGpu ? _kernelForcedPropagationTableAlbedoSampledResults
                                                  : _kernelForcedPropagationTableAlbedoResults;
            void** forcedKernelArgs = sampleOnGpu ? sampledForcedArgs : forcedArgs;
            const char* forcedKernelLabel = sampleOnGpu
                                                ? "cuLaunchKernel(resident forced_propagation_table_albedo_sampled_results)"
                                                : "cuLaunchKernel(resident forced_propagation_table_albedo_results)";
            void* scatterDirectionArgs[] = {&dDirectionv,             &dLambdav,
                                            &dScatterRandomCosthetav, &dScatterRandomPhiv,
                                            &numPaths,                &hgLookupBegin,
                                            &hgLookupCount,           &dLookupWavelengthv,
                                            &dScatterAsymmparv,       &dScatterDirectionOutv};
		    void* compactRfArgs[] = {&dKeyv, &dSumv, &hashCapacityInt, &dCompactCountv, &dCompactKeyv,
		                                     &dCompactSumv};
	            void* validatePathArgs[] = {&dPathCountv, &dPathStatusv, &dPathOffsetv, &numPaths, &dPathScanv};
	            unsigned int compactGridSize = static_cast<unsigned int>((hashCapacity + segmentBlockSize - 1)
	                                                                    / segmentBlockSize);
	            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
	                 && check(_cuMemsetD32(dKeyv, 0xffffffffU, hashCapacity), "cuMemsetD32(resident radiation keys)")
	                 && check(_cuMemsetD32(dSumv, 0, hashCapacity * 2), "cuMemsetD32(resident radiation sums)")
	                 && check(_cuMemsetD32(dCompactCountv, 0, 1), "cuMemsetD32(resident radiation count)")
	                 && check(_cuMemsetD32(dPathScanv, 1, 1), "cuMemsetD32(resident path validation)")
	                 && check(_cuLaunchKernel(_kernelVoronoiTableSectionOpticalDepthsCompact, pathGridSize, 1, 1,
	                                          pathBlockSize, 1, 1, 0, nullptr, compactPathArgs, nullptr),
	                          "cuLaunchKernel(voronoi resident table optical depths)")
                 && check(_cuLaunchKernel(_kernelPathSegmentMetadata, pathGridSize, 1, 1, pathBlockSize, 1, 1,
                                          0, nullptr, metadataArgs, nullptr),
                          "cuLaunchKernel(path_segment_metadata)")
                 && check(_cuLaunchKernel(_kernelRadiationFieldSumBatch, segmentGridSize, 1, 1, segmentBlockSize,
                                          1, 1, 0, nullptr, rfArgs, nullptr),
                          "cuLaunchKernel(resident radiation_field_contribution_sums_batch)")
		                 && check(_cuLaunchKernel(forcedKernel, pathGridSize, 1, 1, pathBlockSize, 1, 1, 0, nullptr,
		                                          forcedKernelArgs, nullptr),
		                          forcedKernelLabel);
                    if (ok && scatterOnGpu)
                        ok = check(_cuLaunchKernel(_kernelHenyeyGreensteinScatteringDirections, pathGridSize, 1, 1,
                                                   pathBlockSize, 1, 1, 0, nullptr, scatterDirectionArgs, nullptr),
                                   "cuLaunchKernel(resident henyey_greenstein_scattering_directions)");
                    ok = ok
		                 && check(_cuLaunchKernel(_kernelRadiationFieldCompactSums, compactGridSize, 1, 1,
		                                          segmentBlockSize, 1, 1, 0, nullptr, compactRfArgs, nullptr),
		                          "cuLaunchKernel(resident radiation_field_compact_sums)")
	                 && check(_cuLaunchKernel(_kernelValidatePathOffsets, pathGridSize, 1, 1, pathBlockSize, 1, 1,
	                                          0, nullptr, validatePathArgs, nullptr),
	                          "cuLaunchKernel(validate_path_offsets)")
	                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
	                 && check(_cuMemcpyDtoH(&pathOutputOk, dPathScanv, sizeof(int)),
	                          "cuMemcpyDtoH(voronoi resident path validation)")
	                 && check(_cuMemcpyDtoH(&compactCount, dCompactCountv, sizeof(int)),
	                          "cuMemcpyDtoH(resident radiation sum count)");
	            captureResidentProfile(residentKernelMs);
	            if (!ok || pathOutputOk != 1 || compactCount < 0 || static_cast<size_t>(compactCount) > totalSegments)
	            {
	                if (ok && pathOutputOk != 1) _error = "voronoi resident table output kernel failed";
	                freeAll();
	                return false;
	            }
            if (accumulateOnGpu && compactCount)
            {
                int numAccumulatorValuesInt = static_cast<int>(numAccumulatorValues);
                unsigned int accumulateGridSize = static_cast<unsigned int>(
                    (static_cast<size_t>(compactCount) + segmentBlockSize - 1) / segmentBlockSize);
                void* accumulateArgs[] = {&dCompactKeyv, &dCompactSumv, &compactCount, &dAccumulator,
                                          &numAccumulatorValuesInt};
                ok = check(_cuLaunchKernel(_kernelAccumulateValuesByKey, accumulateGridSize, 1, 1, segmentBlockSize,
                                           1, 1, 0, nullptr, accumulateArgs, nullptr),
                           "cuLaunchKernel(resident accumulate_values_by_key)")
                     && check(_cuCtxSynchronize(), "cuCtxSynchronize");
                if (!ok)
                {
                    freeAll();
                    return false;
                }
            }

            if (accumulateOnGpu)
            {
                binIndexv.clear();
                Ldsv.clear();
            }
            else
            {
	                binIndexv.assign(static_cast<size_t>(compactCount), 0);
                Ldsv.assign(static_cast<size_t>(compactCount), 0.);
            }
            ok = check(_cuMemcpyDtoH(cellOutv.data(), dCellOutv, cellOutv.size() * sizeof(int)),
                       "cuMemcpyDtoH(resident forced propagation cells)")
                 && check(_cuMemcpyDtoH(distanceOutv.data(), dDistanceOutv, distanceOutv.size() * sizeof(double)),
                          "cuMemcpyDtoH(resident forced propagation distances)")
                 && check(_cuMemcpyDtoH(tauAbsOutv.data(), dTauAbsOutv, tauAbsOutv.size() * sizeof(double)),
                          "cuMemcpyDtoH(resident forced propagation absorption depths)")
	                 && check(_cuMemcpyDtoH(weightOutv.data(), dWeightOutv, weightOutv.size() * sizeof(double)),
	                          "cuMemcpyDtoH(resident forced propagation weights)");
            if (ok && scatterOnGpu)
                ok = check(_cuMemcpyDtoH(scatterDirectionOutv->data(), dScatterDirectionOutv,
                                         scatterDirectionOutv->size() * sizeof(double)),
                           "cuMemcpyDtoH(resident scatter directions)");
            if (ok && compactCount && !accumulateOnGpu)
            {
                ok = check(_cuMemcpyDtoH(binIndexv.data(), dCompactKeyv, binIndexv.size() * sizeof(int)),
                           "cuMemcpyDtoH(resident radiation compact keys)")
	                     && check(_cuMemcpyDtoH(Ldsv.data(), dCompactSumv, Ldsv.size() * sizeof(double)),
	                              "cuMemcpyDtoH(resident radiation compact sums)");
	            }

	            captureResidentProfile(residentOutputMs);
	            printResidentProfile();
	            freeAll();
	            return ok;
	        }

        bool traceVoronoiMeshGridTableExtinctionTotalsFlat(
            const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
            const vector<int>& neighborCountv, const vector<int>& neighborIndexv, const vector<int>& blockBeginv,
            const vector<int>& blockCountv, const vector<int>& blockIndexv, int blockN, int numCells, double eps,
            double xmin, double ymin, double zmin, double xmax, double ymax, double zmax,
            const vector<double>& positionv, const vector<double>& directionv, const MediumState& state,
            const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
            const vector<double>& lookupWavelengthv, const vector<double>& sectionTablev,
            const vector<double>& lambdav, const vector<double>& maxDistancev, vector<double>& tauv)
        {
            if (!gridKey || positionv.empty() || positionv.size() % 3 != 0 || directionv.size() != positionv.size())
                return false;
            size_t numPathsHost = positionv.size() / 3;
            if (lambdav.size() != numPathsHost || maxDistancev.size() != numPathsHost
                || numCells <= 0 || sitev.size() != 3 * static_cast<size_t>(numCells)
                || neighborBeginv.size() != static_cast<size_t>(numCells)
                || neighborCountv.size() != static_cast<size_t>(numCells))
                return false;
            if (mediaIndexv.empty() || lookupBeginv.size() != mediaIndexv.size()
                || lookupCountv.size() != mediaIndexv.size() || lookupWavelengthv.empty()
                || sectionTablev.size() != lookupWavelengthv.size())
                return false;
            if (blockN > 0
                && (blockBeginv.size() != static_cast<size_t>(blockN) * blockN * blockN
                    || blockCountv.size() != blockBeginv.size()))
                return false;
            if (numPathsHost > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            for (int m = 0; m != numCells; ++m)
            {
                int begin = neighborBeginv[m];
                int count = neighborCountv[m];
                if (begin < 0 || count < 0) return false;
                if (static_cast<size_t>(begin) + static_cast<size_t>(count) > neighborIndexv.size()) return false;
                for (int i = 0; i != count; ++i)
                {
                    int neighbor = neighborIndexv[begin + i];
                    if (neighbor >= numCells || neighbor < -6) return false;
                }
            }
            for (size_t b = 0; b != blockBeginv.size(); ++b)
            {
                int begin = blockBeginv[b];
                int count = blockCountv[b];
                if (begin < 0 || count < 0) return false;
                if (static_cast<size_t>(begin) + static_cast<size_t>(count) > blockIndexv.size()) return false;
                for (int i = 0; i != count; ++i)
                {
                    int cell = blockIndexv[begin + i];
                    if (cell < 0 || cell >= numCells) return false;
                }
            }
            for (size_t h = 0; h != mediaIndexv.size(); ++h)
            {
                if (mediaIndexv[h] < 0 || lookupBeginv[h] < 0 || lookupCountv[h] < 2) return false;
                size_t begin = static_cast<size_t>(lookupBeginv[h]);
                size_t count = static_cast<size_t>(lookupCountv[h]);
                if (begin + count > lookupWavelengthv.size()) return false;
            }

            int numPaths = static_cast<int>(numPathsHost);

            vector<int> densityOffsetv(mediaIndexv.size());
            for (size_t h = 0; h != mediaIndexv.size(); ++h)
            {
                int offset = state.numberDensityOffset(mediaIndexv[h]);
                if (offset < 0) return false;
                densityOffsetv[h] = offset;
            }

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;
            if (!ensureVoronoiOnDevice(gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv,
                                       blockCountv, blockIndexv))
                return false;

            tauv.assign(numPathsHost, 0.);
            vector<int> statusv(numPathsHost, 0);

            CUdeviceptr dPositionv = 0;
            CUdeviceptr dDirectionv = 0;
            CUdeviceptr dMaxDistancev = 0;
            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dLookupBeginv = 0;
            CUdeviceptr dLookupCountv = 0;
            CUdeviceptr dLookupWavelengthv = 0;
            CUdeviceptr dSectionTablev = 0;
            CUdeviceptr dLambdav = 0;
            CUdeviceptr dTauv = 0;
            CUdeviceptr dStatusv = 0;

            auto freeAll = [&]() {
                if (dPositionv) _cuMemFree(dPositionv);
                if (dDirectionv) _cuMemFree(dDirectionv);
                if (dMaxDistancev) _cuMemFree(dMaxDistancev);
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dLookupBeginv) _cuMemFree(dLookupBeginv);
                if (dLookupCountv) _cuMemFree(dLookupCountv);
                if (dLookupWavelengthv) _cuMemFree(dLookupWavelengthv);
                if (dSectionTablev) _cuMemFree(dSectionTablev);
                if (dLambdav) _cuMemFree(dLambdav);
                if (dTauv) _cuMemFree(dTauv);
                if (dStatusv) _cuMemFree(dStatusv);
            };

            bool ok = allocateAndCopy(dPositionv, positionv.data(), positionv.size() * sizeof(double))
                      && allocateAndCopy(dDirectionv, directionv.data(), directionv.size() * sizeof(double))
                      && allocateAndCopy(dMaxDistancev, maxDistancev.data(), maxDistancev.size() * sizeof(double))
                      && allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dLookupBeginv, lookupBeginv.data(), lookupBeginv.size() * sizeof(int))
                      && allocateAndCopy(dLookupCountv, lookupCountv.data(), lookupCountv.size() * sizeof(int))
                      && allocateAndCopy(dLookupWavelengthv, lookupWavelengthv.data(),
                                         lookupWavelengthv.size() * sizeof(double))
                      && allocateAndCopy(dSectionTablev, sectionTablev.data(), sectionTablev.size() * sizeof(double))
                      && allocateAndCopy(dLambdav, lambdav.data(), lambdav.size() * sizeof(double))
                      && allocate(dTauv, tauv.size() * sizeof(double))
                      && allocate(dStatusv, statusv.size() * sizeof(int));
            if (!ok)
            {
                freeAll();
                return false;
            }

            int numVars = state.numVars();
            int numTables = static_cast<int>(mediaIndexv.size());
            unsigned int blockSize = 128;
            unsigned int gridSize = static_cast<unsigned int>((numPathsHost + blockSize - 1) / blockSize);
            void* args[] = {&_voronoiSiteDevice,          &_voronoiNeighborBeginDevice,
                            &_voronoiNeighborCountDevice, &_voronoiNeighborIndexDevice,
                            &_voronoiBlockBeginDevice,    &_voronoiBlockCountDevice,
                            &_voronoiBlockIndexDevice,    &blockN,
                            &numCells,                    &eps,
                            &dPositionv,                  &dDirectionv,
                            &numPaths,                    &xmin,
                            &ymin,                        &zmin,
                            &xmax,                        &ymax,
                            &zmax,                        &dMaxDistancev,
                            &_stateDevice,                &numVars,
                            &numTables,                   &dDensityOffsetv,
                            &dLookupBeginv,               &dLookupCountv,
                            &dLookupWavelengthv,          &dSectionTablev,
                            &dLambdav,                    &dTauv,
                            &dStatusv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelVoronoiTableExtinctionOpticalDepthTotals, gridSize, 1, 1,
                                          blockSize, 1, 1, 0, nullptr, args, nullptr),
                          "cuLaunchKernel(voronoi_table_extinction_optical_depth_totals)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(tauv.data(), dTauv, tauv.size() * sizeof(double)),
                          "cuMemcpyDtoH(voronoi table total optical depth)")
                 && check(_cuMemcpyDtoH(statusv.data(), dStatusv, statusv.size() * sizeof(int)),
                          "cuMemcpyDtoH(voronoi table total status)");
            freeAll();
            if (!ok) return false;

            for (size_t p = 0; p != statusv.size(); ++p)
            {
                if (statusv[p] != 1)
                {
                    _error = "voronoi table total optical-depth kernel failed for path " + std::to_string(p);
                    return false;
                }
            }
            return true;
        }

        bool traceVoronoiMeshGridTableExtinctionTotals(
            const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
            const vector<int>& neighborCountv, const vector<int>& neighborIndexv, const vector<int>& blockBeginv,
            const vector<int>& blockCountv, const vector<int>& blockIndexv, int blockN, int numCells, double eps,
            double xmin, double ymin, double zmin, double xmax, double ymax, double zmax,
            const vector<Position>& positions, const vector<Direction>& directions, const MediumState& state,
            const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
            const vector<double>& lookupWavelengthv, const vector<double>& sectionTablev,
            const vector<double>& lambdav, const vector<double>& maxDistancev, vector<double>& tauv)
        {
            if (positions.empty() || directions.size() != positions.size() || lambdav.size() != positions.size()
                || maxDistancev.size() != positions.size()
                || positions.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
                return false;

            vector<double> positionv(3 * positions.size());
            vector<double> directionv(3 * positions.size());
            for (size_t p = 0; p != positions.size(); ++p)
            {
                double rx, ry, rz;
                positions[p].cartesian(rx, ry, rz);
                double kx, ky, kz;
                directions[p].cartesian(kx, ky, kz);
                size_t base = 3 * p;
                positionv[base] = rx;
                positionv[base + 1] = ry;
                positionv[base + 2] = rz;
                directionv[base] = kx;
                directionv[base + 1] = ky;
                directionv[base + 2] = kz;
            }

            return traceVoronoiMeshGridTableExtinctionTotalsFlat(
                gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv, blockCountv,
                blockIndexv, blockN, numCells, eps, xmin, ymin, zmin, xmax, ymax, zmax, positionv, directionv,
                state, mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv, sectionTablev, lambdav,
                maxDistancev, tauv);
        }

        bool traceVoronoiMeshGridTableExtinctionTotals(
            const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
            const vector<int>& neighborCountv, const vector<int>& neighborIndexv, const vector<int>& blockBeginv,
            const vector<int>& blockCountv, const vector<int>& blockIndexv, int blockN, int numCells, double eps,
            double xmin, double ymin, double zmin, double xmax, double ymax, double zmax,
            const vector<Position>& positions, const Direction& direction, const MediumState& state,
            const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
            const vector<double>& lookupWavelengthv, const vector<double>& sectionTablev,
            const vector<double>& lambdav, const vector<double>& maxDistancev, vector<double>& tauv)
        {
            if (positions.empty() || lambdav.size() != positions.size() || maxDistancev.size() != positions.size()
                || positions.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
                return false;

            double kx, ky, kz;
            direction.cartesian(kx, ky, kz);
            vector<double> positionv(3 * positions.size());
            vector<double> directionv(3 * positions.size());
            for (size_t p = 0; p != positions.size(); ++p)
            {
                double rx, ry, rz;
                positions[p].cartesian(rx, ry, rz);
                size_t base = 3 * p;
                positionv[base] = rx;
                positionv[base + 1] = ry;
                positionv[base + 2] = rz;
                directionv[base] = kx;
                directionv[base + 1] = ky;
                directionv[base + 2] = kz;
            }

            return traceVoronoiMeshGridTableExtinctionTotalsFlat(
                gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv, blockCountv,
                blockIndexv, blockN, numCells, eps, xmin, ymin, zmin, xmax, ymax, zmax, positionv, directionv,
                state, mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv, sectionTablev, lambdav,
                maxDistancev, tauv);
        }

        bool traceVoronoiMeshGridTableExtinctionTotals(
            const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
            const vector<int>& neighborCountv, const vector<int>& neighborIndexv, const vector<int>& blockBeginv,
            const vector<int>& blockCountv, const vector<int>& blockIndexv, int blockN, int numCells, double eps,
            double xmin, double ymin, double zmin, double xmax, double ymax, double zmax,
            const vector<const SpatialGridPath*>& paths, const MediumState& state,
            const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
            const vector<double>& lookupWavelengthv, const vector<double>& sectionTablev,
            const vector<double>& lambdav, const vector<double>& maxDistancev, vector<double>& tauv)
        {
            if (paths.empty() || lambdav.size() != paths.size() || maxDistancev.size() != paths.size()
                || paths.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
                return false;
            for (auto path : paths)
                if (!path) return false;

            vector<double> positionv(3 * paths.size());
            vector<double> directionv(3 * paths.size());
            for (size_t p = 0; p != paths.size(); ++p)
            {
                double rx, ry, rz;
                paths[p]->position().cartesian(rx, ry, rz);
                double kx, ky, kz;
                paths[p]->direction().cartesian(kx, ky, kz);
                size_t base = 3 * p;
                positionv[base] = rx;
                positionv[base + 1] = ry;
                positionv[base + 2] = rz;
                directionv[base] = kx;
                directionv[base + 1] = ky;
                directionv[base + 2] = kz;
            }

            return traceVoronoiMeshGridTableExtinctionTotalsFlat(
                gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv, blockCountv,
                blockIndexv, blockN, numCells, eps, xmin, ymin, zmin, xmax, ymax, zmax, positionv, directionv,
                state, mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv, sectionTablev, lambdav,
                maxDistancev, tauv);
        }

        bool traceVoronoiMeshGridTableHenyeyGreensteinObservedLuminosities(
            const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
            const vector<int>& neighborCountv, const vector<int>& neighborIndexv, const vector<int>& blockBeginv,
            const vector<int>& blockCountv, const vector<int>& blockIndexv, int blockN, int numCells, double eps,
            double xmin, double ymin, double zmin, double xmax, double ymax, double zmax,
            const vector<const SpatialGridPath*>& paths, const MediumState& state,
            const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
            const vector<double>& lookupWavelengthv, const vector<double>& sectionTablev,
            const vector<double>& lambdav, const vector<double>& maxDistancev,
            const vector<double>& inputDirectionv, const vector<double>& packetLuminosityv, Direction bfkobs,
            int hgLookupBegin, int hgLookupCount, const vector<double>& asymmparv, vector<double>& luminosityv)
        {
            if (!gridKey || paths.empty() || lambdav.size() != paths.size() || maxDistancev.size() != paths.size()
                || packetLuminosityv.size() != paths.size() || inputDirectionv.size() != 3 * paths.size()
                || numCells <= 0 || sitev.size() != 3 * static_cast<size_t>(numCells)
                || neighborBeginv.size() != static_cast<size_t>(numCells)
                || neighborCountv.size() != static_cast<size_t>(numCells))
                return false;
            if (mediaIndexv.empty() || lookupBeginv.size() != mediaIndexv.size()
                || lookupCountv.size() != mediaIndexv.size() || lookupWavelengthv.empty()
                || sectionTablev.size() != lookupWavelengthv.size() || asymmparv.size() != lookupWavelengthv.size())
                return false;
            if (hgLookupBegin < 0 || hgLookupCount < 2
                || static_cast<size_t>(hgLookupBegin) + static_cast<size_t>(hgLookupCount) > lookupWavelengthv.size())
                return false;
            if (blockN > 0
                && (blockBeginv.size() != static_cast<size_t>(blockN) * blockN * blockN
                    || blockCountv.size() != blockBeginv.size()))
                return false;
            if (paths.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            for (auto path : paths)
                if (!path) return false;

            for (int m = 0; m != numCells; ++m)
            {
                int begin = neighborBeginv[m];
                int count = neighborCountv[m];
                if (begin < 0 || count < 0) return false;
                if (static_cast<size_t>(begin) + static_cast<size_t>(count) > neighborIndexv.size()) return false;
                for (int i = 0; i != count; ++i)
                {
                    int neighbor = neighborIndexv[begin + i];
                    if (neighbor >= numCells || neighbor < -6) return false;
                }
            }
            for (size_t b = 0; b != blockBeginv.size(); ++b)
            {
                int begin = blockBeginv[b];
                int count = blockCountv[b];
                if (begin < 0 || count < 0) return false;
                if (static_cast<size_t>(begin) + static_cast<size_t>(count) > blockIndexv.size()) return false;
                for (int i = 0; i != count; ++i)
                {
                    int cell = blockIndexv[begin + i];
                    if (cell < 0 || cell >= numCells) return false;
                }
            }
            for (size_t h = 0; h != mediaIndexv.size(); ++h)
            {
                if (mediaIndexv[h] < 0 || lookupBeginv[h] < 0 || lookupCountv[h] < 2) return false;
                size_t begin = static_cast<size_t>(lookupBeginv[h]);
                size_t count = static_cast<size_t>(lookupCountv[h]);
                if (begin + count > lookupWavelengthv.size()) return false;
            }

            size_t numPathsHost = paths.size();
            int numPaths = static_cast<int>(numPathsHost);
            vector<double> positionv(3 * numPathsHost);
            vector<double> directionv(3 * numPathsHost);
            for (size_t p = 0; p != numPathsHost; ++p)
            {
                double rx, ry, rz;
                paths[p]->position().cartesian(rx, ry, rz);
                double kx, ky, kz;
                paths[p]->direction().cartesian(kx, ky, kz);
                size_t base = 3 * p;
                positionv[base] = rx;
                positionv[base + 1] = ry;
                positionv[base + 2] = rz;
                directionv[base] = kx;
                directionv[base + 1] = ky;
                directionv[base + 2] = kz;
            }

            vector<int> densityOffsetv(mediaIndexv.size());
            for (size_t h = 0; h != mediaIndexv.size(); ++h)
            {
                int offset = state.numberDensityOffset(mediaIndexv[h]);
                if (offset < 0) return false;
                densityOffsetv[h] = offset;
            }

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;
            if (!ensureVoronoiOnDevice(gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv,
                                       blockCountv, blockIndexv))
                return false;

            double obsx = 0.;
            double obsy = 0.;
            double obsz = 0.;
            bfkobs.cartesian(obsx, obsy, obsz);
            luminosityv.assign(numPathsHost, 0.);
            vector<int> statusv(numPathsHost, 0);

            CUdeviceptr dPositionv = 0;
            CUdeviceptr dDirectionv = 0;
            CUdeviceptr dInputDirectionv = 0;
            CUdeviceptr dPacketLuminosityv = 0;
            CUdeviceptr dMaxDistancev = 0;
            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dLookupBeginv = 0;
            CUdeviceptr dLookupCountv = 0;
            CUdeviceptr dLookupWavelengthv = 0;
            CUdeviceptr dSectionTablev = 0;
            CUdeviceptr dLambdav = 0;
            CUdeviceptr dAsymmparv = 0;
            CUdeviceptr dLuminosityv = 0;
            CUdeviceptr dStatusv = 0;

            bool ok = copyResidentScratch(ResidentPosition, dPositionv, positionv.data(),
                                          positionv.size() * sizeof(double),
                                          "cuMemcpyHtoD(HG observed positions)")
                      && copyResidentScratch(ResidentDirection, dDirectionv, directionv.data(),
                                             directionv.size() * sizeof(double),
                                             "cuMemcpyHtoD(HG observed directions)")
                      && copyResidentScratch(ResidentScatterDirectionOut, dInputDirectionv, inputDirectionv.data(),
                                             inputDirectionv.size() * sizeof(double),
                                             "cuMemcpyHtoD(HG observed input directions)")
                      && copyResidentScratch(ResidentLuminosity, dPacketLuminosityv, packetLuminosityv.data(),
                                             packetLuminosityv.size() * sizeof(double),
                                             "cuMemcpyHtoD(HG observed packet luminosities)")
                      && copyResidentScratch(ResidentDistance, dMaxDistancev, maxDistancev.data(),
                                             maxDistancev.size() * sizeof(double),
                                             "cuMemcpyHtoD(HG observed max distances)")
                      && copyResidentScratch(ResidentDensityOffset, dDensityOffsetv, densityOffsetv.data(),
                                             densityOffsetv.size() * sizeof(int),
                                             "cuMemcpyHtoD(HG observed density offsets)")
                      && copyResidentScratch(ResidentLookupBegin, dLookupBeginv, lookupBeginv.data(),
                                             lookupBeginv.size() * sizeof(int),
                                             "cuMemcpyHtoD(HG observed lookup begins)")
                      && copyResidentScratch(ResidentLookupCount, dLookupCountv, lookupCountv.data(),
                                             lookupCountv.size() * sizeof(int),
                                             "cuMemcpyHtoD(HG observed lookup counts)")
                      && copyResidentScratch(ResidentLookupWavelength, dLookupWavelengthv,
                                             lookupWavelengthv.data(), lookupWavelengthv.size() * sizeof(double),
                                             "cuMemcpyHtoD(HG observed lookup wavelengths)")
                      && copyResidentScratch(ResidentSectionExt, dSectionTablev, sectionTablev.data(),
                                             sectionTablev.size() * sizeof(double),
                                             "cuMemcpyHtoD(HG observed extinction sections)")
                      && copyResidentScratch(ResidentLambda, dLambdav, lambdav.data(), lambdav.size() * sizeof(double),
                                             "cuMemcpyHtoD(HG observed wavelengths)")
                      && copyResidentScratch(ResidentScatterAsymmpar, dAsymmparv, asymmparv.data(),
                                             asymmparv.size() * sizeof(double),
                                             "cuMemcpyHtoD(HG observed asymmetry)")
                      && ensureResidentScratch(ResidentWeightOut, dLuminosityv,
                                               luminosityv.size() * sizeof(double),
                                               "cuMemAlloc(HG observed luminosities)")
                      && ensureResidentScratch(ResidentPathStatus, dStatusv,
                                               statusv.size() * sizeof(int),
                                               "cuMemAlloc(HG observed status)");
            if (!ok) return false;

            int numVars = state.numVars();
            int numTables = static_cast<int>(mediaIndexv.size());
            unsigned int blockSize = 128;
            unsigned int gridSize = static_cast<unsigned int>((numPathsHost + blockSize - 1) / blockSize);
            void* args[] = {&_voronoiSiteDevice,
                            &_voronoiNeighborBeginDevice,
                            &_voronoiNeighborCountDevice,
                            &_voronoiNeighborIndexDevice,
                            &_voronoiBlockBeginDevice,
                            &_voronoiBlockCountDevice,
                            &_voronoiBlockIndexDevice,
                            &blockN,
                            &numCells,
                            &eps,
                            &dPositionv,
                            &dDirectionv,
                            &dInputDirectionv,
                            &dPacketLuminosityv,
                            &numPaths,
                            &obsx,
                            &obsy,
                            &obsz,
                            &xmin,
                            &ymin,
                            &zmin,
                            &xmax,
                            &ymax,
                            &zmax,
                            &dMaxDistancev,
                            &_stateDevice,
                            &numVars,
                            &numTables,
                            &dDensityOffsetv,
                            &dLookupBeginv,
                            &dLookupCountv,
                            &dLookupWavelengthv,
                            &dSectionTablev,
                            &dLambdav,
                            &hgLookupBegin,
                            &hgLookupCount,
                            &dAsymmparv,
                            &dLuminosityv,
                            &dStatusv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelVoronoiTableHenyeyGreensteinScatteringObservedLuminosities,
                                          gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr),
                          "cuLaunchKernel(voronoi_table_hg_scattering_observed_luminosities)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(luminosityv.data(), dLuminosityv, luminosityv.size() * sizeof(double)),
                          "cuMemcpyDtoH(voronoi table HG observed luminosities)")
	                 && check(_cuMemcpyDtoH(statusv.data(), dStatusv, statusv.size() * sizeof(int)),
	                          "cuMemcpyDtoH(voronoi table HG observed status)");
            if (!ok) return false;

            for (size_t p = 0; p != statusv.size(); ++p)
            {
                if (statusv[p] != 1)
                {
                    _error = "voronoi table HG observed-luminosity kernel failed for path " + std::to_string(p);
                    return false;
                }
	            }
	            return true;
	        }

	        bool traceVoronoiMeshGridTableHenyeyGreensteinFrameBandAccumulate(
	            const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
	            const vector<int>& neighborCountv, const vector<int>& neighborIndexv,
	            const vector<int>& blockBeginv, const vector<int>& blockCountv,
	            const vector<int>& blockIndexv, int blockN, int numCells, double eps, double xmin, double ymin,
	            double zmin, double xmax, double ymax, double zmax, const vector<Position>& positionv,
	            const MediumState& state, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
	            const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
	            const vector<double>& sectionTablev, const vector<double>& lambdav,
	            const vector<double>& maxDistancev, const vector<double>& inputDirectionv,
	            const vector<double>& packetLuminosityv, Direction bfkobs, int hgLookupBegin,
	            int hgLookupCount, const vector<double>& asymmparv, const void* accumulatorKey,
	            size_t numAccumulatorValues, double costheta, double sintheta, double cosphi,
	            double sinphi, double cosomega, double sinomega, int numPixelsX, int numPixelsY,
	            double xpmin, double xpsiz, double ypmin, double ypsiz, double redshift,
	            size_t numPixelsInFrame, const vector<int>& bandOffsetv, const vector<double>& bandWavelengthv,
	            const vector<double>& bandTransmissionv, const vector<double>& bandWidthv)
	        {
	            if (!accumulatorKey || !numAccumulatorValues || positionv.empty()) return false;
	            if (lambdav.size() != positionv.size() || maxDistancev.size() != positionv.size()
	                || packetLuminosityv.size() != positionv.size() || inputDirectionv.size() != 3 * positionv.size())
	                return false;
	            if (mediaIndexv.empty() || lookupBeginv.size() != mediaIndexv.size()
	                || lookupCountv.size() != mediaIndexv.size() || lookupWavelengthv.empty()
	                || sectionTablev.empty() || asymmparv.empty())
	                return false;
	            if (numPixelsX <= 0 || numPixelsY <= 0 || xpsiz <= 0.0 || ypsiz <= 0.0 || bandWidthv.empty()
	                || bandOffsetv.size() != bandWidthv.size() + 1
	                || bandWavelengthv.size() != bandTransmissionv.size())
	                return false;
	            if (positionv.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
	            if (numPixelsInFrame > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
	            if (numAccumulatorValues > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
	            if (bandWidthv.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
	            if (numPixelsInFrame > static_cast<size_t>(std::numeric_limits<int>::max()) / bandWidthv.size())
	                return false;
	            for (size_t ell = 0; ell != bandWidthv.size(); ++ell)
	            {
	                if (bandOffsetv[ell] < 0 || bandOffsetv[ell + 1] < bandOffsetv[ell]) return false;
	                if (static_cast<size_t>(bandOffsetv[ell + 1]) > bandWavelengthv.size()) return false;
	                if (bandOffsetv[ell + 1] - bandOffsetv[ell] < 2) return false;
	                if (bandWidthv[ell] <= 0.0) return false;
	            }

	            size_t numPathsHost = positionv.size();
	            int numPaths = static_cast<int>(numPathsHost);
	            int numBands = static_cast<int>(bandWidthv.size());
	            int numPixelsInFrameInt = static_cast<int>(numPixelsInFrame);
	            int numAccumulatorValuesInt = static_cast<int>(numAccumulatorValues);

	            vector<double> positionData(3 * numPathsHost);
	            for (size_t p = 0; p != numPathsHost; ++p)
	            {
	                double x, y, z;
	                positionv[p].cartesian(x, y, z);
	                size_t base = 3 * p;
	                positionData[base] = x;
	                positionData[base + 1] = y;
	                positionData[base + 2] = z;
	            }

	            vector<int> densityOffsetv(mediaIndexv.size());
	            for (size_t h = 0; h != mediaIndexv.size(); ++h)
	            {
	                int offset = state.numberDensityOffset(mediaIndexv[h]);
	                if (offset < 0) return false;
	                densityOffsetv[h] = offset;
	            }

	            double obsx = 0.;
	            double obsy = 0.;
	            double obsz = 0.;
	            bfkobs.cartesian(obsx, obsy, obsz);
	            vector<int> statusv(numPathsHost, 0);

	            std::lock_guard<std::mutex> lock(_mutex);
	            if (!ensureReady()) return false;
	            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
	            if (!ensureStateOnDevice(state)) return false;
	            if (!ensureModule()) return false;
	            if (!ensureVoronoiOnDevice(gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv,
	                                       blockBeginv, blockCountv, blockIndexv))
	                return false;

	            CUdeviceptr dPositionv = 0;
	            CUdeviceptr dInputDirectionv = 0;
	            CUdeviceptr dPacketLuminosityv = 0;
	            CUdeviceptr dMaxDistancev = 0;
	            CUdeviceptr dDensityOffsetv = 0;
	            CUdeviceptr dLookupBeginv = 0;
	            CUdeviceptr dLookupCountv = 0;
	            CUdeviceptr dLookupWavelengthv = 0;
	            CUdeviceptr dSectionTablev = 0;
	            CUdeviceptr dLambdav = 0;
	            CUdeviceptr dAsymmparv = 0;
	            CUdeviceptr dBandOffsetv = 0;
	            CUdeviceptr dBandWavelengthv = 0;
	            CUdeviceptr dBandTransmissionv = 0;
	            CUdeviceptr dBandWidthv = 0;
	            CUdeviceptr dAccumulator = 0;
	            CUdeviceptr dStatusv = 0;

	            bool ok = copyResidentScratch(ResidentPosition, dPositionv, positionData.data(),
	                                          positionData.size() * sizeof(double),
	                                          "cuMemcpyHtoD(HG frame positions)")
	                      && copyResidentScratch(ResidentScatterDirectionOut, dInputDirectionv,
	                                             inputDirectionv.data(), inputDirectionv.size() * sizeof(double),
	                                             "cuMemcpyHtoD(HG frame input directions)")
	                      && copyResidentScratch(ResidentLuminosity, dPacketLuminosityv,
	                                             packetLuminosityv.data(),
	                                             packetLuminosityv.size() * sizeof(double),
	                                             "cuMemcpyHtoD(HG frame packet luminosities)")
	                      && copyResidentScratch(ResidentDistance, dMaxDistancev, maxDistancev.data(),
	                                             maxDistancev.size() * sizeof(double),
	                                             "cuMemcpyHtoD(HG frame max distances)")
	                      && copyResidentScratch(ResidentDensityOffset, dDensityOffsetv, densityOffsetv.data(),
	                                             densityOffsetv.size() * sizeof(int),
	                                             "cuMemcpyHtoD(HG frame density offsets)")
	                      && copyResidentScratch(ResidentLookupBegin, dLookupBeginv, lookupBeginv.data(),
	                                             lookupBeginv.size() * sizeof(int),
	                                             "cuMemcpyHtoD(HG frame lookup begins)")
	                      && copyResidentScratch(ResidentLookupCount, dLookupCountv, lookupCountv.data(),
	                                             lookupCountv.size() * sizeof(int),
	                                             "cuMemcpyHtoD(HG frame lookup counts)")
	                      && copyResidentScratch(ResidentLookupWavelength, dLookupWavelengthv,
	                                             lookupWavelengthv.data(), lookupWavelengthv.size() * sizeof(double),
	                                             "cuMemcpyHtoD(HG frame lookup wavelengths)")
	                      && copyResidentScratch(ResidentSectionExt, dSectionTablev, sectionTablev.data(),
	                                             sectionTablev.size() * sizeof(double),
	                                             "cuMemcpyHtoD(HG frame extinction sections)")
	                      && copyResidentScratch(ResidentLambda, dLambdav, lambdav.data(),
	                                             lambdav.size() * sizeof(double),
	                                             "cuMemcpyHtoD(HG frame wavelengths)")
	                      && copyResidentScratch(ResidentScatterAsymmpar, dAsymmparv, asymmparv.data(),
	                                             asymmparv.size() * sizeof(double),
	                                             "cuMemcpyHtoD(HG frame asymmetry)")
	                      && copyResidentScratch(ResidentBandOffset, dBandOffsetv, bandOffsetv.data(),
	                                             bandOffsetv.size() * sizeof(int),
	                                             "cuMemcpyHtoD(HG frame band offsets)")
	                      && copyResidentScratch(ResidentBandWavelength, dBandWavelengthv,
	                                             bandWavelengthv.data(), bandWavelengthv.size() * sizeof(double),
	                                             "cuMemcpyHtoD(HG frame band wavelengths)")
	                      && copyResidentScratch(ResidentBandTransmission, dBandTransmissionv,
	                                             bandTransmissionv.data(),
	                                             bandTransmissionv.size() * sizeof(double),
	                                             "cuMemcpyHtoD(HG frame band transmissions)")
	                      && copyResidentScratch(ResidentBandWidth, dBandWidthv, bandWidthv.data(),
	                                             bandWidthv.size() * sizeof(double),
	                                             "cuMemcpyHtoD(HG frame band widths)")
	                      && ensureResidentScratch(ResidentPathStatus, dStatusv,
	                                               statusv.size() * sizeof(int),
	                                               "cuMemAlloc(HG frame status)")
	                      && ensureValueAccumulator(accumulatorKey, numAccumulatorValues, dAccumulator);
	            if (!ok) return false;

	            int numVars = state.numVars();
	            int numTables = static_cast<int>(mediaIndexv.size());
	            unsigned int blockSize = 128;
	            unsigned int gridSize = static_cast<unsigned int>((numPathsHost + blockSize - 1) / blockSize);
	            void* args[] = {&_voronoiSiteDevice,
	                            &_voronoiNeighborBeginDevice,
	                            &_voronoiNeighborCountDevice,
	                            &_voronoiNeighborIndexDevice,
	                            &_voronoiBlockBeginDevice,
	                            &_voronoiBlockCountDevice,
	                            &_voronoiBlockIndexDevice,
	                            &blockN,
	                            &numCells,
	                            &eps,
	                            &dPositionv,
	                            &dInputDirectionv,
	                            &dPacketLuminosityv,
	                            &numPaths,
	                            &obsx,
	                            &obsy,
	                            &obsz,
	                            &xmin,
	                            &ymin,
	                            &zmin,
	                            &xmax,
	                            &ymax,
	                            &zmax,
	                            &dMaxDistancev,
	                            &_stateDevice,
	                            &numVars,
	                            &numTables,
	                            &dDensityOffsetv,
	                            &dLookupBeginv,
	                            &dLookupCountv,
	                            &dLookupWavelengthv,
	                            &dSectionTablev,
	                            &dLambdav,
	                            &hgLookupBegin,
	                            &hgLookupCount,
	                            &dAsymmparv,
	                            &costheta,
	                            &sintheta,
	                            &cosphi,
	                            &sinphi,
	                            &cosomega,
	                            &sinomega,
	                            &numPixelsX,
	                            &numPixelsY,
	                            &xpmin,
	                            &xpsiz,
	                            &ypmin,
	                            &ypsiz,
	                            &redshift,
	                            &numPixelsInFrameInt,
	                            &numBands,
	                            &dBandOffsetv,
	                            &dBandWavelengthv,
	                            &dBandTransmissionv,
	                            &dBandWidthv,
	                            &dAccumulator,
	                            &numAccumulatorValuesInt,
	                            &dStatusv};
	            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
	                 && check(_cuLaunchKernel(_kernelVoronoiTableHenyeyGreensteinScatteringFrameBandAccumulate,
	                                          gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr),
	                          "cuLaunchKernel(voronoi_table_hg_scattering_frame_band_accumulate)")
	                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
	                 && check(_cuMemcpyDtoH(statusv.data(), dStatusv, statusv.size() * sizeof(int)),
	                          "cuMemcpyDtoH(voronoi table HG frame status)");
	            if (!ok) return false;

	            for (size_t p = 0; p != statusv.size(); ++p)
	            {
	                if (statusv[p] != 1)
	                {
	                    _error = "voronoi table HG frame-band kernel failed for path " + std::to_string(p);
	                    return false;
	                }
	            }
	            return true;
	        }

		        bool computeConstantSectionContributions(const SpatialGridPath* path, const MediumState& state,
	                                                 const vector<double>& section1v, const vector<double>* section2v,
	                                                 vector<double>& out1v, vector<double>& out2v)
        {
            if (path->segments().size() < minGpuSegments()) return false;
            if (section1v.empty() || (section2v && section2v->size() != section1v.size())) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;

            size_t count = path->segments().size();
            int numSegments = static_cast<int>(count);
            int numMedia = static_cast<int>(section1v.size());
            int numVars = state.numVars();

            vector<int> cellv(count);
            vector<double> dsv(count);
            for (size_t i = 0; i != count; ++i)
            {
                const auto& segment = path->segments()[i];
                cellv[i] = segment.m();
                dsv[i] = segment.ds();
            }

            vector<int> densityOffsetv(numMedia);
            for (int h = 0; h != numMedia; ++h) densityOffsetv[h] = state.numberDensityOffset(h);

            out1v.assign(count, 0.);
            out2v.assign(section2v ? count : 0, 0.);

            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dSection1v = 0;
            CUdeviceptr dSection2v = 0;
            CUdeviceptr dOut1v = 0;
            CUdeviceptr dOut2v = 0;

            auto freeAll = [&]() {
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dSection1v) _cuMemFree(dSection1v);
                if (dSection2v) _cuMemFree(dSection2v);
                if (dOut1v) _cuMemFree(dOut1v);
                if (dOut2v) _cuMemFree(dOut2v);
            };

            bool ok = allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dDsv, dsv.data(), dsv.size() * sizeof(double))
                      && allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dSection1v, section1v.data(), section1v.size() * sizeof(double))
                      && allocate(dOut1v, out1v.size() * sizeof(double));
            if (section2v)
            {
                ok = ok && allocateAndCopy(dSection2v, section2v->data(), section2v->size() * sizeof(double))
                     && allocate(dOut2v, out2v.size() * sizeof(double));
            }

            if (!ok)
            {
                freeAll();
                return false;
            }

            int hasSecond = section2v ? 1 : 0;
            void* args[] = {&dCellv,        &dDsv,      &_stateDevice, &numSegments, &numVars,   &numMedia,
                            &dDensityOffsetv, &dSection1v, &dSection2v, &hasSecond,   &dOut1v,    &dOut2v};

            unsigned int blockSize = 256;
            unsigned int gridSize = static_cast<unsigned int>((count + blockSize - 1) / blockSize);
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelConstantSection, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args,
                                          nullptr),
                          "cuLaunchKernel(constant_section_contribution)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(out1v.data(), dOut1v, out1v.size() * sizeof(double)), "cuMemcpyDtoH");
            if (ok && section2v)
                ok = check(_cuMemcpyDtoH(out2v.data(), dOut2v, out2v.size() * sizeof(double)), "cuMemcpyDtoH");

            freeAll();
            return ok;
        }

        bool computeTableSectionContributions(const SpatialGridPath* path, const MediumState& state,
                                              const vector<int>& mediaIndexv,
                                              const vector<int>& lookupBeginv,
                                              const vector<int>& lookupCountv,
                                              const vector<double>& lookupWavelengthv,
                                              const vector<double>& section1Tablev,
                                              const vector<double>* section2Tablev, double lambda,
                                              vector<double>& out1v, vector<double>& out2v)
        {
            if (path->segments().size() < minGpuSegments()) return false;
            if (mediaIndexv.empty() || lookupBeginv.size() != mediaIndexv.size()
                || lookupCountv.size() != mediaIndexv.size() || lookupWavelengthv.empty()
                || section1Tablev.size() != lookupWavelengthv.size()
                || (section2Tablev && section2Tablev->size() != lookupWavelengthv.size()))
                return false;

            for (size_t h = 0; h != mediaIndexv.size(); ++h)
            {
                if (mediaIndexv[h] < 0 || lookupBeginv[h] < 0 || lookupCountv[h] < 2) return false;
                size_t begin = static_cast<size_t>(lookupBeginv[h]);
                size_t count = static_cast<size_t>(lookupCountv[h]);
                if (begin + count > lookupWavelengthv.size()) return false;
            }

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;

            size_t count = path->segments().size();
            int numSegments = static_cast<int>(count);
            int numTables = static_cast<int>(mediaIndexv.size());
            int numVars = state.numVars();

            vector<int> cellv(count);
            vector<double> dsv(count);
            for (size_t i = 0; i != count; ++i)
            {
                const auto& segment = path->segments()[i];
                cellv[i] = segment.m();
                dsv[i] = segment.ds();
            }

            vector<int> densityOffsetv(numTables);
            for (int h = 0; h != numTables; ++h) densityOffsetv[h] = state.numberDensityOffset(mediaIndexv[h]);

            out1v.assign(count, 0.);
            out2v.assign(section2Tablev ? count : 0, 0.);

            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dLookupBeginv = 0;
            CUdeviceptr dLookupCountv = 0;
            CUdeviceptr dLookupWavelengthv = 0;
            CUdeviceptr dSection1Tablev = 0;
            CUdeviceptr dSection2Tablev = 0;
            CUdeviceptr dOut1v = 0;
            CUdeviceptr dOut2v = 0;

            auto freeAll = [&]() {
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dLookupBeginv) _cuMemFree(dLookupBeginv);
                if (dLookupCountv) _cuMemFree(dLookupCountv);
                if (dLookupWavelengthv) _cuMemFree(dLookupWavelengthv);
                if (dSection1Tablev) _cuMemFree(dSection1Tablev);
                if (dSection2Tablev) _cuMemFree(dSection2Tablev);
                if (dOut1v) _cuMemFree(dOut1v);
                if (dOut2v) _cuMemFree(dOut2v);
            };

            bool ok = allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dDsv, dsv.data(), dsv.size() * sizeof(double))
                      && allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dLookupBeginv, lookupBeginv.data(), lookupBeginv.size() * sizeof(int))
                      && allocateAndCopy(dLookupCountv, lookupCountv.data(), lookupCountv.size() * sizeof(int))
                      && allocateAndCopy(dLookupWavelengthv, lookupWavelengthv.data(),
                                         lookupWavelengthv.size() * sizeof(double))
                      && allocateAndCopy(dSection1Tablev, section1Tablev.data(),
                                         section1Tablev.size() * sizeof(double))
                      && allocate(dOut1v, out1v.size() * sizeof(double));
            if (section2Tablev)
            {
                ok = ok
                     && allocateAndCopy(dSection2Tablev, section2Tablev->data(),
                                        section2Tablev->size() * sizeof(double))
                     && allocate(dOut2v, out2v.size() * sizeof(double));
            }

            if (!ok)
            {
                freeAll();
                return false;
            }

            int hasSecond = section2Tablev ? 1 : 0;
            void* args[] = {&dCellv,          &dDsv,            &_stateDevice,        &numSegments,
                            &numVars,         &numTables,       &dDensityOffsetv,     &dLookupBeginv,
                            &dLookupCountv,   &dLookupWavelengthv, &dSection1Tablev,  &dSection2Tablev,
                            &hasSecond,       &lambda,          &dOut1v,             &dOut2v};

            unsigned int blockSize = 256;
            unsigned int gridSize = static_cast<unsigned int>((count + blockSize - 1) / blockSize);
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelTableSection, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args,
                                          nullptr),
                          "cuLaunchKernel(table_section_contribution)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(out1v.data(), dOut1v, out1v.size() * sizeof(double)),
                          "cuMemcpyDtoH(table contribution)");
            if (ok && section2Tablev)
                ok = check(_cuMemcpyDtoH(out2v.data(), dOut2v, out2v.size() * sizeof(double)),
                           "cuMemcpyDtoH(table second contribution)");

            freeAll();
            return ok;
        }

        bool computeCumulativeTableSectionOpticalDepthsBatch(
            const vector<SpatialGridPath*>& paths, const MediumState& state, const vector<int>& mediaIndexv,
            const vector<int>& lookupBeginv, const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
            const vector<double>& section1Tablev, const vector<double>* section2Tablev, const vector<double>& lambdav,
            vector<int>& pathOffsetv, vector<double>& out1v, vector<double>& out2v)
        {
            if (paths.empty() || lambdav.size() != paths.size()) return false;
            if (mediaIndexv.empty() || lookupBeginv.size() != mediaIndexv.size()
                || lookupCountv.size() != mediaIndexv.size() || lookupWavelengthv.empty()
                || section1Tablev.size() != lookupWavelengthv.size()
                || (section2Tablev && section2Tablev->size() != lookupWavelengthv.size()))
                return false;
            if (paths.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            for (size_t h = 0; h != mediaIndexv.size(); ++h)
            {
                if (mediaIndexv[h] < 0 || lookupBeginv[h] < 0 || lookupCountv[h] < 2) return false;
                size_t begin = static_cast<size_t>(lookupBeginv[h]);
                size_t count = static_cast<size_t>(lookupCountv[h]);
                if (begin + count > lookupWavelengthv.size()) return false;
            }

            pathOffsetv.assign(paths.size() + 1, 0);
            size_t totalSegments = 0;
            for (size_t pathIndex = 0; pathIndex != paths.size(); ++pathIndex)
            {
                if (!paths[pathIndex]) return false;
                totalSegments += paths[pathIndex]->segments().size();
                if (totalSegments > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
                pathOffsetv[pathIndex + 1] = static_cast<int>(totalSegments);
            }
            if (!totalSegments || totalSegments < minGpuSegments()) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;

            int numPaths = static_cast<int>(paths.size());
            int numTables = static_cast<int>(mediaIndexv.size());
            int numVars = state.numVars();

            vector<int> cellv(totalSegments, -1);
            vector<double> dsv(totalSegments, 0.);
            size_t segmentIndex = 0;
            for (const SpatialGridPath* path : paths)
            {
                for (const auto& segment : path->segments())
                {
                    cellv[segmentIndex] = segment.m();
                    dsv[segmentIndex] = segment.ds();
                    ++segmentIndex;
                }
            }

            vector<int> densityOffsetv(numTables);
            for (int h = 0; h != numTables; ++h) densityOffsetv[h] = state.numberDensityOffset(mediaIndexv[h]);

            out1v.assign(totalSegments, 0.);
            out2v.assign(section2Tablev ? totalSegments : 0, 0.);

            CUdeviceptr dPathOffsetv = 0;
            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dLookupBeginv = 0;
            CUdeviceptr dLookupCountv = 0;
            CUdeviceptr dLookupWavelengthv = 0;
            CUdeviceptr dSection1Tablev = 0;
            CUdeviceptr dSection2Tablev = 0;
            CUdeviceptr dLambdav = 0;
            CUdeviceptr dOut1v = 0;
            CUdeviceptr dOut2v = 0;

            auto freeAll = [&]() {
                if (dPathOffsetv) _cuMemFree(dPathOffsetv);
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dLookupBeginv) _cuMemFree(dLookupBeginv);
                if (dLookupCountv) _cuMemFree(dLookupCountv);
                if (dLookupWavelengthv) _cuMemFree(dLookupWavelengthv);
                if (dSection1Tablev) _cuMemFree(dSection1Tablev);
                if (dSection2Tablev) _cuMemFree(dSection2Tablev);
                if (dLambdav) _cuMemFree(dLambdav);
                if (dOut1v) _cuMemFree(dOut1v);
                if (dOut2v) _cuMemFree(dOut2v);
            };

            bool ok = allocateAndCopy(dPathOffsetv, pathOffsetv.data(), pathOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dDsv, dsv.data(), dsv.size() * sizeof(double))
                      && allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dLookupBeginv, lookupBeginv.data(), lookupBeginv.size() * sizeof(int))
                      && allocateAndCopy(dLookupCountv, lookupCountv.data(), lookupCountv.size() * sizeof(int))
                      && allocateAndCopy(dLookupWavelengthv, lookupWavelengthv.data(),
                                         lookupWavelengthv.size() * sizeof(double))
                      && allocateAndCopy(dSection1Tablev, section1Tablev.data(),
                                         section1Tablev.size() * sizeof(double))
                      && allocateAndCopy(dLambdav, lambdav.data(), lambdav.size() * sizeof(double))
                      && allocate(dOut1v, out1v.size() * sizeof(double));
            if (section2Tablev)
            {
                ok = ok
                     && allocateAndCopy(dSection2Tablev, section2Tablev->data(),
                                        section2Tablev->size() * sizeof(double))
                     && allocate(dOut2v, out2v.size() * sizeof(double));
            }
            if (!ok)
            {
                freeAll();
                return false;
            }

            int hasSecond = section2Tablev ? 1 : 0;
            void* args[] = {&dPathOffsetv,   &dCellv,          &dDsv,           &_stateDevice,    &numVars,
                            &numTables,      &dDensityOffsetv, &dLookupBeginv,  &dLookupCountv,   &dLookupWavelengthv,
                            &dSection1Tablev, &dSection2Tablev, &hasSecond,     &dLambdav,        &numPaths,
                            &dOut1v,         &dOut2v};
            unsigned int blockSize = 128;
            unsigned int gridSize = static_cast<unsigned int>((paths.size() + blockSize - 1) / blockSize);
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelCumulativeTableSectionOpticalDepthsBatch, gridSize, 1, 1,
                                          blockSize, 1, 1, 0, nullptr, args, nullptr),
                          "cuLaunchKernel(cumulative_table_section_optical_depths_batch)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(out1v.data(), dOut1v, out1v.size() * sizeof(double)),
                          "cuMemcpyDtoH(cumulative table batch optical depth)");
            if (ok && section2Tablev)
                ok = check(_cuMemcpyDtoH(out2v.data(), dOut2v, out2v.size() * sizeof(double)),
                           "cuMemcpyDtoH(cumulative table batch absorption optical depth)");

            freeAll();
            return ok;
        }

        bool computeCumulativeConstantSectionOpticalDepths(const SpatialGridPath* path, const MediumState& state,
                                                           const vector<double>& section1v,
                                                           const vector<double>* section2v, vector<double>& out1v,
                                                           vector<double>& out2v)
        {
            if (path->segments().size() < minGpuSegments()) return false;
            if (section1v.empty() || (section2v && section2v->size() != section1v.size())) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;

            size_t count = path->segments().size();
            int numSegments = static_cast<int>(count);
            int numMedia = static_cast<int>(section1v.size());
            int numVars = state.numVars();

            vector<int> cellv(count);
            vector<double> dsv(count);
            for (size_t i = 0; i != count; ++i)
            {
                const auto& segment = path->segments()[i];
                cellv[i] = segment.m();
                dsv[i] = segment.ds();
            }

            vector<int> densityOffsetv(numMedia);
            for (int h = 0; h != numMedia; ++h) densityOffsetv[h] = state.numberDensityOffset(h);

            out1v.assign(count, 0.);
            out2v.assign(section2v ? count : 0, 0.);

            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dSection1v = 0;
            CUdeviceptr dSection2v = 0;
            CUdeviceptr dContribution1v = 0;
            CUdeviceptr dContribution2v = 0;
            CUdeviceptr dOut1v = 0;
            CUdeviceptr dOut2v = 0;

            auto freeAll = [&]() {
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dSection1v) _cuMemFree(dSection1v);
                if (dSection2v) _cuMemFree(dSection2v);
                if (dContribution1v) _cuMemFree(dContribution1v);
                if (dContribution2v) _cuMemFree(dContribution2v);
                if (dOut1v) _cuMemFree(dOut1v);
                if (dOut2v) _cuMemFree(dOut2v);
            };

            bool ok = allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dDsv, dsv.data(), dsv.size() * sizeof(double))
                      && allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dSection1v, section1v.data(), section1v.size() * sizeof(double))
                      && allocate(dContribution1v, out1v.size() * sizeof(double))
                      && allocate(dOut1v, out1v.size() * sizeof(double));
            if (section2v)
            {
                ok = ok && allocateAndCopy(dSection2v, section2v->data(), section2v->size() * sizeof(double))
                     && allocate(dContribution2v, out2v.size() * sizeof(double))
                     && allocate(dOut2v, out2v.size() * sizeof(double));
            }
            if (!ok)
            {
                freeAll();
                return false;
            }

            int hasSecond = section2v ? 1 : 0;
            void* contributionArgs[] = {&dCellv,          &dDsv,       &_stateDevice,    &numSegments,
                                        &numVars,         &numMedia,   &dDensityOffsetv, &dSection1v,
                                        &dSection2v,      &hasSecond,  &dContribution1v, &dContribution2v};
            unsigned int blockSize = 256;
            unsigned int gridSize = static_cast<unsigned int>((count + blockSize - 1) / blockSize);
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelConstantSection, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr,
                                          contributionArgs, nullptr),
                          "cuLaunchKernel(constant_section_contribution)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize");
            if (ok)
            {
                void* cumulativeArgs[] = {&dContribution1v, &dContribution2v, &hasSecond, &numSegments, &dOut1v,
                                          &dOut2v};
                ok = check(_cuLaunchKernel(_kernelCumulativeOpticalDepths, 1, 1, 1, 1, 1, 1, 0, nullptr,
                                           cumulativeArgs, nullptr),
                           "cuLaunchKernel(cumulative_optical_depths)")
                     && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                     && check(_cuMemcpyDtoH(out1v.data(), dOut1v, out1v.size() * sizeof(double)),
                              "cuMemcpyDtoH(cumulative optical depth)");
                if (ok && section2v)
                    ok = check(_cuMemcpyDtoH(out2v.data(), dOut2v, out2v.size() * sizeof(double)),
                               "cuMemcpyDtoH(cumulative absorption optical depth)");
            }

            freeAll();
            return ok;
        }

        bool computeCumulativeConstantSectionOpticalDepthsBatch(const vector<SpatialGridPath*>& paths,
                                                                const MediumState& state,
                                                                const vector<double>& section1v,
                                                                const vector<double>* section2v,
                                                                vector<int>& pathOffsetv,
                                                                vector<double>& out1v,
                                                                vector<double>& out2v)
        {
            if (paths.empty() || section1v.empty() || (section2v && section2v->size() != section1v.size()))
                return false;
            if (paths.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            pathOffsetv.assign(paths.size() + 1, 0);
            size_t totalSegments = 0;
            for (size_t pathIndex = 0; pathIndex != paths.size(); ++pathIndex)
            {
                if (!paths[pathIndex]) return false;
                totalSegments += paths[pathIndex]->segments().size();
                if (totalSegments > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
                pathOffsetv[pathIndex + 1] = static_cast<int>(totalSegments);
            }
            if (!totalSegments || totalSegments < minGpuSegments()) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;

            int numPaths = static_cast<int>(paths.size());
            int numMedia = static_cast<int>(section1v.size());
            int numVars = state.numVars();

            vector<int> cellv(totalSegments, -1);
            vector<double> dsv(totalSegments, 0.);
            size_t segmentIndex = 0;
            for (const SpatialGridPath* path : paths)
            {
                for (const auto& segment : path->segments())
                {
                    cellv[segmentIndex] = segment.m();
                    dsv[segmentIndex] = segment.ds();
                    ++segmentIndex;
                }
            }

            vector<int> densityOffsetv(numMedia);
            for (int h = 0; h != numMedia; ++h) densityOffsetv[h] = state.numberDensityOffset(h);

            out1v.assign(totalSegments, 0.);
            out2v.assign(section2v ? totalSegments : 0, 0.);

            CUdeviceptr dPathOffsetv = 0;
            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dSection1v = 0;
            CUdeviceptr dSection2v = 0;
            CUdeviceptr dOut1v = 0;
            CUdeviceptr dOut2v = 0;

            auto freeAll = [&]() {
                if (dPathOffsetv) _cuMemFree(dPathOffsetv);
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dSection1v) _cuMemFree(dSection1v);
                if (dSection2v) _cuMemFree(dSection2v);
                if (dOut1v) _cuMemFree(dOut1v);
                if (dOut2v) _cuMemFree(dOut2v);
            };

            bool ok = allocateAndCopy(dPathOffsetv, pathOffsetv.data(), pathOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dDsv, dsv.data(), dsv.size() * sizeof(double))
                      && allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dSection1v, section1v.data(), section1v.size() * sizeof(double))
                      && allocate(dOut1v, out1v.size() * sizeof(double));
            if (section2v)
            {
                ok = ok && allocateAndCopy(dSection2v, section2v->data(), section2v->size() * sizeof(double))
                     && allocate(dOut2v, out2v.size() * sizeof(double));
            }
            if (!ok)
            {
                freeAll();
                return false;
            }

            int hasSecond = section2v ? 1 : 0;
            void* args[] = {&dPathOffsetv, &dCellv,        &dDsv,       &_stateDevice, &numVars,
                            &numMedia,     &dDensityOffsetv, &dSection1v, &dSection2v,  &hasSecond,
                            &numPaths,     &dOut1v,        &dOut2v};
            unsigned int blockSize = 128;
            unsigned int gridSize = static_cast<unsigned int>((paths.size() + blockSize - 1) / blockSize);
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelCumulativeConstantSectionOpticalDepthsBatch, gridSize, 1, 1,
                                          blockSize, 1, 1, 0, nullptr, args, nullptr),
                          "cuLaunchKernel(cumulative_constant_section_optical_depths_batch)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(out1v.data(), dOut1v, out1v.size() * sizeof(double)),
                          "cuMemcpyDtoH(cumulative batch optical depth)");
            if (ok && section2v)
                ok = check(_cuMemcpyDtoH(out2v.data(), dOut2v, out2v.size() * sizeof(double)),
                           "cuMemcpyDtoH(cumulative batch absorption optical depth)");

            freeAll();
            return ok;
        }

        bool computeExtinctionOpticalDepth(const SpatialGridPath* path, const MediumState& state,
                                           const vector<double>& sectionv, double taumax, double& tau)
        {
            if (path->segments().size() < minGpuSegments()) return false;
            if (sectionv.empty()) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;

            size_t count = path->segments().size();
            int numSegments = static_cast<int>(count);
            int numMedia = static_cast<int>(sectionv.size());
            int numVars = state.numVars();

            vector<int> cellv(count);
            vector<double> dsv(count);
            for (size_t i = 0; i != count; ++i)
            {
                const auto& segment = path->segments()[i];
                cellv[i] = segment.m();
                dsv[i] = segment.ds();
            }

            vector<int> densityOffsetv(numMedia);
            for (int h = 0; h != numMedia; ++h) densityOffsetv[h] = state.numberDensityOffset(h);

            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dSectionv = 0;
            CUdeviceptr dContributionv = 0;
            CUdeviceptr dTauv = 0;
            CUdeviceptr dNull = 0;

            auto freeAll = [&]() {
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dSectionv) _cuMemFree(dSectionv);
                if (dContributionv) _cuMemFree(dContributionv);
                if (dTauv) _cuMemFree(dTauv);
            };

            bool ok = allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dDsv, dsv.data(), dsv.size() * sizeof(double))
                      && allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dSectionv, sectionv.data(), sectionv.size() * sizeof(double))
                      && allocate(dContributionv, count * sizeof(double)) && allocate(dTauv, sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            int hasSecond = 0;
            void* contributionArgs[] = {&dCellv,        &dDsv,      &_stateDevice, &numSegments, &numVars, &numMedia,
                                        &dDensityOffsetv, &dSectionv, &dNull,        &hasSecond,   &dContributionv,
                                        &dNull};
            unsigned int blockSize = 256;
            unsigned int gridSize = static_cast<unsigned int>((count + blockSize - 1) / blockSize);
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelConstantSection, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr,
                                          contributionArgs, nullptr),
                          "cuLaunchKernel(constant_section_contribution)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize");
            if (ok)
            {
                void* sumArgs[] = {&dContributionv, &numSegments, &taumax, &dTauv};
                ok = check(_cuLaunchKernel(_kernelOpticalDepthSum, 1, 1, 1, 1, 1, 1, 0, nullptr, sumArgs, nullptr),
                           "cuLaunchKernel(optical_depth_sum)")
                     && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                     && check(_cuMemcpyDtoH(&tau, dTauv, sizeof(double)), "cuMemcpyDtoH(optical depth)");
            }

            freeAll();
            return ok;
        }

        bool findInteractionPointUsingExtinction(const SpatialGridPath* path, const MediumState& state,
                                                 const vector<double>& sectionv, double tauinteract, bool& found,
                                                 int& m, double& s)
        {
            found = false;
            m = -1;
            s = 0.;
            if (path->segments().size() < minGpuSegments()) return false;
            if (sectionv.empty()) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;

            size_t count = path->segments().size();
            int numSegments = static_cast<int>(count);
            int numMedia = static_cast<int>(sectionv.size());
            int numVars = state.numVars();

            vector<int> cellv(count);
            vector<double> dsv(count);
            for (size_t i = 0; i != count; ++i)
            {
                const auto& segment = path->segments()[i];
                cellv[i] = segment.m();
                dsv[i] = segment.ds();
            }

            vector<int> densityOffsetv(numMedia);
            for (int h = 0; h != numMedia; ++h) densityOffsetv[h] = state.numberDensityOffset(h);

            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dSectionv = 0;
            CUdeviceptr dContributionv = 0;
            CUdeviceptr dFoundv = 0;
            CUdeviceptr dCellOutv = 0;
            CUdeviceptr dDistanceOutv = 0;
            CUdeviceptr dNull = 0;

            auto freeAll = [&]() {
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dSectionv) _cuMemFree(dSectionv);
                if (dContributionv) _cuMemFree(dContributionv);
                if (dFoundv) _cuMemFree(dFoundv);
                if (dCellOutv) _cuMemFree(dCellOutv);
                if (dDistanceOutv) _cuMemFree(dDistanceOutv);
            };

            bool ok = allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dDsv, dsv.data(), dsv.size() * sizeof(double))
                      && allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dSectionv, sectionv.data(), sectionv.size() * sizeof(double))
                      && allocate(dContributionv, count * sizeof(double)) && allocate(dFoundv, sizeof(int))
                      && allocate(dCellOutv, sizeof(int)) && allocate(dDistanceOutv, sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            int hasSecond = 0;
            void* contributionArgs[] = {&dCellv,        &dDsv,      &_stateDevice, &numSegments, &numVars, &numMedia,
                                        &dDensityOffsetv, &dSectionv, &dNull,        &hasSecond,   &dContributionv,
                                        &dNull};
            unsigned int blockSize = 256;
            unsigned int gridSize = static_cast<unsigned int>((count + blockSize - 1) / blockSize);
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelConstantSection, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr,
                                          contributionArgs, nullptr),
                          "cuLaunchKernel(constant_section_contribution)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize");
            if (ok)
            {
                void* interactionArgs[] = {&dCellv, &dDsv, &dContributionv, &numSegments, &tauinteract,
                                           &dFoundv, &dCellOutv, &dDistanceOutv};
                int foundFlag = 0;
                ok = check(_cuLaunchKernel(_kernelInteractionPointExtinction, 1, 1, 1, 1, 1, 1, 0, nullptr,
                                           interactionArgs, nullptr),
                           "cuLaunchKernel(interaction_point_extinction)")
                     && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                     && check(_cuMemcpyDtoH(&foundFlag, dFoundv, sizeof(int)), "cuMemcpyDtoH(interaction found)")
                     && check(_cuMemcpyDtoH(&m, dCellOutv, sizeof(int)), "cuMemcpyDtoH(interaction cell)")
                     && check(_cuMemcpyDtoH(&s, dDistanceOutv, sizeof(double)), "cuMemcpyDtoH(interaction distance)");
                found = foundFlag != 0;
            }

            freeAll();
            return ok;
        }

        bool findInteractionPointUsingScatteringAndAbsorption(const SpatialGridPath* path, const MediumState& state,
                                                              const vector<double>& sectionScav,
                                                              const vector<double>& sectionAbsv,
                                                              double tauinteract, bool& found, int& m, double& s,
                                                              double& tauAbs)
        {
            found = false;
            m = -1;
            s = 0.;
            tauAbs = 0.;
            if (path->segments().size() < minGpuSegments()) return false;
            if (sectionScav.empty() || sectionScav.size() != sectionAbsv.size()) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;

            size_t count = path->segments().size();
            int numSegments = static_cast<int>(count);
            int numMedia = static_cast<int>(sectionScav.size());
            int numVars = state.numVars();

            vector<int> cellv(count);
            vector<double> dsv(count);
            for (size_t i = 0; i != count; ++i)
            {
                const auto& segment = path->segments()[i];
                cellv[i] = segment.m();
                dsv[i] = segment.ds();
            }

            vector<int> densityOffsetv(numMedia);
            for (int h = 0; h != numMedia; ++h) densityOffsetv[h] = state.numberDensityOffset(h);

            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dSectionScav = 0;
            CUdeviceptr dSectionAbsv = 0;
            CUdeviceptr dScaContributionv = 0;
            CUdeviceptr dAbsContributionv = 0;
            CUdeviceptr dFoundv = 0;
            CUdeviceptr dCellOutv = 0;
            CUdeviceptr dDistanceOutv = 0;
            CUdeviceptr dTauAbsOutv = 0;

            auto freeAll = [&]() {
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dSectionScav) _cuMemFree(dSectionScav);
                if (dSectionAbsv) _cuMemFree(dSectionAbsv);
                if (dScaContributionv) _cuMemFree(dScaContributionv);
                if (dAbsContributionv) _cuMemFree(dAbsContributionv);
                if (dFoundv) _cuMemFree(dFoundv);
                if (dCellOutv) _cuMemFree(dCellOutv);
                if (dDistanceOutv) _cuMemFree(dDistanceOutv);
                if (dTauAbsOutv) _cuMemFree(dTauAbsOutv);
            };

            bool ok = allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dDsv, dsv.data(), dsv.size() * sizeof(double))
                      && allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dSectionScav, sectionScav.data(), sectionScav.size() * sizeof(double))
                      && allocateAndCopy(dSectionAbsv, sectionAbsv.data(), sectionAbsv.size() * sizeof(double))
                      && allocate(dScaContributionv, count * sizeof(double))
                      && allocate(dAbsContributionv, count * sizeof(double)) && allocate(dFoundv, sizeof(int))
                      && allocate(dCellOutv, sizeof(int)) && allocate(dDistanceOutv, sizeof(double))
                      && allocate(dTauAbsOutv, sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            int hasSecond = 1;
            void* contributionArgs[] = {&dCellv,        &dDsv,         &_stateDevice,      &numSegments,
                                        &numVars,       &numMedia,     &dDensityOffsetv,   &dSectionScav,
                                        &dSectionAbsv,  &hasSecond,    &dScaContributionv, &dAbsContributionv};
            unsigned int blockSize = 256;
            unsigned int gridSize = static_cast<unsigned int>((count + blockSize - 1) / blockSize);
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelConstantSection, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr,
                                          contributionArgs, nullptr),
                          "cuLaunchKernel(constant_section_contribution)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize");
            if (ok)
            {
                void* interactionArgs[] = {&dCellv,          &dDsv,       &dScaContributionv, &dAbsContributionv,
                                           &numSegments,     &tauinteract, &dFoundv,          &dCellOutv,
                                           &dDistanceOutv,   &dTauAbsOutv};
                int foundFlag = 0;
                ok = check(_cuLaunchKernel(_kernelInteractionPointScaAbs, 1, 1, 1, 1, 1, 1, 0, nullptr,
                                           interactionArgs, nullptr),
                           "cuLaunchKernel(interaction_point_sca_abs)")
                     && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                     && check(_cuMemcpyDtoH(&foundFlag, dFoundv, sizeof(int)), "cuMemcpyDtoH(interaction found)")
                     && check(_cuMemcpyDtoH(&m, dCellOutv, sizeof(int)), "cuMemcpyDtoH(interaction cell)")
                     && check(_cuMemcpyDtoH(&s, dDistanceOutv, sizeof(double)), "cuMemcpyDtoH(interaction distance)")
                     && check(_cuMemcpyDtoH(&tauAbs, dTauAbsOutv, sizeof(double)),
                              "cuMemcpyDtoH(interaction absorption depth)");
                found = foundFlag != 0;
            }

            freeAll();
            return ok;
        }

        bool computeRadiationFieldContributions(const SpatialGridPath* path, double luminosity, vector<double>& Ldsv)
        {
            if (path->segments().size() < minGpuSegments()) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            size_t count = path->segments().size();
            int numSegments = static_cast<int>(count);

            vector<int> cellv(count);
            vector<double> dsv(count);
            vector<double> tauExtv(count);
            for (size_t i = 0; i != count; ++i)
            {
                const auto& segment = path->segments()[i];
                cellv[i] = segment.m();
                dsv[i] = segment.ds();
                tauExtv[i] = segment.tauExt();
            }
            Ldsv.assign(count, 0.);

            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dTauExtv = 0;
            CUdeviceptr dOutv = 0;

            auto freeAll = [&]() {
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dTauExtv) _cuMemFree(dTauExtv);
                if (dOutv) _cuMemFree(dOutv);
            };

            bool ok = allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dDsv, dsv.data(), dsv.size() * sizeof(double))
                      && allocateAndCopy(dTauExtv, tauExtv.data(), tauExtv.size() * sizeof(double))
                      && allocate(dOutv, Ldsv.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dCellv, &dDsv, &dTauExtv, &numSegments, &luminosity, &dOutv};
            unsigned int blockSize = 256;
            unsigned int gridSize = static_cast<unsigned int>((count + blockSize - 1) / blockSize);
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelRadiationField, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args,
                                          nullptr),
                          "cuLaunchKernel(radiation_field_contribution)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(Ldsv.data(), dOutv, Ldsv.size() * sizeof(double)), "cuMemcpyDtoH");

            freeAll();
            return ok;
        }

        bool computeRadiationFieldContributionsBatch(const vector<const SpatialGridPath*>& paths,
                                                     const vector<double>& luminosityv, vector<int>& pathOffsetv,
                                                     vector<double>& Ldsv)
        {
            if (paths.empty() || luminosityv.size() != paths.size()) return false;
            if (paths.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            pathOffsetv.assign(paths.size() + 1, 0);
            size_t totalSegments = 0;
            for (size_t pathIndex = 0; pathIndex != paths.size(); ++pathIndex)
            {
                if (!paths[pathIndex]) return false;
                totalSegments += paths[pathIndex]->segments().size();
                if (totalSegments > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
                pathOffsetv[pathIndex + 1] = static_cast<int>(totalSegments);
            }
            if (!totalSegments || totalSegments < minGpuSegments()) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            int numSegments = static_cast<int>(totalSegments);
            vector<int> pathIndexv(totalSegments, 0);
            vector<int> cellv(totalSegments, -1);
            vector<double> dsv(totalSegments, 0.);
            vector<double> tauExtv(totalSegments, 0.);
            size_t segmentIndex = 0;
            for (size_t pathIndex = 0; pathIndex != paths.size(); ++pathIndex)
            {
                for (const auto& segment : paths[pathIndex]->segments())
                {
                    pathIndexv[segmentIndex] = static_cast<int>(pathIndex);
                    cellv[segmentIndex] = segment.m();
                    dsv[segmentIndex] = segment.ds();
                    tauExtv[segmentIndex] = segment.tauExt();
                    ++segmentIndex;
                }
            }
            Ldsv.assign(totalSegments, 0.);

            CUdeviceptr dPathOffsetv = 0;
            CUdeviceptr dPathIndexv = 0;
            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dTauExtv = 0;
            CUdeviceptr dLuminosityv = 0;
            CUdeviceptr dOutv = 0;

            auto freeAll = [&]() {
                if (dPathOffsetv) _cuMemFree(dPathOffsetv);
                if (dPathIndexv) _cuMemFree(dPathIndexv);
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dTauExtv) _cuMemFree(dTauExtv);
                if (dLuminosityv) _cuMemFree(dLuminosityv);
                if (dOutv) _cuMemFree(dOutv);
            };

            bool ok = allocateAndCopy(dPathOffsetv, pathOffsetv.data(), pathOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dPathIndexv, pathIndexv.data(), pathIndexv.size() * sizeof(int))
                      && allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dDsv, dsv.data(), dsv.size() * sizeof(double))
                      && allocateAndCopy(dTauExtv, tauExtv.data(), tauExtv.size() * sizeof(double))
                      && allocateAndCopy(dLuminosityv, luminosityv.data(), luminosityv.size() * sizeof(double))
                      && allocate(dOutv, Ldsv.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dPathOffsetv, &dPathIndexv, &dCellv, &dDsv, &dTauExtv, &dLuminosityv, &numSegments,
                            &dOutv};
            unsigned int blockSize = 256;
            unsigned int gridSize = static_cast<unsigned int>((totalSegments + blockSize - 1) / blockSize);
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelRadiationFieldBatch, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr,
                                          args, nullptr),
                          "cuLaunchKernel(radiation_field_contributions_batch)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(Ldsv.data(), dOutv, Ldsv.size() * sizeof(double)),
                          "cuMemcpyDtoH(batch radiation field contributions)");

            freeAll();
            return ok;
        }

        bool computeRadiationFieldContributionSumsBatch(const vector<const SpatialGridPath*>& paths,
                                                        const vector<double>& luminosityv,
                                                        const vector<int>& wavelengthBinv, int numWavelengths,
                                                        vector<int>& binIndexv, vector<double>& Ldsv)
        {
            if (paths.empty() || luminosityv.size() != paths.size() || wavelengthBinv.size() != paths.size()
                || numWavelengths <= 0)
                return false;
            if (paths.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            int maxCellForKey =
                (std::numeric_limits<int>::max() - (numWavelengths - 1)) / numWavelengths;
            vector<int> pathOffsetv(paths.size() + 1, 0);
            size_t totalSegments = 0;
            for (size_t pathIndex = 0; pathIndex != paths.size(); ++pathIndex)
            {
                if (!paths[pathIndex]) return false;
                if (wavelengthBinv[pathIndex] < 0 || wavelengthBinv[pathIndex] >= numWavelengths) return false;
                totalSegments += paths[pathIndex]->segments().size();
                if (totalSegments > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
                pathOffsetv[pathIndex + 1] = static_cast<int>(totalSegments);
            }
            if (!totalSegments || totalSegments < minGpuSegments()) return false;
            if (totalSegments > static_cast<size_t>(std::numeric_limits<int>::max()) / 2) return false;
            size_t hashCapacity = nextPowerOfTwo(totalSegments * 2);
            if (hashCapacity > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            int hashCapacityInt = static_cast<int>(hashCapacity);

            vector<int> pathIndexv(totalSegments, 0);
            vector<int> cellv(totalSegments, -1);
            vector<double> dsv(totalSegments, 0.);
            vector<double> tauExtv(totalSegments, 0.);
            size_t segmentIndex = 0;
            for (size_t pathIndex = 0; pathIndex != paths.size(); ++pathIndex)
            {
                for (const auto& segment : paths[pathIndex]->segments())
                {
                    int cell = segment.m();
                    if (cell > maxCellForKey) return false;
                    pathIndexv[segmentIndex] = static_cast<int>(pathIndex);
                    cellv[segmentIndex] = cell;
                    dsv[segmentIndex] = segment.ds();
                    tauExtv[segmentIndex] = segment.tauExt();
                    ++segmentIndex;
                }
            }

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            int numSegments = static_cast<int>(totalSegments);
            CUdeviceptr dPathOffsetv = 0;
            CUdeviceptr dPathIndexv = 0;
            CUdeviceptr dCellv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dTauExtv = 0;
            CUdeviceptr dLuminosityv = 0;
            CUdeviceptr dWavelengthBinv = 0;
            CUdeviceptr dKeyv = 0;
            CUdeviceptr dSumv = 0;
            CUdeviceptr dCountv = 0;
            CUdeviceptr dCompactKeyv = 0;
            CUdeviceptr dCompactSumv = 0;

            auto freeAll = [&]() {
                if (dPathOffsetv) _cuMemFree(dPathOffsetv);
                if (dPathIndexv) _cuMemFree(dPathIndexv);
                if (dCellv) _cuMemFree(dCellv);
                if (dDsv) _cuMemFree(dDsv);
                if (dTauExtv) _cuMemFree(dTauExtv);
                if (dLuminosityv) _cuMemFree(dLuminosityv);
                if (dWavelengthBinv) _cuMemFree(dWavelengthBinv);
                if (dKeyv) _cuMemFree(dKeyv);
                if (dSumv) _cuMemFree(dSumv);
                if (dCountv) _cuMemFree(dCountv);
                if (dCompactKeyv) _cuMemFree(dCompactKeyv);
                if (dCompactSumv) _cuMemFree(dCompactSumv);
            };

            bool ok = allocateAndCopy(dPathOffsetv, pathOffsetv.data(), pathOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dPathIndexv, pathIndexv.data(), pathIndexv.size() * sizeof(int))
                      && allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dDsv, dsv.data(), dsv.size() * sizeof(double))
                      && allocateAndCopy(dTauExtv, tauExtv.data(), tauExtv.size() * sizeof(double))
                      && allocateAndCopy(dLuminosityv, luminosityv.data(), luminosityv.size() * sizeof(double))
                      && allocateAndCopy(dWavelengthBinv, wavelengthBinv.data(),
                                         wavelengthBinv.size() * sizeof(int))
                      && allocate(dKeyv, hashCapacity * sizeof(int))
                      && allocate(dSumv, hashCapacity * sizeof(double))
                      && allocate(dCountv, sizeof(int))
                      && allocate(dCompactKeyv, totalSegments * sizeof(int))
                      && allocate(dCompactSumv, totalSegments * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dPathOffsetv, &dPathIndexv, &dCellv,        &dDsv,
                            &dTauExtv,     &dLuminosityv, &dWavelengthBinv, &numWavelengths,
                            &numSegments,  &hashCapacityInt, &dKeyv,     &dSumv};
            unsigned int blockSize = 256;
            unsigned int gridSize = static_cast<unsigned int>((totalSegments + blockSize - 1) / blockSize);
            unsigned int compactGridSize = static_cast<unsigned int>((hashCapacity + blockSize - 1) / blockSize);
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuMemsetD32(dKeyv, 0xffffffffU, hashCapacity), "cuMemsetD32(radiation sum keys)")
                 && check(_cuMemsetD32(dSumv, 0, hashCapacity * 2), "cuMemsetD32(radiation sums)")
                 && check(_cuMemsetD32(dCountv, 0, 1), "cuMemsetD32(radiation sum count)")
                 && check(_cuLaunchKernel(_kernelRadiationFieldSumBatch, gridSize, 1, 1, blockSize, 1, 1, 0,
                                          nullptr, args, nullptr),
                          "cuLaunchKernel(radiation_field_contribution_sums_batch)");
            void* compactArgs[] = {&dKeyv, &dSumv, &hashCapacityInt, &dCountv, &dCompactKeyv, &dCompactSumv};
            int compactCount = 0;
            ok = ok && check(_cuLaunchKernel(_kernelRadiationFieldCompactSums, compactGridSize, 1, 1, blockSize,
                                             1, 1, 0, nullptr, compactArgs, nullptr),
                             "cuLaunchKernel(radiation_field_compact_sums)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(&compactCount, dCountv, sizeof(int)),
                          "cuMemcpyDtoH(batch radiation field sum count)");
            if (!ok || compactCount < 0 || static_cast<size_t>(compactCount) > totalSegments)
            {
                freeAll();
                return false;
            }

            binIndexv.assign(static_cast<size_t>(compactCount), 0);
            Ldsv.assign(static_cast<size_t>(compactCount), 0.);
            if (compactCount == 0)
            {
                freeAll();
                return true;
            }
            ok = check(_cuMemcpyDtoH(binIndexv.data(), dCompactKeyv, binIndexv.size() * sizeof(int)),
                       "cuMemcpyDtoH(batch radiation field compact keys)")
                 && check(_cuMemcpyDtoH(Ldsv.data(), dCompactSumv, Ldsv.size() * sizeof(double)),
                          "cuMemcpyDtoH(batch radiation field compact sums)");

            freeAll();
            return ok;
        }

        bool computeValueSumsByKey(const vector<int>& keyv, const vector<double>& valuev,
                                   vector<int>& compactKeyv, vector<double>& compactValuev)
        {
            if (keyv.empty() || keyv.size() != valuev.size()) return false;
            if (keyv.size() > static_cast<size_t>(std::numeric_limits<int>::max()) / 2) return false;

            size_t numValuesHost = keyv.size();
            size_t hashCapacity = nextPowerOfTwo(numValuesHost * 2);
            if (hashCapacity > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            int numValues = static_cast<int>(numValuesHost);
            int hashCapacityInt = static_cast<int>(hashCapacity);

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            CUdeviceptr dInputKeyv = 0;
            CUdeviceptr dInputValuev = 0;
            CUdeviceptr dKeyv = 0;
            CUdeviceptr dSumv = 0;
            CUdeviceptr dCountv = 0;
            CUdeviceptr dCompactKeyv = 0;
            CUdeviceptr dCompactSumv = 0;

            auto freeAll = [&]() {
                if (dInputKeyv) _cuMemFree(dInputKeyv);
                if (dInputValuev) _cuMemFree(dInputValuev);
                if (dKeyv) _cuMemFree(dKeyv);
                if (dSumv) _cuMemFree(dSumv);
                if (dCountv) _cuMemFree(dCountv);
                if (dCompactKeyv) _cuMemFree(dCompactKeyv);
                if (dCompactSumv) _cuMemFree(dCompactSumv);
            };

            bool ok = allocateAndCopy(dInputKeyv, keyv.data(), keyv.size() * sizeof(int))
                      && allocateAndCopy(dInputValuev, valuev.data(), valuev.size() * sizeof(double))
                      && allocate(dKeyv, hashCapacity * sizeof(int))
                      && allocate(dSumv, hashCapacity * sizeof(double))
                      && allocate(dCountv, sizeof(int))
                      && allocate(dCompactKeyv, keyv.size() * sizeof(int))
                      && allocate(dCompactSumv, valuev.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dInputKeyv, &dInputValuev, &numValues, &hashCapacityInt, &dKeyv, &dSumv};
            unsigned int blockSize = 256;
            unsigned int gridSize = static_cast<unsigned int>((numValuesHost + blockSize - 1) / blockSize);
            unsigned int compactGridSize = static_cast<unsigned int>((hashCapacity + blockSize - 1) / blockSize);
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuMemsetD32(dKeyv, 0xffffffffU, hashCapacity), "cuMemsetD32(key sums keys)")
                 && check(_cuMemsetD32(dSumv, 0, hashCapacity * 2), "cuMemsetD32(key sums)")
                 && check(_cuMemsetD32(dCountv, 0, 1), "cuMemsetD32(key sum count)")
                 && check(_cuLaunchKernel(_kernelSumKeyValues, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args,
                                          nullptr),
                          "cuLaunchKernel(sum_key_values)");
            void* compactArgs[] = {&dKeyv, &dSumv, &hashCapacityInt, &dCountv, &dCompactKeyv, &dCompactSumv};
            int compactCount = 0;
            ok = ok && check(_cuLaunchKernel(_kernelRadiationFieldCompactSums, compactGridSize, 1, 1, blockSize,
                                             1, 1, 0, nullptr, compactArgs, nullptr),
                             "cuLaunchKernel(radiation_field_compact_sums)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(&compactCount, dCountv, sizeof(int)), "cuMemcpyDtoH(key sum count)");
            if (!ok || compactCount < 0 || static_cast<size_t>(compactCount) > keyv.size())
            {
                freeAll();
                return false;
            }

            compactKeyv.assign(static_cast<size_t>(compactCount), 0);
            compactValuev.assign(static_cast<size_t>(compactCount), 0.);
            if (compactCount == 0)
            {
                freeAll();
                return true;
            }
            ok = check(_cuMemcpyDtoH(compactKeyv.data(), dCompactKeyv, compactKeyv.size() * sizeof(int)),
                       "cuMemcpyDtoH(compact keys)")
                 && check(_cuMemcpyDtoH(compactValuev.data(), dCompactSumv,
                                        compactValuev.size() * sizeof(double)),
                          "cuMemcpyDtoH(compact values)");

            freeAll();
            return ok;
        }

        bool accumulateValuesByKey(const void* accumulatorKey, size_t numAccumulatorValues,
                                   const vector<int>& keyv, const vector<double>& valuev)
        {
            if (!accumulatorKey || !numAccumulatorValues || keyv.empty() || keyv.size() != valuev.size())
                return false;
            if (keyv.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            if (numAccumulatorValues > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            size_t numValuesHost = keyv.size();
            int numValues = static_cast<int>(numValuesHost);
            int numAccumulatorValuesInt = static_cast<int>(numAccumulatorValues);

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            CUdeviceptr dAccumulator = 0;
            if (!ensureValueAccumulator(accumulatorKey, numAccumulatorValues, dAccumulator)) return false;

            CUdeviceptr dInputKeyv = 0;
            CUdeviceptr dInputValuev = 0;
            auto freeAll = [&]() {
                if (dInputKeyv) _cuMemFree(dInputKeyv);
                if (dInputValuev) _cuMemFree(dInputValuev);
            };

            bool ok = allocateAndCopy(dInputKeyv, keyv.data(), keyv.size() * sizeof(int))
                      && allocateAndCopy(dInputValuev, valuev.data(), valuev.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dInputKeyv, &dInputValuev, &numValues, &dAccumulator, &numAccumulatorValuesInt};
            unsigned int blockSize = 256;
            unsigned int gridSize = static_cast<unsigned int>((numValuesHost + blockSize - 1) / blockSize);
            ok = check(_cuLaunchKernel(_kernelAccumulateValuesByKey, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr,
                                       args, nullptr),
                       "cuLaunchKernel(accumulate_values_by_key)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize");

            freeAll();
            return ok;
        }

        bool retrieveAndClearValueAccumulator(const void* accumulatorKey, double* values, size_t numValues)
        {
            if (!accumulatorKey || !values || !numValues) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            auto iterator = _valueAccumulators.find(accumulatorKey);
            if (iterator == _valueAccumulators.end()) return true;

            DeviceBuffer& accumulator = iterator->second;
            size_t accumulatorBytes = numValues * sizeof(double);
            if (accumulator.bytes != accumulatorBytes)
            {
                releaseDeviceBuffer(accumulator);
                _valueAccumulators.erase(iterator);
                _error = "value accumulator size mismatch";
                return false;
            }

            vector<double> partial(numValues);
            bool ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                      && check(_cuMemcpyDtoH(partial.data(), accumulator.ptr, accumulatorBytes),
                               "cuMemcpyDtoH(value accumulator)");
            if (ok)
            {
                for (size_t i = 0; i != numValues; ++i) values[i] += partial[i];
            }
            releaseDeviceBuffer(accumulator);
            _valueAccumulators.erase(iterator);
            return ok;
        }

        void clearValueAccumulator(const void* accumulatorKey)
        {
            if (!accumulatorKey) return;

            std::lock_guard<std::mutex> lock(_mutex);
            auto iterator = _valueAccumulators.find(accumulatorKey);
            if (iterator == _valueAccumulators.end()) return;
            releaseDeviceBuffer(iterator->second);
            _valueAccumulators.erase(iterator);
        }

        bool computeFrameBandTotalFluxSums(
            const vector<Position>& positionv, const vector<double>& wavelengthv, const vector<double>& luminosityv,
            const vector<double>& tauv, bool hasMedium, double costheta, double sintheta, double cosphi,
            double sinphi, double cosomega, double sinomega, int numPixelsX, int numPixelsY, double xpmin,
            double xpsiz, double ypmin, double ypsiz, double redshift, size_t numPixelsInFrame,
            const vector<int>& bandOffsetv, const vector<double>& bandWavelengthv,
            const vector<double>& bandTransmissionv, const vector<double>& bandWidthv,
            vector<int>& compactKeyv, vector<double>& compactValuev,
            const void* accumulatorKey = nullptr, size_t numAccumulatorValues = 0)
        {
            bool accumulateOnGpu = accumulatorKey && numAccumulatorValues > 0;
            if (positionv.empty() || wavelengthv.size() != positionv.size()
                || luminosityv.size() != positionv.size() || numPixelsX <= 0 || numPixelsY <= 0
                || xpsiz <= 0.0 || ypsiz <= 0.0 || bandWidthv.empty()
                || bandOffsetv.size() != bandWidthv.size() + 1
                || bandWavelengthv.size() != bandTransmissionv.size())
                return false;
            if (hasMedium && tauv.size() != positionv.size()) return false;
            if (positionv.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            if (bandWidthv.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            if (numPixelsInFrame > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            if (accumulateOnGpu && numAccumulatorValues > static_cast<size_t>(std::numeric_limits<int>::max()))
                return false;

            for (size_t ell = 0; ell != bandWidthv.size(); ++ell)
            {
                if (bandOffsetv[ell] < 0 || bandOffsetv[ell + 1] < bandOffsetv[ell]) return false;
                if (static_cast<size_t>(bandOffsetv[ell + 1]) > bandWavelengthv.size()) return false;
                if (bandOffsetv[ell + 1] - bandOffsetv[ell] < 2) return false;
                if (bandWidthv[ell] <= 0.0) return false;
            }

            size_t numPacketsHost = positionv.size();
            size_t numBandsHost = bandWidthv.size();
            if (numPacketsHost > static_cast<size_t>(std::numeric_limits<int>::max()) / numBandsHost) return false;
            if (numPixelsInFrame > static_cast<size_t>(std::numeric_limits<int>::max()) / numBandsHost)
                return false;
            size_t numValuesHost = numPacketsHost * numBandsHost;
            if (numValuesHost > static_cast<size_t>(std::numeric_limits<int>::max()) / 2) return false;
            size_t hashCapacity = nextPowerOfTwo(numValuesHost * 2);
            if (hashCapacity > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            int numPackets = static_cast<int>(numPacketsHost);
            int numBands = static_cast<int>(numBandsHost);
            int numValues = static_cast<int>(numValuesHost);
            int hashCapacityInt = static_cast<int>(hashCapacity);
            int hasMediumInt = hasMedium ? 1 : 0;
            int numPixelsInFrameInt = static_cast<int>(numPixelsInFrame);

            vector<double> positionData(3 * numPacketsHost);
            for (size_t p = 0; p != numPacketsHost; ++p)
            {
                double x, y, z;
                positionv[p].cartesian(x, y, z);
                size_t base = 3 * p;
                positionData[base] = x;
                positionData[base + 1] = y;
                positionData[base + 2] = z;
            }

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            CUdeviceptr dPositionv = 0;
            CUdeviceptr dWavelengthv = 0;
            CUdeviceptr dLuminosityv = 0;
            CUdeviceptr dTauv = 0;
            CUdeviceptr dBandOffsetv = 0;
            CUdeviceptr dBandWavelengthv = 0;
            CUdeviceptr dBandTransmissionv = 0;
            CUdeviceptr dBandWidthv = 0;
            CUdeviceptr dInputKeyv = 0;
            CUdeviceptr dInputValuev = 0;
            CUdeviceptr dKeyv = 0;
            CUdeviceptr dSumv = 0;
            CUdeviceptr dCountv = 0;
            CUdeviceptr dCompactKeyv = 0;
            CUdeviceptr dCompactSumv = 0;
            CUdeviceptr dAccumulator = 0;

            bool ok = copyResidentScratch(ResidentPosition, dPositionv, positionData.data(),
                                          positionData.size() * sizeof(double),
                                          "cuMemcpyHtoD(frame detector positions)")
                      && copyResidentScratch(ResidentLambda, dWavelengthv, wavelengthv.data(),
                                             wavelengthv.size() * sizeof(double),
                                             "cuMemcpyHtoD(frame detector wavelengths)")
                      && copyResidentScratch(ResidentLuminosity, dLuminosityv, luminosityv.data(),
                                             luminosityv.size() * sizeof(double),
                                             "cuMemcpyHtoD(frame detector luminosities)")
                      && copyResidentScratch(ResidentLookupBegin, dBandOffsetv, bandOffsetv.data(),
                                             bandOffsetv.size() * sizeof(int),
                                             "cuMemcpyHtoD(frame detector band offsets)")
                      && copyResidentScratch(ResidentLookupWavelength, dBandWavelengthv,
                                             bandWavelengthv.data(), bandWavelengthv.size() * sizeof(double),
                                             "cuMemcpyHtoD(frame detector band wavelengths)")
                      && copyResidentScratch(ResidentSectionSca, dBandTransmissionv,
                                             bandTransmissionv.data(), bandTransmissionv.size() * sizeof(double),
                                             "cuMemcpyHtoD(frame detector band transmissions)")
                      && copyResidentScratch(ResidentSectionExt, dBandWidthv, bandWidthv.data(),
                                             bandWidthv.size() * sizeof(double),
                                             "cuMemcpyHtoD(frame detector band widths)")
                      && ensureResidentScratch(ResidentCell, dInputKeyv, numValuesHost * sizeof(int),
                                               "cuMemAlloc(frame detector input keys)")
                      && ensureResidentScratch(ResidentDistance, dInputValuev, numValuesHost * sizeof(double),
                                               "cuMemAlloc(frame detector input values)")
                      && ensureResidentScratch(ResidentKey, dKeyv, hashCapacity * sizeof(int),
                                               "cuMemAlloc(frame detector keys)")
                      && ensureResidentScratch(ResidentSum, dSumv, hashCapacity * sizeof(double),
                                               "cuMemAlloc(frame detector sums)")
                      && ensureResidentScratch(ResidentCompactCount, dCountv, sizeof(int),
                                               "cuMemAlloc(frame detector count)")
                      && ensureResidentScratch(ResidentCompactKey, dCompactKeyv, numValuesHost * sizeof(int),
                                               "cuMemAlloc(frame detector compact keys)")
                      && ensureResidentScratch(ResidentCompactSum, dCompactSumv, numValuesHost * sizeof(double),
                                               "cuMemAlloc(frame detector compact sums)");
            if (hasMedium)
                ok = ok && copyResidentScratch(ResidentTau, dTauv, tauv.data(), tauv.size() * sizeof(double),
                                               "cuMemcpyHtoD(frame detector optical depths)");
            if (!ok) return false;

            void* buildArgs[] = {&dPositionv,
                                 &dWavelengthv,
                                 &dLuminosityv,
                                 &dTauv,
                                 &hasMediumInt,
                                 &numPackets,
                                 &numBands,
                                 &costheta,
                                 &sintheta,
                                 &cosphi,
                                 &sinphi,
                                 &cosomega,
                                 &sinomega,
                                 &numPixelsX,
                                 &numPixelsY,
                                 &xpmin,
                                 &xpsiz,
                                 &ypmin,
                                 &ypsiz,
                                 &redshift,
                                 &numPixelsInFrameInt,
                                 &dBandOffsetv,
                                 &dBandWavelengthv,
                                 &dBandTransmissionv,
                                 &dBandWidthv,
                                 &dInputKeyv,
                                 &dInputValuev};
            void* sumArgs[] = {&dInputKeyv, &dInputValuev, &numValues, &hashCapacityInt, &dKeyv, &dSumv};
            void* compactArgs[] = {&dKeyv, &dSumv, &hashCapacityInt, &dCountv, &dCompactKeyv, &dCompactSumv};
            unsigned int blockSize = 256;
            unsigned int valueGridSize = static_cast<unsigned int>((numValuesHost + blockSize - 1) / blockSize);
            unsigned int compactGridSize = static_cast<unsigned int>((hashCapacity + blockSize - 1) / blockSize);
            int compactCount = 0;
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuMemsetD32(dKeyv, 0xffffffffU, hashCapacity), "cuMemsetD32(frame detector keys)")
                 && check(_cuMemsetD32(dSumv, 0, hashCapacity * 2), "cuMemsetD32(frame detector sums)")
                 && check(_cuMemsetD32(dCountv, 0, 1), "cuMemsetD32(frame detector sum count)")
                 && check(_cuLaunchKernel(_kernelFrameBandTotalFluxValues, valueGridSize, 1, 1, blockSize, 1, 1,
                                          0, nullptr, buildArgs, nullptr),
                          "cuLaunchKernel(frame_band_total_flux_values)")
                 && check(_cuLaunchKernel(_kernelSumKeyValues, valueGridSize, 1, 1, blockSize, 1, 1, 0, nullptr,
                                          sumArgs, nullptr),
                          "cuLaunchKernel(sum_key_values frame detector)")
                 && check(_cuLaunchKernel(_kernelRadiationFieldCompactSums, compactGridSize, 1, 1, blockSize, 1, 1,
                                          0, nullptr, compactArgs, nullptr),
                          "cuLaunchKernel(radiation_field_compact_sums frame detector)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(&compactCount, dCountv, sizeof(int)),
                          "cuMemcpyDtoH(frame detector compact count)");
            if (!ok || compactCount < 0 || static_cast<size_t>(compactCount) > numValuesHost)
            {
                return false;
            }

            if (accumulateOnGpu)
            {
                if (!ensureValueAccumulator(accumulatorKey, numAccumulatorValues, dAccumulator))
                {
                    return false;
                }
                if (compactCount)
                {
                    int numAccumulatorValuesInt = static_cast<int>(numAccumulatorValues);
                    unsigned int accumulateGridSize = static_cast<unsigned int>(
                        (static_cast<size_t>(compactCount) + blockSize - 1) / blockSize);
                    void* accumulateArgs[] = {&dCompactKeyv, &dCompactSumv, &compactCount, &dAccumulator,
                                              &numAccumulatorValuesInt};
                    ok = check(_cuLaunchKernel(_kernelAccumulateValuesByKey, accumulateGridSize, 1, 1,
                                               blockSize, 1, 1, 0, nullptr, accumulateArgs, nullptr),
                               "cuLaunchKernel(frame detector accumulate_values_by_key)")
                         && check(_cuCtxSynchronize(), "cuCtxSynchronize");
                }
                compactKeyv.clear();
                compactValuev.clear();
                return ok;
            }

            compactKeyv.assign(static_cast<size_t>(compactCount), 0);
            compactValuev.assign(static_cast<size_t>(compactCount), 0.);
            if (compactCount)
            {
                ok = check(_cuMemcpyDtoH(compactKeyv.data(), dCompactKeyv, compactKeyv.size() * sizeof(int)),
                           "cuMemcpyDtoH(frame detector compact keys)")
                     && check(_cuMemcpyDtoH(compactValuev.data(), dCompactSumv,
                                            compactValuev.size() * sizeof(double)),
                              "cuMemcpyDtoH(frame detector compact sums)");
            }

            return ok;
        }

        bool computeDustAbsorbedLuminosities(const MediumState& state, int numCells, int numWavelengths,
                                             const vector<int>& dustMedia, const vector<double>& sectionAbsv,
                                             const double* primaryRadiationField,
                                             const double* secondaryRadiationField,
                                             vector<double>& primaryLuminosities,
                                             vector<double>& secondaryLuminosities)
        {
            if (numCells <= 0 || numWavelengths <= 0 || dustMedia.empty() || !primaryRadiationField) return false;
            int numDustMedia = static_cast<int>(dustMedia.size());
            if (sectionAbsv.size() != static_cast<size_t>(numDustMedia) * static_cast<size_t>(numWavelengths))
                return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;

            int numVars = state.numVars();
            size_t rfSize = static_cast<size_t>(numCells) * static_cast<size_t>(numWavelengths);
            vector<int> densityOffsetv(numDustMedia);
            for (int j = 0; j != numDustMedia; ++j) densityOffsetv[j] = state.numberDensityOffset(dustMedia[j]);
            primaryLuminosities.assign(numCells, 0.);
            secondaryLuminosities.assign(numCells, 0.);

            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dSectionAbsv = 0;
            CUdeviceptr dRf1v = 0;
            CUdeviceptr dRf2v = 0;
            CUdeviceptr dOut1v = 0;
            CUdeviceptr dOut2v = 0;

            auto freeAll = [&]() {
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dSectionAbsv) _cuMemFree(dSectionAbsv);
                if (dRf1v) _cuMemFree(dRf1v);
                if (dRf2v) _cuMemFree(dRf2v);
                if (dOut1v) _cuMemFree(dOut1v);
                if (dOut2v) _cuMemFree(dOut2v);
            };

            bool ok = allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dSectionAbsv, sectionAbsv.data(), sectionAbsv.size() * sizeof(double))
                      && allocateAndCopy(dRf1v, primaryRadiationField, rfSize * sizeof(double))
                      && allocate(dOut1v, primaryLuminosities.size() * sizeof(double))
                      && allocate(dOut2v, secondaryLuminosities.size() * sizeof(double));
            int hasRf2 = secondaryRadiationField ? 1 : 0;
            if (hasRf2) ok = ok && allocateAndCopy(dRf2v, secondaryRadiationField, rfSize * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&_stateDevice, &numCells, &numWavelengths, &numVars, &numDustMedia, &dDensityOffsetv,
                            &dSectionAbsv, &dRf1v, &dRf2v, &hasRf2, &dOut1v, &dOut2v};
            unsigned int blockSize = 128;
            unsigned int gridSize = static_cast<unsigned int>((static_cast<size_t>(numCells) + blockSize - 1) / blockSize);
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelDustAbsorbedLuminosity, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr,
                                          args, nullptr),
                          "cuLaunchKernel(dust_absorbed_luminosity)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(primaryLuminosities.data(), dOut1v,
                                        primaryLuminosities.size() * sizeof(double)),
                          "cuMemcpyDtoH(dust primary luminosities)")
                 && check(_cuMemcpyDtoH(secondaryLuminosities.data(), dOut2v,
                                        secondaryLuminosities.size() * sizeof(double)),
                          "cuMemcpyDtoH(dust secondary luminosities)");

            freeAll();
            return ok;
        }

        bool computeTotalDustAbsorbedLuminosity(const MediumState& state, int numCells, int numWavelengths,
                                                const vector<int>& dustMedia, const vector<double>& sectionAbsv,
                                                const double* primaryRadiationField,
                                                const double* secondaryRadiationField, double& primaryLuminosity,
                                                double& secondaryLuminosity)
        {
            primaryLuminosity = 0.;
            secondaryLuminosity = 0.;
            if (numCells <= 0 || numWavelengths <= 0 || dustMedia.empty() || !primaryRadiationField) return false;
            int numDustMedia = static_cast<int>(dustMedia.size());
            if (sectionAbsv.size() != static_cast<size_t>(numDustMedia) * static_cast<size_t>(numWavelengths))
                return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;

            int numVars = state.numVars();
            size_t rfSize = static_cast<size_t>(numCells) * static_cast<size_t>(numWavelengths);
            vector<int> densityOffsetv(numDustMedia);
            for (int j = 0; j != numDustMedia; ++j) densityOffsetv[j] = state.numberDensityOffset(dustMedia[j]);

            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dSectionAbsv = 0;
            CUdeviceptr dRf1v = 0;
            CUdeviceptr dRf2v = 0;
            CUdeviceptr dOut1v = 0;
            CUdeviceptr dOut2v = 0;
            CUdeviceptr dTotalv = 0;

            auto freeAll = [&]() {
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dSectionAbsv) _cuMemFree(dSectionAbsv);
                if (dRf1v) _cuMemFree(dRf1v);
                if (dRf2v) _cuMemFree(dRf2v);
                if (dOut1v) _cuMemFree(dOut1v);
                if (dOut2v) _cuMemFree(dOut2v);
                if (dTotalv) _cuMemFree(dTotalv);
            };

            bool ok = allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dSectionAbsv, sectionAbsv.data(), sectionAbsv.size() * sizeof(double))
                      && allocateAndCopy(dRf1v, primaryRadiationField, rfSize * sizeof(double))
                      && allocate(dOut1v, static_cast<size_t>(numCells) * sizeof(double))
                      && allocate(dOut2v, static_cast<size_t>(numCells) * sizeof(double))
                      && allocate(dTotalv, 2 * sizeof(double));
            int hasRf2 = secondaryRadiationField ? 1 : 0;
            if (hasRf2) ok = ok && allocateAndCopy(dRf2v, secondaryRadiationField, rfSize * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&_stateDevice, &numCells, &numWavelengths, &numVars, &numDustMedia, &dDensityOffsetv,
                            &dSectionAbsv, &dRf1v, &dRf2v, &hasRf2, &dOut1v, &dOut2v};
            unsigned int blockSize = 128;
            unsigned int gridSize =
                static_cast<unsigned int>((static_cast<size_t>(numCells) + blockSize - 1) / blockSize);
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelDustAbsorbedLuminosity, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr,
                                          args, nullptr),
                          "cuLaunchKernel(dust_absorbed_luminosity)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize");
            if (ok)
            {
                void* sumArgs[] = {&dOut1v, &dOut2v, &numCells, &dTotalv};
                double totals[2] = {0., 0.};
                ok = check(_cuLaunchKernel(_kernelDustAbsorbedLuminositySum, 1, 1, 1, 1, 1, 1, 0, nullptr, sumArgs,
                                           nullptr),
                           "cuLaunchKernel(dust_absorbed_luminosity_sum)")
                     && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                     && check(_cuMemcpyDtoH(totals, dTotalv, 2 * sizeof(double)),
                              "cuMemcpyDtoH(total dust luminosities)");
                if (ok)
                {
                    primaryLuminosity = totals[0];
                    secondaryLuminosity = totals[1];
                }
            }

            freeAll();
            return ok;
        }

        bool computeScatteringProperties(const MediumState& state, int cellIndex, const vector<double>& sectionScav,
                                         const vector<double>& sectionExtv, double& albedo, vector<double>& weights)
        {
            if (cellIndex < 0 || sectionScav.empty() || sectionScav.size() != sectionExtv.size()) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;

            int numMedia = static_cast<int>(sectionScav.size());
            int numVars = state.numVars();
            vector<int> densityOffsetv(numMedia);
            for (int h = 0; h != numMedia; ++h) densityOffsetv[h] = state.numberDensityOffset(h);
            weights.assign(numMedia, 0.);
            albedo = 0.;

            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dSectionScav = 0;
            CUdeviceptr dSectionExtv = 0;
            CUdeviceptr dAlbedov = 0;
            CUdeviceptr dWeightv = 0;

            auto freeAll = [&]() {
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dSectionScav) _cuMemFree(dSectionScav);
                if (dSectionExtv) _cuMemFree(dSectionExtv);
                if (dAlbedov) _cuMemFree(dAlbedov);
                if (dWeightv) _cuMemFree(dWeightv);
            };

            bool ok = allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dSectionScav, sectionScav.data(), sectionScav.size() * sizeof(double))
                      && allocateAndCopy(dSectionExtv, sectionExtv.data(), sectionExtv.size() * sizeof(double))
                      && allocate(dAlbedov, sizeof(double)) && allocate(dWeightv, weights.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&_stateDevice, &cellIndex, &numVars, &numMedia, &dDensityOffsetv,
                            &dSectionScav, &dSectionExtv, &dAlbedov, &dWeightv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelScatteringProperties, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                          "cuLaunchKernel(scattering_properties)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(&albedo, dAlbedov, sizeof(double)), "cuMemcpyDtoH(scattering albedo)")
                 && check(_cuMemcpyDtoH(weights.data(), dWeightv, weights.size() * sizeof(double)),
                          "cuMemcpyDtoH(scattering weights)");

            freeAll();
            return ok;
        }

        bool computeTableScatteringProperties(const MediumState& state, int cellIndex,
                                              const vector<int>& mediaIndexv,
                                              const vector<int>& lookupBeginv,
                                              const vector<int>& lookupCountv,
                                              const vector<double>& lookupWavelengthv,
                                              const vector<double>& sectionScaTablev,
                                              const vector<double>& sectionExtTablev, double lambda,
                                              double& albedo, vector<double>& weights)
        {
            if (cellIndex < 0 || mediaIndexv.empty() || lookupBeginv.size() != mediaIndexv.size()
                || lookupCountv.size() != mediaIndexv.size() || lookupWavelengthv.empty()
                || sectionScaTablev.size() != lookupWavelengthv.size()
                || sectionExtTablev.size() != lookupWavelengthv.size())
                return false;

            for (size_t h = 0; h != mediaIndexv.size(); ++h)
            {
                if (mediaIndexv[h] < 0 || lookupBeginv[h] < 0 || lookupCountv[h] < 2) return false;
                size_t begin = static_cast<size_t>(lookupBeginv[h]);
                size_t count = static_cast<size_t>(lookupCountv[h]);
                if (begin + count > lookupWavelengthv.size()) return false;
            }

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;

            int numTables = static_cast<int>(mediaIndexv.size());
            int numVars = state.numVars();
            vector<int> densityOffsetv(numTables);
            for (int h = 0; h != numTables; ++h) densityOffsetv[h] = state.numberDensityOffset(mediaIndexv[h]);
            weights.assign(numTables, 0.);
            albedo = 0.;

            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dLookupBeginv = 0;
            CUdeviceptr dLookupCountv = 0;
            CUdeviceptr dLookupWavelengthv = 0;
            CUdeviceptr dSectionScaTablev = 0;
            CUdeviceptr dSectionExtTablev = 0;
            CUdeviceptr dAlbedov = 0;
            CUdeviceptr dWeightv = 0;

            auto freeAll = [&]() {
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dLookupBeginv) _cuMemFree(dLookupBeginv);
                if (dLookupCountv) _cuMemFree(dLookupCountv);
                if (dLookupWavelengthv) _cuMemFree(dLookupWavelengthv);
                if (dSectionScaTablev) _cuMemFree(dSectionScaTablev);
                if (dSectionExtTablev) _cuMemFree(dSectionExtTablev);
                if (dAlbedov) _cuMemFree(dAlbedov);
                if (dWeightv) _cuMemFree(dWeightv);
            };

            bool ok = allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dLookupBeginv, lookupBeginv.data(), lookupBeginv.size() * sizeof(int))
                      && allocateAndCopy(dLookupCountv, lookupCountv.data(), lookupCountv.size() * sizeof(int))
                      && allocateAndCopy(dLookupWavelengthv, lookupWavelengthv.data(),
                                         lookupWavelengthv.size() * sizeof(double))
                      && allocateAndCopy(dSectionScaTablev, sectionScaTablev.data(),
                                         sectionScaTablev.size() * sizeof(double))
                      && allocateAndCopy(dSectionExtTablev, sectionExtTablev.data(),
                                         sectionExtTablev.size() * sizeof(double))
                      && allocate(dAlbedov, sizeof(double)) && allocate(dWeightv, weights.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&_stateDevice,        &cellIndex,       &numVars,       &numTables,
                            &dDensityOffsetv,     &dLookupBeginv,   &dLookupCountv, &dLookupWavelengthv,
                            &dSectionScaTablev,   &dSectionExtTablev, &lambda,      &dAlbedov,
                            &dWeightv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelTableScatteringProperties, 1, 1, 1, 1, 1, 1, 0, nullptr, args,
                                          nullptr),
                          "cuLaunchKernel(table_scattering_properties)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(&albedo, dAlbedov, sizeof(double)), "cuMemcpyDtoH(table scattering albedo)")
                 && check(_cuMemcpyDtoH(weights.data(), dWeightv, weights.size() * sizeof(double)),
                          "cuMemcpyDtoH(table scattering weights)");

            freeAll();
            return ok;
        }

        bool computeScatteringAlbedos(const MediumState& state, const vector<int>& cellv,
                                      const vector<double>& sectionScav, const vector<double>& sectionExtv,
                                      vector<double>& albedov)
        {
            if (cellv.empty() || sectionScav.empty() || sectionScav.size() != sectionExtv.size()) return false;
            if (cellv.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;

            int numPaths = static_cast<int>(cellv.size());
            int numMedia = static_cast<int>(sectionScav.size());
            int numVars = state.numVars();
            vector<int> densityOffsetv(numMedia);
            for (int h = 0; h != numMedia; ++h) densityOffsetv[h] = state.numberDensityOffset(h);
            albedov.assign(cellv.size(), 0.);

            CUdeviceptr dCellv = 0;
            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dSectionScav = 0;
            CUdeviceptr dSectionExtv = 0;
            CUdeviceptr dAlbedov = 0;

            auto freeAll = [&]() {
                if (dCellv) _cuMemFree(dCellv);
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dSectionScav) _cuMemFree(dSectionScav);
                if (dSectionExtv) _cuMemFree(dSectionExtv);
                if (dAlbedov) _cuMemFree(dAlbedov);
            };

            bool ok = allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dSectionScav, sectionScav.data(), sectionScav.size() * sizeof(double))
                      && allocateAndCopy(dSectionExtv, sectionExtv.data(), sectionExtv.size() * sizeof(double))
                      && allocate(dAlbedov, albedov.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            unsigned int blockSize = 128;
            unsigned int gridSize = static_cast<unsigned int>((cellv.size() + blockSize - 1) / blockSize);
            void* args[] = {&_stateDevice,      &dCellv,      &numVars,      &numMedia, &dDensityOffsetv,
                            &dSectionScav,      &dSectionExtv, &numPaths,    &dAlbedov};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelScatteringAlbedos, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr,
                                          args, nullptr),
                          "cuLaunchKernel(scattering_albedos)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(albedov.data(), dAlbedov, albedov.size() * sizeof(double)),
                          "cuMemcpyDtoH(scattering albedos)");

            freeAll();
            return ok;
        }

        bool computeTableScatteringAlbedos(const MediumState& state, const vector<int>& cellv,
                                           const vector<double>& lambdav, const vector<int>& mediaIndexv,
                                           const vector<int>& lookupBeginv,
                                           const vector<int>& lookupCountv,
                                           const vector<double>& lookupWavelengthv,
                                           const vector<double>& sectionScaTablev,
                                           const vector<double>& sectionExtTablev, vector<double>& albedov)
        {
            if (cellv.empty() || lambdav.size() != cellv.size()) return false;
            if (mediaIndexv.empty() || lookupBeginv.size() != mediaIndexv.size()
                || lookupCountv.size() != mediaIndexv.size() || lookupWavelengthv.empty()
                || sectionScaTablev.size() != lookupWavelengthv.size()
                || sectionExtTablev.size() != lookupWavelengthv.size())
                return false;
            if (cellv.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            for (size_t h = 0; h != mediaIndexv.size(); ++h)
            {
                if (mediaIndexv[h] < 0 || lookupBeginv[h] < 0 || lookupCountv[h] < 2) return false;
                size_t begin = static_cast<size_t>(lookupBeginv[h]);
                size_t count = static_cast<size_t>(lookupCountv[h]);
                if (begin + count > lookupWavelengthv.size()) return false;
            }

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;

            int numPaths = static_cast<int>(cellv.size());
            int numTables = static_cast<int>(mediaIndexv.size());
            int numVars = state.numVars();
            vector<int> densityOffsetv(numTables);
            for (int h = 0; h != numTables; ++h) densityOffsetv[h] = state.numberDensityOffset(mediaIndexv[h]);
            albedov.assign(cellv.size(), 0.);

            CUdeviceptr dCellv = 0;
            CUdeviceptr dLambdav = 0;
            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dLookupBeginv = 0;
            CUdeviceptr dLookupCountv = 0;
            CUdeviceptr dLookupWavelengthv = 0;
            CUdeviceptr dSectionScaTablev = 0;
            CUdeviceptr dSectionExtTablev = 0;
            CUdeviceptr dAlbedov = 0;

            auto freeAll = [&]() {
                if (dCellv) _cuMemFree(dCellv);
                if (dLambdav) _cuMemFree(dLambdav);
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dLookupBeginv) _cuMemFree(dLookupBeginv);
                if (dLookupCountv) _cuMemFree(dLookupCountv);
                if (dLookupWavelengthv) _cuMemFree(dLookupWavelengthv);
                if (dSectionScaTablev) _cuMemFree(dSectionScaTablev);
                if (dSectionExtTablev) _cuMemFree(dSectionExtTablev);
                if (dAlbedov) _cuMemFree(dAlbedov);
            };

            bool ok = allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dLambdav, lambdav.data(), lambdav.size() * sizeof(double))
                      && allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dLookupBeginv, lookupBeginv.data(), lookupBeginv.size() * sizeof(int))
                      && allocateAndCopy(dLookupCountv, lookupCountv.data(), lookupCountv.size() * sizeof(int))
                      && allocateAndCopy(dLookupWavelengthv, lookupWavelengthv.data(),
                                         lookupWavelengthv.size() * sizeof(double))
                      && allocateAndCopy(dSectionScaTablev, sectionScaTablev.data(),
                                         sectionScaTablev.size() * sizeof(double))
                      && allocateAndCopy(dSectionExtTablev, sectionExtTablev.data(),
                                         sectionExtTablev.size() * sizeof(double))
                      && allocate(dAlbedov, albedov.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            unsigned int blockSize = 128;
            unsigned int gridSize = static_cast<unsigned int>((cellv.size() + blockSize - 1) / blockSize);
            void* args[] = {&_stateDevice,        &dCellv,          &dLambdav,       &numVars,
                            &numTables,           &dDensityOffsetv, &dLookupBeginv,  &dLookupCountv,
                            &dLookupWavelengthv,  &dSectionScaTablev, &dSectionExtTablev,
                            &numPaths,            &dAlbedov};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelTableScatteringAlbedos, gridSize, 1, 1, blockSize, 1, 1, 0,
                                          nullptr, args, nullptr),
                          "cuLaunchKernel(table_scattering_albedos)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(albedov.data(), dAlbedov, albedov.size() * sizeof(double)),
                          "cuMemcpyDtoH(table scattering albedos)");

            freeAll();
            return ok;
        }

        bool henyeyGreensteinScatteringLuminosities(const vector<double>& inputDirectionv,
                                                    const vector<double>& packetLuminosityv,
                                                    const vector<double>& lambdav, Direction bfkobs,
                                                    int lookupBegin, int lookupCount,
                                                    const vector<double>& lookupWavelengthv,
                                                    const vector<double>& asymmparv,
                                                    vector<double>& luminosityv)
        {
            if (packetLuminosityv.empty() || lambdav.size() != packetLuminosityv.size()
                || inputDirectionv.size() != 3 * packetLuminosityv.size())
                return false;
            if (lookupBegin < 0 || lookupCount < 2 || lookupWavelengthv.size() != asymmparv.size()) return false;
            size_t lookupEnd = static_cast<size_t>(lookupBegin) + static_cast<size_t>(lookupCount);
            if (lookupEnd > lookupWavelengthv.size()) return false;
            if (packetLuminosityv.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            int numPackets = static_cast<int>(packetLuminosityv.size());
            double obsx = 0.;
            double obsy = 0.;
            double obsz = 0.;
            bfkobs.cartesian(obsx, obsy, obsz);
            luminosityv.assign(packetLuminosityv.size(), 0.);

            CUdeviceptr dInputDirectionv = 0;
            CUdeviceptr dPacketLuminosityv = 0;
            CUdeviceptr dLambdav = 0;
            CUdeviceptr dLookupWavelengthv = 0;
            CUdeviceptr dAsymmparv = 0;
            CUdeviceptr dLuminosityv = 0;

            auto freeAll = [&]() {
                if (dInputDirectionv) _cuMemFree(dInputDirectionv);
                if (dPacketLuminosityv) _cuMemFree(dPacketLuminosityv);
                if (dLambdav) _cuMemFree(dLambdav);
                if (dLookupWavelengthv) _cuMemFree(dLookupWavelengthv);
                if (dAsymmparv) _cuMemFree(dAsymmparv);
                if (dLuminosityv) _cuMemFree(dLuminosityv);
            };

            bool ok = allocateAndCopy(dInputDirectionv, inputDirectionv.data(),
                                      inputDirectionv.size() * sizeof(double))
                      && allocateAndCopy(dPacketLuminosityv, packetLuminosityv.data(),
                                         packetLuminosityv.size() * sizeof(double))
                      && allocateAndCopy(dLambdav, lambdav.data(), lambdav.size() * sizeof(double))
                      && allocateAndCopy(dLookupWavelengthv, lookupWavelengthv.data(),
                                         lookupWavelengthv.size() * sizeof(double))
                      && allocateAndCopy(dAsymmparv, asymmparv.data(), asymmparv.size() * sizeof(double))
                      && allocate(dLuminosityv, luminosityv.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            unsigned int blockSize = 256;
            unsigned int gridSize = static_cast<unsigned int>((packetLuminosityv.size() + blockSize - 1) / blockSize);
            void* args[] = {&dInputDirectionv, &dPacketLuminosityv, &dLambdav, &numPackets,
                            &obsx,             &obsy,               &obsz,     &lookupBegin,
                            &lookupCount,      &dLookupWavelengthv, &dAsymmparv, &dLuminosityv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelHenyeyGreensteinScatteringLuminosities, gridSize, 1, 1,
                                          blockSize, 1, 1, 0, nullptr, args, nullptr),
                          "cuLaunchKernel(henyey_greenstein_scattering_luminosities)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(luminosityv.data(), dLuminosityv,
                                        luminosityv.size() * sizeof(double)),
                          "cuMemcpyDtoH(HG scattering luminosities)");

            freeAll();
            return ok;
        }

        bool henyeyGreensteinScatteringDirections(const vector<double>& inputDirectionv, const vector<double>& lambdav,
                                                  const vector<double>& randomCosthetav,
                                                  const vector<double>& randomPhiv, int lookupBegin, int lookupCount,
                                                  const vector<double>& lookupWavelengthv,
                                                  const vector<double>& asymmparv,
                                                  vector<double>& outputDirectionv)
        {
            if (lambdav.empty() || randomCosthetav.size() != lambdav.size() || randomPhiv.size() != lambdav.size()
                || inputDirectionv.size() != 3 * lambdav.size())
                return false;
            if (lookupBegin < 0 || lookupCount < 2 || lookupWavelengthv.size() != asymmparv.size()) return false;
            size_t lookupEnd = static_cast<size_t>(lookupBegin) + static_cast<size_t>(lookupCount);
            if (lookupEnd > lookupWavelengthv.size()) return false;
            if (lambdav.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            int numPackets = static_cast<int>(lambdav.size());
            outputDirectionv.assign(inputDirectionv.size(), 0.);

            CUdeviceptr dInputDirectionv = 0;
            CUdeviceptr dLambdav = 0;
            CUdeviceptr dRandomCosthetav = 0;
            CUdeviceptr dRandomPhiv = 0;
            CUdeviceptr dLookupWavelengthv = 0;
            CUdeviceptr dAsymmparv = 0;
            CUdeviceptr dOutputDirectionv = 0;

            auto freeAll = [&]() {
                if (dInputDirectionv) _cuMemFree(dInputDirectionv);
                if (dLambdav) _cuMemFree(dLambdav);
                if (dRandomCosthetav) _cuMemFree(dRandomCosthetav);
                if (dRandomPhiv) _cuMemFree(dRandomPhiv);
                if (dLookupWavelengthv) _cuMemFree(dLookupWavelengthv);
                if (dAsymmparv) _cuMemFree(dAsymmparv);
                if (dOutputDirectionv) _cuMemFree(dOutputDirectionv);
            };

            bool ok = allocateAndCopy(dInputDirectionv, inputDirectionv.data(),
                                      inputDirectionv.size() * sizeof(double))
                      && allocateAndCopy(dLambdav, lambdav.data(), lambdav.size() * sizeof(double))
                      && allocateAndCopy(dRandomCosthetav, randomCosthetav.data(),
                                         randomCosthetav.size() * sizeof(double))
                      && allocateAndCopy(dRandomPhiv, randomPhiv.data(), randomPhiv.size() * sizeof(double))
                      && allocateAndCopy(dLookupWavelengthv, lookupWavelengthv.data(),
                                         lookupWavelengthv.size() * sizeof(double))
                      && allocateAndCopy(dAsymmparv, asymmparv.data(), asymmparv.size() * sizeof(double))
                      && allocate(dOutputDirectionv, outputDirectionv.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            unsigned int blockSize = 256;
            unsigned int gridSize = static_cast<unsigned int>((lambdav.size() + blockSize - 1) / blockSize);
            void* args[] = {&dInputDirectionv, &dLambdav,     &dRandomCosthetav, &dRandomPhiv,
                            &numPackets,       &lookupBegin, &lookupCount,      &dLookupWavelengthv,
                            &dAsymmparv,       &dOutputDirectionv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelHenyeyGreensteinScatteringDirections, gridSize, 1, 1, blockSize, 1, 1,
                                          0, nullptr, args, nullptr),
                          "cuLaunchKernel(henyey_greenstein_scattering_directions)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(outputDirectionv.data(), dOutputDirectionv,
                                        outputDirectionv.size() * sizeof(double)),
                          "cuMemcpyDtoH(HG scattering directions)");

            freeAll();
            return ok;
        }

        bool isotropicDirections(const vector<double>& randomCosthetav, const vector<double>& randomPhiv,
                                 vector<double>& outputDirectionv)
        {
            if (randomCosthetav.empty() || randomPhiv.size() != randomCosthetav.size()) return false;
            if (randomCosthetav.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            int numPackets = static_cast<int>(randomCosthetav.size());
            outputDirectionv.assign(3 * randomCosthetav.size(), 0.);

            CUdeviceptr dRandomCosthetav = 0;
            CUdeviceptr dRandomPhiv = 0;
            CUdeviceptr dOutputDirectionv = 0;

            auto freeAll = [&]() {
                if (dRandomCosthetav) _cuMemFree(dRandomCosthetav);
                if (dRandomPhiv) _cuMemFree(dRandomPhiv);
                if (dOutputDirectionv) _cuMemFree(dOutputDirectionv);
            };

            bool ok = allocateAndCopy(dRandomCosthetav, randomCosthetav.data(),
                                      randomCosthetav.size() * sizeof(double))
                      && allocateAndCopy(dRandomPhiv, randomPhiv.data(), randomPhiv.size() * sizeof(double))
                      && allocate(dOutputDirectionv, outputDirectionv.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            unsigned int blockSize = 256;
            unsigned int gridSize = static_cast<unsigned int>((randomCosthetav.size() + blockSize - 1) / blockSize);
            void* args[] = {&dRandomCosthetav, &dRandomPhiv, &numPackets, &dOutputDirectionv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelIsotropicDirections, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr,
                                          args, nullptr),
                          "cuLaunchKernel(isotropic_directions)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(outputDirectionv.data(), dOutputDirectionv,
                                        outputDirectionv.size() * sizeof(double)),
                          "cuMemcpyDtoH(isotropic directions)");

            freeAll();
            return ok;
        }

        bool scaleWavelengthValues(double* values, size_t numWavelengths, const vector<double>& factorv)
        {
            if (!values || numWavelengths == 0 || factorv.size() != numWavelengths) return false;

            unsigned long long numValues = static_cast<unsigned long long>(numWavelengths);
            unsigned int blockSize = 256;
            unsigned long long blocks = (numValues + blockSize - 1) / blockSize;
            if (blocks > std::numeric_limits<unsigned int>::max()) return false;
            unsigned int gridSize = static_cast<unsigned int>(blocks);

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            CUdeviceptr dValues = 0;
            CUdeviceptr dFactors = 0;
            auto freeAll = [&]() {
                if (dValues) _cuMemFree(dValues);
                if (dFactors) _cuMemFree(dFactors);
            };

            bool ok = allocateAndCopy(dValues, values, numWavelengths * sizeof(double))
                      && allocateAndCopy(dFactors, factorv.data(), factorv.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dValues, &numValues, &dFactors};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelScaleWavelengthValues, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr,
                                          args, nullptr),
                          "cuLaunchKernel(scale_wavelength_values)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(values, dValues, numWavelengths * sizeof(double)),
                          "cuMemcpyDtoH(scale wavelength values)");

            freeAll();
            return ok;
        }

        bool scaleFrameWavelengthValues(double* values, size_t numWavelengths, size_t numPixelsInFrame,
                                        const vector<double>& factorv)
        {
            if (!values || numWavelengths == 0 || numPixelsInFrame == 0 || factorv.size() != numWavelengths)
                return false;

            size_t hostNumValues = numWavelengths * numPixelsInFrame;
            if (hostNumValues / numPixelsInFrame != numWavelengths) return false;
            unsigned long long numValues = static_cast<unsigned long long>(hostNumValues);
            unsigned long long numPixels = static_cast<unsigned long long>(numPixelsInFrame);
            unsigned int blockSize = 256;
            unsigned long long blocks = (numValues + blockSize - 1) / blockSize;
            if (blocks > std::numeric_limits<unsigned int>::max()) return false;
            unsigned int gridSize = static_cast<unsigned int>(blocks);

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            CUdeviceptr dValues = 0;
            CUdeviceptr dFactors = 0;
            auto freeAll = [&]() {
                if (dValues) _cuMemFree(dValues);
                if (dFactors) _cuMemFree(dFactors);
            };

            bool ok = allocateAndCopy(dValues, values, hostNumValues * sizeof(double))
                      && allocateAndCopy(dFactors, factorv.data(), factorv.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dValues, &numValues, &numPixels, &dFactors};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelScaleFrameWavelengthValues, gridSize, 1, 1, blockSize, 1, 1, 0,
                                          nullptr, args, nullptr),
                          "cuLaunchKernel(scale_frame_wavelength_values)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(values, dValues, hostNumValues * sizeof(double)),
                          "cuMemcpyDtoH(scale frame wavelength values)");

            freeAll();
            return ok;
        }

        bool divideValues(double* values, size_t numValues, double divisor)
        {
            if (!values || !numValues) return false;

            unsigned long long numDeviceValues = static_cast<unsigned long long>(numValues);
            unsigned int blockSize = 256;
            unsigned long long blocks = (numDeviceValues + blockSize - 1) / blockSize;
            if (blocks > std::numeric_limits<unsigned int>::max()) return false;
            unsigned int gridSize = static_cast<unsigned int>(blocks);

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            CUdeviceptr dValues = 0;
            auto freeAll = [&]() {
                if (dValues) _cuMemFree(dValues);
            };

            size_t bytes = numValues * sizeof(double);
            bool ok = allocateAndCopy(dValues, values, bytes);
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dValues, &numDeviceValues, &divisor};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelDivideValues, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args,
                                          nullptr),
                          "cuLaunchKernel(divide_values)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(values, dValues, bytes), "cuMemcpyDtoH(divide values)");

            freeAll();
            return ok;
        }

        bool multiplyValues(double* values, size_t numValues, double factor)
        {
            if (!values || !numValues) return false;

            unsigned long long numDeviceValues = static_cast<unsigned long long>(numValues);
            unsigned int blockSize = 256;
            unsigned long long blocks = (numDeviceValues + blockSize - 1) / blockSize;
            if (blocks > std::numeric_limits<unsigned int>::max()) return false;
            unsigned int gridSize = static_cast<unsigned int>(blocks);

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            CUdeviceptr dValues = 0;
            auto freeAll = [&]() {
                if (dValues) _cuMemFree(dValues);
            };

            size_t bytes = numValues * sizeof(double);
            bool ok = allocateAndCopy(dValues, values, bytes);
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dValues, &numDeviceValues, &factor};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelMultiplyValues, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args,
                                          nullptr),
                          "cuLaunchKernel(multiply_values)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(values, dValues, bytes), "cuMemcpyDtoH(multiply values)");

            freeAll();
            return ok;
        }

        bool sumValues(double* output, size_t numValues, const double* value1, const double* value2,
                       const double* value3, const double* value4)
        {
            if (!output || !numValues || !value1 || !value2) return false;
            int numInputs = value4 ? 4 : (value3 ? 3 : 2);

            unsigned long long numDeviceValues = static_cast<unsigned long long>(numValues);
            unsigned int blockSize = 256;
            unsigned long long blocks = (numDeviceValues + blockSize - 1) / blockSize;
            if (blocks > std::numeric_limits<unsigned int>::max()) return false;
            unsigned int gridSize = static_cast<unsigned int>(blocks);

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            CUdeviceptr dOutput = 0;
            CUdeviceptr dValue1 = 0;
            CUdeviceptr dValue2 = 0;
            CUdeviceptr dValue3 = 0;
            CUdeviceptr dValue4 = 0;
            auto freeAll = [&]() {
                if (dOutput) _cuMemFree(dOutput);
                if (dValue1) _cuMemFree(dValue1);
                if (dValue2) _cuMemFree(dValue2);
                if (dValue3) _cuMemFree(dValue3);
                if (dValue4) _cuMemFree(dValue4);
            };

            size_t bytes = numValues * sizeof(double);
            bool ok = allocate(dOutput, bytes) && allocateAndCopy(dValue1, value1, bytes)
                      && allocateAndCopy(dValue2, value2, bytes);
            if (value3) ok = ok && allocateAndCopy(dValue3, value3, bytes);
            if (value4) ok = ok && allocateAndCopy(dValue4, value4, bytes);
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dOutput, &numDeviceValues, &dValue1, &dValue2, &dValue3, &dValue4, &numInputs};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelSumValues, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args,
                                          nullptr),
                          "cuLaunchKernel(sum_values)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(output, dOutput, bytes), "cuMemcpyDtoH(sum values)");

            freeAll();
            return ok;
        }

        bool compositeLaunchWeights(const double* luminosityv, size_t numValues, double spatialBias, double* weightv)
        {
            if (!luminosityv || !weightv || !numValues) return false;

            unsigned long long numDeviceValues = static_cast<unsigned long long>(numValues);
            unsigned int blockSize = 256;
            unsigned long long blocks = (numDeviceValues + blockSize - 1) / blockSize;
            if (blocks > std::numeric_limits<unsigned int>::max()) return false;
            unsigned int gridSize = static_cast<unsigned int>(blocks);

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            CUdeviceptr dLuminosityv = 0;
            CUdeviceptr dWeightv = 0;
            CUdeviceptr dPartialCountv = 0;
            auto freeAll = [&]() {
                if (dLuminosityv) _cuMemFree(dLuminosityv);
                if (dWeightv) _cuMemFree(dWeightv);
                if (dPartialCountv) _cuMemFree(dPartialCountv);
            };

            size_t bytes = numValues * sizeof(double);
            vector<unsigned int> partialCountv(gridSize, 0);
            bool ok = allocateAndCopy(dLuminosityv, luminosityv, bytes) && allocate(dWeightv, bytes)
                      && allocate(dPartialCountv, partialCountv.size() * sizeof(unsigned int));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* countArgs[] = {&dLuminosityv, &numDeviceValues, &dPartialCountv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelEmittingCellCounts, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr,
                                          countArgs, nullptr),
                          "cuLaunchKernel(emitting_cell_counts)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(partialCountv.data(), dPartialCountv,
                                        partialCountv.size() * sizeof(unsigned int)),
                          "cuMemcpyDtoH(emitting cell counts)");
            if (!ok)
            {
                freeAll();
                return false;
            }

            unsigned long long emittingCells = 0;
            for (unsigned int count : partialCountv) emittingCells += count;
            if (!emittingCells)
            {
                freeAll();
                return false;
            }

            void* weightArgs[] = {&dLuminosityv, &numDeviceValues, &emittingCells, &spatialBias, &dWeightv};
            ok = check(_cuLaunchKernel(_kernelCompositeLaunchWeights, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr,
                                       weightArgs, nullptr),
                       "cuLaunchKernel(composite_launch_weights)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(weightv, dWeightv, bytes), "cuMemcpyDtoH(composite launch weights)");

            freeAll();
            return ok;
        }

        bool storedTableCdf(vector<double>& luminosities, int numAxes, const vector<const double*>& axisData,
                            const vector<size_t>& axisSizes, const vector<bool>& axisLog, const double* quantity,
                            size_t quantityStep, bool quantityLog, bool clampFirstAxis, double xmin, double xmax,
                            const vector<double>& parameterValues, const vector<double>& scaleValues,
                            size_t numEntities)
        {
            if (numAxes < 2 || numAxes > 5 || axisData.size() != static_cast<size_t>(numAxes)
                || axisSizes.size() != static_cast<size_t>(numAxes) || axisLog.size() != static_cast<size_t>(numAxes)
                || !quantity || !quantityStep || !numEntities || parameterValues.size() != numEntities * (numAxes - 1)
                || scaleValues.size() != numEntities)
                return false;

            size_t axisValueCount = 0;
            size_t quantityCount = 1;
            vector<int> axisOffsets(numAxes, 0);
            vector<int> axisLengthv(numAxes, 0);
            vector<int> axisLogv(numAxes, 0);
            for (int k = 0; k != numAxes; ++k)
            {
                if (!axisData[k] || axisSizes[k] < 2 || axisSizes[k] > static_cast<size_t>(std::numeric_limits<int>::max()))
                    return false;
                if (quantityCount > std::numeric_limits<size_t>::max() / axisSizes[k]) return false;
                axisOffsets[k] = static_cast<int>(axisValueCount);
                axisLengthv[k] = static_cast<int>(axisSizes[k]);
                axisLogv[k] = axisLog[k] ? 1 : 0;
                axisValueCount += axisSizes[k];
                quantityCount *= axisSizes[k];
            }
            if (axisValueCount > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            vector<double> axisValues(axisValueCount);
            for (int k = 0; k != numAxes; ++k)
            {
                double* target = axisValues.data() + axisOffsets[k];
                std::copy(axisData[k], axisData[k] + axisSizes[k], target);
            }

            vector<double> quantityValues(quantityCount);
            for (size_t i = 0; i != quantityCount; ++i) quantityValues[i] = quantity[i * quantityStep];

            unsigned long long numDeviceEntities = static_cast<unsigned long long>(numEntities);
            unsigned int blockSize = 128;
            unsigned long long blocks = (numDeviceEntities + blockSize - 1) / blockSize;
            if (blocks > std::numeric_limits<unsigned int>::max()) return false;
            unsigned int gridSize = static_cast<unsigned int>(blocks);

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            CUdeviceptr dAxisValues = 0;
            CUdeviceptr dAxisOffsets = 0;
            CUdeviceptr dAxisLengths = 0;
            CUdeviceptr dAxisLog = 0;
            CUdeviceptr dQuantity = 0;
            CUdeviceptr dParameters = 0;
            CUdeviceptr dScaleValues = 0;
            CUdeviceptr dLuminosities = 0;
            auto freeAll = [&]() {
                if (dAxisValues) _cuMemFree(dAxisValues);
                if (dAxisOffsets) _cuMemFree(dAxisOffsets);
                if (dAxisLengths) _cuMemFree(dAxisLengths);
                if (dAxisLog) _cuMemFree(dAxisLog);
                if (dQuantity) _cuMemFree(dQuantity);
                if (dParameters) _cuMemFree(dParameters);
                if (dScaleValues) _cuMemFree(dScaleValues);
                if (dLuminosities) _cuMemFree(dLuminosities);
            };

            luminosities.assign(numEntities, 0.);
            int qtyLog = quantityLog ? 1 : 0;
            int clamp = clampFirstAxis ? 1 : 0;
            bool ok = allocateAndCopy(dAxisValues, axisValues.data(), axisValues.size() * sizeof(double))
                      && allocateAndCopy(dAxisOffsets, axisOffsets.data(), axisOffsets.size() * sizeof(int))
                      && allocateAndCopy(dAxisLengths, axisLengthv.data(), axisLengthv.size() * sizeof(int))
                      && allocateAndCopy(dAxisLog, axisLogv.data(), axisLogv.size() * sizeof(int))
                      && allocateAndCopy(dQuantity, quantityValues.data(), quantityValues.size() * sizeof(double))
                      && allocateAndCopy(dParameters, parameterValues.data(), parameterValues.size() * sizeof(double))
                      && allocateAndCopy(dScaleValues, scaleValues.data(), scaleValues.size() * sizeof(double))
                      && allocate(dLuminosities, luminosities.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dAxisValues,  &dAxisOffsets,       &dAxisLengths, &dAxisLog,
                            &dQuantity,    &numAxes,            &qtyLog,       &clamp,
                            &xmin,         &xmax,               &dParameters,  &dScaleValues,
                            &numDeviceEntities, &dLuminosities};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelStoredTableCdfBatch, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr,
                                          args, nullptr),
                          "cuLaunchKernel(stored_table_cdf_batch)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(luminosities.data(), dLuminosities, luminosities.size() * sizeof(double)),
                          "cuMemcpyDtoH(stored table cdf batch)");

            freeAll();
            return ok;
        }

        bool storedTableSampleWavelengths(vector<double>& wavelengths, vector<double>& specificLuminosities,
                                          int numAxes, const vector<const double*>& axisData,
                                          const vector<size_t>& axisSizes, const vector<bool>& axisLog,
                                          const double* quantity, size_t quantityStep, bool quantityLog,
                                          bool clampFirstAxis, double xmin, double xmax,
                                          const vector<double>& parameterValues,
                                          const vector<double>& intrinsicRandoms,
                                          const vector<double>& forcedWavelengths, size_t numSamples)
        {
            if (numAxes < 2 || numAxes > 5 || axisData.size() != static_cast<size_t>(numAxes)
                || axisSizes.size() != static_cast<size_t>(numAxes) || axisLog.size() != static_cast<size_t>(numAxes)
                || !quantity || !quantityStep || !numSamples || numSamples > static_cast<size_t>(std::numeric_limits<int>::max())
                || parameterValues.size() != numSamples * (numAxes - 1) || intrinsicRandoms.size() != numSamples
                || (!forcedWavelengths.empty() && forcedWavelengths.size() != numSamples))
                return false;

            size_t axisValueCount = 0;
            size_t quantityCount = 1;
            vector<int> axisOffsets(numAxes, 0);
            vector<int> axisLengthv(numAxes, 0);
            vector<int> axisLogv(numAxes, 0);
            for (int k = 0; k != numAxes; ++k)
            {
                if (!axisData[k] || axisSizes[k] < 2 || axisSizes[k] > static_cast<size_t>(std::numeric_limits<int>::max()))
                    return false;
                if (quantityCount > std::numeric_limits<size_t>::max() / axisSizes[k]) return false;
                axisOffsets[k] = static_cast<int>(axisValueCount);
                axisLengthv[k] = static_cast<int>(axisSizes[k]);
                axisLogv[k] = axisLog[k] ? 1 : 0;
                axisValueCount += axisSizes[k];
                quantityCount *= axisSizes[k];
            }
            if (axisValueCount > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            vector<double> axisValues(axisValueCount);
            for (int k = 0; k != numAxes; ++k)
            {
                double* target = axisValues.data() + axisOffsets[k];
                std::copy(axisData[k], axisData[k] + axisSizes[k], target);
            }

            vector<double> quantityValues(quantityCount);
            for (size_t i = 0; i != quantityCount; ++i) quantityValues[i] = quantity[i * quantityStep];

            unsigned long long numDeviceSamples = static_cast<unsigned long long>(numSamples);
            unsigned int blockSize = 128;
            unsigned long long blocks = (numDeviceSamples + blockSize - 1) / blockSize;
            if (blocks > std::numeric_limits<unsigned int>::max()) return false;
            unsigned int gridSize = static_cast<unsigned int>(blocks);

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            CUdeviceptr dAxisValues = 0;
            CUdeviceptr dAxisOffsets = 0;
            CUdeviceptr dAxisLengths = 0;
            CUdeviceptr dAxisLog = 0;
            CUdeviceptr dQuantity = 0;
            CUdeviceptr dParameters = 0;
            CUdeviceptr dIntrinsicRandoms = 0;
            CUdeviceptr dForcedWavelengths = 0;
            CUdeviceptr dWavelengths = 0;
            CUdeviceptr dSpecificLuminosities = 0;
            auto freeAll = [&]() {
                if (dAxisValues) _cuMemFree(dAxisValues);
                if (dAxisOffsets) _cuMemFree(dAxisOffsets);
                if (dAxisLengths) _cuMemFree(dAxisLengths);
                if (dAxisLog) _cuMemFree(dAxisLog);
                if (dQuantity) _cuMemFree(dQuantity);
                if (dParameters) _cuMemFree(dParameters);
                if (dIntrinsicRandoms) _cuMemFree(dIntrinsicRandoms);
                if (dForcedWavelengths) _cuMemFree(dForcedWavelengths);
                if (dWavelengths) _cuMemFree(dWavelengths);
                if (dSpecificLuminosities) _cuMemFree(dSpecificLuminosities);
            };

            wavelengths.assign(numSamples, 0.);
            specificLuminosities.assign(numSamples, 0.);
            int qtyLog = quantityLog ? 1 : 0;
            int clamp = clampFirstAxis ? 1 : 0;
            int hasForcedWavelengths = forcedWavelengths.empty() ? 0 : 1;
            bool ok = allocateAndCopy(dAxisValues, axisValues.data(), axisValues.size() * sizeof(double))
                      && allocateAndCopy(dAxisOffsets, axisOffsets.data(), axisOffsets.size() * sizeof(int))
                      && allocateAndCopy(dAxisLengths, axisLengthv.data(), axisLengthv.size() * sizeof(int))
                      && allocateAndCopy(dAxisLog, axisLogv.data(), axisLogv.size() * sizeof(int))
                      && allocateAndCopy(dQuantity, quantityValues.data(), quantityValues.size() * sizeof(double))
                      && allocateAndCopy(dParameters, parameterValues.data(), parameterValues.size() * sizeof(double))
                      && allocateAndCopy(dIntrinsicRandoms, intrinsicRandoms.data(),
                                         intrinsicRandoms.size() * sizeof(double))
                      && allocate(dWavelengths, wavelengths.size() * sizeof(double))
                      && allocate(dSpecificLuminosities, specificLuminosities.size() * sizeof(double));
            if (hasForcedWavelengths)
                ok = ok && allocateAndCopy(dForcedWavelengths, forcedWavelengths.data(),
                                           forcedWavelengths.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dAxisValues,
                            &dAxisOffsets,
                            &dAxisLengths,
                            &dAxisLog,
                            &dQuantity,
                            &numAxes,
                            &qtyLog,
                            &clamp,
                            &xmin,
                            &xmax,
                            &dParameters,
                            &dIntrinsicRandoms,
                            &dForcedWavelengths,
                            &hasForcedWavelengths,
                            &numDeviceSamples,
                            &dWavelengths,
                            &dSpecificLuminosities};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelStoredTableSampleWavelengthBatch, gridSize, 1, 1, blockSize, 1, 1, 0,
                                          nullptr, args, nullptr),
                          "cuLaunchKernel(stored_table_sample_wavelength_batch)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(wavelengths.data(), dWavelengths, wavelengths.size() * sizeof(double)),
                          "cuMemcpyDtoH(stored table sample wavelengths)")
                 && check(_cuMemcpyDtoH(specificLuminosities.data(), dSpecificLuminosities,
                                        specificLuminosities.size() * sizeof(double)),
                          "cuMemcpyDtoH(stored table sample specific luminosities)");

            freeAll();
            return ok;
        }

        bool findInteractionPointInCumulativePath(const SpatialGridPath* path, double tauinteract, bool hasAbsorption,
                                                  int& m, double& s, double& tauAbs)
        {
            m = -1;
            s = 0.;
            tauAbs = 0.;
            if (path->segments().size() < minGpuSegments()) return false;

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            size_t count = path->segments().size();
            int numSegments = static_cast<int>(count);
            int hasAbs = hasAbsorption ? 1 : 0;

            vector<int> cellv(count);
            vector<double> sv(count);
            vector<double> tauv(count);
            vector<double> tauAbsv(count);
            for (size_t i = 0; i != count; ++i)
            {
                const auto& segment = path->segments()[i];
                cellv[i] = segment.m();
                sv[i] = segment.s();
                tauv[i] = segment.tauExtOrSca();
                tauAbsv[i] = segment.tauAbs();
            }

            CUdeviceptr dCellv = 0;
            CUdeviceptr dSv = 0;
            CUdeviceptr dTauv = 0;
            CUdeviceptr dTauAbsv = 0;
            CUdeviceptr dCellOutv = 0;
            CUdeviceptr dDistanceOutv = 0;
            CUdeviceptr dTauAbsOutv = 0;

            auto freeAll = [&]() {
                if (dCellv) _cuMemFree(dCellv);
                if (dSv) _cuMemFree(dSv);
                if (dTauv) _cuMemFree(dTauv);
                if (dTauAbsv) _cuMemFree(dTauAbsv);
                if (dCellOutv) _cuMemFree(dCellOutv);
                if (dDistanceOutv) _cuMemFree(dDistanceOutv);
                if (dTauAbsOutv) _cuMemFree(dTauAbsOutv);
            };

            bool ok = allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dSv, sv.data(), sv.size() * sizeof(double))
                      && allocateAndCopy(dTauv, tauv.data(), tauv.size() * sizeof(double))
                      && allocateAndCopy(dTauAbsv, tauAbsv.data(), tauAbsv.size() * sizeof(double))
                      && allocate(dCellOutv, sizeof(int)) && allocate(dDistanceOutv, sizeof(double))
                      && allocate(dTauAbsOutv, sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dCellv,     &dSv,          &dTauv,        &dTauAbsv, &hasAbs,
                            &numSegments, &tauinteract, &dCellOutv,    &dDistanceOutv,
                            &dTauAbsOutv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelCumulativePathInteractionPoint, 1, 1, 1, 1, 1, 1, 0, nullptr, args,
                                          nullptr),
                          "cuLaunchKernel(cumulative_path_interaction_point)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(&m, dCellOutv, sizeof(int)), "cuMemcpyDtoH(cumulative interaction cell)")
                 && check(_cuMemcpyDtoH(&s, dDistanceOutv, sizeof(double)),
                          "cuMemcpyDtoH(cumulative interaction distance)")
                 && check(_cuMemcpyDtoH(&tauAbs, dTauAbsOutv, sizeof(double)),
                          "cuMemcpyDtoH(cumulative interaction absorption depth)");

            freeAll();
            return ok;
        }

        bool findInteractionPointsInCumulativePaths(const vector<const SpatialGridPath*>& paths,
                                                    const vector<double>& tauinteractv, bool hasAbsorption,
                                                    vector<int>& cellOutv, vector<double>& distanceOutv,
                                                    vector<double>& tauAbsOutv)
        {
            if (paths.empty() || tauinteractv.size() != paths.size()) return false;
            if (paths.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            vector<int> pathOffsetv(paths.size() + 1, 0);
            size_t totalSegments = 0;
            for (size_t pathIndex = 0; pathIndex != paths.size(); ++pathIndex)
            {
                if (!paths[pathIndex]) return false;
                totalSegments += paths[pathIndex]->segments().size();
                if (totalSegments > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
                pathOffsetv[pathIndex + 1] = static_cast<int>(totalSegments);
            }
            if (!totalSegments) return false;

            vector<int> cellv(totalSegments, -1);
            vector<double> sv(totalSegments, 0.);
            vector<double> tauv(totalSegments, 0.);
            vector<double> tauAbsv(totalSegments, 0.);
            size_t segmentIndex = 0;
            for (const SpatialGridPath* path : paths)
            {
                for (const auto& segment : path->segments())
                {
                    cellv[segmentIndex] = segment.m();
                    sv[segmentIndex] = segment.s();
                    tauv[segmentIndex] = segment.tauExtOrSca();
                    tauAbsv[segmentIndex] = segment.tauAbs();
                    ++segmentIndex;
                }
            }

            int numPaths = static_cast<int>(paths.size());
            int hasAbs = hasAbsorption ? 1 : 0;
            cellOutv.assign(paths.size(), -1);
            distanceOutv.assign(paths.size(), 0.);
            tauAbsOutv.assign(paths.size(), 0.);

            unsigned int blockSize = 128;
            unsigned int gridSize = static_cast<unsigned int>((paths.size() + blockSize - 1) / blockSize);

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            CUdeviceptr dPathOffsetv = 0;
            CUdeviceptr dCellv = 0;
            CUdeviceptr dSv = 0;
            CUdeviceptr dTauv = 0;
            CUdeviceptr dTauAbsv = 0;
            CUdeviceptr dTauinteractv = 0;
            CUdeviceptr dCellOutv = 0;
            CUdeviceptr dDistanceOutv = 0;
            CUdeviceptr dTauAbsOutv = 0;
            auto freeAll = [&]() {
                if (dPathOffsetv) _cuMemFree(dPathOffsetv);
                if (dCellv) _cuMemFree(dCellv);
                if (dSv) _cuMemFree(dSv);
                if (dTauv) _cuMemFree(dTauv);
                if (dTauAbsv) _cuMemFree(dTauAbsv);
                if (dTauinteractv) _cuMemFree(dTauinteractv);
                if (dCellOutv) _cuMemFree(dCellOutv);
                if (dDistanceOutv) _cuMemFree(dDistanceOutv);
                if (dTauAbsOutv) _cuMemFree(dTauAbsOutv);
            };

            bool ok = allocateAndCopy(dPathOffsetv, pathOffsetv.data(), pathOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dSv, sv.data(), sv.size() * sizeof(double))
                      && allocateAndCopy(dTauv, tauv.data(), tauv.size() * sizeof(double))
                      && allocateAndCopy(dTauAbsv, tauAbsv.data(), tauAbsv.size() * sizeof(double))
                      && allocateAndCopy(dTauinteractv, tauinteractv.data(), tauinteractv.size() * sizeof(double))
                      && allocate(dCellOutv, cellOutv.size() * sizeof(int))
                      && allocate(dDistanceOutv, distanceOutv.size() * sizeof(double))
                      && allocate(dTauAbsOutv, tauAbsOutv.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dPathOffsetv, &dCellv,      &dSv,        &dTauv,       &dTauAbsv,
                            &hasAbs,       &dTauinteractv, &numPaths, &dCellOutv,   &dDistanceOutv,
                            &dTauAbsOutv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelCumulativePathInteractionPoints, gridSize, 1, 1, blockSize, 1, 1, 0,
                                          nullptr, args, nullptr),
                          "cuLaunchKernel(cumulative_path_interaction_points)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(cellOutv.data(), dCellOutv, cellOutv.size() * sizeof(int)),
                          "cuMemcpyDtoH(cumulative interaction cells)")
                 && check(_cuMemcpyDtoH(distanceOutv.data(), dDistanceOutv,
                                        distanceOutv.size() * sizeof(double)),
                          "cuMemcpyDtoH(cumulative interaction distances)")
                 && check(_cuMemcpyDtoH(tauAbsOutv.data(), dTauAbsOutv, tauAbsOutv.size() * sizeof(double)),
                          "cuMemcpyDtoH(cumulative interaction absorption depths)");

            freeAll();
            return ok;
        }

        bool forcedPropagationResults(const vector<const SpatialGridPath*>& paths, const vector<double>& tauinteractv,
                                      const vector<double>& pathBiasWeightv, bool hasAbsorption,
                                      const vector<double>& albedov, vector<int>& cellOutv,
                                      vector<double>& distanceOutv, vector<double>& tauAbsOutv,
                                      vector<double>& weightOutv)
        {
            if (paths.empty() || tauinteractv.size() != paths.size() || pathBiasWeightv.size() != paths.size())
                return false;
            if (!hasAbsorption && albedov.size() != paths.size()) return false;
            if (paths.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            vector<int> pathOffsetv(paths.size() + 1, 0);
            size_t totalSegments = 0;
            for (size_t pathIndex = 0; pathIndex != paths.size(); ++pathIndex)
            {
                if (!paths[pathIndex]) return false;
                totalSegments += paths[pathIndex]->segments().size();
                if (totalSegments > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
                pathOffsetv[pathIndex + 1] = static_cast<int>(totalSegments);
            }
            if (!totalSegments) return false;

            vector<int> cellv(totalSegments, -1);
            vector<double> sv(totalSegments, 0.);
            vector<double> tauv(totalSegments, 0.);
            vector<double> tauAbsv(totalSegments, 0.);
            size_t segmentIndex = 0;
            for (const SpatialGridPath* path : paths)
            {
                for (const auto& segment : path->segments())
                {
                    cellv[segmentIndex] = segment.m();
                    sv[segmentIndex] = segment.s();
                    tauv[segmentIndex] = segment.tauExtOrSca();
                    tauAbsv[segmentIndex] = segment.tauAbs();
                    ++segmentIndex;
                }
            }

            int numPaths = static_cast<int>(paths.size());
            int hasAbs = hasAbsorption ? 1 : 0;
            cellOutv.assign(paths.size(), -1);
            distanceOutv.assign(paths.size(), 0.);
            tauAbsOutv.assign(paths.size(), 0.);
            weightOutv.assign(paths.size(), 0.);

            unsigned int blockSize = 128;
            unsigned int gridSize = static_cast<unsigned int>((paths.size() + blockSize - 1) / blockSize);

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureModule()) return false;

            CUdeviceptr dPathOffsetv = 0;
            CUdeviceptr dCellv = 0;
            CUdeviceptr dSv = 0;
            CUdeviceptr dTauv = 0;
            CUdeviceptr dTauAbsv = 0;
            CUdeviceptr dTauinteractv = 0;
            CUdeviceptr dPathBiasWeightv = 0;
            CUdeviceptr dAlbedov = 0;
            CUdeviceptr dCellOutv = 0;
            CUdeviceptr dDistanceOutv = 0;
            CUdeviceptr dTauAbsOutv = 0;
            CUdeviceptr dWeightOutv = 0;
            auto freeAll = [&]() {
                if (dPathOffsetv) _cuMemFree(dPathOffsetv);
                if (dCellv) _cuMemFree(dCellv);
                if (dSv) _cuMemFree(dSv);
                if (dTauv) _cuMemFree(dTauv);
                if (dTauAbsv) _cuMemFree(dTauAbsv);
                if (dTauinteractv) _cuMemFree(dTauinteractv);
                if (dPathBiasWeightv) _cuMemFree(dPathBiasWeightv);
                if (dAlbedov) _cuMemFree(dAlbedov);
                if (dCellOutv) _cuMemFree(dCellOutv);
                if (dDistanceOutv) _cuMemFree(dDistanceOutv);
                if (dTauAbsOutv) _cuMemFree(dTauAbsOutv);
                if (dWeightOutv) _cuMemFree(dWeightOutv);
            };

            bool ok = allocateAndCopy(dPathOffsetv, pathOffsetv.data(), pathOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dSv, sv.data(), sv.size() * sizeof(double))
                      && allocateAndCopy(dTauv, tauv.data(), tauv.size() * sizeof(double))
                      && allocateAndCopy(dTauAbsv, tauAbsv.data(), tauAbsv.size() * sizeof(double))
                      && allocateAndCopy(dTauinteractv, tauinteractv.data(), tauinteractv.size() * sizeof(double))
                      && allocateAndCopy(dPathBiasWeightv, pathBiasWeightv.data(),
                                         pathBiasWeightv.size() * sizeof(double))
                      && allocate(dCellOutv, cellOutv.size() * sizeof(int))
                      && allocate(dDistanceOutv, distanceOutv.size() * sizeof(double))
                      && allocate(dTauAbsOutv, tauAbsOutv.size() * sizeof(double))
                      && allocate(dWeightOutv, weightOutv.size() * sizeof(double));
            if (!hasAbsorption)
                ok = ok && allocateAndCopy(dAlbedov, albedov.data(), albedov.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dPathOffsetv,     &dCellv,        &dSv,        &dTauv,       &dTauAbsv,
                            &hasAbs,           &dTauinteractv, &dPathBiasWeightv, &dAlbedov,
                            &numPaths,         &dCellOutv,     &dDistanceOutv,    &dTauAbsOutv,
                            &dWeightOutv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelForcedPropagationResults, gridSize, 1, 1, blockSize, 1, 1, 0,
                                          nullptr, args, nullptr),
                          "cuLaunchKernel(forced_propagation_results)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(cellOutv.data(), dCellOutv, cellOutv.size() * sizeof(int)),
                          "cuMemcpyDtoH(forced propagation cells)")
                 && check(_cuMemcpyDtoH(distanceOutv.data(), dDistanceOutv,
                                        distanceOutv.size() * sizeof(double)),
                          "cuMemcpyDtoH(forced propagation distances)")
                 && check(_cuMemcpyDtoH(tauAbsOutv.data(), dTauAbsOutv, tauAbsOutv.size() * sizeof(double)),
                          "cuMemcpyDtoH(forced propagation absorption depths)")
                 && check(_cuMemcpyDtoH(weightOutv.data(), dWeightOutv, weightOutv.size() * sizeof(double)),
                          "cuMemcpyDtoH(forced propagation weights)");

            freeAll();
            return ok;
        }

        bool forcedPropagationTableAlbedoResults(
            const vector<const SpatialGridPath*>& paths, const MediumState& state,
            const vector<double>& tauinteractv, const vector<double>& pathBiasWeightv,
            const vector<double>& lambdav, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
            const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
            const vector<double>& sectionScaTablev, const vector<double>& sectionExtTablev,
            vector<int>& cellOutv, vector<double>& distanceOutv, vector<double>& tauAbsOutv,
            vector<double>& weightOutv)
        {
            if (paths.empty() || tauinteractv.size() != paths.size() || pathBiasWeightv.size() != paths.size()
                || lambdav.size() != paths.size())
                return false;
            if (mediaIndexv.empty() || lookupBeginv.size() != mediaIndexv.size()
                || lookupCountv.size() != mediaIndexv.size() || lookupWavelengthv.empty()
                || sectionScaTablev.size() != lookupWavelengthv.size()
                || sectionExtTablev.size() != lookupWavelengthv.size())
                return false;
            if (paths.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            for (size_t h = 0; h != mediaIndexv.size(); ++h)
            {
                if (mediaIndexv[h] < 0 || lookupBeginv[h] < 0 || lookupCountv[h] < 2) return false;
                size_t begin = static_cast<size_t>(lookupBeginv[h]);
                size_t count = static_cast<size_t>(lookupCountv[h]);
                if (begin + count > lookupWavelengthv.size()) return false;
            }

            vector<int> pathOffsetv(paths.size() + 1, 0);
            size_t totalSegments = 0;
            for (size_t pathIndex = 0; pathIndex != paths.size(); ++pathIndex)
            {
                if (!paths[pathIndex]) return false;
                totalSegments += paths[pathIndex]->segments().size();
                if (totalSegments > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
                pathOffsetv[pathIndex + 1] = static_cast<int>(totalSegments);
            }
            if (!totalSegments) return false;

            vector<int> cellv(totalSegments, -1);
            vector<double> sv(totalSegments, 0.);
            vector<double> tauv(totalSegments, 0.);
            size_t segmentIndex = 0;
            for (const SpatialGridPath* path : paths)
            {
                for (const auto& segment : path->segments())
                {
                    cellv[segmentIndex] = segment.m();
                    sv[segmentIndex] = segment.s();
                    tauv[segmentIndex] = segment.tauExtOrSca();
                    ++segmentIndex;
                }
            }

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;

            int numPaths = static_cast<int>(paths.size());
            int numTables = static_cast<int>(mediaIndexv.size());
            int numVars = state.numVars();
            vector<int> densityOffsetv(numTables);
            for (int h = 0; h != numTables; ++h) densityOffsetv[h] = state.numberDensityOffset(mediaIndexv[h]);
            cellOutv.assign(paths.size(), -1);
            distanceOutv.assign(paths.size(), 0.);
            tauAbsOutv.assign(paths.size(), 0.);
            weightOutv.assign(paths.size(), 0.);

            unsigned int blockSize = 128;
            unsigned int gridSize = static_cast<unsigned int>((paths.size() + blockSize - 1) / blockSize);

            CUdeviceptr dPathOffsetv = 0;
            CUdeviceptr dCellv = 0;
            CUdeviceptr dSv = 0;
            CUdeviceptr dTauv = 0;
            CUdeviceptr dTauinteractv = 0;
            CUdeviceptr dPathBiasWeightv = 0;
            CUdeviceptr dLambdav = 0;
            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dLookupBeginv = 0;
            CUdeviceptr dLookupCountv = 0;
            CUdeviceptr dLookupWavelengthv = 0;
            CUdeviceptr dSectionScaTablev = 0;
            CUdeviceptr dSectionExtTablev = 0;
            CUdeviceptr dCellOutv = 0;
            CUdeviceptr dDistanceOutv = 0;
            CUdeviceptr dTauAbsOutv = 0;
            CUdeviceptr dWeightOutv = 0;
            auto freeAll = [&]() {
                if (dPathOffsetv) _cuMemFree(dPathOffsetv);
                if (dCellv) _cuMemFree(dCellv);
                if (dSv) _cuMemFree(dSv);
                if (dTauv) _cuMemFree(dTauv);
                if (dTauinteractv) _cuMemFree(dTauinteractv);
                if (dPathBiasWeightv) _cuMemFree(dPathBiasWeightv);
                if (dLambdav) _cuMemFree(dLambdav);
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dLookupBeginv) _cuMemFree(dLookupBeginv);
                if (dLookupCountv) _cuMemFree(dLookupCountv);
                if (dLookupWavelengthv) _cuMemFree(dLookupWavelengthv);
                if (dSectionScaTablev) _cuMemFree(dSectionScaTablev);
                if (dSectionExtTablev) _cuMemFree(dSectionExtTablev);
                if (dCellOutv) _cuMemFree(dCellOutv);
                if (dDistanceOutv) _cuMemFree(dDistanceOutv);
                if (dTauAbsOutv) _cuMemFree(dTauAbsOutv);
                if (dWeightOutv) _cuMemFree(dWeightOutv);
            };

            bool ok = allocateAndCopy(dPathOffsetv, pathOffsetv.data(), pathOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dSv, sv.data(), sv.size() * sizeof(double))
                      && allocateAndCopy(dTauv, tauv.data(), tauv.size() * sizeof(double))
                      && allocateAndCopy(dTauinteractv, tauinteractv.data(), tauinteractv.size() * sizeof(double))
                      && allocateAndCopy(dPathBiasWeightv, pathBiasWeightv.data(),
                                         pathBiasWeightv.size() * sizeof(double))
                      && allocateAndCopy(dLambdav, lambdav.data(), lambdav.size() * sizeof(double))
                      && allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dLookupBeginv, lookupBeginv.data(), lookupBeginv.size() * sizeof(int))
                      && allocateAndCopy(dLookupCountv, lookupCountv.data(), lookupCountv.size() * sizeof(int))
                      && allocateAndCopy(dLookupWavelengthv, lookupWavelengthv.data(),
                                         lookupWavelengthv.size() * sizeof(double))
                      && allocateAndCopy(dSectionScaTablev, sectionScaTablev.data(),
                                         sectionScaTablev.size() * sizeof(double))
                      && allocateAndCopy(dSectionExtTablev, sectionExtTablev.data(),
                                         sectionExtTablev.size() * sizeof(double))
                      && allocate(dCellOutv, cellOutv.size() * sizeof(int))
                      && allocate(dDistanceOutv, distanceOutv.size() * sizeof(double))
                      && allocate(dTauAbsOutv, tauAbsOutv.size() * sizeof(double))
                      && allocate(dWeightOutv, weightOutv.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            void* args[] = {&dPathOffsetv,      &dCellv,          &dSv,             &dTauv,
                            &dTauinteractv,     &dPathBiasWeightv, &dLambdav,       &_stateDevice,
                            &numVars,           &numTables,       &dDensityOffsetv, &dLookupBeginv,
                            &dLookupCountv,     &dLookupWavelengthv, &dSectionScaTablev,
                            &dSectionExtTablev, &numPaths,        &dCellOutv,      &dDistanceOutv,
                            &dTauAbsOutv,       &dWeightOutv};
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuLaunchKernel(_kernelForcedPropagationTableAlbedoResults, gridSize, 1, 1, blockSize,
                                          1, 1, 0, nullptr, args, nullptr),
                          "cuLaunchKernel(forced_propagation_table_albedo_results)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(cellOutv.data(), dCellOutv, cellOutv.size() * sizeof(int)),
                          "cuMemcpyDtoH(forced propagation table cells)")
                 && check(_cuMemcpyDtoH(distanceOutv.data(), dDistanceOutv,
                                        distanceOutv.size() * sizeof(double)),
                          "cuMemcpyDtoH(forced propagation table distances)")
                 && check(_cuMemcpyDtoH(tauAbsOutv.data(), dTauAbsOutv, tauAbsOutv.size() * sizeof(double)),
                          "cuMemcpyDtoH(forced propagation table absorption depths)")
                 && check(_cuMemcpyDtoH(weightOutv.data(), dWeightOutv, weightOutv.size() * sizeof(double)),
                          "cuMemcpyDtoH(forced propagation table weights)");

            freeAll();
            return ok;
        }

        bool radiationFieldSumsAndForcedPropagationTableAlbedoResults(
            const vector<const SpatialGridPath*>& paths, const MediumState& state,
            const vector<double>& luminosityv, const vector<int>& wavelengthBinv, int numWavelengths,
            const vector<double>& tauinteractv, const vector<double>& pathBiasWeightv,
            const vector<double>& lambdav, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
            const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
            const vector<double>& sectionScaTablev, const vector<double>& sectionExtTablev,
            vector<int>& binIndexv, vector<double>& Ldsv, vector<int>& cellOutv,
            vector<double>& distanceOutv, vector<double>& tauAbsOutv, vector<double>& weightOutv)
        {
            if (paths.empty() || luminosityv.size() != paths.size() || wavelengthBinv.size() != paths.size()
                || tauinteractv.size() != paths.size() || pathBiasWeightv.size() != paths.size()
                || lambdav.size() != paths.size() || numWavelengths <= 0)
                return false;
            if (mediaIndexv.empty() || lookupBeginv.size() != mediaIndexv.size()
                || lookupCountv.size() != mediaIndexv.size() || lookupWavelengthv.empty()
                || sectionScaTablev.size() != lookupWavelengthv.size()
                || sectionExtTablev.size() != lookupWavelengthv.size())
                return false;
            if (paths.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;

            for (size_t h = 0; h != mediaIndexv.size(); ++h)
            {
                if (mediaIndexv[h] < 0 || lookupBeginv[h] < 0 || lookupCountv[h] < 2) return false;
                size_t begin = static_cast<size_t>(lookupBeginv[h]);
                size_t count = static_cast<size_t>(lookupCountv[h]);
                if (begin + count > lookupWavelengthv.size()) return false;
            }

            int maxCellForKey = (std::numeric_limits<int>::max() - (numWavelengths - 1)) / numWavelengths;
            vector<int> pathOffsetv(paths.size() + 1, 0);
            size_t totalSegments = 0;
            for (size_t pathIndex = 0; pathIndex != paths.size(); ++pathIndex)
            {
                if (!paths[pathIndex]) return false;
                int ell = wavelengthBinv[pathIndex];
                if (ell < -1 || ell >= numWavelengths) return false;
                totalSegments += paths[pathIndex]->segments().size();
                if (totalSegments > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
                pathOffsetv[pathIndex + 1] = static_cast<int>(totalSegments);
            }
            if (!totalSegments || totalSegments < minGpuSegments()) return false;
            if (totalSegments > static_cast<size_t>(std::numeric_limits<int>::max()) / 2) return false;
            size_t hashCapacity = nextPowerOfTwo(totalSegments * 2);
            if (hashCapacity > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            int hashCapacityInt = static_cast<int>(hashCapacity);

            vector<int> pathIndexv(totalSegments, 0);
            vector<int> cellv(totalSegments, -1);
            vector<double> sv(totalSegments, 0.);
            vector<double> dsv(totalSegments, 0.);
            vector<double> tauv(totalSegments, 0.);
            size_t segmentIndex = 0;
            for (size_t pathIndex = 0; pathIndex != paths.size(); ++pathIndex)
            {
                for (const auto& segment : paths[pathIndex]->segments())
                {
                    int cell = segment.m();
                    if (cell > maxCellForKey) return false;
                    pathIndexv[segmentIndex] = static_cast<int>(pathIndex);
                    cellv[segmentIndex] = cell;
                    sv[segmentIndex] = segment.s();
                    dsv[segmentIndex] = segment.ds();
                    tauv[segmentIndex] = segment.tauExtOrSca();
                    ++segmentIndex;
                }
            }

            std::lock_guard<std::mutex> lock(_mutex);
            if (!ensureReady()) return false;
            if (!check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")) return false;
            if (!ensureStateOnDevice(state)) return false;
            if (!ensureModule()) return false;

            int numPaths = static_cast<int>(paths.size());
            int numSegments = static_cast<int>(totalSegments);
            int numTables = static_cast<int>(mediaIndexv.size());
            int numVars = state.numVars();
            vector<int> densityOffsetv(numTables);
            for (int h = 0; h != numTables; ++h) densityOffsetv[h] = state.numberDensityOffset(mediaIndexv[h]);

            cellOutv.assign(paths.size(), -1);
            distanceOutv.assign(paths.size(), 0.);
            tauAbsOutv.assign(paths.size(), 0.);
            weightOutv.assign(paths.size(), 0.);

            CUdeviceptr dPathOffsetv = 0;
            CUdeviceptr dPathIndexv = 0;
            CUdeviceptr dCellv = 0;
            CUdeviceptr dSv = 0;
            CUdeviceptr dDsv = 0;
            CUdeviceptr dTauv = 0;
            CUdeviceptr dLuminosityv = 0;
            CUdeviceptr dWavelengthBinv = 0;
            CUdeviceptr dTauinteractv = 0;
            CUdeviceptr dPathBiasWeightv = 0;
            CUdeviceptr dLambdav = 0;
            CUdeviceptr dDensityOffsetv = 0;
            CUdeviceptr dLookupBeginv = 0;
            CUdeviceptr dLookupCountv = 0;
            CUdeviceptr dLookupWavelengthv = 0;
            CUdeviceptr dSectionScaTablev = 0;
            CUdeviceptr dSectionExtTablev = 0;
            CUdeviceptr dKeyv = 0;
            CUdeviceptr dSumv = 0;
            CUdeviceptr dCountv = 0;
            CUdeviceptr dCompactKeyv = 0;
            CUdeviceptr dCompactSumv = 0;
            CUdeviceptr dCellOutv = 0;
            CUdeviceptr dDistanceOutv = 0;
            CUdeviceptr dTauAbsOutv = 0;
            CUdeviceptr dWeightOutv = 0;

            auto freeAll = [&]() {
                if (dPathOffsetv) _cuMemFree(dPathOffsetv);
                if (dPathIndexv) _cuMemFree(dPathIndexv);
                if (dCellv) _cuMemFree(dCellv);
                if (dSv) _cuMemFree(dSv);
                if (dDsv) _cuMemFree(dDsv);
                if (dTauv) _cuMemFree(dTauv);
                if (dLuminosityv) _cuMemFree(dLuminosityv);
                if (dWavelengthBinv) _cuMemFree(dWavelengthBinv);
                if (dTauinteractv) _cuMemFree(dTauinteractv);
                if (dPathBiasWeightv) _cuMemFree(dPathBiasWeightv);
                if (dLambdav) _cuMemFree(dLambdav);
                if (dDensityOffsetv) _cuMemFree(dDensityOffsetv);
                if (dLookupBeginv) _cuMemFree(dLookupBeginv);
                if (dLookupCountv) _cuMemFree(dLookupCountv);
                if (dLookupWavelengthv) _cuMemFree(dLookupWavelengthv);
                if (dSectionScaTablev) _cuMemFree(dSectionScaTablev);
                if (dSectionExtTablev) _cuMemFree(dSectionExtTablev);
                if (dKeyv) _cuMemFree(dKeyv);
                if (dSumv) _cuMemFree(dSumv);
                if (dCountv) _cuMemFree(dCountv);
                if (dCompactKeyv) _cuMemFree(dCompactKeyv);
                if (dCompactSumv) _cuMemFree(dCompactSumv);
                if (dCellOutv) _cuMemFree(dCellOutv);
                if (dDistanceOutv) _cuMemFree(dDistanceOutv);
                if (dTauAbsOutv) _cuMemFree(dTauAbsOutv);
                if (dWeightOutv) _cuMemFree(dWeightOutv);
            };

            bool ok = allocateAndCopy(dPathOffsetv, pathOffsetv.data(), pathOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dPathIndexv, pathIndexv.data(), pathIndexv.size() * sizeof(int))
                      && allocateAndCopy(dCellv, cellv.data(), cellv.size() * sizeof(int))
                      && allocateAndCopy(dSv, sv.data(), sv.size() * sizeof(double))
                      && allocateAndCopy(dDsv, dsv.data(), dsv.size() * sizeof(double))
                      && allocateAndCopy(dTauv, tauv.data(), tauv.size() * sizeof(double))
                      && allocateAndCopy(dLuminosityv, luminosityv.data(), luminosityv.size() * sizeof(double))
                      && allocateAndCopy(dWavelengthBinv, wavelengthBinv.data(),
                                         wavelengthBinv.size() * sizeof(int))
                      && allocateAndCopy(dTauinteractv, tauinteractv.data(), tauinteractv.size() * sizeof(double))
                      && allocateAndCopy(dPathBiasWeightv, pathBiasWeightv.data(),
                                         pathBiasWeightv.size() * sizeof(double))
                      && allocateAndCopy(dLambdav, lambdav.data(), lambdav.size() * sizeof(double))
                      && allocateAndCopy(dDensityOffsetv, densityOffsetv.data(), densityOffsetv.size() * sizeof(int))
                      && allocateAndCopy(dLookupBeginv, lookupBeginv.data(), lookupBeginv.size() * sizeof(int))
                      && allocateAndCopy(dLookupCountv, lookupCountv.data(), lookupCountv.size() * sizeof(int))
                      && allocateAndCopy(dLookupWavelengthv, lookupWavelengthv.data(),
                                         lookupWavelengthv.size() * sizeof(double))
                      && allocateAndCopy(dSectionScaTablev, sectionScaTablev.data(),
                                         sectionScaTablev.size() * sizeof(double))
                      && allocateAndCopy(dSectionExtTablev, sectionExtTablev.data(),
                                         sectionExtTablev.size() * sizeof(double))
                      && allocate(dKeyv, hashCapacity * sizeof(int))
                      && allocate(dSumv, hashCapacity * sizeof(double))
                      && allocate(dCountv, sizeof(int))
                      && allocate(dCompactKeyv, totalSegments * sizeof(int))
                      && allocate(dCompactSumv, totalSegments * sizeof(double))
                      && allocate(dCellOutv, cellOutv.size() * sizeof(int))
                      && allocate(dDistanceOutv, distanceOutv.size() * sizeof(double))
                      && allocate(dTauAbsOutv, tauAbsOutv.size() * sizeof(double))
                      && allocate(dWeightOutv, weightOutv.size() * sizeof(double));
            if (!ok)
            {
                freeAll();
                return false;
            }

            unsigned int blockSize = 256;
            unsigned int segmentGridSize = static_cast<unsigned int>((totalSegments + blockSize - 1) / blockSize);
            unsigned int compactGridSize = static_cast<unsigned int>((hashCapacity + blockSize - 1) / blockSize);
            void* rfArgs[] = {&dPathOffsetv, &dPathIndexv, &dCellv,        &dDsv,
                              &dTauv,        &dLuminosityv, &dWavelengthBinv, &numWavelengths,
                              &numSegments,  &hashCapacityInt, &dKeyv,     &dSumv};
            void* forcedArgs[] = {&dPathOffsetv,      &dCellv,          &dSv,             &dTauv,
                                  &dTauinteractv,     &dPathBiasWeightv, &dLambdav,       &_stateDevice,
                                  &numVars,           &numTables,       &dDensityOffsetv, &dLookupBeginv,
                                  &dLookupCountv,     &dLookupWavelengthv, &dSectionScaTablev,
                                  &dSectionExtTablev, &numPaths,        &dCellOutv,      &dDistanceOutv,
                                  &dTauAbsOutv,       &dWeightOutv};
            void* compactArgs[] = {&dKeyv, &dSumv, &hashCapacityInt, &dCountv, &dCompactKeyv, &dCompactSumv};

            unsigned int forcedBlockSize = 128;
            unsigned int forcedGridSize = static_cast<unsigned int>((paths.size() + forcedBlockSize - 1)
                                                                    / forcedBlockSize);
            int compactCount = 0;
            ok = check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent")
                 && check(_cuMemsetD32(dKeyv, 0xffffffffU, hashCapacity), "cuMemsetD32(combined radiation keys)")
                 && check(_cuMemsetD32(dSumv, 0, hashCapacity * 2), "cuMemsetD32(combined radiation sums)")
                 && check(_cuMemsetD32(dCountv, 0, 1), "cuMemsetD32(combined radiation count)")
                 && check(_cuLaunchKernel(_kernelRadiationFieldSumBatch, segmentGridSize, 1, 1, blockSize, 1, 1,
                                          0, nullptr, rfArgs, nullptr),
                          "cuLaunchKernel(combined radiation_field_contribution_sums_batch)")
                 && check(_cuLaunchKernel(_kernelForcedPropagationTableAlbedoResults, forcedGridSize, 1, 1,
                                          forcedBlockSize, 1, 1, 0, nullptr, forcedArgs, nullptr),
                          "cuLaunchKernel(combined forced_propagation_table_albedo_results)")
                 && check(_cuLaunchKernel(_kernelRadiationFieldCompactSums, compactGridSize, 1, 1, blockSize,
                                          1, 1, 0, nullptr, compactArgs, nullptr),
                          "cuLaunchKernel(combined radiation_field_compact_sums)")
                 && check(_cuCtxSynchronize(), "cuCtxSynchronize")
                 && check(_cuMemcpyDtoH(&compactCount, dCountv, sizeof(int)),
                          "cuMemcpyDtoH(combined radiation sum count)");
            if (!ok || compactCount < 0 || static_cast<size_t>(compactCount) > totalSegments)
            {
                freeAll();
                return false;
            }

            binIndexv.assign(static_cast<size_t>(compactCount), 0);
            Ldsv.assign(static_cast<size_t>(compactCount), 0.);
            ok = check(_cuMemcpyDtoH(cellOutv.data(), dCellOutv, cellOutv.size() * sizeof(int)),
                       "cuMemcpyDtoH(combined forced propagation cells)")
                 && check(_cuMemcpyDtoH(distanceOutv.data(), dDistanceOutv,
                                        distanceOutv.size() * sizeof(double)),
                          "cuMemcpyDtoH(combined forced propagation distances)")
                 && check(_cuMemcpyDtoH(tauAbsOutv.data(), dTauAbsOutv, tauAbsOutv.size() * sizeof(double)),
                          "cuMemcpyDtoH(combined forced propagation absorption depths)")
                 && check(_cuMemcpyDtoH(weightOutv.data(), dWeightOutv, weightOutv.size() * sizeof(double)),
                          "cuMemcpyDtoH(combined forced propagation weights)");
            if (ok && compactCount)
            {
                ok = check(_cuMemcpyDtoH(binIndexv.data(), dCompactKeyv, binIndexv.size() * sizeof(int)),
                           "cuMemcpyDtoH(combined radiation compact keys)")
                     && check(_cuMemcpyDtoH(Ldsv.data(), dCompactSumv, Ldsv.size() * sizeof(double)),
                              "cuMemcpyDtoH(combined radiation compact sums)");
            }

            freeAll();
            return ok;
        }

    private:
        using CuInit = CUresult (*)(unsigned int);
        using CuDeviceGetCount = CUresult (*)(int*);
        using CuDeviceGet = CUresult (*)(CUdevice*, int);
        using CuDeviceGetName = CUresult (*)(char*, int, CUdevice);
        using CuDeviceComputeCapability = CUresult (*)(int*, int*, CUdevice);
        using CuCtxCreate = CUresult (*)(CUcontext*, unsigned int, CUdevice);
        using CuCtxSetCurrent = CUresult (*)(CUcontext);
        using CuCtxSynchronize = CUresult (*)();
        using CuMemAlloc = CUresult (*)(CUdeviceptr*, size_t);
        using CuMemFree = CUresult (*)(CUdeviceptr);
        using CuMemsetD32 = CUresult (*)(CUdeviceptr, unsigned int, size_t);
        using CuMemcpyHtoD = CUresult (*)(CUdeviceptr, const void*, size_t);
        using CuMemcpyDtoH = CUresult (*)(void*, CUdeviceptr, size_t);
        using CuModuleLoadData = CUresult (*)(CUmodule*, const void*);
        using CuModuleUnload = CUresult (*)(CUmodule);
        using CuModuleGetFunction = CUresult (*)(CUfunction*, CUmodule, const char*);
        using CuLaunchKernel = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                                            unsigned int, unsigned int, unsigned int, CUstream, void**, void**);
        using CuGetErrorString = CUresult (*)(CUresult, const char**);

        using NvrtcVersion = nvrtcResult (*)(int*, int*);
        using NvrtcCreateProgram = nvrtcResult (*)(nvrtcProgram*, const char*, const char*, int, const char* const*,
                                                   const char* const*);
        using NvrtcCompileProgram = nvrtcResult (*)(nvrtcProgram, int, const char* const*);
        using NvrtcGetPTXSize = nvrtcResult (*)(nvrtcProgram, size_t*);
        using NvrtcGetPTX = nvrtcResult (*)(nvrtcProgram, char*);
        using NvrtcGetProgramLogSize = nvrtcResult (*)(nvrtcProgram, size_t*);
        using NvrtcGetProgramLog = nvrtcResult (*)(nvrtcProgram, char*);
        using NvrtcDestroyProgram = nvrtcResult (*)(nvrtcProgram*);
        using NvrtcGetErrorString = const char* (*)(nvrtcResult);

        bool ensureReady()
        {
            if (_ready) return true;
            if (_failed) return false;
            if (!loadCuda() || !loadNvrtc() || !initializeCuda())
            {
                _failed = true;
                return false;
            }
            _ready = true;
            return true;
        }

        bool loadCuda()
        {
            _cuda = openLibrary({"libcuda.so.1", "libcuda.so"});
            if (!_cuda)
            {
                _error = "failed to open libcuda.so";
                return false;
            }

            bool ok = loadSymbol(_cuda, "cuInit", _cuInit)
                      && loadSymbol(_cuda, "cuDeviceGetCount", _cuDeviceGetCount)
                      && loadSymbol(_cuda, "cuDeviceGet", _cuDeviceGet)
                      && loadSymbol(_cuda, "cuDeviceGetName", _cuDeviceGetName)
                      && loadSymbol(_cuda, "cuDeviceComputeCapability", _cuDeviceComputeCapability)
                      && loadVersionedSymbol(_cuda, "cuCtxCreate", "cuCtxCreate_v2", _cuCtxCreate)
                      && loadSymbol(_cuda, "cuCtxSetCurrent", _cuCtxSetCurrent)
                      && loadSymbol(_cuda, "cuCtxSynchronize", _cuCtxSynchronize)
                      && loadVersionedSymbol(_cuda, "cuMemAlloc", "cuMemAlloc_v2", _cuMemAlloc)
                      && loadVersionedSymbol(_cuda, "cuMemFree", "cuMemFree_v2", _cuMemFree)
                      && loadVersionedSymbol(_cuda, "cuMemsetD32", "cuMemsetD32_v2", _cuMemsetD32)
                      && loadVersionedSymbol(_cuda, "cuMemcpyHtoD", "cuMemcpyHtoD_v2", _cuMemcpyHtoD)
                      && loadVersionedSymbol(_cuda, "cuMemcpyDtoH", "cuMemcpyDtoH_v2", _cuMemcpyDtoH)
                      && loadSymbol(_cuda, "cuModuleLoadData", _cuModuleLoadData)
                      && loadSymbol(_cuda, "cuModuleUnload", _cuModuleUnload)
                      && loadSymbol(_cuda, "cuModuleGetFunction", _cuModuleGetFunction)
                      && loadSymbol(_cuda, "cuLaunchKernel", _cuLaunchKernel);
            loadSymbol(_cuda, "cuGetErrorString", _cuGetErrorString);
            if (!ok) _error = "missing CUDA driver symbols";
            return ok;
        }

        bool loadNvrtc()
        {
            vector<string> builtinsNames;
            if (const char* env = std::getenv("SKIRTGPU_NVRTC_BUILTINS_LIBRARY")) builtinsNames.emplace_back(env);
            builtinsNames.insert(builtinsNames.end(),
                                 {"libnvrtc-builtins.so", "libnvrtc-builtins.so.12",
                                  "/opt/resolve/libs/libnvrtc-builtins.so.12.8",
                                  "/opt/resolve/libs/libnvrtc-builtins.so.12.8.61",
                                  "/opt/resolve/libs/libnvrtc-builtins.so"});
            _nvrtcBuiltins = openLibraryGlobal(builtinsNames);

            vector<string> names;
            if (const char* env = std::getenv("SKIRTGPU_NVRTC_LIBRARY")) names.emplace_back(env);
            names.insert(names.end(), {"libnvrtc.so", "libnvrtc.so.12", "/opt/resolve/libs/libnvrtc.so",
                                       "/opt/resolve/libs/libnvrtc.so.12"});
            _nvrtc = openLibrary(names);
            if (!_nvrtc)
            {
                _error = "failed to open libnvrtc.so";
                return false;
            }

            bool ok = loadSymbol(_nvrtc, "nvrtcVersion", _nvrtcVersion)
                      && loadSymbol(_nvrtc, "nvrtcCreateProgram", _nvrtcCreateProgram)
                      && loadSymbol(_nvrtc, "nvrtcCompileProgram", _nvrtcCompileProgram)
                      && loadSymbol(_nvrtc, "nvrtcGetPTXSize", _nvrtcGetPTXSize)
                      && loadSymbol(_nvrtc, "nvrtcGetPTX", _nvrtcGetPTX)
                      && loadSymbol(_nvrtc, "nvrtcGetProgramLogSize", _nvrtcGetProgramLogSize)
                      && loadSymbol(_nvrtc, "nvrtcGetProgramLog", _nvrtcGetProgramLog)
                      && loadSymbol(_nvrtc, "nvrtcDestroyProgram", _nvrtcDestroyProgram)
                      && loadSymbol(_nvrtc, "nvrtcGetErrorString", _nvrtcGetErrorString);
            if (!ok) _error = "missing NVRTC symbols";
            return ok;
        }

        bool initializeCuda()
        {
            if (!check(_cuInit(0), "cuInit")) return false;
            int count = 0;
            if (!check(_cuDeviceGetCount(&count), "cuDeviceGetCount")) return false;
            if (count <= 0)
            {
                _error = "no CUDA devices";
                return false;
            }

            int deviceIndex = _deviceSlot >= 0 ? _deviceSlot % count : ProcessManager::rank() % count;
            if (const char* text = std::getenv("SKIRTGPU_DEVICE"))
            {
                char* end = nullptr;
                long value = std::strtol(text, &end, 10);
                if (end != text && value >= 0 && value < count) deviceIndex = static_cast<int>(value);
            }

            if (!check(_cuDeviceGet(&_device, deviceIndex), "cuDeviceGet")) return false;
            _deviceIndex = deviceIndex;
            _deviceCount = count;
            char name[256] = {};
            if (check(_cuDeviceGetName(name, sizeof(name), _device), "cuDeviceGetName")) _deviceName = name;
            if (!check(_cuDeviceComputeCapability(&_major, &_minor, _device), "cuDeviceComputeCapability"))
                return false;
            return check(_cuCtxCreate(&_context, 0, _device), "cuCtxCreate")
                   && check(_cuCtxSetCurrent(_context), "cuCtxSetCurrent");
        }

        bool ensureModule()
        {
            if (_module) return true;

            string ptx;
            if (!compileKernel(ptx)) return false;
            if (!check(_cuModuleLoadData(&_module, ptx.c_str()), "cuModuleLoadData")) return false;
            return check(_cuModuleGetFunction(&_kernelCartesianGridPath, _module, "cartesian_grid_path"),
                         "cuModuleGetFunction(cartesian_grid_path)")
                   && check(_cuModuleGetFunction(&_kernelTreeGridPath, _module, "tree_grid_path"),
                            "cuModuleGetFunction(tree_grid_path)")
                   && check(_cuModuleGetFunction(&_kernelSphere1DGridPath, _module, "sphere1d_grid_path"),
                            "cuModuleGetFunction(sphere1d_grid_path)")
                   && check(_cuModuleGetFunction(&_kernelSphere2DGridPath, _module, "sphere2d_grid_path"),
                            "cuModuleGetFunction(sphere2d_grid_path)")
                   && check(_cuModuleGetFunction(&_kernelSphere3DGridPath, _module, "sphere3d_grid_path"),
                            "cuModuleGetFunction(sphere3d_grid_path)")
                   && check(_cuModuleGetFunction(&_kernelCylinder2DGridPath, _module, "cylinder2d_grid_path"),
                            "cuModuleGetFunction(cylinder2d_grid_path)")
                   && check(_cuModuleGetFunction(&_kernelCylinder3DGridPath, _module, "cylinder3d_grid_path"),
                            "cuModuleGetFunction(cylinder3d_grid_path)")
                   && check(_cuModuleGetFunction(&_kernelTetraMeshGridPath, _module, "tetra_mesh_grid_path"),
                            "cuModuleGetFunction(tetra_mesh_grid_path)")
	                   && check(_cuModuleGetFunction(&_kernelVoronoiMeshGridPath, _module, "voronoi_mesh_grid_path"),
	                            "cuModuleGetFunction(voronoi_mesh_grid_path)")
	                   && check(_cuModuleGetFunction(&_kernelVoronoiMeshGridPaths, _module, "voronoi_mesh_grid_paths"),
	                            "cuModuleGetFunction(voronoi_mesh_grid_paths)")
		                   && check(_cuModuleGetFunction(&_kernelVoronoiMeshGridPathsCount, _module,
		                                                 "voronoi_mesh_grid_paths_count"),
		                            "cuModuleGetFunction(voronoi_mesh_grid_paths_count)")
		                   && check(_cuModuleGetFunction(&_kernelPathCountOffsets, _module, "path_count_offsets"),
		                            "cuModuleGetFunction(path_count_offsets)")
		                   && check(_cuModuleGetFunction(&_kernelValidatePathOffsets, _module,
		                                                 "validate_path_offsets"),
		                            "cuModuleGetFunction(validate_path_offsets)")
		                   && check(_cuModuleGetFunction(&_kernelVoronoiMeshGridPathsCompact, _module,
		                                                 "voronoi_mesh_grid_paths_compact"),
		                            "cuModuleGetFunction(voronoi_mesh_grid_paths_compact)")
	                   && check(_cuModuleGetFunction(&_kernelConstantSection, _module, "constant_section_contribution"),
	                         "cuModuleGetFunction(constant_section_contribution)")
                   && check(_cuModuleGetFunction(&_kernelTableSection, _module, "table_section_contribution"),
                            "cuModuleGetFunction(table_section_contribution)")
                   && check(_cuModuleGetFunction(&_kernelOpticalDepthSum, _module, "optical_depth_sum"),
                            "cuModuleGetFunction(optical_depth_sum)")
                   && check(_cuModuleGetFunction(&_kernelCumulativeOpticalDepths, _module,
                                                 "cumulative_optical_depths"),
                            "cuModuleGetFunction(cumulative_optical_depths)")
                   && check(_cuModuleGetFunction(&_kernelCumulativeConstantSectionOpticalDepthsBatch, _module,
                                                 "cumulative_constant_section_optical_depths_batch"),
                            "cuModuleGetFunction(cumulative_constant_section_optical_depths_batch)")
                   && check(_cuModuleGetFunction(&_kernelCumulativeTableSectionOpticalDepthsBatch, _module,
                                                 "cumulative_table_section_optical_depths_batch"),
                            "cuModuleGetFunction(cumulative_table_section_optical_depths_batch)")
                   && check(_cuModuleGetFunction(&_kernelVoronoiTableSectionOpticalDepthsCompact, _module,
                                                 "voronoi_table_section_optical_depths_compact"),
                            "cuModuleGetFunction(voronoi_table_section_optical_depths_compact)")
                   && check(_cuModuleGetFunction(&_kernelVoronoiTableExtinctionOpticalDepthTotals, _module,
                                                 "voronoi_table_extinction_optical_depth_totals"),
                            "cuModuleGetFunction(voronoi_table_extinction_optical_depth_totals)")
	                   && check(_cuModuleGetFunction(&_kernelVoronoiTableHenyeyGreensteinScatteringObservedLuminosities,
	                                                 _module,
	                                                 "voronoi_table_hg_scattering_observed_luminosities"),
	                            "cuModuleGetFunction(voronoi_table_hg_scattering_observed_luminosities)")
	                   && check(_cuModuleGetFunction(&_kernelVoronoiTableHenyeyGreensteinScatteringFrameBandAccumulate,
	                                                 _module,
	                                                 "voronoi_table_hg_scattering_frame_band_accumulate"),
	                            "cuModuleGetFunction(voronoi_table_hg_scattering_frame_band_accumulate)")
	                   && check(_cuModuleGetFunction(&_kernelInteractionPointExtinction, _module,
	                                                 "interaction_point_extinction"),
	                            "cuModuleGetFunction(interaction_point_extinction)")
                   && check(_cuModuleGetFunction(&_kernelInteractionPointScaAbs, _module, "interaction_point_sca_abs"),
                            "cuModuleGetFunction(interaction_point_sca_abs)")
                   && check(_cuModuleGetFunction(&_kernelCumulativePathInteractionPoint, _module,
                                                 "cumulative_path_interaction_point"),
                            "cuModuleGetFunction(cumulative_path_interaction_point)")
                   && check(_cuModuleGetFunction(&_kernelCumulativePathInteractionPoints, _module,
                                                 "cumulative_path_interaction_points"),
                            "cuModuleGetFunction(cumulative_path_interaction_points)")
                   && check(_cuModuleGetFunction(&_kernelForcedPropagationResults, _module,
                                                 "forced_propagation_results"),
                            "cuModuleGetFunction(forced_propagation_results)")
	                   && check(_cuModuleGetFunction(&_kernelForcedPropagationTableAlbedoResults, _module,
	                                                 "forced_propagation_table_albedo_results"),
	                            "cuModuleGetFunction(forced_propagation_table_albedo_results)")
	                   && check(_cuModuleGetFunction(&_kernelForcedPropagationTableAlbedoSampledResults, _module,
	                                                 "forced_propagation_table_albedo_sampled_results"),
	                            "cuModuleGetFunction(forced_propagation_table_albedo_sampled_results)")
	                   && check(_cuModuleGetFunction(&_kernelRadiationField, _module, "radiation_field_contribution"),
	                            "cuModuleGetFunction(radiation_field_contribution)")
                   && check(_cuModuleGetFunction(&_kernelRadiationFieldBatch, _module,
                                                 "radiation_field_contributions_batch"),
                            "cuModuleGetFunction(radiation_field_contributions_batch)")
                   && check(_cuModuleGetFunction(&_kernelRadiationFieldSumBatch, _module,
                                                 "radiation_field_contribution_sums_batch"),
                            "cuModuleGetFunction(radiation_field_contribution_sums_batch)")
	                   && check(_cuModuleGetFunction(&_kernelRadiationFieldCompactSums, _module,
	                                                 "radiation_field_compact_sums"),
	                            "cuModuleGetFunction(radiation_field_compact_sums)")
	                   && check(_cuModuleGetFunction(&_kernelPathSegmentMetadata, _module,
	                                                 "path_segment_metadata"),
	                            "cuModuleGetFunction(path_segment_metadata)")
		                   && check(_cuModuleGetFunction(&_kernelSumKeyValues, _module, "sum_key_values"),
		                            "cuModuleGetFunction(sum_key_values)")
		                   && check(_cuModuleGetFunction(&_kernelAccumulateValuesByKey, _module,
		                                                 "accumulate_values_by_key"),
		                            "cuModuleGetFunction(accumulate_values_by_key)")
		                   && check(_cuModuleGetFunction(&_kernelFrameBandTotalFluxValues, _module,
		                                                 "frame_band_total_flux_values"),
		                            "cuModuleGetFunction(frame_band_total_flux_values)")
		                   && check(_cuModuleGetFunction(&_kernelHenyeyGreensteinScatteringLuminosities, _module,
		                                                 "henyey_greenstein_scattering_luminosities"),
		                            "cuModuleGetFunction(henyey_greenstein_scattering_luminosities)")
		                   && check(_cuModuleGetFunction(&_kernelHenyeyGreensteinScatteringDirections, _module,
		                                                 "henyey_greenstein_scattering_directions"),
		                            "cuModuleGetFunction(henyey_greenstein_scattering_directions)")
		                   && check(_cuModuleGetFunction(&_kernelIsotropicDirections, _module,
		                                                 "isotropic_directions"),
		                            "cuModuleGetFunction(isotropic_directions)")
	                   && check(_cuModuleGetFunction(&_kernelDustAbsorbedLuminosity, _module,
	                                                 "dust_absorbed_luminosity"),
	                            "cuModuleGetFunction(dust_absorbed_luminosity)")
                   && check(_cuModuleGetFunction(&_kernelDustAbsorbedLuminositySum, _module,
                                                 "dust_absorbed_luminosity_sum"),
                            "cuModuleGetFunction(dust_absorbed_luminosity_sum)")
                   && check(_cuModuleGetFunction(&_kernelScatteringProperties, _module, "scattering_properties"),
                            "cuModuleGetFunction(scattering_properties)")
                   && check(_cuModuleGetFunction(&_kernelScatteringAlbedos, _module, "scattering_albedos"),
                            "cuModuleGetFunction(scattering_albedos)")
                   && check(_cuModuleGetFunction(&_kernelTableScatteringProperties, _module,
                                                 "table_scattering_properties"),
                            "cuModuleGetFunction(table_scattering_properties)")
                   && check(_cuModuleGetFunction(&_kernelTableScatteringAlbedos, _module,
                                                 "table_scattering_albedos"),
                            "cuModuleGetFunction(table_scattering_albedos)")
                   && check(_cuModuleGetFunction(&_kernelScaleWavelengthValues, _module, "scale_wavelength_values"),
                            "cuModuleGetFunction(scale_wavelength_values)")
                   && check(_cuModuleGetFunction(&_kernelScaleFrameWavelengthValues, _module,
                                                 "scale_frame_wavelength_values"),
                            "cuModuleGetFunction(scale_frame_wavelength_values)")
                   && check(_cuModuleGetFunction(&_kernelDivideValues, _module, "divide_values"),
                            "cuModuleGetFunction(divide_values)")
                   && check(_cuModuleGetFunction(&_kernelMultiplyValues, _module, "multiply_values"),
                            "cuModuleGetFunction(multiply_values)")
                   && check(_cuModuleGetFunction(&_kernelSumValues, _module, "sum_values"),
                            "cuModuleGetFunction(sum_values)")
                   && check(_cuModuleGetFunction(&_kernelEmittingCellCounts, _module, "emitting_cell_counts"),
                            "cuModuleGetFunction(emitting_cell_counts)")
                   && check(_cuModuleGetFunction(&_kernelCompositeLaunchWeights, _module,
                                                 "composite_launch_weights"),
                            "cuModuleGetFunction(composite_launch_weights)")
                   && check(_cuModuleGetFunction(&_kernelStoredTableCdfBatch, _module, "stored_table_cdf_batch"),
                            "cuModuleGetFunction(stored_table_cdf_batch)")
                   && check(_cuModuleGetFunction(&_kernelStoredTableSampleWavelengthBatch, _module,
                                                 "stored_table_sample_wavelength_batch"),
                            "cuModuleGetFunction(stored_table_sample_wavelength_batch)");
        }

        bool compileKernel(string& ptx)
        {
            std::ostringstream arch;
            arch << "--gpu-architecture=compute_" << _major << _minor;
            string archOption = arch.str();
            vector<const char*> options = {"--std=c++11", archOption.c_str()};
            if (compileKernelWithOptions(options, ptx)) return true;

            options = {"--std=c++11"};
            return compileKernelWithOptions(options, ptx);
        }

        bool compileKernelWithOptions(const vector<const char*>& options, string& ptx)
        {
            nvrtcProgram program = nullptr;
            nvrtcResult result = _nvrtcCreateProgram(&program, kernelSource(), "SkirtGpuKernels.cu", 0, nullptr, nullptr);
            if (result != NVRTC_SUCCESS_VALUE)
            {
                _error = string("nvrtcCreateProgram failed: ") + nvrtcError(result);
                return false;
            }

            result = _nvrtcCompileProgram(program, static_cast<int>(options.size()), options.data());
            if (result != NVRTC_SUCCESS_VALUE)
            {
                _error = string("NVRTC compile failed: ") + nvrtcError(result) + "\n" + programLog(program);
                _nvrtcDestroyProgram(&program);
                return false;
            }

            size_t size = 0;
            if (_nvrtcGetPTXSize(program, &size) != NVRTC_SUCCESS_VALUE || size == 0)
            {
                _error = "nvrtcGetPTXSize failed";
                _nvrtcDestroyProgram(&program);
                return false;
            }
            ptx.assign(size, '\0');
            if (_nvrtcGetPTX(program, &ptx[0]) != NVRTC_SUCCESS_VALUE)
            {
                _error = "nvrtcGetPTX failed";
                _nvrtcDestroyProgram(&program);
                return false;
            }
            _nvrtcDestroyProgram(&program);
            return true;
        }

        bool ensureStateOnDevice(const MediumState& state)
        {
            const double* host = state.data();
            size_t bytes = state.dataSize() * sizeof(double);
            if (!host || !bytes)
            {
                _error = "empty medium state";
                return false;
            }

            if (_stateDevice && (_stateHost != host || _stateBytes != bytes))
            {
                _cuMemFree(_stateDevice);
                _stateDevice = 0;
            }

            if (!_stateDevice)
            {
                if (!allocate(_stateDevice, bytes)) return false;
                _stateHost = host;
                _stateBytes = bytes;
                _stateDirty = true;
            }
            if (_stateDirty)
            {
                if (!check(_cuMemcpyHtoD(_stateDevice, host, bytes), "cuMemcpyHtoD(state)")) return false;
                _stateDirty = false;
            }
            return true;
        }

        bool ensureTreeOnDevice(const void* gridKey, const vector<double>& nodeBoundsv,
                                const vector<int>& childBeginv, const vector<int>& childCountv,
                                const vector<int>& childIndexv, const vector<int>& cellIndexv)
        {
            size_t nodeBytes = nodeBoundsv.size() * sizeof(double);
            size_t childBeginBytes = childBeginv.size() * sizeof(int);
            size_t childCountBytes = childCountv.size() * sizeof(int);
            size_t childIndexBytes = childIndexv.size() * sizeof(int);
            size_t cellIndexBytes = cellIndexv.size() * sizeof(int);

            bool reuse = _treeKey == gridKey && _treeNodeBytes == nodeBytes && _treeChildBeginBytes == childBeginBytes
                         && _treeChildCountBytes == childCountBytes && _treeChildIndexBytes == childIndexBytes
                         && _treeCellIndexBytes == cellIndexBytes;
            if (reuse) return true;

            freeTreeDeviceData();
            auto copyVector = [this](CUdeviceptr& ptr, const auto& values) {
                if (values.empty()) return true;
                return allocateAndCopy(ptr, values.data(), values.size() * sizeof(values[0]));
            };

            bool ok = copyVector(_treeBoundsDevice, nodeBoundsv) && copyVector(_treeChildBeginDevice, childBeginv)
                      && copyVector(_treeChildCountDevice, childCountv)
                      && copyVector(_treeChildIndexDevice, childIndexv) && copyVector(_treeCellIndexDevice, cellIndexv);
            if (!ok)
            {
                freeTreeDeviceData();
                return false;
            }

            _treeKey = gridKey;
            _treeNodeBytes = nodeBytes;
            _treeChildBeginBytes = childBeginBytes;
            _treeChildCountBytes = childCountBytes;
            _treeChildIndexBytes = childIndexBytes;
            _treeCellIndexBytes = cellIndexBytes;
            return true;
        }

        bool ensureTetraOnDevice(const void* gridKey, const vector<double>& vertexv, const vector<int>& tetraVertexv,
                                 const vector<int>& faceAnchorv, const vector<int>& faceNeighborv,
                                 const vector<double>& faceNormalv, const vector<double>& centroidv)
        {
            size_t vertexBytes = vertexv.size() * sizeof(double);
            size_t tetraVertexBytes = tetraVertexv.size() * sizeof(int);
            size_t faceAnchorBytes = faceAnchorv.size() * sizeof(int);
            size_t faceNeighborBytes = faceNeighborv.size() * sizeof(int);
            size_t faceNormalBytes = faceNormalv.size() * sizeof(double);
            size_t centroidBytes = centroidv.size() * sizeof(double);

            bool reuse = _tetraKey == gridKey && _tetraVertexBytes == vertexBytes
                         && _tetraTetraVertexBytes == tetraVertexBytes && _tetraFaceAnchorBytes == faceAnchorBytes
                         && _tetraFaceNeighborBytes == faceNeighborBytes && _tetraFaceNormalBytes == faceNormalBytes
                         && _tetraCentroidBytes == centroidBytes;
            if (reuse) return true;

            freeTetraDeviceData();
            auto copyVector = [this](CUdeviceptr& ptr, const auto& values) {
                if (values.empty()) return true;
                return allocateAndCopy(ptr, values.data(), values.size() * sizeof(values[0]));
            };

            bool ok = copyVector(_tetraVertexDevice, vertexv) && copyVector(_tetraTetraVertexDevice, tetraVertexv)
                      && copyVector(_tetraFaceAnchorDevice, faceAnchorv)
                      && copyVector(_tetraFaceNeighborDevice, faceNeighborv)
                      && copyVector(_tetraFaceNormalDevice, faceNormalv) && copyVector(_tetraCentroidDevice, centroidv);
            if (!ok)
            {
                freeTetraDeviceData();
                return false;
            }

            _tetraKey = gridKey;
            _tetraVertexBytes = vertexBytes;
            _tetraTetraVertexBytes = tetraVertexBytes;
            _tetraFaceAnchorBytes = faceAnchorBytes;
            _tetraFaceNeighborBytes = faceNeighborBytes;
            _tetraFaceNormalBytes = faceNormalBytes;
            _tetraCentroidBytes = centroidBytes;
            return true;
        }

        bool ensureVoronoiOnDevice(const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
                                   const vector<int>& neighborCountv, const vector<int>& neighborIndexv,
                                   const vector<int>& blockBeginv, const vector<int>& blockCountv,
                                   const vector<int>& blockIndexv)
        {
            size_t siteBytes = sitev.size() * sizeof(double);
            size_t neighborBeginBytes = neighborBeginv.size() * sizeof(int);
            size_t neighborCountBytes = neighborCountv.size() * sizeof(int);
            size_t neighborIndexBytes = neighborIndexv.size() * sizeof(int);
            size_t blockBeginBytes = blockBeginv.size() * sizeof(int);
            size_t blockCountBytes = blockCountv.size() * sizeof(int);
            size_t blockIndexBytes = blockIndexv.size() * sizeof(int);

            bool reuse = _voronoiKey == gridKey && _voronoiSiteBytes == siteBytes
                         && _voronoiNeighborBeginBytes == neighborBeginBytes
                         && _voronoiNeighborCountBytes == neighborCountBytes
                         && _voronoiNeighborIndexBytes == neighborIndexBytes
                         && _voronoiBlockBeginBytes == blockBeginBytes
                         && _voronoiBlockCountBytes == blockCountBytes
                         && _voronoiBlockIndexBytes == blockIndexBytes;
            if (reuse) return true;

            freeVoronoiDeviceData();
            auto copyVector = [this](CUdeviceptr& ptr, const auto& values) {
                if (values.empty()) return true;
                return allocateAndCopy(ptr, values.data(), values.size() * sizeof(values[0]));
            };

            bool ok = copyVector(_voronoiSiteDevice, sitev)
                      && copyVector(_voronoiNeighborBeginDevice, neighborBeginv)
                      && copyVector(_voronoiNeighborCountDevice, neighborCountv)
                      && copyVector(_voronoiNeighborIndexDevice, neighborIndexv)
                      && copyVector(_voronoiBlockBeginDevice, blockBeginv)
                      && copyVector(_voronoiBlockCountDevice, blockCountv)
                      && copyVector(_voronoiBlockIndexDevice, blockIndexv);
            if (!ok)
            {
                freeVoronoiDeviceData();
                return false;
            }

            _voronoiKey = gridKey;
            _voronoiSiteBytes = siteBytes;
            _voronoiNeighborBeginBytes = neighborBeginBytes;
            _voronoiNeighborCountBytes = neighborCountBytes;
            _voronoiNeighborIndexBytes = neighborIndexBytes;
            _voronoiBlockBeginBytes = blockBeginBytes;
            _voronoiBlockCountBytes = blockCountBytes;
            _voronoiBlockIndexBytes = blockIndexBytes;
            return true;
        }

        void freeTetraDeviceData()
        {
            if (_tetraVertexDevice) _cuMemFree(_tetraVertexDevice);
            if (_tetraTetraVertexDevice) _cuMemFree(_tetraTetraVertexDevice);
            if (_tetraFaceAnchorDevice) _cuMemFree(_tetraFaceAnchorDevice);
            if (_tetraFaceNeighborDevice) _cuMemFree(_tetraFaceNeighborDevice);
            if (_tetraFaceNormalDevice) _cuMemFree(_tetraFaceNormalDevice);
            if (_tetraCentroidDevice) _cuMemFree(_tetraCentroidDevice);
            _tetraVertexDevice = 0;
            _tetraTetraVertexDevice = 0;
            _tetraFaceAnchorDevice = 0;
            _tetraFaceNeighborDevice = 0;
            _tetraFaceNormalDevice = 0;
            _tetraCentroidDevice = 0;
            _tetraKey = nullptr;
            _tetraVertexBytes = 0;
            _tetraTetraVertexBytes = 0;
            _tetraFaceAnchorBytes = 0;
            _tetraFaceNeighborBytes = 0;
            _tetraFaceNormalBytes = 0;
            _tetraCentroidBytes = 0;
        }

        void freeVoronoiDeviceData()
        {
            if (_voronoiSiteDevice) _cuMemFree(_voronoiSiteDevice);
            if (_voronoiNeighborBeginDevice) _cuMemFree(_voronoiNeighborBeginDevice);
            if (_voronoiNeighborCountDevice) _cuMemFree(_voronoiNeighborCountDevice);
            if (_voronoiNeighborIndexDevice) _cuMemFree(_voronoiNeighborIndexDevice);
            if (_voronoiBlockBeginDevice) _cuMemFree(_voronoiBlockBeginDevice);
            if (_voronoiBlockCountDevice) _cuMemFree(_voronoiBlockCountDevice);
            if (_voronoiBlockIndexDevice) _cuMemFree(_voronoiBlockIndexDevice);
            _voronoiSiteDevice = 0;
            _voronoiNeighborBeginDevice = 0;
            _voronoiNeighborCountDevice = 0;
            _voronoiNeighborIndexDevice = 0;
            _voronoiBlockBeginDevice = 0;
            _voronoiBlockCountDevice = 0;
            _voronoiBlockIndexDevice = 0;
            _voronoiKey = nullptr;
            _voronoiSiteBytes = 0;
            _voronoiNeighborBeginBytes = 0;
            _voronoiNeighborCountBytes = 0;
            _voronoiNeighborIndexBytes = 0;
            _voronoiBlockBeginBytes = 0;
            _voronoiBlockCountBytes = 0;
            _voronoiBlockIndexBytes = 0;
        }

        void freeTreeDeviceData()
        {
            if (_treeBoundsDevice) _cuMemFree(_treeBoundsDevice);
            if (_treeChildBeginDevice) _cuMemFree(_treeChildBeginDevice);
            if (_treeChildCountDevice) _cuMemFree(_treeChildCountDevice);
            if (_treeChildIndexDevice) _cuMemFree(_treeChildIndexDevice);
            if (_treeCellIndexDevice) _cuMemFree(_treeCellIndexDevice);
            _treeBoundsDevice = 0;
            _treeChildBeginDevice = 0;
            _treeChildCountDevice = 0;
            _treeChildIndexDevice = 0;
            _treeCellIndexDevice = 0;
            _treeKey = nullptr;
            _treeNodeBytes = 0;
            _treeChildBeginBytes = 0;
            _treeChildCountBytes = 0;
            _treeChildIndexBytes = 0;
            _treeCellIndexBytes = 0;
        }

        struct DeviceBuffer
        {
            CUdeviceptr ptr{0};
            size_t bytes{0};
        };

        bool allocate(CUdeviceptr& ptr, size_t bytes)
        {
            if (!bytes) return true;
            return check(_cuMemAlloc(&ptr, bytes), "cuMemAlloc");
        }

        bool allocateAndCopy(CUdeviceptr& ptr, const void* host, size_t bytes)
        {
            return allocate(ptr, bytes) && check(_cuMemcpyHtoD(ptr, host, bytes), "cuMemcpyHtoD");
        }

        void releaseDeviceBuffer(DeviceBuffer& buffer)
        {
            if (buffer.ptr)
            {
                if (_context && _cuCtxSetCurrent) _cuCtxSetCurrent(_context);
                _cuMemFree(buffer.ptr);
            }
            buffer.ptr = 0;
            buffer.bytes = 0;
        }

        bool ensureValueAccumulator(const void* accumulatorKey, size_t numAccumulatorValues, CUdeviceptr& ptr)
        {
            ptr = 0;
            if (!accumulatorKey || !numAccumulatorValues
                || numAccumulatorValues > static_cast<size_t>(std::numeric_limits<int>::max()))
                return false;

            size_t accumulatorBytes = numAccumulatorValues * sizeof(double);
            DeviceBuffer& accumulator = _valueAccumulators[accumulatorKey];
            if (accumulator.ptr && accumulator.bytes != accumulatorBytes) releaseDeviceBuffer(accumulator);
            if (!accumulator.ptr)
            {
                if (!check(_cuMemAlloc(&accumulator.ptr, accumulatorBytes), "cuMemAlloc(value accumulator)"))
                    return false;
                accumulator.bytes = accumulatorBytes;
                if (!check(_cuMemsetD32(accumulator.ptr, 0, numAccumulatorValues * 2),
                           "cuMemsetD32(value accumulator)"))
                {
                    releaseDeviceBuffer(accumulator);
                    return false;
                }
            }

            ptr = accumulator.ptr;
            return true;
        }

        bool ensureResidentScratch(int slot, CUdeviceptr& ptr, size_t bytes, const char* label)
        {
            ptr = 0;
            if (!bytes) return true;

            DeviceBuffer& buffer = _residentScratch[slot];
            if (buffer.ptr && buffer.bytes >= bytes)
            {
                ptr = buffer.ptr;
                return true;
            }

            if (buffer.ptr)
            {
                releaseDeviceBuffer(buffer);
            }
            if (!check(_cuMemAlloc(&buffer.ptr, bytes), label)) return false;
            buffer.bytes = bytes;
            ptr = buffer.ptr;
            return true;
        }

        bool copyResidentScratch(int slot, CUdeviceptr& ptr, const void* host, size_t bytes, const char* label)
        {
            return ensureResidentScratch(slot, ptr, bytes, label) && (!bytes || check(_cuMemcpyHtoD(ptr, host, bytes), label));
        }

        bool check(CUresult result, const char* call)
        {
            if (result == CUDA_SUCCESS_VALUE) return true;
            if (_cuGetErrorString)
            {
                const char* text = nullptr;
                if (_cuGetErrorString(result, &text) == CUDA_SUCCESS_VALUE && text)
                {
                    _error = string(call) + " failed: " + text;
                    return false;
                }
            }
            _error = string(call) + " failed with CUDA error " + std::to_string(result);
            return false;
        }

        string nvrtcError(nvrtcResult result) const
        {
            return _nvrtcGetErrorString ? string(_nvrtcGetErrorString(result)) : std::to_string(result);
        }

        string programLog(nvrtcProgram program) const
        {
            size_t size = 0;
            if (_nvrtcGetProgramLogSize(program, &size) != NVRTC_SUCCESS_VALUE || !size) return "";
            string log(size, '\0');
            if (_nvrtcGetProgramLog(program, &log[0]) != NVRTC_SUCCESS_VALUE) return "";
            return log;
        }

        std::mutex _mutex;
        int _deviceSlot{-1};
        bool _ready{false};
        bool _failed{false};
        string _error{"not initialized"};

        std::unordered_map<const void*, DeviceBuffer> _valueAccumulators;

        enum ResidentScratchSlot {
            ResidentPosition,
            ResidentDirection,
	            ResidentPathOffset,
	            ResidentPathScan,
	            ResidentPathIndex,
            ResidentCell,
            ResidentDs,
            ResidentDistance,
            ResidentTau,
            ResidentPathCount,
            ResidentPathStatus,
            ResidentLuminosity,
            ResidentWavelengthBin,
            ResidentTauInteract,
            ResidentPathBiasWeight,
        ResidentRandomSelect,
        ResidentRandomSample,
        ResidentScatterRandomCostheta,
        ResidentScatterRandomPhi,
        ResidentScatterAsymmpar,
        ResidentScatterDirectionOut,
        ResidentLambda,
            ResidentDensityOffset,
            ResidentLookupBegin,
            ResidentLookupCount,
            ResidentLookupWavelength,
            ResidentSectionSca,
            ResidentSectionExt,
            ResidentBandOffset,
            ResidentBandWavelength,
            ResidentBandTransmission,
            ResidentBandWidth,
            ResidentKey,
            ResidentSum,
            ResidentCompactCount,
            ResidentCompactKey,
            ResidentCompactSum,
            ResidentCellOut,
            ResidentDistanceOut,
            ResidentTauAbsOut,
            ResidentWeightOut,
            ResidentScratchCount
        };

        std::array<DeviceBuffer, ResidentScratchCount> _residentScratch;

        void* _cuda{nullptr};
        void* _nvrtc{nullptr};
        void* _nvrtcBuiltins{nullptr};

        CuInit _cuInit{nullptr};
        CuDeviceGetCount _cuDeviceGetCount{nullptr};
        CuDeviceGet _cuDeviceGet{nullptr};
        CuDeviceGetName _cuDeviceGetName{nullptr};
        CuDeviceComputeCapability _cuDeviceComputeCapability{nullptr};
        CuCtxCreate _cuCtxCreate{nullptr};
        CuCtxSetCurrent _cuCtxSetCurrent{nullptr};
        CuCtxSynchronize _cuCtxSynchronize{nullptr};
        CuMemAlloc _cuMemAlloc{nullptr};
        CuMemFree _cuMemFree{nullptr};
        CuMemsetD32 _cuMemsetD32{nullptr};
        CuMemcpyHtoD _cuMemcpyHtoD{nullptr};
        CuMemcpyDtoH _cuMemcpyDtoH{nullptr};
        CuModuleLoadData _cuModuleLoadData{nullptr};
        CuModuleUnload _cuModuleUnload{nullptr};
        CuModuleGetFunction _cuModuleGetFunction{nullptr};
        CuLaunchKernel _cuLaunchKernel{nullptr};
        CuGetErrorString _cuGetErrorString{nullptr};

        NvrtcVersion _nvrtcVersion{nullptr};
        NvrtcCreateProgram _nvrtcCreateProgram{nullptr};
        NvrtcCompileProgram _nvrtcCompileProgram{nullptr};
        NvrtcGetPTXSize _nvrtcGetPTXSize{nullptr};
        NvrtcGetPTX _nvrtcGetPTX{nullptr};
        NvrtcGetProgramLogSize _nvrtcGetProgramLogSize{nullptr};
        NvrtcGetProgramLog _nvrtcGetProgramLog{nullptr};
        NvrtcDestroyProgram _nvrtcDestroyProgram{nullptr};
        NvrtcGetErrorString _nvrtcGetErrorString{nullptr};

        CUdevice _device{0};
        CUcontext _context{nullptr};
        int _deviceIndex{0};
        int _deviceCount{0};
        int _major{0};
        int _minor{0};
        string _deviceName{"CUDA device"};
        CUmodule _module{nullptr};
        CUfunction _kernelCartesianGridPath{nullptr};
        CUfunction _kernelTreeGridPath{nullptr};
        CUfunction _kernelSphere1DGridPath{nullptr};
        CUfunction _kernelSphere2DGridPath{nullptr};
        CUfunction _kernelSphere3DGridPath{nullptr};
        CUfunction _kernelCylinder2DGridPath{nullptr};
        CUfunction _kernelCylinder3DGridPath{nullptr};
	        CUfunction _kernelTetraMeshGridPath{nullptr};
		        CUfunction _kernelVoronoiMeshGridPath{nullptr};
		        CUfunction _kernelVoronoiMeshGridPaths{nullptr};
		        CUfunction _kernelVoronoiMeshGridPathsCount{nullptr};
		        CUfunction _kernelPathCountOffsets{nullptr};
		        CUfunction _kernelValidatePathOffsets{nullptr};
		        CUfunction _kernelVoronoiMeshGridPathsCompact{nullptr};
	        CUfunction _kernelConstantSection{nullptr};
        CUfunction _kernelTableSection{nullptr};
        CUfunction _kernelOpticalDepthSum{nullptr};
        CUfunction _kernelCumulativeOpticalDepths{nullptr};
        CUfunction _kernelCumulativeConstantSectionOpticalDepthsBatch{nullptr};
        CUfunction _kernelCumulativeTableSectionOpticalDepthsBatch{nullptr};
	        CUfunction _kernelVoronoiTableSectionOpticalDepthsCompact{nullptr};
	        CUfunction _kernelVoronoiTableExtinctionOpticalDepthTotals{nullptr};
		        CUfunction _kernelVoronoiTableHenyeyGreensteinScatteringObservedLuminosities{nullptr};
	        CUfunction _kernelVoronoiTableHenyeyGreensteinScatteringFrameBandAccumulate{nullptr};
	        CUfunction _kernelInteractionPointExtinction{nullptr};
        CUfunction _kernelInteractionPointScaAbs{nullptr};
        CUfunction _kernelCumulativePathInteractionPoint{nullptr};
        CUfunction _kernelCumulativePathInteractionPoints{nullptr};
        CUfunction _kernelForcedPropagationResults{nullptr};
        CUfunction _kernelForcedPropagationTableAlbedoResults{nullptr};
        CUfunction _kernelForcedPropagationTableAlbedoSampledResults{nullptr};
        CUfunction _kernelRadiationField{nullptr};
        CUfunction _kernelRadiationFieldBatch{nullptr};
        CUfunction _kernelRadiationFieldSumBatch{nullptr};
	        CUfunction _kernelRadiationFieldCompactSums{nullptr};
	        CUfunction _kernelPathSegmentMetadata{nullptr};
	        CUfunction _kernelSumKeyValues{nullptr};
	        CUfunction _kernelAccumulateValuesByKey{nullptr};
	        CUfunction _kernelFrameBandTotalFluxValues{nullptr};
	        CUfunction _kernelHenyeyGreensteinScatteringLuminosities{nullptr};
	        CUfunction _kernelHenyeyGreensteinScatteringDirections{nullptr};
	        CUfunction _kernelIsotropicDirections{nullptr};
        CUfunction _kernelDustAbsorbedLuminosity{nullptr};
        CUfunction _kernelDustAbsorbedLuminositySum{nullptr};
        CUfunction _kernelScatteringProperties{nullptr};
        CUfunction _kernelScatteringAlbedos{nullptr};
        CUfunction _kernelTableScatteringProperties{nullptr};
        CUfunction _kernelTableScatteringAlbedos{nullptr};
        CUfunction _kernelScaleWavelengthValues{nullptr};
        CUfunction _kernelScaleFrameWavelengthValues{nullptr};
        CUfunction _kernelDivideValues{nullptr};
        CUfunction _kernelMultiplyValues{nullptr};
        CUfunction _kernelSumValues{nullptr};
        CUfunction _kernelEmittingCellCounts{nullptr};
        CUfunction _kernelCompositeLaunchWeights{nullptr};
        CUfunction _kernelStoredTableCdfBatch{nullptr};
        CUfunction _kernelStoredTableSampleWavelengthBatch{nullptr};

        const double* _stateHost{nullptr};
        size_t _stateBytes{0};
        CUdeviceptr _stateDevice{0};
        bool _stateDirty{true};

        const void* _treeKey{nullptr};
        size_t _treeNodeBytes{0};
        size_t _treeChildBeginBytes{0};
        size_t _treeChildCountBytes{0};
        size_t _treeChildIndexBytes{0};
        size_t _treeCellIndexBytes{0};
        CUdeviceptr _treeBoundsDevice{0};
        CUdeviceptr _treeChildBeginDevice{0};
        CUdeviceptr _treeChildCountDevice{0};
        CUdeviceptr _treeChildIndexDevice{0};
        CUdeviceptr _treeCellIndexDevice{0};

        const void* _tetraKey{nullptr};
        size_t _tetraVertexBytes{0};
        size_t _tetraTetraVertexBytes{0};
        size_t _tetraFaceAnchorBytes{0};
        size_t _tetraFaceNeighborBytes{0};
        size_t _tetraFaceNormalBytes{0};
        size_t _tetraCentroidBytes{0};
        CUdeviceptr _tetraVertexDevice{0};
        CUdeviceptr _tetraTetraVertexDevice{0};
        CUdeviceptr _tetraFaceAnchorDevice{0};
        CUdeviceptr _tetraFaceNeighborDevice{0};
        CUdeviceptr _tetraFaceNormalDevice{0};
        CUdeviceptr _tetraCentroidDevice{0};

        const void* _voronoiKey{nullptr};
        size_t _voronoiSiteBytes{0};
        size_t _voronoiNeighborBeginBytes{0};
        size_t _voronoiNeighborCountBytes{0};
        size_t _voronoiNeighborIndexBytes{0};
        size_t _voronoiBlockBeginBytes{0};
        size_t _voronoiBlockCountBytes{0};
        size_t _voronoiBlockIndexBytes{0};
        CUdeviceptr _voronoiSiteDevice{0};
        CUdeviceptr _voronoiNeighborBeginDevice{0};
        CUdeviceptr _voronoiNeighborCountDevice{0};
        CUdeviceptr _voronoiNeighborIndexDevice{0};
        CUdeviceptr _voronoiBlockBeginDevice{0};
        CUdeviceptr _voronoiBlockCountDevice{0};
        CUdeviceptr _voronoiBlockIndexDevice{0};
    };

    bool useAllCudaDevices()
    {
        if (std::getenv("SKIRTGPU_DEVICE")) return false;
        if (ProcessManager::size() > 1) return false;
        if (const char* text = std::getenv("SKIRTGPU_USE_ALL_DEVICES")) return isTruthy(text);
        return true;
    }

    size_t maxCudaDevices()
    {
        const char* text = std::getenv("SKIRTGPU_MAX_DEVICES");
        if (!text) return static_cast<size_t>(std::numeric_limits<int>::max());
        char* end = nullptr;
        long value = std::strtol(text, &end, 10);
        return end != text && value > 0 ? static_cast<size_t>(value) : 1;
    }

    vector<std::unique_ptr<CudaRuntime>>& runtimePool()
    {
        static std::once_flag once;
        static vector<std::unique_ptr<CudaRuntime>> runtimes;
        std::call_once(once, []() {
            runtimes.push_back(std::make_unique<CudaRuntime>(0));
            int count = runtimes.front()->availableDeviceCount();
            if (useAllCudaDevices() && count > 1)
            {
                size_t limit = std::min(static_cast<size_t>(count), maxCudaDevices());
                for (size_t device = 1; device != limit; ++device)
                    runtimes.push_back(std::make_unique<CudaRuntime>(static_cast<int>(device)));
            }
        });
        return runtimes;
    }

    CudaRuntime& runtime()
    {
        auto& runtimes = runtimePool();
        if (runtimes.size() == 1) return *runtimes.front();

        static std::atomic<size_t> nextRuntime{0};
        thread_local size_t runtimeIndex = nextRuntime.fetch_add(1, std::memory_order_relaxed) % runtimes.size();
        return *runtimes[runtimeIndex];
    }

    CudaRuntime& residentRuntime()
    {
        auto& runtimes = runtimePool();
        if (runtimes.size() == 1) return *runtimes.front();

        static std::atomic<size_t> nextResidentRuntime{0};
        size_t runtimeIndex = nextResidentRuntime.fetch_add(1, std::memory_order_relaxed) % runtimes.size();
        return *runtimes[runtimeIndex];
    }

    CudaRuntime& observerRuntime()
    {
        auto& runtimes = runtimePool();
        if (runtimes.size() == 1) return *runtimes.front();

        static std::atomic<size_t> nextObserverRuntime{0};
        size_t runtimeIndex = nextObserverRuntime.fetch_add(1, std::memory_order_relaxed) % runtimes.size();
        return *runtimes[runtimeIndex];
    }

    CudaRuntime& batchRuntime()
    {
        auto& runtimes = runtimePool();
        if (runtimes.size() == 1) return *runtimes.front();

        static std::atomic<size_t> nextBatchRuntime{0};
        size_t runtimeIndex = nextBatchRuntime.fetch_add(1, std::memory_order_relaxed) % runtimes.size();
        return *runtimes[runtimeIndex];
    }

    string runtimeStatus()
    {
        auto& runtimes = runtimePool();
        if (runtimes.size() == 1) return runtimes.front()->status();

        std::ostringstream out;
        out << "using " << runtimes.size() << " CUDA devices";
        for (size_t i = 0; i != runtimes.size(); ++i) out << (i == 0 ? ": " : "; ") << runtimes[i]->status();
        return out.str();
    }

    void invalidateRuntimeMediumStates(const MediumState& state)
    {
        for (auto& candidate : runtimePool()) candidate->invalidateMediumState(state);
    }

    string lastRuntimeError()
    {
        string fallback = runtime().lastError();
        for (auto& candidate : runtimePool())
        {
            string error = candidate->lastError();
            if (!error.empty() && error != "not initialized") return error;
        }
        return fallback;
    }

    bool computeContributions(const SpatialGridPath* path, const MediumState& state, const vector<double>& section1v,
                              const vector<double>* section2v, vector<double>& out1v, vector<double>& out2v)
    {
        return runtime().computeConstantSectionContributions(path, state, section1v, section2v, out1v, out2v);
    }

    bool computeTableContributions(const SpatialGridPath* path, const MediumState& state,
                                   const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
                                   const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
                                   const vector<double>& section1Tablev, const vector<double>* section2Tablev,
                                   double lambda, vector<double>& out1v, vector<double>& out2v)
    {
        return runtime().computeTableSectionContributions(path, state, mediaIndexv, lookupBeginv, lookupCountv,
                                                          lookupWavelengthv, section1Tablev, section2Tablev, lambda,
                                                          out1v, out2v);
    }

    bool computeCumulativeTableOpticalDepthsBatch(
        const vector<SpatialGridPath*>& paths, const MediumState& state, const vector<int>& mediaIndexv,
        const vector<int>& lookupBeginv, const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
        const vector<double>& section1Tablev, const vector<double>* section2Tablev, const vector<double>& lambdav,
        vector<int>& pathOffsetv, vector<double>& out1v, vector<double>& out2v)
    {
        return batchRuntime().computeCumulativeTableSectionOpticalDepthsBatch(
            paths, state, mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv, section1Tablev, section2Tablev,
            lambdav, pathOffsetv, out1v, out2v);
    }

    bool computeCartesianPath(const double* xv, const double* yv, const double* zv, int nx, int ny, int nz,
                              const Position& position, const Direction& direction, double xmin, double ymin,
                              double zmin, double xmax, double ymax, double zmax, double maxDistance,
                              SpatialGridPath* path)
    {
        return runtime().traceCartesianGridPath(xv, yv, zv, nx, ny, nz, position, direction, xmin, ymin, zmin, xmax,
                                                ymax, zmax, maxDistance, path);
    }

    bool computeTreePath(const void* gridKey, const vector<double>& nodeBoundsv, const vector<int>& childBeginv,
                         const vector<int>& childCountv, const vector<int>& childIndexv,
                         const vector<int>& cellIndexv, const Position& position, const Direction& direction,
                         double xmin, double ymin, double zmin, double xmax, double ymax, double zmax,
                         double maxDistance, SpatialGridPath* path)
    {
        return runtime().traceTreeGridPath(gridKey, nodeBoundsv, childBeginv, childCountv, childIndexv, cellIndexv,
                                           position, direction, xmin, ymin, zmin, xmax, ymax, zmax, maxDistance, path);
    }

    bool computeSphere1DPath(const double* rv, int nr, const Position& position, const Direction& direction,
                             double maxDistance, SpatialGridPath* path)
    {
        return runtime().traceSphere1DGridPath(rv, nr, position, direction, maxDistance, path);
    }

    bool computeSphere2DPath(const double* rv, const double* thetav, const double* cv, int nr, int ntheta,
                             const Position& position, const Direction& direction, double maxDistance,
                             SpatialGridPath* path)
    {
        return runtime().traceSphere2DGridPath(rv, thetav, cv, nr, ntheta, position, direction, maxDistance, path);
    }

    bool computeSphere3DPath(const double* rv, const double* thetav, const double* phiv, const double* cv,
                             const double* sinv, const double* cosv, int nr, int ntheta, int nphi, double eps,
                             const Position& position, const Direction& direction, double maxDistance,
                             SpatialGridPath* path)
    {
        return runtime().traceSphere3DGridPath(rv, thetav, phiv, cv, sinv, cosv, nr, ntheta, nphi, eps, position,
                                               direction, maxDistance, path);
    }

    bool computeCylinder2DPath(const double* Rv, const double* zv, int nR, int nz, const Position& position,
                               const Direction& direction, double maxDistance, SpatialGridPath* path)
    {
        return runtime().traceCylinder2DGridPath(Rv, zv, nR, nz, position, direction, maxDistance, path);
    }

    bool computeCylinder3DPath(const double* Rv, const double* phiv, const double* zv, const double* sinv,
                               const double* cosv, int nR, int nphi, int nz, double eps, bool hasHole,
                               const Position& position, const Direction& direction, double maxDistance,
                               SpatialGridPath* path)
    {
        return runtime().traceCylinder3DGridPath(Rv, phiv, zv, sinv, cosv, nR, nphi, nz, eps, hasHole, position,
                                                direction, maxDistance, path);
    }

    bool computeTetraPath(const void* gridKey, const vector<double>& vertexv, const vector<int>& tetraVertexv,
                          const vector<int>& faceAnchorv, const vector<int>& faceNeighborv,
                          const vector<double>& faceNormalv, const vector<double>& centroidv, int numCells,
                          double eps, const Position& position, const Direction& direction, double xmin, double ymin,
                          double zmin, double xmax, double ymax, double zmax, double maxDistance,
                          SpatialGridPath* path)
    {
        return runtime().traceTetraMeshGridPath(gridKey, vertexv, tetraVertexv, faceAnchorv, faceNeighborv,
                                                faceNormalv, centroidv, numCells, eps, position, direction, xmin, ymin,
                                                zmin, xmax, ymax, zmax, maxDistance, path);
    }

    bool computeVoronoiPath(const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
                            const vector<int>& neighborCountv, const vector<int>& neighborIndexv, int numCells,
                            double eps, const Position& position, const Direction& direction, double xmin, double ymin,
                            double zmin, double xmax, double ymax, double zmax, double maxDistance,
                            SpatialGridPath* path)
    {
        return runtime().traceVoronoiMeshGridPath(gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv,
                                                  numCells, eps, position, direction, xmin, ymin, zmin, xmax, ymax,
                                                  zmax, maxDistance, path);
    }

    bool computeVoronoiPaths(const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
                             const vector<int>& neighborCountv, const vector<int>& neighborIndexv,
                             const vector<int>& blockBeginv, const vector<int>& blockCountv,
                             const vector<int>& blockIndexv, int blockN, int numCells, double eps, double xmin,
                             double ymin, double zmin, double xmax, double ymax, double zmax, double maxDistance,
                             const vector<SpatialGridPath*>& paths)
    {
        return batchRuntime().traceVoronoiMeshGridPaths(gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv,
                                                        blockBeginv, blockCountv, blockIndexv, blockN, numCells, eps,
                                                        xmin, ymin, zmin, xmax, ymax, zmax, maxDistance, paths);
    }

	    bool computeVoronoiTableOpticalDepthPaths(
	        const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
	        const vector<int>& neighborCountv, const vector<int>& neighborIndexv, const vector<int>& blockBeginv,
	        const vector<int>& blockCountv, const vector<int>& blockIndexv, int blockN, int numCells, double eps,
        double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, double maxDistance,
        const vector<SpatialGridPath*>& paths, const MediumState& state, const vector<int>& mediaIndexv,
        const vector<int>& lookupBeginv, const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
        const vector<double>& section1Tablev, const vector<double>* section2Tablev, const vector<double>& lambdav)
    {
        return batchRuntime().traceVoronoiMeshGridPathsTableOpticalDepths(
            gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv, blockCountv, blockIndexv,
            blockN, numCells, eps, xmin, ymin, zmin, xmax, ymax, zmax, maxDistance, paths, state, mediaIndexv,
	            lookupBeginv, lookupCountv, lookupWavelengthv, section1Tablev, section2Tablev, lambdav);
	    }

		    bool computeVoronoiTableRadiationFieldSumsAndForcedPropagationResults(
		        const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
		        const vector<int>& neighborCountv, const vector<int>& neighborIndexv, const vector<int>& blockBeginv,
		        const vector<int>& blockCountv, const vector<int>& blockIndexv, int blockN, int numCells, double eps,
		        double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, double maxDistance,
		        const vector<Position>& positions, const vector<Direction>& directions, const MediumState& state,
		        const vector<double>& luminosityv, const vector<int>& wavelengthBinv, int numWavelengths,
		        const vector<double>& tauinteractv, const vector<double>& pathBiasWeightv, const vector<double>& lambdav,
		        const vector<double>* randomSelectv, const vector<double>* randomSamplev, double pathLengthBias,
		        const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
		        const vector<double>& lookupWavelengthv, const vector<double>& sectionScaTablev,
		        const vector<double>& sectionExtTablev, vector<int>& binIndexv, vector<double>& Ldsv,
		        vector<int>& cellv, vector<double>& distancev, vector<double>& tauAbsv, vector<double>& weightv,
		        const vector<double>* scatterRandomCosthetav, const vector<double>* scatterRandomPhiv,
		        int hgLookupBegin, int hgLookupCount, const vector<double>* hgAsymmparv,
		        vector<double>* scatterDirectionOutv, const void* accumulatorKey = nullptr,
                size_t numAccumulatorValues = 0)
		    {
		        return residentRuntime().voronoiTableRadiationFieldSumsAndForcedPropagationResults(
		            gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv, blockCountv, blockIndexv,
		            blockN, numCells, eps, xmin, ymin, zmin, xmax, ymax, zmax, maxDistance, positions, directions, state,
		            luminosityv, wavelengthBinv, numWavelengths, tauinteractv, pathBiasWeightv, lambdav, randomSelectv,
		            randomSamplev, pathLengthBias, mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv,
		            sectionScaTablev, sectionExtTablev, binIndexv, Ldsv, cellv, distancev, tauAbsv, weightv,
		            scatterRandomCosthetav, scatterRandomPhiv, hgLookupBegin, hgLookupCount, hgAsymmparv,
		            scatterDirectionOutv, accumulatorKey, numAccumulatorValues);
		    }

    bool computeVoronoiTableOpticalDepthTotals(
        const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
        const vector<int>& neighborCountv, const vector<int>& neighborIndexv, const vector<int>& blockBeginv,
        const vector<int>& blockCountv, const vector<int>& blockIndexv, int blockN, int numCells, double eps,
        double xmin, double ymin, double zmin, double xmax, double ymax, double zmax,
        const vector<const SpatialGridPath*>& paths, const MediumState& state, const vector<int>& mediaIndexv,
        const vector<int>& lookupBeginv, const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
        const vector<double>& sectionTablev, const vector<double>& lambdav, const vector<double>& maxDistancev,
        vector<double>& tauv)
    {
        return observerRuntime().traceVoronoiMeshGridTableExtinctionTotals(
            gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv, blockCountv, blockIndexv,
            blockN, numCells, eps, xmin, ymin, zmin, xmax, ymax, zmax, paths, state, mediaIndexv, lookupBeginv,
            lookupCountv, lookupWavelengthv, sectionTablev, lambdav, maxDistancev, tauv);
    }

    bool computeVoronoiTableOpticalDepthTotals(
        const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
        const vector<int>& neighborCountv, const vector<int>& neighborIndexv, const vector<int>& blockBeginv,
        const vector<int>& blockCountv, const vector<int>& blockIndexv, int blockN, int numCells, double eps,
        double xmin, double ymin, double zmin, double xmax, double ymax, double zmax,
        const vector<Position>& positions, const vector<Direction>& directions, const MediumState& state,
        const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
        const vector<double>& lookupWavelengthv, const vector<double>& sectionTablev,
        const vector<double>& lambdav, const vector<double>& maxDistancev, vector<double>& tauv)
    {
        return observerRuntime().traceVoronoiMeshGridTableExtinctionTotals(
            gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv, blockCountv, blockIndexv,
            blockN, numCells, eps, xmin, ymin, zmin, xmax, ymax, zmax, positions, directions, state, mediaIndexv,
            lookupBeginv, lookupCountv, lookupWavelengthv, sectionTablev, lambdav, maxDistancev, tauv);
    }

    bool computeVoronoiTableOpticalDepthTotals(
        const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
        const vector<int>& neighborCountv, const vector<int>& neighborIndexv, const vector<int>& blockBeginv,
        const vector<int>& blockCountv, const vector<int>& blockIndexv, int blockN, int numCells, double eps,
        double xmin, double ymin, double zmin, double xmax, double ymax, double zmax,
        const vector<Position>& positions, const Direction& direction, const MediumState& state,
        const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
        const vector<double>& lookupWavelengthv, const vector<double>& sectionTablev,
        const vector<double>& lambdav, const vector<double>& maxDistancev, vector<double>& tauv)
    {
        return observerRuntime().traceVoronoiMeshGridTableExtinctionTotals(
            gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv, blockCountv, blockIndexv,
            blockN, numCells, eps, xmin, ymin, zmin, xmax, ymax, zmax, positions, direction, state, mediaIndexv,
            lookupBeginv, lookupCountv, lookupWavelengthv, sectionTablev, lambdav, maxDistancev, tauv);
    }

	    bool computeVoronoiTableHenyeyGreensteinObservedLuminosities(
	        const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
	        const vector<int>& neighborCountv, const vector<int>& neighborIndexv, const vector<int>& blockBeginv,
        const vector<int>& blockCountv, const vector<int>& blockIndexv, int blockN, int numCells, double eps,
        double xmin, double ymin, double zmin, double xmax, double ymax, double zmax,
        const vector<const SpatialGridPath*>& paths, const MediumState& state, const vector<int>& mediaIndexv,
        const vector<int>& lookupBeginv, const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
        const vector<double>& sectionTablev, const vector<double>& lambdav, const vector<double>& maxDistancev,
        const vector<double>& inputDirectionv, const vector<double>& packetLuminosityv, Direction bfkobs,
        int hgLookupBegin, int hgLookupCount, const vector<double>& asymmparv, vector<double>& luminosityv)
    {
        return observerRuntime().traceVoronoiMeshGridTableHenyeyGreensteinObservedLuminosities(
            gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv, blockCountv, blockIndexv,
            blockN, numCells, eps, xmin, ymin, zmin, xmax, ymax, zmax, paths, state, mediaIndexv, lookupBeginv,
	            lookupCountv, lookupWavelengthv, sectionTablev, lambdav, maxDistancev, inputDirectionv,
	            packetLuminosityv, bfkobs, hgLookupBegin, hgLookupCount, asymmparv, luminosityv);
	    }

	    bool computeVoronoiTableHenyeyGreensteinFrameBandAccumulate(
	        const void* gridKey, const vector<double>& sitev, const vector<int>& neighborBeginv,
	        const vector<int>& neighborCountv, const vector<int>& neighborIndexv, const vector<int>& blockBeginv,
	        const vector<int>& blockCountv, const vector<int>& blockIndexv, int blockN, int numCells, double eps,
	        double xmin, double ymin, double zmin, double xmax, double ymax, double zmax,
	        const vector<Position>& positionv, const MediumState& state, const vector<int>& mediaIndexv,
	        const vector<int>& lookupBeginv, const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
	        const vector<double>& sectionTablev, const vector<double>& lambdav, const vector<double>& maxDistancev,
	        const vector<double>& inputDirectionv, const vector<double>& packetLuminosityv, Direction bfkobs,
	        int hgLookupBegin, int hgLookupCount, const vector<double>& asymmparv, const void* accumulatorKey,
	        size_t numAccumulatorValues, double costheta, double sintheta, double cosphi, double sinphi,
	        double cosomega, double sinomega, int numPixelsX, int numPixelsY, double xpmin, double xpsiz,
	        double ypmin, double ypsiz, double redshift, size_t numPixelsInFrame, const vector<int>& bandOffsetv,
	        const vector<double>& bandWavelengthv, const vector<double>& bandTransmissionv,
	        const vector<double>& bandWidthv)
	    {
	        auto& runtimes = runtimePool();
	        size_t activeRuntimes = std::min(runtimes.size(), positionv.size());
	        if (activeRuntimes <= 1)
	        {
	            return batchRuntime().traceVoronoiMeshGridTableHenyeyGreensteinFrameBandAccumulate(
	                gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv, blockCountv,
	                blockIndexv, blockN, numCells, eps, xmin, ymin, zmin, xmax, ymax, zmax, positionv, state,
	                mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv, sectionTablev, lambdav,
	                maxDistancev, inputDirectionv, packetLuminosityv, bfkobs, hgLookupBegin, hgLookupCount,
	                asymmparv, accumulatorKey, numAccumulatorValues, costheta, sintheta, cosphi, sinphi,
	                cosomega, sinomega, numPixelsX, numPixelsY, xpmin, xpsiz, ypmin, ypsiz, redshift,
	                numPixelsInFrame, bandOffsetv, bandWavelengthv, bandTransmissionv, bandWidthv);
	        }

	        std::atomic<bool> ok{true};
	        vector<std::thread> workers;
	        workers.reserve(activeRuntimes);
	        for (size_t runtimeIndex = 0; runtimeIndex != activeRuntimes; ++runtimeIndex)
	        {
	            size_t begin = runtimeIndex * positionv.size() / activeRuntimes;
	            size_t end = (runtimeIndex + 1) * positionv.size() / activeRuntimes;
	            CudaRuntime* runtime = runtimes[runtimeIndex].get();
	            workers.emplace_back([&, begin, end, runtime]() {
	                if (begin == end) return;
	                vector<Position> splitPositionv(positionv.begin() + begin, positionv.begin() + end);
	                vector<double> splitLambdav(lambdav.begin() + begin, lambdav.begin() + end);
	                vector<double> splitMaxDistancev(maxDistancev.begin() + begin, maxDistancev.begin() + end);
	                vector<double> splitInputDirectionv(inputDirectionv.begin() + 3 * begin,
	                                                    inputDirectionv.begin() + 3 * end);
	                vector<double> splitPacketLuminosityv(packetLuminosityv.begin() + begin,
	                                                       packetLuminosityv.begin() + end);
	                if (!runtime->traceVoronoiMeshGridTableHenyeyGreensteinFrameBandAccumulate(
	                        gridKey, sitev, neighborBeginv, neighborCountv, neighborIndexv, blockBeginv,
	                        blockCountv, blockIndexv, blockN, numCells, eps, xmin, ymin, zmin, xmax, ymax,
	                        zmax, splitPositionv, state, mediaIndexv, lookupBeginv, lookupCountv,
	                        lookupWavelengthv, sectionTablev, splitLambdav, splitMaxDistancev,
	                        splitInputDirectionv, splitPacketLuminosityv, bfkobs, hgLookupBegin,
	                        hgLookupCount, asymmparv, accumulatorKey, numAccumulatorValues, costheta,
	                        sintheta, cosphi, sinphi, cosomega, sinomega, numPixelsX, numPixelsY, xpmin,
	                        xpsiz, ypmin, ypsiz, redshift, numPixelsInFrame, bandOffsetv, bandWavelengthv,
	                        bandTransmissionv, bandWidthv))
	                    ok.store(false, std::memory_order_relaxed);
	            });
	        }
	        for (std::thread& worker : workers) worker.join();
	        return ok.load(std::memory_order_relaxed);
	    }

	    bool computeCumulativeOpticalDepths(const SpatialGridPath* path, const MediumState& state,
	                                        const vector<double>& section1v, const vector<double>* section2v,
	                                        vector<double>& out1v, vector<double>& out2v)
    {
        return runtime().computeCumulativeConstantSectionOpticalDepths(path, state, section1v, section2v, out1v,
                                                                       out2v);
    }

    bool computeCumulativeOpticalDepthsBatch(const vector<SpatialGridPath*>& paths, const MediumState& state,
                                             const vector<double>& section1v, const vector<double>* section2v,
                                             vector<int>& pathOffsetv, vector<double>& out1v,
                                             vector<double>& out2v)
    {
        return batchRuntime().computeCumulativeConstantSectionOpticalDepthsBatch(paths, state, section1v, section2v,
                                                                                 pathOffsetv, out1v, out2v);
    }

    bool computeOpticalDepth(const SpatialGridPath* path, const MediumState& state, const vector<double>& sectionv,
                             double taumax, double& tau)
    {
        return runtime().computeExtinctionOpticalDepth(path, state, sectionv, taumax, tau);
    }

    bool computeInteractionPointExtinction(const SpatialGridPath* path, const MediumState& state,
                                           const vector<double>& sectionv, double tauinteract, bool& found, int& m,
                                           double& s)
    {
        return runtime().findInteractionPointUsingExtinction(path, state, sectionv, tauinteract, found, m, s);
    }

    bool computeInteractionPointScatteringAndAbsorption(const SpatialGridPath* path, const MediumState& state,
                                                        const vector<double>& sectionScav,
                                                        const vector<double>& sectionAbsv, double tauinteract,
                                                        bool& found, int& m, double& s, double& tauAbs)
    {
        return runtime().findInteractionPointUsingScatteringAndAbsorption(path, state, sectionScav, sectionAbsv,
                                                                          tauinteract, found, m, s, tauAbs);
    }

    bool computeRadiationContributions(const SpatialGridPath* path, double luminosity, vector<double>& Ldsv)
    {
        return runtime().computeRadiationFieldContributions(path, luminosity, Ldsv);
    }

    bool computeRadiationContributionsBatch(const vector<const SpatialGridPath*>& paths,
                                            const vector<double>& luminosityv, vector<int>& pathOffsetv,
                                            vector<double>& Ldsv)
    {
        return batchRuntime().computeRadiationFieldContributionsBatch(paths, luminosityv, pathOffsetv, Ldsv);
    }

    bool computeRadiationContributionSumsBatch(const vector<const SpatialGridPath*>& paths,
                                               const vector<double>& luminosityv,
                                               const vector<int>& wavelengthBinv, int numWavelengths,
                                               vector<int>& binIndexv, vector<double>& Ldsv)
    {
        return batchRuntime().computeRadiationFieldContributionSumsBatch(paths, luminosityv, wavelengthBinv,
                                                                         numWavelengths, binIndexv, Ldsv);
    }

    bool computeDustLuminosities(const MediumState& state, int numCells, int numWavelengths,
                                 const vector<int>& dustMedia, const vector<double>& sectionAbsv,
                                 const double* primaryRadiationField, const double* secondaryRadiationField,
                                 vector<double>& primaryLuminosities, vector<double>& secondaryLuminosities)
    {
        return batchRuntime().computeDustAbsorbedLuminosities(state, numCells, numWavelengths, dustMedia, sectionAbsv,
                                                             primaryRadiationField, secondaryRadiationField,
                                                             primaryLuminosities, secondaryLuminosities);
    }

    bool computeTotalDustLuminosity(const MediumState& state, int numCells, int numWavelengths,
                                    const vector<int>& dustMedia, const vector<double>& sectionAbsv,
                                    const double* primaryRadiationField, const double* secondaryRadiationField,
                                    double& primaryLuminosity, double& secondaryLuminosity)
    {
        return batchRuntime().computeTotalDustAbsorbedLuminosity(state, numCells, numWavelengths, dustMedia,
                                                                sectionAbsv, primaryRadiationField,
                                                                secondaryRadiationField, primaryLuminosity,
                                                                secondaryLuminosity);
    }

    bool computeScatteringProperties(const MediumState& state, int cellIndex, const vector<double>& sectionScav,
                                     const vector<double>& sectionExtv, double& albedo, vector<double>& weights)
    {
        return runtime().computeScatteringProperties(state, cellIndex, sectionScav, sectionExtv, albedo, weights);
    }

    bool computeTableScatteringProperties(const MediumState& state, int cellIndex, const vector<int>& mediaIndexv,
                                          const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
                                          const vector<double>& lookupWavelengthv,
                                          const vector<double>& sectionScaTablev,
                                          const vector<double>& sectionExtTablev, double lambda, double& albedo,
                                          vector<double>& weights)
    {
        return runtime().computeTableScatteringProperties(state, cellIndex, mediaIndexv, lookupBeginv, lookupCountv,
                                                          lookupWavelengthv, sectionScaTablev, sectionExtTablev,
                                                          lambda, albedo, weights);
    }

    bool computeScatteringAlbedos(const MediumState& state, const vector<int>& cellv,
                                  const vector<double>& sectionScav, const vector<double>& sectionExtv,
                                  vector<double>& albedov)
    {
        return batchRuntime().computeScatteringAlbedos(state, cellv, sectionScav, sectionExtv, albedov);
    }

    bool computeTableScatteringAlbedos(const MediumState& state, const vector<int>& cellv,
                                       const vector<double>& lambdav, const vector<int>& mediaIndexv,
                                       const vector<int>& lookupBeginv,
                                       const vector<int>& lookupCountv,
                                       const vector<double>& lookupWavelengthv,
                                       const vector<double>& sectionScaTablev,
                                       const vector<double>& sectionExtTablev, vector<double>& albedov)
    {
        return batchRuntime().computeTableScatteringAlbedos(state, cellv, lambdav, mediaIndexv, lookupBeginv,
                                                            lookupCountv, lookupWavelengthv, sectionScaTablev,
                                                            sectionExtTablev, albedov);
    }

    bool computeScaleWavelengthValues(double* values, size_t numWavelengths, const vector<double>& factorv)
    {
        return batchRuntime().scaleWavelengthValues(values, numWavelengths, factorv);
    }

    bool computeScaleFrameWavelengthValues(double* values, size_t numWavelengths, size_t numPixelsInFrame,
                                           const vector<double>& factorv)
    {
        return batchRuntime().scaleFrameWavelengthValues(values, numWavelengths, numPixelsInFrame, factorv);
    }

    bool computeDivideValues(double* values, size_t numValues, double divisor)
    {
        return batchRuntime().divideValues(values, numValues, divisor);
    }

    bool computeMultiplyValues(double* values, size_t numValues, double factor)
    {
        return batchRuntime().multiplyValues(values, numValues, factor);
    }

    bool computeSumValues(double* output, size_t numValues, const double* value1, const double* value2,
                          const double* value3, const double* value4)
    {
        return batchRuntime().sumValues(output, numValues, value1, value2, value3, value4);
    }

    bool computeKeySums(const vector<int>& keyv, const vector<double>& valuev,
                        vector<int>& compactKeyv, vector<double>& compactValuev)
    {
        return batchRuntime().computeValueSumsByKey(keyv, valuev, compactKeyv, compactValuev);
    }

    bool computeAccumulateValuesByKey(const void* accumulatorKey, size_t numAccumulatorValues,
                                      const vector<int>& keyv, const vector<double>& valuev)
    {
        return batchRuntime().accumulateValuesByKey(accumulatorKey, numAccumulatorValues, keyv, valuev);
    }

    bool computeRetrieveAndClearAccumulatedValues(const void* accumulatorKey, double* values, size_t numValues)
    {
        bool ok = true;
        for (auto& candidate : runtimePool())
            ok = candidate->retrieveAndClearValueAccumulator(accumulatorKey, values, numValues) && ok;
        return ok;
    }

    void computeClearAccumulatedValues(const void* accumulatorKey)
    {
        for (auto& candidate : runtimePool()) candidate->clearValueAccumulator(accumulatorKey);
    }

    bool computeFrameBandTotalFluxSums(
        const vector<Position>& positionv, const vector<double>& wavelengthv, const vector<double>& luminosityv,
        const vector<double>& tauv, bool hasMedium, double costheta, double sintheta, double cosphi,
        double sinphi, double cosomega, double sinomega, int numPixelsX, int numPixelsY, double xpmin,
        double xpsiz, double ypmin, double ypsiz, double redshift, size_t numPixelsInFrame,
        const vector<int>& bandOffsetv, const vector<double>& bandWavelengthv,
        const vector<double>& bandTransmissionv, const vector<double>& bandWidthv,
        vector<int>& compactKeyv, vector<double>& compactValuev)
    {
        return batchRuntime().computeFrameBandTotalFluxSums(
            positionv, wavelengthv, luminosityv, tauv, hasMedium, costheta, sintheta, cosphi, sinphi, cosomega,
            sinomega, numPixelsX, numPixelsY, xpmin, xpsiz, ypmin, ypsiz, redshift, numPixelsInFrame,
            bandOffsetv, bandWavelengthv, bandTransmissionv, bandWidthv, compactKeyv, compactValuev);
    }

    bool computeFrameBandTotalFluxAccumulate(
        const void* accumulatorKey, size_t numAccumulatorValues,
        const vector<Position>& positionv, const vector<double>& wavelengthv, const vector<double>& luminosityv,
        const vector<double>& tauv, bool hasMedium, double costheta, double sintheta, double cosphi,
        double sinphi, double cosomega, double sinomega, int numPixelsX, int numPixelsY, double xpmin,
        double xpsiz, double ypmin, double ypsiz, double redshift, size_t numPixelsInFrame,
        const vector<int>& bandOffsetv, const vector<double>& bandWavelengthv,
        const vector<double>& bandTransmissionv, const vector<double>& bandWidthv)
    {
        vector<int> compactKeyv;
        vector<double> compactValuev;
        return batchRuntime().computeFrameBandTotalFluxSums(
            positionv, wavelengthv, luminosityv, tauv, hasMedium, costheta, sintheta, cosphi, sinphi, cosomega,
            sinomega, numPixelsX, numPixelsY, xpmin, xpsiz, ypmin, ypsiz, redshift, numPixelsInFrame,
            bandOffsetv, bandWavelengthv, bandTransmissionv, bandWidthv, compactKeyv, compactValuev,
            accumulatorKey, numAccumulatorValues);
    }

    bool computeHenyeyGreensteinScatteringLuminosities(const vector<double>& inputDirectionv,
                                                       const vector<double>& packetLuminosityv,
                                                       const vector<double>& lambdav, Direction bfkobs,
                                                       int lookupBegin, int lookupCount,
                                                       const vector<double>& lookupWavelengthv,
                                                       const vector<double>& asymmparv,
                                                       vector<double>& luminosityv)
    {
        return batchRuntime().henyeyGreensteinScatteringLuminosities(inputDirectionv, packetLuminosityv, lambdav,
                                                                    bfkobs, lookupBegin, lookupCount,
                                                                    lookupWavelengthv, asymmparv, luminosityv);
    }

    bool computeHenyeyGreensteinScatteringDirections(const vector<double>& inputDirectionv,
                                                     const vector<double>& lambdav,
                                                     const vector<double>& randomCosthetav,
                                                     const vector<double>& randomPhiv, int lookupBegin,
                                                     int lookupCount, const vector<double>& lookupWavelengthv,
                                                     const vector<double>& asymmparv,
                                                     vector<double>& outputDirectionv)
    {
        return batchRuntime().henyeyGreensteinScatteringDirections(inputDirectionv, lambdav, randomCosthetav,
                                                                   randomPhiv, lookupBegin, lookupCount,
                                                                   lookupWavelengthv, asymmparv, outputDirectionv);
    }

    bool computeIsotropicDirections(const vector<double>& randomCosthetav, const vector<double>& randomPhiv,
                                    vector<double>& outputDirectionv)
    {
        return batchRuntime().isotropicDirections(randomCosthetav, randomPhiv, outputDirectionv);
    }

    bool computeCompositeLaunchWeights(const double* luminosityv, size_t numValues, double spatialBias,
                                       double* weightv)
    {
        return batchRuntime().compositeLaunchWeights(luminosityv, numValues, spatialBias, weightv);
    }

    bool computeStoredTableCdf(vector<double>& luminosities, int numAxes, const vector<const double*>& axisData,
                               const vector<size_t>& axisSizes, const vector<bool>& axisLog, const double* quantity,
                               size_t quantityStep, bool quantityLog, bool clampFirstAxis, double xmin, double xmax,
                               const vector<double>& parameterValues, const vector<double>& scaleValues,
                               size_t numEntities)
    {
        return batchRuntime().storedTableCdf(luminosities, numAxes, axisData, axisSizes, axisLog, quantity,
                                             quantityStep, quantityLog, clampFirstAxis, xmin, xmax, parameterValues,
                                             scaleValues, numEntities);
    }

    bool computeStoredTableSampleWavelengths(vector<double>& wavelengths, vector<double>& specificLuminosities,
                                             int numAxes, const vector<const double*>& axisData,
                                             const vector<size_t>& axisSizes, const vector<bool>& axisLog,
                                             const double* quantity, size_t quantityStep, bool quantityLog,
                                             bool clampFirstAxis, double xmin, double xmax,
                                             const vector<double>& parameterValues,
                                             const vector<double>& intrinsicRandoms,
                                             const vector<double>& forcedWavelengths, size_t numSamples)
    {
        return batchRuntime().storedTableSampleWavelengths(wavelengths, specificLuminosities, numAxes, axisData,
                                                           axisSizes, axisLog, quantity, quantityStep, quantityLog,
                                                           clampFirstAxis, xmin, xmax, parameterValues,
                                                           intrinsicRandoms, forcedWavelengths, numSamples);
    }

    bool computeCumulativePathInteractionPoint(const SpatialGridPath* path, double tauinteract, bool hasAbsorption,
                                               int& m, double& s, double& tauAbs)
    {
        return runtime().findInteractionPointInCumulativePath(path, tauinteract, hasAbsorption, m, s, tauAbs);
    }

    bool computeCumulativePathInteractionPoints(const vector<const SpatialGridPath*>& paths,
                                                const vector<double>& tauinteractv, bool hasAbsorption,
                                                vector<int>& cellv, vector<double>& distancev,
                                                vector<double>& tauAbsv)
    {
        return batchRuntime().findInteractionPointsInCumulativePaths(paths, tauinteractv, hasAbsorption, cellv,
                                                                     distancev, tauAbsv);
    }

    bool computeForcedPropagationResults(const vector<const SpatialGridPath*>& paths,
                                         const vector<double>& tauinteractv,
                                         const vector<double>& pathBiasWeightv, bool hasAbsorption,
                                         const vector<double>& albedov, vector<int>& cellv,
                                         vector<double>& distancev, vector<double>& tauAbsv,
                                         vector<double>& weightv)
    {
        return batchRuntime().forcedPropagationResults(paths, tauinteractv, pathBiasWeightv, hasAbsorption, albedov,
                                                       cellv, distancev, tauAbsv, weightv);
    }

    bool computeForcedPropagationTableAlbedoResults(
        const vector<const SpatialGridPath*>& paths, const MediumState& state,
        const vector<double>& tauinteractv, const vector<double>& pathBiasWeightv,
        const vector<double>& lambdav, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
        const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
        const vector<double>& sectionScaTablev, const vector<double>& sectionExtTablev, vector<int>& cellv,
        vector<double>& distancev, vector<double>& tauAbsv, vector<double>& weightv)
    {
        return batchRuntime().forcedPropagationTableAlbedoResults(
            paths, state, tauinteractv, pathBiasWeightv, lambdav, mediaIndexv, lookupBeginv, lookupCountv,
            lookupWavelengthv, sectionScaTablev, sectionExtTablev, cellv, distancev, tauAbsv, weightv);
    }

    bool computeRadiationFieldSumsAndForcedPropagationTableAlbedoResults(
        const vector<const SpatialGridPath*>& paths, const MediumState& state,
        const vector<double>& luminosityv, const vector<int>& wavelengthBinv, int numWavelengths,
        const vector<double>& tauinteractv, const vector<double>& pathBiasWeightv,
        const vector<double>& lambdav, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
        const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
        const vector<double>& sectionScaTablev, const vector<double>& sectionExtTablev,
        vector<int>& binIndexv, vector<double>& Ldsv, vector<int>& cellv, vector<double>& distancev,
        vector<double>& tauAbsv, vector<double>& weightv)
    {
        return batchRuntime().radiationFieldSumsAndForcedPropagationTableAlbedoResults(
            paths, state, luminosityv, wavelengthBinv, numWavelengths, tauinteractv, pathBiasWeightv, lambdav,
            mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv, sectionScaTablev, sectionExtTablev,
            binIndexv, Ldsv, cellv, distancev, tauAbsv, weightv);
    }
}

#else

namespace
{
    bool computeContributions(const SpatialGridPath*, const MediumState&, const vector<double>&, const vector<double>*,
                              vector<double>&, vector<double>&)
    {
        return false;
    }

    bool computeTableContributions(const SpatialGridPath*, const MediumState&, const vector<int>&,
                                   const vector<int>&, const vector<int>&, const vector<double>&,
                                   const vector<double>&, const vector<double>*, double, vector<double>&,
                                   vector<double>&)
    {
        return false;
    }

    bool computeCumulativeTableOpticalDepthsBatch(const vector<SpatialGridPath*>&, const MediumState&,
                                                  const vector<int>&, const vector<int>&, const vector<int>&,
                                                  const vector<double>&, const vector<double>&,
                                                  const vector<double>*, const vector<double>&, vector<int>&,
                                                  vector<double>&, vector<double>&)
    {
        return false;
    }

    bool computeCartesianPath(const double*, const double*, const double*, int, int, int, const Position&,
                              const Direction&, double, double, double, double, double, double, double,
                              SpatialGridPath*)
    {
        return false;
    }

    bool computeTreePath(const void*, const vector<double>&, const vector<int>&, const vector<int>&,
                         const vector<int>&, const vector<int>&, const Position&, const Direction&, double, double,
                         double, double, double, double, double, SpatialGridPath*)
    {
        return false;
    }

    bool computeSphere1DPath(const double*, int, const Position&, const Direction&, double, SpatialGridPath*)
    {
        return false;
    }

    bool computeSphere2DPath(const double*, const double*, const double*, int, int, const Position&, const Direction&,
                             double, SpatialGridPath*)
    {
        return false;
    }

    bool computeSphere3DPath(const double*, const double*, const double*, const double*, const double*, const double*,
                             int, int, int, double, const Position&, const Direction&, double, SpatialGridPath*)
    {
        return false;
    }

    bool computeCylinder2DPath(const double*, const double*, int, int, const Position&, const Direction&, double,
                               SpatialGridPath*)
    {
        return false;
    }

    bool computeCylinder3DPath(const double*, const double*, const double*, const double*, const double*, int, int,
                               int, double, bool, const Position&, const Direction&, double, SpatialGridPath*)
    {
        return false;
    }

    bool computeTetraPath(const void*, const vector<double>&, const vector<int>&, const vector<int>&,
                          const vector<int>&, const vector<double>&, const vector<double>&, int, double,
                          const Position&, const Direction&, double, double, double, double, double, double, double,
                          SpatialGridPath*)
    {
        return false;
    }

    bool computeVoronoiPath(const void*, const vector<double>&, const vector<int>&, const vector<int>&,
                            const vector<int>&, int, double, const Position&, const Direction&, double, double, double,
                            double, double, double, double, SpatialGridPath*)
    {
        return false;
    }

    bool computeVoronoiPaths(const void*, const vector<double>&, const vector<int>&, const vector<int>&,
                             const vector<int>&, const vector<int>&, const vector<int>&, const vector<int>&, int, int,
                             double, double, double, double, double, double, double, double,
                             const vector<SpatialGridPath*>&)
    {
        return false;
    }

	    bool computeVoronoiTableOpticalDepthPaths(const void*, const vector<double>&, const vector<int>&,
	                                              const vector<int>&, const vector<int>&, const vector<int>&,
	                                              const vector<int>&, const vector<int>&, int, int, double, double,
	                                              double, double, double, double, double, double,
	                                              const vector<SpatialGridPath*>&, const MediumState&,
                                              const vector<int>&, const vector<int>&, const vector<int>&,
                                              const vector<double>&, const vector<double>&, const vector<double>*,
                                              const vector<double>&)
	    {
	        return false;
	    }

		    bool computeVoronoiTableRadiationFieldSumsAndForcedPropagationResults(
		        const void*, const vector<double>&, const vector<int>&, const vector<int>&, const vector<int>&,
		        const vector<int>&, const vector<int>&, const vector<int>&, int, int, double, double, double, double,
		        double, double, double, double, const vector<Position>&, const vector<Direction>&, const MediumState&,
		        const vector<double>&, const vector<int>&, int, const vector<double>&, const vector<double>&,
		        const vector<double>&, const vector<double>*, const vector<double>*, double,
		        const vector<int>&, const vector<int>&, const vector<int>&,
		        const vector<double>&, const vector<double>&, const vector<double>&, vector<int>&, vector<double>&,
		        vector<int>&, vector<double>&, vector<double>&, vector<double>&, const vector<double>*,
		        const vector<double>*, int, int, const vector<double>*, vector<double>*, const void* = nullptr,
                size_t = 0)
		    {
		        return false;
		    }

    bool computeVoronoiTableOpticalDepthTotals(const void*, const vector<double>&, const vector<int>&,
	                                               const vector<int>&, const vector<int>&, const vector<int>&,
                                               const vector<int>&, const vector<int>&, int, int, double, double,
                                               double, double, double, double, double, double,
                                               const vector<const SpatialGridPath*>&, const MediumState&,
                                               const vector<int>&, const vector<int>&, const vector<int>&,
                                               const vector<double>&, const vector<double>&, const vector<double>&,
                                               const vector<double>&, vector<double>&)
    {
        return false;
    }

    bool computeVoronoiTableOpticalDepthTotals(const void*, const vector<double>&, const vector<int>&,
                                               const vector<int>&, const vector<int>&, const vector<int>&,
                                               const vector<int>&, const vector<int>&, int, int, double, double,
                                               double, double, double, double, double, double,
                                               const vector<Position>&, const vector<Direction>&,
                                               const MediumState&, const vector<int>&, const vector<int>&,
                                               const vector<int>&, const vector<double>&, const vector<double>&,
                                               const vector<double>&, const vector<double>&, vector<double>&)
    {
        return false;
    }

    bool computeVoronoiTableOpticalDepthTotals(const void*, const vector<double>&, const vector<int>&,
                                               const vector<int>&, const vector<int>&, const vector<int>&,
                                               const vector<int>&, const vector<int>&, int, int, double, double,
                                               double, double, double, double, double, double,
                                               const vector<Position>&, const Direction&, const MediumState&,
                                               const vector<int>&, const vector<int>&, const vector<int>&,
                                               const vector<double>&, const vector<double>&, const vector<double>&,
                                               const vector<double>&, vector<double>&)
    {
        return false;
    }

	    bool computeVoronoiTableHenyeyGreensteinObservedLuminosities(
	        const void*, const vector<double>&, const vector<int>&, const vector<int>&, const vector<int>&,
	        const vector<int>&, const vector<int>&, const vector<int>&, int, int, double, double, double, double,
	        double, double, double, double, const vector<const SpatialGridPath*>&, const MediumState&,
        const vector<int>&, const vector<int>&, const vector<int>&, const vector<double>&, const vector<double>&,
        const vector<double>&, const vector<double>&, const vector<double>&, const vector<double>&,
        const vector<double>&, Direction, int, int, const vector<double>&, vector<double>&)
	    {
	        return false;
	    }

	    bool computeVoronoiTableHenyeyGreensteinFrameBandAccumulate(
	        const void*, const vector<double>&, const vector<int>&, const vector<int>&, const vector<int>&,
	        const vector<int>&, const vector<int>&, const vector<int>&, int, int, double, double, double, double,
	        double, double, double, double, const vector<Position>&, const MediumState&, const vector<int>&,
	        const vector<int>&, const vector<int>&, const vector<double>&, const vector<double>&,
	        const vector<double>&, const vector<double>&, const vector<double>&, const vector<double>&,
	        Direction, int, int, const vector<double>&, const void*, size_t, double, double, double, double,
	        double, double, int, int, double, double, double, double, double, size_t, const vector<int>&,
	        const vector<double>&, const vector<double>&, const vector<double>&)
	    {
	        return false;
	    }

	    bool computeCumulativeOpticalDepths(const SpatialGridPath*, const MediumState&, const vector<double>&,
	                                        const vector<double>*, vector<double>&, vector<double>&)
    {
        return false;
    }

    bool computeCumulativeOpticalDepthsBatch(const vector<SpatialGridPath*>&, const MediumState&,
                                             const vector<double>&, const vector<double>*, vector<int>&,
                                             vector<double>&, vector<double>&)
    {
        return false;
    }

    bool computeOpticalDepth(const SpatialGridPath*, const MediumState&, const vector<double>&, double, double&)
    {
        return false;
    }

    bool computeInteractionPointExtinction(const SpatialGridPath*, const MediumState&, const vector<double>&, double,
                                           bool&, int&, double&)
    {
        return false;
    }

    bool computeInteractionPointScatteringAndAbsorption(const SpatialGridPath*, const MediumState&,
                                                        const vector<double>&, const vector<double>&, double, bool&,
                                                        int&, double&, double&)
    {
        return false;
    }

    string lastRuntimeError()
    {
        return "runtime loading is not implemented on this platform";
    }

    bool computeRadiationContributions(const SpatialGridPath*, double, vector<double>&)
    {
        return false;
    }

    bool computeRadiationContributionsBatch(const vector<const SpatialGridPath*>&, const vector<double>&,
                                            vector<int>&, vector<double>&)
    {
        return false;
    }

    bool computeDustLuminosities(const MediumState&, int, int, const vector<int>&, const vector<double>&, const double*,
                                 const double*, vector<double>&, vector<double>&)
    {
        return false;
    }

    bool computeTotalDustLuminosity(const MediumState&, int, int, const vector<int>&, const vector<double>&,
                                    const double*, const double*, double&, double&)
    {
        return false;
    }

    bool computeScatteringProperties(const MediumState&, int, const vector<double>&, const vector<double>&, double&,
                                     vector<double>&)
    {
        return false;
    }

    bool computeTableScatteringProperties(const MediumState&, int, const vector<int>&, const vector<int>&,
                                          const vector<int>&, const vector<double>&, const vector<double>&,
                                          const vector<double>&, double, double&, vector<double>&)
    {
        return false;
    }

    bool computeScatteringAlbedos(const MediumState&, const vector<int>&, const vector<double>&,
                                  const vector<double>&, vector<double>&)
    {
        return false;
    }

    bool computeTableScatteringAlbedos(const MediumState&, const vector<int>&, const vector<double>&,
                                       const vector<int>&, const vector<int>&, const vector<int>&,
                                       const vector<double>&, const vector<double>&, const vector<double>&,
                                       vector<double>&)
    {
        return false;
    }

    bool computeScaleWavelengthValues(double*, size_t, const vector<double>&)
    {
        return false;
    }

    bool computeScaleFrameWavelengthValues(double*, size_t, size_t, const vector<double>&)
    {
        return false;
    }

    bool computeDivideValues(double*, size_t, double)
    {
        return false;
    }

    bool computeMultiplyValues(double*, size_t, double)
    {
        return false;
    }

    bool computeSumValues(double*, size_t, const double*, const double*, const double*, const double*)
    {
        return false;
    }

    bool computeKeySums(const vector<int>&, const vector<double>&, vector<int>&, vector<double>&)
    {
        return false;
    }

    bool computeAccumulateValuesByKey(const void*, size_t, const vector<int>&, const vector<double>&)
    {
        return false;
    }

    bool computeRetrieveAndClearAccumulatedValues(const void*, double*, size_t)
    {
        return false;
    }

    void computeClearAccumulatedValues(const void*) {}

    bool computeFrameBandTotalFluxSums(const vector<Position>&, const vector<double>&, const vector<double>&,
                                       const vector<double>&, bool, double, double, double, double, double,
                                       double, int, int, double, double, double, double, double, size_t,
                                       const vector<int>&, const vector<double>&, const vector<double>&,
                                       const vector<double>&, vector<int>&, vector<double>&)
    {
        return false;
    }

    bool computeFrameBandTotalFluxAccumulate(const void*, size_t, const vector<Position>&, const vector<double>&,
                                             const vector<double>&, const vector<double>&, bool, double, double,
                                             double, double, double, double, int, int, double, double, double,
                                             double, double, size_t, const vector<int>&, const vector<double>&,
                                             const vector<double>&, const vector<double>&)
    {
        return false;
    }

    bool computeHenyeyGreensteinScatteringLuminosities(const vector<double>&, const vector<double>&,
                                                       const vector<double>&, Direction, int, int,
                                                       const vector<double>&, const vector<double>&,
                                                       vector<double>&)
    {
        return false;
    }

    bool computeHenyeyGreensteinScatteringDirections(const vector<double>&, const vector<double>&,
                                                     const vector<double>&, const vector<double>&, int, int,
                                                     const vector<double>&, const vector<double>&,
                                                     vector<double>&)
    {
        return false;
    }

    bool computeIsotropicDirections(const vector<double>&, const vector<double>&, vector<double>&)
    {
        return false;
    }

    bool computeCompositeLaunchWeights(const double*, size_t, double, double*)
    {
        return false;
    }

    bool computeStoredTableCdf(vector<double>&, int, const vector<const double*>&, const vector<size_t>&,
                               const vector<bool>&, const double*, size_t, bool, bool, double, double,
                               const vector<double>&, const vector<double>&, size_t)
    {
        return false;
    }

    bool computeStoredTableSampleWavelengths(vector<double>&, vector<double>&, int, const vector<const double*>&,
                                             const vector<size_t>&, const vector<bool>&, const double*, size_t, bool,
                                             bool, double, double, const vector<double>&, const vector<double>&,
                                             const vector<double>&, size_t)
    {
        return false;
    }

    bool computeCumulativePathInteractionPoint(const SpatialGridPath*, double, bool, int&, double&, double&)
    {
        return false;
    }

    bool computeCumulativePathInteractionPoints(const vector<const SpatialGridPath*>&, const vector<double>&, bool,
                                                vector<int>&, vector<double>&, vector<double>&)
    {
        return false;
    }

    bool computeForcedPropagationResults(const vector<const SpatialGridPath*>&, const vector<double>&,
                                         const vector<double>&, bool, const vector<double>&, vector<int>&,
                                         vector<double>&, vector<double>&, vector<double>&)
    {
        return false;
    }

    bool computeForcedPropagationTableAlbedoResults(
        const vector<const SpatialGridPath*>&, const MediumState&, const vector<double>&, const vector<double>&,
        const vector<double>&, const vector<int>&, const vector<int>&, const vector<int>&, const vector<double>&,
        const vector<double>&, const vector<double>&, vector<int>&, vector<double>&, vector<double>&,
        vector<double>&)
    {
        return false;
    }

    bool computeRadiationFieldSumsAndForcedPropagationTableAlbedoResults(
        const vector<const SpatialGridPath*>&, const MediumState&, const vector<double>&, const vector<int>&, int,
        const vector<double>&, const vector<double>&, const vector<double>&, const vector<int>&,
        const vector<int>&, const vector<int>&, const vector<double>&, const vector<double>&,
        const vector<double>&, vector<int>&, vector<double>&, vector<int>&, vector<double>&,
        vector<double>&, vector<double>&)
    {
        return false;
    }
}

#endif

////////////////////////////////////////////////////////////////////

void GpuAcceleration::setProcessEnabled(bool enabled)
{
    _processEnabled = enabled;
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::isProcessEnabled()
{
    return _processEnabled || isTruthy(std::getenv("SKIRTGPU")) || isTruthy(std::getenv("SKIRT_GPU"));
}

////////////////////////////////////////////////////////////////////

string GpuAcceleration::status()
{
    if (!isProcessEnabled()) return "disabled";
#if defined(__unix__) || defined(__APPLE__)
    return runtimeStatus();
#else
    return "unavailable: runtime loading is not implemented on this platform";
#endif
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::isPathGenerationEnabled()
{
    return isProcessEnabled() && pathGenerationEnabled();
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::isSynchronousPhotonCycleEnabled()
{
    return isProcessEnabled() && synchronousPhotonCycleEnabled();
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::selfTest(string& message)
{
    if (!isProcessEnabled())
    {
        message = "GPU acceleration is disabled";
        return false;
    }

    const double tolerance = 1e-12;
    vector<double> xBorders{-1.2, -0.5, 0.1, 0.7, 1.4, 2.0};
    vector<double> yBorders{-0.8, -0.2, 0.25, 0.9};
    vector<double> zBorders{-0.4, 0.0, 0.35, 0.8, 1.3};
    auto locateBasic = [](const vector<double>& xv, double x, int n) {
        int jl = -1;
        int ju = n;
        while (ju - jl > 1)
        {
            int jm = (ju + jl) >> 1;
            if (x < xv[jm])
                ju = jm;
            else
                jl = jm;
        }
        return jl;
    };
    auto locateClip = [&locateBasic](const vector<double>& xv, double x) {
        if (x < xv.front()) return 0;
        return locateBasic(xv, x, static_cast<int>(xv.size()) - 1);
    };
    auto locateFail = [&locateBasic](const vector<double>& xv, double x) {
        if (x > xv.back()) return -1;
        return locateBasic(xv, x, static_cast<int>(xv.size()) - 1);
    };
    auto referenceCartesianPath = [&xBorders, &yBorders, &zBorders, &locateClip](const Position& position,
                                                                                 const Direction& direction,
                                                                                 double maxDistance) {
        SpatialGridPath path(position, direction);
        double rx, ry, rz;
        position.cartesian(rx, ry, rz);
        double kx, ky, kz;
        direction.cartesian(kx, ky, kz);
        double xmin = xBorders.front();
        double ymin = yBorders.front();
        double zmin = zBorders.front();
        double xmax = xBorders.back();
        double ymax = yBorders.back();
        double zmax = zBorders.back();
        double dx = xmax - xmin;
        double dy = ymax - ymin;
        double dz = zmax - zmin;
        double eps = 1e-12 * sqrt(dx * dx + dy * dy + dz * dz);
        double cumds = 0.;

        if (rx <= xmin)
        {
            if (kx <= 0.) return path;
            double ds = (xmin - rx) / kx;
            rx = xmin + eps;
            ry += ky * ds;
            rz += kz * ds;
            cumds += ds;
        }
        else if (rx >= xmax)
        {
            if (kx >= 0.) return path;
            double ds = (xmax - rx) / kx;
            rx = xmax - eps;
            ry += ky * ds;
            rz += kz * ds;
            cumds += ds;
        }

        if (ry <= ymin)
        {
            if (ky <= 0.) return path;
            double ds = (ymin - ry) / ky;
            rx += kx * ds;
            ry = ymin + eps;
            rz += kz * ds;
            cumds += ds;
        }
        else if (ry >= ymax)
        {
            if (ky >= 0.) return path;
            double ds = (ymax - ry) / ky;
            rx += kx * ds;
            ry = ymax - eps;
            rz += kz * ds;
            cumds += ds;
        }

        if (rz <= zmin)
        {
            if (kz <= 0.) return path;
            double ds = (zmin - rz) / kz;
            rx += kx * ds;
            ry += ky * ds;
            rz = zmin + eps;
            cumds += ds;
        }
        else if (rz >= zmax)
        {
            if (kz >= 0.) return path;
            double ds = (zmax - rz) / kz;
            rx += kx * ds;
            ry += ky * ds;
            rz = zmax - eps;
            cumds += ds;
        }

        if (!(rx >= xmin && rx <= xmax && ry >= ymin && ry <= ymax && rz >= zmin && rz <= zmax)) return path;
        if (cumds > 0.)
        {
            path.addSegment(-1, cumds);
            if (path.segments().back().s() > maxDistance) return path;
        }

        int nx = static_cast<int>(xBorders.size()) - 1;
        int ny = static_cast<int>(yBorders.size()) - 1;
        int nz = static_cast<int>(zBorders.size()) - 1;
        int i = locateClip(xBorders, rx);
        int j = locateClip(yBorders, ry);
        int k = locateClip(zBorders, rz);
        while (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
        {
            int m = k + nz * j + nz * ny * i;
            double xE = (kx < 0.0) ? xBorders[i] : xBorders[i + 1];
            double yE = (ky < 0.0) ? yBorders[j] : yBorders[j + 1];
            double zE = (kz < 0.0) ? zBorders[k] : zBorders[k + 1];
            double dsx = (fabs(kx) > 1e-15) ? (xE - rx) / kx : DBL_MAX;
            double dsy = (fabs(ky) > 1e-15) ? (yE - ry) / ky : DBL_MAX;
            double dsz = (fabs(kz) > 1e-15) ? (zE - rz) / kz : DBL_MAX;

            int axis = 2;
            double ds = dsz;
            if (dsx <= dsy && dsx <= dsz)
            {
                axis = 0;
                ds = dsx;
            }
            else if (dsy < dsx && dsy <= dsz)
            {
                axis = 1;
                ds = dsy;
            }

            if (ds > 0.)
            {
                path.addSegment(m, ds);
                if (path.segments().back().s() > maxDistance) return path;
            }

            if (axis == 0)
            {
                rx = xE;
                ry += ky * ds;
                rz += kz * ds;
                i += (kx < 0.0) ? -1 : 1;
            }
            else if (axis == 1)
            {
                rx += kx * ds;
                ry = yE;
                rz += kz * ds;
                j += (ky < 0.0) ? -1 : 1;
            }
            else
            {
                rx += kx * ds;
                ry += ky * ds;
                rz = zE;
                k += (kz < 0.0) ? -1 : 1;
            }
        }
        return path;
    };
    auto checkCartesianPath = [&message, &xBorders, &yBorders, &zBorders, &referenceCartesianPath,
                               tolerance](const Position& position, const Direction& direction, double maxDistance) {
        SpatialGridPath expected = referenceCartesianPath(position, direction, maxDistance);
        SpatialGridPath actual(position, direction);
        if (!computeCartesianPath(xBorders.data(), yBorders.data(), zBorders.data(),
                                  static_cast<int>(xBorders.size()) - 1, static_cast<int>(yBorders.size()) - 1,
                                  static_cast<int>(zBorders.size()) - 1, position, direction, xBorders.front(),
                                  yBorders.front(), zBorders.front(), xBorders.back(), yBorders.back(),
                                  zBorders.back(), maxDistance, &actual))
        {
            message = "GPU Cartesian grid path kernel failed: " + lastRuntimeError();
            return false;
        }
        if (actual.segments().size() != expected.segments().size())
        {
            message = "Cartesian grid path segment-count mismatch";
            return false;
        }
        for (size_t i = 0; i != expected.segments().size(); ++i)
        {
            const auto& a = actual.segments()[i];
            const auto& e = expected.segments()[i];
            if (a.m() != e.m() || abs(a.ds() - e.ds()) > tolerance * max(1., abs(e.ds()))
                || abs(a.s() - e.s()) > tolerance * max(1., abs(e.s())))
            {
                message = "Cartesian grid path segment mismatch";
                return false;
            }
        }
        return true;
    };
    if (!checkCartesianPath(Position(-2.1, -0.55, 0.2), Direction(1.0, 0.26, 0.08, true),
                            std::numeric_limits<double>::infinity())
        || !checkCartesianPath(Position(0.1, -0.2, 0.0), Direction(1.0, 1.0, 1.0, true),
                               std::numeric_limits<double>::infinity())
        || !checkCartesianPath(Position(-2.1, -0.55, 0.2), Direction(1.0, 0.26, 0.08, true), 2.0))
        return false;

    vector<double> treeBounds;
    vector<int> treeChildBegin;
    vector<int> treeChildCount;
    vector<int> treeChildIndex;
    vector<int> treeCellIndex;
    auto addTreeNode = [&treeBounds, &treeChildBegin, &treeChildCount, &treeCellIndex](double xmin, double ymin,
                                                                                       double zmin, double xmax,
                                                                                       double ymax, double zmax,
                                                                                       int cellIndex) {
        int id = static_cast<int>(treeCellIndex.size());
        treeBounds.insert(treeBounds.end(), {xmin, ymin, zmin, xmax, ymax, zmax});
        treeChildBegin.push_back(0);
        treeChildCount.push_back(0);
        treeCellIndex.push_back(cellIndex);
        return id;
    };
    addTreeNode(0., 0., 0., 1., 1., 1., -1);
    for (int z = 0; z != 2; ++z)
        for (int y = 0; y != 2; ++y)
            for (int x = 0; x != 2; ++x)
            {
                double xmin = 0.5 * x;
                double ymin = 0.5 * y;
                double zmin = 0.5 * z;
                int childOrdinal = x + 2 * y + 4 * z;
                int cellIndex = childOrdinal == 7 ? -1 : childOrdinal;
                addTreeNode(xmin, ymin, zmin, xmin + 0.5, ymin + 0.5, zmin + 0.5, cellIndex);
            }
    for (int z = 0; z != 2; ++z)
        for (int y = 0; y != 2; ++y)
            for (int x = 0; x != 2; ++x)
            {
                double xmin = 0.5 + 0.25 * x;
                double ymin = 0.5 + 0.25 * y;
                double zmin = 0.5 + 0.25 * z;
                int childOrdinal = x + 2 * y + 4 * z;
                addTreeNode(xmin, ymin, zmin, xmin + 0.25, ymin + 0.25, zmin + 0.25, 7 + childOrdinal);
            }
    treeChildBegin[0] = 0;
    treeChildCount[0] = 8;
    for (int id = 1; id <= 8; ++id) treeChildIndex.push_back(id);
    treeChildBegin[8] = treeChildIndex.size();
    treeChildCount[8] = 8;
    for (int id = 9; id <= 16; ++id) treeChildIndex.push_back(id);

    auto treeContains = [&treeBounds](int node, double x, double y, double z) {
        int base = 6 * node;
        return x >= treeBounds[base] && x <= treeBounds[base + 3] && y >= treeBounds[base + 1]
               && y <= treeBounds[base + 4] && z >= treeBounds[base + 2] && z <= treeBounds[base + 5];
    };
    auto findTreeLeaf = [&treeChildBegin, &treeChildCount, &treeChildIndex, &treeCellIndex,
                         &treeContains](double x, double y, double z) {
        if (!treeContains(0, x, y, z)) return -1;
        int node = 0;
        for (int guard = 0; guard != static_cast<int>(treeCellIndex.size()); ++guard)
        {
            int count = treeChildCount[node];
            if (count <= 0) return node;
            int begin = treeChildBegin[node];
            int next = -1;
            for (int i = count - 1; i >= 0; --i)
            {
                int child = treeChildIndex[begin + i];
                if (treeContains(child, x, y, z))
                {
                    next = child;
                    break;
                }
            }
            if (next < 0) return -1;
            node = next;
        }
        return -1;
    };
    auto referenceTreePath = [&treeBounds, &treeCellIndex, &findTreeLeaf, &treeContains](const Position& position,
                                                                                        const Direction& direction,
                                                                                        double maxDistance) {
        SpatialGridPath path(position, direction);
        double rx, ry, rz;
        position.cartesian(rx, ry, rz);
        double kx, ky, kz;
        direction.cartesian(kx, ky, kz);
        double xmin = treeBounds[0];
        double ymin = treeBounds[1];
        double zmin = treeBounds[2];
        double xmax = treeBounds[3];
        double ymax = treeBounds[4];
        double zmax = treeBounds[5];
        double dx = xmax - xmin;
        double dy = ymax - ymin;
        double dz = zmax - zmin;
        double eps = 1e-12 * sqrt(dx * dx + dy * dy + dz * dz);
        double cumds = 0.;

        if (rx <= xmin)
        {
            if (kx <= 0.) return path;
            double ds = (xmin - rx) / kx;
            rx = xmin + eps;
            ry += ky * ds;
            rz += kz * ds;
            cumds += ds;
        }
        else if (rx >= xmax)
        {
            if (kx >= 0.) return path;
            double ds = (xmax - rx) / kx;
            rx = xmax - eps;
            ry += ky * ds;
            rz += kz * ds;
            cumds += ds;
        }
        if (ry <= ymin)
        {
            if (ky <= 0.) return path;
            double ds = (ymin - ry) / ky;
            rx += kx * ds;
            ry = ymin + eps;
            rz += kz * ds;
            cumds += ds;
        }
        else if (ry >= ymax)
        {
            if (ky >= 0.) return path;
            double ds = (ymax - ry) / ky;
            rx += kx * ds;
            ry = ymax - eps;
            rz += kz * ds;
            cumds += ds;
        }
        if (rz <= zmin)
        {
            if (kz <= 0.) return path;
            double ds = (zmin - rz) / kz;
            rx += kx * ds;
            ry += ky * ds;
            rz = zmin + eps;
            cumds += ds;
        }
        else if (rz >= zmax)
        {
            if (kz >= 0.) return path;
            double ds = (zmax - rz) / kz;
            rx += kx * ds;
            ry += ky * ds;
            rz = zmax - eps;
            cumds += ds;
        }
        if (!treeContains(0, rx, ry, rz)) return path;

        int node = findTreeLeaf(rx, ry, rz);
        if (node < 0) return path;
        if (cumds > 0.)
        {
            path.addSegment(-1, cumds);
            if (path.segments().back().s() > maxDistance) return path;
        }

        while (node >= 0)
        {
            int base = 6 * node;
            double xnext = (kx < 0.) ? treeBounds[base] : treeBounds[base + 3];
            double ynext = (ky < 0.) ? treeBounds[base + 1] : treeBounds[base + 4];
            double znext = (kz < 0.) ? treeBounds[base + 2] : treeBounds[base + 5];
            double dsx = (fabs(kx) > 1e-15) ? (xnext - rx) / kx : DBL_MAX;
            double dsy = (fabs(ky) > 1e-15) ? (ynext - ry) / ky : DBL_MAX;
            double dsz = (fabs(kz) > 1e-15) ? (znext - rz) / kz : DBL_MAX;
            double ds = dsz;
            if (dsx <= dsy && dsx <= dsz)
                ds = dsx;
            else if (dsy <= dsx && dsy <= dsz)
                ds = dsy;

            if (ds > 0.)
            {
                path.addSegment(treeCellIndex[node], ds);
                if (path.segments().back().s() > maxDistance) return path;
            }

            int oldNode = node;
            double step = ds + eps;
            rx += kx * step;
            ry += ky * step;
            rz += kz * step;
            node = findTreeLeaf(rx, ry, rz);
            if (node == oldNode)
            {
                rx = std::nextafter(rx, (kx < 0.) ? -DBL_MAX : DBL_MAX);
                ry = std::nextafter(ry, (ky < 0.) ? -DBL_MAX : DBL_MAX);
                rz = std::nextafter(rz, (kz < 0.) ? -DBL_MAX : DBL_MAX);
                node = findTreeLeaf(rx, ry, rz);
            }
            if (node == oldNode) node = -1;
        }
        return path;
    };
    auto checkTreePath = [&message, &treeBounds, &treeChildBegin, &treeChildCount, &treeChildIndex, &treeCellIndex,
                          &referenceTreePath, tolerance](const Position& position, const Direction& direction,
                                                         double maxDistance) {
        SpatialGridPath expected = referenceTreePath(position, direction, maxDistance);
        SpatialGridPath actual(position, direction);
        if (!computeTreePath(treeBounds.data(), treeBounds, treeChildBegin, treeChildCount, treeChildIndex,
                             treeCellIndex, position, direction, treeBounds[0], treeBounds[1], treeBounds[2],
                             treeBounds[3], treeBounds[4], treeBounds[5], maxDistance, &actual))
        {
            message = "GPU tree grid path kernel failed: " + lastRuntimeError();
            return false;
        }
        if (actual.segments().size() != expected.segments().size())
        {
            message = "tree grid path segment-count mismatch";
            return false;
        }
        for (size_t i = 0; i != expected.segments().size(); ++i)
        {
            const auto& a = actual.segments()[i];
            const auto& e = expected.segments()[i];
            if (a.m() != e.m() || abs(a.ds() - e.ds()) > tolerance * max(1., abs(e.ds()))
                || abs(a.s() - e.s()) > tolerance * max(1., abs(e.s())))
            {
                message = "tree grid path segment mismatch";
                return false;
            }
        }
        return true;
    };
    if (!checkTreePath(Position(-0.25, 0.2, 0.25), Direction(1.0, 0.42, 0.35, true),
                       std::numeric_limits<double>::infinity())
        || !checkTreePath(Position(0.25, 0.25, 0.25), Direction(1.0, 1.0, 1.0, true),
                          std::numeric_limits<double>::infinity())
        || !checkTreePath(Position(-0.25, 0.2, 0.25), Direction(1.0, 0.42, 0.35, true), 0.8))
        return false;

    vector<double> radialBorders{0.15, 0.35, 0.7, 1.05, 1.6};
    auto locateRadial = [&locateBasic](const vector<double>& xv, double x) {
        if (x == xv.back()) return static_cast<int>(xv.size()) - 2;
        return locateBasic(xv, x, static_cast<int>(xv.size()));
    };
    auto referenceSphere1DPath = [&radialBorders, &locateRadial, &locateClip](const Position& position,
                                                                               const Direction& direction,
                                                                               double maxDistance) {
        SpatialGridPath path(position, direction);
        double rx, ry, rz;
        position.cartesian(rx, ry, rz);
        double kx, ky, kz;
        direction.cartesian(kx, ky, kz);
        int nr = static_cast<int>(radialBorders.size()) - 1;
        double r = sqrt(rx * rx + ry * ry + rz * rz);
        double q = rx * kx + ry * ky + rz * kz;
        double p = sqrt((r - q) * (r + q));
        int i = -1;
        int imin = -1;
        bool inwards = false;
        auto addSegment = [&path, maxDistance](int m, double ds) {
            path.addSegment(m, ds);
            return !path.segments().empty() && path.segments().back().s() > maxDistance;
        };

        if (r > radialBorders.back())
        {
            if (q > 0. || p > radialBorders.back()) return path;
            double qmax = -sqrt((radialBorders.back() - p) * (radialBorders.back() + p));
            double ds = qmax - q;
            i = nr - 1;
            q = qmax;
            imin = locateRadial(radialBorders, p);
            inwards = i > imin;
            if (ds > 0. && addSegment(-1, ds)) return path;
        }
        else if (r < radialBorders.front())
        {
            double qmin = sqrt((radialBorders.front() - p) * (radialBorders.front() + p));
            double ds = qmin - q;
            i = 0;
            q = qmin;
            inwards = false;
            if (ds > 0. && addSegment(-1, ds)) return path;
        }
        else
        {
            i = locateClip(radialBorders, r);
            inwards = false;
            if (q < 0.)
            {
                imin = locateRadial(radialBorders, p);
                if (i > imin) inwards = true;
            }
        }

        while (i < nr)
        {
            int m = i;
            double ds = 0.;
            if (inwards)
            {
                double rN = radialBorders[i];
                double qN = -sqrt((rN - p) * (rN + p));
                ds = qN - q;
                --i;
                q = qN;
                if (i <= imin) inwards = false;
            }
            else
            {
                double rN = radialBorders[i + 1];
                double qN = sqrt((rN - p) * (rN + p));
                ds = qN - q;
                ++i;
                q = qN;
            }
            if (ds > 0. && addSegment(m, ds)) return path;
        }
        return path;
    };
    auto checkSphere1DPath = [&message, &radialBorders, &referenceSphere1DPath, tolerance](
                                 const Position& position, const Direction& direction, double maxDistance) {
        SpatialGridPath expected = referenceSphere1DPath(position, direction, maxDistance);
        SpatialGridPath actual(position, direction);
        if (!computeSphere1DPath(radialBorders.data(), static_cast<int>(radialBorders.size()) - 1, position,
                                 direction, maxDistance, &actual))
        {
            message = "GPU spherical 1D grid path kernel failed: " + lastRuntimeError();
            return false;
        }
        if (actual.segments().size() != expected.segments().size())
        {
            message = "spherical 1D grid path segment-count mismatch";
            return false;
        }
        for (size_t i = 0; i != expected.segments().size(); ++i)
        {
            const auto& a = actual.segments()[i];
            const auto& e = expected.segments()[i];
            if (a.m() != e.m() || abs(a.ds() - e.ds()) > 1e-11 * max(1., abs(e.ds()))
                || abs(a.s() - e.s()) > 1e-11 * max(1., abs(e.s())))
            {
                message = "spherical 1D grid path segment mismatch";
                return false;
            }
        }
        return true;
    };
    if (!checkSphere1DPath(Position(2.0, 0.2, -0.1), Direction(-1.0, -0.18, 0.05, true),
                           std::numeric_limits<double>::infinity())
        || !checkSphere1DPath(Position(0.05, 0.01, 0.0), Direction(1.0, 0.4, 0.2, true),
                              std::numeric_limits<double>::infinity())
        || !checkSphere1DPath(Position(0.4, 0.0, 0.0), Direction(-1.0, 0.0, 0.0, false),
                              std::numeric_limits<double>::infinity())
        || !checkSphere1DPath(Position(2.0, 0.2, -0.1), Direction(-1.0, -0.18, 0.05, true), 1.5))
        return false;

    vector<double> sphere2DRv{0.2, 0.45, 0.9, 1.35, 1.8};
    vector<double> sphere2DThetav{0.0, 0.7, M_PI_2, 2.35, M_PI};
    vector<double> sphere2DCv;
    for (double theta : sphere2DThetav) sphere2DCv.push_back(cos(theta));
    sphere2DCv.front() = 1.;
    sphere2DCv.back() = -1.;
    auto quadraticSmallestPositive = [](double b, double c) {
        if (b * b > c)
        {
            if (b > 0.)
            {
                if (c < 0.)
                {
                    double x1 = -b - sqrt(b * b - c);
                    return c / x1;
                }
            }
            else
            {
                double x2 = -b + sqrt(b * b - c);
                if (c > 0.)
                {
                    double x1 = c / x2;
                    if (x1 < x2) return x1;
                }
                return x2;
            }
        }
        return 0.;
    };
    auto scaledSmallestPositive = [&quadraticSmallestPositive](double a, double b, double c) {
        if (fabs(a) > 1e-9) return quadraticSmallestPositive(b / a, c / a);
        double x = -0.5 * c / b;
        return x > 0. ? x : 0.;
    };
    auto locateArray = [&locateBasic](const vector<double>& xv, double x) {
        if (x == xv.back()) return static_cast<int>(xv.size()) - 2;
        return locateBasic(xv, x, static_cast<int>(xv.size()));
    };
    auto referenceSphere2DPath = [&sphere2DRv, &sphere2DThetav, &sphere2DCv, &quadraticSmallestPositive,
                                  &scaledSmallestPositive, &locateArray, &locateClip](
                                     const Position& position, const Direction& direction, double maxDistance) {
        SpatialGridPath path(position, direction);
        double rx, ry, rz;
        position.cartesian(rx, ry, rz);
        double kx, ky, kz;
        direction.cartesian(kx, ky, kz);
        int nr = static_cast<int>(sphere2DRv.size()) - 1;
        int ntheta = static_cast<int>(sphere2DThetav.size()) - 1;
        double eps = 1e-11 * sphere2DRv.back();
        int i = -1;
        int j = -1;
        auto setCellIndices = [&]() {
            double radius = sqrt(rx * rx + ry * ry + rz * rz);
            double theta = radius == 0. ? 0. : acos(rz / radius);
            i = locateArray(sphere2DRv, radius);
            j = locateClip(sphere2DThetav, theta);
            return i < nr;
        };
        auto addSegment = [&path, maxDistance](int m, double ds) {
            path.addSegment(m, ds);
            return !path.segments().empty() && path.segments().back().s() > maxDistance;
        };

        double radius = sqrt(rx * rx + ry * ry + rz * rz);
        if (radius > sphere2DRv.back())
        {
            double ds = quadraticSmallestPositive(rx * kx + ry * ky + rz * kz,
                                                 rx * rx + ry * ry + rz * rz
                                                     - sphere2DRv.back() * sphere2DRv.back());
            if (ds <= 0.) return path;
            rx += kx * (ds + eps);
            ry += ky * (ds + eps);
            rz += kz * (ds + eps);
            if (!setCellIndices()) return path;
            if (addSegment(-1, ds)) return path;
        }
        else
        {
            if (radius < eps)
            {
                rx += kx * eps;
                ry += ky * eps;
                rz += kz * eps;
            }
            if (!setCellIndices()) return path;
        }

        int guard = 0;
        int maxGuard = 16 * (nr + ntheta + 1);
        while (guard++ < maxGuard)
        {
            int icur = i;
            int jcur = j;
            double ds = DBL_MAX;
            double r2 = rx * rx + ry * ry + rz * rz;
            double rk = rx * kx + ry * ky + rz * kz;

            if (icur > 0 || (icur == 0 && sphere2DRv[0] > 0.))
            {
                double rB = sphere2DRv[icur];
                double s = quadraticSmallestPositive(rk, r2 - rB * rB);
                if (s > 0. && s < ds)
                {
                    ds = s;
                    i = icur - 1;
                    j = jcur;
                }
            }

            {
                double rB = sphere2DRv[icur + 1];
                double s = quadraticSmallestPositive(rk, r2 - rB * rB);
                if (s > 0. && s < ds)
                {
                    ds = s;
                    i = icur + 1;
                    j = jcur;
                }
            }

            if (jcur > 0)
            {
                double c = sphere2DCv[jcur];
                double s = c ? scaledSmallestPositive(c * c - kz * kz, c * c * rk - rz * kz, c * c * r2 - rz * rz)
                             : -rz / kz;
                if (s > 0. && s < ds)
                {
                    ds = s;
                    i = icur;
                    j = jcur - 1;
                }
            }

            if (jcur < ntheta - 1)
            {
                double c = sphere2DCv[jcur + 1];
                double s = c ? scaledSmallestPositive(c * c - kz * kz, c * c * rk - rz * kz, c * c * r2 - rz * rz)
                             : -rz / kz;
                if (s > 0. && s < ds)
                {
                    ds = s;
                    i = icur;
                    j = jcur + 1;
                }
            }

            if (i != icur || j != jcur)
            {
                if (addSegment(jcur + ntheta * icur, ds)) return path;
                rx += kx * (ds + eps);
                ry += ky * (ds + eps);
                rz += kz * (ds + eps);
                if (i >= nr) return path;
            }
            else
            {
                rx += kx * eps;
                ry += ky * eps;
                rz += kz * eps;
                if (!setCellIndices()) return path;
            }
        }
        return path;
    };
    auto checkSphere2DPath = [&message, &sphere2DRv, &sphere2DThetav, &sphere2DCv, &referenceSphere2DPath](
                                 const string& label, const Position& position, const Direction& direction,
                                 double maxDistance) {
        SpatialGridPath expected = referenceSphere2DPath(position, direction, maxDistance);
        SpatialGridPath actual(position, direction);
        if (!computeSphere2DPath(sphere2DRv.data(), sphere2DThetav.data(), sphere2DCv.data(),
                                 static_cast<int>(sphere2DRv.size()) - 1,
                                 static_cast<int>(sphere2DThetav.size()) - 1, position, direction, maxDistance,
                                 &actual))
        {
            message = "GPU spherical 2D grid path kernel failed (" + label + "): " + lastRuntimeError();
            return false;
        }
        if (actual.segments().size() != expected.segments().size())
        {
            std::ostringstream details;
            details << " expected";
            for (const auto& segment : expected.segments()) details << " [" << segment.m() << "," << segment.ds() << "]";
            details << " actual";
            for (const auto& segment : actual.segments()) details << " [" << segment.m() << "," << segment.ds() << "]";
            message = "spherical 2D grid path segment-count mismatch (" + label + "): expected "
                      + std::to_string(expected.segments().size()) + ", got "
                      + std::to_string(actual.segments().size()) + details.str();
            return false;
        }
        for (size_t i = 0; i != expected.segments().size(); ++i)
        {
            const auto& a = actual.segments()[i];
            const auto& e = expected.segments()[i];
            if (a.m() != e.m() || abs(a.ds() - e.ds()) > 1e-10 * max(1., abs(e.ds()))
                || abs(a.s() - e.s()) > 1e-10 * max(1., abs(e.s())))
            {
                message = "spherical 2D grid path segment mismatch (" + label + ")";
                return false;
            }
        }
        return true;
    };
    if (!checkSphere2DPath("outside-entry", Position(2.1, 0.3, 0.45), Direction(-1.0, -0.22, -0.35, true),
                           std::numeric_limits<double>::infinity())
        || !checkSphere2DPath("inner-hole", Position(0.05, 0.02, 0.01), Direction(1.0, 0.35, 0.2, true),
                              std::numeric_limits<double>::infinity())
        || !checkSphere2DPath("polar-crossing", Position(0.55, 0.2, 0.9), Direction(0.4, 0.1, -1.0, true),
                              std::numeric_limits<double>::infinity())
        || !checkSphere2DPath("limited", Position(2.1, 0.3, 0.45), Direction(-1.0, -0.22, -0.35, true), 1.3))
        return false;

    vector<double> sphere3DRv{0.12, 0.42, 0.85, 1.3, 1.8};
    vector<double> sphere3DThetav{0.0, 0.75, M_PI_2, 2.35, M_PI};
    vector<double> sphere3DPhiv{-M_PI, -1.2, 0.0, 1.2, M_PI};
    vector<double> sphere3DCv;
    for (double theta : sphere3DThetav) sphere3DCv.push_back(cos(theta));
    sphere3DCv.front() = 1.;
    sphere3DCv.back() = -1.;
    vector<double> sphere3DSinv;
    vector<double> sphere3DCosv;
    for (double phi : sphere3DPhiv)
    {
        sphere3DSinv.push_back(sin(phi));
        sphere3DCosv.push_back(cos(phi));
    }
    sphere3DSinv.front() = 0.;
    sphere3DCosv.front() = -1.;
    sphere3DSinv.back() = 0.;
    sphere3DCosv.back() = -1.;
    double sphere3DEps = 1e-12 * sphere3DRv.back();

    auto referenceSphere3DPath = [&sphere3DRv, &sphere3DThetav, &sphere3DPhiv, &sphere3DCv, &sphere3DSinv,
                                  &sphere3DCosv, sphere3DEps, &quadraticSmallestPositive, &scaledSmallestPositive,
                                  &locateArray, &locateClip](const Position& position, const Direction& direction,
                                                             double maxDistance) {
        SpatialGridPath path(position, direction);
        double rx, ry, rz;
        position.cartesian(rx, ry, rz);
        double kx, ky, kz;
        direction.cartesian(kx, ky, kz);
        int nr = static_cast<int>(sphere3DRv.size()) - 1;
        int ntheta = static_cast<int>(sphere3DThetav.size()) - 1;
        int nphi = static_cast<int>(sphere3DPhiv.size()) - 1;
        int i = -1;
        int j = -1;
        int k = -1;
        auto setCellIndices = [&]() {
            double radius = sqrt(rx * rx + ry * ry + rz * rz);
            double theta = radius == 0. ? 0. : acos(rz / radius);
            double phi = radius == 0. ? 0. : atan2(ry, rx);
            i = locateArray(sphere3DRv, radius);
            j = locateClip(sphere3DThetav, theta);
            k = locateClip(sphere3DPhiv, phi);
            return i < nr;
        };
        auto addSegment = [&path, maxDistance](int m, double ds) {
            path.addSegment(m, ds);
            return !path.segments().empty() && path.segments().back().s() > maxDistance;
        };
        auto meridionalPlane = [&sphere3DSinv, &sphere3DCosv](int kk, double rx, double ry, double kx, double ky) {
            double q = kx * sphere3DSinv[kk] - ky * sphere3DCosv[kk];
            if (abs(q) < 1e-12) return 0.;
            return -(rx * sphere3DSinv[kk] - ry * sphere3DCosv[kk]) / q;
        };

        double radius = sqrt(rx * rx + ry * ry + rz * rz);
        if (radius > sphere3DRv.back())
        {
            double ds = quadraticSmallestPositive(rx * kx + ry * ky + rz * kz,
                                                 radius * radius - sphere3DRv.back() * sphere3DRv.back());
            if (ds <= 0.) return path;
            rx += kx * (ds + sphere3DEps);
            ry += ky * (ds + sphere3DEps);
            rz += kz * (ds + sphere3DEps);
            if (!setCellIndices()) return path;
            if (addSegment(-1, ds)) return path;
        }
        else
        {
            if (!setCellIndices()) return path;
        }

        int guard = 0;
        int maxGuard = 24 * (nr + ntheta + nphi + 1);
        while (guard++ < maxGuard)
        {
            if (i >= 0)
            {
                int icur = i;
                int jcur = j;
                int kcur = k;
                double ds = DBL_MAX;
                double r2 = rx * rx + ry * ry + rz * rz;
                double rk = rx * kx + ry * ky + rz * kz;

                {
                    double rB = sphere3DRv[icur];
                    double s = quadraticSmallestPositive(rk, r2 - rB * rB);
                    if (s > 0. && s < ds)
                    {
                        ds = s;
                        i = icur - 1;
                        j = jcur;
                        k = kcur;
                    }
                }

                {
                    double rB = sphere3DRv[icur + 1];
                    double s = quadraticSmallestPositive(rk, r2 - rB * rB);
                    if (s > 0. && s < ds)
                    {
                        ds = s;
                        i = icur + 1;
                        j = jcur;
                        k = kcur;
                    }
                }

                if (jcur > 0)
                {
                    double c = sphere3DCv[jcur];
                    double s =
                        c ? scaledSmallestPositive(c * c - kz * kz, c * c * rk - rz * kz, c * c * r2 - rz * rz)
                          : -rz / kz;
                    if (s > 0. && s < ds)
                    {
                        ds = s;
                        i = icur;
                        j = jcur - 1;
                        k = kcur;
                    }
                }

                if (jcur < ntheta - 1)
                {
                    double c = sphere3DCv[jcur + 1];
                    double s =
                        c ? scaledSmallestPositive(c * c - kz * kz, c * c * rk - rz * kz, c * c * r2 - rz * rz)
                          : -rz / kz;
                    if (s > 0. && s < ds)
                    {
                        ds = s;
                        i = icur;
                        j = jcur + 1;
                        k = kcur;
                    }
                }

                {
                    double s = meridionalPlane(kcur, rx, ry, kx, ky);
                    if (s > 0. && s < ds)
                    {
                        ds = s;
                        i = icur;
                        j = jcur;
                        k = kcur > 0 ? kcur - 1 : nphi - 1;
                    }
                }

                {
                    double s = meridionalPlane(kcur + 1, rx, ry, kx, ky);
                    if (s > 0. && s < ds)
                    {
                        ds = s;
                        i = icur;
                        j = jcur;
                        k = (kcur + 1) % nphi;
                    }
                }

                if (ds == DBL_MAX) return path;
                if (addSegment(kcur + (jcur + icur * ntheta) * nphi, ds)) return path;
                rx += kx * (ds + sphere3DEps);
                ry += ky * (ds + sphere3DEps);
                rz += kz * (ds + sphere3DEps);
                if (i >= nr) return path;
            }
            else
            {
                double r2 = rx * rx + ry * ry + rz * rz;
                double ds = quadraticSmallestPositive(rx * kx + ry * ky + rz * kz, r2 - sphere3DRv[0] * sphere3DRv[0]);
                if (ds <= 0.) return path;
                if (addSegment(-1, ds)) return path;
                rx += kx * (ds + sphere3DEps);
                ry += ky * (ds + sphere3DEps);
                rz += kz * (ds + sphere3DEps);
                if (!setCellIndices()) return path;
            }
        }
        return path;
    };
    auto checkSphere3DPath = [&message, &sphere3DRv, &sphere3DThetav, &sphere3DPhiv, &sphere3DCv, &sphere3DSinv,
                              &sphere3DCosv, sphere3DEps, &referenceSphere3DPath](
                                 const string& label, const Position& position, const Direction& direction,
                                 double maxDistance) {
        SpatialGridPath expected = referenceSphere3DPath(position, direction, maxDistance);
        SpatialGridPath actual(position, direction);
        if (!computeSphere3DPath(sphere3DRv.data(), sphere3DThetav.data(), sphere3DPhiv.data(), sphere3DCv.data(),
                                 sphere3DSinv.data(), sphere3DCosv.data(), static_cast<int>(sphere3DRv.size()) - 1,
                                 static_cast<int>(sphere3DThetav.size()) - 1,
                                 static_cast<int>(sphere3DPhiv.size()) - 1, sphere3DEps, position, direction,
                                 maxDistance, &actual))
        {
            message = "GPU spherical 3D grid path kernel failed (" + label + "): " + lastRuntimeError();
            return false;
        }
        if (actual.segments().size() != expected.segments().size())
        {
            message = "spherical 3D grid path segment-count mismatch (" + label + "): expected "
                      + std::to_string(expected.segments().size()) + ", got "
                      + std::to_string(actual.segments().size());
            return false;
        }
        for (size_t i = 0; i != expected.segments().size(); ++i)
        {
            const auto& a = actual.segments()[i];
            const auto& e = expected.segments()[i];
            if (a.m() != e.m() || abs(a.ds() - e.ds()) > 1e-10 * max(1., abs(e.ds()))
                || abs(a.s() - e.s()) > 1e-10 * max(1., abs(e.s())))
            {
                message = "spherical 3D grid path segment mismatch (" + label + ")";
                return false;
            }
        }
        return true;
    };
    if (!checkSphere3DPath("outside-entry", Position(2.2, 0.35, 0.42), Direction(-1.0, -0.31, -0.27, true),
                           std::numeric_limits<double>::infinity())
        || !checkSphere3DPath("inner-hole", Position(0.05, 0.02, 0.01), Direction(1.0, 0.35, 0.25, true),
                              std::numeric_limits<double>::infinity())
        || !checkSphere3DPath("azimuth-polar", Position(0.75, -0.25, 0.55), Direction(-0.2, 0.85, -0.5, true),
                              std::numeric_limits<double>::infinity())
        || !checkSphere3DPath("limited", Position(2.2, 0.35, 0.42), Direction(-1.0, -0.31, -0.27, true), 1.4))
        return false;

    vector<double> cylinder2DRv{0.15, 0.45, 0.9, 1.35, 1.8};
    vector<double> cylinder2DZv{-1.2, -0.4, 0.25, 0.9, 1.6};
    auto referenceCylinder2DPath = [&cylinder2DRv, &cylinder2DZv, &locateArray, &locateClip](
                                       const Position& position, const Direction& direction, double maxDistance) {
        SpatialGridPath path(position, direction);
        double rx, ry, rz;
        position.cartesian(rx, ry, rz);
        double kx, ky, kz;
        direction.cartesian(kx, ky, kz);
        int nR = static_cast<int>(cylinder2DRv.size()) - 1;
        int nz = static_cast<int>(cylinder2DZv.size()) - 1;
        double R = sqrt(rx * rx + ry * ry);
        double z = rz;
        double kq = sqrt(kx * kx + ky * ky);
        if (kq == 0.) kq = 1e-20;
        if (kz == 0.) kz = 1e-20;
        double q = (rx * kx + ry * ky) / kq;
        double p = sqrt(max(0., (R - q) * (R + q)));
        double cumds = 0.;
        auto addSegment = [&path, maxDistance](int m, double ds) {
            path.addSegment(m, ds);
            return !path.segments().empty() && path.segments().back().s() > maxDistance;
        };

        if (R >= cylinder2DRv.back())
        {
            if (q > 0. || p > cylinder2DRv.back()) return path;
            double qmin = -sqrt((cylinder2DRv.back() - p) * (cylinder2DRv.back() + p));
            double ds = (qmin - q) / kq;
            q = qmin;
            R = cylinder2DRv.back() - 1e-8 * (cylinder2DRv[nR] - cylinder2DRv[nR - 1]);
            z += kz * ds;
            cumds += ds;
        }

        if (z < cylinder2DZv.front())
        {
            if (kz <= 0.) return path;
            double ds = (cylinder2DZv.front() - z) / kz;
            q += kq * ds;
            R = sqrt(p * p + q * q);
            z = cylinder2DZv.front() + 1e-8 * (cylinder2DZv[1] - cylinder2DZv[0]);
            cumds += ds;
        }
        else if (z > cylinder2DZv.back())
        {
            if (kz >= 0.) return path;
            double ds = (cylinder2DZv.back() - z) / kz;
            q += kq * ds;
            R = sqrt(p * p + q * q);
            z = cylinder2DZv.back() - 1e-8 * (cylinder2DZv[nz] - cylinder2DZv[nz - 1]);
            cumds += ds;
        }

        if (std::isinf(R) || std::isnan(R) || std::isinf(z) || std::isnan(z)) return path;
        if (R >= cylinder2DRv.back() || z <= cylinder2DZv.front() || z >= cylinder2DZv.back()) return path;
        if (cumds > 0. && addSegment(-1, cumds)) return path;

        int i = locateArray(cylinder2DRv, R);
        int k = locateClip(cylinder2DZv, z);
        int imin = -1;
        enum class Phase { UpInwards, UpOutwards, DownInwards, DownOutwards };
        Phase phase = Phase::UpOutwards;
        if (kz >= 0.)
        {
            phase = Phase::UpOutwards;
            if (q < 0.)
            {
                imin = locateArray(cylinder2DRv, p);
                if (i > imin) phase = Phase::UpInwards;
            }
        }
        else
        {
            phase = Phase::DownOutwards;
            if (q < 0.)
            {
                imin = locateArray(cylinder2DRv, p);
                if (i > imin) phase = Phase::DownInwards;
            }
        }

        int guard = 0;
        int maxGuard = 8 * (nR + nz + 1);
        while (guard++ < maxGuard)
        {
            int m = i < 0 ? -1 : k + nz * i;
            double ds = 0.;
            if (phase == Phase::UpInwards)
            {
                double RN = cylinder2DRv[i];
                double qN = -sqrt((RN - p) * (RN + p));
                double zN = cylinder2DZv[k + 1];
                double dsq = (qN - q) / kq;
                double dsz = (zN - z) / kz;
                if (dsq < dsz)
                {
                    ds = dsq;
                    --i;
                    q = qN;
                    z += kz * ds;
                    if (i <= imin) phase = Phase::UpOutwards;
                }
                else
                {
                    ds = dsz;
                    ++k;
                    if (k < nz)
                    {
                        q += kq * ds;
                        z = zN;
                    }
                }
            }
            else if (phase == Phase::UpOutwards)
            {
                double RN = cylinder2DRv[i + 1];
                double qN = sqrt((RN - p) * (RN + p));
                double zN = cylinder2DZv[k + 1];
                double dsq = (qN - q) / kq;
                double dsz = (zN - z) / kz;
                if (dsq < dsz)
                {
                    ds = dsq;
                    ++i;
                    if (i < nR)
                    {
                        q = qN;
                        z += kz * ds;
                    }
                }
                else
                {
                    ds = dsz;
                    ++k;
                    if (k < nz)
                    {
                        q += kq * ds;
                        z = zN;
                    }
                }
            }
            else if (phase == Phase::DownInwards)
            {
                double RN = cylinder2DRv[i];
                double qN = -sqrt((RN - p) * (RN + p));
                double zN = cylinder2DZv[k];
                double dsq = (qN - q) / kq;
                double dsz = (zN - z) / kz;
                if (dsq < dsz)
                {
                    ds = dsq;
                    --i;
                    q = qN;
                    z += kz * ds;
                    if (i <= imin) phase = Phase::DownOutwards;
                }
                else
                {
                    ds = dsz;
                    --k;
                    if (k >= 0)
                    {
                        q += kq * ds;
                        z = zN;
                    }
                }
            }
            else
            {
                double RN = cylinder2DRv[i + 1];
                double qN = sqrt((RN - p) * (RN + p));
                double zN = cylinder2DZv[k];
                double dsq = (qN - q) / kq;
                double dsz = (zN - z) / kz;
                if (dsq < dsz)
                {
                    ds = dsq;
                    ++i;
                    if (i < nR)
                    {
                        q = qN;
                        z += kz * ds;
                    }
                }
                else
                {
                    ds = dsz;
                    --k;
                    if (k >= 0)
                    {
                        q += kq * ds;
                        z = zN;
                    }
                }
            }

            if (ds > 0. && addSegment(m, ds)) return path;
            if (i >= nR || k < 0 || k >= nz) return path;
        }
        return path;
    };
    auto checkCylinder2DPath = [&message, &cylinder2DRv, &cylinder2DZv, &referenceCylinder2DPath](
                                   const string& label, const Position& position, const Direction& direction,
                                   double maxDistance) {
        SpatialGridPath expected = referenceCylinder2DPath(position, direction, maxDistance);
        SpatialGridPath actual(position, direction);
        if (!computeCylinder2DPath(cylinder2DRv.data(), cylinder2DZv.data(),
                                   static_cast<int>(cylinder2DRv.size()) - 1,
                                   static_cast<int>(cylinder2DZv.size()) - 1, position, direction, maxDistance,
                                   &actual))
        {
            message = "GPU cylindrical 2D grid path kernel failed (" + label + "): " + lastRuntimeError();
            return false;
        }
        if (actual.segments().size() != expected.segments().size())
        {
            message = "cylindrical 2D grid path segment-count mismatch (" + label + "): expected "
                      + std::to_string(expected.segments().size()) + ", got "
                      + std::to_string(actual.segments().size());
            return false;
        }
        for (size_t i = 0; i != expected.segments().size(); ++i)
        {
            const auto& a = actual.segments()[i];
            const auto& e = expected.segments()[i];
            if (a.m() != e.m() || abs(a.ds() - e.ds()) > 1e-10 * max(1., abs(e.ds()))
                || abs(a.s() - e.s()) > 1e-10 * max(1., abs(e.s())))
            {
                message = "cylindrical 2D grid path segment mismatch (" + label + ")";
                return false;
            }
        }
        return true;
    };
    if (!checkCylinder2DPath("outside-radial", Position(2.2, 0.25, -0.8), Direction(-1.0, -0.15, 0.32, true),
                             std::numeric_limits<double>::infinity())
        || !checkCylinder2DPath("below-z", Position(0.6, 0.15, -2.0), Direction(0.2, 0.1, 1.0, true),
                                std::numeric_limits<double>::infinity())
        || !checkCylinder2DPath("inner-hole", Position(0.05, 0.02, 0.0), Direction(1.0, 0.25, 0.45, true),
                                std::numeric_limits<double>::infinity())
        || !checkCylinder2DPath("downward", Position(1.1, 0.2, 1.3), Direction(-0.4, 0.15, -1.0, true),
                                std::numeric_limits<double>::infinity())
        || !checkCylinder2DPath("limited", Position(2.2, 0.25, -0.8), Direction(-1.0, -0.15, 0.32, true), 1.4))
        return false;

    double cylinder3DEps = 1e-12 * 1.8;
    vector<double> cylinder3DRv{2. * cylinder3DEps, 0.45, 0.9, 1.35, 1.8};
    vector<double> cylinder3DHoleRv{0.2, 0.55, 1.0, 1.5};
    vector<double> cylinder3DPhiv{-M_PI, -1.2, 0.0, 1.1, M_PI};
    vector<double> cylinder3DZv{-1.0, -0.25, 0.5, 1.2};
    vector<double> cylinder3DSinv;
    vector<double> cylinder3DCosv;
    for (double phi : cylinder3DPhiv)
    {
        cylinder3DSinv.push_back(sin(phi));
        cylinder3DCosv.push_back(cos(phi));
    }
    cylinder3DSinv.front() = 0.;
    cylinder3DCosv.front() = -1.;
    cylinder3DSinv.back() = 0.;
    cylinder3DCosv.back() = -1.;
    auto referenceCylinder3DPath = [&quadraticSmallestPositive, &locateArray, &locateClip, &locateFail](
                                       const vector<double>& Rv, const vector<double>& phiv,
                                       const vector<double>& zv, const vector<double>& sinv,
                                       const vector<double>& cosv, double eps, bool hasHole, const Position& position,
                                       const Direction& direction, double maxDistance) {
        SpatialGridPath path(position, direction);
        double rx, ry, rz;
        position.cartesian(rx, ry, rz);
        double kx, ky, kz;
        direction.cartesian(kx, ky, kz);
        int nR = static_cast<int>(Rv.size()) - 1;
        int nphi = static_cast<int>(phiv.size()) - 1;
        int nz = static_cast<int>(zv.size()) - 1;
        double kq2 = kx * kx + ky * ky;
        int i = -1;
        int j = -1;
        int k = -1;
        auto addSegment = [&path, maxDistance](int m, double ds) {
            path.addSegment(m, ds);
            return !path.segments().empty() && path.segments().back().s() > maxDistance;
        };
        auto setCellIndices = [&]() {
            double R = sqrt(rx * rx + ry * ry);
            double phi = atan2(ry, rx);
            i = locateArray(Rv, R);
            j = locateClip(phiv, phi);
            k = locateFail(zv, rz);
            return i < nR && k >= 0;
        };
        auto firstIntersectionCylinder = [&](int ii) {
            if (abs(kq2) < 1e-12) return 0.;
            double b = rx * kx + ry * ky;
            double c = rx * rx + ry * ry - Rv[ii] * Rv[ii];
            return quadraticSmallestPositive(b / kq2, c / kq2);
        };
        auto intersectionMeridionalPlane = [&](int jj) {
            double q = kx * sinv[jj] - ky * cosv[jj];
            if (abs(q) < 1e-12) return 0.;
            return -(rx * sinv[jj] - ry * cosv[jj]) / q;
        };
        auto intersectionHorizontalPlane = [&](int kk) {
            if (abs(kz) < 1e-12) return 0.;
            return (zv[kk] - rz) / kz;
        };

        double cumds = 0.;
        if (sqrt(rx * rx + ry * ry) > Rv[nR])
        {
            double ds = firstIntersectionCylinder(nR);
            if (ds <= 0.) return path;
            rx += kx * (ds + eps);
            ry += ky * (ds + eps);
            rz += kz * (ds + eps);
            cumds += ds;
        }

        if (rz < zv.front())
        {
            double ds = intersectionHorizontalPlane(0);
            if (ds <= 0.) return path;
            rx += kx * (ds + eps);
            ry += ky * (ds + eps);
            rz += kz * (ds + eps);
            cumds += ds;
        }
        else if (rz > zv.back())
        {
            double ds = intersectionHorizontalPlane(nz);
            if (ds <= 0.) return path;
            rx += kx * (ds + eps);
            ry += ky * (ds + eps);
            rz += kz * (ds + eps);
            cumds += ds;
        }

        if (!setCellIndices()) return path;
        if (cumds > 0. && addSegment(-1, cumds)) return path;

        int guard = 0;
        int maxGuard = 12 * (nR + nphi + nz + 1);
        while (guard++ < maxGuard)
        {
            if (i >= 0)
            {
                int icur = i;
                int jcur = j;
                int kcur = k;
                double ds = DBL_MAX;

                {
                    double s = firstIntersectionCylinder(icur);
                    if (s > 0. && s < ds)
                    {
                        ds = s;
                        i = icur - 1;
                        j = jcur;
                        k = kcur;
                    }
                }

                {
                    double s = firstIntersectionCylinder(icur + 1);
                    if (s > 0. && s < ds)
                    {
                        ds = s;
                        i = icur + 1;
                        j = jcur;
                        k = kcur;
                    }
                }

                {
                    double s = intersectionMeridionalPlane(jcur);
                    if (s > 0. && s < ds)
                    {
                        ds = s;
                        i = icur;
                        j = jcur > 0 ? jcur - 1 : nphi - 1;
                        k = kcur;
                    }
                }

                {
                    double s = intersectionMeridionalPlane(jcur + 1);
                    if (s > 0. && s < ds)
                    {
                        ds = s;
                        i = icur;
                        j = (jcur + 1) % nphi;
                        k = kcur;
                    }
                }

                {
                    double s = intersectionHorizontalPlane(kcur);
                    if (s > 0. && s < ds)
                    {
                        ds = s;
                        i = icur;
                        j = jcur;
                        k = kcur - 1;
                    }
                }

                {
                    double s = intersectionHorizontalPlane(kcur + 1);
                    if (s > 0. && s < ds)
                    {
                        ds = s;
                        i = icur;
                        j = jcur;
                        k = kcur + 1;
                    }
                }

                if (ds == DBL_MAX) return path;
                if (addSegment(kcur + (jcur + icur * nphi) * nz, ds)) return path;
                rx += kx * (ds + eps);
                ry += ky * (ds + eps);
                rz += kz * (ds + eps);
                if (i >= nR || k < 0 || k >= nz) return path;
            }
            else if (!hasHole)
            {
                int kcur = k;
                double ds = DBL_MAX;
                {
                    double s = firstIntersectionCylinder(0);
                    if (s > 0. && s < ds) ds = s;
                }
                {
                    double s = intersectionHorizontalPlane(kcur);
                    if (s > 0. && s < ds)
                    {
                        ds = s;
                        k = kcur - 1;
                    }
                }
                {
                    double s = intersectionHorizontalPlane(kcur + 1);
                    if (s > 0. && s < ds)
                    {
                        ds = s;
                        k = kcur + 1;
                    }
                }
                if (ds == DBL_MAX) return path;
                if (addSegment(kcur + j * nz, ds)) return path;
                rx += kx * (ds + eps);
                ry += ky * (ds + eps);
                rz += kz * (ds + eps);
                if (k == kcur)
                {
                    if (!setCellIndices()) return path;
                }
                else if (k < 0 || k >= nz)
                    return path;
            }
            else
            {
                double ds = firstIntersectionCylinder(0);
                if (ds <= 0.) return path;
                if (addSegment(-1, ds)) return path;
                rx += kx * (ds + eps);
                ry += ky * (ds + eps);
                rz += kz * (ds + eps);
                if (!setCellIndices()) return path;
            }
        }
        return path;
    };
    auto checkCylinder3DPath = [&message, &cylinder3DRv, &cylinder3DHoleRv, &cylinder3DPhiv, &cylinder3DZv,
                                &cylinder3DSinv, &cylinder3DCosv, cylinder3DEps, &referenceCylinder3DPath](
                                   const string& label, bool hasHole, const Position& position,
                                   const Direction& direction, double maxDistance) {
        const vector<double>& Rv = hasHole ? cylinder3DHoleRv : cylinder3DRv;
        double eps = hasHole ? 1e-12 * Rv.back() : cylinder3DEps;
        SpatialGridPath expected =
            referenceCylinder3DPath(Rv, cylinder3DPhiv, cylinder3DZv, cylinder3DSinv, cylinder3DCosv, eps, hasHole,
                                    position, direction, maxDistance);
        SpatialGridPath actual(position, direction);
        if (!computeCylinder3DPath(Rv.data(), cylinder3DPhiv.data(), cylinder3DZv.data(), cylinder3DSinv.data(),
                                   cylinder3DCosv.data(), static_cast<int>(Rv.size()) - 1,
                                   static_cast<int>(cylinder3DPhiv.size()) - 1,
                                   static_cast<int>(cylinder3DZv.size()) - 1, eps, hasHole, position, direction,
                                   maxDistance, &actual))
        {
            message = "GPU cylindrical 3D grid path kernel failed (" + label + "): " + lastRuntimeError();
            return false;
        }
        if (actual.segments().size() != expected.segments().size())
        {
            message = "cylindrical 3D grid path segment-count mismatch (" + label + "): expected "
                      + std::to_string(expected.segments().size()) + ", got "
                      + std::to_string(actual.segments().size());
            return false;
        }
        for (size_t i = 0; i != expected.segments().size(); ++i)
        {
            const auto& a = actual.segments()[i];
            const auto& e = expected.segments()[i];
            if (a.m() != e.m() || abs(a.ds() - e.ds()) > 1e-10 * max(1., abs(e.ds()))
                || abs(a.s() - e.s()) > 1e-10 * max(1., abs(e.s())))
            {
                message = "cylindrical 3D grid path segment mismatch (" + label + ")";
                return false;
            }
        }
        return true;
    };
    if (!checkCylinder3DPath("outside-entry", false, Position(2.2, 0.35, -0.5),
                             Direction(-1.0, -0.22, 0.35, true), std::numeric_limits<double>::infinity())
        || !checkCylinder3DPath("artificial-hole", false, Position(0.0, 0.0, -0.4),
                                Direction(0.2, 0.05, 0.35, true), std::numeric_limits<double>::infinity())
        || !checkCylinder3DPath("real-hole", true, Position(0.05, 0.02, 0.0),
                                Direction(1.0, 0.3, 0.2, true), std::numeric_limits<double>::infinity())
        || !checkCylinder3DPath("below-z", false, Position(0.6, -0.2, -1.8),
                                Direction(0.2, 0.4, 1.0, true), std::numeric_limits<double>::infinity())
        || !checkCylinder3DPath("limited", false, Position(2.2, 0.35, -0.5),
                                Direction(-1.0, -0.22, 0.35, true), 1.3))
        return false;

    vector<double> tetraVertices{0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0.,
                                 0., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1., 1.};
    vector<int> tetraIndices{0, 1, 3, 7, 0, 3, 2, 7, 0, 2, 6, 7, 0, 6, 4, 7, 0, 4, 5, 7, 0, 5, 1, 7};
    vector<int> tetraFaceAnchor;
    vector<int> tetraFaceNeighbor;
    vector<double> tetraFaceNormals;
    vector<double> tetraCentroids;
    auto tetraClockwiseVertices = [](int face) {
        std::array<int, 3> cv = {(face + 3) % 4, (face + 2) % 4, (face + 1) % 4};
        if (face % 2 == 0) std::swap(cv[0], cv[2]);
        return cv;
    };
    auto vertexAt = [&tetraVertices](int index) {
        int base = 3 * index;
        return std::array<double, 3>{tetraVertices[base], tetraVertices[base + 1], tetraVertices[base + 2]};
    };
    auto cross = [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return std::array<double, 3>{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                                     a[0] * b[1] - a[1] * b[0]};
    };
    auto dot = [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    auto subtract = [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return std::array<double, 3>{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    };
    auto faceKey = [&tetraIndices, &tetraClockwiseVertices](int cell, int face) {
        auto cv = tetraClockwiseVertices(face);
        std::array<int, 3> key{tetraIndices[4 * cell + cv[0]], tetraIndices[4 * cell + cv[1]],
                               tetraIndices[4 * cell + cv[2]]};
        std::sort(key.begin(), key.end());
        return key;
    };

    int numTetra = static_cast<int>(tetraIndices.size()) / 4;
    tetraFaceNeighbor.assign(4 * numTetra, -1);
    for (int m = 0; m != numTetra; ++m)
    {
        std::array<double, 3> centroid{0., 0., 0.};
        for (int t = 0; t != 4; ++t)
        {
            auto v = vertexAt(tetraIndices[4 * m + t]);
            centroid[0] += 0.25 * v[0];
            centroid[1] += 0.25 * v[1];
            centroid[2] += 0.25 * v[2];
        }
        tetraCentroids.insert(tetraCentroids.end(), {centroid[0], centroid[1], centroid[2]});

        for (int f = 0; f != 4; ++f)
        {
            auto cv = tetraClockwiseVertices(f);
            int anchor = tetraIndices[4 * m + cv[0]];
            tetraFaceAnchor.push_back(anchor);
            auto v0 = vertexAt(anchor);
            auto v1 = vertexAt(tetraIndices[4 * m + cv[1]]);
            auto v2 = vertexAt(tetraIndices[4 * m + cv[2]]);
            auto opposite = vertexAt(tetraIndices[4 * m + f]);
            auto normal = cross(subtract(v1, v0), subtract(v2, v0));
            double norm = sqrt(dot(normal, normal));
            normal = {normal[0] / norm, normal[1] / norm, normal[2] / norm};
            if (dot(normal, subtract(opposite, v0)) > 0.)
                normal = {-normal[0], -normal[1], -normal[2]};
            tetraFaceNormals.insert(tetraFaceNormals.end(), {normal[0], normal[1], normal[2]});
        }
    }
    for (int m = 0; m != numTetra; ++m)
        for (int f = 0; f != 4; ++f)
            for (int n = 0; n != numTetra; ++n)
                if (n != m)
                    for (int nf = 0; nf != 4; ++nf)
                        if (faceKey(m, f) == faceKey(n, nf)) tetraFaceNeighbor[4 * m + f] = n;

    auto referenceTetraPath = [&tetraVertices, &tetraFaceAnchor, &tetraFaceNeighbor, &tetraFaceNormals,
                               &tetraCentroids, dot](const Position& position, const Direction& direction,
                                                     double maxDistance) {
        SpatialGridPath path(position, direction);
        double rx, ry, rz;
        position.cartesian(rx, ry, rz);
        double kx, ky, kz;
        direction.cartesian(kx, ky, kz);
        double eps = 1e-12 * sqrt(3.);
        double cumds = 0.;
        auto containsCell = [&](int cell) {
            std::array<double, 3> r{rx, ry, rz};
            for (int f = 0; f != 4; ++f)
            {
                int anchor = tetraFaceAnchor[4 * cell + f];
                std::array<double, 3> v{tetraVertices[3 * anchor], tetraVertices[3 * anchor + 1],
                                        tetraVertices[3 * anchor + 2]};
                std::array<double, 3> n{tetraFaceNormals[12 * cell + 3 * f],
                                        tetraFaceNormals[12 * cell + 3 * f + 1],
                                        tetraFaceNormals[12 * cell + 3 * f + 2]};
                if (dot({v[0] - r[0], v[1] - r[1], v[2] - r[2]}, n) < 0.) return false;
            }
            return true;
        };
        auto findCell = [&]() {
            for (int m = 0; m != 6; ++m)
                if (containsCell(m)) return m;
            return -1;
        };
        auto addSegment = [&path, maxDistance](int m, double ds) {
            path.addSegment(m, ds);
            return !path.segments().empty() && path.segments().back().s() > maxDistance;
        };

        if (rx <= 0.)
        {
            if (kx <= 0.) return path;
            double ds = -rx / kx;
            rx = eps;
            ry += ky * ds;
            rz += kz * ds;
            cumds += ds;
        }
        else if (rx >= 1.)
        {
            if (kx >= 0.) return path;
            double ds = (1. - rx) / kx;
            rx = 1. - eps;
            ry += ky * ds;
            rz += kz * ds;
            cumds += ds;
        }
        if (ry <= 0.)
        {
            if (ky <= 0.) return path;
            double ds = -ry / ky;
            rx += kx * ds;
            ry = eps;
            rz += kz * ds;
            cumds += ds;
        }
        else if (ry >= 1.)
        {
            if (ky >= 0.) return path;
            double ds = (1. - ry) / ky;
            rx += kx * ds;
            ry = 1. - eps;
            rz += kz * ds;
            cumds += ds;
        }
        if (rz <= 0.)
        {
            if (kz <= 0.) return path;
            double ds = -rz / kz;
            rx += kx * ds;
            ry += ky * ds;
            rz = eps;
            cumds += ds;
        }
        else if (rz >= 1.)
        {
            if (kz >= 0.) return path;
            double ds = (1. - rz) / kz;
            rx += kx * ds;
            ry += ky * ds;
            rz = 1. - eps;
            cumds += ds;
        }
        if (rx < 0. || rx > 1. || ry < 0. || ry > 1. || rz < 0. || rz > 1.) return path;
        if (cumds > 0. && addSegment(-1, cumds)) return path;

        int cell = findCell();
        if (cell < 0) return path;
        for (int guard = 0; guard != 8 * (6 + 1); ++guard)
        {
            int exitFace = -1;
            double ds = DBL_MAX;
            for (int f = 0; f != 4; ++f)
            {
                std::array<double, 3> n{tetraFaceNormals[12 * cell + 3 * f],
                                        tetraFaceNormals[12 * cell + 3 * f + 1],
                                        tetraFaceNormals[12 * cell + 3 * f + 2]};
                double ndotk = n[0] * kx + n[1] * ky + n[2] * kz;
                if (ndotk > 0.)
                {
                    int anchor = tetraFaceAnchor[4 * cell + f];
                    std::array<double, 3> v{tetraVertices[3 * anchor], tetraVertices[3 * anchor + 1],
                                            tetraVertices[3 * anchor + 2]};
                    double s = (n[0] * (v[0] - rx) + n[1] * (v[1] - ry) + n[2] * (v[2] - rz)) / ndotk;
                    if (s > 0. && s < ds)
                    {
                        ds = s;
                        exitFace = f;
                    }
                }
            }
            if (exitFace < 0 || ds < eps)
            {
                rx += kx * eps;
                ry += ky * eps;
                rz += kz * eps;
                cell = findCell();
                if (cell < 0) return path;
                continue;
            }
            int nextCell = tetraFaceNeighbor[4 * cell + exitFace];
            if (nextCell < 0) return path;
            if (addSegment(cell, ds)) return path;
            rx += kx * ds;
            ry += ky * ds;
            rz += kz * ds;
            cell = nextCell;
        }
        return path;
    };
    auto checkTetraPath = [&message, &tetraVertices, &tetraIndices, &tetraFaceAnchor, &tetraFaceNeighbor,
                           &tetraFaceNormals, &tetraCentroids, &referenceTetraPath](
                              const string& label, const Position& position, const Direction& direction,
                              double maxDistance) {
        SpatialGridPath expected = referenceTetraPath(position, direction, maxDistance);
        if (expected.segments().empty())
        {
            message = "tetrahedral grid path test did not cross an internal face (" + label + ")";
            return false;
        }
        SpatialGridPath actual(position, direction);
        if (!computeTetraPath(tetraIndices.data(), tetraVertices, tetraIndices, tetraFaceAnchor, tetraFaceNeighbor,
                              tetraFaceNormals, tetraCentroids, 6, 1e-12 * sqrt(3.), position, direction, 0., 0., 0.,
                              1., 1., 1., maxDistance, &actual))
        {
            message = "GPU tetrahedral grid path kernel failed (" + label + "): " + lastRuntimeError();
            return false;
        }
        if (actual.segments().size() != expected.segments().size())
        {
            message = "tetrahedral grid path segment-count mismatch (" + label + "): expected "
                      + std::to_string(expected.segments().size()) + ", got "
                      + std::to_string(actual.segments().size());
            return false;
        }
        for (size_t i = 0; i != expected.segments().size(); ++i)
        {
            const auto& a = actual.segments()[i];
            const auto& e = expected.segments()[i];
            if (a.m() != e.m() || abs(a.ds() - e.ds()) > 1e-10 * max(1., abs(e.ds()))
                || abs(a.s() - e.s()) > 1e-10 * max(1., abs(e.s())))
            {
                message = "tetrahedral grid path segment mismatch (" + label + ")";
                return false;
            }
        }
        return true;
    };
    if (!checkTetraPath("inside", Position(0.12, 0.18, 0.08), Direction(0.75, 0.55, 1.0, true),
                        std::numeric_limits<double>::infinity())
        || !checkTetraPath("outside-entry", Position(-0.2, 0.2, 0.1), Direction(1.0, 0.35, 0.65, true),
                           std::numeric_limits<double>::infinity())
        || !checkTetraPath("limited", Position(-0.2, 0.2, 0.1), Direction(1.0, 0.35, 0.65, true), 0.6))
        return false;

    vector<double> voronoiSites{0.25, 0.5, 0.5, 0.75, 0.5, 0.5};
    vector<int> voronoiNeighborBegin{0, 6};
    vector<int> voronoiNeighborCount{6, 6};
    vector<int> voronoiNeighborIndex{1, -1, -3, -4, -5, -6, 0, -2, -3, -4, -5, -6};
    double voronoiEps = 1e-12 * sqrt(3.);
    auto findVoronoiCell = [&voronoiSites](double rx, double ry, double rz) {
        if (rx < 0. || rx > 1. || ry < 0. || ry > 1. || rz < 0. || rz > 1.) return -1;
        int best = -1;
        double bestDistance = DBL_MAX;
        for (int m = 0; m != 2; ++m)
        {
            double dx = rx - voronoiSites[3 * m];
            double dy = ry - voronoiSites[3 * m + 1];
            double dz = rz - voronoiSites[3 * m + 2];
            double distance = dx * dx + dy * dy + dz * dz;
            if (distance < bestDistance)
            {
                best = m;
                bestDistance = distance;
            }
        }
        return best;
    };
    auto referenceVoronoiPath = [&voronoiSites, &voronoiNeighborBegin, &voronoiNeighborCount, &voronoiNeighborIndex,
                                 &findVoronoiCell, voronoiEps](const Position& position, const Direction& direction,
                                                               double maxDistance) {
        SpatialGridPath path(position, direction);
        double rx, ry, rz;
        position.cartesian(rx, ry, rz);
        double kx, ky, kz;
        direction.cartesian(kx, ky, kz);
        double cumds = 0.;
        auto addSegment = [&path, maxDistance](int m, double ds) {
            path.addSegment(m, ds);
            return !path.segments().empty() && path.segments().back().s() > maxDistance;
        };

        if (rx <= 0.)
        {
            if (kx <= 0.) return path;
            double ds = -rx / kx;
            rx = voronoiEps;
            ry += ky * ds;
            rz += kz * ds;
            cumds += ds;
        }
        else if (rx >= 1.)
        {
            if (kx >= 0.) return path;
            double ds = (1. - rx) / kx;
            rx = 1. - voronoiEps;
            ry += ky * ds;
            rz += kz * ds;
            cumds += ds;
        }
        if (ry <= 0.)
        {
            if (ky <= 0.) return path;
            double ds = -ry / ky;
            rx += kx * ds;
            ry = voronoiEps;
            rz += kz * ds;
            cumds += ds;
        }
        else if (ry >= 1.)
        {
            if (ky >= 0.) return path;
            double ds = (1. - ry) / ky;
            rx += kx * ds;
            ry = 1. - voronoiEps;
            rz += kz * ds;
            cumds += ds;
        }
        if (rz <= 0.)
        {
            if (kz <= 0.) return path;
            double ds = -rz / kz;
            rx += kx * ds;
            ry += ky * ds;
            rz = voronoiEps;
            cumds += ds;
        }
        else if (rz >= 1.)
        {
            if (kz >= 0.) return path;
            double ds = (1. - rz) / kz;
            rx += kx * ds;
            ry += ky * ds;
            rz = 1. - voronoiEps;
            cumds += ds;
        }

        int cell = findVoronoiCell(rx, ry, rz);
        if (cell < 0) return path;
        if (cumds > 0. && addSegment(-1, cumds)) return path;

        for (int guard = 0; guard != 8 * (2 + 1); ++guard)
        {
            int begin = voronoiNeighborBegin[cell];
            int count = voronoiNeighborCount[cell];
            int nextCell = -99;
            double ds = DBL_MAX;
            double prx = voronoiSites[3 * cell];
            double pry = voronoiSites[3 * cell + 1];
            double prz = voronoiSites[3 * cell + 2];
            for (int i = 0; i != count; ++i)
            {
                int neighbor = voronoiNeighborIndex[begin + i];
                double si = 0.;
                if (neighbor >= 0)
                {
                    double pix = voronoiSites[3 * neighbor];
                    double piy = voronoiSites[3 * neighbor + 1];
                    double piz = voronoiSites[3 * neighbor + 2];
                    double nx = pix - prx;
                    double ny = piy - pry;
                    double nz = piz - prz;
                    double ndotk = nx * kx + ny * ky + nz * kz;
                    if (ndotk > 0.)
                    {
                        double px = 0.5 * (pix + prx);
                        double py = 0.5 * (piy + pry);
                        double pz = 0.5 * (piz + prz);
                        si = (nx * (px - rx) + ny * (py - ry) + nz * (pz - rz)) / ndotk;
                    }
                }
                else if (neighbor == -1)
                    si = -rx / kx;
                else if (neighbor == -2)
                    si = (1. - rx) / kx;
                else if (neighbor == -3)
                    si = -ry / ky;
                else if (neighbor == -4)
                    si = (1. - ry) / ky;
                else if (neighbor == -5)
                    si = -rz / kz;
                else if (neighbor == -6)
                    si = (1. - rz) / kz;

                if (si > 0. && si < ds)
                {
                    ds = si;
                    nextCell = neighbor;
                }
            }

            if (nextCell == -99)
            {
                rx += kx * voronoiEps;
                ry += ky * voronoiEps;
                rz += kz * voronoiEps;
                cell = findVoronoiCell(rx, ry, rz);
                if (cell < 0) return path;
                continue;
            }

            if (addSegment(cell, ds)) return path;
            rx += kx * (ds + voronoiEps);
            ry += ky * (ds + voronoiEps);
            rz += kz * (ds + voronoiEps);
            if (nextCell < 0) return path;
            cell = nextCell;
        }
        return path;
    };
    auto checkVoronoiPath = [&message, &voronoiSites, &voronoiNeighborBegin, &voronoiNeighborCount,
                             &voronoiNeighborIndex, voronoiEps, &referenceVoronoiPath](
                                const string& label, const Position& position, const Direction& direction,
                                double maxDistance) {
        SpatialGridPath expected = referenceVoronoiPath(position, direction, maxDistance);
        if (expected.segments().empty())
        {
            message = "Voronoi grid path test did not cross a cell boundary (" + label + ")";
            return false;
        }
        SpatialGridPath actual(position, direction);
        if (!computeVoronoiPath(voronoiSites.data(), voronoiSites, voronoiNeighborBegin, voronoiNeighborCount,
                                voronoiNeighborIndex, 2, voronoiEps, position, direction, 0., 0., 0., 1., 1., 1.,
                                maxDistance, &actual))
        {
            message = "GPU Voronoi grid path kernel failed (" + label + "): " + lastRuntimeError();
            return false;
        }
        if (actual.segments().size() != expected.segments().size())
        {
            message = "Voronoi grid path segment-count mismatch (" + label + "): expected "
                      + std::to_string(expected.segments().size()) + ", got "
                      + std::to_string(actual.segments().size());
            return false;
        }
        for (size_t i = 0; i != expected.segments().size(); ++i)
        {
            const auto& a = actual.segments()[i];
            const auto& e = expected.segments()[i];
            if (a.m() != e.m() || abs(a.ds() - e.ds()) > 1e-10 * max(1., abs(e.ds()))
                || abs(a.s() - e.s()) > 1e-10 * max(1., abs(e.s())))
            {
                message = "Voronoi grid path segment mismatch (" + label + ")";
                return false;
            }
        }
        return true;
    };
    if (!checkVoronoiPath("inside", Position(0.2, 0.45, 0.45), Direction(1.0, 0.1, 0.05, true),
                          std::numeric_limits<double>::infinity())
        || !checkVoronoiPath("outside-entry", Position(-0.3, 0.4, 0.45), Direction(1.0, 0.2, 0.1, true),
                             std::numeric_limits<double>::infinity())
        || !checkVoronoiPath("limited", Position(-0.3, 0.4, 0.45), Direction(1.0, 0.2, 0.1, true), 0.7)
        || !checkVoronoiPath("reverse", Position(0.8, 0.55, 0.5), Direction(-1.0, 0.05, 0.02, true),
                             std::numeric_limits<double>::infinity()))
        return false;

    {
        vector<Position> positions{Position(0.2, 0.45, 0.45), Position(-0.3, 0.4, 0.45),
                                   Position(0.8, 0.55, 0.5)};
        vector<Direction> directions{Direction(1.0, 0.1, 0.05, true), Direction(1.0, 0.2, 0.1, true),
                                     Direction(-1.0, 0.05, 0.02, true)};
        double batchMaxDistance = std::numeric_limits<double>::infinity();
        vector<SpatialGridPath> actualv;
        vector<SpatialGridPath*> pathv;
        actualv.reserve(positions.size());
        pathv.reserve(positions.size());
        for (size_t i = 0; i != positions.size(); ++i)
        {
            actualv.emplace_back(positions[i], directions[i]);
            pathv.push_back(&actualv.back());
        }
        vector<int> emptyVoronoiBlockv;
        if (!computeVoronoiPaths(voronoiSites.data(), voronoiSites, voronoiNeighborBegin, voronoiNeighborCount,
                                 voronoiNeighborIndex, emptyVoronoiBlockv, emptyVoronoiBlockv, emptyVoronoiBlockv, 0, 2,
                                 voronoiEps, 0., 0., 0., 1., 1., 1., batchMaxDistance, pathv))
        {
            message = "GPU Voronoi batch path kernel failed: " + lastRuntimeError();
            return false;
        }
        for (size_t p = 0; p != actualv.size(); ++p)
        {
            SpatialGridPath expected = referenceVoronoiPath(positions[p], directions[p], batchMaxDistance);
            if (actualv[p].segments().size() != expected.segments().size())
            {
                message = "Voronoi batch path segment-count mismatch";
                return false;
            }
            for (size_t i = 0; i != expected.segments().size(); ++i)
            {
                const auto& a = actualv[p].segments()[i];
                const auto& e = expected.segments()[i];
                if (a.m() != e.m() || abs(a.ds() - e.ds()) > 1e-10 * max(1., abs(e.ds()))
                    || abs(a.s() - e.s()) > 1e-10 * max(1., abs(e.s())))
                {
                    message = "Voronoi batch path segment mismatch";
                    return false;
                }
            }
        }

        vector<int> blockBeginv(8);
        vector<int> blockCountv(8, 1);
        vector<int> blockIndexv(8);
        for (int block = 0; block != 8; ++block)
        {
            blockBeginv[block] = block;
            blockIndexv[block] = block >= 4 ? 1 : 0;
        }

        vector<SpatialGridPath> actualBlockv;
        vector<SpatialGridPath*> blockPathv;
        actualBlockv.reserve(positions.size());
        blockPathv.reserve(positions.size());
        for (size_t i = 0; i != positions.size(); ++i)
        {
            actualBlockv.emplace_back(positions[i], directions[i]);
            blockPathv.push_back(&actualBlockv.back());
        }
        if (!computeVoronoiPaths(voronoiSites.data(), voronoiSites, voronoiNeighborBegin, voronoiNeighborCount,
                                 voronoiNeighborIndex, blockBeginv, blockCountv, blockIndexv, 2, 2, voronoiEps, 0.,
                                 0., 0., 1., 1., 1., batchMaxDistance, blockPathv))
        {
            message = "GPU Voronoi block-list batch path kernel failed: " + lastRuntimeError();
            return false;
        }
        for (size_t p = 0; p != actualBlockv.size(); ++p)
        {
            SpatialGridPath expected = referenceVoronoiPath(positions[p], directions[p], batchMaxDistance);
            if (actualBlockv[p].segments().size() != expected.segments().size())
            {
                message = "Voronoi block-list batch path segment-count mismatch";
                return false;
            }
            for (size_t i = 0; i != expected.segments().size(); ++i)
            {
                const auto& a = actualBlockv[p].segments()[i];
                const auto& e = expected.segments()[i];
                if (a.m() != e.m() || abs(a.ds() - e.ds()) > 1e-10 * max(1., abs(e.ds()))
                    || abs(a.s() - e.s()) > 1e-10 * max(1., abs(e.s())))
                {
                    message = "Voronoi block-list batch path segment mismatch";
                    return false;
                }
            }
        }
    }

    MediumState state;
    state.initConfiguration(5, 2, 0);
    state.initCommonStateVariables({StateVariable::volume()});
    state.initSpecificStateVariables({StateVariable::numberDensity()});
    state.initSpecificStateVariables({StateVariable::numberDensity()});
    state.initAllocate();
    for (int m = 0; m != 5; ++m)
    {
        state.setVolume(m, 1.);
        state.setNumberDensity(m, 0, 0.25 * (m + 1));
        state.setNumberDensity(m, 1, 0.50 * (m + 1));
    }
    invalidateMediumState(state);

    auto fillPath = []() {
        SpatialGridPath path(Position(), Direction(1., 0., 0., false));
        for (int i = 0; i != 128; ++i)
        {
            int m = (i % 13 == 0) ? -1 : i % 5;
            double ds = 0.1 + 0.01 * (i % 7);
            path.addSegment(m, ds);
        }
        return path;
    };

    vector<double> extSections{2.0, 0.5};
    vector<double> expectedExt;
    double tau = 0.;
    SpatialGridPath extPath = fillPath();
    for (const auto& segment : extPath.segments())
    {
        if (segment.m() >= 0)
        {
            double n0 = state.numberDensity(segment.m(), 0);
            double n1 = state.numberDensity(segment.m(), 1);
            tau += (extSections[0] * n0 + extSections[1] * n1) * segment.ds();
        }
        expectedExt.push_back(tau);
    }
    if (!setExtinctionOpticalDepths(&extPath, state, extSections))
    {
        message = "GPU extinction optical-depth kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t i = 0; i != extPath.segments().size(); ++i)
    {
        if (abs(extPath.segments()[i].tauExtOrSca() - expectedExt[i]) > tolerance)
        {
            message = "extinction optical-depth kernel mismatch";
            return false;
        }
    }

    vector<double> expectedLds;
    double luminosity = 3.25;
    double lnExtBeg = 0.;
    double extBeg = 1.;
    for (const auto& segment : extPath.segments())
    {
        double lnExtEnd = -segment.tauExt();
        double extEnd = exp(lnExtEnd);
        double Lds = 0.;
        if (segment.m() >= 0)
        {
            double extMean = SpecialFunctions::lnmean(extEnd, extBeg, lnExtEnd, lnExtBeg);
            Lds = luminosity * extMean * segment.ds();
        }
        expectedLds.push_back(Lds);
        lnExtBeg = lnExtEnd;
        extBeg = extEnd;
    }
    vector<double> gpuLdsv;
    if (!radiationFieldContributions(&extPath, luminosity, gpuLdsv))
    {
        message = "GPU radiation-field contribution kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t i = 0; i != expectedLds.size(); ++i)
    {
        if (abs(gpuLdsv[i] - expectedLds[i]) > 1e-11 * max(1., abs(expectedLds[i])))
        {
            message = "radiation-field contribution kernel mismatch";
            return false;
        }
    }

    auto expectedRadiationFieldContributions = [](const SpatialGridPath& path, double packetLuminosity) {
        vector<double> expected;
        double lnExtBeg = 0.;
        double extBeg = 1.;
        for (const auto& segment : path.segments())
        {
            double lnExtEnd = -segment.tauExt();
            double extEnd = exp(lnExtEnd);
            double Lds = 0.;
            if (segment.m() >= 0)
            {
                double extMean = SpecialFunctions::lnmean(extEnd, extBeg, lnExtEnd, lnExtBeg);
                Lds = packetLuminosity * extMean * segment.ds();
            }
            expected.push_back(Lds);
            lnExtBeg = lnExtEnd;
            extBeg = extEnd;
        }
        return expected;
    };
    SpatialGridPath rfPathShort = fillPath();
    rfPathShort.segments().erase(rfPathShort.segments().begin() + 73, rfPathShort.segments().end());
    if (!setExtinctionOpticalDepths(&rfPathShort, state, extSections))
    {
        message = "GPU short extinction optical-depth kernel failed: " + lastRuntimeError();
        return false;
    }
    vector<const SpatialGridPath*> rfBatchPaths{&extPath, &rfPathShort};
    vector<double> rfBatchLuminosities{1.25, 4.5};
    vector<vector<double>> expectedRfBatch;
    for (size_t p = 0; p != rfBatchPaths.size(); ++p)
        expectedRfBatch.push_back(expectedRadiationFieldContributions(*rfBatchPaths[p], rfBatchLuminosities[p]));
    vector<int> rfBatchOffsets;
    vector<double> gpuBatchLdsv;
    if (!radiationFieldContributions(rfBatchPaths, rfBatchLuminosities, rfBatchOffsets, gpuBatchLdsv))
    {
        message = "GPU batched radiation-field contribution kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t p = 0; p != rfBatchPaths.size(); ++p)
    {
        if (static_cast<size_t>(rfBatchOffsets[p + 1] - rfBatchOffsets[p]) != rfBatchPaths[p]->segments().size())
        {
            message = "batched radiation-field contribution offset mismatch";
            return false;
        }
        size_t begin = static_cast<size_t>(rfBatchOffsets[p]);
        for (size_t i = 0; i != expectedRfBatch[p].size(); ++i)
        {
            double expected = expectedRfBatch[p][i];
            if (abs(gpuBatchLdsv[begin + i] - expected) > 1e-11 * max(1., abs(expected)))
            {
                message = "batched radiation-field contribution kernel mismatch";
                return false;
            }
        }
    }

    const int rfNumWavelengths = 7;
    vector<int> rfWavelengthBins{3, 3};
    vector<int> gpuRfBinIndices;
    vector<double> gpuRfSums;
    if (!radiationFieldContributionSums(rfBatchPaths, rfBatchLuminosities, rfWavelengthBins, rfNumWavelengths,
                                        gpuRfBinIndices, gpuRfSums))
    {
        message = "GPU batched radiation-field sum kernel failed: " + lastRuntimeError();
        return false;
    }
    vector<double> expectedRfSums(5 * rfNumWavelengths, 0.);
    for (size_t p = 0; p != rfBatchPaths.size(); ++p)
    {
        int ell = rfWavelengthBins[p];
        const auto& segments = rfBatchPaths[p]->segments();
        for (size_t i = 0; i != segments.size(); ++i)
        {
            int m = segments[i].m();
            if (m >= 0) expectedRfSums[m * rfNumWavelengths + ell] += expectedRfBatch[p][i];
        }
    }
    vector<double> actualRfSums(expectedRfSums.size(), 0.);
    if (gpuRfBinIndices.size() != gpuRfSums.size())
    {
        message = "batched radiation-field sum size mismatch";
        return false;
    }
    for (size_t i = 0; i != gpuRfBinIndices.size(); ++i)
    {
        int index = gpuRfBinIndices[i];
        if (index < 0 || static_cast<size_t>(index) >= actualRfSums.size())
        {
            message = "batched radiation-field sum index mismatch";
            return false;
        }
        actualRfSums[index] += gpuRfSums[i];
    }
    for (size_t i = 0; i != expectedRfSums.size(); ++i)
    {
        if (abs(actualRfSums[i] - expectedRfSums[i]) > 1e-11 * max(1., abs(expectedRfSums[i])))
        {
            message = "batched radiation-field sum kernel mismatch";
            return false;
        }
    }

    double gpuTau = 0.;
    if (!getExtinctionOpticalDepth(&extPath, state, extSections, std::numeric_limits<double>::infinity(), gpuTau))
    {
        message = "GPU peel-off extinction optical-depth kernel failed: " + lastRuntimeError();
        return false;
    }
    if (abs(gpuTau - expectedExt.back()) > tolerance)
    {
        message = "peel-off extinction optical-depth kernel mismatch";
        return false;
    }
    double gpuTauLimited = 0.;
    if (!getExtinctionOpticalDepth(&extPath, state, extSections, expectedExt[50], gpuTauLimited))
    {
        message = "GPU limited peel-off extinction optical-depth kernel failed: " + lastRuntimeError();
        return false;
    }
    if (!std::isinf(gpuTauLimited))
    {
        message = "peel-off extinction optical-depth threshold mismatch";
        return false;
    }

    bool found = false;
    int interactionCell = -1;
    double interactionDistance = 0.;
    double tauinteract = expectedExt[60] + 0.25 * (expectedExt[61] - expectedExt[60]);
    if (!findInteractionPointUsingExtinction(&extPath, state, extSections, tauinteract, found, interactionCell,
                                             interactionDistance))
    {
        message = "GPU extinction interaction-point kernel failed: " + lastRuntimeError();
        return false;
    }
    if (!found || interactionCell != extPath.segments()[61].m())
    {
        message = "extinction interaction-point cell mismatch";
        return false;
    }
    double expectedInteractionDistance =
        interpolateLinLin(tauinteract, expectedExt[60], expectedExt[61], extPath.segments()[60].s(),
                          extPath.segments()[61].s());
    if (abs(interactionDistance - expectedInteractionDistance) > tolerance)
    {
        message = "extinction interaction-point distance mismatch";
        return false;
    }

    SpatialGridPath extReferencePath = extPath;
    tauinteract = expectedExt[82] + 0.6 * (expectedExt[83] - expectedExt[82]);
    extReferencePath.findInteractionPoint(tauinteract);
    double cumulativeInteractionTauAbs = 0.;
    if (!findInteractionPointInCumulativePath(&extPath, tauinteract, false, interactionCell, interactionDistance,
                                              cumulativeInteractionTauAbs))
    {
        message = "GPU cumulative extinction interaction-point kernel failed: " + lastRuntimeError();
        return false;
    }
    if (interactionCell != extReferencePath.interactionCellIndex()
        || abs(interactionDistance - extReferencePath.interactionDistance()) > tolerance
        || abs(cumulativeInteractionTauAbs) > tolerance)
    {
        message = "cumulative extinction interaction-point mismatch";
        return false;
    }

    vector<double> scaSections{0.2, 0.4};
    vector<double> absSections{0.7, 0.1};
    vector<double> expectedSca;
    vector<double> expectedAbs;
    double tauSca = 0.;
    double tauAbs = 0.;
    SpatialGridPath scaAbsPath = fillPath();
    for (const auto& segment : scaAbsPath.segments())
    {
        if (segment.m() >= 0)
        {
            double n0 = state.numberDensity(segment.m(), 0);
            double n1 = state.numberDensity(segment.m(), 1);
            tauSca += (scaSections[0] * n0 + scaSections[1] * n1) * segment.ds();
            tauAbs += (absSections[0] * n0 + absSections[1] * n1) * segment.ds();
        }
        expectedSca.push_back(tauSca);
        expectedAbs.push_back(tauAbs);
    }
    if (!setScatteringAndAbsorptionOpticalDepths(&scaAbsPath, state, scaSections, absSections))
    {
        message = "GPU scattering/absorption optical-depth kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t i = 0; i != scaAbsPath.segments().size(); ++i)
    {
        if (abs(scaAbsPath.segments()[i].tauExtOrSca() - expectedSca[i]) > tolerance
            || abs(scaAbsPath.segments()[i].tauAbs() - expectedAbs[i]) > tolerance)
        {
            message = "scattering/absorption optical-depth kernel mismatch";
            return false;
        }
    }

    auto expectedExtinctionForPath = [&](const SpatialGridPath& path) {
        vector<double> expected;
        double cumulativeTau = 0.;
        for (const auto& segment : path.segments())
        {
            if (segment.m() >= 0)
            {
                double n0 = state.numberDensity(segment.m(), 0);
                double n1 = state.numberDensity(segment.m(), 1);
                cumulativeTau += (extSections[0] * n0 + extSections[1] * n1) * segment.ds();
            }
            expected.push_back(cumulativeTau);
        }
        return expected;
    };

    auto expectedScatteringAbsorptionForPath = [&](const SpatialGridPath& path, vector<double>& expectedScaOut,
                                                   vector<double>& expectedAbsOut) {
        double cumulativeSca = 0.;
        double cumulativeAbs = 0.;
        expectedScaOut.clear();
        expectedAbsOut.clear();
        for (const auto& segment : path.segments())
        {
            if (segment.m() >= 0)
            {
                double n0 = state.numberDensity(segment.m(), 0);
                double n1 = state.numberDensity(segment.m(), 1);
                cumulativeSca += (scaSections[0] * n0 + scaSections[1] * n1) * segment.ds();
                cumulativeAbs += (absSections[0] * n0 + absSections[1] * n1) * segment.ds();
            }
            expectedScaOut.push_back(cumulativeSca);
            expectedAbsOut.push_back(cumulativeAbs);
        }
    };

    SpatialGridPath batchExtPathA = fillPath();
    SpatialGridPath batchExtPathB = fillPath();
    batchExtPathB.segments().erase(batchExtPathB.segments().begin() + 37, batchExtPathB.segments().end());
    SpatialGridPath batchExtPathC = fillPath();
    batchExtPathC.segments().erase(batchExtPathC.segments().begin() + 91, batchExtPathC.segments().end());
    vector<SpatialGridPath*> batchExtOpticalDepthPaths{&batchExtPathA, &batchExtPathB, &batchExtPathC};
    vector<vector<double>> expectedExtBatch;
    for (const SpatialGridPath* path : batchExtOpticalDepthPaths)
        expectedExtBatch.push_back(expectedExtinctionForPath(*path));
    if (!setExtinctionOpticalDepths(batchExtOpticalDepthPaths, state, extSections))
    {
        message = "GPU batched extinction optical-depth kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t p = 0; p != batchExtOpticalDepthPaths.size(); ++p)
    {
        const auto& segments = batchExtOpticalDepthPaths[p]->segments();
        for (size_t i = 0; i != segments.size(); ++i)
        {
            if (abs(segments[i].tauExtOrSca() - expectedExtBatch[p][i]) > tolerance)
            {
                message = "batched extinction optical-depth kernel mismatch";
                return false;
            }
        }
    }

    SpatialGridPath batchScaAbsPathA = fillPath();
    SpatialGridPath batchScaAbsPathB = fillPath();
    batchScaAbsPathB.segments().erase(batchScaAbsPathB.segments().begin() + 43,
                                      batchScaAbsPathB.segments().end());
    SpatialGridPath batchScaAbsPathC = fillPath();
    batchScaAbsPathC.segments().erase(batchScaAbsPathC.segments().begin() + 79,
                                      batchScaAbsPathC.segments().end());
    vector<SpatialGridPath*> batchScaAbsOpticalDepthPaths{&batchScaAbsPathA, &batchScaAbsPathB,
                                                          &batchScaAbsPathC};
    vector<vector<double>> expectedScaBatch(batchScaAbsOpticalDepthPaths.size());
    vector<vector<double>> expectedAbsBatch(batchScaAbsOpticalDepthPaths.size());
    for (size_t p = 0; p != batchScaAbsOpticalDepthPaths.size(); ++p)
        expectedScatteringAbsorptionForPath(*batchScaAbsOpticalDepthPaths[p], expectedScaBatch[p],
                                            expectedAbsBatch[p]);
    if (!setScatteringAndAbsorptionOpticalDepths(batchScaAbsOpticalDepthPaths, state, scaSections, absSections))
    {
        message = "GPU batched scattering/absorption optical-depth kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t p = 0; p != batchScaAbsOpticalDepthPaths.size(); ++p)
    {
        const auto& segments = batchScaAbsOpticalDepthPaths[p]->segments();
        for (size_t i = 0; i != segments.size(); ++i)
        {
            if (abs(segments[i].tauExtOrSca() - expectedScaBatch[p][i]) > tolerance
                || abs(segments[i].tauAbs() - expectedAbsBatch[p][i]) > tolerance)
            {
                message = "batched scattering/absorption optical-depth kernel mismatch";
                return false;
            }
        }
    }

    double interactionTauAbs = 0.;
    tauinteract = expectedSca[73] + 0.4 * (expectedSca[74] - expectedSca[73]);
    if (!findInteractionPointUsingScatteringAndAbsorption(&scaAbsPath, state, scaSections, absSections, tauinteract,
                                                          found, interactionCell, interactionDistance,
                                                          interactionTauAbs))
    {
        message = "GPU scattering/absorption interaction-point kernel failed: " + lastRuntimeError();
        return false;
    }
    if (!found || interactionCell != scaAbsPath.segments()[74].m())
    {
        message = "scattering/absorption interaction-point cell mismatch";
        return false;
    }
    expectedInteractionDistance =
        interpolateLinLin(tauinteract, expectedSca[73], expectedSca[74], scaAbsPath.segments()[73].s(),
                          scaAbsPath.segments()[74].s());
    double expectedInteractionTauAbs =
        interpolateLinLin(tauinteract, expectedSca[73], expectedSca[74], expectedAbs[73], expectedAbs[74]);
    if (abs(interactionDistance - expectedInteractionDistance) > tolerance
        || abs(interactionTauAbs - expectedInteractionTauAbs) > tolerance)
    {
        message = "scattering/absorption interaction-point interpolation mismatch";
        return false;
    }

    SpatialGridPath scaAbsReferencePath = scaAbsPath;
    tauinteract = expectedSca[92] + 0.7 * (expectedSca[93] - expectedSca[92]);
    scaAbsReferencePath.findInteractionPoint(tauinteract);
    if (!findInteractionPointInCumulativePath(&scaAbsPath, tauinteract, true, interactionCell, interactionDistance,
                                              interactionTauAbs))
    {
        message = "GPU cumulative scattering/absorption interaction-point kernel failed: " + lastRuntimeError();
        return false;
    }
    if (interactionCell != scaAbsReferencePath.interactionCellIndex()
        || abs(interactionDistance - scaAbsReferencePath.interactionDistance()) > tolerance
        || abs(interactionTauAbs - scaAbsReferencePath.interactionOpticalDepth()) > tolerance)
    {
        message = "cumulative scattering/absorption interaction-point mismatch";
        return false;
    }

    SpatialGridPath extPathShort = fillPath();
    extPathShort.segments().erase(extPathShort.segments().begin() + 37, extPathShort.segments().end());
    double shortTauExt = 0.;
    for (auto& segment : extPathShort.segments())
    {
        if (segment.m() >= 0)
        {
            double n0 = state.numberDensity(segment.m(), 0);
            double n1 = state.numberDensity(segment.m(), 1);
            shortTauExt += (extSections[0] * n0 + extSections[1] * n1) * segment.ds();
        }
        segment.setOpticalDepth(shortTauExt);
    }
    vector<const SpatialGridPath*> batchedExtPaths{&extPath, &extPathShort};
    vector<double> batchedExtTau{expectedExt[12] + 0.35 * (expectedExt[13] - expectedExt[12]),
                                 extPathShort.totalOpticalDepth() * 0.65};
    vector<int> batchedCells;
    vector<double> batchedDistances;
    vector<double> batchedTauAbs;
    if (!findInteractionPointsInCumulativePaths(batchedExtPaths, batchedExtTau, false, batchedCells,
                                                batchedDistances, batchedTauAbs))
    {
        message = "GPU batched cumulative extinction interaction-point kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t i = 0; i != batchedExtPaths.size(); ++i)
    {
        SpatialGridPath reference = *batchedExtPaths[i];
        reference.findInteractionPoint(batchedExtTau[i]);
        if (batchedCells[i] != reference.interactionCellIndex()
            || abs(batchedDistances[i] - reference.interactionDistance()) > tolerance
            || abs(batchedTauAbs[i]) > tolerance)
        {
            message = "batched cumulative extinction interaction-point mismatch";
            return false;
        }
    }

    SpatialGridPath scaAbsPathShort = fillPath();
    scaAbsPathShort.segments().erase(scaAbsPathShort.segments().begin() + 53, scaAbsPathShort.segments().end());
    double shortTauSca = 0.;
    double shortTauAbs = 0.;
    for (auto& segment : scaAbsPathShort.segments())
    {
        if (segment.m() >= 0)
        {
            double n0 = state.numberDensity(segment.m(), 0);
            double n1 = state.numberDensity(segment.m(), 1);
            shortTauSca += (scaSections[0] * n0 + scaSections[1] * n1) * segment.ds();
            shortTauAbs += (absSections[0] * n0 + absSections[1] * n1) * segment.ds();
        }
        segment.setOpticalDepth(shortTauSca, shortTauAbs);
    }
    vector<const SpatialGridPath*> batchedScaAbsPaths{&scaAbsPath, &scaAbsPathShort};
    vector<double> batchedScaAbsTau{expectedSca[21] + 0.45 * (expectedSca[22] - expectedSca[21]),
                                    scaAbsPathShort.totalOpticalDepth() * 0.55};
    if (!findInteractionPointsInCumulativePaths(batchedScaAbsPaths, batchedScaAbsTau, true, batchedCells,
                                                batchedDistances, batchedTauAbs))
    {
        message = "GPU batched cumulative scattering/absorption interaction-point kernel failed: "
                  + lastRuntimeError();
        return false;
    }
    for (size_t i = 0; i != batchedScaAbsPaths.size(); ++i)
    {
        SpatialGridPath reference = *batchedScaAbsPaths[i];
        reference.findInteractionPoint(batchedScaAbsTau[i]);
        if (batchedCells[i] != reference.interactionCellIndex()
            || abs(batchedDistances[i] - reference.interactionDistance()) > tolerance
            || abs(batchedTauAbs[i] - reference.interactionOpticalDepth()) > tolerance)
        {
            message = "batched cumulative scattering/absorption interaction-point mismatch";
            return false;
        }
    }

    vector<double> forcedPathBiasWeights{1.2, 0.7};
    vector<double> forcedAlbedos{0.42, 0.86};
    vector<double> forcedWeights;
    if (!forcedPropagationResults(batchedExtPaths, batchedExtTau, forcedPathBiasWeights, false, forcedAlbedos,
                                  batchedCells, batchedDistances, batchedTauAbs, forcedWeights))
    {
        message = "GPU batched forced-propagation extinction kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t i = 0; i != batchedExtPaths.size(); ++i)
    {
        SpatialGridPath reference = *batchedExtPaths[i];
        reference.findInteractionPoint(batchedExtTau[i]);
        double expectedWeight = forcedPathBiasWeights[i] * (-expm1(-reference.totalOpticalDepth())) * forcedAlbedos[i];
        if (batchedCells[i] != reference.interactionCellIndex()
            || abs(batchedDistances[i] - reference.interactionDistance()) > tolerance
            || abs(batchedTauAbs[i]) > tolerance || abs(forcedWeights[i] - expectedWeight) > tolerance)
        {
            message = "batched forced-propagation extinction result mismatch";
            return false;
        }
    }

    forcedPathBiasWeights = {0.9, 1.1};
    if (!forcedPropagationResults(batchedScaAbsPaths, batchedScaAbsTau, forcedPathBiasWeights, true, {},
                                  batchedCells, batchedDistances, batchedTauAbs, forcedWeights))
    {
        message = "GPU batched forced-propagation scattering/absorption kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t i = 0; i != batchedScaAbsPaths.size(); ++i)
    {
        SpatialGridPath reference = *batchedScaAbsPaths[i];
        reference.findInteractionPoint(batchedScaAbsTau[i]);
        double expectedWeight = forcedPathBiasWeights[i] * (-expm1(-reference.totalOpticalDepth()))
                                * exp(-reference.interactionOpticalDepth());
        if (batchedCells[i] != reference.interactionCellIndex()
            || abs(batchedDistances[i] - reference.interactionDistance()) > tolerance
            || abs(batchedTauAbs[i] - reference.interactionOpticalDepth()) > tolerance
            || abs(forcedWeights[i] - expectedWeight) > tolerance)
        {
            message = "batched forced-propagation scattering/absorption result mismatch";
            return false;
        }
    }

    vector<double> scaPropertySections{0.15, 0.55};
    vector<double> extPropertySections{0.80, 1.25};
    double expectedKsca = 0.;
    double expectedKext = 0.;
    vector<double> expectedWeights(scaPropertySections.size());
    int scatteringCell = 3;
    for (int h = 0; h != static_cast<int>(scaPropertySections.size()); ++h)
    {
        double n = state.numberDensity(scatteringCell, h);
        expectedWeights[h] = scaPropertySections[h] * n;
        expectedKsca += expectedWeights[h];
        expectedKext += extPropertySections[h] * n;
    }
    for (double& weight : expectedWeights) weight /= expectedKsca;
    double gpuAlbedo = 0.;
    vector<double> gpuWeights;
    if (!scatteringProperties(state, scatteringCell, scaPropertySections, extPropertySections, gpuAlbedo, gpuWeights))
    {
        message = "GPU scattering properties kernel failed: " + lastRuntimeError();
        return false;
    }
    if (abs(gpuAlbedo - expectedKsca / expectedKext) > tolerance || gpuWeights.size() != expectedWeights.size())
    {
        message = "scattering properties albedo mismatch";
        return false;
    }
    for (size_t h = 0; h != expectedWeights.size(); ++h)
    {
        if (abs(gpuWeights[h] - expectedWeights[h]) > tolerance)
        {
            message = "scattering properties weight mismatch";
            return false;
        }
    }

    vector<int> albedoCells{0, 1, -1, 4, 2};
    vector<double> expectedAlbedos(albedoCells.size(), 0.);
    for (size_t i = 0; i != albedoCells.size(); ++i)
    {
        int m = albedoCells[i];
        if (m < 0) continue;
        double ksca = 0.;
        double kext = 0.;
        for (int h = 0; h != static_cast<int>(scaPropertySections.size()); ++h)
        {
            double n = state.numberDensity(m, h);
            ksca += scaPropertySections[h] * n;
            kext += extPropertySections[h] * n;
        }
        expectedAlbedos[i] = kext > 0. ? ksca / kext : 0.;
    }
    vector<double> gpuAlbedos;
    if (!scatteringAlbedos(state, albedoCells, scaPropertySections, extPropertySections, gpuAlbedos))
    {
        message = "GPU batched scattering albedo kernel failed: " + lastRuntimeError();
        return false;
    }
    if (gpuAlbedos.size() != expectedAlbedos.size())
    {
        message = "batched scattering albedo size mismatch";
        return false;
    }
    for (size_t i = 0; i != expectedAlbedos.size(); ++i)
    {
        if (abs(gpuAlbedos[i] - expectedAlbedos[i]) > tolerance)
        {
            message = "batched scattering albedo mismatch";
            return false;
        }
    }

    vector<int> tableMedia{0, 1};
    vector<int> tableLookupBegin{0, 5};
    vector<int> tableLookupCount{5, 5};
    vector<double> tableLookupWavelengths{0.1, 0.2, 0.5, 1.0, 2.0, 0.05, 0.3, 0.6, 1.2, 2.4};
    vector<double> tableScaSections(tableLookupWavelengths.size());
    vector<double> tableAbsSections(tableLookupWavelengths.size());
    vector<double> tableExtSections(tableLookupWavelengths.size());
    for (size_t i = 0; i != tableLookupWavelengths.size(); ++i)
    {
        tableScaSections[i] = 0.07 + 0.013 * (i + 1);
        tableAbsSections[i] = 0.19 + 0.017 * (i + 1);
        tableExtSections[i] = tableScaSections[i] + tableAbsSections[i];
    }
    auto tableLookupIndex = [&tableLookupWavelengths, &tableLookupBegin, &tableLookupCount](int h, double lambda) {
        int begin = tableLookupBegin[h];
        int count = tableLookupCount[h];
        if (lambda < tableLookupWavelengths[begin]) return begin;
        int jl = -1;
        int ju = count - 1;
        while (ju - jl > 1)
        {
            int jm = (ju + jl) >> 1;
            if (lambda < tableLookupWavelengths[begin + jm])
                ju = jm;
            else
                jl = jm;
        }
        return begin + (jl < 0 ? 0 : jl);
    };
    double tableLambda = 0.75;
    vector<double> selectedTableSca(tableMedia.size());
    vector<double> selectedTableAbs(tableMedia.size());
    vector<double> selectedTableExt(tableMedia.size());
    for (int h = 0; h != static_cast<int>(tableMedia.size()); ++h)
    {
        int index = tableLookupIndex(h, tableLambda);
        selectedTableSca[h] = tableScaSections[index];
        selectedTableAbs[h] = tableAbsSections[index];
        selectedTableExt[h] = tableExtSections[index];
    }

    SpatialGridPath tableExtPath = fillPath();
    vector<double> expectedTableExt;
    tau = 0.;
    for (const auto& segment : tableExtPath.segments())
    {
        if (segment.m() >= 0)
        {
            double n0 = state.numberDensity(segment.m(), 0);
            double n1 = state.numberDensity(segment.m(), 1);
            tau += (selectedTableExt[0] * n0 + selectedTableExt[1] * n1) * segment.ds();
        }
        expectedTableExt.push_back(tau);
    }
    if (!setExtinctionOpticalDepthsFromTables(&tableExtPath, state, tableMedia, tableLookupBegin, tableLookupCount,
                                              tableLookupWavelengths, tableExtSections, tableLambda))
    {
        message = "GPU table extinction optical-depth kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t i = 0; i != tableExtPath.segments().size(); ++i)
    {
        if (abs(tableExtPath.segments()[i].tauExtOrSca() - expectedTableExt[i]) > tolerance)
        {
            message = "table extinction optical-depth kernel mismatch";
            return false;
        }
    }
    if (!getExtinctionOpticalDepthFromTables(&tableExtPath, state, tableMedia, tableLookupBegin, tableLookupCount,
                                             tableLookupWavelengths, tableExtSections, tableLambda,
                                             std::numeric_limits<double>::infinity(), gpuTau))
    {
        message = "GPU table peel-off extinction optical-depth kernel failed: " + lastRuntimeError();
        return false;
    }
    if (abs(gpuTau - expectedTableExt.back()) > tolerance)
    {
        message = "table peel-off extinction optical-depth kernel mismatch";
        return false;
    }

    tauinteract = expectedTableExt[43] + 0.35 * (expectedTableExt[44] - expectedTableExt[43]);
    if (!findInteractionPointUsingExtinctionFromTables(&tableExtPath, state, tableMedia, tableLookupBegin,
                                                       tableLookupCount, tableLookupWavelengths, tableExtSections,
                                                       tableLambda, tauinteract, found, interactionCell,
                                                       interactionDistance))
    {
        message = "GPU table extinction interaction-point kernel failed: " + lastRuntimeError();
        return false;
    }
    expectedInteractionDistance =
        interpolateLinLin(tauinteract, expectedTableExt[43], expectedTableExt[44], tableExtPath.segments()[43].s(),
                          tableExtPath.segments()[44].s());
    if (!found || interactionCell != tableExtPath.segments()[44].m()
        || abs(interactionDistance - expectedInteractionDistance) > tolerance)
    {
        message = "table extinction interaction-point mismatch";
        return false;
    }

    SpatialGridPath tableScaAbsPath = fillPath();
    vector<double> expectedTableSca;
    vector<double> expectedTableAbs;
    tauSca = 0.;
    tauAbs = 0.;
    for (const auto& segment : tableScaAbsPath.segments())
    {
        if (segment.m() >= 0)
        {
            double n0 = state.numberDensity(segment.m(), 0);
            double n1 = state.numberDensity(segment.m(), 1);
            tauSca += (selectedTableSca[0] * n0 + selectedTableSca[1] * n1) * segment.ds();
            tauAbs += (selectedTableAbs[0] * n0 + selectedTableAbs[1] * n1) * segment.ds();
        }
        expectedTableSca.push_back(tauSca);
        expectedTableAbs.push_back(tauAbs);
    }
    if (!setScatteringAndAbsorptionOpticalDepthsFromTables(
            &tableScaAbsPath, state, tableMedia, tableLookupBegin, tableLookupCount, tableLookupWavelengths,
            tableScaSections, tableAbsSections, tableLambda))
    {
        message = "GPU table scattering/absorption optical-depth kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t i = 0; i != tableScaAbsPath.segments().size(); ++i)
    {
        if (abs(tableScaAbsPath.segments()[i].tauExtOrSca() - expectedTableSca[i]) > tolerance
            || abs(tableScaAbsPath.segments()[i].tauAbs() - expectedTableAbs[i]) > tolerance)
        {
            message = "table scattering/absorption optical-depth kernel mismatch";
            return false;
        }
    }
    tauinteract = expectedTableSca[52] + 0.55 * (expectedTableSca[53] - expectedTableSca[52]);
    if (!findInteractionPointUsingScatteringAndAbsorptionFromTables(
            &tableScaAbsPath, state, tableMedia, tableLookupBegin, tableLookupCount, tableLookupWavelengths,
            tableScaSections, tableAbsSections, tableLambda, tauinteract, found, interactionCell,
            interactionDistance, interactionTauAbs))
    {
        message = "GPU table scattering/absorption interaction-point kernel failed: " + lastRuntimeError();
        return false;
    }
    expectedInteractionDistance =
        interpolateLinLin(tauinteract, expectedTableSca[52], expectedTableSca[53],
                          tableScaAbsPath.segments()[52].s(), tableScaAbsPath.segments()[53].s());
    expectedInteractionTauAbs =
        interpolateLinLin(tauinteract, expectedTableSca[52], expectedTableSca[53], expectedTableAbs[52],
                          expectedTableAbs[53]);
    if (!found || interactionCell != tableScaAbsPath.segments()[53].m()
        || abs(interactionDistance - expectedInteractionDistance) > tolerance
        || abs(interactionTauAbs - expectedInteractionTauAbs) > tolerance)
    {
        message = "table scattering/absorption interaction-point mismatch";
        return false;
    }

    auto expectedTableExtinctionForPath = [&](const SpatialGridPath& path, double lambda) {
        vector<double> selected(tableMedia.size());
        for (int h = 0; h != static_cast<int>(tableMedia.size()); ++h)
            selected[h] = tableExtSections[tableLookupIndex(h, lambda)];
        vector<double> expected;
        double cumulativeTau = 0.;
        for (const auto& segment : path.segments())
        {
            if (segment.m() >= 0)
            {
                double n0 = state.numberDensity(segment.m(), 0);
                double n1 = state.numberDensity(segment.m(), 1);
                cumulativeTau += (selected[0] * n0 + selected[1] * n1) * segment.ds();
            }
            expected.push_back(cumulativeTau);
        }
        return expected;
    };

    auto expectedTableScatteringAbsorptionForPath = [&](const SpatialGridPath& path, double lambda,
                                                        vector<double>& expectedScaOut,
                                                        vector<double>& expectedAbsOut) {
        vector<double> selectedSca(tableMedia.size());
        vector<double> selectedAbs(tableMedia.size());
        for (int h = 0; h != static_cast<int>(tableMedia.size()); ++h)
        {
            selectedSca[h] = tableScaSections[tableLookupIndex(h, lambda)];
            selectedAbs[h] = tableAbsSections[tableLookupIndex(h, lambda)];
        }
        double cumulativeSca = 0.;
        double cumulativeAbs = 0.;
        expectedScaOut.clear();
        expectedAbsOut.clear();
        for (const auto& segment : path.segments())
        {
            if (segment.m() >= 0)
            {
                double n0 = state.numberDensity(segment.m(), 0);
                double n1 = state.numberDensity(segment.m(), 1);
                cumulativeSca += (selectedSca[0] * n0 + selectedSca[1] * n1) * segment.ds();
                cumulativeAbs += (selectedAbs[0] * n0 + selectedAbs[1] * n1) * segment.ds();
            }
            expectedScaOut.push_back(cumulativeSca);
            expectedAbsOut.push_back(cumulativeAbs);
        }
    };

    SpatialGridPath batchTableExtPathA = fillPath();
    SpatialGridPath batchTableExtPathB = fillPath();
    batchTableExtPathB.segments().erase(batchTableExtPathB.segments().begin() + 48,
                                        batchTableExtPathB.segments().end());
    SpatialGridPath batchTableExtPathC = fillPath();
    batchTableExtPathC.segments().erase(batchTableExtPathC.segments().begin() + 83,
                                        batchTableExtPathC.segments().end());
    vector<SpatialGridPath*> batchTableExtPaths{&batchTableExtPathA, &batchTableExtPathB, &batchTableExtPathC};
    vector<double> batchTableLambdav{0.18, 0.75, 1.5};
    vector<vector<double>> expectedTableExtBatch;
    for (size_t p = 0; p != batchTableExtPaths.size(); ++p)
        expectedTableExtBatch.push_back(expectedTableExtinctionForPath(*batchTableExtPaths[p],
                                                                       batchTableLambdav[p]));
    if (!setExtinctionOpticalDepthsFromTables(batchTableExtPaths, state, tableMedia, tableLookupBegin,
                                              tableLookupCount, tableLookupWavelengths, tableExtSections,
                                              batchTableLambdav))
    {
        message = "GPU batched table extinction optical-depth kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t p = 0; p != batchTableExtPaths.size(); ++p)
    {
        const auto& segments = batchTableExtPaths[p]->segments();
        for (size_t i = 0; i != segments.size(); ++i)
        {
            if (abs(segments[i].tauExtOrSca() - expectedTableExtBatch[p][i]) > tolerance)
            {
                message = "batched table extinction optical-depth kernel mismatch";
                return false;
            }
        }
    }

    SpatialGridPath batchTableScaAbsPathA = fillPath();
    SpatialGridPath batchTableScaAbsPathB = fillPath();
    batchTableScaAbsPathB.segments().erase(batchTableScaAbsPathB.segments().begin() + 57,
                                           batchTableScaAbsPathB.segments().end());
    SpatialGridPath batchTableScaAbsPathC = fillPath();
    batchTableScaAbsPathC.segments().erase(batchTableScaAbsPathC.segments().begin() + 96,
                                           batchTableScaAbsPathC.segments().end());
    vector<SpatialGridPath*> batchTableScaAbsPaths{&batchTableScaAbsPathA, &batchTableScaAbsPathB,
                                                   &batchTableScaAbsPathC};
    vector<vector<double>> expectedTableScaBatch(batchTableScaAbsPaths.size());
    vector<vector<double>> expectedTableAbsBatch(batchTableScaAbsPaths.size());
    for (size_t p = 0; p != batchTableScaAbsPaths.size(); ++p)
        expectedTableScatteringAbsorptionForPath(*batchTableScaAbsPaths[p], batchTableLambdav[p],
                                                 expectedTableScaBatch[p], expectedTableAbsBatch[p]);
    if (!setScatteringAndAbsorptionOpticalDepthsFromTables(
            batchTableScaAbsPaths, state, tableMedia, tableLookupBegin, tableLookupCount, tableLookupWavelengths,
            tableScaSections, tableAbsSections, batchTableLambdav))
    {
        message = "GPU batched table scattering/absorption optical-depth kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t p = 0; p != batchTableScaAbsPaths.size(); ++p)
    {
        const auto& segments = batchTableScaAbsPaths[p]->segments();
        for (size_t i = 0; i != segments.size(); ++i)
        {
            if (abs(segments[i].tauExtOrSca() - expectedTableScaBatch[p][i]) > tolerance
                || abs(segments[i].tauAbs() - expectedTableAbsBatch[p][i]) > tolerance)
            {
                message = "batched table scattering/absorption optical-depth kernel mismatch";
                return false;
            }
        }
    }

    {
        vector<Position> fusedPositions{Position(0.2, 0.45, 0.45), Position(-0.3, 0.4, 0.45),
                                        Position(0.8, 0.55, 0.5)};
        vector<Direction> fusedDirections{Direction(1.0, 0.1, 0.05, true), Direction(1.0, 0.2, 0.1, true),
                                          Direction(-1.0, 0.05, 0.02, true)};
        vector<double> fusedLambdav{0.18, 0.75, 1.5};
        vector<int> blockBeginv(8);
        vector<int> blockCountv(8, 1);
        vector<int> blockIndexv(8);
        for (int block = 0; block != 8; ++block)
        {
            blockBeginv[block] = block;
            blockIndexv[block] = block >= 4 ? 1 : 0;
        }

        vector<SpatialGridPath> fusedExtPathv;
        vector<SpatialGridPath*> fusedExtPathPtrv;
        fusedExtPathv.reserve(fusedPositions.size());
        fusedExtPathPtrv.reserve(fusedPositions.size());
        for (size_t p = 0; p != fusedPositions.size(); ++p)
        {
            fusedExtPathv.emplace_back(fusedPositions[p], fusedDirections[p]);
            fusedExtPathPtrv.push_back(&fusedExtPathv.back());
        }
        if (!computeVoronoiTableOpticalDepthPaths(
                voronoiSites.data(), voronoiSites, voronoiNeighborBegin, voronoiNeighborCount, voronoiNeighborIndex,
                blockBeginv, blockCountv, blockIndexv, 2, 2, voronoiEps, 0., 0., 0., 1., 1., 1.,
                std::numeric_limits<double>::infinity(), fusedExtPathPtrv, state, tableMedia, tableLookupBegin,
                tableLookupCount, tableLookupWavelengths, tableExtSections, nullptr, fusedLambdav))
        {
            message = "GPU fused Voronoi/table extinction path kernel failed: " + lastRuntimeError();
            return false;
        }
        for (size_t p = 0; p != fusedExtPathv.size(); ++p)
        {
            SpatialGridPath expected = referenceVoronoiPath(fusedPositions[p], fusedDirections[p],
                                                            std::numeric_limits<double>::infinity());
            vector<double> expectedTau = expectedTableExtinctionForPath(expected, fusedLambdav[p]);
            const auto& actualSegments = fusedExtPathv[p].segments();
            const auto& expectedSegments = expected.segments();
            if (actualSegments.size() != expectedSegments.size())
            {
                message = "fused Voronoi/table extinction path segment-count mismatch";
                return false;
            }
            for (size_t i = 0; i != expectedSegments.size(); ++i)
            {
                if (actualSegments[i].m() != expectedSegments[i].m()
                    || abs(actualSegments[i].ds() - expectedSegments[i].ds()) > 1e-10 * max(1., abs(expectedSegments[i].ds()))
                    || abs(actualSegments[i].tauExtOrSca() - expectedTau[i]) > tolerance)
                {
                    message = "fused Voronoi/table extinction path mismatch";
                    return false;
                }
            }
        }

        vector<const SpatialGridPath*> fusedExtConstPathPtrv;
        fusedExtConstPathPtrv.reserve(fusedExtPathv.size());
        for (const auto& path : fusedExtPathv) fusedExtConstPathPtrv.push_back(&path);
        vector<double> fusedMaxDistancev(fusedExtPathv.size(), std::numeric_limits<double>::infinity());
        vector<double> fusedTotalTauv;
        if (!computeVoronoiTableOpticalDepthTotals(
                voronoiSites.data(), voronoiSites, voronoiNeighborBegin, voronoiNeighborCount, voronoiNeighborIndex,
                blockBeginv, blockCountv, blockIndexv, 2, 2, voronoiEps, 0., 0., 0., 1., 1., 1.,
                fusedExtConstPathPtrv, state, tableMedia, tableLookupBegin, tableLookupCount,
                tableLookupWavelengths, tableExtSections, fusedLambdav, fusedMaxDistancev, fusedTotalTauv))
        {
            message = "GPU fused Voronoi/table total extinction kernel failed: " + lastRuntimeError();
            return false;
        }
        if (fusedTotalTauv.size() != fusedExtPathv.size())
        {
            message = "fused Voronoi/table total extinction output size mismatch";
            return false;
        }
        for (size_t p = 0; p != fusedExtPathv.size(); ++p)
        {
            SpatialGridPath expected = referenceVoronoiPath(fusedPositions[p], fusedDirections[p],
                                                            std::numeric_limits<double>::infinity());
            vector<double> expectedTau = expectedTableExtinctionForPath(expected, fusedLambdav[p]);
            double expectedTotal = expectedTau.empty() ? 0. : expectedTau.back();
            if (abs(fusedTotalTauv[p] - expectedTotal) > tolerance)
            {
                message = "fused Voronoi/table total extinction mismatch";
                return false;
            }
        }

        Direction fusedObserverDirection(1.0, 0.1, 0.05, true);
        vector<SpatialGridPath> fusedObservedPathv;
        vector<const SpatialGridPath*> fusedObservedConstPathPtrv;
        fusedObservedPathv.reserve(fusedPositions.size());
        fusedObservedConstPathPtrv.reserve(fusedPositions.size());
        for (size_t p = 0; p != fusedPositions.size(); ++p)
        {
            fusedObservedPathv.emplace_back(fusedPositions[p], fusedObserverDirection);
            fusedObservedConstPathPtrv.push_back(&fusedObservedPathv.back());
        }
        vector<double> fusedObservedTauv;
        if (!computeVoronoiTableOpticalDepthTotals(
                voronoiSites.data(), voronoiSites, voronoiNeighborBegin, voronoiNeighborCount, voronoiNeighborIndex,
                blockBeginv, blockCountv, blockIndexv, 2, 2, voronoiEps, 0., 0., 0., 1., 1., 1.,
                fusedObservedConstPathPtrv, state, tableMedia, tableLookupBegin, tableLookupCount,
                tableLookupWavelengths, tableExtSections, fusedLambdav, fusedMaxDistancev, fusedObservedTauv))
        {
            message = "GPU fused Voronoi/table observed extinction reference failed: " + lastRuntimeError();
            return false;
        }
        vector<Direction> fusedObserverDirectionv(fusedPositions.size(), fusedObserverDirection);
        vector<double> fusedObservedDirectTauv;
        if (!computeVoronoiTableOpticalDepthTotals(
                voronoiSites.data(), voronoiSites, voronoiNeighborBegin, voronoiNeighborCount, voronoiNeighborIndex,
                blockBeginv, blockCountv, blockIndexv, 2, 2, voronoiEps, 0., 0., 0., 1., 1., 1.,
                fusedPositions, fusedObserverDirectionv, state, tableMedia, tableLookupBegin, tableLookupCount,
                tableLookupWavelengths, tableExtSections, fusedLambdav, fusedMaxDistancev,
                fusedObservedDirectTauv))
        {
            message = "GPU fused Voronoi/table direct observed extinction kernel failed: " + lastRuntimeError();
            return false;
        }
        if (fusedObservedDirectTauv.size() != fusedObservedTauv.size())
        {
            message = "fused Voronoi/table direct observed extinction output size mismatch";
            return false;
        }
        for (size_t p = 0; p != fusedObservedTauv.size(); ++p)
        {
            if (abs(fusedObservedDirectTauv[p] - fusedObservedTauv[p]) > tolerance)
            {
                message = "fused Voronoi/table direct observed extinction mismatch";
                return false;
            }
        }
        vector<double> fusedObservedSharedDirectionTauv;
        if (!computeVoronoiTableOpticalDepthTotals(
                voronoiSites.data(), voronoiSites, voronoiNeighborBegin, voronoiNeighborCount, voronoiNeighborIndex,
                blockBeginv, blockCountv, blockIndexv, 2, 2, voronoiEps, 0., 0., 0., 1., 1., 1.,
                fusedPositions, fusedObserverDirection, state, tableMedia, tableLookupBegin, tableLookupCount,
                tableLookupWavelengths, tableExtSections, fusedLambdav, fusedMaxDistancev,
                fusedObservedSharedDirectionTauv))
        {
            message = "GPU fused Voronoi/table shared-direction observed extinction kernel failed: "
                      + lastRuntimeError();
            return false;
        }
        if (fusedObservedSharedDirectionTauv.size() != fusedObservedTauv.size())
        {
            message = "fused Voronoi/table shared-direction observed extinction output size mismatch";
            return false;
        }
        for (size_t p = 0; p != fusedObservedTauv.size(); ++p)
        {
            if (abs(fusedObservedSharedDirectionTauv[p] - fusedObservedTauv[p]) > tolerance)
            {
                message = "fused Voronoi/table shared-direction observed extinction mismatch";
                return false;
            }
        }
        vector<double> fusedInputDirectionv{0., 0., 1.,
                                            sqrt(0.75), 0., 0.5,
                                            sqrt(1. - 0.25 * 0.25), 0., -0.25};
        vector<double> fusedPacketLuminosityv{2.0, 3.0, 4.0};
        vector<double> fusedAsymmparv{0.0, 0.35, 0.7, 0.96, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8};
        vector<double> fusedHgOnlyLuminosityv;
        if (!henyeyGreensteinScatteringLuminosities(
                fusedInputDirectionv, fusedPacketLuminosityv, fusedLambdav, fusedObserverDirection,
                tableLookupBegin.front(), tableLookupCount.front(), tableLookupWavelengths, fusedAsymmparv,
                fusedHgOnlyLuminosityv))
        {
            message = "GPU fused HG observed reference luminosity kernel failed: " + lastRuntimeError();
            return false;
        }
        vector<double> fusedObservedLuminosityv;
        if (!computeVoronoiTableHenyeyGreensteinObservedLuminosities(
                voronoiSites.data(), voronoiSites, voronoiNeighborBegin, voronoiNeighborCount, voronoiNeighborIndex,
                blockBeginv, blockCountv, blockIndexv, 2, 2, voronoiEps, 0., 0., 0., 1., 1., 1.,
                fusedObservedConstPathPtrv, state, tableMedia, tableLookupBegin, tableLookupCount,
                tableLookupWavelengths, tableExtSections, fusedLambdav, fusedMaxDistancev, fusedInputDirectionv,
                fusedPacketLuminosityv, fusedObserverDirection, tableLookupBegin.front(), tableLookupCount.front(),
                fusedAsymmparv, fusedObservedLuminosityv))
        {
            message = "GPU fused Voronoi/table HG observed luminosity kernel failed: " + lastRuntimeError();
            return false;
        }
        if (fusedObservedLuminosityv.size() != fusedHgOnlyLuminosityv.size())
        {
            message = "fused Voronoi/table HG observed luminosity output size mismatch";
            return false;
        }
	        for (size_t p = 0; p != fusedObservedLuminosityv.size(); ++p)
	        {
	            double expected = fusedHgOnlyLuminosityv[p] * exp(-fusedObservedTauv[p]);
	            if (abs(fusedObservedLuminosityv[p] - expected) > 1e-11 * max(1., abs(expected)))
	            {
	                message = "fused Voronoi/table HG observed luminosity mismatch";
	                return false;
	            }
	        }

	        double frameCostheta = 1.0;
	        double frameSintheta = 0.0;
	        double frameCosphi = 1.0;
	        double frameSinphi = 0.0;
	        double frameCosomega = 1.0;
	        double frameSinomega = 0.0;
	        int frameNumPixelsX = 3;
	        int frameNumPixelsY = 3;
	        double frameXpmin = -1.0;
	        double frameXpsiz = 1.0;
	        double frameYpmin = -1.0;
	        double frameYpsiz = 1.0;
	        double frameRedshift = 0.1;
	        size_t frameNumPixels = static_cast<size_t>(frameNumPixelsX) * static_cast<size_t>(frameNumPixelsY);
	        vector<int> frameBandOffsetv{0, 3, 6, 9};
	        vector<double> frameBandWavelengthv{0.1, 0.3, 0.5, 0.6, 1.0, 1.4, 1.4, 1.8, 2.2};
	        vector<double> frameBandTransmissionv{0.0, 2.0, 0.0, 0.0, 1.5, 0.0, 0.0, 3.0, 0.0};
	        vector<double> frameBandWidthv{0.4, 0.8, 0.5};
	        vector<double> expectedHgFrameSums(frameBandWidthv.size() * frameNumPixels, 0.);
	        auto hgFrameTransmission = [&](size_t ell, double lambda) {
	            int begin = frameBandOffsetv[ell];
	            int end = frameBandOffsetv[ell + 1];
	            if (!(lambda > frameBandWavelengthv[begin] && lambda < frameBandWavelengthv[end - 1])) return 0.;
	            auto first = frameBandWavelengthv.begin() + begin;
	            auto last = frameBandWavelengthv.begin() + end;
	            int i = static_cast<int>(std::upper_bound(first, last, lambda) - frameBandWavelengthv.begin());
	            if (i <= begin || i >= end) return 0.;
	            double x0 = frameBandWavelengthv[i - 1];
	            double x1 = frameBandWavelengthv[i];
	            double y0 = frameBandTransmissionv[i - 1];
	            double y1 = frameBandTransmissionv[i];
	            double t = x1 != x0 ? (lambda - x0) / (x1 - x0) : 0.;
	            return (y0 + t * (y1 - y0)) * frameBandWidthv[ell];
	        };
	        for (size_t p = 0; p != fusedPositions.size(); ++p)
	        {
	            double x, y, z;
	            fusedPositions[p].cartesian(x, y, z);
	            double xpp = -frameSinphi * x + frameCosphi * y;
	            double ypp = -frameCosphi * frameCostheta * x - frameSinphi * frameCostheta * y
	                         + frameSintheta * z;
	            double xp = frameCosomega * xpp - frameSinomega * ypp;
	            double yp = frameSinomega * xpp + frameCosomega * ypp;
	            int i = static_cast<int>(floor((xp - frameXpmin) / frameXpsiz));
	            int j = static_cast<int>(floor((yp - frameYpmin) / frameYpsiz));
	            if (i < 0 || i >= frameNumPixelsX || j < 0 || j >= frameNumPixelsY) continue;
	            size_t pixel = static_cast<size_t>(i + frameNumPixelsX * j);
	            double lambda = fusedLambdav[p] * (1.0 + frameRedshift);
	            for (size_t ell = 0; ell != frameBandWidthv.size(); ++ell)
	            {
	                double transmission = hgFrameTransmission(ell, lambda);
	                if (transmission == 0.) continue;
	                expectedHgFrameSums[pixel + ell * frameNumPixels] +=
	                    fusedObservedLuminosityv[p] * transmission;
	            }
	        }
	        int fusedFrameAccumulatorKeyObject = 0;
	        const void* fusedFrameAccumulatorKey = &fusedFrameAccumulatorKeyObject;
	        vector<double> actualHgFrameSums(expectedHgFrameSums.size(), 0.);
	        clearAccumulatedValues(fusedFrameAccumulatorKey);
	        if (!computeVoronoiTableHenyeyGreensteinFrameBandAccumulate(
	                voronoiSites.data(), voronoiSites, voronoiNeighborBegin, voronoiNeighborCount,
	                voronoiNeighborIndex, blockBeginv, blockCountv, blockIndexv, 2, 2, voronoiEps,
	                0., 0., 0., 1., 1., 1., fusedPositions, state, tableMedia, tableLookupBegin,
	                tableLookupCount, tableLookupWavelengths, tableExtSections, fusedLambdav,
	                fusedMaxDistancev, fusedInputDirectionv, fusedPacketLuminosityv, fusedObserverDirection,
	                tableLookupBegin.front(), tableLookupCount.front(), fusedAsymmparv,
	                fusedFrameAccumulatorKey, actualHgFrameSums.size(), frameCostheta, frameSintheta,
	                frameCosphi, frameSinphi, frameCosomega, frameSinomega, frameNumPixelsX,
	                frameNumPixelsY, frameXpmin, frameXpsiz, frameYpmin, frameYpsiz, frameRedshift,
	                frameNumPixels, frameBandOffsetv, frameBandWavelengthv, frameBandTransmissionv,
	                frameBandWidthv))
	        {
	            message = "GPU fused Voronoi/table HG frame-band accumulator failed: " + lastRuntimeError();
	            return false;
	        }
	        if (!retrieveAndClearAccumulatedValues(fusedFrameAccumulatorKey, actualHgFrameSums.data(),
	                                               actualHgFrameSums.size()))
	        {
	            message = "GPU fused Voronoi/table HG frame-band accumulator retrieval failed: "
	                      + lastRuntimeError();
	            return false;
	        }
	        for (size_t i = 0; i != expectedHgFrameSums.size(); ++i)
	        {
	            if (abs(actualHgFrameSums[i] - expectedHgFrameSums[i])
	                > 1e-11 * max(1., abs(expectedHgFrameSums[i])))
	            {
	                message = "fused Voronoi/table HG frame-band accumulator mismatch";
	                return false;
	            }
	        }

	        vector<SpatialGridPath> fusedScaAbsPathv;
        vector<SpatialGridPath*> fusedScaAbsPathPtrv;
        fusedScaAbsPathv.reserve(fusedPositions.size());
        fusedScaAbsPathPtrv.reserve(fusedPositions.size());
        for (size_t p = 0; p != fusedPositions.size(); ++p)
        {
            fusedScaAbsPathv.emplace_back(fusedPositions[p], fusedDirections[p]);
            fusedScaAbsPathPtrv.push_back(&fusedScaAbsPathv.back());
        }
        if (!computeVoronoiTableOpticalDepthPaths(
                voronoiSites.data(), voronoiSites, voronoiNeighborBegin, voronoiNeighborCount, voronoiNeighborIndex,
                blockBeginv, blockCountv, blockIndexv, 2, 2, voronoiEps, 0., 0., 0., 1., 1., 1.,
                std::numeric_limits<double>::infinity(), fusedScaAbsPathPtrv, state, tableMedia, tableLookupBegin,
                tableLookupCount, tableLookupWavelengths, tableScaSections, &tableAbsSections, fusedLambdav))
        {
            message = "GPU fused Voronoi/table scattering-absorption path kernel failed: " + lastRuntimeError();
            return false;
        }
	        for (size_t p = 0; p != fusedScaAbsPathv.size(); ++p)
	        {
	            SpatialGridPath expected = referenceVoronoiPath(fusedPositions[p], fusedDirections[p],
	                                                            std::numeric_limits<double>::infinity());
            vector<double> expectedSca;
            vector<double> expectedAbs;
            expectedTableScatteringAbsorptionForPath(expected, fusedLambdav[p], expectedSca, expectedAbs);
            const auto& actualSegments = fusedScaAbsPathv[p].segments();
            const auto& expectedSegments = expected.segments();
            if (actualSegments.size() != expectedSegments.size())
            {
                message = "fused Voronoi/table scattering-absorption path segment-count mismatch";
                return false;
            }
            for (size_t i = 0; i != expectedSegments.size(); ++i)
            {
                if (actualSegments[i].m() != expectedSegments[i].m()
                    || abs(actualSegments[i].ds() - expectedSegments[i].ds()) > 1e-10 * max(1., abs(expectedSegments[i].ds()))
                    || abs(actualSegments[i].tauExtOrSca() - expectedSca[i]) > tolerance
                    || abs(actualSegments[i].tauAbs() - expectedAbs[i]) > tolerance)
                {
                    message = "fused Voronoi/table scattering-absorption path mismatch";
                    return false;
	                }
	            }
	        }

	        vector<double> residentRfLuminosities{0.7, 1.3, 0.9};
	        vector<int> residentRfBins{2, -1, 4};
	        const int residentRfNumWavelengths = 7;
	        vector<double> residentForcedTau{fusedExtPathv[0].totalOpticalDepth() * 0.25,
	                                         fusedExtPathv[1].totalOpticalDepth() * 0.55,
	                                         fusedExtPathv[2].totalOpticalDepth() * 0.80};
	        vector<double> residentForcedBias{1.15, 0.85, 1.05};
	        vector<double> expectedResidentRfDense(2 * residentRfNumWavelengths, 0.);
	        for (size_t p = 0; p != fusedExtPathv.size(); ++p)
	        {
	            int ell = residentRfBins[p];
	            if (ell < 0) continue;
	            const auto& segments = fusedExtPathv[p].segments();
	            for (size_t i = 0; i != segments.size(); ++i)
	            {
	                int m = segments[i].m();
	                if (m < 0) continue;
	                double lnExtBeg = i == 0 ? 0. : -segments[i - 1].tauExt();
	                double lnExtEnd = -segments[i].tauExt();
	                double extBeg = exp(lnExtBeg);
	                double extEnd = exp(lnExtEnd);
	                double extMean = SpecialFunctions::lnmean(extEnd, extBeg, lnExtEnd, lnExtBeg);
	                expectedResidentRfDense[m * residentRfNumWavelengths + ell] +=
	                    residentRfLuminosities[p] * extMean * segments[i].ds();
	            }
	        }
	        vector<int> residentRfKeys;
	        vector<double> residentRfSums;
	        vector<int> residentForcedCells;
	        vector<double> residentForcedDistances;
	        vector<double> residentForcedTauAbs;
	        vector<double> residentForcedWeights;
	        if (!computeVoronoiTableRadiationFieldSumsAndForcedPropagationResults(
	                voronoiSites.data(), voronoiSites, voronoiNeighborBegin, voronoiNeighborCount, voronoiNeighborIndex,
		                blockBeginv, blockCountv, blockIndexv, 2, 2, voronoiEps, 0., 0., 0., 1., 1., 1.,
		                std::numeric_limits<double>::infinity(), fusedPositions, fusedDirections, state,
			                residentRfLuminosities, residentRfBins, residentRfNumWavelengths, residentForcedTau,
			                residentForcedBias, fusedLambdav, nullptr, nullptr, 0.0, tableMedia, tableLookupBegin,
			                tableLookupCount, tableLookupWavelengths, tableScaSections, tableExtSections,
			                residentRfKeys, residentRfSums, residentForcedCells, residentForcedDistances,
			                residentForcedTauAbs, residentForcedWeights, nullptr, nullptr, 0, 0, nullptr, nullptr))
	        {
	            message = "GPU resident Voronoi/table radiation-field/forced-propagation kernel failed: "
	                      + lastRuntimeError();
	            return false;
	        }
	        if (residentRfKeys.size() != residentRfSums.size())
	        {
	            message = "resident Voronoi/table radiation-field output size mismatch";
	            return false;
	        }
	        vector<double> actualResidentRfDense(expectedResidentRfDense.size(), 0.);
	        for (size_t i = 0; i != residentRfKeys.size(); ++i)
	        {
	            int key = residentRfKeys[i];
	            if (key < 0 || static_cast<size_t>(key) >= actualResidentRfDense.size())
	            {
	                message = "resident Voronoi/table radiation-field output key mismatch";
	                return false;
	            }
	            actualResidentRfDense[key] += residentRfSums[i];
	        }
	        for (size_t i = 0; i != expectedResidentRfDense.size(); ++i)
	        {
	            if (abs(actualResidentRfDense[i] - expectedResidentRfDense[i])
	                > 1e-11 * max(1., abs(expectedResidentRfDense[i])))
	            {
	                message = "resident Voronoi/table radiation-field sum mismatch";
	                return false;
	            }
	        }
	        if (residentForcedCells.size() != fusedExtPathv.size()
	            || residentForcedDistances.size() != fusedExtPathv.size()
	            || residentForcedTauAbs.size() != fusedExtPathv.size()
	            || residentForcedWeights.size() != fusedExtPathv.size())
	        {
	            message = "resident Voronoi/table forced-propagation output size mismatch";
	            return false;
	        }
	        for (size_t p = 0; p != fusedExtPathv.size(); ++p)
	        {
	            SpatialGridPath reference = fusedExtPathv[p];
	            reference.findInteractionPoint(residentForcedTau[p]);
	            double ksca = 0.;
	            double kext = 0.;
	            if (reference.interactionCellIndex() >= 0)
	            {
	                for (int h = 0; h != static_cast<int>(tableMedia.size()); ++h)
	                {
	                    int index = tableLookupIndex(h, fusedLambdav[p]);
	                    double n = state.numberDensity(reference.interactionCellIndex(), tableMedia[h]);
	                    ksca += tableScaSections[index] * n;
	                    kext += tableExtSections[index] * n;
	                }
	            }
	            double albedo = kext > 0. ? ksca / kext : 0.;
	            double expectedWeight = residentForcedBias[p] * (-expm1(-reference.totalOpticalDepth())) * albedo;
	            if (residentForcedCells[p] != reference.interactionCellIndex()
	                || abs(residentForcedDistances[p] - reference.interactionDistance()) > tolerance
	                || abs(residentForcedTauAbs[p]) > tolerance
	                || abs(residentForcedWeights[p] - expectedWeight) > tolerance)
	            {
	                message = "resident Voronoi/table forced-propagation result mismatch";
	                return false;
	            }
	        }

		        vector<double> sampledRandomSelect;
		        vector<double> sampledRandomSample(fusedExtPathv.size(), 0.0);
		        vector<double> sampledScatterCostheta{0.25, 0.65, 0.40};
		        vector<double> sampledScatterPhi{0.10, 0.35, 0.80};
		        vector<double> residentAsymmpar(tableLookupWavelengths.size(), 0.0);
		        for (size_t i = 0; i != residentAsymmpar.size(); ++i)
		            residentAsymmpar[i] = -0.25 + 0.08 * static_cast<double>(i);
		        sampledRandomSelect.reserve(fusedExtPathv.size());
	        for (size_t p = 0; p != fusedExtPathv.size(); ++p)
	        {
	            double taupath = fusedExtPathv[p].totalOpticalDepth();
	            double tau = residentForcedTau[p];
	            double u = 0.5;
	            if (taupath > 0.)
	            {
	                if (taupath < 1.0e-10)
	                    u = tau / taupath;
	                else
	                    u = (1.0 - exp(-tau)) / (1.0 - exp(-taupath));
	                if (u <= 0.) u = std::nextafter(0., 1.);
	                if (u >= 1.) u = std::nextafter(1., 0.);
	            }
	            sampledRandomSelect.push_back(u);
	        }
	        vector<int> sampledResidentRfKeys;
	        vector<double> sampledResidentRfSums;
	        vector<int> sampledResidentForcedCells;
	        vector<double> sampledResidentForcedDistances;
		        vector<double> sampledResidentForcedTauAbs;
		        vector<double> sampledResidentForcedWeights;
		        vector<double> sampledResidentScatterDirections;
		        const vector<double> noTauinteractv;
		        const vector<double> noPathBiasv;
		        if (!computeVoronoiTableRadiationFieldSumsAndForcedPropagationResults(
	                voronoiSites.data(), voronoiSites, voronoiNeighborBegin, voronoiNeighborCount, voronoiNeighborIndex,
	                blockBeginv, blockCountv, blockIndexv, 2, 2, voronoiEps, 0., 0., 0., 1., 1., 1.,
	                std::numeric_limits<double>::infinity(), fusedPositions, fusedDirections, state,
		                residentRfLuminosities, residentRfBins, residentRfNumWavelengths, noTauinteractv,
		                noPathBiasv, fusedLambdav, &sampledRandomSelect, &sampledRandomSample, 0.0, tableMedia,
		                tableLookupBegin, tableLookupCount, tableLookupWavelengths, tableScaSections,
		                tableExtSections, sampledResidentRfKeys, sampledResidentRfSums, sampledResidentForcedCells,
		                sampledResidentForcedDistances, sampledResidentForcedTauAbs, sampledResidentForcedWeights,
		                &sampledScatterCostheta, &sampledScatterPhi, tableLookupBegin.front(), tableLookupCount.front(),
		                &residentAsymmpar, &sampledResidentScatterDirections))
	        {
	            message = "GPU sampled resident Voronoi/table radiation-field/forced-propagation kernel failed: "
	                      + lastRuntimeError();
	            return false;
	        }
	        if (sampledResidentRfKeys.size() != residentRfKeys.size()
	            || sampledResidentRfSums.size() != residentRfSums.size())
	        {
	            message = "sampled resident Voronoi/table radiation-field output size mismatch";
	            return false;
	        }
	        vector<double> actualSampledResidentRfDense(expectedResidentRfDense.size(), 0.);
	        for (size_t i = 0; i != sampledResidentRfKeys.size(); ++i)
	        {
	            int key = sampledResidentRfKeys[i];
	            if (key < 0 || static_cast<size_t>(key) >= actualSampledResidentRfDense.size())
	            {
	                message = "sampled resident Voronoi/table radiation-field output key mismatch";
	                return false;
	            }
	            actualSampledResidentRfDense[key] += sampledResidentRfSums[i];
	        }
	        for (size_t i = 0; i != expectedResidentRfDense.size(); ++i)
	        {
	            if (abs(actualSampledResidentRfDense[i] - expectedResidentRfDense[i])
	                > 1e-11 * max(1., abs(expectedResidentRfDense[i])))
	            {
	                message = "sampled resident Voronoi/table radiation-field sum mismatch";
	                return false;
	            }
	        }
	        for (size_t p = 0; p != fusedExtPathv.size(); ++p)
	        {
	            double expectedSampledWeight = residentForcedWeights[p] / residentForcedBias[p];
	            if (sampledResidentForcedCells[p] != residentForcedCells[p]
	                || abs(sampledResidentForcedDistances[p] - residentForcedDistances[p]) > tolerance
	                || abs(sampledResidentForcedTauAbs[p] - residentForcedTauAbs[p]) > tolerance
	                || abs(sampledResidentForcedWeights[p] - expectedSampledWeight) > tolerance)
	            {
	                message = "sampled resident Voronoi/table forced-propagation result mismatch";
		                return false;
		            }
		        }
		        if (sampledResidentScatterDirections.size() != 3 * fusedExtPathv.size())
		        {
		            message = "sampled resident HG scatter direction output size mismatch";
		            return false;
		        }
		        auto residentHgDirection = [&](size_t p) {
		            int index = tableLookupIndex(0, fusedLambdav[p]);
		            double g = residentAsymmpar[index];
		            double u = sampledScatterCostheta[p];
		            double v = sampledScatterPhi[p];
		            double phi = 2. * M_PI * v;
		            double cosphi = cos(phi);
		            double sinphi = sin(phi);
		            double costheta = 0.;
		            if (abs(g) < 1e-6)
		            {
		                costheta = 2. * u - 1.;
		                double sintheta = sqrt(abs((1. - costheta) * (1. + costheta)));
		                return vector<double>{cosphi * sintheta, sinphi * sintheta, costheta};
		            }
		            double f = ((1. - g) * (1. + g)) / (1. - g + 2. * g * u);
		            costheta = (1. + g * g - f * f) / (2. * g);
		            double sintheta = sqrt(abs((1. - costheta) * (1. + costheta)));
		            double kx, ky, kz;
		            fusedDirections[p].cartesian(kx, ky, kz);
		            if (kz > 0.99999) return vector<double>{cosphi * sintheta, sinphi * sintheta, costheta};
		            if (kz < -0.99999) return vector<double>{cosphi * sintheta, sinphi * sintheta, -costheta};
		            double root = sqrt((1. - kz) * (1. + kz));
		            return vector<double>{sintheta / root * (-kx * kz * cosphi + ky * sinphi) + kx * costheta,
		                                  -sintheta / root * (ky * kz * cosphi + kx * sinphi) + ky * costheta,
		                                  root * sintheta * cosphi + kz * costheta};
		        };
		        for (size_t p = 0; p != fusedExtPathv.size(); ++p)
		        {
		            vector<double> expected = residentHgDirection(p);
		            for (size_t c = 0; c != 3; ++c)
		            {
		                if (abs(sampledResidentScatterDirections[3 * p + c] - expected[c]) > tolerance)
		                {
		                    message = "sampled resident HG scatter direction mismatch";
		                    return false;
		                }
		            }
		        }
	    }

	    auto expectedTableAlbedo = [&](int m, double lambda) {
        if (m < 0) return 0.;
        double ksca = 0.;
        double kext = 0.;
        for (int h = 0; h != static_cast<int>(tableMedia.size()); ++h)
        {
            int index = tableLookupIndex(h, lambda);
            double n = state.numberDensity(m, tableMedia[h]);
            ksca += tableScaSections[index] * n;
            kext += tableExtSections[index] * n;
        }
        return kext > 0. ? ksca / kext : 0.;
    };
    vector<const SpatialGridPath*> forcedTablePaths;
    for (const SpatialGridPath* path : batchTableExtPaths) forcedTablePaths.push_back(path);
    vector<double> forcedTableTau{batchTableExtPathA.totalOpticalDepth() * 0.25,
                                  batchTableExtPathB.totalOpticalDepth() * 0.55,
                                  batchTableExtPathC.totalOpticalDepth() * 0.80};
    vector<double> forcedTableBias{1.15, 0.85, 1.05};
    vector<double> forcedTableWeights;
    if (!forcedPropagationResultsFromTables(forcedTablePaths, state, forcedTableTau, forcedTableBias,
                                            batchTableLambdav, tableMedia, tableLookupBegin, tableLookupCount,
                                            tableLookupWavelengths, tableScaSections, tableExtSections,
                                            batchedCells, batchedDistances, batchedTauAbs, forcedTableWeights))
    {
        message = "GPU batched table forced-propagation kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t i = 0; i != forcedTablePaths.size(); ++i)
    {
        SpatialGridPath reference = *forcedTablePaths[i];
        reference.findInteractionPoint(forcedTableTau[i]);
        double albedo = expectedTableAlbedo(reference.interactionCellIndex(), batchTableLambdav[i]);
        double expectedWeight = forcedTableBias[i] * (-expm1(-reference.totalOpticalDepth())) * albedo;
        if (batchedCells[i] != reference.interactionCellIndex()
            || abs(batchedDistances[i] - reference.interactionDistance()) > tolerance
            || abs(batchedTauAbs[i]) > tolerance
            || abs(forcedTableWeights[i] - expectedWeight) > tolerance)
        {
            message = "batched table forced-propagation result mismatch";
            return false;
        }
    }

    vector<double> combinedRfLuminosities{0.7, 1.3, 0.9};
    vector<int> combinedRfBins{2, -1, 4};
    const int combinedRfNumWavelengths = 7;
    vector<double> expectedCombinedRfDense(5 * combinedRfNumWavelengths, 0.);
    for (size_t p = 0; p != forcedTablePaths.size(); ++p)
    {
        int ell = combinedRfBins[p];
        if (ell < 0) continue;
        const auto& segments = forcedTablePaths[p]->segments();
        for (size_t i = 0; i != segments.size(); ++i)
        {
            int m = segments[i].m();
            if (m < 0) continue;
            double lnExtBeg = i == 0 ? 0. : -segments[i - 1].tauExt();
            double lnExtEnd = -segments[i].tauExt();
            double extBeg = exp(lnExtBeg);
            double extEnd = exp(lnExtEnd);
            double extMean = SpecialFunctions::lnmean(extEnd, extBeg, lnExtEnd, lnExtBeg);
            expectedCombinedRfDense[m * combinedRfNumWavelengths + ell] +=
                combinedRfLuminosities[p] * extMean * segments[i].ds();
        }
    }
    vector<int> combinedRfKeys;
    vector<double> combinedRfSums;
    vector<int> combinedForcedCells;
    vector<double> combinedForcedDistances;
    vector<double> combinedForcedTauAbs;
    vector<double> combinedForcedWeights;
    if (!radiationFieldSumsAndForcedPropagationResultsFromTables(
            forcedTablePaths, state, combinedRfLuminosities, combinedRfBins, combinedRfNumWavelengths,
            forcedTableTau, forcedTableBias, batchTableLambdav, tableMedia, tableLookupBegin, tableLookupCount,
            tableLookupWavelengths, tableScaSections, tableExtSections, combinedRfKeys, combinedRfSums,
            combinedForcedCells, combinedForcedDistances, combinedForcedTauAbs, combinedForcedWeights))
    {
        message = "GPU combined radiation-field/forced-propagation kernel failed: " + lastRuntimeError();
        return false;
    }
    vector<double> actualCombinedRfDense(expectedCombinedRfDense.size(), 0.);
    if (combinedRfKeys.size() != combinedRfSums.size())
    {
        message = "combined radiation-field output size mismatch";
        return false;
    }
    for (size_t i = 0; i != combinedRfKeys.size(); ++i)
    {
        int key = combinedRfKeys[i];
        if (key < 0 || static_cast<size_t>(key) >= actualCombinedRfDense.size())
        {
            message = "combined radiation-field output key mismatch";
            return false;
        }
        actualCombinedRfDense[key] += combinedRfSums[i];
    }
    for (size_t i = 0; i != expectedCombinedRfDense.size(); ++i)
    {
        if (abs(actualCombinedRfDense[i] - expectedCombinedRfDense[i])
            > 1e-11 * max(1., abs(expectedCombinedRfDense[i])))
        {
            message = "combined radiation-field sum mismatch";
            return false;
        }
    }
    for (size_t i = 0; i != forcedTablePaths.size(); ++i)
    {
        if (combinedForcedCells[i] != batchedCells[i]
            || abs(combinedForcedDistances[i] - batchedDistances[i]) > tolerance
            || abs(combinedForcedTauAbs[i] - batchedTauAbs[i]) > tolerance
            || abs(combinedForcedWeights[i] - forcedTableWeights[i]) > tolerance)
        {
            message = "combined forced-propagation result mismatch";
            return false;
        }
    }

    expectedKsca = 0.;
    expectedKext = 0.;
    expectedWeights.assign(tableMedia.size(), 0.);
    for (int h = 0; h != static_cast<int>(tableMedia.size()); ++h)
    {
        double n = state.numberDensity(scatteringCell, h);
        expectedWeights[h] = selectedTableSca[h] * n;
        expectedKsca += expectedWeights[h];
        expectedKext += selectedTableExt[h] * n;
    }
    for (double& weight : expectedWeights) weight /= expectedKsca;
    if (!scatteringPropertiesFromTables(state, scatteringCell, tableMedia, tableLookupBegin, tableLookupCount,
                                        tableLookupWavelengths, tableScaSections, tableExtSections, tableLambda,
                                        gpuAlbedo, gpuWeights))
    {
        message = "GPU table scattering properties kernel failed: " + lastRuntimeError();
        return false;
    }
    if (abs(gpuAlbedo - expectedKsca / expectedKext) > tolerance || gpuWeights.size() != expectedWeights.size())
    {
        message = "table scattering properties albedo mismatch";
        return false;
    }
    for (size_t h = 0; h != expectedWeights.size(); ++h)
    {
        if (abs(gpuWeights[h] - expectedWeights[h]) > tolerance)
        {
            message = "table scattering properties weight mismatch";
            return false;
        }
    }

    vector<double> tableAlbedoLambdas{0.18, 0.75, 1.5, 0.6, 2.0};
    vector<double> expectedTableAlbedos(albedoCells.size(), 0.);
    for (size_t i = 0; i != albedoCells.size(); ++i)
    {
        int m = albedoCells[i];
        if (m < 0) continue;
        double ksca = 0.;
        double kext = 0.;
        for (int h = 0; h != static_cast<int>(tableMedia.size()); ++h)
        {
            int index = tableLookupIndex(h, tableAlbedoLambdas[i]);
            double n = state.numberDensity(m, tableMedia[h]);
            ksca += tableScaSections[index] * n;
            kext += tableExtSections[index] * n;
        }
        expectedTableAlbedos[i] = kext > 0. ? ksca / kext : 0.;
    }
    if (!scatteringAlbedosFromTables(state, albedoCells, tableAlbedoLambdas, tableMedia, tableLookupBegin,
                                     tableLookupCount, tableLookupWavelengths, tableScaSections, tableExtSections,
                                     gpuAlbedos))
    {
        message = "GPU batched table scattering albedo kernel failed: " + lastRuntimeError();
        return false;
    }
    if (gpuAlbedos.size() != expectedTableAlbedos.size())
    {
        message = "batched table scattering albedo size mismatch";
        return false;
    }
    for (size_t i = 0; i != expectedTableAlbedos.size(); ++i)
    {
        if (abs(gpuAlbedos[i] - expectedTableAlbedos[i]) > tolerance)
        {
            message = "batched table scattering albedo mismatch";
            return false;
        }
    }

    const int numWavelengths = 9;
    vector<int> dustMedia{0, 1};
    vector<double> dustAbsSections(dustMedia.size() * numWavelengths);
    vector<double> primaryRadiationField(5 * numWavelengths);
    vector<double> secondaryRadiationField(5 * numWavelengths);
    vector<double> expectedPrimaryDustLuminosities(5, 0.);
    vector<double> expectedSecondaryDustLuminosities(5, 0.);
    for (int j = 0; j != static_cast<int>(dustMedia.size()); ++j)
        for (int ell = 0; ell != numWavelengths; ++ell)
            dustAbsSections[j * numWavelengths + ell] = 0.03 * (j + 1) * (ell + 2);
    for (int m = 0; m != 5; ++m)
    {
        for (int ell = 0; ell != numWavelengths; ++ell)
        {
            double opacity = 0.;
            for (int j = 0; j != static_cast<int>(dustMedia.size()); ++j)
            {
                int h = dustMedia[j];
                opacity += dustAbsSections[j * numWavelengths + ell] * state.numberDensity(m, h);
            }
            int index = m * numWavelengths + ell;
            primaryRadiationField[index] = 0.2 * (m + 1) + 0.01 * ell;
            secondaryRadiationField[index] = 0.05 * (ell + 1) + 0.02 * m;
            expectedPrimaryDustLuminosities[m] += opacity * primaryRadiationField[index];
            expectedSecondaryDustLuminosities[m] += opacity * secondaryRadiationField[index];
        }
    }
    vector<double> gpuPrimaryDustLuminosities;
    vector<double> gpuSecondaryDustLuminosities;
    if (!dustAbsorbedLuminosities(state, 5, numWavelengths, dustMedia, dustAbsSections, primaryRadiationField.data(),
                                  secondaryRadiationField.data(), gpuPrimaryDustLuminosities,
                                  gpuSecondaryDustLuminosities))
    {
        message = "GPU dust absorbed luminosity kernel failed: " + lastRuntimeError();
        return false;
    }
    for (int m = 0; m != 5; ++m)
    {
        if (abs(gpuPrimaryDustLuminosities[m] - expectedPrimaryDustLuminosities[m]) > tolerance
            || abs(gpuSecondaryDustLuminosities[m] - expectedSecondaryDustLuminosities[m]) > tolerance)
        {
            message = "dust absorbed luminosity kernel mismatch";
            return false;
        }
    }
    double gpuPrimaryDustLuminosity = 0.;
    double gpuSecondaryDustLuminosity = 0.;
    if (!totalDustAbsorbedLuminosity(state, 5, numWavelengths, dustMedia, dustAbsSections,
                                     primaryRadiationField.data(), secondaryRadiationField.data(),
                                     gpuPrimaryDustLuminosity, gpuSecondaryDustLuminosity))
    {
        message = "GPU total dust absorbed luminosity kernel failed: " + lastRuntimeError();
        return false;
    }
    double expectedPrimaryDustLuminosity = 0.;
    double expectedSecondaryDustLuminosity = 0.;
    for (int m = 0; m != 5; ++m)
    {
        expectedPrimaryDustLuminosity += expectedPrimaryDustLuminosities[m];
        expectedSecondaryDustLuminosity += expectedSecondaryDustLuminosities[m];
    }
    if (abs(gpuPrimaryDustLuminosity - expectedPrimaryDustLuminosity) > tolerance
        || abs(gpuSecondaryDustLuminosity - expectedSecondaryDustLuminosity) > tolerance)
    {
        message = "total dust absorbed luminosity kernel mismatch";
        return false;
    }

    vector<double> wavelengthFactors{0.3, 0.5, 0.9, 1.1, 1.7, 2.0};
    vector<double> spectralValues{1.0, -2.0, 3.5, 4.25, -0.75, 8.0};
    vector<double> expectedSpectralValues = spectralValues;
    for (size_t ell = 0; ell != wavelengthFactors.size(); ++ell) expectedSpectralValues[ell] *= wavelengthFactors[ell];
    if (!scaleWavelengthValues(spectralValues.data(), spectralValues.size(), wavelengthFactors))
    {
        message = "GPU wavelength scaling kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t ell = 0; ell != spectralValues.size(); ++ell)
    {
        if (abs(spectralValues[ell] - expectedSpectralValues[ell]) > tolerance)
        {
            message = "wavelength scaling kernel mismatch";
            return false;
        }
    }

    size_t framePixels = 5;
    vector<double> frameValues(wavelengthFactors.size() * framePixels);
    for (size_t i = 0; i != frameValues.size(); ++i) frameValues[i] = 0.2 + 0.03 * static_cast<double>(i);
    vector<double> expectedFrameValues = frameValues;
    for (size_t ell = 0; ell != wavelengthFactors.size(); ++ell)
        for (size_t l = 0; l != framePixels; ++l) expectedFrameValues[l + ell * framePixels] *= wavelengthFactors[ell];
    if (!scaleFrameWavelengthValues(frameValues.data(), wavelengthFactors.size(), framePixels, wavelengthFactors))
    {
        message = "GPU IFU frame scaling kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t i = 0; i != frameValues.size(); ++i)
    {
        if (abs(frameValues[i] - expectedFrameValues[i]) > tolerance)
        {
            message = "IFU frame scaling kernel mismatch";
            return false;
        }
    }

    vector<double> dividedValues{3.0, -6.0, 9.0, 12.0, -1.5, 0.75};
    vector<double> expectedDividedValues = dividedValues;
    double divisor = 1.5;
    for (double& value : expectedDividedValues) value /= divisor;
    if (!divideValues(dividedValues.data(), dividedValues.size(), divisor))
    {
        message = "GPU scalar division kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t i = 0; i != dividedValues.size(); ++i)
    {
        if (abs(dividedValues[i] - expectedDividedValues[i]) > tolerance)
        {
            message = "scalar division kernel mismatch";
            return false;
        }
    }

    vector<double> multipliedValues{3.0, -6.0, 9.0, 12.0, -1.5, 0.75};
    vector<double> expectedMultipliedValues = multipliedValues;
    double factor = -2.0;
    for (double& value : expectedMultipliedValues) value *= factor;
    if (!multiplyValues(multipliedValues.data(), multipliedValues.size(), factor))
    {
        message = "GPU scalar multiplication kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t i = 0; i != multipliedValues.size(); ++i)
    {
        if (abs(multipliedValues[i] - expectedMultipliedValues[i]) > tolerance)
        {
            message = "scalar multiplication kernel mismatch";
            return false;
        }
    }

    vector<double> sumA{1.0, -2.0, 3.0, 4.5, 0.25, -0.75};
    vector<double> sumB{0.5, 1.5, -4.0, 2.0, -0.25, 3.0};
    vector<double> sumC{2.0, -0.25, 0.75, 1.0, 5.0, -3.5};
    vector<double> sumD{-1.0, 0.75, 2.25, -6.0, 1.5, 0.5};
    vector<double> gpuSum(sumA.size(), 0.);
    vector<double> expectedSum(sumA.size(), 0.);
    for (size_t i = 0; i != expectedSum.size(); ++i) expectedSum[i] = sumA[i] + sumB[i] + sumC[i] + sumD[i];
    if (!sumValues(gpuSum.data(), gpuSum.size(), sumA.data(), sumB.data(), sumC.data(), sumD.data()))
    {
        message = "GPU output array sum kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t i = 0; i != gpuSum.size(); ++i)
    {
        if (abs(gpuSum[i] - expectedSum[i]) > tolerance)
        {
            message = "output array sum kernel mismatch";
            return false;
        }
    }

    vector<int> keySumKeys{3, 7, 3, -1, 9, 7, 3};
    vector<double> keySumValues{1.5, -2.0, 0.25, 100.0, 4.0, 5.5, -0.75};
    vector<int> compactKeySumKeys;
    vector<double> compactKeySumValues;
    if (!sumValuesByKey(keySumKeys, keySumValues, compactKeySumKeys, compactKeySumValues))
    {
        message = "GPU key-sum kernel failed: " + lastRuntimeError();
        return false;
    }
    vector<double> expectedKeySums(10, 0.);
    for (size_t i = 0; i != keySumKeys.size(); ++i)
        if (keySumKeys[i] >= 0) expectedKeySums[keySumKeys[i]] += keySumValues[i];
    vector<double> actualKeySums(expectedKeySums.size(), 0.);
    if (compactKeySumKeys.size() != compactKeySumValues.size())
    {
        message = "key-sum output size mismatch";
        return false;
    }
    for (size_t i = 0; i != compactKeySumKeys.size(); ++i)
    {
        int key = compactKeySumKeys[i];
        if (key < 0 || static_cast<size_t>(key) >= actualKeySums.size())
        {
            message = "key-sum output key mismatch";
            return false;
        }
        actualKeySums[key] += compactKeySumValues[i];
    }
    for (size_t i = 0; i != expectedKeySums.size(); ++i)
    {
        if (abs(actualKeySums[i] - expectedKeySums[i]) > tolerance)
        {
            message = "key-sum kernel mismatch";
            return false;
        }
    }

    int accumulatorKeyObject = 0;
    const void* accumulatorKey = &accumulatorKeyObject;
    vector<int> accumulatorKeys{0, 2, 2, -1, 5, 9};
    vector<double> accumulatorValues{1.0, 2.0, 3.5, 100.0, -4.0, 8.0};
    vector<double> accumulatorTable(6, 0.);
    clearAccumulatedValues(accumulatorKey);
    if (!accumulateValuesByKey(accumulatorKey, accumulatorTable.size(), accumulatorKeys, accumulatorValues))
    {
        message = "GPU key accumulator kernel failed: " + lastRuntimeError();
        return false;
    }
    if (!retrieveAndClearAccumulatedValues(accumulatorKey, accumulatorTable.data(), accumulatorTable.size()))
    {
        message = "GPU key accumulator retrieval failed: " + lastRuntimeError();
        return false;
    }
    vector<double> expectedAccumulator{1.0, 0.0, 5.5, 0.0, 0.0, -4.0};
    for (size_t i = 0; i != expectedAccumulator.size(); ++i)
    {
        if (abs(accumulatorTable[i] - expectedAccumulator[i]) > tolerance)
        {
            message = "key accumulator kernel mismatch";
            return false;
        }
    }

    {
        double costheta = 0.8;
        double sintheta = 0.6;
        double cosphi = 0.6;
        double sinphi = 0.8;
        double cosomega = 0.5;
        double sinomega = sqrt(0.75);
        int numPixelsX = 3;
        int numPixelsY = 2;
        double xpmin = -1.5;
        double xpsiz = 1.0;
        double ypmin = -1.2;
        double ypsiz = 1.1;
        double redshift = 0.1;
        size_t numPixelsInFrame = static_cast<size_t>(numPixelsX) * static_cast<size_t>(numPixelsY);
        vector<Position> framePositions{Position(0.2, 0.1, 0.4), Position(-0.7, 0.3, -0.2),
                                        Position(0.8, -0.5, 0.1), Position(3.0, 3.0, 0.0)};
        vector<double> frameWavelengths{1.0, 1.5, 1.15, 2.1};
        vector<double> frameLuminosities{2.0, 3.0, 4.0, 5.0};
        vector<double> frameTau{0.2, 0.0, 0.3, 0.1};
        vector<int> frameBandOffsetv{0, 3, 6, 9};
        vector<double> frameBandWavelengthv{0.90, 1.10, 1.30, 1.40, 1.65, 1.90, 2.00, 2.30, 2.60};
        vector<double> frameBandTransmissionv{0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0};
        vector<double> frameBandWidthv{0.5, 1.2, 0.4};
        vector<double> expectedFrameSums(frameBandWidthv.size() * numPixelsInFrame, 0.);

        auto frameTransmission = [&](size_t ell, double lambda) {
            int begin = frameBandOffsetv[ell];
            int end = frameBandOffsetv[ell + 1];
            if (!(lambda > frameBandWavelengthv[begin] && lambda < frameBandWavelengthv[end - 1])) return 0.;
            auto first = frameBandWavelengthv.begin() + begin;
            auto last = frameBandWavelengthv.begin() + end;
            int i = static_cast<int>(std::upper_bound(first, last, lambda) - frameBandWavelengthv.begin());
            if (i <= begin || i >= end) return 0.;
            double x0 = frameBandWavelengthv[i - 1];
            double x1 = frameBandWavelengthv[i];
            double y0 = frameBandTransmissionv[i - 1];
            double y1 = frameBandTransmissionv[i];
            double t = x1 != x0 ? (lambda - x0) / (x1 - x0) : 0.;
            return (y0 + t * (y1 - y0)) * frameBandWidthv[ell];
        };

        for (size_t p = 0; p != framePositions.size(); ++p)
        {
            double x, y, z;
            framePositions[p].cartesian(x, y, z);
            double xpp = -sinphi * x + cosphi * y;
            double ypp = -cosphi * costheta * x - sinphi * costheta * y + sintheta * z;
            double xp = cosomega * xpp - sinomega * ypp;
            double yp = sinomega * xpp + cosomega * ypp;
            int i = static_cast<int>(floor((xp - xpmin) / xpsiz));
            int j = static_cast<int>(floor((yp - ypmin) / ypsiz));
            if (i < 0 || i >= numPixelsX || j < 0 || j >= numPixelsY) continue;
            size_t pixel = static_cast<size_t>(i + numPixelsX * j);
            double lambda = frameWavelengths[p] * (1. + redshift);
            for (size_t ell = 0; ell != frameBandWidthv.size(); ++ell)
            {
                double transmission = frameTransmission(ell, lambda);
                if (transmission == 0.) continue;
                expectedFrameSums[pixel + ell * numPixelsInFrame] +=
                    frameLuminosities[p] * transmission * exp(-frameTau[p]);
            }
        }

        vector<int> compactFrameKeys;
        vector<double> compactFrameValues;
        if (!frameBandTotalFluxSums(framePositions, frameWavelengths, frameLuminosities, frameTau, true, costheta,
                                    sintheta, cosphi, sinphi, cosomega, sinomega, numPixelsX, numPixelsY,
                                    xpmin, xpsiz, ypmin, ypsiz, redshift, numPixelsInFrame, frameBandOffsetv,
                                    frameBandWavelengthv, frameBandTransmissionv, frameBandWidthv,
                                    compactFrameKeys, compactFrameValues))
        {
            message = "GPU frame band detector kernel failed: " + lastRuntimeError();
            return false;
        }
        if (compactFrameKeys.size() != compactFrameValues.size())
        {
            message = "frame band detector output size mismatch";
            return false;
        }
        vector<double> actualFrameSums(expectedFrameSums.size(), 0.);
        for (size_t i = 0; i != compactFrameKeys.size(); ++i)
        {
            int key = compactFrameKeys[i];
            if (key < 0 || static_cast<size_t>(key) >= actualFrameSums.size())
            {
                message = "frame band detector output key mismatch";
                return false;
            }
            actualFrameSums[key] += compactFrameValues[i];
        }
        for (size_t i = 0; i != expectedFrameSums.size(); ++i)
        {
            if (abs(actualFrameSums[i] - expectedFrameSums[i]) > 1e-11)
            {
                message = "frame band detector kernel mismatch";
                return false;
            }
        }

        int frameAccumulatorKeyObject = 0;
        const void* frameAccumulatorKey = &frameAccumulatorKeyObject;
        vector<double> accumulatedFrameSums(expectedFrameSums.size(), 0.);
        clearAccumulatedValues(frameAccumulatorKey);
        if (!frameBandTotalFluxAccumulate(frameAccumulatorKey, accumulatedFrameSums.size(), framePositions,
                                          frameWavelengths, frameLuminosities, frameTau, true, costheta,
                                          sintheta, cosphi, sinphi, cosomega, sinomega, numPixelsX, numPixelsY,
                                          xpmin, xpsiz, ypmin, ypsiz, redshift, numPixelsInFrame,
                                          frameBandOffsetv, frameBandWavelengthv, frameBandTransmissionv,
                                          frameBandWidthv))
        {
            message = "GPU frame band detector accumulator failed: " + lastRuntimeError();
            return false;
        }
        if (!retrieveAndClearAccumulatedValues(frameAccumulatorKey, accumulatedFrameSums.data(),
                                               accumulatedFrameSums.size()))
        {
            message = "GPU frame band detector accumulator retrieval failed: " + lastRuntimeError();
            return false;
        }
        for (size_t i = 0; i != expectedFrameSums.size(); ++i)
        {
            if (abs(accumulatedFrameSums[i] - expectedFrameSums[i]) > 1e-11)
            {
                message = "frame band detector accumulator mismatch";
                return false;
            }
        }
    }

    {
        auto hgValue = [](double g, double costheta) {
            double t = 1. + g * g - 2. * g * costheta;
            return (1. - g) * (1. + g) / sqrt(t * t * t);
        };
        auto hgIntegral = [](double g, double cosalpha, double cosbeta) {
            double ta = sqrt(1. + g * g - 2. * g * cosalpha);
            double tb = sqrt(1. + g * g - 2. * g * cosbeta);
            double f1 = (1. - g) * (1. + g) / g;
            double f2 = (tb - ta) / (tb * ta);
            return f1 * f2;
        };
        auto hgMean = [&](double g, double costheta) {
            constexpr double delta = 4. * M_PI / 180.;
            double theta = acos(costheta);
            double cosalpha = cos(theta - delta);
            double cosbeta = cos(theta + delta);
            if (theta < delta)
                return (hgIntegral(g, 1., cosalpha) + hgIntegral(g, 1., cosbeta))
                       / (2. - cosalpha - cosbeta);
            if (theta > M_PI - delta)
                return (hgIntegral(g, cosalpha, -1.) + hgIntegral(g, cosbeta, -1.))
                       / (2. + cosalpha + cosbeta);
            return hgIntegral(g, cosalpha, cosbeta) / (cosalpha - cosbeta);
        };

        vector<double> inputDirectionv{0., 0., 1.,
                                       sqrt(0.75), 0., 0.5,
                                       sqrt(1. - 0.25 * 0.25), 0., -0.25};
        vector<double> packetLuminosityv{2.0, 3.0, 4.0};
        vector<double> hgLambdav{0.2, 0.7, 1.3};
        vector<double> hgLookupWavelengthv{0.1, 0.5, 1.0, 2.0};
        vector<double> hgAsymmparv{0.0, 0.5, 0.96, -0.2};
        vector<double> gpuHgLuminosityv;
        if (!henyeyGreensteinScatteringLuminosities(inputDirectionv, packetLuminosityv, hgLambdav,
                                                    Direction(0., 0., 1., true), 0, hgLookupWavelengthv.size(),
                                                    hgLookupWavelengthv, hgAsymmparv, gpuHgLuminosityv))
        {
            message = "GPU HG scattering luminosity kernel failed: " + lastRuntimeError();
            return false;
        }
        vector<double> expectedHgLuminosityv{packetLuminosityv[0] * hgValue(0.0, 1.0),
                                             packetLuminosityv[1] * hgValue(0.5, 0.5),
                                             packetLuminosityv[2] * hgMean(0.96, -0.25)};
        if (gpuHgLuminosityv.size() != expectedHgLuminosityv.size())
        {
            message = "HG scattering luminosity output size mismatch";
            return false;
        }
	        for (size_t i = 0; i != expectedHgLuminosityv.size(); ++i)
	        {
	            if (abs(gpuHgLuminosityv[i] - expectedHgLuminosityv[i]) > tolerance)
	            {
	                message = "HG scattering luminosity kernel mismatch";
	                return false;
	            }
	        }

	        auto hgLookupIndex = [&](double lambda) {
	            if (lambda < hgLookupWavelengthv.front()) return 0;
	            int jl = -1;
	            int ju = static_cast<int>(hgLookupWavelengthv.size()) - 1;
	            while (ju - jl > 1)
	            {
	                int jm = (ju + jl) >> 1;
	                if (lambda < hgLookupWavelengthv[jm])
	                    ju = jm;
	                else
	                    jl = jm;
	            }
	            return jl < 0 ? 0 : jl;
	        };
	        auto hgDirection = [&](size_t i, double u, double v) {
	            constexpr double pi = M_PI;
	            double g = hgAsymmparv[hgLookupIndex(hgLambdav[i])];
	            double phi = 2. * pi * v;
	            double cosphi = cos(phi);
	            double sinphi = sin(phi);
	            double costheta = 0.;
	            if (abs(g) < 1e-6)
	            {
	                costheta = 2. * u - 1.;
	                double sintheta = sqrt(abs((1. - costheta) * (1. + costheta)));
	                return vector<double>{cosphi * sintheta, sinphi * sintheta, costheta};
	            }
	            double f = ((1. - g) * (1. + g)) / (1. - g + 2. * g * u);
	            costheta = (1. + g * g - f * f) / (2. * g);
	            double sintheta = sqrt(abs((1. - costheta) * (1. + costheta)));
	            double kx = inputDirectionv[3 * i];
	            double ky = inputDirectionv[3 * i + 1];
	            double kz = inputDirectionv[3 * i + 2];
	            if (kz > 0.99999) return vector<double>{cosphi * sintheta, sinphi * sintheta, costheta};
	            if (kz < -0.99999) return vector<double>{cosphi * sintheta, sinphi * sintheta, -costheta};
	            double root = sqrt((1. - kz) * (1. + kz));
	            return vector<double>{sintheta / root * (-kx * kz * cosphi + ky * sinphi) + kx * costheta,
	                                  -sintheta / root * (ky * kz * cosphi + kx * sinphi) + ky * costheta,
	                                  root * sintheta * cosphi + kz * costheta};
	        };
	        vector<double> hgRandomCosthetav{0.25, 0.70, 0.40};
	        vector<double> hgRandomPhiv{0.10, 0.30, 0.80};
	        vector<double> gpuHgDirectionv;
	        if (!henyeyGreensteinScatteringDirections(inputDirectionv, hgLambdav, hgRandomCosthetav, hgRandomPhiv, 0,
	                                                  hgLookupWavelengthv.size(), hgLookupWavelengthv,
	                                                  hgAsymmparv, gpuHgDirectionv))
	        {
	            message = "GPU HG scattering direction kernel failed: " + lastRuntimeError();
	            return false;
	        }
	        if (gpuHgDirectionv.size() != inputDirectionv.size())
	        {
	            message = "HG scattering direction output size mismatch";
	            return false;
	        }
	        for (size_t i = 0; i != hgLambdav.size(); ++i)
	        {
	            vector<double> expected = hgDirection(i, hgRandomCosthetav[i], hgRandomPhiv[i]);
	            for (size_t c = 0; c != 3; ++c)
	            {
	                if (abs(gpuHgDirectionv[3 * i + c] - expected[c]) > tolerance)
	                {
	                    message = "HG scattering direction kernel mismatch";
	                    return false;
	                }
	            }
	        }

	        vector<double> isoRandomCosthetav{0.0, 0.25, 0.70, 1.0};
	        vector<double> isoRandomPhiv{0.10, 0.30, 0.80, 0.45};
	        vector<double> gpuIsoDirectionv;
	        if (!isotropicDirections(isoRandomCosthetav, isoRandomPhiv, gpuIsoDirectionv))
	        {
	            message = "GPU isotropic direction kernel failed: " + lastRuntimeError();
	            return false;
	        }
	        if (gpuIsoDirectionv.size() != 3 * isoRandomCosthetav.size())
	        {
	            message = "isotropic direction output size mismatch";
	            return false;
	        }
	        for (size_t i = 0; i != isoRandomCosthetav.size(); ++i)
	        {
	            double costheta = 2. * isoRandomCosthetav[i] - 1.;
	            double sintheta = sqrt(abs((1. - costheta) * (1. + costheta)));
	            double phi = 2. * M_PI * isoRandomPhiv[i];
	            vector<double> expected{sintheta * cos(phi), sintheta * sin(phi), costheta};
	            for (size_t c = 0; c != 3; ++c)
	            {
	                if (abs(gpuIsoDirectionv[3 * i + c] - expected[c]) > tolerance)
	                {
	                    message = "isotropic direction kernel mismatch";
	                    return false;
	                }
	            }
	        }
	    }

	    vector<double> launchLuminosities{0.0, 0.20, 0.0, 0.15, 0.40, 0.0, 0.25};
    vector<double> gpuLaunchWeights(launchLuminosities.size(), 0.);
    vector<double> expectedLaunchWeights(launchLuminosities.size(), 0.);
    double spatialBias = 0.35;
    int emittingCount = 0;
    for (double value : launchLuminosities)
        if (value > 0.) ++emittingCount;
    for (size_t i = 0; i != launchLuminosities.size(); ++i)
    {
        double uniformWeight = launchLuminosities[i] > 0. ? 1. / emittingCount : 0.;
        expectedLaunchWeights[i] = (1. - spatialBias) * launchLuminosities[i] + spatialBias * uniformWeight;
    }
    if (!compositeLaunchWeights(launchLuminosities.data(), launchLuminosities.size(), spatialBias,
                                gpuLaunchWeights.data()))
    {
        message = "GPU composite launch weight kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t i = 0; i != gpuLaunchWeights.size(); ++i)
    {
        if (abs(gpuLaunchWeights[i] - expectedLaunchWeights[i]) > tolerance)
        {
            message = "composite launch weight kernel mismatch";
            return false;
        }
    }

    vector<double> axis0{1.0, 2.0, 4.0};
    vector<double> axis1{10.0, 20.0};
    vector<const double*> tableAxes{axis0.data(), axis1.data()};
    vector<size_t> tableAxisSizes{axis0.size(), axis1.size()};
    vector<bool> tableAxisLog{false, false};
    vector<double> quantity{10.0, 20.0, 40.0, 20.0, 40.0, 80.0};
    vector<double> tableParameters{10.0, 15.0, 20.0};
    vector<double> tableScales{1.0, 2.0, 3.0};
    vector<double> tableLuminosities;
    vector<double> expectedTableLuminosities{75.0, 225.0, 450.0};
    if (!storedTableCdf(tableLuminosities, 2, tableAxes, tableAxisSizes, tableAxisLog, quantity.data(), 1, false,
                        true, 1.0, 4.0, tableParameters, tableScales, tableScales.size()))
    {
        message = "GPU stored-table CDF batch kernel failed: " + lastRuntimeError();
        return false;
    }
    for (size_t i = 0; i != tableLuminosities.size(); ++i)
    {
        if (abs(tableLuminosities[i] - expectedTableLuminosities[i]) > tolerance)
        {
            message = "stored-table CDF batch kernel mismatch";
            return false;
        }
    }

    vector<double> sampleRandoms{0.2, 0.75, 0.5};
    vector<double> forcedWavelengths{0.0, 3.0, 0.0};
    vector<double> sampledWavelengths;
    vector<double> sampledSpecificLuminosities;
    if (!storedTableSampleWavelengths(sampledWavelengths, sampledSpecificLuminosities, 2, tableAxes, tableAxisSizes,
                                      tableAxisLog, quantity.data(), 1, false, true, 1.0, 4.0, tableParameters,
                                      sampleRandoms, forcedWavelengths, tableParameters.size()))
    {
        message = "GPU stored-table wavelength sampling kernel failed: " + lastRuntimeError();
        return false;
    }
    vector<double> expectedSampledWavelengths{2.0, 3.0, sqrt(8.5)};
    vector<double> expectedSampledSpecificLuminosities{2.0 / 7.5, 3.0 / 7.5, sqrt(8.5) / 7.5};
    for (size_t i = 0; i != sampledWavelengths.size(); ++i)
    {
        if (abs(sampledWavelengths[i] - expectedSampledWavelengths[i]) > tolerance
            || abs(sampledSpecificLuminosities[i] - expectedSampledSpecificLuminosities[i]) > tolerance)
        {
            message = "stored-table wavelength sampling kernel mismatch";
            return false;
        }
    }

    message =
        "GPU Cartesian/tree/adaptive-mesh/Voronoi/Voronoi-compact/block-list/fused-table-batch/tetrahedral/cylindrical-2D/3D/spherical-1D/2D/3D grid path, stored-table CDF/sampling batch, constant/table/observer-total optical-depth, radiation-field sums, key-sum reduction, key accumulator, frame-band detector/accumulator, HG scatter-peel luminosity/direction, fused HG observer luminosity/frame accumulator, cumulative-path, albedo, forced-propagation, table-albedo forced-propagation, and combined radiation-field/forced-propagation batches, dust-section table, secondary-launch, flux-output, and photon-cycle kernel self-test passed";
    return true;
}

////////////////////////////////////////////////////////////////////

void GpuAcceleration::invalidateMediumState(const MediumState& state)
{
#if defined(__unix__) || defined(__APPLE__)
    invalidateRuntimeMediumStates(state);
#else
    (void)state;
#endif
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setCartesianSpatialGridPath(const CartesianSpatialGrid& grid, SpatialGridPath* path,
                                                  double maxDistance)
{
    if (!isProcessEnabled() || !path) return false;
    const Array& xv = grid.xBorderPositions();
    const Array& yv = grid.yBorderPositions();
    const Array& zv = grid.zBorderPositions();
    int nx = grid.numBinsX();
    int ny = grid.numBinsY();
    int nz = grid.numBinsZ();
    if (xv.size() != static_cast<size_t>(nx + 1) || yv.size() != static_cast<size_t>(ny + 1)
        || zv.size() != static_cast<size_t>(nz + 1))
        return false;

    return computeCartesianPath(std::begin(xv), std::begin(yv), std::begin(zv), nx, ny, nz, path->position(),
                                path->direction(), grid.xmin(), grid.ymin(), grid.zmin(), grid.xmax(), grid.ymax(),
                                grid.zmax(), maxDistance, path);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setTreeSpatialGridPath(const TreeSpatialGrid& grid, SpatialGridPath* path, double maxDistance)
{
    if (!isProcessEnabled() || !path) return false;
    return computeTreePath(&grid, grid.traversalNodeBounds(), grid.traversalChildBegin(), grid.traversalChildCount(),
                           grid.traversalChildIndex(), grid.traversalCellIndex(), path->position(), path->direction(),
                           grid.xmin(), grid.ymin(), grid.zmin(), grid.xmax(), grid.ymax(), grid.zmax(), maxDistance,
                           path);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setAdaptiveMeshSpatialGridPath(const AdaptiveMeshSpatialGrid& grid, SpatialGridPath* path,
                                                     double maxDistance)
{
    if (!isProcessEnabled() || !path) return false;
    Box box = grid.boundingBox();
    return computeTreePath(&grid, grid.traversalNodeBounds(), grid.traversalChildBegin(), grid.traversalChildCount(),
                           grid.traversalChildIndex(), grid.traversalCellIndex(), path->position(), path->direction(),
                           box.xmin(), box.ymin(), box.zmin(), box.xmax(), box.ymax(), box.zmax(), maxDistance, path);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setSphere1DSpatialGridPath(const Sphere1DSpatialGrid& grid, SpatialGridPath* path,
                                                 double maxDistance)
{
    if (!isProcessEnabled() || !path) return false;
    const Array& rv = grid.radialBorderPositions();
    int nr = grid.numRadialBins();
    if (rv.size() != static_cast<size_t>(nr + 1)) return false;
    return computeSphere1DPath(std::begin(rv), nr, path->position(), path->direction(), maxDistance, path);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setSphere2DSpatialGridPath(const Sphere2DSpatialGrid& grid, SpatialGridPath* path,
                                                 double maxDistance)
{
    if (!isProcessEnabled() || !path) return false;
    const Array& rv = grid.radialBorderPositions();
    const Array& thetav = grid.polarBorderAngles();
    const Array& cv = grid.polarBorderCosines();
    int nr = grid.numRadialBins();
    int ntheta = grid.numPolarBins();
    if (rv.size() != static_cast<size_t>(nr + 1) || thetav.size() != static_cast<size_t>(ntheta + 1)
        || cv.size() != static_cast<size_t>(ntheta + 1))
        return false;
    return computeSphere2DPath(std::begin(rv), std::begin(thetav), std::begin(cv), nr, ntheta, path->position(),
                               path->direction(), maxDistance, path);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setSphere3DSpatialGridPath(const Sphere3DSpatialGrid& grid, SpatialGridPath* path,
                                                 double maxDistance)
{
    if (!isProcessEnabled() || !path) return false;
    const Array& rv = grid.radialBorderPositions();
    const Array& thetav = grid.polarBorderAngles();
    const Array& phiv = grid.azimuthalBorderAngles();
    const Array& cv = grid.polarBorderCosines();
    const Array& sinv = grid.azimuthalBorderSines();
    const Array& cosv = grid.azimuthalBorderCosines();
    int nr = grid.numRadialBins();
    int ntheta = grid.numPolarBins();
    int nphi = grid.numAzimuthalBins();
    if (rv.size() != static_cast<size_t>(nr + 1) || thetav.size() != static_cast<size_t>(ntheta + 1)
        || phiv.size() != static_cast<size_t>(nphi + 1) || cv.size() != static_cast<size_t>(ntheta + 1)
        || sinv.size() != static_cast<size_t>(nphi + 1) || cosv.size() != static_cast<size_t>(nphi + 1))
        return false;
    return computeSphere3DPath(std::begin(rv), std::begin(thetav), std::begin(phiv), std::begin(cv), std::begin(sinv),
                               std::begin(cosv), nr, ntheta, nphi, grid.traversalEpsilon(), path->position(),
                               path->direction(), maxDistance, path);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setCylinder2DSpatialGridPath(const Cylinder2DSpatialGrid& grid, SpatialGridPath* path,
                                                   double maxDistance)
{
    if (!isProcessEnabled() || !path) return false;
    const Array& Rv = grid.radialBorderPositions();
    const Array& zv = grid.zBorderPositions();
    int nR = grid.numRadialBins();
    int nz = grid.numZBins();
    if (Rv.size() != static_cast<size_t>(nR + 1) || zv.size() != static_cast<size_t>(nz + 1)) return false;
    return computeCylinder2DPath(std::begin(Rv), std::begin(zv), nR, nz, path->position(), path->direction(),
                                 maxDistance, path);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setCylinder3DSpatialGridPath(const Cylinder3DSpatialGrid& grid, SpatialGridPath* path,
                                                   double maxDistance)
{
    if (!isProcessEnabled() || !path) return false;
    const Array& Rv = grid.radialBorderPositions();
    const Array& phiv = grid.azimuthalBorderAngles();
    const Array& zv = grid.zBorderPositions();
    const Array& sinv = grid.azimuthalBorderSines();
    const Array& cosv = grid.azimuthalBorderCosines();
    int nR = grid.numRadialBins();
    int nphi = grid.numAzimuthalBins();
    int nz = grid.numZBins();
    if (Rv.size() != static_cast<size_t>(nR + 1) || phiv.size() != static_cast<size_t>(nphi + 1)
        || zv.size() != static_cast<size_t>(nz + 1) || sinv.size() != static_cast<size_t>(nphi + 1)
        || cosv.size() != static_cast<size_t>(nphi + 1))
        return false;
    return computeCylinder3DPath(std::begin(Rv), std::begin(phiv), std::begin(zv), std::begin(sinv), std::begin(cosv),
                                 nR, nphi, nz, grid.traversalEpsilon(), grid.hasCentralHole(), path->position(),
                                 path->direction(), maxDistance, path);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setTetraMeshSpatialGridPath(const TetraMeshSpatialGrid& grid, SpatialGridPath* path,
                                                  double maxDistance)
{
    if (!isProcessEnabled() || !path) return false;
    int numCells = grid.numCells();
    if (numCells <= 0) return false;
    return computeTetraPath(&grid, grid.traversalVertexCoordinates(), grid.traversalTetraVertexIndices(),
                            grid.traversalFaceAnchorIndices(), grid.traversalFaceNeighborIndices(),
                            grid.traversalFaceNormals(), grid.traversalCentroids(), numCells, grid.traversalEpsilon(),
                            path->position(), path->direction(), grid.xmin(), grid.ymin(), grid.zmin(), grid.xmax(),
                            grid.ymax(), grid.zmax(), maxDistance, path);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setVoronoiMeshSpatialGridPath(const VoronoiMeshSpatialGrid& grid, SpatialGridPath* path,
                                                    double maxDistance)
{
    if (!isProcessEnabled() || !path) return false;
    int numCells = grid.numCells();
    if (numCells <= 0) return false;
    const vector<double>& sitev = grid.traversalSiteCoordinates();
    const vector<int>& neighborBeginv = grid.traversalNeighborBegin();
    const vector<int>& neighborCountv = grid.traversalNeighborCount();
    const vector<int>& neighborIndexv = grid.traversalNeighborIndex();
    if (sitev.size() != 3 * static_cast<size_t>(numCells)
        || neighborBeginv.size() != static_cast<size_t>(numCells)
        || neighborCountv.size() != static_cast<size_t>(numCells))
        return false;
    for (int m = 0; m != numCells; ++m)
    {
        int begin = neighborBeginv[m];
        int count = neighborCountv[m];
        if (begin < 0 || count < 0) return false;
        if (static_cast<size_t>(begin) + static_cast<size_t>(count) > neighborIndexv.size()) return false;
        for (int i = 0; i != count; ++i)
        {
            int neighbor = neighborIndexv[begin + i];
            if (neighbor >= numCells || neighbor < -6) return false;
        }
    }
    return computeVoronoiPath(&grid, sitev, neighborBeginv, neighborCountv, neighborIndexv, numCells,
                              grid.traversalEpsilon(), path->position(), path->direction(), grid.xmin(), grid.ymin(),
                              grid.zmin(), grid.xmax(), grid.ymax(), grid.zmax(), maxDistance, path);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setVoronoiMeshSpatialGridPaths(const VoronoiMeshSpatialGrid& grid,
                                                     const vector<SpatialGridPath*>& paths, double maxDistance)
{
    if (!isProcessEnabled() || paths.empty()) return false;
    for (auto path : paths)
        if (!path) return false;
    int numCells = grid.numCells();
    if (numCells <= 0) return false;
    const vector<double>& sitev = grid.traversalSiteCoordinates();
    const vector<int>& neighborBeginv = grid.traversalNeighborBegin();
    const vector<int>& neighborCountv = grid.traversalNeighborCount();
    const vector<int>& neighborIndexv = grid.traversalNeighborIndex();
    const vector<int>& blockBeginv = grid.traversalBlockBegin();
    const vector<int>& blockCountv = grid.traversalBlockCount();
    const vector<int>& blockIndexv = grid.traversalBlockIndex();
    int blockN = grid.traversalBlockGridN();
    if (sitev.size() != 3 * static_cast<size_t>(numCells)
        || neighborBeginv.size() != static_cast<size_t>(numCells)
        || neighborCountv.size() != static_cast<size_t>(numCells))
        return false;
    if (blockN > 0
        && (blockBeginv.size() != static_cast<size_t>(blockN) * blockN * blockN
            || blockCountv.size() != blockBeginv.size()))
        return false;
    for (int m = 0; m != numCells; ++m)
    {
        int begin = neighborBeginv[m];
        int count = neighborCountv[m];
        if (begin < 0 || count < 0) return false;
        if (static_cast<size_t>(begin) + static_cast<size_t>(count) > neighborIndexv.size()) return false;
        for (int i = 0; i != count; ++i)
        {
            int neighbor = neighborIndexv[begin + i];
            if (neighbor >= numCells || neighbor < -6) return false;
        }
    }
    for (size_t b = 0; b != blockBeginv.size(); ++b)
    {
        int begin = blockBeginv[b];
        int count = blockCountv[b];
        if (begin < 0 || count < 0) return false;
        if (static_cast<size_t>(begin) + static_cast<size_t>(count) > blockIndexv.size()) return false;
        for (int i = 0; i != count; ++i)
        {
            int cell = blockIndexv[begin + i];
            if (cell < 0 || cell >= numCells) return false;
        }
    }
    return computeVoronoiPaths(&grid, sitev, neighborBeginv, neighborCountv, neighborIndexv,
                               blockBeginv, blockCountv, blockIndexv, blockN, numCells, grid.traversalEpsilon(),
                               grid.xmin(), grid.ymin(), grid.zmin(), grid.xmax(), grid.ymax(), grid.zmax(),
                               maxDistance, paths);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setVoronoiMeshSpatialGridExtinctionOpticalDepthsFromTables(
    const VoronoiMeshSpatialGrid& grid, const vector<SpatialGridPath*>& paths, const MediumState& state,
    const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
    const vector<double>& lookupWavelengthv, const vector<double>& sectionExtv, const vector<double>& lambdav)
{
    if (!isProcessEnabled() || paths.empty()) return false;
    int numCells = grid.numCells();
    if (numCells <= 0) return false;
    return computeVoronoiTableOpticalDepthPaths(
        &grid, grid.traversalSiteCoordinates(), grid.traversalNeighborBegin(), grid.traversalNeighborCount(),
        grid.traversalNeighborIndex(), grid.traversalBlockBegin(), grid.traversalBlockCount(),
        grid.traversalBlockIndex(), grid.traversalBlockGridN(), numCells, grid.traversalEpsilon(), grid.xmin(),
        grid.ymin(), grid.zmin(), grid.xmax(), grid.ymax(), grid.zmax(), std::numeric_limits<double>::infinity(),
        paths, state, mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv, sectionExtv, nullptr, lambdav);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::getVoronoiMeshSpatialGridExtinctionOpticalDepthsFromTables(
    const VoronoiMeshSpatialGrid& grid, const vector<const SpatialGridPath*>& paths, const MediumState& state,
    const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
    const vector<double>& lookupWavelengthv, const vector<double>& sectionExtv, const vector<double>& lambdav,
    const vector<double>& maxDistancev, vector<double>& tauv)
{
    if (!isProcessEnabled() || paths.empty()) return false;
    int numCells = grid.numCells();
    if (numCells <= 0) return false;
    return computeVoronoiTableOpticalDepthTotals(
        &grid, grid.traversalSiteCoordinates(), grid.traversalNeighborBegin(), grid.traversalNeighborCount(),
        grid.traversalNeighborIndex(), grid.traversalBlockBegin(), grid.traversalBlockCount(),
        grid.traversalBlockIndex(), grid.traversalBlockGridN(), numCells, grid.traversalEpsilon(), grid.xmin(),
        grid.ymin(), grid.zmin(), grid.xmax(), grid.ymax(), grid.zmax(), paths, state, mediaIndexv, lookupBeginv,
        lookupCountv, lookupWavelengthv, sectionExtv, lambdav, maxDistancev, tauv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::getVoronoiMeshSpatialGridExtinctionOpticalDepthsFromTables(
    const VoronoiMeshSpatialGrid& grid, const vector<Position>& positions, const vector<Direction>& directions,
    const MediumState& state, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
    const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
    const vector<double>& sectionExtv, const vector<double>& lambdav,
    const vector<double>& maxDistancev, vector<double>& tauv)
{
    if (!isProcessEnabled() || positions.empty()) return false;
    int numCells = grid.numCells();
    if (numCells <= 0) return false;
    return computeVoronoiTableOpticalDepthTotals(
        &grid, grid.traversalSiteCoordinates(), grid.traversalNeighborBegin(), grid.traversalNeighborCount(),
        grid.traversalNeighborIndex(), grid.traversalBlockBegin(), grid.traversalBlockCount(),
        grid.traversalBlockIndex(), grid.traversalBlockGridN(), numCells, grid.traversalEpsilon(), grid.xmin(),
        grid.ymin(), grid.zmin(), grid.xmax(), grid.ymax(), grid.zmax(), positions, directions, state,
        mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv, sectionExtv, lambdav, maxDistancev, tauv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::getVoronoiMeshSpatialGridExtinctionOpticalDepthsFromTables(
    const VoronoiMeshSpatialGrid& grid, const vector<Position>& positions, const Direction& direction,
    const MediumState& state, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
    const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
    const vector<double>& sectionExtv, const vector<double>& lambdav,
    const vector<double>& maxDistancev, vector<double>& tauv)
{
    if (!isProcessEnabled() || positions.empty()) return false;
    int numCells = grid.numCells();
    if (numCells <= 0) return false;
    return computeVoronoiTableOpticalDepthTotals(
        &grid, grid.traversalSiteCoordinates(), grid.traversalNeighborBegin(), grid.traversalNeighborCount(),
        grid.traversalNeighborIndex(), grid.traversalBlockBegin(), grid.traversalBlockCount(),
        grid.traversalBlockIndex(), grid.traversalBlockGridN(), numCells, grid.traversalEpsilon(), grid.xmin(),
        grid.ymin(), grid.zmin(), grid.xmax(), grid.ymax(), grid.zmax(), positions, direction, state, mediaIndexv,
        lookupBeginv, lookupCountv, lookupWavelengthv, sectionExtv, lambdav, maxDistancev, tauv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::getVoronoiMeshSpatialGridHenyeyGreensteinScatteringObservedLuminositiesFromTables(
    const VoronoiMeshSpatialGrid& grid, const vector<const SpatialGridPath*>& paths, const MediumState& state,
    const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
    const vector<double>& lookupWavelengthv, const vector<double>& sectionExtv, const vector<double>& lambdav,
    const vector<double>& maxDistancev, const vector<double>& inputDirectionv,
    const vector<double>& packetLuminosityv, Direction bfkobs, int hgLookupBegin, int hgLookupCount,
    const vector<double>& asymmparv, vector<double>& luminosityv)
{
    if (!isProcessEnabled() || paths.empty()) return false;
    int numCells = grid.numCells();
    if (numCells <= 0) return false;
    return computeVoronoiTableHenyeyGreensteinObservedLuminosities(
        &grid, grid.traversalSiteCoordinates(), grid.traversalNeighborBegin(), grid.traversalNeighborCount(),
        grid.traversalNeighborIndex(), grid.traversalBlockBegin(), grid.traversalBlockCount(),
        grid.traversalBlockIndex(), grid.traversalBlockGridN(), numCells, grid.traversalEpsilon(), grid.xmin(),
        grid.ymin(), grid.zmin(), grid.xmax(), grid.ymax(), grid.zmax(), paths, state, mediaIndexv, lookupBeginv,
	        lookupCountv, lookupWavelengthv, sectionExtv, lambdav, maxDistancev, inputDirectionv, packetLuminosityv,
	        bfkobs, hgLookupBegin, hgLookupCount, asymmparv, luminosityv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::getVoronoiMeshSpatialGridHenyeyGreensteinScatteringFrameBandAccumulateFromTables(
    const VoronoiMeshSpatialGrid& grid, const vector<Position>& positionv, const MediumState& state,
    const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
    const vector<double>& lookupWavelengthv, const vector<double>& sectionExtv, const vector<double>& lambdav,
    const vector<double>& maxDistancev, const vector<double>& inputDirectionv,
    const vector<double>& packetLuminosityv, Direction bfkobs, int hgLookupBegin, int hgLookupCount,
    const vector<double>& asymmparv, const void* accumulatorKey, size_t numAccumulatorValues,
    double costheta, double sintheta, double cosphi, double sinphi, double cosomega, double sinomega,
    int numPixelsX, int numPixelsY, double xpmin, double xpsiz, double ypmin, double ypsiz,
    double redshift, size_t numPixelsInFrame, const vector<int>& bandOffsetv,
    const vector<double>& bandWavelengthv, const vector<double>& bandTransmissionv,
    const vector<double>& bandWidthv)
{
    if (!isProcessEnabled() || positionv.empty()) return false;
    int numCells = grid.numCells();
    if (numCells <= 0) return false;
    return computeVoronoiTableHenyeyGreensteinFrameBandAccumulate(
        &grid, grid.traversalSiteCoordinates(), grid.traversalNeighborBegin(), grid.traversalNeighborCount(),
        grid.traversalNeighborIndex(), grid.traversalBlockBegin(), grid.traversalBlockCount(),
        grid.traversalBlockIndex(), grid.traversalBlockGridN(), numCells, grid.traversalEpsilon(), grid.xmin(),
        grid.ymin(), grid.zmin(), grid.xmax(), grid.ymax(), grid.zmax(), positionv, state, mediaIndexv,
        lookupBeginv, lookupCountv, lookupWavelengthv, sectionExtv, lambdav, maxDistancev, inputDirectionv,
        packetLuminosityv, bfkobs, hgLookupBegin, hgLookupCount, asymmparv, accumulatorKey,
        numAccumulatorValues, costheta, sintheta, cosphi, sinphi, cosomega, sinomega, numPixelsX, numPixelsY,
        xpmin, xpsiz, ypmin, ypsiz, redshift, numPixelsInFrame, bandOffsetv, bandWavelengthv,
        bandTransmissionv, bandWidthv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setVoronoiMeshSpatialGridScatteringAndAbsorptionOpticalDepthsFromTables(
    const VoronoiMeshSpatialGrid& grid, const vector<SpatialGridPath*>& paths, const MediumState& state,
    const vector<int>& mediaIndexv, const vector<int>& lookupBeginv, const vector<int>& lookupCountv,
    const vector<double>& lookupWavelengthv, const vector<double>& sectionScav, const vector<double>& sectionAbsv,
    const vector<double>& lambdav)
{
    if (!isProcessEnabled() || paths.empty()) return false;
    int numCells = grid.numCells();
    if (numCells <= 0) return false;
    return computeVoronoiTableOpticalDepthPaths(
        &grid, grid.traversalSiteCoordinates(), grid.traversalNeighborBegin(), grid.traversalNeighborCount(),
        grid.traversalNeighborIndex(), grid.traversalBlockBegin(), grid.traversalBlockCount(),
        grid.traversalBlockIndex(), grid.traversalBlockGridN(), numCells, grid.traversalEpsilon(), grid.xmin(),
        grid.ymin(), grid.zmin(), grid.xmax(), grid.ymax(), grid.zmax(), std::numeric_limits<double>::infinity(),
        paths, state, mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv, sectionScav, &sectionAbsv, lambdav);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setExtinctionOpticalDepths(SpatialGridPath* path, const MediumState& state,
                                                 const vector<double>& sectionv)
{
    if (!isProcessEnabled()) return false;

    vector<double> cumulative;
    vector<double> unusedCumulative;
    if (computeCumulativeOpticalDepths(path, state, sectionv, nullptr, cumulative, unusedCumulative))
    {
        auto& segments = path->segments();
        for (size_t i = 0; i != segments.size(); ++i) segments[i].setOpticalDepth(cumulative[i]);
        return true;
    }

    vector<double> contributions;
    vector<double> unused;
    if (!computeContributions(path, state, sectionv, nullptr, contributions, unused)) return false;

    double tau = 0.;
    auto& segments = path->segments();
    for (size_t i = 0; i != segments.size(); ++i)
    {
        tau += contributions[i];
        segments[i].setOpticalDepth(tau);
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setExtinctionOpticalDepths(const vector<SpatialGridPath*>& paths, const MediumState& state,
                                                 const vector<double>& sectionv)
{
    if (!isProcessEnabled()) return false;

    vector<int> pathOffsetv;
    vector<double> cumulative;
    vector<double> unusedCumulative;
    if (!computeCumulativeOpticalDepthsBatch(paths, state, sectionv, nullptr, pathOffsetv, cumulative,
                                             unusedCumulative))
        return false;

    for (size_t pathIndex = 0; pathIndex != paths.size(); ++pathIndex)
    {
        auto& segments = paths[pathIndex]->segments();
        size_t begin = static_cast<size_t>(pathOffsetv[pathIndex]);
        for (size_t i = 0; i != segments.size(); ++i) segments[i].setOpticalDepth(cumulative[begin + i]);
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setExtinctionOpticalDepthsFromTables(SpatialGridPath* path, const MediumState& state,
                                                           const vector<int>& mediaIndexv,
                                                           const vector<int>& lookupBeginv,
                                                           const vector<int>& lookupCountv,
                                                           const vector<double>& lookupWavelengthv,
                                                           const vector<double>& sectionExtv, double lambda)
{
    if (!isProcessEnabled()) return false;

    vector<double> contributions;
    vector<double> unused;
    if (!computeTableContributions(path, state, mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv,
                                   sectionExtv, nullptr, lambda, contributions, unused))
        return false;

    double tau = 0.;
    auto& segments = path->segments();
    for (size_t i = 0; i != segments.size(); ++i)
    {
        tau += contributions[i];
        segments[i].setOpticalDepth(tau);
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setExtinctionOpticalDepthsFromTables(const vector<SpatialGridPath*>& paths,
                                                           const MediumState& state,
                                                           const vector<int>& mediaIndexv,
                                                           const vector<int>& lookupBeginv,
                                                           const vector<int>& lookupCountv,
                                                           const vector<double>& lookupWavelengthv,
                                                           const vector<double>& sectionExtv,
                                                           const vector<double>& lambdav)
{
    if (!isProcessEnabled()) return false;

    vector<int> pathOffsetv;
    vector<double> cumulative;
    vector<double> unusedCumulative;
    if (!computeCumulativeTableOpticalDepthsBatch(paths, state, mediaIndexv, lookupBeginv, lookupCountv,
                                                  lookupWavelengthv, sectionExtv, nullptr, lambdav, pathOffsetv,
                                                  cumulative, unusedCumulative))
        return false;

    for (size_t pathIndex = 0; pathIndex != paths.size(); ++pathIndex)
    {
        auto& segments = paths[pathIndex]->segments();
        size_t begin = static_cast<size_t>(pathOffsetv[pathIndex]);
        for (size_t i = 0; i != segments.size(); ++i) segments[i].setOpticalDepth(cumulative[begin + i]);
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::getExtinctionOpticalDepth(const SpatialGridPath* path, const MediumState& state,
                                                const vector<double>& sectionv, double taumax, double& tau)
{
    if (!isProcessEnabled()) return false;
    if (computeOpticalDepth(path, state, sectionv, taumax, tau)) return true;

    vector<double> contributions;
    vector<double> unused;
    if (!computeContributions(path, state, sectionv, nullptr, contributions, unused)) return false;

    tau = 0.;
    for (double contribution : contributions)
    {
        tau += contribution;
        if (tau >= taumax)
        {
            tau = std::numeric_limits<double>::infinity();
            break;
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::getExtinctionOpticalDepthFromTables(const SpatialGridPath* path, const MediumState& state,
                                                          const vector<int>& mediaIndexv,
                                                          const vector<int>& lookupBeginv,
                                                          const vector<int>& lookupCountv,
                                                          const vector<double>& lookupWavelengthv,
                                                          const vector<double>& sectionExtv, double lambda,
                                                          double taumax, double& tau)
{
    if (!isProcessEnabled()) return false;

    vector<double> contributions;
    vector<double> unused;
    if (!computeTableContributions(path, state, mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv,
                                   sectionExtv, nullptr, lambda, contributions, unused))
        return false;

    tau = 0.;
    for (double contribution : contributions)
    {
        tau += contribution;
        if (tau >= taumax)
        {
            tau = std::numeric_limits<double>::infinity();
            break;
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::findInteractionPointUsingExtinction(const SpatialGridPath* path, const MediumState& state,
                                                          const vector<double>& sectionv, double tauinteract,
                                                          bool& found, int& m, double& s)
{
    found = false;
    m = -1;
    s = 0.;
    if (!isProcessEnabled()) return false;
    if (computeInteractionPointExtinction(path, state, sectionv, tauinteract, found, m, s)) return true;

    vector<double> contributions;
    vector<double> unused;
    if (!computeContributions(path, state, sectionv, nullptr, contributions, unused)) return false;

    double tau = 0.;
    double distance = 0.;
    const auto& segments = path->segments();
    for (size_t i = 0; i != segments.size(); ++i)
    {
        double tau0 = tau;
        double s0 = distance;
        tau += contributions[i];
        distance += segments[i].ds();
        if (tauinteract < tau)
        {
            found = true;
            m = segments[i].m();
            s = interpolateLinLin(tauinteract, tau0, tau, s0, distance);
            return true;
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::findInteractionPointUsingExtinctionFromTables(
    const SpatialGridPath* path, const MediumState& state, const vector<int>& mediaIndexv,
    const vector<int>& lookupBeginv, const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
    const vector<double>& sectionExtv, double lambda, double tauinteract, bool& found, int& m, double& s)
{
    found = false;
    m = -1;
    s = 0.;
    if (!isProcessEnabled()) return false;

    vector<double> contributions;
    vector<double> unused;
    if (!computeTableContributions(path, state, mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv,
                                   sectionExtv, nullptr, lambda, contributions, unused))
        return false;

    double tau = 0.;
    double distance = 0.;
    const auto& segments = path->segments();
    for (size_t i = 0; i != segments.size(); ++i)
    {
        double tau0 = tau;
        double s0 = distance;
        tau += contributions[i];
        distance += segments[i].ds();
        if (tauinteract < tau)
        {
            found = true;
            m = segments[i].m();
            s = interpolateLinLin(tauinteract, tau0, tau, s0, distance);
            return true;
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setScatteringAndAbsorptionOpticalDepths(SpatialGridPath* path, const MediumState& state,
                                                              const vector<double>& sectionScav,
                                                              const vector<double>& sectionAbsv)
{
    if (!isProcessEnabled()) return false;

    vector<double> cumulativeSca;
    vector<double> cumulativeAbs;
    if (computeCumulativeOpticalDepths(path, state, sectionScav, &sectionAbsv, cumulativeSca, cumulativeAbs))
    {
        auto& segments = path->segments();
        for (size_t i = 0; i != segments.size(); ++i) segments[i].setOpticalDepth(cumulativeSca[i], cumulativeAbs[i]);
        return true;
    }

    vector<double> scaContributions;
    vector<double> absContributions;
    if (!computeContributions(path, state, sectionScav, &sectionAbsv, scaContributions, absContributions)) return false;

    double tauSca = 0.;
    double tauAbs = 0.;
    auto& segments = path->segments();
    for (size_t i = 0; i != segments.size(); ++i)
    {
        tauSca += scaContributions[i];
        tauAbs += absContributions[i];
        segments[i].setOpticalDepth(tauSca, tauAbs);
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setScatteringAndAbsorptionOpticalDepths(const vector<SpatialGridPath*>& paths,
                                                              const MediumState& state,
                                                              const vector<double>& sectionScav,
                                                              const vector<double>& sectionAbsv)
{
    if (!isProcessEnabled()) return false;

    vector<int> pathOffsetv;
    vector<double> cumulativeSca;
    vector<double> cumulativeAbs;
    if (!computeCumulativeOpticalDepthsBatch(paths, state, sectionScav, &sectionAbsv, pathOffsetv, cumulativeSca,
                                             cumulativeAbs))
        return false;

    for (size_t pathIndex = 0; pathIndex != paths.size(); ++pathIndex)
    {
        auto& segments = paths[pathIndex]->segments();
        size_t begin = static_cast<size_t>(pathOffsetv[pathIndex]);
        for (size_t i = 0; i != segments.size(); ++i)
            segments[i].setOpticalDepth(cumulativeSca[begin + i], cumulativeAbs[begin + i]);
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setScatteringAndAbsorptionOpticalDepthsFromTables(
    SpatialGridPath* path, const MediumState& state, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
    const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv, const vector<double>& sectionScav,
    const vector<double>& sectionAbsv, double lambda)
{
    if (!isProcessEnabled()) return false;

    vector<double> scaContributions;
    vector<double> absContributions;
    if (!computeTableContributions(path, state, mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv,
                                   sectionScav, &sectionAbsv, lambda, scaContributions, absContributions))
        return false;

    double tauSca = 0.;
    double tauAbs = 0.;
    auto& segments = path->segments();
    for (size_t i = 0; i != segments.size(); ++i)
    {
        tauSca += scaContributions[i];
        tauAbs += absContributions[i];
        segments[i].setOpticalDepth(tauSca, tauAbs);
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::setScatteringAndAbsorptionOpticalDepthsFromTables(
    const vector<SpatialGridPath*>& paths, const MediumState& state, const vector<int>& mediaIndexv,
    const vector<int>& lookupBeginv, const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
    const vector<double>& sectionScav, const vector<double>& sectionAbsv, const vector<double>& lambdav)
{
    if (!isProcessEnabled()) return false;

    vector<int> pathOffsetv;
    vector<double> cumulativeSca;
    vector<double> cumulativeAbs;
    if (!computeCumulativeTableOpticalDepthsBatch(paths, state, mediaIndexv, lookupBeginv, lookupCountv,
                                                  lookupWavelengthv, sectionScav, &sectionAbsv, lambdav, pathOffsetv,
                                                  cumulativeSca, cumulativeAbs))
        return false;

    for (size_t pathIndex = 0; pathIndex != paths.size(); ++pathIndex)
    {
        auto& segments = paths[pathIndex]->segments();
        size_t begin = static_cast<size_t>(pathOffsetv[pathIndex]);
        for (size_t i = 0; i != segments.size(); ++i)
            segments[i].setOpticalDepth(cumulativeSca[begin + i], cumulativeAbs[begin + i]);
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::findInteractionPointUsingScatteringAndAbsorption(const SpatialGridPath* path,
                                                                       const MediumState& state,
                                                                       const vector<double>& sectionScav,
                                                                       const vector<double>& sectionAbsv,
                                                                       double tauinteract, bool& found, int& m,
                                                                       double& s, double& tauAbs)
{
    found = false;
    m = -1;
    s = 0.;
    tauAbs = 0.;
    if (!isProcessEnabled()) return false;
    if (computeInteractionPointScatteringAndAbsorption(path, state, sectionScav, sectionAbsv, tauinteract, found, m, s,
                                                       tauAbs))
        return true;

    vector<double> scaContributions;
    vector<double> absContributions;
    if (!computeContributions(path, state, sectionScav, &sectionAbsv, scaContributions, absContributions)) return false;

    double tauSca = 0.;
    double cumulativeTauAbs = 0.;
    double distance = 0.;
    const auto& segments = path->segments();
    for (size_t i = 0; i != segments.size(); ++i)
    {
        double tauSca0 = tauSca;
        double tauAbs0 = cumulativeTauAbs;
        double s0 = distance;
        tauSca += scaContributions[i];
        cumulativeTauAbs += absContributions[i];
        distance += segments[i].ds();
        if (tauinteract < tauSca)
        {
            found = true;
            m = segments[i].m();
            s = interpolateLinLin(tauinteract, tauSca0, tauSca, s0, distance);
            tauAbs = interpolateLinLin(tauinteract, tauSca0, tauSca, tauAbs0, cumulativeTauAbs);
            return true;
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::findInteractionPointUsingScatteringAndAbsorptionFromTables(
    const SpatialGridPath* path, const MediumState& state, const vector<int>& mediaIndexv,
    const vector<int>& lookupBeginv, const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
    const vector<double>& sectionScav, const vector<double>& sectionAbsv, double lambda, double tauinteract,
    bool& found, int& m, double& s, double& tauAbs)
{
    found = false;
    m = -1;
    s = 0.;
    tauAbs = 0.;
    if (!isProcessEnabled()) return false;

    vector<double> scaContributions;
    vector<double> absContributions;
    if (!computeTableContributions(path, state, mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv,
                                   sectionScav, &sectionAbsv, lambda, scaContributions, absContributions))
        return false;

    double tauSca = 0.;
    double cumulativeTauAbs = 0.;
    double distance = 0.;
    const auto& segments = path->segments();
    for (size_t i = 0; i != segments.size(); ++i)
    {
        double tauSca0 = tauSca;
        double tauAbs0 = cumulativeTauAbs;
        double s0 = distance;
        tauSca += scaContributions[i];
        cumulativeTauAbs += absContributions[i];
        distance += segments[i].ds();
        if (tauinteract < tauSca)
        {
            found = true;
            m = segments[i].m();
            s = interpolateLinLin(tauinteract, tauSca0, tauSca, s0, distance);
            tauAbs = interpolateLinLin(tauinteract, tauSca0, tauSca, tauAbs0, cumulativeTauAbs);
            return true;
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::scatteringProperties(const MediumState& state, int cellIndex, const vector<double>& sectionScav,
                                           const vector<double>& sectionExtv, double& albedo,
                                           vector<double>& weights)
{
    if (!isProcessEnabled()) return false;
    return computeScatteringProperties(state, cellIndex, sectionScav, sectionExtv, albedo, weights);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::scatteringPropertiesFromTables(const MediumState& state, int cellIndex,
                                                     const vector<int>& mediaIndexv,
                                                     const vector<int>& lookupBeginv,
                                                     const vector<int>& lookupCountv,
                                                     const vector<double>& lookupWavelengthv,
                                                     const vector<double>& sectionScav,
                                                     const vector<double>& sectionExtv, double lambda,
                                                     double& albedo, vector<double>& weights)
{
    if (!isProcessEnabled()) return false;
    return computeTableScatteringProperties(state, cellIndex, mediaIndexv, lookupBeginv, lookupCountv,
                                            lookupWavelengthv, sectionScav, sectionExtv, lambda, albedo, weights);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::scatteringAlbedos(const MediumState& state, const vector<int>& cellv,
                                        const vector<double>& sectionScav, const vector<double>& sectionExtv,
                                        vector<double>& albedov)
{
    if (!isProcessEnabled()) return false;
    return computeScatteringAlbedos(state, cellv, sectionScav, sectionExtv, albedov);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::scatteringAlbedosFromTables(const MediumState& state, const vector<int>& cellv,
                                                  const vector<double>& lambdav,
                                                  const vector<int>& mediaIndexv,
                                                  const vector<int>& lookupBeginv,
                                                  const vector<int>& lookupCountv,
                                                  const vector<double>& lookupWavelengthv,
                                                  const vector<double>& sectionScav,
                                                  const vector<double>& sectionExtv, vector<double>& albedov)
{
    if (!isProcessEnabled()) return false;
    return computeTableScatteringAlbedos(state, cellv, lambdav, mediaIndexv, lookupBeginv, lookupCountv,
                                         lookupWavelengthv, sectionScav, sectionExtv, albedov);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::findInteractionPointInCumulativePath(const SpatialGridPath* path, double tauinteract,
                                                           bool hasAbsorption, int& m, double& s, double& tauAbs)
{
    if (!isProcessEnabled()) return false;
    return computeCumulativePathInteractionPoint(path, tauinteract, hasAbsorption, m, s, tauAbs);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::findInteractionPointsInCumulativePaths(const vector<const SpatialGridPath*>& paths,
                                                             const vector<double>& tauinteractv,
                                                             bool hasAbsorption, vector<int>& cellv,
                                                             vector<double>& distancev, vector<double>& tauAbsv)
{
    if (!isProcessEnabled()) return false;
    return computeCumulativePathInteractionPoints(paths, tauinteractv, hasAbsorption, cellv, distancev, tauAbsv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::forcedPropagationResults(const vector<const SpatialGridPath*>& paths,
                                               const vector<double>& tauinteractv,
                                               const vector<double>& pathBiasWeightv, bool hasAbsorption,
                                               const vector<double>& albedov, vector<int>& cellv,
                                               vector<double>& distancev, vector<double>& tauAbsv,
                                               vector<double>& weightv)
{
    if (!isProcessEnabled()) return false;
    return computeForcedPropagationResults(paths, tauinteractv, pathBiasWeightv, hasAbsorption, albedov, cellv,
                                           distancev, tauAbsv, weightv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::forcedPropagationResultsFromTables(
    const vector<const SpatialGridPath*>& paths, const MediumState& state, const vector<double>& tauinteractv,
    const vector<double>& pathBiasWeightv, const vector<double>& lambdav, const vector<int>& mediaIndexv,
    const vector<int>& lookupBeginv, const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
    const vector<double>& sectionScav, const vector<double>& sectionExtv, vector<int>& cellv,
    vector<double>& distancev, vector<double>& tauAbsv, vector<double>& weightv)
{
    if (!isProcessEnabled()) return false;
    return computeForcedPropagationTableAlbedoResults(paths, state, tauinteractv, pathBiasWeightv, lambdav,
                                                      mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv,
                                                      sectionScav, sectionExtv, cellv, distancev, tauAbsv, weightv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::radiationFieldSumsAndForcedPropagationResultsFromTables(
    const vector<const SpatialGridPath*>& paths, const MediumState& state,
    const vector<double>& luminosityv, const vector<int>& wavelengthBinv, int numWavelengths,
    const vector<double>& tauinteractv, const vector<double>& pathBiasWeightv,
    const vector<double>& lambdav, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
    const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
    const vector<double>& sectionScav, const vector<double>& sectionExtv, vector<int>& binIndexv,
    vector<double>& Ldsv, vector<int>& cellv, vector<double>& distancev, vector<double>& tauAbsv,
    vector<double>& weightv)
{
    if (!isProcessEnabled()) return false;
    return computeRadiationFieldSumsAndForcedPropagationTableAlbedoResults(
        paths, state, luminosityv, wavelengthBinv, numWavelengths, tauinteractv, pathBiasWeightv, lambdav,
        mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv, sectionScav, sectionExtv, binIndexv, Ldsv,
        cellv, distancev, tauAbsv, weightv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::voronoiMeshSpatialGridRadiationFieldAndForcedPropagationResultsFromTables(
    const VoronoiMeshSpatialGrid& grid, const vector<Position>& positionv,
    const vector<Direction>& directionv, const MediumState& state,
    const vector<double>& luminosityv, const vector<int>& wavelengthBinv, int numWavelengths,
    const vector<double>& tauinteractv, const vector<double>& pathBiasWeightv,
    const vector<double>& lambdav, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
    const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
    const vector<double>& sectionScav, const vector<double>& sectionExtv, vector<int>& binIndexv,
    vector<double>& Ldsv, vector<int>& cellv, vector<double>& distancev, vector<double>& tauAbsv,
    vector<double>& weightv, const void* accumulatorKey, size_t numAccumulatorValues)
{
    if (!isProcessEnabled()) return false;
    int numCells = grid.numCells();
    if (numCells <= 0) return false;
    return computeVoronoiTableRadiationFieldSumsAndForcedPropagationResults(
        &grid, grid.traversalSiteCoordinates(), grid.traversalNeighborBegin(), grid.traversalNeighborCount(),
        grid.traversalNeighborIndex(), grid.traversalBlockBegin(), grid.traversalBlockCount(),
	        grid.traversalBlockIndex(), grid.traversalBlockGridN(), numCells, grid.traversalEpsilon(), grid.xmin(),
	        grid.ymin(), grid.zmin(), grid.xmax(), grid.ymax(), grid.zmax(),
	        std::numeric_limits<double>::infinity(), positionv, directionv, state, luminosityv, wavelengthBinv,
	        numWavelengths, tauinteractv, pathBiasWeightv, lambdav, nullptr, nullptr, 0.0, mediaIndexv, lookupBeginv,
	        lookupCountv, lookupWavelengthv, sectionScav, sectionExtv,
		        binIndexv, Ldsv, cellv, distancev, tauAbsv, weightv, nullptr, nullptr, 0, 0, nullptr, nullptr,
            accumulatorKey, numAccumulatorValues);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::voronoiMeshSpatialGridSampledRadiationFieldAndForcedPropagationResultsFromTables(
    const VoronoiMeshSpatialGrid& grid, const vector<Position>& positionv,
    const vector<Direction>& directionv, const MediumState& state,
    const vector<double>& luminosityv, const vector<int>& wavelengthBinv, int numWavelengths,
    const vector<double>& randomSelectv, const vector<double>& randomSamplev, double pathLengthBias,
    const vector<double>& lambdav, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
    const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
    const vector<double>& sectionScav, const vector<double>& sectionExtv, vector<int>& binIndexv,
    vector<double>& Ldsv, vector<int>& cellv, vector<double>& distancev, vector<double>& tauAbsv,
    vector<double>& weightv, const void* accumulatorKey, size_t numAccumulatorValues)
{
    if (!isProcessEnabled()) return false;
    int numCells = grid.numCells();
    if (numCells <= 0) return false;
    const vector<double> tauinteractv;
    const vector<double> pathBiasWeightv;
    return computeVoronoiTableRadiationFieldSumsAndForcedPropagationResults(
        &grid, grid.traversalSiteCoordinates(), grid.traversalNeighborBegin(), grid.traversalNeighborCount(),
        grid.traversalNeighborIndex(), grid.traversalBlockBegin(), grid.traversalBlockCount(),
        grid.traversalBlockIndex(), grid.traversalBlockGridN(), numCells, grid.traversalEpsilon(), grid.xmin(),
        grid.ymin(), grid.zmin(), grid.xmax(), grid.ymax(), grid.zmax(),
        std::numeric_limits<double>::infinity(), positionv, directionv, state, luminosityv, wavelengthBinv,
	        numWavelengths, tauinteractv, pathBiasWeightv, lambdav, &randomSelectv, &randomSamplev, pathLengthBias,
	        mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv, sectionScav, sectionExtv, binIndexv, Ldsv,
	        cellv, distancev, tauAbsv, weightv, nullptr, nullptr, 0, 0, nullptr, nullptr, accumulatorKey,
            numAccumulatorValues);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::voronoiMeshSpatialGridSampledRadiationFieldForcedPropagationAndHenyeyGreensteinScatteringResultsFromTables(
    const VoronoiMeshSpatialGrid& grid, const vector<Position>& positionv,
    const vector<Direction>& directionv, const MediumState& state,
    const vector<double>& luminosityv, const vector<int>& wavelengthBinv, int numWavelengths,
    const vector<double>& randomSelectv, const vector<double>& randomSamplev, double pathLengthBias,
    const vector<double>& scatterRandomCosthetav, const vector<double>& scatterRandomPhiv,
    int hgLookupBegin, int hgLookupCount, const vector<double>& asymmparv,
    const vector<double>& lambdav, const vector<int>& mediaIndexv, const vector<int>& lookupBeginv,
    const vector<int>& lookupCountv, const vector<double>& lookupWavelengthv,
    const vector<double>& sectionScav, const vector<double>& sectionExtv, vector<int>& binIndexv,
    vector<double>& Ldsv, vector<int>& cellv, vector<double>& distancev, vector<double>& tauAbsv,
    vector<double>& weightv, vector<double>& scatterDirectionv, const void* accumulatorKey,
    size_t numAccumulatorValues)
{
    if (!isProcessEnabled()) return false;
    int numCells = grid.numCells();
    if (numCells <= 0) return false;
    const vector<double> tauinteractv;
    const vector<double> pathBiasWeightv;
    return computeVoronoiTableRadiationFieldSumsAndForcedPropagationResults(
        &grid, grid.traversalSiteCoordinates(), grid.traversalNeighborBegin(), grid.traversalNeighborCount(),
        grid.traversalNeighborIndex(), grid.traversalBlockBegin(), grid.traversalBlockCount(),
        grid.traversalBlockIndex(), grid.traversalBlockGridN(), numCells, grid.traversalEpsilon(), grid.xmin(),
        grid.ymin(), grid.zmin(), grid.xmax(), grid.ymax(), grid.zmax(),
        std::numeric_limits<double>::infinity(), positionv, directionv, state, luminosityv, wavelengthBinv,
        numWavelengths, tauinteractv, pathBiasWeightv, lambdav, &randomSelectv, &randomSamplev, pathLengthBias,
        mediaIndexv, lookupBeginv, lookupCountv, lookupWavelengthv, sectionScav, sectionExtv, binIndexv, Ldsv,
        cellv, distancev, tauAbsv, weightv, &scatterRandomCosthetav, &scatterRandomPhiv, hgLookupBegin,
        hgLookupCount, &asymmparv, &scatterDirectionv, accumulatorKey, numAccumulatorValues);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::dustAbsorbedLuminosities(const MediumState& state, int numCells, int numWavelengths,
                                               const vector<int>& dustMedia, const vector<double>& sectionAbsv,
                                               const double* primaryRadiationField,
                                               const double* secondaryRadiationField,
                                               vector<double>& primaryLuminosities,
                                               vector<double>& secondaryLuminosities)
{
    if (!isProcessEnabled()) return false;
    return computeDustLuminosities(state, numCells, numWavelengths, dustMedia, sectionAbsv, primaryRadiationField,
                                   secondaryRadiationField, primaryLuminosities, secondaryLuminosities);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::totalDustAbsorbedLuminosity(const MediumState& state, int numCells, int numWavelengths,
                                                  const vector<int>& dustMedia, const vector<double>& sectionAbsv,
                                                  const double* primaryRadiationField,
                                                  const double* secondaryRadiationField, double& primaryLuminosity,
                                                  double& secondaryLuminosity)
{
    if (!isProcessEnabled()) return false;
    return computeTotalDustLuminosity(state, numCells, numWavelengths, dustMedia, sectionAbsv, primaryRadiationField,
                                      secondaryRadiationField, primaryLuminosity, secondaryLuminosity);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::radiationFieldContributions(const SpatialGridPath* path, double luminosity, vector<double>& Ldsv)
{
    if (!isProcessEnabled()) return false;
    return computeRadiationContributions(path, luminosity, Ldsv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::radiationFieldContributions(const vector<const SpatialGridPath*>& paths,
                                                  const vector<double>& luminosityv, vector<int>& pathOffsetv,
                                                  vector<double>& Ldsv)
{
    if (!isProcessEnabled()) return false;
    return computeRadiationContributionsBatch(paths, luminosityv, pathOffsetv, Ldsv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::radiationFieldContributionSums(const vector<const SpatialGridPath*>& paths,
                                                     const vector<double>& luminosityv,
                                                     const vector<int>& wavelengthBinv, int numWavelengths,
                                                     vector<int>& binIndexv, vector<double>& Ldsv)
{
    if (!isProcessEnabled()) return false;
    return computeRadiationContributionSumsBatch(paths, luminosityv, wavelengthBinv, numWavelengths, binIndexv, Ldsv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::scaleWavelengthValues(double* values, size_t numWavelengths, const vector<double>& factorv)
{
    if (!isProcessEnabled()) return false;
    return computeScaleWavelengthValues(values, numWavelengths, factorv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::scaleFrameWavelengthValues(double* values, size_t numWavelengths, size_t numPixelsInFrame,
                                                 const vector<double>& factorv)
{
    if (!isProcessEnabled()) return false;
    return computeScaleFrameWavelengthValues(values, numWavelengths, numPixelsInFrame, factorv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::divideValues(double* values, size_t numValues, double divisor)
{
    if (!isProcessEnabled()) return false;
    return computeDivideValues(values, numValues, divisor);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::multiplyValues(double* values, size_t numValues, double factor)
{
    if (!isProcessEnabled()) return false;
    return computeMultiplyValues(values, numValues, factor);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::sumValues(double* output, size_t numValues, const double* value1, const double* value2,
                                const double* value3, const double* value4)
{
    if (!isProcessEnabled()) return false;
    return computeSumValues(output, numValues, value1, value2, value3, value4);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::sumValuesByKey(const vector<int>& keyv, const vector<double>& valuev,
                                     vector<int>& compactKeyv, vector<double>& compactValuev)
{
    if (!isProcessEnabled()) return false;
    return computeKeySums(keyv, valuev, compactKeyv, compactValuev);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::accumulateValuesByKey(const void* accumulatorKey, size_t numAccumulatorValues,
                                            const vector<int>& keyv, const vector<double>& valuev)
{
    if (!isProcessEnabled()) return false;
    return computeAccumulateValuesByKey(accumulatorKey, numAccumulatorValues, keyv, valuev);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::retrieveAndClearAccumulatedValues(const void* accumulatorKey, double* values, size_t numValues)
{
    if (!isProcessEnabled()) return false;
    return computeRetrieveAndClearAccumulatedValues(accumulatorKey, values, numValues);
}

////////////////////////////////////////////////////////////////////

void GpuAcceleration::clearAccumulatedValues(const void* accumulatorKey)
{
    if (!isProcessEnabled()) return;
    computeClearAccumulatedValues(accumulatorKey);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::frameBandTotalFluxSums(
    const vector<Position>& positionv, const vector<double>& wavelengthv, const vector<double>& luminosityv,
    const vector<double>& tauv, bool hasMedium, double costheta, double sintheta, double cosphi,
    double sinphi, double cosomega, double sinomega, int numPixelsX, int numPixelsY, double xpmin,
    double xpsiz, double ypmin, double ypsiz, double redshift, size_t numPixelsInFrame,
    const vector<int>& bandOffsetv, const vector<double>& bandWavelengthv,
    const vector<double>& bandTransmissionv, const vector<double>& bandWidthv,
    vector<int>& compactKeyv, vector<double>& compactValuev)
{
    if (!isProcessEnabled()) return false;
    return computeFrameBandTotalFluxSums(
        positionv, wavelengthv, luminosityv, tauv, hasMedium, costheta, sintheta, cosphi, sinphi, cosomega,
        sinomega, numPixelsX, numPixelsY, xpmin, xpsiz, ypmin, ypsiz, redshift, numPixelsInFrame,
        bandOffsetv, bandWavelengthv, bandTransmissionv, bandWidthv, compactKeyv, compactValuev);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::frameBandTotalFluxAccumulate(
    const void* accumulatorKey, size_t numAccumulatorValues,
    const vector<Position>& positionv, const vector<double>& wavelengthv, const vector<double>& luminosityv,
    const vector<double>& tauv, bool hasMedium, double costheta, double sintheta, double cosphi,
    double sinphi, double cosomega, double sinomega, int numPixelsX, int numPixelsY, double xpmin,
    double xpsiz, double ypmin, double ypsiz, double redshift, size_t numPixelsInFrame,
    const vector<int>& bandOffsetv, const vector<double>& bandWavelengthv,
    const vector<double>& bandTransmissionv, const vector<double>& bandWidthv)
{
    if (!isProcessEnabled()) return false;
    return computeFrameBandTotalFluxAccumulate(
        accumulatorKey, numAccumulatorValues, positionv, wavelengthv, luminosityv, tauv, hasMedium, costheta,
        sintheta, cosphi, sinphi, cosomega, sinomega, numPixelsX, numPixelsY, xpmin, xpsiz, ypmin, ypsiz,
        redshift, numPixelsInFrame, bandOffsetv, bandWavelengthv, bandTransmissionv, bandWidthv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::henyeyGreensteinScatteringLuminosities(const vector<double>& inputDirectionv,
                                                             const vector<double>& packetLuminosityv,
                                                             const vector<double>& lambdav, Direction bfkobs,
                                                             int lookupBegin, int lookupCount,
                                                             const vector<double>& lookupWavelengthv,
                                                             const vector<double>& asymmparv,
                                                             vector<double>& luminosityv)
{
    if (!isProcessEnabled()) return false;
    return computeHenyeyGreensteinScatteringLuminosities(inputDirectionv, packetLuminosityv, lambdav, bfkobs,
                                                        lookupBegin, lookupCount, lookupWavelengthv,
                                                        asymmparv, luminosityv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::henyeyGreensteinScatteringDirections(const vector<double>& inputDirectionv,
                                                           const vector<double>& lambdav,
                                                           const vector<double>& randomCosthetav,
                                                           const vector<double>& randomPhiv, int lookupBegin,
                                                           int lookupCount,
                                                           const vector<double>& lookupWavelengthv,
                                                           const vector<double>& asymmparv,
                                                           vector<double>& outputDirectionv)
{
    if (!isProcessEnabled()) return false;
    return computeHenyeyGreensteinScatteringDirections(inputDirectionv, lambdav, randomCosthetav, randomPhiv,
                                                       lookupBegin, lookupCount, lookupWavelengthv, asymmparv,
                                                       outputDirectionv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::isotropicDirections(const vector<double>& randomCosthetav, const vector<double>& randomPhiv,
                                          vector<double>& outputDirectionv)
{
    if (!isProcessEnabled()) return false;
    return computeIsotropicDirections(randomCosthetav, randomPhiv, outputDirectionv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::compositeLaunchWeights(const double* luminosityv, size_t numValues, double spatialBias,
                                             double* weightv)
{
    if (!isProcessEnabled()) return false;
    return computeCompositeLaunchWeights(luminosityv, numValues, spatialBias, weightv);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::storedTableCdf(vector<double>& luminosities, int numAxes, const vector<const double*>& axisData,
                                     const vector<size_t>& axisSizes, const vector<bool>& axisLog,
                                     const double* quantity, size_t quantityStep, bool quantityLog,
                                     bool clampFirstAxis, double xmin, double xmax,
                                     const vector<double>& parameterValues, const vector<double>& scaleValues,
                                     size_t numEntities)
{
    if (!isProcessEnabled()) return false;
    return computeStoredTableCdf(luminosities, numAxes, axisData, axisSizes, axisLog, quantity, quantityStep,
                                 quantityLog, clampFirstAxis, xmin, xmax, parameterValues, scaleValues, numEntities);
}

////////////////////////////////////////////////////////////////////

bool GpuAcceleration::storedTableSampleWavelengths(vector<double>& wavelengths,
                                                   vector<double>& specificLuminosities, int numAxes,
                                                   const vector<const double*>& axisData,
                                                   const vector<size_t>& axisSizes,
                                                   const vector<bool>& axisLog, const double* quantity,
                                                   size_t quantityStep, bool quantityLog, bool clampFirstAxis,
                                                   double xmin, double xmax,
                                                   const vector<double>& parameterValues,
                                                   const vector<double>& intrinsicRandoms,
                                                   const vector<double>& forcedWavelengths, size_t numSamples)
{
    if (!isProcessEnabled()) return false;
    return computeStoredTableSampleWavelengths(wavelengths, specificLuminosities, numAxes, axisData, axisSizes,
                                               axisLog, quantity, quantityStep, quantityLog, clampFirstAxis, xmin,
                                               xmax, parameterValues, intrinsicRandoms, forcedWavelengths,
                                               numSamples);
}

////////////////////////////////////////////////////////////////////
