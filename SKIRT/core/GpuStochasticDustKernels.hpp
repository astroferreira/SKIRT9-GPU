#ifndef GPUSTOCHASTICDUSTKERNELS_HPP
#define GPUSTOCHASTICDUSTKERNELS_HPP

// Appended to the runtime CUDA source after its shared lookup helpers.
//
// One warp computes the emissivity of one cell; blocks hold several warps and loop over cells, so device
// memory no longer scales with the number of cells in a batch. The heating and radiation-field bin tables
// are stored column-major per lower triangle (column i holds rows f = i+1 .. N-1 contiguously).
//
// The CPU algorithm builds the cumulative transition matrix A'(f,i) = sum_{f'>=f} A(f',i) and solves
// P[i] = sum_{j<i} A'(i,j) P[j] / C[i]. Exchanging the two sums gives sum_{f>=i} Q(f) with
// Q(f) = sum_{j<i} A(f,j) P[j]; Q is updated with column i of A once P[i] is known. This removes the
// O(N^2) scratch matrix while keeping every term positive, so the result differs from the CPU only by
// floating-point summation order. Each step's table column is loaded while the previous row sum is
// reduced, which hides most of the global-memory latency of this sequential recurrence.
inline const char* stochasticDustKernelSource()
{
    return R"cuda(
// register slots per lane for the probability recurrence; the fixed SKIRT temperature grids need at most 750 points
#define SKIRT_DUST_SLOTS 24
static __device__ unsigned long long skirt_dust_column(int column, int size)
{
    return (unsigned long long)column * (size - 1) - (unsigned long long)column * (column - 1) / 2;
}

static __device__ double skirt_dust_warp_sum(double value)
{
    for (int stride = 16; stride; stride >>= 1) value += __shfl_xor_sync(0xffffffff, value, stride);
    return value;
}

static __device__ double skirt_dust_warp_max(double value)
{
    for (int stride = 16; stride; stride >>= 1) value = fmax(value, __shfl_xor_sync(0xffffffff, value, stride));
    return value;
}

// Computes probabilities over [tmin,tmax] on one temperature grid and narrows the range in place.
// P is a warp-private shared array with room for every possible count; J includes the CMB.
static __device__ bool skirt_dust_warp_probabilities(
    const double* temperatures, int size, const double* heating, const short* bins, const double* cooling,
    const double* J, double* P, int maxCount, int& offset, double& tmin, double& tmax)
{
    int lane = threadIdx.x & 31;
    offset = skirt_locate_clip(temperatures, tmin, size);
    int count = skirt_locate_clip(temperatures, tmax, size) - offset + 2;
    if (count > maxCount || count > 32 * SKIRT_DUST_SLOTS) return false;

    // Q(f) = A(f,0) P[0] with P[0] = 1; lane l keeps Q(f) for f = 32 s + l in register slot s
    double q[SKIRT_DUST_SLOTS];
    unsigned long long column = skirt_dust_column(offset, size);
    double partial = 0.;
#pragma unroll
    for (int s = 0; s != SKIRT_DUST_SLOTS; ++s)
    {
        int f = 32 * s + lane;
        q[s] = 0.;
        if (f >= 1 && f < count)
        {
            int k = bins[column + f - 1];
            q[s] = k >= 0 ? heating[column + f - 1] * J[k] : 0.;
            partial += q[s];
        }
    }
    if (!lane) P[0] = 1.;
    double sum = skirt_dust_warp_sum(partial);
    // software pipeline: the column for step i is loaded during the reduction of step i-1
    double h[SKIRT_DUST_SLOTS];
    int kk[SKIRT_DUST_SLOTS];
    column = skirt_dust_column(1 + offset, size) - 2;
#pragma unroll
    for (int s = 0; s != SKIRT_DUST_SLOTS; ++s)
    {
        int f = 32 * s + lane;
        kk[s] = -1;
        if (f > 1 && f < count) { kk[s] = __ldg(bins + column + f); h[s] = __ldg(heating + column + f); }
    }
    for (int i = 1; i < count; ++i)
    {
        double value = sum / cooling[i + offset];
        __syncwarp();
        if (!lane) P[i] = value;
        if (value > 1e10)
        {
            __syncwarp();
            for (int j = lane; j <= i; j += 32) P[j] /= value;
#pragma unroll
            for (int s = 0; s != SKIRT_DUST_SLOTS; ++s) q[s] /= value;
            value = 1.;
        }
        partial = 0.;
#pragma unroll
        for (int s = 0; s != SKIRT_DUST_SLOTS; ++s)
        {
            int f = 32 * s + lane;
            if (32 * s + 31 > i && f > i && f < count)
            {
                double a = kk[s] >= 0 ? h[s] * J[kk[s]] : 0.;
                q[s] += a * value;
                partial += q[s];
            }
        }
        column = skirt_dust_column(i + 1 + offset, size) - i - 2;
#pragma unroll
        for (int s = 0; s != SKIRT_DUST_SLOTS; ++s)
        {
            int f = 32 * s + lane;
            kk[s] = -1;
            if (32 * s + 31 > i + 1 && f > i + 1 && f < count)
            {
                kk[s] = __ldg(bins + column + f);
                h[s] = __ldg(heating + column + f);
            }
        }
        sum = skirt_dust_warp_sum(partial);
    }
    __syncwarp();

    // normalize and determine the range where the probability exceeds 1e-20 of its maximum
    partial = 0.;
    for (int i = lane; i < count; i += 32) partial += P[i];
    sum = skirt_dust_warp_sum(partial);
    double maximum = 0.;
    for (int i = lane; i < count; i += 32)
    {
        double value = P[i] / sum;
        P[i] = value;
        maximum = fmax(maximum, value);
    }
    double threshold = 1e-20 * skirt_dust_warp_max(maximum);
    __syncwarp();
    int first = count - 2;
    for (int base = 0; base < count - 2; base += 32)
    {
        int k = base + lane;
        unsigned mask = __ballot_sync(0xffffffff, k < count - 2 && P[k] > threshold);
        if (mask)
        {
            first = base + __ffs(mask) - 1;
            break;
        }
    }
    int last = 1;
    for (int top = count - 2; top > 1; top -= 32)
    {
        int k = top - lane;
        unsigned mask = __ballot_sync(0xffffffff, k > 1 && P[k] > threshold);
        if (mask)
        {
            last = top - (__ffs(mask) - 1);
            break;
        }
    }
    tmin = temperatures[first + offset];
    tmax = temperatures[last + 1 + offset];
    return true;
}

extern "C" __global__ void stochastic_dust_emission(
    int numCells, int numInput, int numOutput, int numGrains, int numTypes,
    int sizeA, int sizeB, int sizeC, double planckFactor1, double planckFactor2,
    double maxTemperature, double equilibriumWidth, double mediumWidth, int maxCount,
    const int* types, const double* temperature, const double* blackbody,
    const double* heating, const short* bins, const double* cooling, const double* planckAbs,
    const double* rfCrossSection, const double* emCrossSection, const double* rfWidth,
    const double* cmb, const double* wavelength, const double* grainMass, const double* maxEnthalpyTemp,
    const double* radiationFields, double* spectra)
{
    extern __shared__ double skirtDustShared[];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;
    double* J = skirtDustShared + (unsigned long long)warp * (numInput + maxCount);
    double* P = J + numInput;
    int sizes[3] = {sizeA, sizeB, sizeC};
    int temperatureOffsets[3] = {0, sizeA, sizeA + sizeB};
    unsigned long long triangleOffsets[3] = {0, (unsigned long long)sizeA * (sizeA - 1) / 2,
                                           (unsigned long long)sizeA * (sizeA - 1) / 2
                                               + (unsigned long long)sizeB * (sizeB - 1) / 2};
    int totalTemperatures = sizeA + sizeB + sizeC;
    unsigned long long totalTriangles = triangleOffsets[2] + (unsigned long long)sizeC * (sizeC - 1) / 2;

    for (int cell = blockIdx.x * warpsPerBlock + warp; cell < numCells; cell += gridDim.x * warpsPerBlock)
    {
        double* output = spectra + (unsigned long long)cell * numOutput;
        const double* field = radiationFields + (unsigned long long)cell * numInput;
        for (int k = lane; k < numInput; k += 32) J[k] = field[k] + cmb[k];
        for (int ell = lane; ell < numOutput; ell += 32) output[ell] = 0.;
        // lane t holds the equilibrium mass cutoff for grain type t
        double equilibriumMass = 1.7976931348623157e308;
        __syncwarp();
        for (int b = 0; b != numGrains; ++b)
        {
            unsigned long long rateBase = (unsigned long long)b * totalTemperatures;
            unsigned long long heatingBase = (unsigned long long)b * totalTriangles;
            const double* sections = emCrossSection + (unsigned long long)b * numOutput;
            const double* rfSections = rfCrossSection + (unsigned long long)b * numInput;
            double partial = 0.;
            for (int k = lane; k < numInput; k += 32) partial += rfSections[k] * J[k] * rfWidth[k];
            double input = skirt_dust_warp_sum(partial);
            double teq = 0.;
            if (input > 0.)
            {
                const double* abs = planckAbs + rateBase + temperatureOffsets[2];
                const double* temp = temperature + temperatureOffsets[2];
                if (input <= abs[0]) teq = temp[0];
                else if (input >= abs[sizeC - 1]) teq = temp[sizeC - 1];
                else
                {
                    int i = skirt_locate_clip(abs, input, sizeC);
                    teq = temp[i] + (input - abs[i]) * (temp[i + 1] - temp[i]) / (abs[i + 1] - abs[i]);
                }
            }
            double tmin = 0., tmax = fmin(maxTemperature, maxEnthalpyTemp[b]);
            int type = types[b];
            bool stochastic = grainMass[b] < __shfl_sync(0xffffffff, equilibriumMass, type);
            int grid = 0, offset = 0;
            if (stochastic)
            {
                bool valid = skirt_dust_warp_probabilities(temperature, sizeA, heating + heatingBase,
                                                           bins + heatingBase, cooling + rateBase, J, P,
                                                           maxCount, offset, tmin, tmax);
                if (valid && tmax - tmin > equilibriumWidth && teq < tmax)
                {
                    grid = tmax - tmin > mediumWidth ? 1 : 2;
                    valid = skirt_dust_warp_probabilities(temperature + temperatureOffsets[grid], sizes[grid],
                                                          heating + heatingBase + triangleOffsets[grid],
                                                          bins + heatingBase + triangleOffsets[grid],
                                                          cooling + rateBase + temperatureOffsets[grid],
                                                          J, P, maxCount, offset, tmin, tmax);
                }
                // the host bounds maxCount; an unexpected overflow poisons the cell so the host falls back
                if (!valid)
                {
                    for (int ell = lane; ell < numOutput; ell += 32) output[ell] = __longlong_as_double(0x7ff8000000000000LL);
                    break;
                }
                stochastic = grid && tmax - tmin > equilibriumWidth && teq < tmax;
                if (!stochastic && lane == type) equilibriumMass = grainMass[b];
            }
            if (stochastic)
            {
                const double* temp = temperature + temperatureOffsets[grid];
                int first = skirt_locate_clip(temp, tmin, sizes[grid]);
                int last = skirt_locate_clip(temp, tmax, sizes[grid]);
                const double* planck = blackbody + (unsigned long long)temperatureOffsets[grid] * numOutput;
                // four independent wavelength chains per lane; each keeps the sequential temperature order
                for (int base = lane; base < numOutput; base += 128)
                {
                    double value[4] = {0., 0., 0., 0.};
                    const double* row = planck + (unsigned long long)first * numOutput + base;
                    for (int i = first; i <= last; ++i, row += numOutput)
                    {
                        double p = P[i - offset];
#pragma unroll
                        for (int u = 0; u != 4; ++u)
                            if (base + 32 * u < numOutput) value[u] += __ldg(row + 32 * u) * p;
                    }
#pragma unroll
                    for (int u = 0; u != 4; ++u)
                        if (base + 32 * u < numOutput) output[base + 32 * u] += sections[base + 32 * u] * value[u];
                }
            }
            else
            {
                // wavelength[numOutput + ell] holds the host-computed planckFactor2 / pow(lambda, 5)
                double factor = planckFactor1 / teq;
                for (int ell = lane; ell < numOutput; ell += 32)
                {
                    double value = wavelength[numOutput + ell] / (exp(factor / wavelength[ell]) - 1.);
                    output[ell] += sections[ell] * value;
                }
            }
            __syncwarp();
        }
    }
}
)cuda";
}

#endif
