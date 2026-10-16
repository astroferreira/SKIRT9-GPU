#ifndef GPUSOURCESPECTRUMKERNELS_HPP
#define GPUSOURCESPECTRUMKERNELS_HPP

// Appended to the runtime CUDA source after its shared stored-table helpers.
inline const char* sourceSpectrumKernelSource()
{
    return R"cuda(
static __device__ double skirt_source_negative_infinity()
{
    return __longlong_as_double(static_cast<long long>(0xfff0000000000000ULL));
}

extern "C" __global__ void source_spectrum_log_quantity(double* values, unsigned long long count)
{
    unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (; i < count; i += stride)
    {
        double value = values[i];
        if (value > 0.0)
            values[i] = log(value);
        else if (value <= 0.0)
            values[i] = skirt_source_negative_infinity();
        else
            values[i] = log(value);
    }
}

// Returns the interpolated table value, or for logarithmic quantities its natural logarithm, with -infinity
// representing zero (outside an unclamped axis or at a non-positive corner value).
static __device__ double skirt_source_spectrum_value(
    const double* firstAxis, int firstLength, int firstAxisLog, const double* quantity,
    int quantityLog, int clampFirstAxis, const unsigned long long* cornerBases,
    const double* cornerWeights, int numCorners, double x, int pointIndex, int minRight, int lastPoint)
{
    double zero = quantityLog ? skirt_source_negative_infinity() : 0.0;
    int right;
    double fraction;
    if (pointIndex == 0 || pointIndex == lastPoint)
    {
        right = skirt_lower_bound_double(firstAxis, firstLength, x);
        if (right == 0)
        {
            if (!clampFirstAxis && x != firstAxis[0]) return zero;
            right++;
            x = firstAxis[0];
        }
        else if (right == firstLength)
        {
            if (!clampFirstAxis) return zero;
            right--;
            x = firstAxis[right];
        }
        double x1 = firstAxis[right - 1];
        double x2 = firstAxis[right];
        if (firstAxisLog)
        {
            x = log(x);
            x1 = log(x1);
            x2 = log(x2);
        }
        fraction = (x - x1) / (x2 - x1);
    }
    else
    {
        right = pointIndex + minRight - 1;
        fraction = 1.0;
    }

    double y = 0.0;
    double negativeInfinity = skirt_source_negative_infinity();
    for (int corner = 0; corner != numCorners; ++corner)
    {
        double front = cornerWeights[corner] * fraction;
        if (front)
        {
            double value = quantity[cornerBases[corner] + right];
            if (quantityLog && value == negativeInfinity) return zero;
            y += front * value;
        }
        front = cornerWeights[corner] * (1.0 - fraction);
        if (front)
        {
            double value = quantity[cornerBases[corner] + right - 1];
            if (quantityLog && value == negativeInfinity) return zero;
            y += front * value;
        }
    }
    return y;
}

static __device__ double skirt_source_spectrum_pdf(
    const double* firstAxis, int firstLength, int firstAxisLog, const double* quantity,
    int quantityLog, int clampFirstAxis, const unsigned long long* cornerBases,
    const double* cornerWeights, int numCorners, double x, int pointIndex, int minRight, int lastPoint)
{
    double y = skirt_source_spectrum_value(firstAxis, firstLength, firstAxisLog, quantity, quantityLog,
                                           clampFirstAxis, cornerBases, cornerWeights, numCorners, x, pointIndex,
                                           minRight, lastPoint);
    return quantityLog ? exp(y) : y;
}

// Power-law segment area from the log-pdf difference dy = ln(p1/p0) and lnx = ln(x1/x0); see skirt_powerlaw_area.
static __device__ double skirt_powerlaw_area_logs(double p0, double p1, double x0, double x1, double lnx, double dy)
{
    double q = 1.0 + dy / lnx;
    if (q == 0.0) return p0 * x0 * lnx;
    if (fabs(q) < 1e-3)
    {
        double s = q * lnx;
        return p0 * x0 * (lnx * (1.0 + 0.5 * s + 1.0 / 6.0 * s * s + 1.0 / 24.0 * s * s * s));
    }
    return (x1 * p1 - x0 * p0) / q;
}

// Segment area for row integrals; uses the stored log-pdf and log wavelength ratio when available.
static __device__ double skirt_source_segment_area(int loglog, const double* logPdf, const double* logRatio,
                                                   int interval, double p0, double p1, double x0, double x1)
{
    if (!loglog) return 0.5 * (p0 + p1) * (x1 - x0);
    if (!(p0 > 0.0 && p1 > 0.0)) return 0.0;
    if (logPdf && logRatio)
        return skirt_powerlaw_area_logs(p0, p1, x0, x1, logRatio[interval], logPdf[interval + 1] - logPdf[interval]);
    return skirt_powerlaw_area(p0, p1, x0, x1);
}


extern "C" __global__ void source_spectrum_pdf_rows(
    double* axisValues, int* axisOffsets, int* axisLengths, int* axisLog, double* quantity,
    int numAxes, int quantityLog, int clampFirstAxis, double* parameters, double* wavelengths,
    int numWavelengths, int firstEntity, int numRows, double* pdfScratch,
    const unsigned long long* historyBeginv, double* logScratch, int floatExp)
{
    int row = static_cast<int>(blockIdx.x);
    if (row >= numRows || numAxes < 2 || numAxes > 5 || numWavelengths < 2) return;
    int entity = firstEntity + row;
    if (historyBeginv && historyBeginv[entity] == historyBeginv[entity + 1]) return;

    __shared__ int nonWavelengthRight[4];
    __shared__ double nonWavelengthFraction[4];
    __shared__ unsigned long long cornerBases[16];
    __shared__ double cornerWeights[16];
    int lane = static_cast<int>(threadIdx.x);
    if (lane == 0)
    {
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
            double x1 = axis[right - 1];
            double x2 = axis[right];
            if (axisLog[k])
            {
                value = log(value);
                x1 = log(x1);
                x2 = log(x2);
            }
            nonWavelengthRight[k - 1] = right;
            nonWavelengthFraction[k - 1] = (value - x1) / (x2 - x1);
        }
    }
    __syncthreads();

    int numCorners = 1 << (numAxes - 1);
    if (lane < numCorners)
    {
        int indices[5] = {0, 0, 0, 0, 0};
        int term = lane;
        double weight = 1.0;
        for (int k = 1; k != numAxes; ++k)
        {
            int left = term & 1;
            indices[k] = nonWavelengthRight[k - 1] - left;
            double fraction = nonWavelengthFraction[k - 1];
            weight *= left ? (1.0 - fraction) : fraction;
            term >>= 1;
        }
        cornerBases[lane] = skirt_table_flattened_index(indices, axisLengths, numAxes);
        cornerWeights[lane] = weight;
    }
    __syncthreads();

    const double* firstAxis = axisValues + axisOffsets[0];
    int minRight = skirt_upper_bound_double(firstAxis, axisLengths[0], wavelengths[0]);
    unsigned long long rowOffset = static_cast<unsigned long long>(row) * numWavelengths;
    if (!quantityLog || !logScratch)
    {
        for (int i = lane; i < numWavelengths; i += static_cast<int>(blockDim.x))
            pdfScratch[rowOffset + i] = skirt_source_spectrum_pdf(
                firstAxis, axisLengths[0], axisLog[0], quantity, quantityLog, clampFirstAxis,
                cornerBases, cornerWeights, numCorners, wavelengths[i], i, minRight, numWavelengths - 1);
        return;
    }

    // keep the log-pdf for the segment integrals; with floatExp the pdf is evaluated in FP32 relative to the row
    // maximum, which only rescales the row (the cache stores normalized float/bfloat16 values)
    double localMax = skirt_source_negative_infinity();
    for (int i = lane; i < numWavelengths; i += static_cast<int>(blockDim.x))
    {
        double y = skirt_source_spectrum_value(
            firstAxis, axisLengths[0], axisLog[0], quantity, quantityLog, clampFirstAxis,
            cornerBases, cornerWeights, numCorners, wavelengths[i], i, minRight, numWavelengths - 1);
        logScratch[rowOffset + i] = y;
        localMax = fmax(localMax, y);
        if (!floatExp) pdfScratch[rowOffset + i] = exp(y);
    }
    if (!floatExp) return;
    __shared__ double rowMax[256];
    rowMax[lane] = localMax;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) >> 1; stride; stride >>= 1)
    {
        if (lane < stride) rowMax[lane] = fmax(rowMax[lane], rowMax[lane + stride]);
        __syncthreads();
    }
    double shift = rowMax[0] > skirt_source_negative_infinity() ? rowMax[0] : 0.0;
    for (int i = lane; i < numWavelengths; i += static_cast<int>(blockDim.x))
    {
        double y = logScratch[rowOffset + i] - shift;
        logScratch[rowOffset + i] = y;
        pdfScratch[rowOffset + i] = static_cast<double>(expf(static_cast<float>(y)));
    }
}

extern "C" __global__ void source_spectrum_luminosity_rows(
    const double* pdfScratch, const double* wavelengths, int numWavelengths, int numRows,
    int firstEntity, int loglog, const double* scaleValues, double* luminosities,
    const double* logScratch, const double* logRatio)
{
    int row = static_cast<int>(blockIdx.x);
    if (row >= numRows) return;

    int lane = static_cast<int>(threadIdx.x);
    unsigned long long scratchOffset = static_cast<unsigned long long>(row) * numWavelengths;
    double norm = 0.0;
    for (int interval = lane; interval + 1 < numWavelengths; interval += static_cast<int>(blockDim.x))
    {
        double leftPdf = pdfScratch[scratchOffset + interval];
        double rightPdf = pdfScratch[scratchOffset + interval + 1];
        double leftX = wavelengths[interval];
        double rightX = wavelengths[interval + 1];
        norm += skirt_source_segment_area(loglog, logScratch ? logScratch + scratchOffset : nullptr, logRatio,
                                          interval, leftPdf, rightPdf, leftX, rightX);
    }

    __shared__ double reduction[256];
    reduction[lane] = norm;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) >> 1; stride; stride >>= 1)
    {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    if (lane == 0)
    {
        int entity = firstEntity + row;
        luminosities[entity] = reduction[0] * scaleValues[entity];
    }
}

static __device__ unsigned short skirt_source_bfloat16_rne(float value)
{
    unsigned int bits = __float_as_uint(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<unsigned short>(bits >> 16);
}

extern "C" __global__ void source_spectrum_normalize_rows(
    double* pdfScratch, double* wavelengths, int numWavelengths, int numRows, int firstEntity,
    int loglog, double* cdfScratch, float* pdfOutput, unsigned short* pdfBfloat16Output,
    float* cdfOutput, int* status, const unsigned long long* historyBeginv,
    const double* logScratch, const double* logRatio)
{
    int row = static_cast<int>(blockIdx.x);
    if (row >= numRows) return;
    if (numWavelengths < 2 || blockDim.x > 256 || (!pdfOutput && !pdfBfloat16Output) || !cdfOutput)
    {
        if (threadIdx.x == 0 && status) atomicExch(status, 0);
        return;
    }

    int lane = static_cast<int>(threadIdx.x);
    int entityIndex = firstEntity + row;
    unsigned long long scratchOffset = static_cast<unsigned long long>(row) * numWavelengths;
    unsigned long long entity = static_cast<unsigned long long>(firstEntity) + static_cast<unsigned int>(row);
    unsigned long long outputOffset = entity * static_cast<unsigned int>(numWavelengths);
    if (historyBeginv && historyBeginv[entityIndex] == historyBeginv[entityIndex + 1])
    {
        for (int i = lane; i < numWavelengths; i += static_cast<int>(blockDim.x))
        {
            if (pdfBfloat16Output)
                pdfBfloat16Output[outputOffset + i] = 0;
            else
                pdfOutput[outputOffset + i] = 0.0f;
            cdfOutput[outputOffset + i] = 0.0f;
        }
        return;
    }
    __shared__ double scan[256];
    __shared__ double carry;

    if (lane == 0)
    {
        carry = 0.0;
        cdfScratch[scratchOffset] = 0.0;
    }
    __syncthreads();

    int numIntervals = numWavelengths - 1;
    for (int tile = 0; tile < numIntervals; tile += static_cast<int>(blockDim.x))
    {
        int interval = tile + lane;
        int tileCount = numIntervals - tile;
        if (tileCount > static_cast<int>(blockDim.x)) tileCount = static_cast<int>(blockDim.x);

        double area = 0.0;
        if (interval < numIntervals)
        {
            double leftPdf = pdfScratch[scratchOffset + interval];
            double rightPdf = pdfScratch[scratchOffset + interval + 1];
            double leftX = wavelengths[interval];
            double rightX = wavelengths[interval + 1];
            area = skirt_source_segment_area(loglog, logScratch ? logScratch + scratchOffset : nullptr, logRatio,
                                             interval, leftPdf, rightPdf, leftX, rightX);
            if (!isfinite(area) && status) atomicExch(status, 0);
        }
        scan[lane] = area;
        __syncthreads();

        for (int offset = 1; offset < static_cast<int>(blockDim.x); offset <<= 1)
        {
            double prefix = scan[lane];
            if (lane >= offset) prefix += scan[lane - offset];
            __syncthreads();
            scan[lane] = prefix;
            __syncthreads();
        }

        double tileCarry = carry;
        if (interval < numIntervals) cdfScratch[scratchOffset + interval + 1] = tileCarry + scan[lane];
        __syncthreads();
        if (lane == 0) carry = tileCarry + scan[tileCount - 1];
        __syncthreads();
    }

    double norm = carry;
    if (!isfinite(norm) && status) atomicExch(status, 0);
    bool normalize = norm > 0.0;
    for (int i = lane; i < numWavelengths; i += static_cast<int>(blockDim.x))
    {
        double pdf = pdfScratch[scratchOffset + i];
        double cdf = cdfScratch[scratchOffset + i];
        if (normalize)
        {
            pdf /= norm;
            cdf /= norm;
        }
        if ((!isfinite(pdf) || !isfinite(cdf)) && status) atomicExch(status, 0);

        float pdfFloat = static_cast<float>(pdf);
        float cdfFloat = static_cast<float>(cdf);
        if ((!isfinite(pdfFloat) || !isfinite(cdfFloat)) && status) atomicExch(status, 0);
        if (pdfBfloat16Output)
        {
            unsigned short packed = skirt_source_bfloat16_rne(pdfFloat);
            if ((packed & 0x7f80U) == 0x7f80U && status) atomicExch(status, 0);
            pdfBfloat16Output[outputOffset + i] = packed;
        }
        else
            pdfOutput[outputOffset + i] = pdfFloat;
        cdfOutput[outputOffset + i] = cdfFloat;
    }
}
)cuda";
}

#endif
