/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       Copyright 2026 Astronomical Observatory, Ghent University ////
///////////////////////////////////////////////////////////////// */

#ifndef GPUSOURCELAUNCHDATA_HPP
#define GPUSOURCELAUNCHDATA_HPP

#include "Basics.hpp"
#include "GpuSourceSpectrumTable.hpp"

////////////////////////////////////////////////////////////////////

/** Describes a prepared source for GPU packet launch. The wavelength, PDF, and CDF arrays use a
    common wavelength grid and store one contiguous row per entity unless spectrumRowIndexv maps
    entities to compact rows. Exactly one of pdfv and pdfBfloat16v is populated for host-prepared
    spectra. Alternatively, spectrumTable describes imported spectra generated in bounded device
    cache windows, with all three host spectral arrays empty. Imported-particle sources use
    particleCenterSizev; dust sources use
    cellIndexv and cellExtentv. */
struct GpuSourceLaunchData
{
    unsigned long long cacheToken{0};
    size_t generation{0};
    unsigned long long randomSeed{0};
    vector<unsigned long long> historyBeginv;
    vector<double> luminosityWeightv;
    vector<double> particleCenterSizev;
    vector<int> cellIndexv;
    vector<double> cellExtentv;
    vector<double> wavelengthv;
    vector<float> pdfv;
    vector<unsigned short> pdfBfloat16v;
    vector<float> cdfv;
    // Imported spectra may be generated in the resident device cache instead of uploaded as dense rows.
    GpuSourceSpectrumTable spectrumTable;
    bool tablePdfBfloat16{false};
    vector<int> validationEntityv;
    vector<float> validationPdfv;
    vector<float> validationCdfv;
    // Dust-only indirection; -1 is allowed only for entities assigned no packet histories.
    // Geometry and launch ordering remain dense, including non-emitting absorbing cells.
    vector<int> spectrumRowIndexv;
    double wavelengthBias{0.};
    double biasLogMin{0.};
    double biasLogWidth{0.};

    bool hasValidSpectrumLayout(bool checkHistories = true) const
    {
        size_t entities = luminosityWeightv.size(), wavelengths = wavelengthv.size();
        if (!entities || wavelengths < 2 || historyBeginv.size() != entities + 1)
            return false;
        bool table = spectrumTable.values != nullptr;
        size_t rows = entities;
        if (table)
        {
            size_t axes = spectrumTable.axes.size();
            if (axes < 2 || axes > 5 || spectrumTable.axisSizes.size() != axes
                || spectrumTable.axisLog.size() != axes || !spectrumTable.quantityStride
                || spectrumTable.parameters.size() != entities * (axes - 1)
                || !pdfv.empty() || !pdfBfloat16v.empty() || !cdfv.empty()
                || !spectrumRowIndexv.empty() || !cellIndexv.empty()
                || particleCenterSizev.size() != 4 * entities)
                return false;
            for (size_t axis = 0; axis != axes; ++axis)
                if (!spectrumTable.axes[axis] || spectrumTable.axisSizes[axis] < 2) return false;
        }
        else
        {
            if (cdfv.empty() || cdfv.size() % wavelengths) return false;
            rows = cdfv.size() / wavelengths;
            if (!((pdfv.size() == cdfv.size() && pdfBfloat16v.empty())
                  || (pdfv.empty() && pdfBfloat16v.size() == cdfv.size())))
                return false;
        }
        if (spectrumRowIndexv.empty())
        {
            if (rows != entities) return false;
        }
        else if (spectrumRowIndexv.size() != entities || cellIndexv.size() != entities
                 || !particleCenterSizev.empty() || rows > entities)
            return false;
        if (!checkHistories) return true;
        for (size_t p = 0; p != entities; ++p)
        {
            if (historyBeginv[p] > historyBeginv[p + 1]) return false;
            if (!spectrumRowIndexv.empty())
            {
                int row = spectrumRowIndexv[p];
                if (row < -1 || (row >= 0 && static_cast<size_t>(row) >= rows)
                    || (row < 0 && historyBeginv[p] != historyBeginv[p + 1]))
                    return false;
            }
        }
        return true;
    }
};

////////////////////////////////////////////////////////////////////

#endif
