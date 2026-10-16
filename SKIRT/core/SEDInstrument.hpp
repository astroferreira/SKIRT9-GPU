/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#ifndef SEDINSTRUMENT_HPP
#define SEDINSTRUMENT_HPP

#include "ApertureInstrument.hpp"
#include "Position.hpp"
struct GpuSedDetectorInfo;

////////////////////////////////////////////////////////////////////

/** An SEDInstrument object represents a distant instrument with an optional circular aperture that
    records the spatially integrated flux density for each wavelength and outputs an %SED text
    column file. */
class SEDInstrument : public ApertureInstrument
{
    ITEM_CONCRETE(SEDInstrument, ApertureInstrument,
                  "a distant instrument that outputs the spatially integrated flux density as an SED")
    ITEM_END()

    //============= Construction - Setup - Destruction =============

protected:
    /** This function configures the FluxRecorder instance associated with this instrument. */
    void setupSelfBefore() override;

    //======================== Other Functions =======================

public:
    /** This function simulates the detection of a photon packet by the instrument. It verifies
        that the arriving photon packet projects within the aperture and then calls the detect()
        function of the FluxRecorder instance associated with this instrument. */
    void detect(PhotonPacket* pp) override;

    /** This function detects a batch of photon packets for total-only SED outputs. It returns
        false if the associated flux recorder requires the generic scalar detect() path. */
    bool detectTotalBatch(const vector<PhotonPacket*>& ppv);

    /** This function detects a batch of direct luminosity contributions for total-only SED
        outputs. It returns false if the associated flux recorder requires the generic scalar
        detect() path. */
    bool detectTotalBatch(const vector<Position>& positionv, const vector<double>& wavelengthv,
                          const vector<double>& luminosityv, const vector<double>& tauv);

    /** Fills a descriptor for the fused GPU observer/detector path. */
    bool getGpuDetectorInfo(GpuSedDetectorInfo& info);
};

////////////////////////////////////////////////////////////////////

#endif
