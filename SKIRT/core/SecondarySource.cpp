/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#include "SecondarySource.hpp"
#include "PhotonPacket.hpp"

////////////////////////////////////////////////////////////////////

SecondarySource::SecondarySource(SimulationItem* parent)
{
    parent->addChild(this);

    // because virtual functions don't work properly from with a constructor,
    // this calls setupSelfBefore/After in base classes but NOT in subclasses
    setup();
}

////////////////////////////////////////////////////////////////////

void SecondarySource::launchBatch(PhotonPacket* ppv, size_t firstHistoryIndex, size_t numPackets, double L) const
{
    for (size_t i = 0; i != numPackets; ++i) launch(&ppv[i], firstHistoryIndex + i, L);
}

////////////////////////////////////////////////////////////////////
