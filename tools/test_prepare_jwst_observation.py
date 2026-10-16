"""Check local and cosmological observer distances in derived configurations."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from prepare_jwst_observation import main


class ObservationPreparationTests(unittest.TestCase):
    def prepare(self, redshift, distance):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.ski"
            output = Path(directory) / "jwst.ski"
            base.write_text('''<skirt-simulation-hierarchy><MonteCarloSimulation>
                <Random/><cosmology><LocalUniverseCosmology/></cosmology>
                <SourceSystem/><MediumSystem/><InstrumentSystem>
                <defaultWavelengthGrid/><instruments><FrameInstrument instrumentName="xy"
                distance="10 Mpc" inclination="0 deg" azimuth="0 deg" roll="0 deg">
                <wavelengthGrid/></FrameInstrument></instruments></InstrumentSystem>
                </MonteCarloSimulation></skirt-simulation-hierarchy>''')
            argv = ["prepare", "--base-ski", str(base), "--output", str(output),
                    "--redshift", str(redshift), "--distance-mpc", str(distance)]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            return ET.parse(output).getroot(), json.loads(output.with_suffix(".manifest.json").read_text())

    def test_local_frame_and_sed_use_finite_distance(self):
        root, report = self.prepare(0, 20)
        self.assertEqual(root.find(".//FrameInstrument").get("distance"), "20 Mpc")
        self.assertEqual(root.find(".//SEDInstrument").get("distance"), "20 Mpc")
        self.assertEqual(report["fiducial_distance_mpc"], 20)

    def test_positive_redshift_uses_cosmological_distance(self):
        root, report = self.prepare(1, 20)
        self.assertEqual(root.find(".//FrameInstrument").get("distance"), "0 Mpc")
        self.assertEqual(root.find(".//SEDInstrument").get("distance"), "0 Mpc")
        self.assertIsNone(report["fiducial_distance_mpc"])

    def test_rejects_invalid_local_distance(self):
        for distance in (0, -1, float("nan"), float("inf")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.prepare(0, distance)


if __name__ == "__main__":
    unittest.main()
