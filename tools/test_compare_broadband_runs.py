import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parent))
from compare_broadband_runs import canonical_model, compare, metrics


XML = '''<skirt producer="SKIRT build" time="2026-01-01T00:00:00">
<MonteCarloSimulation packets="10"/><instruments><FrameInstrument instrumentName="xy" numPixelsX="2" numPixelsY="2">
<wavelengthGrid><ConfigurableBandWavelengthGrid><bands>
<BroadBand bandName="first"/><BroadBand bandName="second"/>
</bands></ConfigurableBandWavelengthGrid></wavelengthGrid>
</FrameInstrument></instruments></skirt>'''

FILE_BAND_XML = '''<skirt producer="SKIRT build" time="2026-01-01T00:00:00">
<instruments><defaultWavelengthGrid><ConfigurableBandWavelengthGrid><bands>
<FileBand filename="cfht_u.dat"/>
</bands></ConfigurableBandWavelengthGrid></defaultWavelengthGrid>
<FrameInstrument instrumentName="xy" numPixelsX="2" numPixelsY="2"/>
</instruments></skirt>'''


def write_run(directory, data, producer="SKIRT build", time="2026-01-01T00:00:00", simulation=10.0, xml=XML):
    directory = Path(directory)
    directory.mkdir()
    (directory / "optical_parameters.xml").write_text(
        xml.replace("SKIRT build", producer).replace("2026-01-01T00:00:00", time))
    (directory / "optical_log.txt").write_text(
        "Finished setup in 2.0 s.\nFinished primary emission in 4.0 s.\n"
        "Finished secondary emission in 1.0 s.\n"
        f"Finished simulation optical using 1 thread in {simulation} s.\n")
    primary = fits.PrimaryHDU(np.asarray(data, dtype=np.float32))
    primary.header["BUNIT"] = "W m-2"
    for axis in (1, 2):
        primary.header[f"CUNIT{axis}"] = "deg"
        primary.header[f"CTYPE{axis}"] = "LINEAR"
        primary.header[f"CRPIX{axis}"] = 1.0
        primary.header[f"CRVAL{axis}"] = 0.0
        primary.header[f"CDELT{axis}"] = 0.1
    primary.header["CUNIT3"] = "micron"
    wave = fits.BinTableHDU.from_columns([fits.Column(name="GRID_POINTS", format="D",
                                                       array=np.arange(1, data.shape[0] + 1, dtype=float))])
    fits.HDUList([primary, wave]).writeto(directory / "optical_xy_total.fits")


class BroadbandComparisonTests(unittest.TestCase):
    def test_metrics_rejects_invalid_or_empty_bands(self):
        for reference, candidate in (([0], [0]), ([1], [-1]), ([1], [np.nan]), ([1], [1, 2])):
            with self.assertRaises(ValueError):
                metrics(reference, candidate)

    def test_canonical_model_ignores_writer_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "one.xml", Path(temp) / "two.xml"
            first.write_text(XML)
            second.write_text(XML.replace("SKIRT build", "other build").replace("2026-01-01T00:00:00", "2026-02-02T00:00:00"))
            self.assertEqual(canonical_model(first), canonical_model(second))

    def test_compare_reports_bands_and_speedups(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = np.array([[[1, 2], [3, 4]], [[2, 2], [2, 2]]])
            write_run(root / "reference", data, simulation=10)
            write_run(root / "candidate", data * 1.1, producer="other", time="2026-02-01T00:00:00", simulation=5)
            report = compare(root / "reference", root / "candidate")
            self.assertEqual(len(report["comparisons"]), 2)
            self.assertAlmostEqual(report["max_errors"]["relative_sum_error"], 0.1)
            self.assertEqual(report["stage_speedups_reference_over_candidate"]["simulation"], 2)
            self.assertEqual(report["reference"]["timings"]["setup"], 2.0)
            self.assertEqual(report["stage_speedups_reference_over_candidate"]["setup"], 1.0)
            self.assertFalse(report["noise_parity_certificate"])

    def test_cli_only_enforces_tolerance_when_requested(self):
        with tempfile.TemporaryDirectory() as temp:
            root, tool = Path(temp), Path(__file__).with_name("compare_broadband_runs.py")
            data = np.ones((2, 2, 2))
            write_run(root / "reference", data)
            write_run(root / "candidate", data * 1.1)
            output = root / "report.json"
            command = [sys.executable, str(tool), "--reference", str(root / "reference"),
                       "--candidate", str(root / "candidate"), "--output", str(output)]
            self.assertEqual(subprocess.run(command, capture_output=True, text=True).returncode, 0)
            self.assertIsNone(json.loads(output.read_text())["passes"])
            self.assertNotEqual(subprocess.run(command + ["--require-identical", "0.01"],
                                               capture_output=True, text=True).returncode, 0)

    def test_compare_rejects_wavelength_or_header_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = np.ones((2, 2, 2))
            write_run(root / "reference", data)
            write_run(root / "candidate", data)
            with fits.open(root / "candidate" / "optical_xy_total.fits", mode="update") as hdus:
                hdus[0].header["BUNIT"] = "Jy"
            with self.assertRaisesRegex(ValueError, "coordinate/unit mismatch"):
                compare(root / "reference", root / "candidate")

    def test_compare_rejects_shape_outside_frame_declaration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = np.ones((2, 3, 2))
            write_run(root / "reference", data)
            write_run(root / "candidate", data)
            with self.assertRaisesRegex(ValueError, "FrameInstrument declaration"):
                compare(root / "reference", root / "candidate")

    def test_inherited_file_band_grid_is_a_valid_broadband_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = np.ones((1, 2, 2))
            write_run(root / "reference", data, xml=FILE_BAND_XML)
            write_run(root / "candidate", data, xml=FILE_BAND_XML)
            report = compare(root / "reference", root / "candidate")
            self.assertEqual(report["comparisons"][0]["band"], "cfht_u.dat")


if __name__ == "__main__":
    unittest.main()
