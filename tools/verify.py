"""Run focused tests and record reproducible numerical measurements."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
RUNTIME.mkdir(exist_ok=True)
os.chdir(ROOT)
for key in ("TEMP", "TMP", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "MPLCONFIGDIR",
            "NUMBA_CACHE_DIR", "JUPYTER_CONFIG_DIR", "JUPYTER_RUNTIME_DIR", "IPYTHONDIR"):
    os.environ[key] = str(RUNTIME / key.lower())
    Path(os.environ[key]).mkdir(exist_ok=True)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["NAPARI_CONFIG"] = str(RUNTIME / "napari.yaml")
os.environ["NAPARI_ASYNC"] = "0"
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
tempfile.tempdir = str(RUNTIME)

import numpy as np
import pytest

from napari_iq_reader import read_descriptor


class Results:
    def __init__(self):
        self.tests = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.outcome != "passed":
            item = {"test": report.nodeid, "phase": report.when, "outcome": report.outcome,
                    "seconds": report.duration}
            if report.failed or report.skipped:
                item["detail"] = str(report.longrepr)
            self.tests.append(item)


def main():
    results = Results()
    started = time.perf_counter()
    status = pytest.main(["-q", "-p", "no:cacheprovider", "--basetemp",
                          str(RUNTIME / "pytest"), "tests"], plugins=[results])
    elapsed = time.perf_counter() - started
    fixtures = []
    for path in sorted((ROOT / "examples").glob("*.iq.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        dataset = (path.parent / doc["dataset"]).resolve()
        if not dataset.is_file():
            fixtures.append({"file": dataset.name, "available": False,
                             "status": "optional fixture not present"})
            continue
        start = time.perf_counter()
        data, attrs, _ = read_descriptor(path)[0]
        duration = time.perf_counter() - start
        meta = attrs["metadata"]
        fixtures.append({
            "file": dataset.name, "source_bytes": dataset.stat().st_size,
            "source_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "shape": list(data.shape), "dtype": str(data.dtype),
            "all_finite": bool(np.isfinite(data).all()), "seconds": duration,
            "samples_read": meta["samples_read"], "payload_bytes_read": meta["payload_bytes_read"],
            "frequency_reference": meta["frequency_reference"],
            "frequency_first_last_hz": meta["frequencies_hz"][[0, -1]].tolist(),
            "frame_center_first_last_seconds": meta["frame_centers_seconds"][[0, -1]].tolist(),
            "scale_MHz_ms": list(attrs["scale"]), "translate_MHz_ms": list(attrs["translate"]),
            "sample_rate_basis": meta["sample_rate_basis"],
        })
    tone_dir = RUNTIME / "measured-tone"
    tone_dir.mkdir(exist_ok=True)
    indices = np.arange(4352)
    samples = np.where(indices < 2304,
                       0.5 * np.exp(2j * np.pi * 1024 * indices / 8192),
                       0.25 * np.exp(-2j * np.pi * 768 * indices / 8192))
    pairs = np.empty((len(samples), 2), dtype="<f4")
    pairs[:, 0], pairs[:, 1] = samples.real, samples.imag
    (tone_dir / "tone.iq").write_bytes(pairs.tobytes())
    descriptor = tone_dir / "tone.iq.json"
    descriptor.write_text(json.dumps({"dataset": "tone.iq", "format": "cf32",
                                      "sample_rate_hz": 8192, "start_sample": 256,
                                      "max_samples": 4096, "n_fft": 256,
                                      "hop_length": 128}), encoding="utf-8")
    data, attrs, _ = read_descriptor(descriptor)[0]
    meta = attrs["metadata"]
    rows = [int(np.argmax(data[:, 0])), int(np.argmax(data[:, -1]))]
    original = ROOT.parent / "rf-ocusync-classifier/capture_io.py"
    vendored = ROOT / "src/napari_iq_reader/_capture_io.py"
    original_available = original.is_file()
    viewer_tests = [t for t in results.tests if "test_viewer.py" in t["test"]]
    report = {
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pytest", "npe2", "napari", "PyQt6")},
        "test_exit_code": int(status), "test_seconds": elapsed,
        "test_counts": {name: sum(t["outcome"] == name for t in results.tests)
                        for name in ("passed", "failed", "skipped")},
        "tests": results.tests,
        "viewer_status": "passed" if len(viewer_tests) == 2 and all(
            t["outcome"] == "passed" for t in viewer_tests) else "unavailable or failed; GUI unverified",
        "viewer_scope": "Viewer construction attempted; Viewer.open verified only if both Viewer tests pass",
        "headless_integration_scope": "installed npe2 dispatch on four real fixtures; actual napari Image units and data_to_world",
        "unverified": ["interactive screen rendering", "OpenGL screenshot rendering",
                       "other OS and Python versions", "independent real-capture rate calibration",
                       "physical SDR acquisition", "RF payload decoding", "classification accuracy"],
        "bounds": {"max_samples": 262144, "max_descriptor_bytes": 16384,
                   "max_transform_estimate_bytes": 33554432,
                   "allocation_estimate_is_not_measured_peak_rss": True,
                   "fixture_hashing_reads_whole_files_outside_plugin": True},
        "tone": {"shape": list(data.shape), "peak_rows_first_last": rows,
                 "expected_peak_frequencies_hz": [1024, -768],
                 "measured_peak_frequencies_hz": [float(meta["frequencies_hz"][row]) for row in rows],
                 "expected_peak_db": [float(20 * np.log10(0.5)), float(20 * np.log10(0.25))],
                 "measured_peak_db": [float(data[rows[0], 0]), float(data[rows[1], -1])],
                 "frame_center_first_last_seconds": meta["frame_centers_seconds"][[0, -1]].tolist()},
        "fixtures": fixtures,
        "provenance": {"source_file_available": original_available,
                       "source_module_sha256": hashlib.sha256(original.read_bytes()).hexdigest() if original_available else None,
                       "vendored_module_sha256": hashlib.sha256(vendored.read_bytes()).hexdigest(),
                       "normalized_text_equal": original.read_text(encoding="utf-8") == vendored.read_text(encoding="utf-8") if original_available else None},
    }
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("test_counts", "viewer_status", "test_seconds")}, indent=2))
    return int(status)


if __name__ == "__main__":
    raise SystemExit(main())
