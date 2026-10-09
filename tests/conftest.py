import json

import numpy as np
import pytest


@pytest.fixture
def make_capture(tmp_path):
    def create(samples=None, payload=None, **options):
        raw = tmp_path / "tone.iq"
        if payload is None:
            components = np.empty((len(samples), 2), dtype="<f4")
            components[:, 0], components[:, 1] = samples.real, samples.imag
            payload = components.tobytes()
        raw.write_bytes(payload)
        descriptor = tmp_path / "tone.iq.json"
        document = {"dataset": raw.name, "format": "cf32", "sample_rate_hz": 8192,
                    "n_fft": 256, "hop_length": 128, "max_samples": 4096}
        document.update(options)
        descriptor.write_text(json.dumps(document), encoding="utf-8")
        return descriptor, raw
    return create


@pytest.fixture
def switching_tone():
    indices = np.arange(4352)
    return np.where(indices < 2304,
                    0.5 * np.exp(2j * np.pi * 1024 * indices / 8192),
                    0.25 * np.exp(-2j * np.pi * 768 * indices / 8192)).astype(np.complex64)
