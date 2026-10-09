"""Adapt a bounded capture transform to npe2 layer data."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ._capture_io import CaptureError, compute_spectrogram, open_capture

MAX_DESCRIPTOR_BYTES = 16_384
MAX_SAMPLES = 262_144
MAX_OUTPUT_BYTES = 32 * 1024 * 1024


def _integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise CaptureError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CaptureError(f"duplicate descriptor key: {key}")
        result[key] = value
    return result


def _load_descriptor(path):
    path = Path(path)
    if not path.name.endswith(".iq.json"):
        raise CaptureError("expected an .iq.json descriptor")
    with path.open("rb") as stream:
        payload = stream.read(MAX_DESCRIPTOR_BYTES + 1)
    if len(payload) > MAX_DESCRIPTOR_BYTES:
        raise CaptureError("descriptor exceeds 16384 bytes")
    try:
        document = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise CaptureError("invalid I/Q descriptor JSON") from error
    allowed = {"dataset", "format", "sample_rate_hz", "center_frequency_hz", "endian",
               "start_sample", "max_samples", "n_fft", "hop_length", "sample_rate_basis"}
    if not isinstance(document, dict) or document.keys() - allowed:
        raise CaptureError("descriptor must be an object with supported keys")
    dataset = document.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        raise CaptureError("dataset must be a file path")
    if document.get("format") not in ("cf32", "ci16"):
        raise CaptureError("format must be cf32 or ci16")
    rate = document.get("sample_rate_hz")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 < rate <= 1e12 or not math.isfinite(rate):
        raise CaptureError("sample_rate_hz must be finite and positive, at most 1e12")
    center = document.get("center_frequency_hz")
    if center is not None and (isinstance(center, bool) or not isinstance(center, (int, float))
                              or not 0 <= center <= 1e308 or not math.isfinite(center)):
        raise CaptureError("center_frequency_hz must be finite and nonnegative")
    if document.get("endian", "little") not in ("little", "big"):
        raise CaptureError("endian must be little or big")
    basis = document.get("sample_rate_basis", "user supplied; acquisition accuracy unverified")
    if not isinstance(basis, str):
        raise CaptureError("sample_rate_basis must be a string")
    n_fft = _integer(document.get("n_fft", 1024), "n_fft", 16, 8192)
    hop = _integer(document.get("hop_length", n_fft // 2), "hop_length", 1, n_fft)
    limit = _integer(document.get("max_samples", MAX_SAMPLES), "max_samples", 1, MAX_SAMPLES)
    start = _integer(document.get("start_sample", 0), "start_sample", 0, 2**53 - 1)
    if limit < n_fft:
        raise CaptureError("max_samples must allow one complete FFT frame")
    return path, document, n_fft, hop, limit, start, basis


def get_reader(path):
    """Probe only a descriptor; refuse stacks and unrelated paths without raising."""
    if not isinstance(path, (str, Path)):
        return None
    try:
        _load_descriptor(path)
    except (ValueError, OSError, TypeError, OverflowError):
        return None
    return read_descriptor


def read_descriptor(path):
    """Return one scalar image: rows descend in frequency, columns advance in time."""
    path, doc, n_fft, hop, limit, start, basis = _load_descriptor(path)
    capture = open_capture(path.parent / doc["dataset"], format=doc["format"],
                           sample_rate=doc["sample_rate_hz"],
                           center_frequency=doc.get("center_frequency_hz"),
                           endian=doc.get("endian", "little"))
    available = max(0, capture.sample_count - start)
    selected = min(limit, available)
    if selected < n_fft:
        raise CaptureError("selected capture has fewer than n_fft samples")
    frames = 1 + (selected - n_fft) // hop
    estimate = n_fft * frames * 32 + (n_fft + frames) * 8 + n_fft * 64
    if estimate > MAX_OUTPUT_BYTES:
        raise CaptureError("spectrogram exceeds the 32 MiB transform budget")
    samples = capture.read_samples(start_sample=start, max_samples=limit)
    result = compute_spectrogram(samples, capture.metadata.sample_rate, n_fft=n_fft,
                                 hop_length=hop, window="hamming",
                                 center_frequency=capture.metadata.center_frequency,
                                 max_output_bytes=MAX_OUTPUT_BYTES)
    frequencies = result.frequencies_hz[::-1].copy()
    times = result.times_seconds + start / capture.metadata.sample_rate
    data = np.ascontiguousarray(result.power_db[::-1], dtype=np.float32)
    metadata = {
        "axes": ["frequency", "time"],
        "axis_units": ["Hz", "s"],
        "frequencies_hz": frequencies,
        "frame_centers_seconds": times,
        "value_units": "dB relative to unit complex amplitude squared",
        "power_definition": "10*log10(max(abs(FFT(x*hamming)/sum(hamming))**2, 1e-12))",
        "frequency_reference": "baseband" if capture.metadata.center_frequency is None else "absolute",
        "sample_rate_hz": capture.metadata.sample_rate,
        "sample_rate_basis": basis,
        "center_frequency_hz": capture.metadata.center_frequency,
        "start_sample": start,
        "samples_read": len(samples),
        "payload_bytes_read": len(samples) * capture.metadata.bytes_per_sample,
        "source_sample_count": capture.sample_count,
        "limited": len(samples) < available,
        "n_fft": n_fft,
        "hop_length": hop,
        "discarded_tail_samples": len(samples) - ((frames - 1) * hop + n_fft),
        "transform_budget_bytes": MAX_OUTPUT_BYTES,
        "validation_scope": "whole-file byte alignment; selected sample values only",
    }
    attributes = {
        "name": f"{Path(doc['dataset']).name}: relative power (dB)",
        "rgb": False,
        "colormap": "viridis",
        "axis_labels": ("Frequency (MHz)", "Time (ms)"),
        "units": ("MHz", "ms"),
        "scale": (-capture.metadata.sample_rate / n_fft / 1e6,
                  hop / capture.metadata.sample_rate * 1e3),
        "translate": (float(frequencies[0]) / 1e6, float(times[0]) * 1e3),
        "metadata": metadata,
    }
    return [(data, attributes, "image")]
