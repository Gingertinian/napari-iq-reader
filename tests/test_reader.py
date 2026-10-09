import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from napari_iq_reader import get_reader, read_descriptor
from napari_iq_reader._capture_io import CaptureError, CaptureMetadata, iter_iq
from napari_iq_reader.reader import MAX_SAMPLES

ROOT = Path(__file__).resolve().parents[1]


def test_signed_tone_axes_time_and_power(make_capture, switching_tone):
    path, _ = make_capture(switching_tone, start_sample=256)
    data, attributes, kind = read_descriptor(path)[0]
    meta = attributes["metadata"]
    assert kind == "image" and data.dtype == np.float32 and data.flags.c_contiguous
    assert data.shape == (256, 31) and attributes["rgb"] is False
    assert np.all(np.diff(meta["frequencies_hz"]) < 0)
    assert np.all(np.diff(meta["frame_centers_seconds"]) > 0)
    first, last = np.argmax(data[:, 0]), np.argmax(data[:, -1])
    assert meta["frequencies_hz"][first] == 1024
    assert meta["frequencies_hz"][last] == -768
    assert first < last
    np.testing.assert_allclose(data[first, 0], 20 * np.log10(0.5), atol=1e-5)
    np.testing.assert_allclose(data[last, -1], 20 * np.log10(0.25), atol=1e-5)
    np.testing.assert_allclose(meta["frame_centers_seconds"][[0, -1]], [384 / 8192, 4224 / 8192])
    for row, column in [(first, 0), (last, 30)]:
        world = np.array(attributes["translate"]) + np.array(attributes["scale"]) * [row, column]
        np.testing.assert_allclose(world, [meta["frequencies_hz"][row] / 1e6,
                                          meta["frame_centers_seconds"][column] * 1e3])
    assert attributes["units"] == ("MHz", "ms")
    assert meta["samples_read"] == 4096 and meta["payload_bytes_read"] == 32768


def test_absolute_frequency(make_capture):
    path, _ = make_capture(np.ones(512, dtype=np.complex64), center_frequency_hz=5745000000)
    data, attrs, _ = read_descriptor(path)[0]
    meta = attrs["metadata"]
    assert meta["frequency_reference"] == "absolute"
    assert meta["frequencies_hz"][np.argmax(data[:, 0])] == 5745000000


@pytest.mark.parametrize("payload", [b"", b"x", b"1234", b"123456789"])
def test_malformed_raw(make_capture, payload):
    path, _ = make_capture(payload=payload)
    with pytest.raises(CaptureError):
        read_descriptor(path)


def test_nonfinite_selected_samples(make_capture):
    values = np.ones(4096, dtype=np.complex64)
    values[99] = np.nan
    path, _ = make_capture(values)
    with pytest.raises(CaptureError, match="NaN"):
        read_descriptor(path)


@pytest.mark.parametrize("options", [
    {"sample_rate_hz": 0}, {"sample_rate_hz": float("nan")},
    {"sample_rate_hz": True}, {"format": "guess"}, {"endian": "native"},
    {"max_samples": MAX_SAMPLES + 1}, {"max_samples": True}, {"max_samples": 1},
    {"n_fft": 8193}, {"hop_length": 0}, {"hop_length": 257},
    {"start_sample": -1}, {"center_frequency_hz": -1}, {"unrecognized": 3},
    {"sample_rate_hz": 10**400}, {"center_frequency_hz": 10**400},
])
def test_descriptor_validation(make_capture, options):
    path, _ = make_capture(np.ones(4096, dtype=np.complex64), **options)
    assert get_reader(path) is None
    with pytest.raises(CaptureError):
        read_descriptor(path)


@pytest.mark.parametrize("text", ['[]', '{"dataset":', '{"dataset":"a","dataset":"b"}'])
def test_invalid_json(tmp_path, text):
    path = tmp_path / "bad.iq.json"
    path.write_text(text, encoding="utf-8")
    assert get_reader(path) is None
    with pytest.raises(CaptureError):
        read_descriptor(path)


def test_descriptor_size_bound(tmp_path):
    path = tmp_path / "oversized.iq.json"
    path.write_bytes(b" " * 16385)
    assert get_reader(path) is None
    with pytest.raises(CaptureError, match="16384"):
        read_descriptor(path)


def test_probe_does_not_read_payload(make_capture):
    path, raw = make_capture(np.ones(4096, dtype=np.complex64))
    actual = Path.open
    def guarded(file, *args, **kwargs):
        if file == raw:
            raise AssertionError("probe read raw payload")
        return actual(file, *args, **kwargs)
    with patch.object(Path, "open", guarded):
        assert get_reader(str(path)) is read_descriptor
    for unsupported in [raw, [str(path)], 1, None, path.parent, path.parent / "missing.iq.json"]:
        assert get_reader(unsupported) is None


def test_payload_read_bounds_and_offset(make_capture):
    values = np.exp(2j * np.pi * np.arange(8192) / 8).astype(np.complex64)
    path, raw = make_capture(values, start_sample=256, max_samples=1024)
    payload = raw.read_bytes()
    class CountingStream(io.BytesIO):
        def __init__(self, value):
            super().__init__(value)
            self.requests = []
            self.bytes_read = 0
        def read(self, count=-1):
            self.requests.append((self.tell(), count))
            data = super().read(count)
            self.bytes_read += len(data)
            return data
    stream = CountingStream(payload)
    actual = Path.open
    def tracked(file, *args, **kwargs):
        return stream if file == raw else actual(file, *args, **kwargs)
    with patch.object(Path, "open", tracked):
        _, attrs, _ = read_descriptor(path)[0]
    assert stream.bytes_read == 1024 * 8
    assert stream.requests == [(256 * 8, 1024 * 8)]
    assert attrs["metadata"]["limited"] is True


def test_memory_guard_precedes_payload_read(make_capture):
    path, raw = make_capture(np.ones(MAX_SAMPLES, dtype=np.complex64), n_fft=8192,
                            hop_length=1, max_samples=MAX_SAMPLES)
    actual = Path.open
    def guarded(file, *args, **kwargs):
        if file == raw:
            raise AssertionError("memory guard read payload")
        return actual(file, *args, **kwargs)
    with patch.object(Path, "open", guarded), pytest.raises(CaptureError, match="budget"):
        read_descriptor(path)


def test_shortened_stream_is_rejected():
    class Shortened(io.BytesIO):
        def read(self, count=-1):
            return b""
    with pytest.raises(CaptureError, match="shortened"):
        list(iter_iq(Shortened(bytes(4096 * 8)), CaptureMetadata("cf32_le", 8192),
                     max_samples=256))


@pytest.mark.parametrize("start", [4095, 4096, 9999])
def test_incomplete_frame_or_past_eof(make_capture, start):
    path, _ = make_capture(np.ones(4096, dtype=np.complex64), start_sample=start)
    with pytest.raises(CaptureError, match="fewer"):
        read_descriptor(path)


def test_int16_big_endian(make_capture):
    pairs = np.zeros((512, 2), dtype=">i2")
    pairs[:, 0] = 16384
    path, _ = make_capture(payload=pairs.tobytes(), format="ci16", endian="big")
    data, attrs, _ = read_descriptor(path)[0]
    row = np.argmax(data[:, 0])
    assert attrs["metadata"]["frequencies_hz"][row] == 0
    np.testing.assert_allclose(data[row, 0], 20 * np.log10(0.5), atol=1e-5)


@pytest.mark.fixture
@pytest.mark.parametrize("number", [1, 2, 3, 4])
def test_public_fixture_shape_and_finiteness(number):
    path = ROOT / "examples" / f"signal_{number:02d}.iq.json"
    dataset = path.parent / json.loads(path.read_text(encoding="utf-8"))["dataset"]
    if not dataset.is_file():
        pytest.skip("optional public recording is not present; see PROVENANCE.md")
    data, attrs, _ = read_descriptor(path)[0]
    assert data.shape == (1024, 511)
    assert np.isfinite(data).all()
    meta = attrs["metadata"]
    assert meta["samples_read"] == 262144
    assert meta["source_sample_count"] == 1000000 and meta["limited"]
    assert meta["payload_bytes_read"] == 2097152
    assert meta["frame_centers_seconds"][0] == 512 / 1e8
    assert meta["frame_centers_seconds"][-1] == 261632 / 1e8
