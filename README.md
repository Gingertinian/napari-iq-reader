# I/Q spectrogram reader

A small npe2 reader by Jeronimo Munoz Larreta. It connects an existing bounded
I/Q reader to napari without importing Qt in the numerical module. Open an
`*.iq.json` descriptor to get one float32 image, not an RGB screenshot.

Rows run from positive to negative frequency; columns advance in time. Metadata
contains exact Hz and second coordinates. The layer transform uses MHz and ms,
including a signed frequency scale and frame-center time offset. Camera
orientation determines screen direction. Values are relative power in dB,
with a -120 dB floor, not calibrated dBm or a power spectral density.

## Reproduce

From this folder, in PowerShell with Python 3.11 or later:

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install --no-cache-dir -r requirements-tested.txt -e ".[test,viewer]"
./tools/verify.ps1
```

The Windows test wrapper removes its temporary drive alias on exit, avoiding DLL
path-length limits in deeply nested folders. Elsewhere, run `python tools/verify.py`.
For an existing napari environment, install `-e .` and open a descriptor with:

```python
viewer.open("examples/signal_01.iq.json", plugin="napari-iq-reader")
```

The four descriptors use recordings from a sibling checkout:

```sh
git clone --depth 1 https://github.com/Gingertinian/rf-ocusync-classifier.git ../rf-ocusync-classifier
```

Recordings are not bundled. Raw layout, sample rate and optional center frequency
are explicit. Mini 3 uses an adopted corpus rate. See [provenance](PROVENANCE.md).
Absent recordings skip the optional fixture checks; the generated-tone and
installed plugin/Image tests run independently.

Each read selects at most 262144 complex samples. The reader checks a 32 MiB
transform allocation estimate before reading the payload. It rejects truncated
pairs, invalid metadata and nonfinite selected samples. It checks whole-file
alignment but does not inspect values outside the selected interval. FFT windows
have no padding; an incomplete final frame is discarded.

Tests cover signed generated tones, absolute frequency, time offsets, amplitude
normalization, malformed input, actual read counts and all four public fixtures.
The verifier writes a detailed local report to `verification.json`.
`CHECKS.txt` summarizes the verified run.
Installed npe2 dispatch and an actual napari Image layer are verified headlessly.
Both offscreen Viewer tests skip on this Windows system because Qt cannot create
an OpenGL context. Viewer.open, interactive rendering and screenshots remain
unverified. These skips preserve the native error in the report.

This is a visualization adapter, with no payload decoding or classifier.
