import json
from pathlib import Path

import numpy as np
import pytest
from napari.layers import Image
from npe2 import PluginManager, io_utils

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.fixture
def test_installed_manifest_dispatches_all_real_fixtures():
    manager = PluginManager.instance()
    manager.discover()
    manifest = manager.get_manifest("napari-iq-reader")
    manifest.validate_imports()
    assert len(manifest.contributions.readers) == 1
    for number in range(1, 5):
        path = ROOT / "examples" / f"signal_{number:02d}.iq.json"
        dataset = path.parent / json.loads(path.read_text(encoding="utf-8"))["dataset"]
        if not dataset.is_file():
            pytest.skip("optional public recording is not present; see PROVENANCE.md")
        layers = io_utils.read([str(path)], stack=False, plugin_name="napari-iq-reader")
        assert len(layers) == 1 and layers[0][2] == "image"
        assert layers[0][0].shape == (1024, 511)


def test_napari_image_accepts_units_and_signed_world_transform(make_capture, switching_tone):
    path, _ = make_capture(switching_tone, start_sample=256)
    PluginManager.instance().discover()
    data, attrs, _ = io_utils.read([str(path)], stack=False, plugin_name="napari-iq-reader")[0]
    layer = Image(data, **attrs)
    assert layer.rgb is False
    assert tuple(layer.axis_labels) == ("Frequency (MHz)", "Time (ms)")
    assert str(layer.units[0]) == "megahertz" and str(layer.units[1]) == "millisecond"
    for column, frequency in [(0, 1024), (30, -768)]:
        row = int(np.argmax(layer.data[:, column]))
        time_seconds = (256 + 128 + column * 128) / 8192
        np.testing.assert_allclose(layer.data_to_world((row, column)),
                                   [frequency / 1e6, time_seconds * 1e3])
