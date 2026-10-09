import os
import sys
from pathlib import Path

import numpy as np
import pytest

from napari_iq_reader import read_descriptor

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def offscreen_viewer():
    if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
        pytest.skip("set QT_QPA_PLATFORM=offscreen for this test")
    napari = pytest.importorskip("napari")
    from OpenGL.error import GLError
    try:
        viewer = napari.Viewer(show=False)
    except GLError as error:
        if sys.platform == "win32" and error.err == 1282 and "glGetIntegerv" in str(error):
            pytest.skip(f"Windows offscreen Qt cannot create an OpenGL context: {error}")
        raise
    try:
        yield viewer
    finally:
        viewer.close()


@pytest.mark.viewer
def test_offscreen_viewer_plugin_and_coordinates(make_capture, switching_tone, offscreen_viewer):
    from npe2 import PluginManager
    path, _ = make_capture(switching_tone, start_sample=256)
    manager = PluginManager.instance()
    manager.discover()
    manifest = manager.get_manifest("napari-iq-reader")
    manifest.validate_imports()
    viewer = offscreen_viewer
    try:
        layer = viewer.open(str(path), plugin="napari-iq-reader")[0]
        assert not viewer.window._qt_window.isVisible()
        assert layer.data.shape == (256, 31) and layer.rgb is False
        assert tuple(layer.axis_labels) == ("Frequency (MHz)", "Time (ms)")
        assert str(layer.units[0]) == "megahertz" and str(layer.units[1]) == "millisecond"
        for column, frequency in [(0, 1024), (30, -768)]:
            row = np.argmax(layer.data[:, column])
            world = layer.data_to_world((int(row), column))
            expected_time = (256 + 128 + column * 128) / 8192
            np.testing.assert_allclose(world, [frequency / 1e6, expected_time * 1e3])
    finally:
        viewer.close()


@pytest.mark.viewer
@pytest.mark.fixture
def test_offscreen_viewer_real_fixtures(offscreen_viewer):
    viewer = offscreen_viewer
    try:
        for number in range(1, 5):
            path = ROOT / "examples" / f"signal_{number:02d}.iq.json"
            layer = viewer.open(str(path), plugin="napari-iq-reader")[0]
            expected, attrs, _ = read_descriptor(path)[0]
            np.testing.assert_array_equal(layer.data, expected)
            assert layer.data.shape == (1024, 511)
            np.testing.assert_allclose(layer.data_to_world((0, 0)), attrs["translate"])
            viewer.layers.remove(layer)
    finally:
        viewer.close()
