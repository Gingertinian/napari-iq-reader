# Provenance and terms

Author of the existing software and this adapter: Jeronimo Munoz Larreta.
Original software has no additional license grant. The adapter does not
relicense upstream software or data.

`src/napari_iq_reader/_capture_io.py` is an isolated copy of `capture_io.py`
from [rf-ocusync-classifier](https://github.com/Gingertinian/rf-ocusync-classifier),
local commit `e81fde7a02f6598ae39a2f61cc198b4163c5f5ae`. The local copy is
byte-identical. Original SHA256:
`22837df07f8dd287d3c6c4802f9e0d5daeff9a6f59e9d3944eaf0562a01aa240`.
The verification report compares normalized source text and records both hashes.
No classifier, trained model, or other project code is imported.

The descriptors reference the existing 1000000-sample little-endian float32
excerpts without modifying or copying them. The verification report records
their SHA256 hashes. Source provenance is preserved in the original
`DATA_SOURCES.md`, `NOTICE.md`, `examples.json` and `LICENSE-RFUAV.txt`.

Examples 01 to 03 are from RFUAV by Rui Shi, Xiaodong Yu, Shengming Wang,
Yijia Zhang, Lu Xu, Peng Pan and Chunlai Ma. Dataset:
[RFUAV](https://huggingface.co/datasets/kitofrank/RFUAV),
revision `d91868143eb148bb5d87aa078e15ed3f8e1e7ed3`, Apache-2.0.
Members: `DJI MINI3/VTSBW=10/pack2_4-5s.iq`,
`DJI FPV COMBO/VTSBW=20/pack3_15-16s.iq`,
`DJI AVATA2/VTSBW=10/pack1_10-11s.iq`. Each excerpt starts at source
sample 21000000.

Example 04 is from DRFF-R2 by Haolin Zheng, Ning Gao, Zhenghang Zhu,
Zhijun Huang, Shi Jin and Michail Matthaiou:
[dataset](https://doi.org/10.57760/sciencedb.36815),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode).
The original repository converted the first 1000000 environmental samples
from MAT RF0_I/RF0_Q arrays to interleaved float32, without resampling.
Recorded rate: 100 MS/s; center: 5745 MHz. No endorsement is implied.

Implementation references:

- [Reader contracts](https://napari.org/stable/plugins/building_a_plugin/guides.html#readers)
- [Plugin packaging](https://napari.org/stable/plugins/building_a_plugin/first_plugin.html)
- [Image axes, units and transforms](https://napari.org/stable/api/napari.layers.Image.html)
