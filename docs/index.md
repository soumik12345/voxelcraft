# VoxelCraft

VoxelCraft is a Python library for training automatic building systems for
Minecraft. Its current tools include a Gradio gallery for exploring Meta's
3D-Craft dataset, with textured Minecraft geometry, construction replay, and
source metadata.

## Install

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From a checkout of the repository, install the project dependencies:

```bash
uv sync
```

The visualization app runs on a CPU and does not require PyTorch, CUDA, or a
Minecraft installation.

## Open the 3D-Craft gallery

Extract the dataset so each house has its own directory under `data/houses`:

```text
data/
└── houses/
    └── <house-id>/
        ├── placed.json
        ├── schematic.npy
        └── stats.json
```

Additional archive previews and metadata files are optional. Houses with missing
files remain browsable, with coverage notes in the app.

```bash
uv run voxelcraft-visualize --data-dir data/houses --open
```

The gallery opens at <http://127.0.0.1:7860>. Use `--port` to change the port.
The first textured view downloads Minecraft assets and caches them locally.

| Mode | Use it to |
| --- | --- |
| Textured blocks | Orbit and zoom around objects with Minecraft 1.12 textures and block geometry. |
| Voxels | Inspect block IDs and metadata on hover, with material, build-order, or player colors. |
| Points | Explore large objects with lighter geometry and per-block hover details. |

Use the timeline to replay recorded edits, switch to the final schematic, filter
materials, or lower the cutaway height to inspect interiors. Expand the panels
below the object to inspect original JSON, inventory, action history, previews,
and file downloads.

Read the [visualization documentation](api/visualization.md) for dataset conventions,
texture packs, rendering limits, and reusable app examples.

## Extend the viewer

Subclass `BaseVisualizationApp` to add another dataset, or embed the existing
3D-Craft app in a Gradio project:

```python
from voxelcraft.visualization import Meta3DCraftVisualizationApp

viewer = Meta3DCraftVisualizationApp("data/houses")
viewer.launch(server_port=7861)
```

See the [visualization API](api/visualization.md) for the available app classes,
replay helpers, and Minecraft rendering interfaces.
