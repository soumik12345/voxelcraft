# voxelcraft

A library for training automatic building systems for Minecraft.

## 3D dataset visualization

Start the Gradio gallery from the project root:

```bash
uv run voxelcraft-visualize --data-dir data/houses --open
```

Alternatively, run `uv run python -m voxelcraft.visualization`. The app serves at
<http://127.0.0.1:7860>. Use `--port` to select another port.

The gallery defaults to Minecraft 1.12 block models with their original textures.
It includes orbit/zoom, recorded construction steps, final schematics, height
cutaways, and material filters. Choose **Voxels** for block hover details and
material/order/player colors. Expand the panels below the model to inspect source JSON, player statistics,
schematic shape and dtype, block inventories, the paginated action log, archive
renderings, and original file downloads. Folders with missing files remain browsable.

Game model geometry includes slabs, stairs, fences, panes, doors, and plants where
supported. Unsupported models or metadata variants appear as checkerboards and
are listed in the current-step metadata. Textures download on first use into
`.cache/voxelcraft/minecraft-1.12`; subsequent views use the cache. Animated textures
use a static first frame. The original colored voxel and point views remain available.

To use a Minecraft 1.12-compatible texture pack:

```bash
uv run voxelcraft-visualize --texture-pack /path/to/pack.zip
```

See [visualization architecture and rendering choices](docs/api/visualization.md) for
the dataset conventions, extension API, and references.

Run the visualization tests with:

```bash
uv run python -m unittest discover -s tests -v
```

## Documentation

The Zensical site includes getting started and visualization documentation with
API references generated from Python source. Preview it locally:

```bash
uv run --group docs zensical serve
```

Open <http://localhost:8000>. Build and validate the static site with:

```bash
uv run --group docs zensical build --strict
```

The build writes to `site/`. Documentation pages live in `docs/`, with navigation
configured in `zensical.toml`.
