# Visualization API

!!! warning

    The visualization apps are generated using AI and only tested lightly without
    thorough review.

The public app classes are exported from `voxelcraft.visualization`. The base
class supplies dataset discovery, navigation, and the Gradio lifecycle. The
3D-Craft implementation loads house artifacts and connects geometry, replay, and
metadata controls.

The Minecraft rendering backend provides asset loading and textured GLB export
for the viewer.

The 3D-Craft viewer uses a gallery layout with a large interactive object and
expandable metadata panels. It runs locally in Gradio and requires
no CUDA device or Minecraft installation.

## Launch and reuse

```bash
uv run voxelcraft-visualize --data-dir data/houses --open
# Equivalent module entry point:
uv run python -m voxelcraft.visualization --data-dir data/houses
```

The default address is `http://127.0.0.1:7860`. `--host` and `--port` override the
binding; `--share` explicitly enables a public Gradio share link. Only the selected
dataset directory is added to Gradio's allowed file paths. Restart the app after
changing the dataset on disk; loaded samples use a bounded eight-house cache.
The first textured view requires network access to fetch the pinned legacy assets.
They are cached under `.cache/voxelcraft/minecraft-1.12`, which is ignored by Git.
Generated scene GLBs use a separate temporary cache bounded to 32 scenes; Gradio
also expires its served artifact cache after an hour.

`--texture-pack /path/to/pack.zip` (or a directory) overrides texture PNGs using the
Minecraft 1.12 resource paths `assets/minecraft/textures/blocks/*.png` and
`assets/minecraft/textures/entity/**/*.png`. Missing pack
textures fall back to the vanilla assets. This option changes images, not model
definitions. Restart after editing a pack. No game textures are bundled in the
Python package.

```python
from voxelcraft.visualization import Meta3DCraftVisualizationApp

viewer = Meta3DCraftVisualizationApp("data/houses")
demo = viewer.build()  # A gr.Blocks instance; can be embedded or mounted.
# Or launch with Gradio options:
viewer.launch(server_port=7861)
```

`BaseVisualizationApp` supplies dataset path validation, sample navigation,
idempotent Blocks construction, and local launch defaults. To support another
dataset, subclass it and implement:

- `discover_samples() -> list[str]`: stable sample directory identifiers.
- `create_content(app: gr.Blocks) -> None`: dataset components and callbacks inside
  the active Blocks context. Register initial loading with `app.load(...)`.

Selections, source choices, and construction positions live in each browser
session's components, rather than shared instance attributes. Dataset records can
be cached across sessions because rendering does not mutate them.

## Dataset fidelity

The supplied `data/README` defines these files:

| File | Interpretation |
| --- | --- |
| `placed.json` | Ordered `(tick, entity_id, [x,y,z], [block_id,meta], action)` records. `P` places/replaces, `B` removes. There are 20 game ticks per second. |
| `schematic.npy` | Final integer array in **Y, Z, X, id/meta** order. ID zero is air. The app loads it with `allow_pickle=False`. |
| `stats.json` | Original aggregate statistics, including dimensions, placement/removal counts, and named players' minutes. All original fields are preserved. |
| `chunky.*.png` | Original rendered viewpoints, when present; available in the preview gallery. |
| `depth.*.bin`, `block.*.bin` | Original binary buffers, where present. Their byte sizes and downloads are exposed; the archive README does not specify their binary layouts. |

Any additional per-house JSON files are included in the metadata panel. Every
per-house source file is available in the download panel. A neighboring
`data/splits.json`, if provided, adds train/validation/test membership.

Recorded construction replays actions **in file order** from an empty scene. It
does not reconstruct pre-existing terrain from break actions. Timestamps and
entity IDs are preserved, and elapsed seconds are measured relative to the first
recorded action. Entity IDs are session-local; the files do not provide a mapping
from entity IDs to the player names in `stats.json`.

The final schematic has local index coordinates; the archive does not provide
its world-space origin. It is shown as a separate source, without inventing an
alignment with action coordinates. Replayed recordings and schematic contents can
differ. Missing or malformed files produce coverage notes; malformed action rows
are reported and skipped while preserving original row numbers. Original files
remain downloadable. Action pages contain 100 valid records, so large logs are
fully accessible without sending the entire log to the browser on every update.

## Rendering choices and references

The default **Textured blocks** mode exports a self-contained binary glTF (GLB)
with embedded Minecraft PNGs to Gradio's native
[Model3D](https://gradio.app/docs/gradio/model3d). It uses the original Minecraft
1.12 models, parent inheritance, face UVs, rotations, and metadata-selected variants.
Legacy numeric ID definitions and game assets come from pinned snapshots of
[minecraft-data](https://github.com/PrismarineJS/minecraft-data/tree/bd8b5c6048e5af9694374b83f25f5b4aade619cc/data/pc/1.12)
and [minecraft-assets](https://github.com/PrismarineJS/minecraft-assets/tree/6c571b0825ee27b81f01127c2bd0d4ba2c89abb9/data/1.12).
Texture downloads use the broader
[MCAsset Cloud 1.12.2 archive](https://github.com/InventivetalentDev/minecraft-assets/tree/3360cc4a4e7be0183be2fae38622dfece4468401/assets/minecraft/textures),
including entity texture sheets omitted from the simplified block catalog.
Previously cached vanilla block PNGs remain usable. Both catalog and texture
archive revisions appear in `minecraft_rendering` metadata.
Nearest-neighbor texture samplers retain pixel edges. Opaque, cutout, and
semi-transparent materials retain the PNG alpha channel. Mesh primitives are
batched by texture/material; original images are embedded once per GLB.

Models include slabs, oriented straight stairs, connected fences/panes, two-part
doors, flowers, and double plants where their necessary state is present. Special
meshes add closed normal/trapped single and double chests, Ender chests, bed halves,
standing/wall signs, and water/lava surfaces. These use the legacy entity texture
layouts and metadata orientations. Double-chest halves retain their source pairing
when filtering or cutting away the other half. Fluid corner heights use source
neighbors and level metadata; shared faces are removed between visible fluids.
Entity sheets retain their full aspect ratio, including resource pack overrides.
Model selection sees the full source neighborhood even when material filters hide a
neighbor. The renderer is a static dataset viewer, with these limits:

- Stairs currently use the straight variant; inner/outer corner joins are not simulated.
- Animated block textures use the first frame. Fluid flow animation, lighting,
  and biome-specific tints are not a full Minecraft client simulation.
- Beds default to red and signs have no text: these files do not contain the
  necessary tile-entity color/text data. Chest lids remain closed. Other special
  block entities (such as banners and skulls) can still require a fallback.
- Grass/foliage use an illustrative temperate tint because the dataset does not
  supply biome information.
- Missing models or uninterpreted nonzero metadata are rendered as conspicuous
  checkerboards and listed under `minecraft_rendering.fallback_variants`.
- If asset downloads fail, the app shows colored voxels and an explanatory status
  message while keeping metadata and controls usable.

**Voxels** and **Points** use Plotly through
[gr.Plot](https://gradio.app/docs/gradio/plot), providing per-block hover details and
analysis colors. These modes use full cubes or point centers. Their camera persists
within a house/source. Model3D can reframe after a GLB update; it does not offer
the same block-hover interaction. All rendering modes share timeline, cutaway,
material filters, metadata, and original file downloads.

The following are implementation choices for this dataset, informed by those
APIs rather than a claim that Gradio specifies a universal voxel standard:

- **Exact cube surfaces:** explicit indexed triangles in one WebGL `Mesh3d` trace
  preserve voxel boundaries and holes. No Delaunay triangulation, convex hull, or
  smoothed isosurface is used. Plotly's [Mesh3d documentation](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Mesh3d.html)
  recommends explicit `i`, `j`, and `k` indices for general meshes.
- **Exposed faces only:** adjacent occupied cells share no rendered internal
  faces. Sparse neighbor lookups avoid dense allocations spanning Minecraft world
  coordinates. Filtering or cutting away blocks reveals the newly exposed faces.
- **Batched geometry:** float32 positions and int32 triangle indices keep the
  numeric payload compact. The renderer uses one trace, rather than one trace per
  voxel. Point mode is available for a lighter view.
- **Readable geometry:** opaque faces and flat shading preserve corners and avoid
  transparent-surface sorting artifacts. Materials use deterministic approximate
  colors; block IDs and metadata remain authoritative in hover and inventory.
- **Correct proportions:** Minecraft Y is mapped to the plotting library's
  vertical Z axis. Hover retains original Minecraft XYZ. Fixed scene bounds and
  [`aspectmode="data"`](https://plotly.com/python/3d-axes/) preserve unit-cube
  proportions through construction steps and cutaways.
- **Camera continuity in Plotly modes:** scene and layout
  [`uirevision`](https://plotly.com/python/reference/layout/scene/#layout-scene-uirevision)
  stay stable within a house/source. Gradio 6.31's bundled Plotly component uses
  `Plotly.react`; changing houses or geometry sources resets the revision.
- **Controlled updates:** sliders handle input from both pointers and keyboards.
  Callbacks use `trigger_mode="always_last"` to coalesce pending changes and share
  a serial concurrency group so old renders cannot finish over a new selection.
  Metadata and preview panels are collapsed by default to give the object space.

The geometry/replay tests cover replacement and removal, air, signed legacy IDs,
axis order, outward triangle winding, shared-face culling, empty states, cutaways,
filters, stable bounds, malformed data, pagination, source selection, and Gradio's
JSON encoder.

Legacy mesh checks additionally cover chest facing, double-chest UV continuity,
bed legs and halves, wall-sign placement, fluid level transitions and filtering,
entity texture aspect ratios, and ZIP/directory resource pack overrides.

## Base app

::: voxelcraft.visualization.base.BaseVisualizationApp
    options:
      members:
        - discover_samples
        - create_content
        - sample_directory
        - adjacent_sample
        - random_sample
        - build
        - launch

## 3D-Craft app

::: voxelcraft.visualization.meta_3d_craft.Meta3DCraftVisualizationApp
    options:
      members:
        - discover_samples
        - render_sample
        - render_scene
        - action_page
        - sample_metadata
        - select_sample
        - create_content

## Dataset records

::: voxelcraft.visualization.meta_3d_craft.CraftEvent

::: voxelcraft.visualization.meta_3d_craft.Block

::: voxelcraft.visualization.meta_3d_craft.CraftSample

## Geometry and replay helpers

::: voxelcraft.visualization.meta_3d_craft.replay_events

::: voxelcraft.visualization.meta_3d_craft.schematic_blocks

::: voxelcraft.visualization.meta_3d_craft.voxel_figure

## Minecraft rendering

The textured rendering backend resolves legacy Minecraft block IDs, metadata,
models, and texture sheets, then exports a self-contained GLB for Gradio's
`Model3D` component. It does not require a Minecraft client or GPU server.

See the [rendering details](#rendering-choices-and-references)
for asset sources, supported geometry, caching, and fidelity limits.

### Asset loading

`MinecraftAssets` downloads pinned vanilla assets into a local cache and supports
texture overrides from a resource-pack directory or ZIP. Block and entity texture
paths follow Minecraft 1.12 conventions.

::: voxelcraft.visualization.minecraft.MinecraftAssets
    options:
      heading_level: 4
      members:
        - catalogs
        - model
        - texture
        - applications

### Scene export

`MinecraftRenderer.export()` accepts a mapping from Minecraft XYZ coordinates to
blocks and returns the generated GLB path plus a rendering report. Pass the full
source scene as `context_blocks` when exporting a filtered view so connected
models retain their original state. Call `close()` after the server has stopped
using its generated files.

::: voxelcraft.visualization.minecraft.MinecraftRenderer
    options:
      heading_level: 4
      members:
        - export
        - close

### Errors and block interface

::: voxelcraft.visualization.minecraft.MinecraftAssetError
    options:
      heading_level: 4

::: voxelcraft.visualization.minecraft.BlockLike
    options:
      heading_level: 4
