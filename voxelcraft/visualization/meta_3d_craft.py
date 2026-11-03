"""Gallery and metadata inspector for Meta's 3D-Craft Minecraft dataset.

Run with ``python -m voxelcraft.visualization --data-dir data/houses``.
"""

import argparse
import colorsys
import json
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypedDict

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from voxelcraft.visualization.base import BaseVisualizationApp
from voxelcraft.visualization.minecraft import (
    TEXTURED,
    MinecraftAssetError,
    MinecraftAssets,
    MinecraftRenderer,
)

RECORDING = "Recorded construction"
SCHEMATIC = "Final schematic"
PAGE_SIZE = 100


class SceneEventOptions(TypedDict):
    """Options shared by scene callbacks, retaining each keyword's type."""

    concurrency_id: str
    concurrency_limit: int
    trigger_mode: Literal["always_last"]


# These are legacy Minecraft numeric IDs, not modern namespaced IDs. Unknown
# IDs and variants remain visible as numeric id:meta pairs, without guessing.
MATERIALS = {
    1: ("Stone", "#8d9298"),
    2: ("Grass", "#7f9955"),
    3: ("Dirt", "#927154"),
    4: ("Cobblestone", "#777e83"),
    5: ("Wood planks", "#bf9c67"),
    12: ("Sand", "#d9c98d"),
    17: ("Log", "#806046"),
    18: ("Leaves", "#5e854f"),
    20: ("Glass", "#b0d7de"),
    24: ("Sandstone", "#d5c28a"),
    35: ("Wool", "#c6adbb"),
    41: ("Gold block", "#d4ad40"),
    42: ("Iron block", "#bfc6ca"),
    43: ("Double slab", "#9d9a90"),
    44: ("Slab", "#aaa79c"),
    45: ("Bricks", "#ac7262"),
    47: ("Bookshelf", "#a48b59"),
    49: ("Obsidian", "#51465f"),
    50: ("Torch", "#dfb55d"),
    53: ("Wood stairs", "#b68d59"),
    64: ("Wood door", "#a67d4c"),
    65: ("Ladder", "#b79152"),
    67: ("Cobblestone stairs", "#858a8d"),
    78: ("Snow layer", "#e0e9ef"),
    79: ("Ice", "#9dbfdb"),
    80: ("Snow", "#dfe7ec"),
    85: ("Fence", "#b5925c"),
    98: ("Stone bricks", "#858c8d"),
}


@dataclass(frozen=True)
class CraftEvent:
    index: int  # One-based index in the original file, including skipped rows.
    tick: int
    entity_id: int
    xyz: tuple[int, int, int]
    block_id: int
    block_meta: int
    action: str


@dataclass(frozen=True)
class Block:
    block_id: int
    block_meta: int
    event_index: int = -1
    tick: int = -1
    entity_id: int = -1


@dataclass(frozen=True)
class CraftSample:
    sample_id: str
    events: tuple[CraftEvent, ...]
    action_count: int
    json_metadata: dict[str, Any]
    schematic: np.ndarray | None
    schematic_info: dict[str, Any]
    files: tuple[Path, ...]
    warnings: tuple[str, ...]


def replay_events(
    events: tuple[CraftEvent, ...], steps: int
) -> dict[tuple[int, int, int], Block]:
    """Replay a prefix in file order, respecting replacement, air, and removal."""
    blocks = {}
    for event in events:
        if event.index > steps:
            break
        if event.action == "B" or event.block_id == 0:
            blocks.pop(event.xyz, None)
        else:
            blocks[event.xyz] = Block(
                event.block_id,
                event.block_meta,
                event.index,
                event.tick,
                event.entity_id,
            )
    return blocks


def schematic_blocks(array: np.ndarray) -> dict[tuple[int, int, int], Block]:
    """Convert the documented (Y, Z, X, id/meta) array to Minecraft XYZ."""
    return {
        (int(x), int(y), int(z)): Block(
            int(array[y, z, x, 0]) % 256, int(array[y, z, x, 1]) % 256
        )
        for y, z, x in np.argwhere(array[..., 0] != 0)
    }


def material_name(block_id: int) -> str:
    return MATERIALS.get(block_id, (f"Block {block_id}", ""))[0]


def block_color(block: Block, mode: str, total_actions: int) -> str:
    if mode == "Build order" and block.event_index > 0:
        fraction = block.event_index / max(total_actions, 1)
        rgb = colorsys.hsv_to_rgb(0.63 - 0.5 * fraction, 0.55, 0.85)
    elif mode == "Player" and block.entity_id >= 0:
        rgb = colorsys.hsv_to_rgb((block.entity_id * 0.618034) % 1, 0.45, 0.8)
    elif block.block_id in MATERIALS:
        return MATERIALS[block.block_id][1]
    else:
        rgb = colorsys.hsv_to_rgb(
            ((block.block_id * 37 + block.block_meta * 7) % 360) / 360, 0.35, 0.78
        )
    return "#" + "".join(f"{round(value * 255):02x}" for value in rgb)


# Quads face outward in XYZ. Winding is reversed after swapping Y/Z for Plotly.
FACES = (
    ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
    ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
)


def voxel_figure(
    blocks: dict[tuple[int, int, int], Block],
    bounds: tuple[tuple[int, int, int], tuple[int, int, int]],
    revision: str,
    color_mode: str = "Material",
    render_mode: str = "Voxels",
    total_actions: int = 0,
) -> go.Figure:
    """One WebGL trace, explicit exposed faces, and metadata on each block.

    Sparse neighbor lookups avoid allocating a dense world-coordinate volume.
    Fixed bounds and uirevision keep the view stable during timeline/filter edits.
    """
    figure = go.Figure()
    hover = "X %{customdata[0]} · Y %{customdata[1]} · Z %{customdata[2]}<br>Block %{customdata[3]}:%{customdata[4]}"
    if any(block.event_index > 0 for block in blocks.values()):
        hover += "<br>Action %{customdata[5]} · tick %{customdata[6]}<br>Entity %{customdata[7]}"
    hover += "<extra></extra>"
    vertices, triangles, colors, metadata = [], [], [], []
    for xyz, block in blocks.items():
        color = block_color(block, color_mode, total_actions)
        record = [
            *xyz,
            block.block_id,
            block.block_meta,
            block.event_index,
            block.tick,
            block.entity_id,
        ]
        x, y, z = xyz
        if render_mode == "Points":
            vertices.append((x + 0.5, z + 0.5, y + 0.5))
            colors.append(color)
            metadata.append(record)
            continue
        for direction, corners in FACES:
            neighbor = tuple(value + delta for value, delta in zip(xyz, direction))
            if neighbor in blocks:
                continue
            start = len(vertices)
            vertices.extend((x + dx, z + dz, y + dy) for dx, dy, dz in corners)
            triangles.extend(
                ((start, start + 2, start + 1), (start, start + 3, start + 2))
            )
            colors.extend([color] * 4)
            metadata.extend([record] * 4)
    if vertices:
        coordinates = np.asarray(vertices, dtype=np.float32)
        common = dict(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
            customdata=np.asarray(metadata, dtype=np.int32),
            hovertemplate=hover,
        )
        if render_mode == "Points":
            figure.add_trace(
                go.Scatter3d(
                    **common,
                    mode="markers",
                    marker=dict(size=3, color=colors, opacity=1),
                )
            )
        else:
            faces = np.asarray(triangles, dtype=np.int32)
            figure.add_trace(
                go.Mesh3d(
                    **common,
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    vertexcolor=colors,
                    flatshading=True,
                    opacity=1,
                    lighting=dict(
                        ambient=0.65, diffuse=0.75, specular=0.05, roughness=1
                    ),
                    lightposition=dict(x=100, y=100, z=200),
                )
            )
    else:
        figure.add_annotation(
            text="No blocks at this step or with these filters.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
    low, high = bounds
    scene = {
        key: dict(
            title=title,
            range=[low[index] - 1, high[index] + 2],
            gridcolor="#dce2e6",
            showbackground=False,
            showspikes=False,
        )
        for key, title, index in [
            ("xaxis", "X", 0),
            ("yaxis", "Z", 2),
            ("zaxis", "Y · height", 1),
        ]
    }
    figure.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=5, b=0),
        showlegend=False,
        paper_bgcolor="#f8fafb",
        font=dict(color="#334155"),
        uirevision=revision,
        scene=dict(
            **scene,
            aspectmode="data",
            uirevision=revision,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)),
        ),
    )
    return figure


class Meta3DCraftVisualizationApp(BaseVisualizationApp):
    """Browse 3D-Craft house geometry, construction history, metadata, and artifacts.

    Dataset source: Facebook AI Research (now Meta), released through the
    [CraftAssist repository](https://github.com/facebookresearch/craftassist#datasets)
    as the [house dataset archive](https://craftassist.s3-us-west-2.amazonaws.com/pubr/house_data.tar.gz).

    Paper: Chen et al.,
    [Order-Aware Generative Modeling Using the 3D-Craft Dataset](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Order-Aware_Generative_Modeling_Using_the_3D-Craft_Dataset_ICCV_2019_paper.pdf),
    ICCV 2019, Section 3.2.

    Collection: Crowdsourced players built houses from scratch in Minecraft's
    creative mode, with a 30-minute time budget and no additional building
    instructions. A Cuberite server logged timestamped block placements and
    removals, including player IDs, XYZ coordinates, and block IDs/metadata.
    """

    title = "3D-Craft gallery"
    description = (
        "Explore human-built Minecraft houses, then unfold their construction history and metadata.\n\n"
        "**Dataset source:** Facebook AI Research (now Meta) · "
        "[CraftAssist](https://github.com/facebookresearch/craftassist#datasets) · "
        "[House dataset archive](https://craftassist.s3-us-west-2.amazonaws.com/pubr/house_data.tar.gz).\n\n"
        "**Paper:** Chen et al., "
        "[Order-Aware Generative Modeling Using the 3D-Craft Dataset]"
        "(https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Order-Aware_Generative_Modeling_Using_the_3D-Craft_Dataset_ICCV_2019_paper.pdf) "
        "(ICCV 2019, Section 3.2).\n\n"
        "**Collection method:** Crowdsourced players built houses from scratch in "
        "Minecraft creative mode with a 30-minute time budget and no additional "
        "building instructions. A Cuberite server recorded timestamped block "
        "placements and removals, player IDs, XYZ coordinates, and block IDs/metadata."
    )

    def __init__(
        self,
        data_dir: str | Path = "data/houses",
        *,
        texture_pack: str | Path | None = None,
    ) -> None:
        super().__init__(data_dir)
        # A bounded instance cache serves multiple sessions without storing any
        # session's selected house, filters, or timeline position on the app.
        self.load_sample = lru_cache(maxsize=8)(self._load_sample)
        self.minecraft_renderer = MinecraftRenderer(
            MinecraftAssets(texture_pack=texture_pack)
        )

    def discover_samples(self) -> list[str]:
        return sorted(
            path.name
            for path in self.data_dir.iterdir()
            if path.is_dir() and path.resolve().is_relative_to(self.data_dir)
        )

    def _load_sample(self, sample_id: str) -> CraftSample:
        directory = self.sample_directory(sample_id)
        files = tuple(
            sorted(
                path
                for path in directory.iterdir()
                if path.is_file() and path.resolve().is_relative_to(self.data_dir)
            )
        )
        warnings, metadata, events = [], {}, []
        action_count = 0
        for path in files:
            if path.suffix != ".json":
                continue
            try:
                content = json.loads(path.read_text())
                if path.name != "placed.json":
                    metadata[path.name] = content
                    continue
                if not isinstance(content, list):
                    raise ValueError("expected a list of actions")
                action_count = len(content)
                invalid = 0
                for index, row in enumerate(content, 1):
                    try:
                        tick, entity, xyz, block_info, action = row
                        if (
                            len(xyz) != 3
                            or len(block_info) != 2
                            or action not in ("P", "B")
                        ):
                            raise ValueError("invalid action fields")
                        numbers = [tick, entity, *xyz, *block_info]
                        if any(
                            isinstance(n, bool)
                            or not isinstance(n, (int, float))
                            or not math.isfinite(n)
                            or n != int(n)
                            for n in numbers
                        ):
                            raise ValueError("expected finite integer fields")
                        events.append(
                            CraftEvent(
                                index,
                                int(tick),
                                int(entity),
                                (int(xyz[0]), int(xyz[1]), int(xyz[2])),
                                int(block_info[0]) % 256,
                                int(block_info[1]) % 256,
                                action,
                            )
                        )
                    except (ValueError, TypeError):
                        invalid += 1
                if invalid:
                    warnings.append(
                        f"placed.json: skipped {invalid} invalid action rows; original row numbers are preserved."
                    )
                if any(a.tick > b.tick for a, b in zip(events, events[1:])):
                    warnings.append(
                        "Action ticks are not monotonic; replay follows the original file order."
                    )
            except (OSError, ValueError, TypeError) as error:
                warnings.append(f"{path.name}: {error}")
        schematic, schematic_info = None, {}
        path = directory / "schematic.npy"
        if path in files:
            try:
                array = np.load(path, allow_pickle=False, mmap_mode="r")
                if (
                    array.ndim != 4
                    or array.shape[-1] != 2
                    or array.dtype.kind not in "iu"
                ):
                    raise ValueError(
                        "expected an integer array with shape (Y, Z, X, 2)"
                    )
                schematic = array
                schematic_info = {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "axes": ["Y (height)", "Z", "X", "block ID / metadata"],
                    "occupied_cells": int(np.count_nonzero(array[..., 0])),
                    "coordinate_frame": "Local indices; the archive does not supply a world origin.",
                }
            except (OSError, ValueError, TypeError) as error:
                warnings.append(f"schematic.npy: {error}")
        for name in ("placed.json", "stats.json", "schematic.npy"):
            if not any(path.name == name for path in files):
                warnings.append(f"{name} is missing.")
        return CraftSample(
            sample_id,
            tuple(events),
            action_count,
            metadata,
            schematic,
            schematic_info,
            files,
            tuple(warnings),
        )

    @staticmethod
    def _source_blocks(sample: CraftSample, source: str, steps: int) -> dict:
        if source == SCHEMATIC:
            return (
                schematic_blocks(sample.schematic)
                if sample.schematic is not None
                else {}
            )
        return replay_events(sample.events, steps)

    @staticmethod
    def _bounds(sample: CraftSample, source: str) -> tuple:
        if source == SCHEMATIC and sample.schematic is not None:
            y, z, x, _ = sample.schematic.shape
            return (0, 0, 0), (max(x - 1, 0), max(y - 1, 0), max(z - 1, 0))
        if sample.events:
            coords = np.asarray([event.xyz for event in sample.events])
            return tuple(coords.min(axis=0).tolist()), tuple(
                coords.max(axis=0).tolist()
            )
        return (0, 0, 0), (1, 1, 1)

    def render_sample(
        self,
        sample_id,
        source,
        steps,
        color_mode,
        render_mode,
        materials,
        max_height,
        *,
        render_geometry=True,
    ):
        sample = self.load_sample(sample_id)
        steps = min(max(int(steps or 0), 0), sample.action_count)
        blocks = self._source_blocks(sample, source, steps)
        visible = {
            xyz: block
            for xyz, block in blocks.items()
            if xyz[1] <= max_height
            and (not materials or f"{block.block_id}:{block.block_meta}" in materials)
        }
        figure = (
            voxel_figure(
                visible,
                self._bounds(sample, source),
                f"{sample_id}:{source}",
                color_mode,
                render_mode,
                sample.action_count,
            )
            if render_geometry
            else go.Figure()
        )
        counts = Counter(
            (block.block_id, block.block_meta) for block in blocks.values()
        )
        shown = Counter(
            (block.block_id, block.block_meta) for block in visible.values()
        )
        inventory = [
            [
                block_id,
                meta,
                material_name(block_id),
                count,
                shown[(block_id, meta)],
                block_color(Block(block_id, meta), "Material", 0),
            ]
            for (block_id, meta), count in sorted(counts.items())
        ]
        status = f"**{len(visible):,} visible / {len(blocks):,} occupied blocks**"
        if source == RECORDING:
            status += (
                f" · action {steps:,} / {sample.action_count:,} · world coordinates"
            )
        else:
            status += " · local schematic coordinates"
        if source == SCHEMATIC and color_mode != "Material":
            status += " · schematic has no action/player data; using material colors"
        applied = [event for event in sample.events if event.index <= steps]
        last = applied[-1] if applied and source == RECORDING else None
        current = {
            "source": source,
            "occupied_blocks": len(blocks),
            "visible_blocks": len(visible),
            "action_position": steps if source == RECORDING else None,
            "last_action": vars(last) if last else None,
            "elapsed_seconds": (last.tick - sample.events[0].tick) / 20
            if last
            else None,
        }
        return figure, status, inventory, current

    def action_page(self, sample_id: str, page: int) -> tuple[list, str]:
        sample = self.load_sample(sample_id)
        page_count = max(1, math.ceil(len(sample.events) / PAGE_SIZE))
        page = min(max(int(page or 1), 1), page_count)
        events = sample.events[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]
        start_tick = sample.events[0].tick if sample.events else 0
        rows = [
            [
                event.index,
                event.tick,
                (event.tick - start_tick) / 20,
                event.entity_id,
                *event.xyz,
                event.block_id,
                event.block_meta,
                "Place" if event.action == "P" else "Break",
            ]
            for event in events
        ]
        return (
            rows,
            f"Page {page} / {page_count} · {len(sample.events):,} valid actions · 20 ticks = 1 second",
        )

    def render_scene(
        self, sample_id, source, steps, color_mode, render_mode, materials, max_height
    ):
        """Keep inspection outputs consistent when switching rendering backends."""
        figure, status, inventory, current = self.render_sample(
            sample_id,
            source,
            steps,
            color_mode,
            "Voxels" if render_mode == TEXTURED else render_mode,
            materials,
            max_height,
            render_geometry=render_mode != TEXTURED,
        )
        model_path = None
        textured = render_mode == TEXTURED
        if textured:
            sample = self.load_sample(sample_id)
            blocks = self._source_blocks(
                sample, source, current["action_position"] or 0
            )
            visible = {
                xyz: block
                for xyz, block in blocks.items()
                if xyz[1] <= max_height
                and (
                    not materials or f"{block.block_id}:{block.block_meta}" in materials
                )
            }
            try:
                model_path, report = self.minecraft_renderer.export(
                    visible, self._bounds(sample, source), context_blocks=blocks
                )
                current["minecraft_rendering"] = report
                if report["fallback_variants"]:
                    status += (
                        " · checkerboard fallback for unsupported variants: "
                        + ", ".join(report["fallback_variants"])
                    )
                if color_mode != "Material":
                    status += " · textures show original materials; choose Voxels for analysis colors"
                if not visible:
                    status += " · no blocks at this step or with these filters"
                    textured = False
                    figure = voxel_figure(
                        {}, self._bounds(sample, source), f"{sample_id}:{source}"
                    )
            except MinecraftAssetError as error:
                textured = False
                figure = self.render_sample(
                    sample_id,
                    source,
                    steps,
                    color_mode,
                    "Voxels",
                    materials,
                    max_height,
                )[0]
                status += f"\n\nTextures unavailable: {error}. Showing colored voxels; retry Textured blocks when assets are available."
                current["minecraft_rendering"] = {"error": str(error)}
        return (
            # Keep inactive WebGL viewers mounted and retain their last value.
            # Clearing/hard-hiding Model3D tears down its asynchronous Babylon
            # canvas; restoring the cached GLB URL can then leave a blank viewer.
            gr.update(visible="hidden")
            if textured
            else gr.update(value=figure, visible=True),
            status,
            inventory,
            current,
            gr.update(value=model_path, visible=True)
            if textured
            else gr.update(visible="hidden"),
        )

    def sample_metadata(self, sample: CraftSample) -> dict:
        entity_counts = Counter(event.entity_id for event in sample.events)
        counts = Counter(event.action for event in sample.events)
        split_names = []
        split_path = self.data_dir.parent / "splits.json"
        if split_path.is_file():
            try:
                splits = json.loads(split_path.read_text())
                split_names = [
                    name
                    for name, ids in splits.items()
                    if isinstance(ids, list) and sample.sample_id in ids
                ]
            except (OSError, ValueError, AttributeError):
                split_names = ["Could not read splits.json"]
        return {
            "house_id": sample.sample_id,
            "split": split_names or "Not provided",
            "source_metadata": sample.json_metadata,
            "recording": {
                "raw_action_rows": sample.action_count,
                "valid_actions": len(sample.events),
                "placements": counts["P"],
                "breaks": counts["B"],
                "first_tick": sample.events[0].tick if sample.events else None,
                "last_tick": sample.events[-1].tick if sample.events else None,
                "entity_action_counts": {
                    str(entity): count for entity, count in entity_counts.items()
                },
                "entity_colors": {
                    str(entity): block_color(Block(0, 0, entity_id=entity), "Player", 0)
                    for entity in sorted(entity_counts)
                },
                "elapsed_seconds": (sample.events[-1].tick - sample.events[0].tick) / 20
                if sample.events
                else None,
                "entity_note": "Entity IDs are session-local. Player names in stats.json are not mapped to entity IDs.",
            },
            "schematic": sample.schematic_info or "Not available",
            "files": [
                {"name": path.name, "bytes": path.stat().st_size}
                for path in sample.files
            ],
            "warnings": list(sample.warnings),
        }

    def select_sample(self, sample_id: str, source: str | None = None):
        try:
            sample = self.load_sample(sample_id)
            source = source or (RECORDING if sample.events else SCHEMATIC)
            low, high = self._bounds(sample, source)
            material_pairs = {
                (event.block_id, event.block_meta)
                for event in sample.events
                if event.block_id
            }
            if sample.schematic is not None:
                material_pairs.update(
                    (block.block_id, block.block_meta)
                    for block in schematic_blocks(sample.schematic).values()
                )
            choices = [
                (f"{block_id}:{meta} · {material_name(block_id)}", f"{block_id}:{meta}")
                for block_id, meta in sorted(material_pairs)
            ]
            figure, status, inventory, current, model = self.render_scene(
                sample_id,
                source,
                sample.action_count,
                "Material",
                TEXTURED,
                [],
                high[1],
            )
            previews = sorted(
                (
                    path
                    for path in sample.files
                    if path.suffix.lower() in (".png", ".jpg", ".jpeg")
                ),
                key=lambda path: (
                    int(path.stem.split(".")[-1])
                    if path.stem.split(".")[-1].isdigit()
                    else 0
                ),
            )
            rows, page_status = self.action_page(sample_id, 1)
            notes = "\n\n".join(sample.warnings)
            if not notes:
                notes = "All three core files are present."
            return (
                gr.update(value=source),
                gr.update(
                    minimum=0,
                    maximum=max(sample.action_count, 1),
                    value=sample.action_count,
                    interactive=bool(sample.events) and source == RECORDING,
                ),
                gr.update(choices=choices, value=[]),
                gr.update(
                    minimum=low[1], maximum=max(high[1], low[1] + 1), value=high[1]
                ),
                "Material",
                TEXTURED,
                figure,
                status,
                inventory,
                current,
                self.sample_metadata(sample),
                rows,
                gr.update(
                    minimum=1,
                    maximum=max(2, math.ceil(len(sample.events) / PAGE_SIZE)),
                    value=1,
                    interactive=len(sample.events) > PAGE_SIZE,
                ),
                page_status,
                [(str(path), path.name) for path in previews],
                "" if previews else "No rendered previews are included for this house.",
                [str(path) for path in sample.files],
                notes,
                model,
            )
        except (OSError, ValueError, TypeError) as error:
            raise gr.Error(
                f"Could not load this house: {error}. Select another house or check its files."
            ) from error

    def create_content(self, app: gr.Blocks) -> None:
        if not self.sample_ids:
            gr.Markdown(
                f"No house folders found in `{self.data_dir}`. Extract the dataset here and restart the app."
            )
            return
        initial = next(
            (
                name
                for name in self.sample_ids
                if name.startswith("workdir")
                and (self.data_dir / name / "placed.json").is_file()
            ),
            self.sample_ids[0],
        )
        with gr.Row():
            house = gr.Dropdown(
                self.sample_ids,
                value=initial,
                label=f"House · {len(self.sample_ids):,} folders",
                filterable=True,
                scale=6,
                min_width=250,
            )
            previous = gr.Button("Previous", scale=1, min_width=90)
            following = gr.Button("Next", scale=1, min_width=90)
            random = gr.Button("Random house", scale=1, min_width=120)
        with gr.Row():
            source = gr.Radio(
                [RECORDING, SCHEMATIC], value=RECORDING, label="Geometry source"
            )
            color = gr.Radio(
                ["Material", "Build order", "Player"],
                value="Material",
                label="Color by",
            )
            mode = gr.Radio(
                [TEXTURED, "Voxels", "Points"], value=TEXTURED, label="Rendering"
            )
        textured_model = gr.Model3D(
            label="Minecraft blocks",
            show_label=False,
            height=620,
            interactive=False,
            clear_color=(0.94, 0.96, 0.98, 1),
            elem_id="textured-object-view",
            display_mode="solid",
        )
        plot = gr.Plot(
            label="Interactive house",
            show_label=False,
            elem_id="object-view",
            visible="hidden",
        )
        status = gr.Markdown("Loading house…")
        gr.Markdown(
            "Drag to orbit · scroll to zoom. Textured blocks uses Minecraft 1.12 models and textures. Choose Voxels for block hover details and analysis colors."
        )
        with gr.Row():
            timeline = gr.Slider(
                0,
                1,
                value=1,
                step=1,
                label="Construction · actions applied",
                info="Replay edits from an empty scene. Changes to pre-existing terrain are not reconstructed.",
            )
            height = gr.Slider(
                0,
                1,
                value=1,
                step=1,
                label="Cutaway · maximum Y",
                info="Lower the ceiling to look inside.",
            )
        with gr.Accordion("Filter block materials", open=False):
            materials = gr.Dropdown(
                [],
                multiselect=True,
                label="Materials",
                info="Leave empty to show all block types.",
            )
        with gr.Accordion("House metadata & file coverage", open=False):
            notes = gr.Markdown()
            metadata = gr.JSON(label="Source metadata and derived summaries", open=True)
            current = gr.JSON(label="Current construction step", open=True)
        with gr.Accordion("Block inventory & color key", open=False):
            inventory = gr.Dataframe(
                headers=[
                    "Block ID",
                    "Meta",
                    "Material",
                    "Occupied",
                    "Visible",
                    "Material color",
                ],
                datatype=["number", "number", "str", "number", "number", "str"],
                interactive=False,
                show_search="filter",
                max_height=320,
            )
        with gr.Accordion("Construction action log", open=False):
            page = gr.Slider(1, 2, value=1, step=1, label="Action log page")
            page_status = gr.Markdown()
            actions = gr.Dataframe(
                headers=[
                    "Action",
                    "Tick",
                    "Elapsed seconds",
                    "Entity",
                    "X",
                    "Y",
                    "Z",
                    "Block ID",
                    "Meta",
                    "Edit",
                ],
                datatype=[
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "str",
                ],
                interactive=False,
                max_height=360,
                show_search="filter",
            )
        with gr.Accordion("Rendered previews", open=False):
            preview_note = gr.Markdown()
            gallery = gr.Gallery(
                label="Archive renderings", columns=4, height="auto", interactive=False
            )
        with gr.Accordion("Download original files", open=False):
            gr.Markdown(
                "Original JSON, NumPy schematics, previews, and any raw depth/block buffers are preserved. Binary buffer layouts are not specified in the dataset README."
            )
            files = gr.File(
                file_count="multiple", label="House files", interactive=False
            )

        outputs = [
            source,
            timeline,
            materials,
            height,
            color,
            mode,
            plot,
            status,
            inventory,
            current,
            metadata,
            actions,
            page,
            page_status,
            gallery,
            preview_note,
            files,
            notes,
            textured_model,
        ]
        scene_events = SceneEventOptions(
            concurrency_id="scene", concurrency_limit=1, trigger_mode="always_last"
        )
        app.load(
            self.select_sample,
            inputs=house,
            outputs=outputs,
            api_name="select_house",
            **scene_events,
        )
        house.change(
            self.select_sample,
            inputs=house,
            outputs=outputs,
            api_visibility="private",
            **scene_events,
        )
        source.input(
            self.select_sample,
            inputs=[house, source],
            outputs=outputs,
            api_visibility="private",
            **scene_events,
        )
        previous.click(
            lambda value: self.adjacent_sample(value, -1),
            house,
            house,
            api_visibility="private",
        )
        following.click(
            lambda value: self.adjacent_sample(value, 1),
            house,
            house,
            api_visibility="private",
        )
        random.click(self.random_sample, outputs=house, api_visibility="private")
        render_inputs = [house, source, timeline, color, mode, materials, height]
        render_outputs = [plot, status, inventory, current, textured_model]
        gr.on(
            triggers=[
                control.input for control in (timeline, height, color, mode, materials)
            ],
            fn=self.render_scene,
            inputs=render_inputs,
            outputs=render_outputs,
            api_name="render_house",
            **scene_events,
        )
        page.input(
            self.action_page,
            [house, page],
            [actions, page_status],
            api_name="action_page",
            **scene_events,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/houses"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--texture-pack",
        type=Path,
        help="Optional Minecraft 1.12 texture-pack ZIP or directory.",
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link."
    )
    parser.add_argument(
        "--open", action="store_true", help="Open the app in a browser."
    )
    args = parser.parse_args()
    try:
        app = Meta3DCraftVisualizationApp(args.data_dir, texture_pack=args.texture_pack)
    except ValueError as error:
        parser.error(str(error))
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.open,
    )


if __name__ == "__main__":
    main()
