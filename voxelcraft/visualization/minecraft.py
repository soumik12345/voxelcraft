"""Minecraft 1.12 block models/textures exported as self-contained binary glTF.

Assets are fetched lazily from pinned catalogs and the MCAsset Cloud archive
into an ignored local cache. No game images are distributed in the package.
"""

import hashlib
import io
import json
import math
import struct
import tempfile
import threading
import urllib.error
import urllib.request
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Mapping, Protocol
from zipfile import ZipFile

import numpy as np
from PIL import Image

from voxelcraft.visualization.minecraft_legacy import entity_faces, fluid_faces

ASSET_REVISION = "6c571b0825ee27b81f01127c2bd0d4ba2c89abb9"
DATA_REVISION = "bd8b5c6048e5af9694374b83f25f5b4aade619cc"
ARCHIVE_REVISION = "3360cc4a4e7be0183be2fae38622dfece4468401"
ARCHIVE_URL = f"https://raw.githubusercontent.com/InventivetalentDev/minecraft-assets/{ARCHIVE_REVISION}/assets/minecraft/textures/"
ASSET_URL = f"https://raw.githubusercontent.com/PrismarineJS/minecraft-assets/{ASSET_REVISION}/data/1.12/"
DATA_URL = f"https://raw.githubusercontent.com/PrismarineJS/minecraft-data/{DATA_REVISION}/data/pc/1.12/blocks.json"
TEXTURED = "Textured blocks"
COLORS = "white orange magenta light_blue yellow lime pink gray silver cyan purple blue brown green red black".split()
WOODS = "oak spruce birch jungle acacia dark_oak".split()
DIRECTIONS = {
    "west": (-1, 0, 0),
    "east": (1, 0, 0),
    "down": (0, -1, 0),
    "up": (0, 1, 0),
    "north": (0, 0, -1),
    "south": (0, 0, 1),
}


class BlockLike(Protocol):
    block_id: int
    block_meta: int


class MinecraftAssetError(RuntimeError):
    """A required model or texture asset could not be loaded."""


def rotation(axis: str, degrees: float) -> np.ndarray:
    """Minecraft blockstate rotations are clockwise when viewed along the axis."""
    angle = math.radians(-degrees)
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class MinecraftAssets:
    """Load legacy model inheritance and cache original texture PNGs locally."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        texture_pack: str | Path | None = None,
    ):
        self.cache_dir = Path(cache_dir or ".cache/voxelcraft/minecraft-1.12").resolve()
        self.texture_pack = (
            Path(texture_pack).expanduser().resolve() if texture_pack else None
        )
        if self.texture_pack and not self.texture_pack.exists():
            raise ValueError(f"Texture pack does not exist: {self.texture_pack}")
        self._lock = threading.RLock()

    def _fetch(self, name: str, url: str | None = None) -> bytes:
        if ".." in PurePosixPath(name).parts or name.startswith("/"):
            raise MinecraftAssetError("Invalid asset path")
        path = self.cache_dir / name
        with self._lock:
            if path.is_file():
                return path.read_bytes()
        try:
            request = urllib.request.Request(
                url or ASSET_URL + name,
                headers={"User-Agent": "voxelcraft-visualization"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()
            with self._lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=path.parent, delete=False
                ) as temporary:
                    temporary.write(content)
                    temporary_path = Path(temporary.name)
                temporary_path.replace(path)
            return content
        except (OSError, urllib.error.URLError) as error:
            raise MinecraftAssetError(
                f"Could not fetch Minecraft asset {name}: {error}"
            ) from error

    @lru_cache(maxsize=1)
    def catalogs(self):
        try:
            models = json.loads(self._fetch("blocks_models.json"))
            states = json.loads(self._fetch("blocks_states.json"))
            textures = {
                item["name"]: item
                for item in json.loads(self._fetch("blocks_textures.json"))
            }
            blocks = {
                item["id"]: item
                for item in json.loads(self._fetch("blocks.json", DATA_URL))
            }
        except (ValueError, TypeError, KeyError) as error:
            raise MinecraftAssetError(
                f"Invalid cached Minecraft asset metadata in {self.cache_dir}"
            ) from error
        return models, states, textures, blocks

    @lru_cache(maxsize=1024)
    def model(self, name: str) -> dict:
        models = self.catalogs()[0]
        name = name.removeprefix("minecraft:").removeprefix("block/")
        if name not in models:
            raise MinecraftAssetError(f"Model {name!r} is unavailable")
        child = models[name]
        parent = (
            self.model(child["parent"])
            if child.get("parent", "").startswith("block/")
            else {}
        )
        return {
            **parent,
            **child,
            "textures": {**parent.get("textures", {}), **child.get("textures", {})},
        }

    @lru_cache(maxsize=1024)
    def texture(self, name: str) -> Image.Image:
        if name == "missing":
            pixels = np.zeros((16, 16, 4), dtype=np.uint8)
            pixels[:] = [25, 25, 25, 255]
            pixels[:8, :8] = pixels[8:, 8:] = [220, 0, 220, 255]
            return Image.fromarray(pixels)
        name = name.removeprefix("minecraft:")
        if ".." in PurePosixPath(name).parts or name.startswith("/"):
            raise MinecraftAssetError("Invalid texture path")
        data = None
        if self.texture_pack:
            resource = f"assets/minecraft/textures/{name}.png"
            if self.texture_pack.is_dir():
                candidate = (self.texture_pack / resource).resolve()
                if candidate.is_relative_to(self.texture_pack) and candidate.is_file():
                    data = candidate.read_bytes()
            else:
                with ZipFile(self.texture_pack) as archive:
                    if resource in archive.namelist():
                        data = archive.read(resource)
        try:
            image = Image.open(
                io.BytesIO(
                    data
                    if data is not None
                    else self._fetch(name + ".png", ARCHIVE_URL + name + ".png")
                )
            ).convert("RGBA")
            # Animated vanilla images are vertical strips. Use their first frame.
            return (
                image.crop((0, 0, image.width, image.width))
                if name.startswith("blocks/") and image.height > image.width
                else image
            )
        except (OSError, ValueError) as error:
            raise MinecraftAssetError(f"Invalid texture {name}: {error}") from error

    def applications(
        self, xyz: tuple, block: BlockLike, neighbors: Mapping
    ) -> list[dict]:
        """Choose model variants from legacy metadata and neighboring connections."""
        models, states, textures, registry = self.catalogs()
        info = registry.get(block.block_id, {})
        name, meta = info.get("name", ""), block.block_meta
        if not 0 <= meta <= 15:
            return []
        limits = {1: 6, 3: 2, 12: 1, 19: 1, 24: 2, 97: 5, 98: 3, 155: 4, 179: 2}
        if block.block_id in limits and meta > limits[block.block_id]:
            return []
        entry = textures.get(name, {})
        model_name, properties = entry.get("model"), {}
        state_name = name
        colors = {
            35: "wool",
            95: "stained_glass",
            159: "stained_hardened_clay",
            160: "stained_glass_pane",
            171: "carpet",
            251: "concrete",
            252: "concrete_powder",
        }
        if block.block_id in colors:
            color = COLORS[meta & 15]
            suffix = colors[block.block_id]
            # Legacy model filenames differ from their blockstate names.
            candidates = [
                f"{color}_{suffix}",
                f"{suffix}_{color}",
                f"glass_{color}",
                f"hardened_clay_{color}",
            ]
            model_name = next((key for key in candidates if key in models), model_name)
            if block.block_id == 160:
                state_name = f"{color}_stained_glass_pane"
        elif block.block_id in (5, 6, 17, 18, 125, 126, 161, 162):
            species_index = (meta & 3) + (4 if block.block_id in (161, 162) else 0)
            if block.block_id in (5, 6, 125, 126):
                species_index = meta & 7
            if species_index >= len(WOODS):
                return []
            species = WOODS[species_index]
            if block.block_id in (17, 162):
                axis = meta & 12
                model_name = f"{species}_{'bark' if axis == 12 else 'log'}"
                return [
                    {
                        "model": model_name,
                        "x": 90 if axis in (4, 8) else 0,
                        "y": 90 if axis == 4 else 0,
                    }
                ]
            if block.block_id in (18, 161):
                model_name = f"{species}_leaves"
            elif block.block_id in (125, 126):
                model_name = (
                    f"{species}_planks"
                    if block.block_id == 125
                    else f"{'upper' if meta & 8 else 'half'}_slab_{species}"
                )
            else:
                model_name = (
                    f"{species}_{'sapling' if block.block_id == 6 else 'planks'}"
                )
        elif block.block_id in (43, 44):
            material = [
                "stone",
                "sandstone",
                "oak",
                "cobblestone",
                "brick",
                "stone_brick",
                "nether_brick",
                "quartz",
            ][meta & 7]
            if block.block_id == 44:
                model_name = f"{'upper' if meta & 8 else 'half'}_slab_{material}"
            else:
                model_name = {
                    "stone": "double_stone",
                    "oak": "oak_planks",
                    "brick": "brick",
                    "stone_brick": "stonebrick_normal",
                    "sandstone": "sandstone_normal",
                    "quartz": "quartz_normal",
                }.get(material, material)
        elif block.block_id == 1:
            model_name = [
                "stone",
                "granite",
                "granite_smooth",
                "diorite",
                "diorite_smooth",
                "andesite",
                "andesite_smooth",
            ][min(meta, 6)]
        elif block.block_id == 3:
            model_name = ["dirt", "coarse_dirt", "podzol"][min(meta, 2)]
        elif block.block_id == 12:
            model_name = "red_sand" if meta & 1 else "sand"
        elif block.block_id == 98:
            model_name = [
                "stonebrick_normal",
                "stonebrick_mossy",
                "stonebrick_cracked",
                "stonebrick_chiseled",
            ][meta & 3]
        elif block.block_id in (24, 179):
            prefix = "red_sandstone" if block.block_id == 179 else "sandstone"
            model_name = prefix + "_" + ["normal", "chiseled", "smooth"][min(meta, 2)]
        elif block.block_id == 155:
            model_name = ["quartz_normal", "quartz_chiseled", "quartz_lines"][
                min(meta, 2)
            ]
            if meta in (3, 4):
                return [{"model": model_name, "x": 90, "y": 90 if meta == 3 else 0}]
        elif block.block_id == 97:
            model_name = [
                "stone",
                "cobblestone",
                "stonebrick_normal",
                "stonebrick_mossy",
                "stonebrick_cracked",
                "stonebrick_chiseled",
            ][min(meta, 5)]
        elif block.block_id == 19:
            model_name = "sponge_wet" if meta else "sponge"
        elif block.block_id == 38:
            flower_names = [
                "poppy",
                "orchid",
                "allium",
                "houstonia",
                "red_tulip",
                "orange_tulip",
                "white_tulip",
                "pink_tulip",
                "daisy",
            ]
            if meta >= len(flower_names):
                return []
            model_name = flower_names[meta]
        elif block.block_id == 175:
            lower = neighbors.get((xyz[0], xyz[1] - 1, xyz[2]))
            if meta & 8 and (not lower or lower.block_id != 175):
                return []
            variant = (
                lower.block_meta
                if meta & 8 and lower and lower.block_id == 175
                else meta
            )
            if variant & 7 > 5:
                return []
            plant = ["sunflower", "syringa", "grass", "fern", "rose", "paeonia"][
                variant & 7
            ]
            model_name = f"double_{plant}_{'top' if meta & 8 else 'bottom'}"
        elif block.block_id == 31:
            if meta > 2:
                return []
            model_name = ["dead_bush", "tall_grass", "fern"][meta]
        if name.endswith("_stairs"):
            properties = {
                "facing": ["east", "west", "south", "north"][meta & 3],
                "half": "top" if meta & 4 else "bottom",
                "shape": "straight",
            }
        elif name.endswith("fence_gate"):
            facing = ["south", "west", "north", "east"][meta & 3]
            sides = (
                [(1, 0, 0), (-1, 0, 0)]
                if facing in ("north", "south")
                else [(0, 0, 1), (0, 0, -1)]
            )
            in_wall = any(
                neighbors.get(tuple(a + b for a, b in zip(xyz, delta)))
                and neighbors[tuple(a + b for a, b in zip(xyz, delta))].block_id == 139
                for delta in sides
            )
            properties = {
                "facing": facing,
                "open": "true" if meta & 4 else "false",
                "in_wall": str(in_wall).lower(),
                "powered": "true" if meta & 8 else "false",
            }
        elif name.endswith("_door") or name == "wooden_door":
            upper = bool(meta & 8)
            other = neighbors.get((xyz[0], xyz[1] + (-1 if upper else 1), xyz[2]))
            lower_meta = (
                other.block_meta
                if upper and other and other.block_id == block.block_id
                else (0 if upper else meta)
            )
            upper_meta = (
                meta
                if upper
                else (
                    other.block_meta
                    if other and other.block_id == block.block_id
                    else 8
                )
            )
            properties = {
                "facing": ["east", "south", "west", "north"][lower_meta & 3],
                "half": "upper" if upper else "lower",
                "hinge": "right" if upper_meta & 1 else "left",
                "open": "true" if lower_meta & 4 else "false",
            }
        elif name in ("torch", "redstone_torch", "unlit_redstone_torch"):
            properties = {
                "facing": {1: "east", 2: "west", 3: "south", 4: "north"}.get(meta, "up")
            }
        elif name in ("ladder", "furnace", "lit_furnace", "chest", "trapped_chest"):
            properties = {
                "facing": {2: "north", 3: "south", 4: "west", 5: "east"}.get(
                    meta, "north"
                )
            }
        elif block.block_id in (96, 167):
            properties = {
                "facing": ["south", "north", "east", "west"][meta & 3],
                "half": "top" if meta & 8 else "bottom",
                "open": "true" if meta & 4 else "false",
            }
        elif block.block_id == 78:
            properties = {"layers": str((meta & 7) + 1)}
        state = (
            states.get(state_name)
            or states.get(name)
            or states.get(entry.get("blockState"))
            or {}
        )
        handled_ids = set(colors) | {
            1,
            3,
            5,
            6,
            12,
            17,
            18,
            19,
            24,
            31,
            38,
            43,
            44,
            97,
            98,
            125,
            126,
            155,
            161,
            162,
            175,
            179,
        }
        # Never silently reuse a default model for metadata we do not interpret.
        if meta and block.block_id not in handled_ids and not properties:
            numeric_state = next(
                (
                    field
                    for field in ("age", "moisture", "stage")
                    if any(
                        key.startswith(field + "=") for key in state.get("variants", {})
                    )
                ),
                None,
            )
            if numeric_state:
                properties = {numeric_state: str(meta)}
            else:
                return []
        if "multipart" in state:
            connections = {}
            for direction in ("north", "east", "south", "west"):
                delta = DIRECTIONS[direction]
                neighbor = neighbors.get(tuple(a + b for a, b in zip(xyz, delta)))
                connections[direction] = bool(
                    neighbor
                    and (
                        neighbor.block_id == block.block_id
                        or not registry.get(neighbor.block_id, {}).get(
                            "transparent", True
                        )
                    )
                )

            def matches(condition):
                if "OR" in condition:
                    return any(matches(item) for item in condition["OR"])
                return all(
                    str(connections.get(key, False)).lower()
                    in str(value).lower().split("|")
                    for key, value in condition.items()
                )

            result = []
            for part in state["multipart"]:
                if not part.get("when") or matches(part["when"]):
                    apply = part["apply"]
                    selected = dict(apply[0] if isinstance(apply, list) else apply)
                    if block.block_id == 160:
                        selected["model"] = selected["model"].replace(
                            "white_", COLORS[meta & 15] + "_", 1
                        )
                    result.append(selected)
            return result
        if properties:
            for key, application in state.get("variants", {}).items():
                values = dict(
                    item.split("=", 1) for item in key.split(",") if "=" in item
                )
                if all(properties.get(k) == v for k, v in values.items()) and values:
                    return [
                        application[0] if isinstance(application, list) else application
                    ]
            return []
        return [{"model": model_name}] if model_name else []


def face_geometry(direction: str, low, high, uv=None):
    """Outward quads in Minecraft/glTF Y-up coordinates; UV origin is top left."""
    x0, y0, z0 = low
    x1, y1, z1 = high
    corners = {
        "down": [(x0, y0, z1), (x0, y0, z0), (x1, y0, z0), (x1, y0, z1)],
        "up": [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)],
        "north": [(x1, y1, z0), (x1, y0, z0), (x0, y0, z0), (x0, y1, z0)],
        "south": [(x0, y1, z1), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1)],
        "west": [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],
        "east": [(x1, y1, z1), (x1, y0, z1), (x1, y0, z0), (x1, y1, z0)],
    }
    defaults = {
        "down": [x0, 16 - z1, x1, 16 - z0],
        "up": [x0, z0, x1, z1],
        "north": [16 - x1, 16 - y1, 16 - x0, 16 - y0],
        "south": [x0, 16 - y1, x1, 16 - y0],
        "west": [z0, 16 - y1, z1, 16 - y0],
        "east": [16 - z1, 16 - y1, 16 - z0, 16 - y0],
    }
    u0, v0, u1, v1 = np.asarray(uv if uv is not None else defaults[direction]) / 16
    return np.array(corners[direction], dtype=float), np.array(
        [(u0, v0), (u0, v1), (u1, v1), (u1, v0)]
    )


def face_uvs(direction, uv, face_rotation, state_rotation, uvlock):
    """Apply clockwise face turns and compensate blockstate rotation in UV space."""
    uv = np.roll(uv, -(face_rotation // 90), axis=0)
    if not uvlock:
        return uv
    normal = np.rint(state_rotation @ DIRECTIONS[direction]).astype(int)
    world_face = next(
        name for name, value in DIRECTIONS.items() if tuple(normal) == value
    )
    local, _ = face_geometry(direction, [0, 0, 0], [16, 16, 16])
    world, _ = face_geometry(world_face, [0, 0, 0], [16, 16, 16])
    local_u, local_v = (
        state_rotation @ ((local[3] - local[0]) / 16),
        state_rotation @ ((local[1] - local[0]) / 16),
    )
    world_u, world_v = (world[3] - world[0]) / 16, (world[1] - world[0]) / 16
    transform = np.array(
        [
            [np.dot(local_u, world_u), np.dot(local_v, world_u)],
            [np.dot(local_u, world_v), np.dot(local_v, world_v)],
        ]
    )
    return (uv - 0.5) @ transform.T + 0.5


class MinecraftRenderer:
    """Generate textured GLBs, batched by material, with a bounded scene cache."""

    def __init__(self, assets: MinecraftAssets | None = None):
        self.assets = assets or MinecraftAssets()
        self._directory = tempfile.TemporaryDirectory(prefix="voxelcraft-textured-")
        self._scenes = OrderedDict()
        self._lock = threading.RLock()

    def close(self):
        """Release generated files after the app/server has stopped using them."""
        with self._lock:
            self._scenes.clear()
            self._directory.cleanup()

    def export(
        self,
        blocks: Mapping[tuple[int, int, int], BlockLike],
        bounds: tuple,
        *,
        context_blocks: Mapping | None = None,
    ) -> tuple[str | None, dict]:
        if not blocks:
            return None, {"textured_blocks": 0, "fallback_variants": []}
        context_blocks = blocks if context_blocks is None else context_blocks
        key = hashlib.sha256(
            repr(
                (
                    bounds,
                    sorted(
                        (xyz, b.block_id, b.block_meta) for xyz, b in blocks.items()
                    ),
                    sorted(
                        (xyz, b.block_id, b.block_meta)
                        for xyz, b in context_blocks.items()
                    ),
                )
            ).encode()
        ).hexdigest()
        with self._lock:
            if key in self._scenes:
                self._scenes.move_to_end(key)
                return self._scenes[key]
            path, report = self._export(blocks, bounds, key, context_blocks)
            self._scenes[key] = (path, report)
            if len(self._scenes) > 32:
                _, (old_path, _) = self._scenes.popitem(last=False)
                Path(old_path).unlink(missing_ok=True)
            return path, report

    def _export(self, blocks, bounds, key, context_blocks):
        groups, fallback, plans, opaque = defaultdict(list), set(), {}, set()
        origin = (np.asarray(bounds[0]) + np.asarray(bounds[1]) + 1) / 2
        for xyz, block in blocks.items():
            special = entity_faces(xyz, block, context_blocks)
            if special is None:
                special = fluid_faces(
                    xyz,
                    block,
                    context_blocks,
                    blocks,
                    self.assets.catalogs()[3],
                    face_geometry,
                )
            if special is not None:
                for texture, tint, vertices, uv in special:
                    normal = np.cross(
                        vertices[1] - vertices[0], vertices[2] - vertices[0]
                    )
                    length = np.linalg.norm(normal)
                    if length > 1e-8:
                        groups[(texture, tint)].append(
                            (vertices / 16 + xyz - origin, uv, normal / length)
                        )
                continue
            applications = self.assets.applications(xyz, block, context_blocks)
            resolved = []
            for application in applications:
                try:
                    model = self.assets.model(application["model"])
                    if not model.get("elements"):
                        continue
                    resolved.append((application, model))
                except (MinecraftAssetError, TypeError):
                    continue
            if not resolved:
                fallback.add(f"{block.block_id}:{block.block_meta}")
                resolved = [
                    (
                        {},
                        {
                            "elements": [
                                {
                                    "from": [0, 0, 0],
                                    "to": [16, 16, 16],
                                    "faces": {
                                        face: {"texture": "missing"}
                                        for face in DIRECTIONS
                                    },
                                }
                            ]
                        },
                    )
                ]
            plans[xyz] = resolved
            # Only full, opaque cubes can hide a neighboring model's cullface.
            registry = self.assets.catalogs()[3].get(block.block_id, {})
            elements = resolved[0][1]["elements"]
            if (
                len(resolved) == 1
                and len(elements) == 1
                and elements[0].get("from") == [0, 0, 0]
                and elements[0].get("to") == [16, 16, 16]
                and not registry.get("transparent", True)
            ):
                opaque.add(xyz)
        for xyz, applications in plans.items():
            for application, model in applications:
                state_rotation = rotation("y", application.get("y", 0)) @ rotation(
                    "x", application.get("x", 0)
                )
                for element in model["elements"]:
                    for direction, face in element.get("faces", {}).items():
                        if "cullface" in face:
                            delta = np.rint(
                                state_rotation @ DIRECTIONS[face["cullface"]]
                            ).astype(int)
                            if tuple(np.asarray(xyz) + delta) in opaque:
                                continue
                        texture = face["texture"]
                        seen = set()
                        while texture.startswith("#") and texture not in seen:
                            seen.add(texture)
                            texture = model.get("textures", {}).get(
                                texture[1:], "missing"
                            )
                        if texture.startswith("#"):
                            texture = "missing"
                        vertices, uv = face_geometry(
                            direction, element["from"], element["to"], face.get("uv")
                        )
                        uv = face_uvs(
                            direction,
                            uv,
                            face.get("rotation", 0),
                            state_rotation,
                            application.get("uvlock", False),
                        )
                        if "rotation" in element:
                            transform = element["rotation"]
                            pivot = np.asarray(transform["origin"])
                            vertices = (vertices - pivot) @ rotation(
                                transform["axis"], -transform["angle"]
                            ).T
                            if transform.get("rescale"):
                                scale = np.ones(3) / math.cos(
                                    math.radians(transform["angle"])
                                )
                                scale["xyz".index(transform["axis"])] = 1
                                vertices *= scale
                            vertices += pivot
                        vertices = (
                            ((vertices - 8) @ state_rotation.T + 8) / 16 + xyz - origin
                        )
                        normal = np.cross(
                            vertices[1] - vertices[0], vertices[2] - vertices[0]
                        )
                        norm = np.linalg.norm(normal)
                        if not norm:
                            continue
                        normal /= norm
                        # Grass's transparent side overlay is intentionally coplanar.
                        if face["texture"] == "#overlay":
                            vertices += normal * 0.0005
                        tint = (1.0, 1.0, 1.0)
                        if "tintindex" in face:
                            tint = (
                                (0.57, 0.74, 0.35)
                                if blocks[xyz].block_id == 2
                                else (0.47, 0.67, 0.26)
                            )
                        groups[(texture, tint)].append((vertices, uv, normal))
        texture_names = sorted({texture for texture, _ in groups})
        with ThreadPoolExecutor(max_workers=8) as pool:
            images = dict(
                zip(texture_names, pool.map(self.assets.texture, texture_names))
            )
        report = {
            "textured_blocks": sum(
                f"{b.block_id}:{b.block_meta}" not in fallback for b in blocks.values()
            ),
            "fallback_variants": sorted(fallback),
            "minecraft_version": "1.12",
            "asset_revision": ASSET_REVISION,
            "texture_archive_revision": ARCHIVE_REVISION,
            "texture_pack": str(self.assets.texture_pack)
            if self.assets.texture_pack
            else "Minecraft 1.12.2 default (MCAsset Cloud archive)",
            "notes": "Static models and fluid surfaces; closed chests. Beds default to red and signs are blank because tile-entity data is absent. Biome tint is illustrative; animated textures use their first frame. Unsupported variants use a checkerboard fallback.",
        }
        path = Path(self._directory.name) / f"{key}.glb"
        path.write_bytes(
            build_glb(groups, images, {"minecraft_origin": origin.tolist(), **report})
        )
        return str(path), report


def build_glb(groups: dict, images: dict, extras: dict) -> bytes:
    """Encode glTF 2.0 with embedded PNGs and nearest-neighbor samplers."""
    binary, views, accessors, materials, textures, image_defs, primitives = (
        bytearray(),
        [],
        [],
        [],
        [],
        [],
        [],
    )

    def view(data, target=None):
        binary.extend(b"\0" * (-len(binary) % 4))
        item = {"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)}
        if target:
            item["target"] = target
        binary.extend(data)
        views.append(item)
        return len(views) - 1

    def accessor(array, kind, component=5126, target=34962):
        item = {
            "bufferView": view(array.tobytes(), target),
            "componentType": component,
            "count": len(array),
            "type": kind,
        }
        if kind == "VEC3":
            item.update(min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())
        accessors.append(item)
        return len(accessors) - 1

    image_indices = {}
    for name, image in images.items():
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_indices[name] = len(textures)
        textures.append({"source": len(image_defs), "sampler": 0})
        image_defs.append(
            {
                "bufferView": view(buffer.getvalue()),
                "mimeType": "image/png",
                "name": name,
            }
        )
    for (texture, tint), faces in groups.items():
        image = images[texture]
        alpha = np.asarray(image)[..., 3]
        alpha_mode = (
            "OPAQUE"
            if np.all(alpha == 255)
            else ("BLEND" if np.any((alpha > 0) & (alpha < 255)) else "MASK")
        )
        materials.append(
            {
                "name": texture,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": image_indices[texture]},
                    "baseColorFactor": [*tint, 1],
                    "metallicFactor": 0,
                    "roughnessFactor": 1,
                },
                "alphaMode": alpha_mode,
                "alphaCutoff": 0.5,
                "doubleSided": True,
            }
        )
        positions = np.concatenate([face[0] for face in faces]).astype("<f4")
        uv = np.concatenate([face[1] for face in faces]).astype("<f4")
        normals = np.repeat(np.asarray([face[2] for face in faces]), 4, axis=0).astype(
            "<f4"
        )
        indices = (
            (np.arange(len(faces), dtype=np.uint32)[:, None] * 4 + [0, 1, 2, 0, 2, 3])
            .astype("<u4")
            .reshape(-1)
        )
        primitives.append(
            {
                "attributes": {
                    "POSITION": accessor(positions, "VEC3"),
                    "NORMAL": accessor(normals, "VEC3"),
                    "TEXCOORD_0": accessor(uv, "VEC2"),
                },
                "indices": accessor(indices, "SCALAR", 5125, 34963),
                "material": len(materials) - 1,
            }
        )
    document = {
        "asset": {
            "version": "2.0",
            "generator": "voxelcraft",
            "copyright": "Minecraft textures and models: Mojang / Microsoft",
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": primitives}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
        "images": image_defs,
        "textures": textures,
        "materials": materials,
        "samplers": [
            {"magFilter": 9728, "minFilter": 9728, "wrapS": 10497, "wrapT": 10497}
        ],
        "extras": extras,
    }
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    return (
        struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary))
        + struct.pack("<I4s", len(encoded), b"JSON")
        + encoded
        + struct.pack("<I4s", len(binary), b"BIN\0")
        + binary
    )
