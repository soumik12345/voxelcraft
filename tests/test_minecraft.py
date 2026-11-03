"""Offline checks for game asset selection and textured glTF fidelity."""

import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from voxelcraft.visualization.meta_3d_craft import (
    RECORDING,
    Block,
    Meta3DCraftVisualizationApp,
)
from voxelcraft.visualization.minecraft import (
    DIRECTIONS,
    TEXTURED,
    MinecraftAssetError,
    MinecraftAssets,
    MinecraftRenderer,
    face_geometry,
    face_uvs,
    rotation,
)


def unpack_glb(path):
    data = Path(path).read_bytes()
    magic, version, total = struct.unpack_from("<4sII", data)
    assert (magic, version, total) == (b"glTF", 2, len(data))
    length, kind = struct.unpack_from("<I4s", data, 12)
    assert kind == b"JSON"
    document = json.loads(data[20 : 20 + length])
    binary_length, kind = struct.unpack_from("<I4s", data, 20 + length)
    assert kind == b"BIN\0"
    binary = data[28 + length :]
    assert len(binary) == binary_length
    return document, binary


class MinecraftTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        cube = {
            "from": [0, 0, 0],
            "to": [16, 16, 16],
            "faces": {
                face: {"texture": "#all", "cullface": face} for face in DIRECTIONS
            },
        }
        models = {
            "cube": {"elements": [cube]},
            "stone": {"parent": "block/cube", "textures": {"all": "blocks/stone"}},
            "oak_planks": {"parent": "block/stone"},
            "spruce_planks": {
                "parent": "block/cube",
                "textures": {"all": "blocks/spruce"},
            },
            "glass": {"parent": "block/cube", "textures": {"all": "blocks/glass"}},
            "half_slab_stone": {
                "elements": [{**cube, "to": [16, 8, 16]}],
                "textures": {"all": "blocks/stone"},
            },
            "upper_slab_stone": {
                "elements": [{**cube, "from": [0, 8, 0]}],
                "textures": {"all": "blocks/stone"},
            },
        }
        blocks = [
            {"id": i, "name": name, "transparent": transparent}
            for i, name, transparent in [
                (1, "stone", False),
                (5, "planks", False),
                (20, "glass", True),
                (44, "stone_slab", True),
            ]
        ]
        textures = [
            {"name": name, "model": model}
            for name, model in [
                ("stone", "stone"),
                ("planks", "oak_planks"),
                ("glass", "glass"),
                ("stone_slab", "half_slab_stone"),
            ]
        ]
        for filename, content in [
            ("blocks.json", blocks),
            ("blocks_models.json", models),
            ("blocks_states.json", {}),
            ("blocks_textures.json", textures),
        ]:
            (self.root / filename).write_text(json.dumps(content))
        (self.root / "blocks").mkdir()
        for name in ["stone", "spruce", "glass"]:
            image = Image.new("RGBA", (16, 16), (10, 20, 30, 255))
            image.putpixel((0, 0), (255, 0, 0, 0 if name == "glass" else 255))
            image.save(self.root / "blocks" / (name + ".png"))
        self.assets = MinecraftAssets(self.root)
        self.renderer = MinecraftRenderer(self.assets)
        self.addCleanup(self.renderer.close)

    def test_cached_assets_work_without_network_and_inherit_textures(self):
        with patch(
            "urllib.request.urlopen", side_effect=AssertionError("unexpected download")
        ):
            self.assertEqual(
                self.assets.model("oak_planks")["textures"]["all"], "blocks/stone"
            )
            self.assertEqual(self.assets.texture("blocks/stone").size, (16, 16))

    def test_metadata_selects_wood_variants_and_upper_slabs(self):
        self.assertEqual(
            self.assets.applications((0, 0, 0), Block(5, 1), {})[0]["model"],
            "spruce_planks",
        )
        self.assertEqual(
            self.assets.applications((0, 0, 0), Block(44, 8), {})[0]["model"],
            "upper_slab_stone",
        )

    def test_every_face_has_outward_winding_and_unit_uvs(self):
        for direction, normal in DIRECTIONS.items():
            vertices, uv = face_geometry(direction, [0, 0, 0], [16, 16, 16])
            computed = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            self.assertGreater(np.dot(computed, normal), 0)
            self.assertEqual(set(map(tuple, uv)), {(0, 0), (0, 1), (1, 0), (1, 1)})

    def test_clockwise_face_rotation_uses_asymmetric_corners(self):
        _, uv = face_geometry("north", [0, 0, 0], [16, 16, 16])
        turned = face_uvs("north", uv, 90, np.eye(3), False)
        np.testing.assert_allclose(turned, [[0, 1], [1, 1], [1, 0], [0, 0]])

    def test_uvlock_keeps_each_face_aligned_in_world_space(self):
        for transform in (rotation("y", 90), rotation("x", 90), rotation("y", 180)):
            for direction, normal in DIRECTIONS.items():
                points, uv = face_geometry(direction, [0, 0, 0], [16, 16, 16])
                locked = face_uvs(direction, uv, 0, transform, True)
                points = (points - 8) @ transform.T + 8
                world_normal = tuple(np.rint(transform @ normal).astype(int))
                world_face = next(
                    key for key, value in DIRECTIONS.items() if value == world_normal
                )
                world, _ = face_geometry(world_face, [0, 0, 0], [16, 16, 16])
                u, v = (world[3] - world[0]) / 16, (world[1] - world[0]) / 16
                projected = np.array(
                    [
                        [np.dot((p - world[0]) / 16, u), np.dot((p - world[0]) / 16, v)]
                        for p in points
                    ]
                )
                np.testing.assert_allclose(locked, projected, atol=1e-6)

    def test_double_slab_aliases_and_unsupported_metadata(self):
        for meta, expected in [
            (1, "sandstone_normal"),
            (5, "stonebrick_normal"),
            (7, "quartz_normal"),
        ]:
            self.assertEqual(
                self.assets.applications((0, 0, 0), Block(43, meta), {})[0]["model"],
                expected,
            )
        self.assertEqual(self.assets.applications((0, 0, 0), Block(1, 99), {}), [])
        self.assertEqual(self.assets.applications((0, 0, 0), Block(20, 1), {}), [])
        self.assertEqual(
            self.assets.applications((0, 0, 0), Block(38, 1), {})[0]["model"], "orchid"
        )
        self.assertEqual(
            self.assets.applications((0, 0, 0), Block(38, 8), {})[0]["model"], "daisy"
        )
        self.assertTrue(
            self.assets.applications(
                (0, 0, 0), Block(175, 8), {(0, -1, 0): Block(175, 0)}
            )[0]["model"].endswith("_top")
        )

    def test_embedded_original_png_uvs_normals_nearest_sampler_and_face_culling(self):
        path, report = self.renderer.export(
            {(0, 0, 0): Block(1, 0), (1, 0, 0): Block(1, 0)}, ((0, 0, 0), (1, 0, 0))
        )
        document, binary = unpack_glb(path)
        self.assertEqual(report["fallback_variants"], [])
        self.assertEqual(document["samplers"][0]["magFilter"], 9728)
        self.assertEqual(document["samplers"][0]["minFilter"], 9728)
        primitive = document["meshes"][0]["primitives"][0]
        self.assertEqual(document["accessors"][primitive["indices"]]["count"], 60)
        self.assertIn("NORMAL", primitive["attributes"])
        self.assertIn("TEXCOORD_0", primitive["attributes"])
        for view in document["bufferViews"]:
            self.assertEqual(view["byteOffset"] % 4, 0)
            self.assertLessEqual(view["byteOffset"] + view["byteLength"], len(binary))
        image_view = document["bufferViews"][document["images"][0]["bufferView"]]
        png = binary[
            image_view["byteOffset"] : image_view["byteOffset"]
            + image_view["byteLength"]
        ]
        self.assertEqual(Image.open(io.BytesIO(png)).getpixel((0, 0)), (255, 0, 0, 255))

    def test_transparent_textures_keep_alpha_and_unknown_models_are_explicit(self):
        path, report = self.renderer.export(
            {(0, 0, 0): Block(20, 0), (2, 0, 0): Block(250, 4)}, ((0, 0, 0), (2, 0, 0))
        )
        document, _ = unpack_glb(path)
        self.assertIn(
            "MASK", [material["alphaMode"] for material in document["materials"]]
        )
        self.assertEqual(report["fallback_variants"], ["250:4"])
        self.assertEqual(report["textured_blocks"], 1)

    def test_empty_and_repeated_scene_exports(self):
        self.assertIsNone(self.renderer.export({}, ((0, 0, 0), (1, 1, 1)))[0])
        blocks = {(0, 0, 0): Block(1, 0)}
        first = self.renderer.export(blocks, ((0, 0, 0), (1, 1, 1)))
        self.assertEqual(first, self.renderer.export(blocks, ((0, 0, 0), (1, 1, 1))))

    def test_texture_pack_override_and_animated_first_frame(self):
        pack = self.root / "pack"
        texture = pack / "assets/minecraft/textures/blocks/stone.png"
        texture.parent.mkdir(parents=True)
        Image.new("RGBA", (16, 32), (120, 45, 50, 255)).save(texture)
        overridden = MinecraftAssets(self.root, texture_pack=pack).texture(
            "blocks/stone"
        )
        self.assertEqual(overridden.size, (16, 16))
        self.assertEqual(overridden.getpixel((0, 0)), (120, 45, 50, 255))

    def test_asset_failure_keeps_inspection_usable(self):
        houses = self.root / "houses"
        house = houses / "example"
        house.mkdir(parents=True)
        (house / "placed.json").write_text(json.dumps([[0, 1, [0, 0, 0], [1, 0], "P"]]))
        app = Meta3DCraftVisualizationApp(houses)
        self.addCleanup(app.minecraft_renderer.close)
        with patch.object(
            app.minecraft_renderer, "export", side_effect=MinecraftAssetError("offline")
        ):
            scene = app.render_scene(
                "example", RECORDING, 1, "Material", TEXTURED, [], 10
            )
        self.assertTrue(scene[0]["visible"])
        self.assertEqual(scene[4]["visible"], "hidden")
        self.assertNotIn("value", scene[4])
        self.assertIn("Textures unavailable", scene[1])
        self.assertEqual(scene[3]["visible_blocks"], 1)

    def test_textured_voxel_round_trip_keeps_cached_model_mounted(self):
        houses = self.root / "switch-houses"
        house = houses / "example"
        house.mkdir(parents=True)
        (house / "placed.json").write_text(
            json.dumps(
                [
                    [0, 1, [0, 0, 0], [1, 0], "P"],
                    [20, 1, [1, 0, 0], [20, 0], "P"],
                ]
            )
        )
        app = Meta3DCraftVisualizationApp(houses)
        self.addCleanup(app.minecraft_renderer.close)
        app.minecraft_renderer = self.renderer

        def render(mode, steps=2):
            return app.render_scene(
                "example", RECORDING, steps, "Material", mode, [], 1
            )

        initial = render(TEXTURED)
        path = initial[4]["value"]
        self.assertTrue(Path(path).is_file())
        self.assertEqual(initial[0], {"__type__": "update", "visible": "hidden"})
        for mode in ("Voxels", "Points", "Voxels"):
            inspection = render(mode)
            self.assertTrue(inspection[0]["visible"])
            # Hard-hiding or clearing this value remounts Gradio's async canvas.
            self.assertEqual(inspection[4], {"__type__": "update", "visible": "hidden"})
            restored = render(TEXTURED)
            self.assertTrue(restored[4]["visible"])
            self.assertEqual(restored[4]["value"], path)
            self.assertEqual(restored[3]["visible_blocks"], 2)

        empty = render(TEXTURED, steps=0)
        self.assertEqual(empty[4], {"__type__": "update", "visible": "hidden"})
        self.assertTrue(empty[0]["visible"])
        self.assertFalse(empty[0]["value"].data)
        self.assertIn("No blocks", empty[0]["value"].layout.annotations[0].text)
        self.assertEqual(render(TEXTURED)[4]["value"], path)


if __name__ == "__main__":
    unittest.main()
