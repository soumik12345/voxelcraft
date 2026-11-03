"""Coverage for legacy special meshes, independently of downloaded game assets."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image

from voxelcraft.visualization.meta_3d_craft import Block
from voxelcraft.visualization.minecraft import (
    MinecraftAssets,
    MinecraftRenderer,
    face_geometry,
)
from voxelcraft.visualization.minecraft_legacy import (
    entity_box,
    entity_faces,
    fluid_faces,
)


class LegacyMeshesTests(unittest.TestCase):
    def test_legacy_box_uvs_use_sheet_dimensions_and_top_flip(self):
        faces = entity_box(
            (1, 2, 1), (14, 5, 14), (0, 0), (64, 64), np.eye(3), np.zeros(3), "chest"
        )
        np.testing.assert_allclose(
            faces[0][3], np.array([[42, 14], [28, 14], [28, 19], [42, 19]]) / 64
        )
        np.testing.assert_allclose(
            faces[3][3], np.array([[42, 14], [28, 14], [28, 0], [42, 0]]) / 64
        )
        for _, _, vertices, _ in faces:
            normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            self.assertGreater(np.dot(normal, vertices.mean(axis=0) - [8, 4.5, 8]), 0)

    def test_chest_latches_face_metadata_direction(self):
        for bid in (54, 130, 146):
            for meta, axis, coordinate in (
                (2, 2, 0),
                (3, 2, 16),
                (4, 0, 0),
                (5, 0, 16),
            ):
                faces = entity_faces((0, 0, 0), Block(bid, meta), {})
                latch = np.concatenate([f[2] for f in faces[-6:]])
                self.assertAlmostEqual(
                    latch[:, axis].min() if coordinate == 0 else latch[:, axis].max(),
                    coordinate,
                )
        self.assertIsNone(entity_faces((0, 0, 0), Block(54, 1), {}))

    def test_double_chest_halves_have_continuous_geometry_and_uvs(self):
        for meta, axis in ((2, 0), (3, 0), (4, 2), (5, 2)):
            xyz = tuple(1 if i == axis else 0 for i in range(3))
            context = {(0, 0, 0): Block(54, meta), xyz: Block(54, meta)}
            halves = [entity_faces(pos, b, context) for pos, b in context.items()]
            for half in halves:
                points = np.concatenate([f[2] for f in half])
                self.assertGreaterEqual(points[:, axis].min(), -1e-8)
                self.assertLessEqual(points[:, axis].max(), 16 + 1e-8)
                self.assertTrue(all(f[0].endswith("normal_double") for f in half))
                for _, _, vertices, uv in half:
                    self.assertTrue(np.isfinite(uv).all())
                    self.assertGreater(
                        np.linalg.norm(
                            np.cross(
                                vertices[1] - vertices[0], vertices[2] - vertices[0]
                            )
                        ),
                        1e-8,
                    )
            # At the shared plane, each face must sample the same atlas location.
            seam = []
            for half, pos in zip(halves, context):
                entries = set()
                for _, _, vertices, uv in half:
                    world = vertices + np.array(pos) * 16
                    normal = np.cross(world[1] - world[0], world[2] - world[0])
                    normal /= np.linalg.norm(normal)
                    for point, tex in zip(world, uv):
                        if abs(point[axis] - 16) < 1e-7:
                            entries.add(tuple(np.round(np.r_[point, normal, tex], 6)))
                seam.append(entries)
            self.assertEqual(seam[0], seam[1])

    def test_bed_halves_have_two_legs_and_stay_in_their_cells(self):
        for meta in range(16):
            faces = entity_faces((0, 0, 0), Block(26, meta), {})
            self.assertEqual(len(faces), 18)
            points = np.concatenate([f[2] for f in faces])
            np.testing.assert_allclose(points.min(axis=0), (0, 0, 0), atol=1e-8)
            np.testing.assert_allclose(points.max(axis=0), (16, 9, 16), atol=1e-8)
            self.assertTrue(all(f[0] == "entity/bed/red" for f in faces))

    def test_wall_signs_touch_support_edge_and_have_no_post(self):
        for meta, axis, coordinate in ((2, 2, 15), (3, 2, 1), (4, 0, 15), (5, 0, 1)):
            faces = entity_faces((0, 0, 0), Block(68, meta), {})
            self.assertEqual(len(faces), 6)
            points = np.concatenate([f[2] for f in faces])
            self.assertAlmostEqual(points[:, axis].mean(), coordinate)
        self.assertEqual(len(entity_faces((0, 0, 0), Block(63, 7), {})), 12)

    def test_fluid_shared_edges_levels_and_filter_reveal(self):
        a, b = (0, 0, 0), (1, 0, 0)
        context = {a: Block(8, 0), b: Block(9, 5)}
        left = fluid_faces(a, context[a], context, context, {}, face_geometry)
        right = fluid_faces(b, context[b], context, context, {}, face_geometry)
        self.assertEqual((len(left), len(right)), (5, 5))
        left_top, right_top = left[0][2], right[0][2]
        np.testing.assert_allclose(
            sorted(left_top[left_top[:, 0] == 16, 1]),
            sorted(right_top[right_top[:, 0] == 0, 1]),
        )
        self.assertGreater(left_top[:, 1].mean(), right_top[:, 1].mean())
        revealed = fluid_faces(
            a, context[a], context, {a: context[a]}, {}, face_geometry
        )
        self.assertEqual(len(revealed), 6)
        np.testing.assert_allclose(revealed[0][2], left_top)
        stacked = {a: Block(8, 0), (0, 1, 0): Block(9, 0)}
        faces = fluid_faces(a, stacked[a], stacked, {a: stacked[a]}, {}, face_geometry)
        np.testing.assert_allclose(faces[0][2][:, 1], 16)
        self.assertIsNone(fluid_faces(a, Block(8, 16), {}, {}, {}, face_geometry))

    def test_entity_sheets_preserved_and_resource_packs_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cached = root / "entity/chest"
            cached.mkdir(parents=True)
            Image.new("RGBA", (64, 128), (100, 50, 20, 255)).save(cached / "normal.png")
            self.assertEqual(
                MinecraftAssets(root).texture("entity/chest/normal").size, (64, 128)
            )
            pack = root / "pack"
            sheet = pack / "assets/minecraft/textures/entity/sign.png"
            sheet.parent.mkdir(parents=True)
            Image.new("RGBA", (128, 64), (10, 20, 30, 255)).save(sheet)
            zipped = root / "pack.zip"
            with ZipFile(zipped, "w") as archive:
                archive.write(sheet, "assets/minecraft/textures/entity/sign.png")
            for source in (pack, zipped):
                with patch(
                    "urllib.request.urlopen", side_effect=AssertionError("network")
                ):
                    self.assertEqual(
                        MinecraftAssets(root, source).texture("entity/sign").size,
                        (128, 64),
                    )

    def test_reported_variants_export_without_fallback_offline(self):
        blocks = {
            (i * 2, 0, 0): Block(bid, meta)
            for i, (bid, meta) in enumerate(
                [
                    (130, 3),
                    (26, 10),
                    (26, 2),
                    (54, 4),
                    (54, 5),
                    (68, 2),
                    (68, 3),
                    (68, 4),
                    (68, 5),
                    (8, 0),
                    (10, 0),
                    (146, 2),
                ]
            )
        }
        with tempfile.TemporaryDirectory() as tmp:
            assets = MinecraftAssets(tmp)
            renderer = MinecraftRenderer(assets)
            self.addCleanup(renderer.close)
            with (
                patch.object(assets, "catalogs", return_value=({}, {}, {}, {})),
                patch.object(
                    assets,
                    "texture",
                    return_value=Image.new("RGBA", (64, 64), (20, 50, 70, 255)),
                ),
            ):
                path, report = renderer.export(blocks, ((0, 0, 0), (24, 1, 1)))
                self.assertEqual(report["fallback_variants"], [])
                self.assertEqual(report["textured_blocks"], len(blocks))
                self.assertGreater(Path(path).stat().st_size, 1000)
