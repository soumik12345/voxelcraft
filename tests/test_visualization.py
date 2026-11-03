"""Geometry, dataset fidelity, and Gradio callback checks without a browser."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from voxelcraft.visualization import BaseVisualizationApp, Meta3DCraftVisualizationApp
from voxelcraft.visualization.meta_3d_craft import (
    RECORDING,
    SCHEMATIC,
    Block,
    CraftEvent,
    replay_events,
    schematic_blocks,
    voxel_figure,
)


class GeometryTests(unittest.TestCase):
    def test_replay_replacement_removal_air_and_zero_step(self):
        events = tuple(
            CraftEvent(i, i * 20, 7, xyz, block_id, 0, action)
            for i, (xyz, block_id, action) in enumerate(
                [
                    ((-4, 63, 9), 1, "P"),
                    ((-4, 63, 9), 5, "P"),
                    ((5, 63, 9), 1, "P"),
                    ((-4, 63, 9), 5, "B"),
                    ((5, 63, 9), 0, "P"),
                    ((100, 60, 100), 1, "B"),
                ],
                1,
            )
        )
        self.assertEqual(replay_events(events, 0), {})
        self.assertEqual(replay_events(events, 2)[(-4, 63, 9)].block_id, 5)
        self.assertEqual(set(replay_events(events, 4)), {(5, 63, 9)})
        self.assertEqual(replay_events(events, 6), {})

    def test_schematic_axis_order_and_signed_ids(self):
        array = np.zeros((2, 3, 4, 2), dtype=np.int8)
        array[1, 2, 3] = (-1, 2)
        self.assertEqual(schematic_blocks(array), {(3, 1, 2): Block(255, 2)})

    def test_mesh_culls_shared_faces_and_keeps_original_xyz_for_hover(self):
        blocks = {(0, 63, 0): Block(1, 0), (1, 63, 0): Block(5, 0)}
        figure = voxel_figure(blocks, ((0, 63, 0), (1, 63, 0)), "house")
        mesh = figure.data[0]
        self.assertEqual(len(figure.data), 1)
        self.assertEqual(len(mesh.i), 20)
        self.assertTrue(mesh.flatshading)
        self.assertEqual(mesh.opacity, 1)
        self.assertEqual(tuple(mesh.customdata[0][:3]), (0, 63, 0))
        self.assertEqual(min(mesh.z), 63)
        self.assertEqual(figure.layout.scene.aspectmode, "data")

    def test_triangle_normals_face_outward_after_axis_swap(self):
        mesh = voxel_figure(
            {(0, 0, 0): Block(1, 0)}, ((0, 0, 0), (0, 0, 0)), "one"
        ).data[0]
        vertices = np.array([mesh.x, mesh.y, mesh.z]).T
        for i, j, k in zip(mesh.i, mesh.j, mesh.k):
            triangle = vertices[[i, j, k]]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            self.assertGreater(np.dot(normal, triangle.mean(axis=0) - 0.5), 0)

    def test_empty_scene_and_points(self):
        empty = voxel_figure({}, ((0, 0, 0), (5, 5, 5)), "empty")
        self.assertFalse(empty.data)
        self.assertIn("No blocks", empty.layout.annotations[0].text)
        points = voxel_figure(
            {(0, 0, 0): Block(1, 0)},
            ((0, 0, 0), (1, 1, 1)),
            "one",
            render_mode="Points",
        )
        self.assertEqual(points.data[0].type, "scatter3d")


class DatasetTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        house = self.root / "house-a"
        house.mkdir()
        self.events = [[i * 20, 55, [i, 63, 0], [5, 2], "P"] for i in range(205)]
        (house / "placed.json").write_text(json.dumps(self.events))
        (house / "stats.json").write_text(
            json.dumps({"player_minutes": {"Builder": 3}, "custom": [1, 2]})
        )
        array = np.zeros((2, 3, 4, 2), dtype=np.uint8)
        array[1, 2, 3] = (5, 2)
        np.save(house / "schematic.npy", array)
        (house / "depth.0.bin").write_bytes(b"original binary")
        (self.root / "empty-house").mkdir()
        self.app = Meta3DCraftVisualizationApp(self.root)
        self.addCleanup(self.app.minecraft_renderer.close)
        # Dataset/callback tests stay offline; renderer fidelity has its own fixtures.
        renderer = patch.object(
            self.app.minecraft_renderer,
            "export",
            return_value=(
                None,
                {
                    "textured_blocks": 0,
                    "fallback_variants": [],
                },
            ),
        )
        renderer.start()
        self.addCleanup(renderer.stop)

    def test_base_discovery_navigation_and_path_boundaries(self):
        self.assertIsInstance(self.app, BaseVisualizationApp)
        self.assertEqual(self.app.sample_ids, ("empty-house", "house-a"))
        self.assertEqual(self.app.adjacent_sample("house-a", 1), "empty-house")
        with self.assertRaises(ValueError):
            self.app.sample_directory("../outside")

    def test_source_metadata_and_original_files_are_available(self):
        sample = self.app.load_sample("house-a")
        metadata = self.app.sample_metadata(sample)
        self.assertEqual(metadata["source_metadata"]["stats.json"]["custom"], [1, 2])
        self.assertEqual(metadata["schematic"]["shape"], [2, 3, 4, 2])
        self.assertEqual(metadata["recording"]["entity_action_counts"], {"55": 205})
        import gradio as gr

        gr.JSON().postprocess(metadata)  # Exercise Gradio's stricter JSON encoder.
        self.assertIn("depth.0.bin", {item["name"] for item in metadata["files"]})
        self.assertEqual(sample.warnings, ())

    def test_action_log_pagination_includes_tail_and_tick_conversion(self):
        first, _ = self.app.action_page("house-a", 1)
        last, status = self.app.action_page("house-a", 3)
        self.assertEqual(len(first), 100)
        self.assertEqual(first[1][2], 1)
        self.assertEqual([row[0] for row in last], [201, 202, 203, 204, 205])
        self.assertIn("Page 3 / 3", status)

    def test_filters_cutaway_and_fixed_bounds(self):
        full = self.app.render_sample(
            "house-a", RECORDING, 205, "Material", "Voxels", [], 63
        )
        early = self.app.render_sample(
            "house-a", RECORDING, 2, "Build order", "Voxels", [], 63
        )
        self.assertEqual(full[3]["occupied_blocks"], 205)
        self.assertEqual(early[3]["occupied_blocks"], 2)
        self.assertEqual(
            full[0].layout.scene.xaxis.range, early[0].layout.scene.xaxis.range
        )
        self.assertEqual(full[0].layout.uirevision, early[0].layout.uirevision)
        filtered = self.app.render_sample(
            "house-a", RECORDING, 205, "Material", "Points", ["1:0"], 63
        )
        self.assertEqual(filtered[3]["visible_blocks"], 0)
        cut = self.app.render_sample(
            "house-a", RECORDING, 205, "Material", "Voxels", [], 62
        )
        self.assertEqual(cut[3]["visible_blocks"], 0)

    def test_callbacks_reset_controls_for_recording_and_schematic(self):
        result = self.app.select_sample("house-a")
        self.assertEqual(result[0]["value"], RECORDING)
        self.assertEqual(result[1]["maximum"], 205)
        self.assertEqual(result[2]["value"], [])
        self.assertEqual(len(result), 19)
        self.assertEqual(result[9]["last_action"]["index"], 205)
        schematic = self.app.select_sample("house-a", SCHEMATIC)
        self.assertFalse(schematic[1]["interactive"])
        self.assertEqual(schematic[9]["occupied_blocks"], 1)
        self.assertEqual(schematic[3]["maximum"], 1)

    def test_missing_files_keep_metadata_and_empty_geometry_browsable(self):
        result = self.app.select_sample("empty-house")
        self.assertIn("placed.json is missing", result[17])
        self.assertEqual(result[9]["visible_blocks"], 0)
        self.assertEqual(result[16], [])

    def test_malformed_rows_reported_with_original_action_numbers(self):
        (self.root / "house-a" / "placed.json").write_text(
            json.dumps([self.events[0], ["bad"], self.events[1]])
        )
        sample = self.app.load_sample("house-a")
        self.assertEqual([event.index for event in sample.events], [1, 3])
        self.assertIn("skipped 1 invalid", sample.warnings[0])
        self.assertEqual(len(replay_events(sample.events, 2)), 1)

    def test_bad_schematic_does_not_hide_other_metadata(self):
        np.save(self.root / "house-a" / "schematic.npy", np.ones((3, 3)))
        sample = self.app.load_sample("house-a")
        self.assertIsNone(sample.schematic)
        self.assertEqual(len(sample.events), 205)
        self.assertTrue(any("schematic.npy" in note for note in sample.warnings))

    def test_schematic_only_house_defaults_to_schematic(self):
        (self.root / "house-a" / "placed.json").unlink()
        result = self.app.select_sample("house-a")
        self.assertEqual(result[0]["value"], SCHEMATIC)
        self.assertEqual(result[9]["occupied_blocks"], 1)

    def test_build_reuse_and_empty_dataset(self):
        app = self.app.build()
        self.assertIs(self.app.build(), app)
        with tempfile.TemporaryDirectory() as root:
            empty = Meta3DCraftVisualizationApp(root)
            self.addCleanup(empty.minecraft_renderer.close)
            self.assertEqual(empty.sample_ids, ())
            self.assertIsNone(empty.random_sample())
            self.assertIsNotNone(empty.build())

    def test_scene_callbacks_share_a_queue_and_sliders_support_keyboard_input(self):
        app = self.app.build()
        callbacks = [
            fn
            for fn in app.fns.values()
            if fn.fn
            in (self.app.select_sample, self.app.render_scene, self.app.action_page)
        ]
        self.assertTrue(callbacks)
        self.assertTrue(
            all(
                fn.concurrency_id == "scene" and fn.concurrency_limit == 1
                for fn in callbacks
            )
        )
        render = next(fn for fn in callbacks if fn.fn == self.app.render_scene)
        self.assertEqual(len(render.targets), 5)
        self.assertTrue(all(event == "input" for _, event in render.targets))


if __name__ == "__main__":
    unittest.main()
