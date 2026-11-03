"""Static Minecraft 1.12 block-entity and fluid meshes.

Entity boxes use the legacy ModelBox unfolded UV layout. Poses follow the 1.12
special renderers, rather than substituting modern block models or cube textures.
All vertices returned here are in block-local model units (16 units per block).
"""

import math

import numpy as np

ENTITY_IDS = {26, 54, 63, 68, 130, 146}
FLUID_IDS = {8, 9, 10, 11}


def turn(axis, degrees):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def entity_box(low, size, offset, sheet, transform, translation, texture):
    """Emit a ModelBox with exact per-face UV rectangles, including top flips."""
    x, y, z = low
    dx, dy, dz = size
    u, v = offset
    w, h = sheet
    points = (
        np.array(
            [
                [x, y, z],
                [x + dx, y, z],
                [x + dx, y + dy, z],
                [x, y + dy, z],
                [x, y, z + dz],
                [x + dx, y, z + dz],
                [x + dx, y + dy, z + dz],
                [x, y + dy, z + dz],
            ]
        )
        @ transform.T
        + translation
    )
    layout = [
        ([5, 1, 2, 6], (u + dz + dx, v + dz, u + 2 * dz + dx, v + dz + dy)),
        ([0, 4, 7, 3], (u, v + dz, u + dz, v + dz + dy)),
        ([5, 4, 0, 1], (u + dz, v, u + dz + dx, v + dz)),
        ([2, 3, 7, 6], (u + dz + dx, v + dz, u + dz + 2 * dx, v)),
        ([1, 0, 3, 2], (u + dz, v + dz, u + dz + dx, v + dz + dy)),
        ([4, 5, 6, 7], (u + 2 * dz + dx, v + dz, u + 2 * dz + 2 * dx, v + dz + dy)),
    ]
    return [
        (
            texture,
            (1.0, 1.0, 1.0),
            points[indices],
            np.array(
                [[u1 / w, v0 / h], [u0 / w, v0 / h], [u0 / w, v1 / h], [u1 / w, v1 / h]]
            ),
        )
        for indices, (u0, v0, u1, v1) in layout
    ]


def clip_chest(faces):
    """Clip a double chest to this cell, interpolating UVs at the join."""
    result = []
    for texture, tint, vertices, uv in faces:
        # Each face remains an axis-aligned rectangle after cardinal rotations.
        # An internal end face is outside the cell; crossing side faces are split.
        for axis in (0, 2):
            if vertices[:, axis].max() <= 0 or vertices[:, axis].min() >= 16:
                break
            low, high = vertices[:, axis].min(), vertices[:, axis].max()
            if high - low < 1e-8:
                continue
            for i, point in enumerate(vertices.copy()):
                clipped = np.clip(point[axis], 0, 16)
                if clipped == point[axis]:
                    continue
                j = next(
                    j
                    for j in range(4)
                    if abs(vertices[j, axis] - point[axis]) > 1e-8
                    and np.allclose(
                        np.delete(vertices[j], axis), np.delete(point, axis)
                    )
                )
                ratio = (clipped - point[axis]) / (vertices[j, axis] - point[axis])
                uv[i] += ratio * (uv[j] - uv[i])
                vertices[i, axis] = clipped
        else:
            result.append((texture, tint, vertices, uv))
    return result


def entity_faces(xyz, block, neighbors):
    """Return a static special mesh, or None for unrecognized/invalid states."""
    bid, meta = block.block_id, block.block_meta
    if bid not in ENTITY_IDS or not 0 <= meta <= 15:
        return None
    faces = []
    if bid in (54, 130, 146):
        if meta not in (2, 3, 4, 5):
            return None
        angle = {2: 180, 3: 0, 4: 90, 5: -90}[meta]
        # This matches translate(0,16,16), scale(1,-1,-1), rotate about (8,8,8).
        flip = np.diag([1, -1, -1])
        transform = flip @ turn("y", angle)
        translation = np.array([0, 16, 16]) + flip @ (
            np.full(3, 8.0) - turn("y", angle) @ np.full(3, 8.0)
        )
        paired = None
        axis = 0 if meta in (2, 3) else 2
        if bid != 130:
            for sign in (-1, 1):
                pos = list(xyz)
                pos[axis] += sign
                other = neighbors.get(tuple(pos))
                if other and (other.block_id, other.block_meta) == (bid, meta):
                    paired = sign
                    break
        width = 30 if paired else 14
        if paired:
            # The long raw X axis determines which half occupies this cell.
            world_axis = transform[:, 0]
            if world_axis[axis] * paired < 0:
                translation -= world_axis * 16
        texture = (
            "entity/chest/"
            + ({54: "normal", 130: "ender", 146: "trapped"}[bid])
            + ("_double" if paired else "")
        )
        for low, size, uv in [
            ((1, 2, 1), (width, 5, 14), (0, 0)),
            ((1, 6, 1), (width, 10, 14), (0, 19)),
            (((16 if paired else 8) - 1, 5, 0), (2, 4, 1), (0, 0)),
        ]:
            faces += entity_box(
                low,
                size,
                uv,
                (128 if paired else 64, 64),
                transform,
                translation,
                texture,
            )
        return clip_chest(faces) if paired else faces
    if bid in (63, 68):
        if bid == 68 and meta not in (2, 3, 4, 5):
            return None
        angle = -meta * 22.5 if bid == 63 else -{2: 180, 3: 0, 4: 90, 5: -90}[meta]
        yaw = turn("y", angle)
        transform = yaw @ np.diag([2 / 3, -2 / 3, -2 / 3])
        translation = np.array([8.0, 8.0, 8.0])
        if bid == 68:
            translation += yaw @ np.array([0, -5, -7])
        faces += entity_box(
            (-12, -14, -1),
            (24, 12, 2),
            (0, 0),
            (64, 32),
            transform,
            translation,
            "entity/sign",
        )
        if bid == 63:
            faces += entity_box(
                (-1, -2, -1),
                (2, 14, 2),
                (0, 14),
                (64, 32),
                transform,
                translation,
                "entity/sign",
            )
        return faces
    # Bed color is tile-entity data, absent in this dataset. Use historical red.
    head = bool(meta & 8)
    angle, offset = {
        0: (180, (16, 9, 16)),
        1: (-90, (0, 9, 16)),
        2: (0, (0, 9, 0)),
        3: (90, (16, 9, 0)),
    }[meta & 3]
    transform = turn("x", 90) @ turn("z", angle)
    faces += entity_box(
        (0, 0, 0),
        (16, 16, 6),
        (0, 0 if head else 22),
        (64, 64),
        transform,
        np.array(offset),
        "entity/bed/red",
    )
    for i in (1, 3) if head else (0, 2):
        low = [(0, 6, -16), (0, 6, 0), (-16, 6, -16), (-16, 6, 0)][i]
        leg_rotation = turn("z", [0, 90, 270, 180][i]) @ turn("x", 90)
        faces += entity_box(
            low,
            (3, 3, 3),
            (50, i * 6),
            (64, 64),
            transform @ leg_rotation,
            np.array(offset),
            "entity/bed/red",
        )
    return faces


def fluid_faces(xyz, block, context, visible, registry, face_geometry):
    """Static fluid surfaces with shared corner heights and visible-face culling."""
    bid, meta = block.block_id, block.block_meta
    if bid not in FLUID_IDS or not 0 <= meta <= 15:
        return None
    family = {8, 9} if bid in (8, 9) else {10, 11}
    name = "water" if bid in (8, 9) else "lava"

    def same(pos, source):
        other = source.get(tuple(pos))
        return other is not None and other.block_id in family

    def solid(pos, source):
        other = source.get(tuple(pos))
        info = registry.get(other.block_id, {}) if other else {}
        return info.get("boundingBox") == "block" and not info.get("transparent", True)

    def height(cx, cz):
        total, weight = 0.0, 0
        for dx in (0, -1):
            for dz in (0, -1):
                pos = (xyz[0] + cx + dx, xyz[1], xyz[2] + cz + dz)
                if same((pos[0], pos[1] + 1, pos[2]), context):
                    return 16.0
                if same(pos, context):
                    level = context[pos].block_meta
                    amount = ((0 if level >= 8 else level) + 1) / 9
                    count = 11 if level == 0 or level >= 8 else 1
                    total += amount * count
                    weight += count
                elif not solid(pos, context):
                    total += 1
                    weight += 1
        return 16 * (1 - total / weight) if weight else 16.0

    heights = {(x, z): height(x, z) for x in (0, 1) for z in (0, 1)}
    faces = []
    directions = {
        "up": (0, 1, 0),
        "down": (0, -1, 0),
        "north": (0, 0, -1),
        "south": (0, 0, 1),
        "west": (-1, 0, 0),
        "east": (1, 0, 0),
    }
    for direction, delta in directions.items():
        pos = tuple(a + b for a, b in zip(xyz, delta))
        if same(pos, visible) or solid(pos, visible):
            continue
        vertices, uv = face_geometry(direction, (0, 0, 0), (16, 16, 16))
        for i, vertex in enumerate(vertices):
            if vertex[1] > 0:
                vertex[1] = heights[(int(vertex[0] / 16), int(vertex[2] / 16))]
                if direction not in ("up", "down"):
                    uv[i, 1] = 1 - vertex[1] / 16
        texture = f"blocks/{name}_{'still' if direction in ('up', 'down') else 'flow'}"
        faces.append((texture, (1.0, 1.0, 1.0), vertices, uv))
    return faces
