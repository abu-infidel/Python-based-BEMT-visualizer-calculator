"""Loft a blade definition into a watertight-ish triangle mesh.

Frame convention
----------------
The shaft axis is ``+Z``, which is also the thrust direction, and the propeller
turns toward ``+theta``.  For a station at azimuth ``psi``:

    e_r = (cos psi, sin psi, 0)      radial, out along the blade
    e_t = (-sin psi, cos psi, 0)     tangential, the direction of motion
    e_a = (0, 0, 1)                  axial, the thrust direction

A section at blade angle ``theta`` has its chord running leading edge to
trailing edge along

    c_hat = -cos(theta) e_t - sin(theta) e_a

which puts the leading edge both *ahead* in rotation and *forward* in thrust --
the orientation that makes ``alpha = theta - phi`` positive and the lift point
in ``+Z``.  Getting this backwards yields a mesh that looks fine and is a
pusher, so it is worth being explicit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..geometry import BladeGeometry, BladeStations
from .sections import naca4_section, camber_from_zero_lift

PITCH_AXIS = 0.25   # sections rotate about the quarter chord

PART_BLADE = 0
PART_HUB = 1
PART_SPINNER = 2


@dataclass(slots=True)
class PropellerMesh:
    """Triangle mesh plus the per-vertex bookkeeping the visualiser needs."""

    vertices: np.ndarray                 # (N, 3) float32
    faces: np.ndarray                    # (M, 3) int32
    normals: np.ndarray                  # (N, 3) float32
    station_x: np.ndarray                # (N,) r/R, for mapping spanwise fields
    part: np.ndarray                     # (N,) PART_* tag
    blade_id: np.ndarray                 # (N,) which blade
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def radius(self) -> float:
        return float(np.hypot(self.vertices[:, 0], self.vertices[:, 1]).max())

    def rotated(self, angle: float) -> np.ndarray:
        """Vertices spun about the shaft axis -- one frame of the animation."""
        c, s = math.cos(angle), math.sin(angle)
        v = self.vertices
        out = np.empty_like(v)
        out[:, 0] = c * v[:, 0] - s * v[:, 1]
        out[:, 1] = s * v[:, 0] + c * v[:, 1]
        out[:, 2] = v[:, 2]
        return out

    def sample_field(self, x_stations: np.ndarray, values: np.ndarray,
                     hub_value: float | None = None) -> np.ndarray:
        """Interpolate a spanwise solver field onto every vertex.

        Blade vertices take the value at their own ``r/R``; hub and spinner
        vertices take ``hub_value`` (by default the root station's value), so a
        colour map does not paint the spinner with meaningless data.
        """
        x = np.asarray(x_stations, dtype=float)
        v = np.asarray(values, dtype=float)
        order = np.argsort(x)
        out = np.interp(self.station_x, x[order], v[order])
        if hub_value is None:
            hub_value = float(v[order][0])
        out = np.where(self.part == PART_BLADE, out, hub_value)
        return out

    def to_obj(self) -> str:
        """Wavefront OBJ text -- opens in anything."""
        lines = ["# Propwash propeller mesh",
                 f"# {self.n_vertices} vertices, {self.n_faces} faces"]
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in self.vertices]
        lines += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in self.normals]
        lines += [f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}"
                  for a, b, c in self.faces]
        return "\n".join(lines) + "\n"

    def to_stl(self) -> bytes:
        """Binary STL -- what a slicer or a CFD pre-processor wants."""
        import struct
        tris = self.vertices[self.faces]
        n = _face_normals(self.vertices, self.faces)
        buf = bytearray(b"Propwash propeller mesh".ljust(80, b"\0"))
        buf += struct.pack("<I", self.n_faces)
        for i in range(self.n_faces):
            buf += struct.pack("<3f", *n[i])
            for j in range(3):
                buf += struct.pack("<3f", *tris[i, j])
            buf += b"\0\0"
        return bytes(buf)

    def describe(self) -> str:
        lo, hi = self.bounds
        return (f"{self.n_vertices:,} vertices / {self.n_faces:,} triangles, "
                f"R={self.radius * 1000:.1f} mm, "
                f"axial extent {(hi[2] - lo[2]) * 1000:.1f} mm")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-20)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (the cross products are already area-scaled)."""
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    out = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(out, faces[:, k], fn)
    ln = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(ln, 1e-20)).astype(np.float32)


def _grid_faces(n_span: int, n_loop: int, offset: int = 0) -> np.ndarray:
    """Triangulate an ``n_span x n_loop`` lofted grid (loop closes on itself)."""
    i = np.arange(n_span - 1).reshape(-1, 1)
    j = np.arange(n_loop).reshape(1, -1)
    jn = (j + 1) % n_loop

    v00 = i * n_loop + j
    v01 = i * n_loop + jn
    v10 = (i + 1) * n_loop + j
    v11 = (i + 1) * n_loop + jn

    tri1 = np.stack([v00, v10, v11], axis=-1).reshape(-1, 3)
    tri2 = np.stack([v00, v11, v01], axis=-1).reshape(-1, 3)
    return (np.vstack([tri1, tri2]) + offset).astype(np.int32)


def _fan_faces(centre: int, ring: np.ndarray, flip: bool = False) -> np.ndarray:
    """Triangle fan closing a ring onto a single centre vertex (tip cap)."""
    n = ring.size
    a = np.full(n, centre, dtype=np.int32)
    b = ring.astype(np.int32)
    c = np.roll(ring, -1).astype(np.int32)
    return np.stack([a, c, b] if flip else [a, b, c], axis=1)


def build_blade_surface(stations: BladeStations, n_chord: int = 61,
                        azimuth: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One blade as ``(vertices, faces, station_x)``, including a tip cap."""
    n_span = stations.n

    # Per-station section shapes, reusing the polar's thickness and camber.
    loops = []
    for i in range(n_span):
        polar = stations.polars[i]
        loops.append(naca4_section(float(stations.thickness[i]),
                                   camber_from_zero_lift(getattr(polar, "alpha_0", 0.0)),
                                   camber_pos=0.4, n_points=n_chord))
    n_loop = min(len(l) for l in loops) - 1   # drop the duplicated closing point
    loops = [l[:n_loop] for l in loops]

    ca, sa = math.cos(azimuth), math.sin(azimuth)
    e_r = np.array([ca, sa, 0.0])
    e_t = np.array([-sa, ca, 0.0])
    e_a = np.array([0.0, 0.0, 1.0])

    verts = np.empty((n_span * n_loop, 3), dtype=np.float64)
    st_x = np.empty(n_span * n_loop, dtype=np.float64)

    for i in range(n_span):
        theta = float(stations.twist[i])
        chord = float(stations.chord[i])
        ct, stt = math.cos(theta), math.sin(theta)

        c_hat = -ct * e_t - stt * e_a           # leading edge -> trailing edge
        n_hat = np.cross(c_hat, e_r)            # suction side, toward +Z at zero pitch

        origin = (stations.r[i] * e_r
                  + float(stations.sweep[i]) * e_t
                  + float(stations.dihedral[i]) * e_a)

        xc = loops[i][:, 0] - PITCH_AXIS
        yc = loops[i][:, 1]
        block = (origin[None, :]
                 + (xc * chord)[:, None] * c_hat[None, :]
                 + (yc * chord)[:, None] * n_hat[None, :])
        verts[i * n_loop:(i + 1) * n_loop] = block
        st_x[i * n_loop:(i + 1) * n_loop] = stations.x[i]

    faces = _grid_faces(n_span, n_loop)

    # Cap the tip with a fan to the section centroid; the root disappears
    # inside the hub so it needs no cap.
    tip_ring = np.arange((n_span - 1) * n_loop, n_span * n_loop)
    centre = verts[tip_ring].mean(axis=0)
    verts = np.vstack([verts, centre[None, :]])
    st_x = np.append(st_x, stations.x[-1])
    faces = np.vstack([faces, _fan_faces(verts.shape[0] - 1, tip_ring)])

    return verts, faces, st_x


def surface_of_revolution(z: np.ndarray, r: np.ndarray, n_theta: int = 40,
                          close_ends: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Revolve a ``(z, r)`` profile about the shaft axis."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    ang = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)

    verts = np.stack([(r[:, None] * ca[None, :]).ravel(),
                      (r[:, None] * sa[None, :]).ravel(),
                      np.repeat(z, n_theta)], axis=1)
    faces = _grid_faces(z.size, n_theta)

    if close_ends:
        for idx, flip in ((0, True), (z.size - 1, False)):
            if r[idx] > 1e-9:
                ring = np.arange(idx * n_theta, (idx + 1) * n_theta)
                centre = np.array([0.0, 0.0, z[idx]])
                verts = np.vstack([verts, centre[None, :]])
                faces = np.vstack([faces, _fan_faces(verts.shape[0] - 1, ring, flip)])
    return verts, faces


def build_spinner(radius: float, nose_length: float, back_length: float,
                  n_axial: int = 22, n_theta: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """An ogive spinner nose blended into a short cylindrical backplate."""
    t = np.linspace(0.0, 1.0, n_axial)
    z_nose = nose_length * (1.0 - t)
    r_nose = radius * np.sqrt(np.clip(1.0 - (1.0 - t) ** 2, 0.0, 1.0)) ** 0.75

    z_back = np.linspace(0.0, -back_length, 5)[1:]
    r_back = np.full(z_back.size, radius)

    z = np.concatenate([z_nose, z_back])
    r = np.concatenate([r_nose, r_back])
    return surface_of_revolution(z, r, n_theta)


def build_propeller_mesh(geometry: BladeGeometry, n_span: int = 44, n_chord: int = 49,
                         spinner: bool = True, spacing: str = "cosine",
                         stations: BladeStations | None = None) -> PropellerMesh:
    """Assemble every blade plus the spinner into one mesh."""
    st = stations if stations is not None else geometry.discretize(n_span, spacing)

    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    all_x: list[np.ndarray] = []
    all_part: list[np.ndarray] = []
    all_blade: list[np.ndarray] = []
    offset = 0

    for b in range(geometry.n_blades):
        psi = 2.0 * math.pi * b / geometry.n_blades
        v, f, x = build_blade_surface(st, n_chord, psi)
        all_v.append(v)
        all_f.append(f + offset)
        all_x.append(x)
        all_part.append(np.full(v.shape[0], PART_BLADE, dtype=np.int8))
        all_blade.append(np.full(v.shape[0], b, dtype=np.int8))
        offset += v.shape[0]

    if spinner:
        r_spin = max(geometry.hub_radius, 0.10 * geometry.radius)
        v, f = build_spinner(r_spin, 1.35 * r_spin, 0.55 * r_spin)
        all_v.append(v)
        all_f.append(f + offset)
        all_x.append(np.full(v.shape[0], geometry.hub_radius_frac))
        all_part.append(np.full(v.shape[0], PART_SPINNER, dtype=np.int8))
        all_blade.append(np.full(v.shape[0], -1, dtype=np.int8))
        offset += v.shape[0]

    vertices = np.vstack(all_v).astype(np.float32)
    faces = np.vstack(all_f).astype(np.int32)
    station_x = np.concatenate(all_x).astype(np.float32)
    part = np.concatenate(all_part)
    blade_id = np.concatenate(all_blade)

    faces = _orient_outward(vertices, faces)
    normals = vertex_normals(vertices, faces)

    return PropellerMesh(
        vertices=vertices, faces=faces, normals=normals, station_x=station_x,
        part=part, blade_id=blade_id,
        meta={"blade": geometry.name, "n_blades": geometry.n_blades,
              "radius": geometry.radius, "n_span": st.n, "n_chord": n_chord,
              "pitch_offset_deg": math.degrees(geometry.pitch_offset)},
    )


def _orient_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Flip the winding if the lofted surface came out inside-out.

    Cheaper and more robust than reasoning about the handedness of every frame:
    compare each face normal against the outward direction from the shaft axis
    and flip the whole mesh if the majority disagree.
    """
    centroids = vertices[faces].mean(axis=1)
    radial = centroids.copy()
    radial[:, 2] = 0.0
    norm = np.linalg.norm(radial, axis=1, keepdims=True)
    radial = radial / np.maximum(norm, 1e-12)

    fn = _face_normals(vertices.astype(np.float64), faces)
    if float(np.sum(np.einsum("ij,ij->i", fn, radial))) < 0.0:
        return faces[:, ::-1].copy()
    return faces


__all__ = ["PropellerMesh", "build_propeller_mesh", "build_blade_surface",
           "build_spinner", "surface_of_revolution", "vertex_normals",
           "PART_BLADE", "PART_HUB", "PART_SPINNER"]
