"""
Mesh generation and STL export.

Converts a relief depth map (in mm) to a watertight 3D mesh suitable
for CNC carving. The mesh includes:
- Top surface: the relief portrait
- Bottom surface: flat base
- Side walls: connecting top and bottom

Output is a binary STL file.
"""

import io
import numpy as np
import stl
from stl import mesh as stl_mesh
from scipy import ndimage


class MeshGenerator:
    """Generates watertight 3D meshes from relief depth maps."""

    def generate(
        self,
        relief_map: np.ndarray,
        width_mm: float = 150.0,
        height_mm: float | None = None,
        base_thickness_mm: float = 2.0,
        mesh_resolution: int | None = None,
        smooth_normals: bool = True,
    ) -> stl_mesh.Mesh:
        """
        Generate a watertight 3D mesh from a relief depth map.

        Args:
            relief_map: Relief depth map in mm (float32). Values are height
                        above the base plane.
            width_mm: Physical width of the output in mm.
            height_mm: Physical height. None = auto from aspect ratio.
            base_thickness_mm: Thickness of the flat base in mm.
            mesh_resolution: Max grid dimension for mesh vertices.
                           None = use full resolution. Lower = fewer triangles.
            smooth_normals: Apply Laplacian smoothing to surface normals.

        Returns:
            numpy-stl Mesh object ready for STL export.
        """
        h_px, w_px = relief_map.shape

        # Calculate physical dimensions
        if height_mm is None:
            height_mm = width_mm * (h_px / w_px)

        # Optionally downsample for mesh complexity control
        if mesh_resolution is not None and max(h_px, w_px) > mesh_resolution:
            scale = mesh_resolution / max(h_px, w_px)
            new_w = max(2, int(w_px * scale))
            new_h = max(2, int(h_px * scale))
            # Use order=3 (cubic) for smooth interpolation
            relief_map = ndimage.zoom(
                relief_map, (new_h / h_px, new_w / w_px), order=3
            )
            h_px, w_px = relief_map.shape

        # Light smoothing to prevent jagged surfaces at lower resolutions
        if smooth_normals:
            relief_map = ndimage.gaussian_filter(relief_map, sigma=0.5)

        # Create vertex grid
        vertices, top_faces = self._create_top_surface(
            relief_map, width_mm, height_mm, base_thickness_mm
        )

        # Create bottom (flat base) surface
        bottom_vertices, bottom_faces = self._create_bottom_surface(
            h_px, w_px, width_mm, height_mm
        )

        # Create side walls
        side_vertices, side_faces = self._create_side_walls(
            relief_map, width_mm, height_mm, base_thickness_mm
        )

        # Combine all geometry
        combined_mesh = self._combine_geometry(
            vertices, top_faces,
            bottom_vertices, bottom_faces,
            side_vertices, side_faces,
        )

        return combined_mesh

    def _create_top_surface(
        self,
        relief: np.ndarray,
        width_mm: float,
        height_mm: float,
        base_thickness: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create the relief top surface as a triangle mesh."""
        h, w = relief.shape

        # Create grid coordinates in mm
        x = np.linspace(0, width_mm, w)
        y = np.linspace(0, height_mm, h)
        xx, yy = np.meshgrid(x, y)

        # Z = relief height + base thickness
        zz = relief + base_thickness

        # Flatten to vertex array: (h*w, 3)
        vertices = np.column_stack([
            xx.ravel(),
            yy.ravel(),
            zz.ravel(),
        ])

        # Create triangle indices
        # Each grid cell becomes 2 triangles
        faces = []
        for row in range(h - 1):
            for col in range(w - 1):
                # Vertex indices for this cell
                tl = row * w + col       # top-left
                tr = row * w + col + 1   # top-right
                bl = (row + 1) * w + col  # bottom-left
                br = (row + 1) * w + col + 1  # bottom-right

                # Two triangles per cell (consistent winding for outward normals)
                faces.append([tl, bl, tr])
                faces.append([tr, bl, br])

        return vertices, np.array(faces, dtype=np.int32)

    def _create_bottom_surface(
        self,
        h: int,
        w: int,
        width_mm: float,
        height_mm: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create the flat bottom surface."""
        # Just 4 corner vertices for the flat bottom
        vertices = np.array([
            [0, 0, 0],
            [width_mm, 0, 0],
            [0, height_mm, 0],
            [width_mm, height_mm, 0],
        ], dtype=np.float64)

        # Two triangles (winding for downward-facing normals)
        faces = np.array([
            [0, 1, 2],
            [1, 3, 2],
        ], dtype=np.int32)

        return vertices, faces

    def _create_side_walls(
        self,
        relief: np.ndarray,
        width_mm: float,
        height_mm: float,
        base_thickness: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create side walls connecting top surface to bottom plane."""
        h, w = relief.shape
        vertices = []
        faces = []
        v_offset = 0

        x_coords = np.linspace(0, width_mm, w)
        y_coords = np.linspace(0, height_mm, h)

        # Bottom edge (y=0): row 0
        for i in range(w - 1):
            z_left = relief[0, i] + base_thickness
            z_right = relief[0, i + 1] + base_thickness
            x_left = x_coords[i]
            x_right = x_coords[i + 1]

            quad_verts = [
                [x_left, 0, z_left],    # top-left
                [x_right, 0, z_right],  # top-right
                [x_left, 0, 0],          # bottom-left
                [x_right, 0, 0],        # bottom-right
            ]
            vertices.extend(quad_verts)
            faces.append([v_offset, v_offset + 2, v_offset + 1])
            faces.append([v_offset + 1, v_offset + 2, v_offset + 3])
            v_offset += 4

        # Top edge (y=height): last row
        for i in range(w - 1):
            z_left = relief[h - 1, i] + base_thickness
            z_right = relief[h - 1, i + 1] + base_thickness
            x_left = x_coords[i]
            x_right = x_coords[i + 1]

            quad_verts = [
                [x_left, height_mm, z_left],
                [x_right, height_mm, z_right],
                [x_left, height_mm, 0],
                [x_right, height_mm, 0],
            ]
            vertices.extend(quad_verts)
            faces.append([v_offset, v_offset + 1, v_offset + 2])
            faces.append([v_offset + 1, v_offset + 3, v_offset + 2])
            v_offset += 4

        # Left edge (x=0): column 0
        for j in range(h - 1):
            z_top = relief[j, 0] + base_thickness
            z_bottom = relief[j + 1, 0] + base_thickness
            y_top = y_coords[j]
            y_bottom = y_coords[j + 1]

            quad_verts = [
                [0, y_top, z_top],
                [0, y_bottom, z_bottom],
                [0, y_top, 0],
                [0, y_bottom, 0],
            ]
            vertices.extend(quad_verts)
            faces.append([v_offset, v_offset + 1, v_offset + 2])
            faces.append([v_offset + 1, v_offset + 3, v_offset + 2])
            v_offset += 4

        # Right edge (x=width): last column
        for i in range(h - 1):
            z_top = relief[i, w - 1] + base_thickness
            z_bottom = relief[i + 1, w - 1] + base_thickness
            y_top = y_coords[i]
            y_bottom = y_coords[i + 1]

            quad_verts = [
                [width_mm, y_top, z_top],
                [width_mm, y_bottom, z_bottom],
                [width_mm, y_top, 0],
                [width_mm, y_bottom, 0],
            ]
            vertices.extend(quad_verts)
            faces.append([v_offset, v_offset + 2, v_offset + 1])
            faces.append([v_offset + 1, v_offset + 2, v_offset + 3])
            v_offset += 4

        return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)

    def _combine_geometry(
        self,
        top_v, top_f,
        bottom_v, bottom_f,
        side_v, side_f,
    ) -> stl_mesh.Mesh:
        """Combine top, bottom, and side geometry into a single STL mesh."""
        # Offset face indices
        top_offset = 0
        bottom_offset = len(top_v)
        side_offset = bottom_offset + len(bottom_v)

        bottom_f_offset = bottom_f + bottom_offset
        side_f_offset = side_f + side_offset

        # Combine all vertices and faces
        all_vertices = np.vstack([top_v, bottom_v, side_v])
        all_faces = np.vstack([top_f, bottom_f_offset, side_f_offset])

        # Create STL mesh
        stl = stl_mesh.Mesh(np.zeros(len(all_faces), dtype=stl_mesh.Mesh.dtype))

        for i, face in enumerate(all_faces):
            for j in range(3):
                stl.vectors[i][j] = all_vertices[face[j]]

        return stl

    def save_stl(
        self,
        mesh: stl_mesh.Mesh,
        filepath: str,
        binary: bool = True,
    ):
        """Save mesh to STL file."""
        if binary:
            mesh.save(filepath, mode=stl.Mode.BINARY)
        else:
            mesh.save(filepath, mode=stl.Mode.ASCII)

    def to_bytes(self, mesh: stl_mesh.Mesh) -> bytes:
        """Export mesh to STL bytes (for API responses)."""
        buf = io.BytesIO()
        mesh.save("output.stl", fh=buf, mode=stl.Mode.BINARY)
        return buf.getvalue()
