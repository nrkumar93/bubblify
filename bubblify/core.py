"""Core data structures and enhanced URDF visualizer for Bubblify."""

from __future__ import annotations

import dataclasses
import itertools
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import trimesh
import viser
import yourdfpy
from trimesh.scene import Scene

from viser import transforms as tf


@dataclasses.dataclass
class Sphere:
    """Represents a collision sphere attached to a URDF link."""

    id: int
    link: str
    local_xyz: Tuple[float, float, float]
    radius: float
    color: Tuple[int, int, int] = (255, 180, 60)
    node: Optional[viser.SceneNodeHandle] = dataclasses.field(default=None, repr=False)


class SphereStore:
    """Manages collection of spheres and their relationships to URDF links."""

    def __init__(self):
        self._next_id = itertools.count(0)
        self.by_id: Dict[int, Sphere] = {}
        self.ids_by_link: Dict[str, List[int]] = {}
        self.group_nodes: Dict[str, viser.FrameHandle] = {}  # /spheres/<link> parents

    def add(self, link: str, xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0), radius: float = 0.05) -> Sphere:
        """Add a new sphere to the specified link."""
        s = Sphere(id=next(self._next_id), link=link, local_xyz=xyz, radius=radius)
        self.by_id[s.id] = s
        self.ids_by_link.setdefault(link, []).append(s.id)
        return s

    def remove(self, sphere_id: int) -> Optional[Sphere]:
        """Remove a sphere by ID."""
        if sphere_id not in self.by_id:
            return None

        sphere = self.by_id.pop(sphere_id)
        self.ids_by_link[sphere.link].remove(sphere_id)

        # Clean up empty link lists
        if not self.ids_by_link[sphere.link]:
            del self.ids_by_link[sphere.link]

        # Remove from scene
        if sphere.node is not None:
            sphere.node.remove()

        return sphere

    def get_spheres_for_link(self, link: str) -> List[Sphere]:
        """Get all spheres attached to a specific link."""
        return [self.by_id[sid] for sid in self.ids_by_link.get(link, [])]

    def clear(self):
        """Remove all spheres."""
        for sphere in list(self.by_id.values()):
            self.remove(sphere.id)


class EnhancedViserUrdf:
    """Enhanced URDF visualizer with per-link control capabilities.

    Extends the basic ViserUrdf functionality to provide:
    - Individual link visibility control
    - Direct access to link frames
    - Mesh node handles for fine-grained control
    """

    def __init__(
        self,
        target: viser.ViserServer | viser.ClientHandle,
        urdf_or_path: yourdfpy.URDF | Path,
        scale: float = 1.0,
        root_node_name: str = "/",
        mesh_color_override: Tuple[float, float, float] | Tuple[float, float, float, float] | None = None,
        collision_mesh_color_override: Tuple[float, float, float] | Tuple[float, float, float, float] | None = None,
        load_meshes: bool = True,
        load_collision_meshes: bool = False,
    ) -> None:
        """Initialize enhanced URDF visualizer."""
        assert root_node_name.startswith("/")
        assert len(root_node_name) == 1 or not root_node_name.endswith("/")

        if isinstance(urdf_or_path, Path):
            urdf = yourdfpy.URDF.load(
                urdf_or_path,
                build_scene_graph=load_meshes,
                build_collision_scene_graph=load_collision_meshes,
                load_meshes=load_meshes,
                load_collision_meshes=load_collision_meshes,
                filename_handler=partial(
                    yourdfpy.filename_handler_magic,
                    dir=urdf_or_path.parent,
                ),
            )
        else:
            urdf = urdf_or_path
        assert isinstance(urdf, yourdfpy.URDF)

        self._target = target
        self._urdf = urdf
        self._scale = scale
        self._root_node_name = root_node_name
        self._load_meshes = load_meshes
        self._collision_root_frame: viser.FrameHandle | None = None
        self._visual_root_frame: viser.FrameHandle | None = None
        self._joint_frames: List[viser.SceneNodeHandle] = []
        self._meshes: List[viser.SceneNodeHandle] = []

        # Enhanced functionality: per-link control
        self.link_frame: Dict[str, viser.FrameHandle] = {}
        self.link_meshes: Dict[str, List[viser.SceneNodeHandle]] = {}

        # Cache of transformed visual meshes for live opacity rebuilds.
        # Maps viser node name -> (transformed trimesh, urdf link name).
        self._visual_mesh_cache: Dict[str, Tuple[trimesh.Trimesh, str]] = {}
        # Cached wireframe edge segments per mesh (viser node name -> (E,2,3)
        # array of line endpoints and base color), computed once at load.
        self._visual_edge_cache: Dict[str, Tuple[np.ndarray, Tuple[int, int, int]]] = {}
        self._visual_opacity: float = 1.0

        # Per-link collision geometry expressed in the link's own frame, used to
        # test whether a collision sphere protrudes past the robot surface.
        # Maps urdf link name -> trimesh.Trimesh (concatenated link geometry).
        self._link_local_meshes: Dict[str, trimesh.Trimesh] = {}

        num_joints_to_repeat = 0
        if load_meshes:
            if urdf.scene is not None:
                num_joints_to_repeat += 1
                self._visual_root_frame = self._add_joint_frames_and_meshes(
                    urdf.scene,
                    root_node_name,
                    collision_geometry=False,
                    mesh_color_override=mesh_color_override,
                )
                self._index_scene(urdf.scene, collision=False)
            else:
                warnings.warn(
                    "load_meshes is enabled but the URDF model does not have a visual scene configured. Not displaying."
                )
        if load_collision_meshes:
            if urdf.collision_scene is not None:
                num_joints_to_repeat += 1
                self._collision_root_frame = self._add_joint_frames_and_meshes(
                    urdf.collision_scene,
                    root_node_name,
                    collision_geometry=True,
                    mesh_color_override=collision_mesh_color_override,
                )
                self._index_scene(urdf.collision_scene, collision=True)
            else:
                warnings.warn(
                    "load_collision_meshes is enabled but the URDF model does not have a collision scene configured. Not displaying."
                )

        self._joint_map_values = [*self._urdf.joint_map.values()] * num_joints_to_repeat

        # Build per-link meshes (in each link's local frame) for protrusion tests.
        self._build_link_local_meshes()

    def _build_link_local_meshes(self) -> None:
        """Build a per-link mesh in each link's local frame.

        These meshes are used to test whether a collision sphere protrudes past
        the robot surface. The collision scene is preferred when available (it is
        what the spheres approximate); otherwise the visual scene is used.
        """
        scene = self._urdf.collision_scene or self._urdf.scene
        if scene is None:
            return

        collision_geometry = self._urdf.collision_scene is not None
        link_pieces: Dict[str, List[trimesh.Trimesh]] = {}

        for geom_name, mesh in scene.geometry.items():
            if not isinstance(mesh, trimesh.Trimesh):
                continue
            link_name = scene.graph.transforms.parents.get(geom_name)
            if link_name is None:
                continue
            try:
                # Transform from the link frame to this geometry's frame.
                T_link_geom = self._urdf.get_transform(
                    geom_name, link_name, collision_geometry=collision_geometry
                )
            except Exception:
                continue

            piece = mesh.copy()
            piece.apply_scale(self._scale)
            piece.apply_transform(T_link_geom)
            link_pieces.setdefault(link_name, []).append(piece)

        for link_name, pieces in link_pieces.items():
            try:
                combined = trimesh.util.concatenate(pieces) if len(pieces) > 1 else pieces[0]
            except Exception:
                combined = pieces[0]
            self._link_local_meshes[link_name] = combined

    def sphere_protrudes(
        self,
        link_name: str,
        local_xyz: Tuple[float, float, float],
        radius: float,
    ) -> bool:
        """Return True if a sphere pokes outside the link's mesh surface.

        The sphere is defined in the link's local frame (the same frame Bubblify
        stores sphere centers in). A sphere is fully contained when the signed
        distance from its center to the mesh surface is at least its radius
        (trimesh signed distance is positive inside, negative outside). If the
        link has no mesh, the sphere cannot be judged and is treated as not
        protruding.
        """
        mesh = self._link_local_meshes.get(link_name)
        if mesh is None:
            return False

        try:
            center = np.asarray([local_xyz], dtype=float)
            signed = trimesh.proximity.signed_distance(mesh, center)[0]
        except Exception:
            return False

        # Contained iff signed >= radius; otherwise part of the surface is outside.
        return bool(signed < radius)

    def _index_scene(self, scene: Scene, collision: bool) -> None:
        """Index link frames and meshes for per-link control."""
        # Add the base frame explicitly (it's not a joint child, so gets missed otherwise)
        if not collision and self._visual_root_frame is not None:
            # The base link is the scene's base frame
            base_link = scene.graph.base_frame
            self.link_frame[base_link] = self._visual_root_frame
        elif collision and self._collision_root_frame is not None:
            base_link = scene.graph.base_frame
            self.link_frame[base_link] = self._collision_root_frame

        # Index joint frames (link frames) by matching joint child names
        joint_offset = len(self._joint_frames) - len(self._urdf.joint_map)
        for i, joint in enumerate(self._urdf.joint_map.values()):
            child = joint.child
            frame_index = joint_offset + i
            if frame_index < len(self._joint_frames):
                frame_handle = self._joint_frames[frame_index]
                if isinstance(frame_handle, viser.FrameHandle):
                    self.link_frame[child] = frame_handle

    @property
    def show_visual(self) -> bool:
        """Returns whether the visual meshes are currently visible."""
        return self._visual_root_frame is not None and self._visual_root_frame.visible

    @show_visual.setter
    def show_visual(self, visible: bool) -> None:
        """Set whether the visual meshes are currently visible."""
        if self._visual_root_frame is not None:
            self._visual_root_frame.visible = visible
        else:
            warnings.warn("Cannot set `.show_visual`, since no visual meshes were loaded.")

    @property
    def show_collision(self) -> bool:
        """Returns whether the collision meshes are currently visible."""
        return self._collision_root_frame is not None and self._collision_root_frame.visible

    @show_collision.setter
    def show_collision(self, visible: bool) -> None:
        """Set whether the collision meshes are currently visible."""
        if self._collision_root_frame is not None:
            self._collision_root_frame.visible = visible
        else:
            warnings.warn("Cannot set `.show_collision`, since no collision meshes were loaded.")

    def set_link_visible(self, link_name: str, visible: bool, which: str = "visual"):
        """Set visibility of a specific link's meshes."""
        if which in ("visual", "both") and self._load_meshes:
            for mesh_handle in self.link_meshes.get(link_name, []):
                mesh_handle.visible = visible
        if which in ("collision", "both") and self._collision_root_frame is not None:
            # Handle collision meshes if needed
            pass

    def remove(self) -> None:
        """Remove URDF from scene."""
        for frame in self._joint_frames:
            frame.remove()
        for mesh in self._meshes:
            mesh.remove()

    def update_cfg(self, configuration: np.ndarray) -> None:
        """Update the joint angles of the visualized URDF."""
        self._urdf.update_cfg(configuration)
        for joint, frame_handle in zip(self._joint_map_values, self._joint_frames):
            assert isinstance(joint, yourdfpy.Joint)
            T_parent_child = self._urdf.get_transform(joint.child, joint.parent, collision_geometry=not self._load_meshes)
            frame_handle.wxyz = tf.SO3.from_matrix(T_parent_child[:3, :3]).wxyz
            frame_handle.position = T_parent_child[:3, 3] * self._scale

    def get_actuated_joint_limits(self) -> dict[str, tuple[float | None, float | None]]:
        """Returns an ordered mapping from actuated joint names to position limits."""
        out: dict[str, tuple[float | None, float | None]] = {}
        for joint_name, joint in zip(self._urdf.actuated_joint_names, self._urdf.actuated_joints):
            assert isinstance(joint_name, str)
            assert isinstance(joint, yourdfpy.Joint)
            if joint.limit is None:
                out[joint_name] = (-np.pi, np.pi)
            else:
                out[joint_name] = (joint.limit.lower, joint.limit.upper)
        return out

    def get_actuated_joint_names(self) -> Tuple[str, ...]:
        """Returns a tuple of actuated joint names, in order."""
        return tuple(self._urdf.actuated_joint_names)

    def get_all_link_names(self) -> List[str]:
        """Get all link names in the URDF."""
        return list(self.link_frame.keys())

    def _add_joint_frames_and_meshes(
        self,
        scene: Scene,
        root_node_name: str,
        collision_geometry: bool,
        mesh_color_override: Tuple[float, float, float] | Tuple[float, float, float, float] | None,
    ) -> viser.FrameHandle:
        """Helper function to add joint frames and meshes to the ViserUrdf object."""
        prefix = "collision" if collision_geometry else "visual"
        prefixed_root_node_name = (f"{root_node_name}/{prefix}").replace("//", "/")
        root_frame = self._target.scene.add_frame(prefixed_root_node_name, show_axes=False)

        # Add coordinate frame for each joint.
        for joint in self._urdf.joint_map.values():
            assert isinstance(joint, yourdfpy.Joint)
            self._joint_frames.append(
                self._target.scene.add_frame(
                    _viser_name_from_frame(
                        scene,
                        joint.child,
                        prefixed_root_node_name,
                    ),
                    show_axes=False,
                )
            )

        # Add the URDF's meshes/geometry to viser.
        for mesh_name, mesh in scene.geometry.items():
            assert isinstance(mesh, trimesh.Trimesh)
            T_parent_child = self._urdf.get_transform(
                mesh_name,
                scene.graph.transforms.parents[mesh_name],
                collision_geometry=collision_geometry,
            )
            name = _viser_name_from_frame(scene, mesh_name, prefixed_root_node_name)

            # Scale + transform the mesh. (these will mutate it!)
            mesh = mesh.copy()
            mesh.apply_scale(self._scale)
            mesh.apply_transform(T_parent_child)

            # Create the mesh handle and store it with the corresponding link
            mesh_handle = None
            if mesh_color_override is None:
                mesh_handle = self._target.scene.add_mesh_trimesh(name, mesh)
            elif len(mesh_color_override) == 3:
                mesh_handle = self._target.scene.add_mesh_simple(
                    name,
                    mesh.vertices,
                    mesh.faces,
                    color=mesh_color_override,
                )
            elif len(mesh_color_override) == 4:
                mesh_handle = self._target.scene.add_mesh_simple(
                    name,
                    mesh.vertices,
                    mesh.faces,
                    color=mesh_color_override[:3],
                    opacity=mesh_color_override[3],
                )
            else:
                raise ValueError("Invalid mesh_color_override format")

            # Store mesh handle and map it to the correct URDF link
            if mesh_handle is not None:
                self._meshes.append(mesh_handle)
                # Get the actual URDF link name that this mesh belongs to
                urdf_link_name = scene.graph.transforms.parents[mesh_name]
                self.link_meshes.setdefault(urdf_link_name, []).append(mesh_handle)
                # Cache visual meshes so we can rebuild them with a new opacity,
                # plus their edge segments for the faded wireframe rendering.
                if not collision_geometry:
                    self._visual_mesh_cache[name] = (mesh, urdf_link_name)
                    try:
                        edges = mesh.edges_unique
                        segments = np.asarray(mesh.vertices[edges], dtype=np.float32)
                        try:
                            base_color = tuple(int(c) for c in mesh.visual.main_color[:3])
                        except Exception:
                            base_color = (120, 120, 120)
                        self._visual_edge_cache[name] = (segments, base_color)
                    except Exception:
                        pass
        return root_frame

    def set_visual_opacity(self, opacity: float) -> None:
        """Set the opacity of all visual robot meshes.

        Viser mesh handles do not expose a mutable opacity (or any depth-write
        control), so this rebuilds the visual meshes in place by re-adding nodes
        with their existing names (viser overwrites a scene node when a node with
        the same name is added).

        Because a transparent *solid* surface still writes depth and would
        occlude collision spheres sitting inside a link, fading does not use a
        translucent solid surface. Instead:

        - ``opacity >= 1.0``: the original textured meshes are restored via
          ``add_mesh_trimesh``.
        - ``0 < opacity < 1.0``: the mesh is replaced by an opaque line-segment
          wireframe (``add_line_segments``) built from its edges. Real line
          segments are fully opaque and have no faces, so — unlike a translucent
          surface or a ``wireframe=True`` mesh — they never enter the renderer's
          transparent pass and do not smear or accumulate blur as the camera
          moves. The fade is simulated by blending the line color toward a light
          background as opacity drops.
        - ``opacity == 0``: the mesh is hidden entirely.
        """
        if not self._load_meshes or not self._visual_mesh_cache:
            return

        opacity = float(min(max(opacity, 0.0), 1.0))
        self._visual_opacity = opacity

        for name, (mesh, urdf_link_name) in self._visual_mesh_cache.items():
            # Preserve the current visibility of the link's existing handles.
            existing = self.link_meshes.get(urdf_link_name, [])
            visible = existing[0].visible if existing else True

            if opacity >= 1.0:
                new_handle = self._target.scene.add_mesh_trimesh(name, mesh, visible=visible)
            else:
                segments, base_color = self._visual_edge_cache.get(
                    name, (None, (150, 150, 150))
                )
                # Fall back to a solid mesh only if edges are unavailable.
                if segments is None:
                    new_handle = self._target.scene.add_mesh_simple(
                        name,
                        mesh.vertices,
                        mesh.faces,
                        color=base_color,
                        opacity=1.0,
                        wireframe=True,
                        visible=visible and opacity > 0.0,
                    )
                else:
                    # Blend the line color toward a light background as opacity
                    # drops (1.0 -> base color, 0.0 -> background). Lines stay
                    # fully opaque, so no transparent-pass accumulation.
                    bg = 245
                    blend = max(opacity, 0.15)
                    color = tuple(
                        int(round(c * blend + bg * (1.0 - blend))) for c in base_color
                    )
                    new_handle = self._target.scene.add_line_segments(
                        name,
                        points=segments,
                        colors=color,
                        line_width=1.0,
                        visible=visible and opacity > 0.0,
                    )

            # Replace the stored handle for this link/mesh.
            self.link_meshes.setdefault(urdf_link_name, [])
            for i, handle in enumerate(self._meshes):
                if handle.name == name:
                    self._meshes[i] = new_handle
                    break
            link_handles = self.link_meshes[urdf_link_name]
            for i, handle in enumerate(link_handles):
                if handle.name == name:
                    link_handles[i] = new_handle
                    break


def _viser_name_from_frame(
    scene: Scene,
    frame_name: str,
    root_node_name: str = "/",
) -> str:
    """Given the name of a frame in our URDF's kinematic tree, return a scene node name for viser."""
    assert root_node_name.startswith("/")
    assert len(root_node_name) == 1 or not root_node_name.endswith("/")

    frames = []
    while frame_name != scene.graph.base_frame:
        frames.append(frame_name)
        frame_name = scene.graph.transforms.parents[frame_name]
    if root_node_name != "/":
        frames.append(root_node_name)
    return "/".join(frames[::-1])


def inject_spheres_into_urdf_xml(original_urdf_path: Optional[Path], urdf_obj: yourdfpy.URDF, store: SphereStore) -> str:
    """Inject collision spheres into URDF XML, replacing all existing collision elements."""
    if original_urdf_path is not None:
        root = ET.parse(original_urdf_path).getroot()
    else:
        # Reconstruct from urdf_obj using correct method
        root = ET.fromstring(urdf_obj.write_xml_string())

    # Map link name to element
    link_elems = {e.get("name"): e for e in root.findall("link")}

    # Remove ALL existing collision elements from ALL links (not just sphere links)
    for link_elem in link_elems.values():
        # Find and remove all collision elements
        collision_elems = link_elem.findall("collision")
        for collision_elem in collision_elems:
            link_elem.remove(collision_elem)

    # Add sphere collision elements
    for link_name, sphere_ids in store.ids_by_link.items():
        link_elem = link_elems.get(link_name)
        if link_elem is None:
            continue

        for sphere_id in sphere_ids:
            sphere = store.by_id[sphere_id]
            coll = ET.SubElement(link_elem, "collision", {"name": f"sphere_{sphere.id}"})
            origin = ET.SubElement(
                coll, "origin", {"xyz": f"{sphere.local_xyz[0]} {sphere.local_xyz[1]} {sphere.local_xyz[2]}", "rpy": "0 0 0"}
            )
            geom = ET.SubElement(coll, "geometry")
            sph = ET.SubElement(geom, "sphere", {"radius": f"{sphere.radius}"})

    # Pretty format the XML with proper indentation (Python 3.8 compatible)
    def indent_xml(elem, level=0, indent="  "):
        """Indent XML for pretty printing (Python 3.8 compatible)."""
        i = "\n" + level * indent
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + indent
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                indent_xml(child, level + 1, indent)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    indent_xml(root)

    # Add XML declaration and return
    xml_content = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + xml_content
