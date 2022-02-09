import math
from pathlib import Path

import gym_miniworld.entity
import gym_miniworld.objmesh
from gym_miniworld.utils import get_file_path


class ObjMesh(gym_miniworld.objmesh.ObjMesh):
    @classmethod
    def get(cls, mesh_name: str, **kwargs):
        """
        Load a mesh or used a cached version
        """
        path = Path(mesh_name)
        if path.is_absolute():
            subdir = path.parent
            mesh_name = path.stem
        else:
            subdir = "meshes"

        # Assemble the absolute path to the mesh file
        file_path = get_file_path(subdir, mesh_name, "obj")

        if file_path in cls.cache:
            return cls.cache[file_path]
        print(f"Loading {file_path}...")

        mesh = ObjMesh(file_path, **kwargs)
        print(f"Loaded {file_path}.")
        cls.cache[file_path] = mesh

        return mesh


class MeshEnt(gym_miniworld.entity.MeshEnt):
    """
    Entity whose appearance is defined by a mesh file

    height -- scale the model to this height
    static -- flag indicating this object cannot move
    """

    def __init__(
        self,
        mesh_name,
        height,
        static=True,
        tex_name=None,
    ):
        gym_miniworld.entity.Entity.__init__(self)

        self.static = static

        # Load the mesh
        self.mesh = ObjMesh.get(mesh_name, tex_path=tex_name)

        # Get the mesh extents
        sx, sy, sz = self.mesh.max_coords

        # Compute the mesh scaling factor
        self.scale = height / sy

        # Compute the radius and height
        self.radius = math.sqrt(sx * sx + sz * sz) * self.scale
        self.height = height
