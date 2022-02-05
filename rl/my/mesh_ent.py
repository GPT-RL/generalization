import math

from gym_miniworld.entity import Entity
from gym_miniworld.objmesh import ObjMesh
from pyglet.gl import (
    glColor3f,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glScalef,
    glTranslatef,
)


class MeshEnt(Entity):
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
        super().__init__()

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

    def render(self):
        """
        Draw the object
        """

        glPushMatrix()
        glTranslatef(*self.pos)
        glScalef(self.scale, self.scale, self.scale)
        glRotatef(self.dir * 180 / math.pi, 0, 1, 0)
        glColor3f(1, 1, 1)
        self.mesh.render()
        glPopMatrix()

    @property
    def is_static(self):
        return self.static
