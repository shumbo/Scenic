""" Module containing the Shape class and its subclasses, which represent unoriented shapes centered at the origin"""

from abc import ABC, abstractmethod

import trimesh
from trimesh.transformations import translation_matrix, quaternion_matrix, concatenate_matrices

from scenic.core.distributions import distributionFunction, distributionMethod, Samplable, needsSampling

class Shape(ABC):
    """An abstract base class for Scenic objects.

    Scenic objects have a shape property associated with them. This
    abstract class implements the procedure to perform mesh processing
    as well as several common methods supported by meshes that an object
    will use.
    """
    def __init__(self):
        pass

    @abstractmethod
    def getBoundingBoxExtents(self):
        pass

class DefaultShape(Shape):
    def __init__(self, parent):
        self.parent = parent

    def getBoundingBoxExtents(self):
        return (1,1,1)

    def resolve(self):
        self.parent.shape = BoxShape(self.parent.width, self.parent.length, self.parent.height)

class MeshShape(Shape):
    """ A Shape subclass defined by a Trimesh object.

    :param mesh: A trimesh.Trimesh mesh object.
    """
    @distributionMethod
    def __init__(self, mesh):
        self.mesh = mesh.copy()

        # Center vertices around origin
        self.mesh.vertices -= self.mesh.bounding_box.center_mass

    def getBoundingBoxExtents(self):
        _ , extents = trimesh.bounds.oriented_bounds(self.mesh)

        return extents

    # def containsPoint(self, position, orientation, point):
    #     # Apply translation and rotation to properly apply mesh position
    #     # and rotation.
    #     oriented_mesh = self.mesh.copy()

    #     position_matrix = translation_matrix(position)
    #     rotation_matrix = quaternion_matrix(orientation.q)

    #     transform_matrix = concatenate_matrices(position_matrix, rotation_matrix)

    #     oriented_mesh.apply_transform(transform_matrix)

    #     return oriented_mesh.contains([point])

    # def intersects(self, position, orientation, shape):
    #     assert isinstance(shape, MeshShape) # TODO: Generalize to non mesh objects

    #     collision_manager = trimesh.collision.CollisionManager()

    #     collision_manager.add_object("ShapeA", self.get_oriented_mesh())
    #     collision_manager.add_object("ShapeB", shape.get_oriented_mesh())

    #     return not intersection_mesh.is_empty

    # ## TEMPORARY: Should be replaced by regions in the future.
    # def get_oriented_mesh(self, position, orientation):
    #     oriented_mesh = self.mesh.copy()

    #     position_matrix = translation_matrix(position)
    #     rotation_matrix = quaternion_matrix(orientation.q)

    #     transform_matrix = concatenate_matrices(position_matrix, rotation_matrix)

    #     oriented_mesh.apply_transform(transform_matrix)

    #     return oriented_mesh

class BoxShape(MeshShape, Samplable):
    def __init__(self, width, length, height):
        Samplable.__init__(self, [width, length, height])

        self.width = width
        self.length = length
        self.height = height

        self.mesh = None

    def sampleGiven(self, value):
        width = value[self.width]
        length = value[self.length]
        height = value[self.height]

        # TODO: Lazy transformation into a mesh if needed.
        return MeshShape(trimesh.creation.box((width, length, height)))
