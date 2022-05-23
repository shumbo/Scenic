""" Module containing the Shape class and its subclasses, which represent unoriented shapes centered at the origin"""

from abc import ABC, abstractmethod

import trimesh
from trimesh.transformations import translation_matrix, quaternion_matrix, concatenate_matrices

from scenic.core.distributions import distributionFunction, distributionMethod, Samplable, needsSampling


###################################################################################################
# Abstract Classes and Utilities
###################################################################################################

class Shape(Samplable):
    """ An abstract base class for Scenic shapes.

    Represents physical shape and dimensions in Scenc.
    """
    def __init__(self, *dependencies):
        super().__init__(dependencies)

    def getBoundingBoxExtents(self):
        raise NotImplementedError()

###################################################################################################
# 3D Shape Classes
###################################################################################################

class MeshShape(Shape):
    """ A Shape subclass defined by a Trimesh object.

    :param mesh: A trimesh.Trimesh mesh object.
    """
    @distributionMethod
    def __init__(self, mesh, *dependencies):
        super().__init__(dependencies)

        # Ensure the mesh is watertight so volume is well defined
        if not mesh.is_watertight:
            raise ValueError("A MeshShape cannot be defined with a mesh that is not watertight.")

        self.mesh = mesh.copy()

        # Center vertices around origin
        self.mesh.vertices -= self.mesh.bounding_box.center_mass

    def getBoundingBoxExtents(self):
        _ , extents = trimesh.bounds.oriented_bounds(self.mesh)

        return extents

class DefaultShape(Shape):
    def __init__(self, parent, *dependencies):
        super().__init__(dependencies)

        self.parent = parent

    def getBoundingBoxExtents(self):
        return (1,1,1)

    def resolve(self):
        if needsSampling(self.parent.width) or needsSampling(self.parent.length) or needsSampling(self.parent.height):
            self.parent.shape = BoxShape(self.parent.width, self.parent.length, self.parent.height)
        else:
            self.parent.shape = MeshShape(trimesh.creation.box((self.parent.width, self.parent.length, self.parent.height)))

class BoxShape(Shape):
    def __init__(self, width, length, height):
        super().__init__([width, length, height])

        self.width = width
        self.length = length
        self.height = height

    def sampleGiven(self, value):
        width = value[self.width]
        length = value[self.length]
        height = value[self.height]

        # TODO: Lazy transformation into a mesh only if needed.
        return BoxShape(width, length, height)

    @property
    def mesh(self):
        return MeshShape(trimesh.creation.box((self.width, self.length, self.height)))
    