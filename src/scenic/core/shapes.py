""" Module containing the Shape class and its subclasses, which represent unoriented shapes centered at the origin"""

from abc import ABC, abstractmethod

import trimesh
from trimesh.transformations import translation_matrix, quaternion_matrix, concatenate_matrices
import numpy

from scenic.core.distributions import (distributionFunction, distributionMethod, Samplable,
                                       needsSampling, toDistribution)


###################################################################################################
# Abstract Classes and Utilities
###################################################################################################

class Shape(Samplable, ABC):
    """ An abstract base class for Scenic shapes.

    Represents physical shape in Scenic. Does not represent position or orientation,
    which is all handled by the region class. Does contain dimension information, which
    is used as a default value by any Object with this shape and can be overwritten.

    :param dimensions: The raw (before scaling) dimensions of the shape. If dimensions
        and scale are both specified the dimensions are first set by dimensions, and then
        scaled by scale.
    :param scale: Scales all the dimensions of the shape by a multiplicative factor.
        If dimensions and scale are both specified the dimensions are first set by dimensions,
        and then scaled by scale.
    """
    def __init__(self, dimensions, scale):
        # Report dimensions and scale as samplable
        dimensions = toDistribution(dimensions)
        super().__init__([dimensions, scale])

        # Store values
        self.raw_dimensions = dimensions
        self.scale = scale

    @property
    def dimensions(self):
        return [dim * self.scale for dim in self.raw_dimensions]

###################################################################################################
# 3D Shape Classes
###################################################################################################

class MeshShape(Shape):
    """ A Shape subclass defined by a Trimesh object.

    :param mesh: A trimesh.Trimesh mesh object.
    :param dimensions: The raw (before scaling) dimensions of the shape. If dimensions
        and scale are both specified the dimensions are first set by dimensions, and then
        scaled by scale.
    :param scale: Scales all the dimensions of the shape by a multiplicative factor.
        If dimensions and scale are both specified the dimensions are first set by dimensions,
        and then scaled by scale.
    """
    def __init__(self, mesh, dimensions=None, scale=1):
        # Ensure the mesh is watertight so volume is well defined
        if not mesh.is_watertight:
            raise ValueError("A MeshShape cannot be defined with a mesh that does not have a well defined volume.")

        # Copy mesh and center vertices around origin
        self.mesh = mesh.copy()

        # If dimensions are not specified, infer them.
        if dimensions is None:
            dimensions = self.mesh.extents

        # Report samplables
        super().__init__(dimensions, scale)

    def sampleGiven(self, values):
        return MeshShape(self.mesh, values[self.raw_dimensions], values[self.scale])

class BoxShape(Shape):
    def __init__(self, dimensions=(1,1,1), scale=1):
        # Report samplables
        super().__init__(dimensions, scale)
    
    def sampleGiven(self, values):
        return BoxShape(values[self.raw_dimensions], values[self.scale])

    @property
    def mesh(self):
        return trimesh.creation.box((1,1,1))
