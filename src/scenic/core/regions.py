"""Objects representing regions in space.

Manipulations of polygons and line segments are done using the
`shapely <https://github.com/shapely/shapely>`_ package.
"""

import math
import random
import itertools
from abc import ABC

import numpy
import scipy
import shapely
import shapely.geometry
import shapely.ops
import shapely.prepared

import trimesh
from trimesh.transformations import translation_matrix, quaternion_matrix, concatenate_matrices

from subprocess import CalledProcessError

from scenic.core.distributions import (Samplable, RejectionException, needsSampling,
									   distributionMethod, toDistribution)
from scenic.core.lazy_eval import valueInContext
from scenic.core.vectors import Vector, OrientedVector, VectorDistribution, VectorField, Orientation
from scenic.core.geometry import _RotatedRectangle
from scenic.core.geometry import sin, cos, hypot, findMinMax, pointIsInCone, averageVectors
from scenic.core.geometry import headingOfSegment, triangulatePolygon, plotPolygon, polygonUnion
from scenic.core.type_support import toVector, toScalar
from scenic.core.utils import cached, cached_property

###################################################################################################
# Abstract Classes and Utilities
###################################################################################################

class Region(Samplable, ABC):
	""" An abstract base class for Scenic Regions.
	
	Extends Scenic shapes with translation and rotation.
	"""
	def __init__(self, name, *dependencies, orientation=None):
		super().__init__(dependencies)
		self.name = name
		self.orientation = orientation

	## Abstract Methods ##

	def intersects(self, other, triedReversed=False) -> bool:
		"""Check if this `Region` intersects another."""
		if triedReversed:
			raise NotImplementedError
		else:
			return other.intersects(self, triedReversed=True)

	def uniformPointInner(self):
		"""Do the actual random sampling. Implemented by subclasses."""
		raise NotImplementedError

	def containsPoint(self, point) -> bool:
		"""Check if the `Region` contains a point. Implemented by subclasses."""
		raise NotImplementedError

	# TODO: Move this to subclass for 2D regions.
	def containsObject(self, obj) -> bool:
		"""Check if the `Region` contains an :obj:`~scenic.core.object_types.Object`.
		The default implementation assumes the `Region` is convex; subclasses must
		override the method if this is not the case.
		"""
		for corner in obj.corners:
			if not self.containsPoint(corner):
				return False
		return True

	def distanceTo(self, point) -> float:
		"""Distance to this region from a given point.

		Not supported by all region types.
        """
		raise NotImplementedError

	## Overridable Methods ##
	# The following methods can be overriden to get better performance or if the region
	# has dependencies (in the case of sampleGiven).

	def intersect(self, other, triedReversed=False) -> 'Region':
		"""intersect(other)

		Get a `Region` representing the intersection of this one with another.
		"""
		if triedReversed:
			orientation = self.orientation
			if orientation is None:
				orientation = other.orientation
			elif other.orientation is not None:
				orientation = None 		# would be ambiguous, so pick no orientation
			return IntersectionRegion(self, other, orientation=orientation)
		else:
			return other.intersect(self, triedReversed=True)

	def union(self, other, triedReversed=False) -> 'Region':
		"""Get a `Region` representing the union of this one with another.
		Not supported by all region types.
		"""
		if triedReversed:
			raise NotImplementedError
		else:
			return other.union(self, triedReversed=True)

	def difference(self, other) -> 'Region':
		"""Get a `Region` representing the difference of this one and another."""
		if isinstance(other, EmptyRegion):
			return self
		elif isinstance(other, AllRegion):
			return nowhere
		return DifferenceRegion(self, other)

	def getAABB(self):
		"""Axis-aligned bounding box for this `Region`. Implemented by some subclasses."""
		raise NotImplementedError

	def sampleGiven(self, value):
		return self

	## API Methods ##

	@staticmethod
	def uniformPointIn(region):
		"""Get a uniform `Distribution` over points in a `Region`."""
		return PointInRegionDistribution(region)

	def uniformPoint(self):
		"""Sample a uniformly-random point in this `Region`.
		Can only be called on fixed Regions with no random parameters.
		"""
		assert not needsSampling(self)
		return self.uniformPointInner()

	def __contains__(self, thing) -> bool:
		"""Check if this `Region` contains an object or vector."""
		from scenic.core.object_types import Object
		if isinstance(thing, Object):
			return self.containsObject(thing)
		vec = toVector(thing, '"X in Y" with X not an Object or a vector')
		return self.containsPoint(vec)

	def orient(self, vec):
		"""Orient the given vector along the region's orientation, if any."""
		if self.orientation is None:
			return vec
		else:
			return OrientedVector(vec.x, vec.y, vec.z, self.orientation[vec])

	def __str__(self):
		s = f'<{type(self).__name__}'
		if self.name:
			s += f' {self.name}'
		return s + '>'

	def __repr__(self):
		s = f'<{type(self).__name__}'
		if self.name:
			s += f' {self.name}'
		return s + f' at {hex(id(self))}>'

class PointInRegionDistribution(VectorDistribution):
	"""Uniform distribution over points in a Region"""
	def __init__(self, region):
		super().__init__(region)
		self.region = region

	def sampleGiven(self, value):
		return value[self.region].uniformPointInner()

	@property
	def heading(self):
		if self.region.orientation is not None:
			return self.region.orientation[self]
		else:
			return 0

	def __repr__(self):
		return f'PointIn({self.region})'

###################################################################################################
# Utility Regions and Functions
###################################################################################################

class AllRegion(Region):
	"""Region consisting of all space."""
	def intersect(self, other, triedReversed=False):
		return other

	def intersects(self, other, triedReversed=False):
		return not isinstance(other, EmptyRegion)

	def union(self, other, triedReversed=False):
		return self

	def containsPoint(self, point):
		return True

	def containsObject(self, obj):
		return True

	def distanceTo(self, point):
		return 0

	def __eq__(self, other):
		return type(other) is AllRegion

	def __hash__(self):
		return hash(AllRegion)

class EmptyRegion(Region):
	"""Region containing no points."""
	def intersect(self, other, triedReversed=False):
		return self

	def intersects(self, other, triedReversed=False):
		return False

	def difference(self, other):
		return self

	def union(self, other, triedReversed=False):
		return other

	def uniformPointInner(self):
		raise RejectionException(f'sampling empty Region')

	def containsPoint(self, point):
		return False

	def containsObject(self, obj):
		return False

	def distanceTo(self, point):
		return float('inf')

	def show(self, plt, style=None, **kwargs):
		pass

	def __eq__(self, other):
		return type(other) is EmptyRegion

	def __hash__(self):
		return hash(EmptyRegion)

#: A `Region` containing all points.
#:
#: Points may not be sampled from this region, as no uniform distribution over it exists.
everywhere = AllRegion('everywhere')

#: A `Region` containing no points.
#:
#: Attempting to sample from this region causes the sample to be rejected.
nowhere = EmptyRegion('nowhere')

class IntersectionRegion(Region):
	def __init__(self, *regions, orientation=None, sampler=None, name=None):
		self.regions = tuple(regions)
		if len(self.regions) < 2:
			raise RuntimeError('tried to take intersection of fewer than 2 regions')
		super().__init__(name, *self.regions, orientation=orientation)
		if sampler is None:
			sampler = self.genericSampler
		self.sampler = sampler

	def sampleGiven(self, value):
		regs = [value[reg] for reg in self.regions]
		# Now that regions have been sampled, attempt intersection again in the hopes
		# there is a specialized sampler to handle it (unless we already have one)
		if self.sampler is self.genericSampler:
			failed = False
			intersection = regs[0]
			for region in regs[1:]:
				intersection = intersection.intersect(region)
				if isinstance(intersection, IntersectionRegion):
					failed = True
					break
			if not failed:
				intersection.orientation = value[self.orientation]
				return intersection
		return IntersectionRegion(*regs, orientation=value[self.orientation],
								  sampler=self.sampler, name=self.name)

	def evaluateInner(self, context, modifying):
		regs = (valueInContext(reg, context, modifying) for reg in self.regions)
		orientation = valueInContext(self.orientation, context, modifying)
		return IntersectionRegion(*regs, orientation=orientation, sampler=self.sampler,
								  name=self.name)

	def containsPoint(self, point):
		return all(region.containsPoint(point) for region in self.regions)

	def uniformPointInner(self):
		return self.orient(self.sampler(self))

	@staticmethod
	def genericSampler(intersection):
		regs = intersection.regions

		# Get a candidate point from each region
		points = [reg.uniformPointInner() for reg in regs]

		# Filter all points that aren't contained in all regions
		for region in regs:
			points = [point for point in points if region.containsPoint(point)]

		# If no points remain, reject. Otherwise return one at random.
		if len(points) == 0:
			raise RejectionException(f'sampling intersection of Regions {regs}')

		return random.choice(points)

		# point = regs[0].uniformPointInner()
		# for region in regs[1:]:
		# 	if not region.containsPoint(point):
		# 		raise RejectionException(
		# 			f'sampling intersection of Regions {regs[0]} and {region}')
		# return point

	def isEquivalentTo(self, other):
		if type(other) is not IntersectionRegion:
			return False
		return (areEquivalent(set(other.regions), set(self.regions))
				and other.orientation == self.orientation)

	def __repr__(self):
		return f'IntersectionRegion({self.regions})'

class DifferenceRegion(Region):
	def __init__(self, regionA, regionB, sampler=None, name=None):
		self.regionA, self.regionB = regionA, regionB
		super().__init__(name, regionA, regionB, orientation=regionA.orientation)
		if sampler is None:
			sampler = self.genericSampler
		self.sampler = sampler

	def sampleGiven(self, value):
		regionA, regionB = value[self.regionA], value[self.regionB]
		# Now that regions have been sampled, attempt difference again in the hopes
		# there is a specialized sampler to handle it (unless we already have one)
		if self.sampler is self.genericSampler:
			diff = regionA.difference(regionB)
			if not isinstance(diff, DifferenceRegion):
				diff.orientation = value[self.orientation]
				return diff
		return DifferenceRegion(regionA, regionB, orientation=value[self.orientation],
								sampler=self.sampler, name=self.name)

	def evaluateInner(self, context, modifying):
		regionA = valueInContext(self.regionA, context, modifying)
		regionB = valueInContext(self.regionB, context, modifying)
		orientation = valueInContext(self.orientation, context, modifying)
		return DifferenceRegion(regionA, regionB, orientation=orientation,
								sampler=self.sampler, name=self.name)

	def containsPoint(self, point):
		return self.regionA.containsPoint(point) and not self.regionB.containsPoint(point)

	def uniformPointInner(self):
		return self.orient(self.sampler(self))

	@staticmethod
	def genericSampler(difference):
		regionA, regionB = difference.regionA, difference.regionB
		point = regionA.uniformPointInner()
		if regionB.containsPoint(point):
			raise RejectionException(
				f'sampling difference of Regions {regionA} and {regionB}')
		return point

	def isEquivalentTo(self, other):
		if type(other) is not DifferenceRegion:
			return False
		return (areEquivalent(self.regionA, other.regionA)
				and areEquivalent(self.regionB, other.regionB)
				and other.orientation == self.orientation)

	def __repr__(self):
		return f'DifferenceRegion({self.regionA}, {self.regionB})'

def toPolygon(thing):
	if needsSampling(thing):
		return None
	if hasattr(thing, 'polygon'):
		poly = thing.polygon
	elif hasattr(thing, 'polygons'):
		poly = thing.polygons
	elif hasattr(thing, 'lineString'):
		poly = thing.lineString
	else:
		return None

	if poly.has_z:	# TODO revisit once we have 3D regions
		return shapely.ops.transform(lambda x, y, z: (x, y), poly)
	else:
		return poly

def regionFromShapelyObject(obj, orientation=None):
	"""Build a 'Region' from Shapely geometry."""
	assert obj.is_valid, obj
	if obj.is_empty:
		return nowhere
	elif isinstance(obj, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
		return PolygonalRegion(polygon=obj, orientation=orientation)
	elif isinstance(obj, (shapely.geometry.LineString, shapely.geometry.MultiLineString)):
		return PolylineRegion(polyline=obj, orientation=orientation)
	else:
		raise RuntimeError(f'unhandled type of Shapely geometry: {obj}')

###################################################################################################
# 3D Regions
###################################################################################################

class _MeshRegion(Region):
	"""Region given by an oriented and positioned mesh. This region can be subclassed to define
	whether operations are performed over the volume or surface of the mesh.

	The mesh is first placed so the origin is at the center of the bounding box. The mesh is then
	translated so the center of the bounding box of the mesh is at positon, and rotated to orientation.

	:param mesh: The mesh representing the shape of this MeshRegion
	:param name: An optional name to help with debugging.
	:param position: An optional position, which determines where the center of the region will be.
	:param position: An optional Orientation object which determines the rotation of the object in space.
	:param dimensions: An optional 3-tuple, with the values representing width, length, height respectively.
		The mesh will be scaled such that the bounding box for the mesh has these dimensions.
	:param orientation: An optional vector field describing the preferred orientation at every point in
		the region.
	:param tolerance: Tolerance for collision computations.
	:param center_mesh: Whether or not to center the mesh after copying and before transformations. Only turn this off
		if you know what you're doing and don't plan to scale or translate the mesh.
	:param additional_deps: Any additional sampling dependencies this region relies on.
	"""
	def __init__(self, mesh, name=None, dimensions=None, position=None, rotation=None, orientation=None, on_direction=(0,0,1), \
	  	tolerance=1e-8, center_mesh=True, engine="blender", additional_deps=[]):
		# Copy the mesh and parameters
		self._mesh = mesh.copy()
		self.dimensions = toDistribution(dimensions)
		self.position = toDistribution(position)
		self.rotation = toDistribution(rotation)
		self.orientation = orientation
		self.tolerance = tolerance
		self.center_mesh = center_mesh
		self.engine = engine

		# Initialize superclass with samplables
		super().__init__(name, self.dimensions, self.position, self.rotation, orientation=orientation, *additional_deps)

		# If sampling is needed, delay transformations
		if needsSampling(self):
			return

		# Center mesh unless disabled
		if center_mesh:
			self.mesh.vertices -= self.mesh.bounding_box.center_mass

		# If dimensions are provided, scale mesh to those dimension
		if dimensions is not None:
			scale = self.mesh.extents / numpy.array(dimensions)

			scale_matrix = numpy.eye(4)
			scale_matrix[:3, :3] /= scale

			self.mesh.apply_transform(scale_matrix)

		# If rotation is provided, apply rotation
		if rotation is not None:
			rotation_matrix = quaternion_matrix((rotation.w, rotation.x, rotation.y, rotation.z))
			self.mesh.apply_transform(rotation_matrix)

		# If position is provided, translate mesh.
		if position is not None:
			position_matrix = translation_matrix(position)
			self.mesh.apply_transform(position_matrix)

		# Set default orientation to one inferred from face norms if none is provided.
		if orientation is None:
			self.orientation = VectorField("DefaultSurfaceVectorField", lambda pos: self.getFlatOrientation(pos))
		else:
			self.orientation = orientation

		self.on_direction = toVector(on_direction)

	## API Methods ##
	# Mesh Access #
	@property
	def mesh(self):
		# Prevent access to mesh unless it actually represents region.
		if needsSampling(self):
			raise ValueError("Attempting to access the Mesh of an unsampled MeshRegion.")

		return self._mesh

	# Composition methods #
	def intersect(self, other, triedReversed=False):
		""" Get a `Region` representing the intersection of this region's
		volume with another region. If the resulting mesh is watertight,
		the resulting region will be a MeshVolumeRegion. Otherwise returns
		a MeshSurfaceRegion.
		"""
		# If one of the regions isn't fixed fall back on default behavior
		if needsSampling(self) or needsSampling(other):
			return super().intersect(other, triedReversed)

		if isinstance(other, (_MeshRegion)):
			# Other region is a MeshRegion.
			# We can extract the mesh to perform boolean operations on it
			other_mesh = other.mesh

			# Compute intersection using Trimesh (CalledProcessError usually means empty intersection)
			try:
				new_mesh = self.mesh.intersection(other_mesh, engine=self.engine)
			except CalledProcessError:
				return EmptyRegion("EmptyMesh")

			if new_mesh.is_empty:
				return EmptyRegion("EmptyMesh")
			elif new_mesh.is_volume:
				return MeshVolumeRegion(new_mesh, tolerance=min(self.tolerance, other.tolerance), center_mesh=False, engine=self.engine)
			else:
				return MeshSurfaceRegion(new_mesh, tolerance=min(self.tolerance, other.tolerance), center_mesh=False, engine=self.engine)

		elif isinstance(other_polygon := toPolygon(other),shapely.geometry.polygon.Polygon):
			# Other region is a polygon, which we can extrude it to cover the entire mesh vertically
			# and then take the intersection.

			# Determine the mesh's vertical bounds, and extrude the polygon to have height equal to the mesh
			# (plus a little extra).
			vertical_bounds = (self.mesh.bounds[0][2], self.mesh.bounds[1][2])
			polygon_height = vertical_bounds[1] - vertical_bounds[0] + 1

			polygon_mesh = trimesh.creation.extrude_polygon(polygon=other_polygon, height=polygon_height)

			# Translate the polygon mesh vertically so it covers the main mesh.
			polygon_z_pos = (vertical_bounds[1] + vertical_bounds[0])/2
			polygon_mesh.vertices[:,2] += polygon_z_pos - polygon_mesh.bounding_box.center_mass[2]

			# Compute intersection using Trimesh (CalledProcessError usually means empty intersection)
			try:
				new_mesh = self.mesh.intersection(polygon_mesh, engine=self.engine)
			except CalledProcessError:
				return EmptyRegion("EmptyMesh")


			if new_mesh.is_empty:
				return EmptyRegion("EmptyMesh")
			elif new_mesh.is_volume:
				return MeshVolumeRegion(new_mesh, tolerance=self.tolerance, center_mesh=False, engine=self.engine)
			else:
				return MeshSurfaceRegion(new_mesh, tolerance=self.tolerance, center_mesh=False, engine=self.engine)

		# Don't know how to compute this intersection, fall back to default behavior.
		return super().intersect(other, triedReversed)

	def union(self, other, triedReversed=False):
		""" Get a `Region` representing the union of this region's
		volume with another region. If the resulting mesh is watertight,
		the resulting region will be a MeshVolumeRegion. Otherwise returns
		a MeshSurfaceRegion.
		"""
		# If one of the regions isn't fixed fall back on default behavior
		if needsSampling(self) or needsSampling(other):
			return super().union(other, triedReversed)

		# If other region is represented by a mesh, we can extract the mesh to
		# perform boolean operations on it
		if isinstance(other, (_MeshRegion)):
			other_mesh = other.mesh

			# Compute union using Trimesh
			new_mesh = self.mesh.union(other_mesh, engine=self.engine)

			if new_mesh.is_empty:
				return EmptyRegion("EmptyMesh")
			elif new_mesh.is_volume:
				return MeshVolumeRegion(new_mesh, tolerance=min(self.tolerance, other.tolerance), center_mesh=False, engine=self.engine)
			else:
				return MeshSurfaceRegion(new_mesh, tolerance=min(self.tolerance, other.tolerance), center_mesh=False, engine=self.engine)
		# TODO Look into union between mesh and polygon.
		# If mesh projection onto plane contained by polygon then the union is the polygon.

		# Don't know how to compute this union, fall back to default behavior.
		return super().union(other, triedReversed)

	def difference(self, other, debug=False):
		""" Get a `Region` representing the difference of this region's
		volume with another region. If the resulting mesh is watertight,
		the resulting region will be a MeshVolumeRegion. Otherwise returns
		a MeshSurfaceRegion.
		"""
		# If one of the regions isn't fixed fall back on default behavior
		if needsSampling(self) or needsSampling(other):
			return super().difference(other)

		# If other region is represented by a mesh, we can extract the mesh to
		# perform boolean operations on it
		if isinstance(other, (_MeshRegion)):
			other_mesh = other.mesh

			# Compute difference using Trimesh (CalledProcessError usually means empty intersection)
			try:
				new_mesh = self.mesh.difference(other_mesh, engine=self.engine, debug=debug)
			except CalledProcessError:
				return EmptyRegion("EmptyMesh")

			if new_mesh.is_empty:
				return EmptyRegion("EmptyMesh")
			elif new_mesh.is_volume:
				return MeshVolumeRegion(new_mesh, tolerance=min(self.tolerance, other.tolerance), center_mesh=False, engine=self.engine)
			else:
				return MeshSurfaceRegion(new_mesh, tolerance=min(self.tolerance, other.tolerance), center_mesh=False, engine=self.engine)
		elif toPolygon(other) is not None:
			# Other region is a polygon.
			# We can extrude it to cover the entire mesh vertically
			# and then take the difference.
			other_polygon = toPolygon(other)

			# Determine the mesh's vertical bounds, and extrude the polygon to have height equal to the mesh
			# (plus a little extra).
			vertical_bounds = (self.mesh.bounds[0][2], self.mesh.bounds[1][2])
			polygon_height = vertical_bounds[1] - vertical_bounds[0] + 1

			polygon_mesh = trimesh.creation.extrude_polygon(polygon=other_polygon, height=polygon_height)

			# Translate the polygon mesh vertically so it covers the main mesh.
			polygon_z_pos = (vertical_bounds[1] + vertical_bounds[0])/2
			polygon_mesh.vertices[:,2] += polygon_z_pos - polygon_mesh.bounding_box.center_mass[2]

			# Compute difference using Trimesh (CalledProcessError usually means empty intersection)
			try:
				new_mesh = self.mesh.difference(polygon_mesh, engine=self.engine)
			except CalledProcessError:
				return EmptyRegion("EmptyMesh")

			if new_mesh.is_empty:
				return EmptyRegion("EmptyMesh")
			elif new_mesh.is_volume:
				return MeshVolumeRegion(new_mesh, tolerance=self.tolerance, center_mesh=False, engine=self.engine)
			else:
				return MeshSurfaceRegion(new_mesh, tolerance=self.tolerance, center_mesh=False, engine=self.engine)

		# Don't know how to compute this difference, fall back to default behavior.
		return super().difference(other)

	def findOn(self, point, on_direction):
		""" Find the nearest point in the region following the on_direction.
		Returns None if no such points exist.
		"""
		# Get first point hit in both directions of ray
		point = point.coordinates

		intersection_data, _, _ = self.mesh.ray.intersects_location(ray_origins=[point, point], ray_directions=[on_direction, -1*on_direction], multiple_hits=False)

		if len(intersection_data) == 0:
			return None

		# Get point with least euclidean distance
		def euclidean_distance(p_1, p_2):
		    diff_list = [p_1[i] - p_2[i] for i in range(3)]
		    square_list = [math.pow(p, 2) for p in diff_list]
		    return math.sqrt(sum(square_list))

		distances = [euclidean_distance(point, p) for p in intersection_data]

		closest_point = intersection_data[distances.index(min(distances))]

		return Vector(*closest_point)

	def getFlatOrientation(self, pos):
		prox_query = trimesh.proximity.ProximityQuery(self.mesh)

		_, distance, triangle_id = prox_query.on_surface([pos.coordinates])

		if distance > self.tolerance:
			return (0,0,0)

		face_normal_vector = self.mesh.face_normals[triangle_id][0]

		transform = trimesh.geometry.align_vectors([0,0,1], face_normal_vector)
		orientation = tuple(scipy.spatial.transform.Rotation.from_matrix(transform[:3,:3]).as_euler('ZXY'))

		return orientation

	@cached_property
	def circumcircle(self):
		assert not needsSampling(self)

		center_point = Vector(*self.mesh.bounding_box.center_mass)
		half_extents = [val/2 for val in self.mesh.extents]
		circumradius = hypot(*half_extents)

		return (center_point, circumradius)

class MeshVolumeRegion(_MeshRegion):
	""" An instance of _MeshRegion that performs operations over the volume
	of the mesh.
	"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# Ensure the mesh is watertight so volume is well defined
		if not self._mesh.is_volume:
			raise ValueError("A MeshVolumeRegion cannot be defined with a mesh that does not have a well defined volume.")

		# Compute how many samples are necessary to achieve 99% probability
		# of success when rejection sampling volume.
		p_volume = self._mesh.volume/self._mesh.bounding_box.volume

		if p_volume > 0.99:
			self.num_samples = 1
		else:
			self.num_samples = min(1e6, max(1, math.ceil(math.log(0.01, 1 - p_volume))))

	# Property testing methods #
	def intersects(self, other, triedReversed=False):
		"""Check if this region's volume intersects another.

		This function handles intersect calculations for MeshVolumeRegion with:
			- MeshVolumeRegon
			- MeshSurfaceRegion
		"""
		# Check if region is fixed, and if not returns a default implementation
		if needsSampling(self):
			return super().intersects(other)

		elif isinstance(other, MeshVolumeRegion):
			# PASS 1
			# Check if bounding boxes intersect. If not, volumes cannot intersect.
			# For bounding boxes to intersect there must be overlap of the bounds
			# in all 3 dimensions.
			range_overlaps = [(self.mesh.bounds[0,dim] <= other.mesh.bounds[1,dim]) and \
							  (other.mesh.bounds[0,dim] <= self.mesh.bounds[1,dim]) \
							  for dim in range(3)]
			bb_overlap = all(range_overlaps)

			if not bb_overlap:
				return False

			# PASS 2
			# Compute inradius and circumradius for a candidate point in each region,
			# and compute the inradius and circumradius of each point. If the candidate 
			# points are closer than the sum of the inradius values, they must intersect.
			# If the candidate points are farther apart than the sum of the circumradius
			# values, they can't intersect.

			# Get a candidate point from each mesh. If the center of the object is in the mesh use that.
			# Otherwise try to sample a point as a candidate, skipping this pass if the sample fails.
			if self.containsPoint(Vector(*self.mesh.bounding_box.center_mass)):
				s_candidate_point = Vector(*self.mesh.bounding_box.center_mass)
			elif len(samples:=trimesh.sample.volume_mesh(self.mesh, self.num_samples)) > 0:
				s_candidate_point = Vector(*samples[0])
			else:
				s_candidate_point = None

			if other.containsPoint(Vector(*other.mesh.bounding_box.center_mass)):
				o_candidate_point = Vector(*other.mesh.bounding_box.center_mass)
			elif len(samples:=trimesh.sample.volume_mesh(other.mesh, other.num_samples)) > 0:
				o_candidate_point = Vector(*samples[0])
			else:
				o_candidate_point = None

			if s_candidate_point is not None and o_candidate_point is not None:
				# Compute the inradius of each object from its candidate point.
				s_inradius = abs(trimesh.proximity.ProximityQuery(self.mesh).signed_distance([s_candidate_point])[0])
				o_inradius = abs(trimesh.proximity.ProximityQuery(other.mesh).signed_distance([o_candidate_point])[0])

				# Compute the circumradius of each object from its candidate point.
				s_circumradius = numpy.max(numpy.linalg.norm(self.mesh.vertices - s_candidate_point, axis=1))
				o_circumradius = numpy.max(numpy.linalg.norm(other.mesh.vertices - o_candidate_point, axis=1))

				# Get the distance between the two points and check for mandatory or impossible collision.
				point_distance = s_candidate_point.distanceTo(o_candidate_point)

				if point_distance < s_inradius + o_inradius:
					return True

				if point_distance > s_circumradius + o_circumradius:
					return False

			# PASS 3
			# Use Trimesh's collision manager to check for intersection.
			# If the surfaces collide, that implies a collision of the volumes.
			# Cheaper than computing volumes immediately.
			collision_manager = trimesh.collision.CollisionManager()

			collision_manager.add_object("SelfRegion", self.mesh)
			collision_manager.add_object("OtherRegion", other.mesh)

			surface_collision = collision_manager.in_collision_internal()

			if surface_collision:
				return True

			# PASS 4
			# Compute intersection and check if it's empty. Expensive but guaranteed
			# to give the right answer.
			return not isinstance(self.intersect(other), EmptyRegion)

		elif isinstance(other, MeshSurfaceRegion):
			# PASS 1
			# Check if bounding boxes intersect. If not, volumes cannot intersect.
			# For bounding boxes to intersect there must be overlap of the bounds
			# in all 3 dimensions.
			range_overlaps = [(self.mesh.bounds[0,dim] <= other.mesh.bounds[1,dim]) and \
							  (other.mesh.bounds[0,dim] <= self.mesh.bounds[1,dim]) \
							  for dim in range(3)]
			bb_overlap = all(range_overlaps)

			if not bb_overlap:
				return False

			# PASS 2
			# Use Trimesh's collision manager to check for intersection.
			# If the surfaces collide, that implies a collision of the volumes.
			# Cheaper than computing volumes immediately.
			collision_manager = trimesh.collision.CollisionManager()

			collision_manager.add_object("SelfRegion", self.mesh)
			collision_manager.add_object("OtherRegion", other.mesh)

			surface_collision = collision_manager.in_collision_internal()

			if surface_collision:
				return True

			# PASS 3
			# Compute intersection and check if it's empty. Expensive but guaranteed
			# to give the right answer.
			return not isinstance(self.intersect(other), EmptyRegion)

		elif not triedReversed:
			return other.intersects(self)

		raise NotImplementedError("Cannot check intersection of MeshRegion with " +
			type(other) + ".")

	def containsPoint(self, point):
		"""Check if this region's volume contains a point."""
		return self.distanceTo(point) < self.tolerance

	def containsObject(self, obj):
		"""Check if this region's volume contains an :obj:`~scenic.core.object_types.Object`.
		The object must support coercion to a mesh.
		"""
		# PASS 1
		# Check if bounding boxes intersect. If not, volumes cannot intersect and so
		# the object cannot be contained in this region.
		range_overlaps = [(self.mesh.bounds[0,dim] <= obj.occupiedSpace.mesh.bounds[1,dim]) and \
						  (obj.occupiedSpace.mesh.bounds[0,dim] <= self.mesh.bounds[1,dim]) \
						  for dim in range(3)]
		bb_overlap = all(range_overlaps)

		if not bb_overlap:
			return False

		# PASS 2
		# Take the object's position if contained in the mesh, or a random sample otherwise.
		# Then check if the point is not in the region, return False if so. Otherwise, compute
		# the circumradius of the object from that point and see if the closest point on the
		# mesh is farther than the circumradius. If it is, then the object must be contained and
		# return True.

		# Get a candidate point from the object mesh. If the position of the object is in the mesh use that.
		# Otherwise try to sample a point as a candidate, skipping this pass if the sample fails.
		if obj.containsPoint(obj.position):
			obj_candidate_point = obj.position
		elif len(samples:=trimesh.sample.volume_mesh(obj.occupiedSpace.mesh, obj.occupiedSpace.num_samples)) > 0:
			obj_candidate_point = Vector(*samples[0])
		else:
			obj_candidate_point = None

		if obj_candidate_point is not None:
			# If this region doesn't contain the candidate point, it can't contain the object.
			if not self.containsPoint(obj_candidate_point):
				return False

			# Compute the circumradius of the object from the candidate point.
			obj_circumradius = numpy.max(numpy.linalg.norm(obj.occupiedSpace.mesh.vertices - obj_candidate_point, axis=1))

			# Compute the minimum distance from the region to this point.
			pq = trimesh.proximity.ProximityQuery(self.mesh)
			region_distance = abs(pq.signed_distance([obj_candidate_point])[0])

			if region_distance > obj_circumradius:
				return True

		# PASS 3
		# Take the region's center_mass if contained in the mesh, or a random sample otherwise.
		# Then get the circumradius of the region from that point and the farthest point on
		# the object from this point. If the maximum distance is greater than the circumradius,
		# return False.

		# Get a candidate point from the rgion mesh. If the center of mass of the region is in the mesh use that.
		# Otherwise try to sample a point as a candidate, skipping this pass if the sample fails.
		if self.containsPoint(Vector(*self.mesh.bounding_box.center_mass)):
			reg_candidate_point = Vector(*self.mesh.bounding_box.center_mass)
		elif len(samples:=trimesh.sample.volume_mesh(self.mesh, self.num_samples)) > 0:
			reg_candidate_point = Vector(*samples[0])
		else:
			reg_candidate_point = None

		if obj_candidate_point is not None:
			# Calculate circumradius of the region from the candidate_point
			reg_circumradius = numpy.max(numpy.linalg.norm(self.mesh.vertices - reg_candidate_point, axis=1))

			# Calculate maximum distance to the object.
			obj_max_distance = numpy.max(numpy.linalg.norm(obj.occupiedSpace.mesh.vertices - reg_candidate_point, axis=1))

			if obj_max_distance > reg_circumradius:
				return False

		# PASS 4
		# If the difference between the object's region and this region is empty,
		# i.e. obj_region - self_region = EmptyRegion, that means the object is
		# entirely contained in this region. We also return true if the result is a MeshSurfaceRegion,
		# as this usually means the object and this region share a surface. 
		# right answer. 
		diff_region = obj.occupiedSpace.difference(self)
		if isinstance(diff_region, EmptyRegion) or isinstance(diff_region, MeshSurfaceRegion):
			return True

	def uniformPointInner(self):
		""" Samples a point uniformly from the volume of the region"""
		# TODO: Look into tetrahedralization, perhaps to be turned on when a heuristic
		# is met. Currently using Trimesh's rejection sampling.
		sample = trimesh.sample.volume_mesh(self.mesh, self.num_samples)

		if len(sample) == 0:
			raise RejectionException("Rejection sampling MeshVolumeRegion failed.")
		else:
			return Vector(*sample[0])

	def distanceTo(self, point):
		""" Get the minimum distance from this region (including volume) to the specified point."""
		point = toVector(point, "Could not convert 'point' to vector.")

		pq = trimesh.proximity.ProximityQuery(self.mesh)

		dist = pq.signed_distance([point.coordinates])[0]

		# Positive distance indicates being contained in the mesh.
		if dist > 0:
			dist = 0

		return abs(dist)

	## Sampling Methods ##
	def sampleGiven(self, value):
		return MeshVolumeRegion(mesh=self._mesh, name=self.name, \
			dimensions=value[self.dimensions], position=value[self.position], rotation=value[self.rotation], \
			orientation=self.orientation, tolerance=self.tolerance, center_mesh=self.center_mesh, engine=self.engine)

	## Utility Methods ##
	def getSurfaceRegion(self):
		""" Return a region equivalent to this one except as a MeshSurfaceRegion"""
		return MeshSurfaceRegion(self.mesh, self.name, center_mesh=False)

	def getVolumeRegion(self):
		""" Returns this object, as it is already a MeshVolumeRegion"""
		return self


class MeshSurfaceRegion(_MeshRegion):
	""" An instance of _MeshRegion that performs operations over the surface
	of the mesh.
	"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	# Property testing methods #
	def intersects(self, other, triedReversed=False):
		"""	Check if this region's surface intersects another region.
		This is equivalent to checking if the two meshes collide
		"""
		# Check if region is fixed, and if not returns a default implementation
		if needsSampling(self):
			return super().intersects(other)

		elif isinstance(other, MeshSurfaceRegion):
			# Uses Trimesh's collision manager to check for intersection of the
			# polygons.
			collision_manager = trimesh.collision.CollisionManager()

			collision_manager.add_object("SelfRegion", self.mesh)
			collision_manager.add_object("OtherRegion", other.mesh)

			surface_collision = collision_manager.in_collision_internal()

			return surface_collision

		elif not triedReversed:
			return other.intersects(self)

		raise NotImplementedError("Cannot check intersection of MeshRegion with " +
			type(other) + ".")

	def containsPoint(self, point):
		"""Check if this region's surface contains a point."""
		# First compute the minimum distance to the point.
		min_distance = self.distanceTo(point)

		# If the minimum distance is within tolerance of 0, the mesh contains the point.
		return min_distance < self.tolerance

	def containsObject(self, obj):
		# A surface cannot contain an object, which must have a volume.
		return False

	def uniformPointInner(self):
		""" Sample a point uniformly at random from the surface of them mesh"""
		return Vector(*trimesh.sample.sample_surface(self.mesh, 1)[0][0])

	def distanceTo(self, point):
		""" Get the minimum distance from this object to the specified point."""
		pq = trimesh.proximity.ProximityQuery(self.mesh)

		dist = abs(pq.signed_distance([point.coordinates])[0])

		return dist

	## Sampling Methods ##
	def sampleGiven(self, value):
		return MeshSurfaceRegion(mesh=self._mesh, name=self.name, \
			dimensions=value[self.dimensions], position=value[self.position], rotation=value[self.rotation], \
			orientation=self.orientation, tolerance=self.tolerance, center_mesh=self.center_mesh, engine=self.engine)

	## Utility Methods ##
	def getVolumeRegion(self):
		""" Return a region equivalent to this one except as a MeshVolumeRegion"""
		return MeshVolumeRegion(self.mesh, self.name, center_mesh=False)

	def getSurfaceRegion(self):
		return self

class BoxRegion(MeshVolumeRegion):
	"""Region in the shape of a rectangular cuboid, i.e. a box. By default the unit box centered at the origin
	and aligned with the axes is used.

	:param name: An optional name to help with debugging.
	:param position: An optional position, which determines where the center of the region will be.
	:param position: An optional Orientation object which determines the rotation of the object in space.
	:param dimensions: An optional 3-tuple, describing the length, width, and height of the box.
	:param orientation: An optional vector field describing the preferred orientation at every point in
		the region.
	:param tolerance: Tolerance for collision computations.
	"""
	def __init__(self, name=None, position=None, rotation=None, dimensions=None, orientation=None, tolerance=1e-8, engine="blender"):
		box_mesh = trimesh.creation.box((1, 1, 1))
		super().__init__(mesh=box_mesh, name=name, position=position, rotation=rotation, dimensions=dimensions,\
		orientation=orientation, tolerance=tolerance, engine=engine)

class SpheroidRegion(MeshVolumeRegion):
	"""Region in the shape of a spheroid. By default the unit sphere centered at the origin
	and aligned with the axes is used.

	:param name: An optional name to help with debugging.
	:param position: An optional position, which determines where the center of the region will be.
	:param position: An optional Orientation object which determines the rotation of the object in space.
	:param dimensions: An optional 3-tuple, describing the length, width, and height of the box.
	:param orientation: An optional vector field describing the preferred orientation at every point in
		the region.
	:param tolerance: Tolerance for collision computations.
	"""
	def __init__(self, name=None, position=None, rotation=None, dimensions=None, orientation=None, tolerance=1e-8, engine="blender"):
		sphere_mesh = trimesh.creation.icosphere(radius=1)
		super().__init__(mesh=sphere_mesh, name=name, position=position, rotation=rotation, dimensions=dimensions, \
			orientation=orientation, tolerance=tolerance, engine=engine)

class PyramidViewRegion(MeshVolumeRegion):
	"""
	:param visibleDistance: The view distance for this region (will be slightly amplified to 
		prevent mesh intersection errors).
	:param viewAngles: The view angles for this region.
	:param rotation: An optional Orientation object which determines the rotation of the object in space.
	"""
	def __init__(self, visibleDistance, viewAngles, rotation=None):
		if min(viewAngles) <= 0 or max(viewAngles) >= math.pi:
			raise ValueError("viewAngles members must be between 0 and Pi.")

		x_dim = 2*visibleDistance*math.tan(viewAngles[0]/2)
		z_dim = 2*visibleDistance*math.tan(viewAngles[1]/2)

		dimensions = (x_dim, visibleDistance*1.01, z_dim)

		# Create pyramid mesh and scale it appropriately.
		vertices = [[ 0,  0,  0],
		            [-1,  1,  1],
		            [ 1,  1,  1],
		            [ 1,  1, -1],
		            [-1,  1, -1]]

		faces = [[0,2,1],
		         [0,3,2],
		         [0,4,3],
	             [0,1,4],
		         [1,2,4],
		         [2,3,4]]

		pyramid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

		scale = pyramid_mesh.extents / numpy.array(dimensions)

		scale_matrix = numpy.eye(4)
		scale_matrix[:3, :3] /= scale

		pyramid_mesh.apply_transform(scale_matrix)

		super().__init__(mesh=pyramid_mesh, rotation=rotation, center_mesh=False)


class TriangularPrismViewRegion(MeshVolumeRegion):
	"""
	:param visibleDistance: The view distance for this region (will be slightly amplified to 
		prevent mesh intersection errors).
	:param viewAngles: The view angles for this region.
	:param rotation: An optional Orientation object which determines the rotation of the object in space.
	"""
	def __init__(self, visibleDistance, viewAngle, rotation=None):
		if viewAngle <= 0 or viewAngle >= math.pi:
			raise ValueError("viewAngles members must be between 0 and Pi.")

		y_dim = 1.01*visibleDistance
		z_dim = 2*y_dim
		x_dim = 2*math.tan(viewAngle/2)*y_dim

		dimensions = (x_dim, y_dim, z_dim)

		# Create triangualr prism mesh and scale it appropriately.
		vertices = [[ 0,  0,  1], # 0 - Top origin
					[ 0,  0, -1], # 1 - Bottom origin
					[-1,  1,  1], # 2 - Top left
					[ 1,  1,  1], # 3 - Top right
					[-1,  1, -1], # 4 - Bottom left
					[ 1,  1, -1]] # 5 - Bottom right

		faces = [
				 [0,3,2], # Top triangle
				 [1,4,5], # Bottom triangle
				 [1,0,2], # Left 1
				 [1,2,4], # Left 2
				 [1,3,0], # Right 1
				 [1,5,3], # Right 2
				 [4,2,3], # Back 1
				 [4,3,5], # Back 2
				]

		tprism_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

		scale = tprism_mesh.extents / numpy.array(dimensions)

		scale_matrix = numpy.eye(4)
		scale_matrix[:3, :3] /= scale

		tprism_mesh.apply_transform(scale_matrix)

		super().__init__(mesh=tprism_mesh, rotation=rotation, center_mesh=False)

class DefaultViewRegion(MeshVolumeRegion):
	""" The default view region shape.
	:param visibleDistance: The view distance for this region.
	:param viewAngles: The view angles for this region.
	:param name: An optional name to help with debugging.
	:param position: An optional position, which determines where the center of the region will be.
	:param position: An optional Orientation object which determines the rotation of the object in space.
	:param orientation: An optional vector field describing the preferred orientation at every point in
		the region.
	:param tolerance: Tolerance for collision computations.
	"""
	def __init__(self, visibleDistance, viewAngles, name=None, position=Vector(0,0,0), rotation=None,\
		orientation=None, tolerance=1e-8):
		# Bound viewAngles from either side.
		viewAngles = tuple([min(angle, math.tau) for angle in viewAngles])

		if min(viewAngles) <= 0:
			raise ValueError("viewAngles cannot have a component less than or equal to 0")

		# Cases in view region computation
		# Case 1: 		Azimuth view angle = 360 degrees
		# 	Case 1.a: 	Altitude view angle = 180 degrees 	=> Full Sphere View Region
		# 	Case 1.b: 	Altitude view angle < 180 degrees  	=> Sphere - (Cone + Cone) (Cones on z axis expanding from origin)
		# Case 2: 		Azimuth view angle = 180 degrees
		# 	Case 2.a:	Altitude view angle = 180 degrees 	=> Hemisphere View Region
		# 	Case 2.b:	Altitude view angle < 180 degrees	=> Hemisphere - (Cone + Cone) (Cones on appropriate hemispheres)
		# Case 3:		Altitude view angle = 180 degrees	
		#	Case 3.a: 	Azimuth view angle < 180 degrees	=> Sphere intersected with Pyramid View Region
		#	Case 3.b: 	Azimuth view angle > 180 degrees	=> Sphere - Backwards Pyramid View Region 
		# Case 4: 		Both view angles < 180 				=> Capped Pyramid View Region
		# Case 5:		Azimuth > 180, Altitude < 180		=> (Sphere - (Cone + Cone) (Cones on appropriate hemispheres)) - Backwards Capped Pyramid View Region

		view_region = None
		diameter = 2*visibleDistance
		base_sphere = SpheroidRegion(dimensions=(diameter, diameter, diameter), engine="scad")

		if (math.tau-0.01 <= viewAngles[0] <= math.tau+0.01):
			# Case 1
			if viewAngles[1] > math.pi-0.01:
				#Case 1.a
				view_region = base_sphere
			else:
				# Case 1.b
				# Create cone with yaw oriented around (0,0,-1)
				padded_height = visibleDistance * 2
				radius = padded_height*math.tan((math.pi-viewAngles[1])/2)

				cone_mesh = trimesh.creation.cone(radius=radius, height=padded_height)

				position_matrix = translation_matrix((0,0,-1*padded_height))
				cone_mesh.apply_transform(position_matrix)

				# Create two cones around the yaw axis
				orientation_1 = Orientation.fromEuler(0,0,0)
				orientation_2 = Orientation.fromEuler(0,0,math.pi)

				cone_1 = MeshVolumeRegion(mesh=cone_mesh, rotation=orientation_1, center_mesh=False)
				cone_2 = MeshVolumeRegion(mesh=cone_mesh, rotation=orientation_2, center_mesh=False)

				view_region = base_sphere.difference(cone_1).difference(cone_2)

		elif (math.pi-0.01 <= viewAngles[0] <= math.pi+0.01):
			# Case 2
			if viewAngles[1] > math.pi-0.01:
				# Case 2.a
				padded_diameter = 1.1*diameter
				view_region = base_sphere.intersect(BoxRegion(dimensions=(padded_diameter, padded_diameter, padded_diameter), position=(0,padded_diameter/2,0)))
			else:
				# Case 2.b
				# Create cone with yaw oriented around (0,0,-1)
				padded_height = visibleDistance * 2
				radius = padded_height*math.tan((math.pi-viewAngles[1])/2)

				cone_mesh = trimesh.creation.cone(radius=radius, height=padded_height)

				position_matrix = translation_matrix((0,0,-1*padded_height))
				cone_mesh.apply_transform(position_matrix)

				# Create two cones around the yaw axis
				orientation_1 = Orientation.fromEuler(0,0,0)
				orientation_2 = Orientation.fromEuler(0,0,math.pi)

				cone_1 = MeshVolumeRegion(mesh=cone_mesh, rotation=orientation_1, center_mesh=False)
				cone_2 = MeshVolumeRegion(mesh=cone_mesh, rotation=orientation_2, center_mesh=False)

				padded_diameter = 1.1*diameter

				base_hemisphere = base_sphere.intersect(BoxRegion(dimensions=(padded_diameter, padded_diameter, padded_diameter), position=(0,padded_diameter/2,0)))

				view_region = base_hemisphere.difference(cone_1).difference(cone_2)

		elif viewAngles[1] > math.pi-0.01:
			# Case 3
			if viewAngles[0] < math.pi:
				view_region = base_sphere.intersect(TriangularPrismViewRegion(visibleDistance, viewAngles[0]))
			elif viewAngles[0] > math.pi:
				back_tprism = TriangularPrismViewRegion(visibleDistance, math.tau - viewAngles[0], rotation=Orientation.fromEuler(math.pi, 0, 0))
				view_region = base_sphere.difference(back_tprism)
			else:
				assert False, f"{viewAngles=}"

		elif viewAngles[0] < math.pi and viewAngles[1] < math.pi:
			# Case 4
			view_region = base_sphere.intersect(PyramidViewRegion(visibleDistance, viewAngles))
		elif viewAngles[0] > math.pi and viewAngles[1] < math.pi:
			# Case 5
			# Create cone with yaw oriented around (0,0,-1)
			padded_height = visibleDistance * 2
			radius = padded_height*math.tan((math.pi-viewAngles[1])/2)

			cone_mesh = trimesh.creation.cone(radius=radius, height=padded_height)

			position_matrix = translation_matrix((0,0,-1*padded_height))
			cone_mesh.apply_transform(position_matrix)

			# Position on the yaw axis
			orientation_1 = Orientation.fromEuler(0,0,0)
			orientation_2 = Orientation.fromEuler(0,0,math.pi)

			cone_1 = MeshVolumeRegion(mesh=cone_mesh, rotation=orientation_1, center_mesh=False)
			cone_2 = MeshVolumeRegion(mesh=cone_mesh, rotation=orientation_2, center_mesh=False)

			backwards_view_angle = (math.tau-viewAngles[0], math.pi-0.01)
			back_pyramid = PyramidViewRegion(visibleDistance, backwards_view_angle, rotation=Orientation.fromEuler(math.pi, 0, 0))

			# Note: Openscad does not like the result of the difference with the cones, so they must be done last.
			view_region = base_sphere.difference(back_pyramid).difference(cone_1).difference(cone_2)
		else:
			assert False, f"{viewAngles=}"

		assert view_region is not None

		# Initialize volume region
		super().__init__(mesh=view_region.mesh, name=name, position=position, rotation=rotation, orientation=orientation, \
			tolerance=tolerance, center_mesh=False)

###################################################################################################
# 2D Regions
###################################################################################################

class CircularRegion(Region):
	"""A circular region with a possibly-random center and radius.

	Args:
		center (`Vector`): center of the disc.
		radius (float): radius of the disc.
		resolution (int; optional): number of vertices to use when approximating this region as a
			polygon.
		name (str; optional): name for debugging.
	"""
	def __init__(self, center, radius, resolution=32, name=None):
		super().__init__(name, center, radius)
		self.center = toVector(center, "center of CircularRegion not a vector")
		self.radius = toScalar(radius, "radius of CircularRegion not a scalar")
		self.circumcircle = (self.center, self.radius)
		self.resolution = resolution

	@cached_property
	def polygon(self):
		assert not (needsSampling(self.center) or needsSampling(self.radius))
		ctr = shapely.geometry.Point(self.center)
		return ctr.buffer(self.radius, resolution=self.resolution)

	def sampleGiven(self, value):
		return CircularRegion(value[self.center], value[self.radius],
							  name=self.name, resolution=self.resolution)

	def evaluateInner(self, context, modifying):
		center = valueInContext(self.center, context, modifying)
		radius = valueInContext(self.radius, context, modifying)
		return CircularRegion(center, radius,
							  name=self.name, resolution=self.resolution)

	def intersects(self, other, triedReversed=False):
		if isinstance(other, CircularRegion):
			return self.center.distanceTo(other.center) <= self.radius + other.radius
		return super().intersects(other, triedReversed)

	def containsPoint(self, point):
		point = point.toVector()
		return point.distanceTo(self.center) <= self.radius

	def distanceTo(self, point):
		return max(0, point.distanceTo(self.center) - self.radius)

	def uniformPointInner(self):
		x, y, z = self.center
		r = random.triangular(0, self.radius, self.radius)
		t = random.uniform(-math.pi, math.pi)
		pt = Vector(x + (r * cos(t)), y + (r * sin(t)), z)
		return self.orient(pt)

	def getAABB(self):
		x, y, _ = self.center
		r = self.radius
		return ((x - r, y - r), (x + r, y + r))

	def __repr__(self):
		return f'CircularRegion({self.center}, {self.radius})'

class SectorRegion(Region):
	"""A sector of a `CircularRegion`.

	This region consists of a sector of a disc, i.e. the part of a disc subtended by a
	given arc.

	Args:
		center (`Vector`): center of the corresponding disc.
		radius (float): radius of the disc.
		heading (float): heading of the centerline of the sector.
		angle (float): angle subtended by the sector.
		resolution (int; optional): number of vertices to use when approximating this region as a
			polygon.
		name (str; optional): name for debugging.
	"""
	def __init__(self, center, radius, heading, angle, resolution=32, name=None):
		self.center = toVector(center, "center of SectorRegion not a vector")
		self.radius = toScalar(radius, "radius of SectorRegion not a scalar")
		self.heading = toScalar(heading, "heading of SectorRegion not a scalar")
		self.angle = toScalar(angle, "angle of SectorRegion not a scalar")
		super().__init__(name, self.center, radius, heading, angle)
		r = (radius / 2) * cos(angle / 2)
		self.circumcircle = (self.center.offsetRadially(r, heading), r)
		self.resolution = resolution

	@cached_property
	def polygon(self):
		center, radius = self.center, self.radius
		ctr = shapely.geometry.Point(center)
		circle = ctr.buffer(radius, resolution=self.resolution)
		if self.angle >= math.tau - 0.001:
			return circle
		else:
			heading = self.heading
			half_angle = self.angle / 2
			mask = shapely.geometry.Polygon([
				center,
				center.offsetRadially(radius, heading + half_angle),
				center.offsetRadially(2*radius, heading),
				center.offsetRadially(radius, heading - half_angle)
			])
			return circle & mask

	def sampleGiven(self, value):
		return SectorRegion(value[self.center], value[self.radius],
			value[self.heading], value[self.angle],
			name=self.name, resolution=self.resolution)

	def evaluateInner(self, context, modifying):
		center = valueInContext(self.center, context, modifying)
		radius = valueInContext(self.radius, context, modifying)
		heading = valueInContext(self.heading, context, modifying)
		angle = valueInContext(self.angle, context, modifying)
		return SectorRegion(center, radius, heading, angle,
							name=self.name, resolution=self.resolution)

	def containsPoint(self, point):
		point = point.toVector()
		if not pointIsInCone(tuple(point), tuple(self.center), self.heading, self.angle):
			return False
		return point.distanceTo(self.center) <= self.radius

	def uniformPointInner(self):
		x, y, z = self.center
		heading, angle, maxDist = self.heading, self.angle, self.radius
		r = random.triangular(0, maxDist, maxDist)
		ha = angle / 2.0
		t = random.uniform(-ha, ha) + (heading + (math.pi / 2))
		pt = Vector(x + (r * cos(t)), y + (r * sin(t)), z)
		return self.orient(pt)

	def __repr__(self):
		return f'SectorRegion({self.center},{self.radius},{self.heading},{self.angle})'

class RectangularRegion(_RotatedRectangle, Region):
	"""A rectangular region with a possibly-random position, heading, and size.

	Args:
		position (`Vector`): center of the rectangle.
		heading (float): the heading of the ``length`` axis of the rectangle.
		width (float): width of the rectangle.
		length (float): length of the rectangle.
		name (str; optional): name for debugging.
	"""
	def __init__(self, position, heading, width, length, name=None):
		super().__init__(name, position, heading, width, length)
		self.position = toVector(position, "position of RectangularRegion not a vector")
		self.heading = toScalar(heading, "heading of RectangularRegion not a scalar")
		self.width = toScalar(width, "width of RectangularRegion not a scalar")
		self.length = toScalar(length, "length of RectangularRegion not a scalar")
		self.hw = hw = width / 2
		self.hl = hl = length / 2
		self.radius = hypot(hw, hl)		# circumcircle; for collision detection
		self.corners = tuple(self.position.offsetRotated(heading, Vector(*offset))
			for offset in ((hw, hl), (-hw, hl), (-hw, -hl), (hw, -hl)))
		self.circumcircle = (self.position, self.radius)

	def sampleGiven(self, value):
		return RectangularRegion(value[self.position], value[self.heading],
			value[self.width], value[self.length],
			name=self.name)

	def evaluateInner(self, context, modifying):
		position = valueInContext(self.position, context, modifying)
		heading = valueInContext(self.heading, context, modifying)
		width = valueInContext(self.width, context, modifying)
		length = valueInContext(self.length, context, modifying)
		return RectangularRegion(position, heading, width, length,
								 name=self.name)

	def uniformPointInner(self):
		hw, hl = self.hw, self.hl
		rx = random.uniform(-hw, hw)
		ry = random.uniform(-hl, hl)
		pt = self.position.offsetRotated(self.heading, Vector(rx, ry))
		return self.orient(pt)

	def getAABB(self):
		x, y, z = zip(*self.corners)
		minx, maxx = findMinMax(x)
		miny, maxy = findMinMax(y)
		return ((minx, miny), (maxx, maxy))

	def __repr__(self):
		return f'RectangularRegion({self.position},{self.heading},{self.width},{self.length})'

class PolylineRegion(Region):
	"""Region given by one or more polylines (chain of line segments).

	The region may be specified by giving either a sequence of points or ``shapely``
	polylines (a ``LineString`` or ``MultiLineString``).

	Args:
		points: sequence of points making up the polyline (or `None` if using the
			**polyline** argument instead).
		polyline: ``shapely`` polyline or collection of polylines (or `None` if using
			the **points** argument instead).
		orientation (optional): :term:`preferred orientation` to use, or `True` to use an
			orientation aligned with the direction of the polyline (the default).
		name (str; optional): name for debugging.
	"""
	def __init__(self, points=None, polyline=None, orientation=True, name=None):
		if orientation is True:
			orientation = VectorField('Polyline', self.defaultOrientation)
			self.usingDefaultOrientation = True
		else:
			self.usingDefaultOrientation = False
		super().__init__(name, orientation=orientation)
		if points is not None:
			points = tuple(pt[:2] for pt in points)
			if len(points) < 2:
				raise RuntimeError('tried to create PolylineRegion with < 2 points')
			self.points = points
			self.lineString = shapely.geometry.LineString(points)
		elif polyline is not None:
			if isinstance(polyline, shapely.geometry.LineString):
				if len(polyline.coords) < 2:
					raise RuntimeError('tried to create PolylineRegion with <2-point LineString')
			elif isinstance(polyline, shapely.geometry.MultiLineString):
				if len(polyline.geoms) == 0:
					raise RuntimeError('tried to create PolylineRegion from empty MultiLineString')
				for line in polyline.geoms:
					assert len(line.coords) >= 2
			else:
				raise RuntimeError('tried to create PolylineRegion from non-LineString')
			self.lineString = polyline
			self.points = None
		else:
			raise RuntimeError('must specify points or polyline for PolylineRegion')
		if not self.lineString.is_valid:
			raise RuntimeError('tried to create PolylineRegion with '
							   f'invalid LineString {self.lineString}')
		self.segments = self.segmentsOf(self.lineString)
		cumulativeLengths = []
		total = 0
		for p, q in self.segments:
			dx, dy = p[0] - q[0], p[1] - q[1]
			total += math.hypot(dx, dy)
			cumulativeLengths.append(total)
		self.cumulativeLengths = cumulativeLengths
		if self.points is None:
			pts = []
			for p, q in self.segments:
				pts.append(p)
			pts.append(q)
			self.points = pts

	@classmethod
	def segmentsOf(cls, lineString):
		if isinstance(lineString, shapely.geometry.LineString):
			segments = []
			points = list(lineString.coords)
			if len(points) < 2:
				raise RuntimeError('LineString has fewer than 2 points')
			last = points[0]
			for point in points[1:]:
				segments.append((last, point))
				last = point
			return segments
		elif isinstance(lineString, shapely.geometry.MultiLineString):
			allSegments = []
			for line in lineString.geoms:
				allSegments.extend(cls.segmentsOf(line))
			return allSegments
		else:
			raise RuntimeError('called segmentsOf on non-linestring')

	@cached_property
	def start(self):
		"""Get an `OrientedPoint` at the start of the polyline.

		The OP's heading will be aligned with the orientation of the region, if
		there is one (the default orientation pointing along the polyline).
		"""
		pointA, pointB = self.segments[0]
		if self.usingDefaultOrientation:
			heading = headingOfSegment(pointA, pointB)
		elif self.orientation is not None:
			heading = self.orientation[pointA]
		else:
			heading = 0
		from scenic.core.object_types import OrientedPoint
		return OrientedPoint(position=pointA, yaw=heading)

	@cached_property
	def end(self):
		"""Get an `OrientedPoint` at the end of the polyline.

		The OP's heading will be aligned with the orientation of the region, if
		there is one (the default orientation pointing along the polyline).
		"""
		pointA, pointB = self.segments[-1]
		if self.usingDefaultOrientation:
			heading = headingOfSegment(pointA, pointB)
		elif self.orientation is not None:
			heading = self.orientation[pointB]
		else:
			heading = 0
		from scenic.core.object_types import OrientedPoint
		return OrientedPoint(position=pointB, yaw=heading)

	def defaultOrientation(self, point):
		start, end = self.nearestSegmentTo(point)
		return start.angleTo(end)

	def uniformPointInner(self):
		pointA, pointB = random.choices(self.segments,
										cum_weights=self.cumulativeLengths)[0]
		interpolation = random.random()
		x, y = averageVectors(pointA, pointB, weight=interpolation)
		if self.usingDefaultOrientation:
			return OrientedVector(x, y, 0, headingOfSegment(pointA, pointB))
		else:
			return self.orient(Vector(x, y, 0))

	def intersect(self, other, triedReversed=False):
		poly = toPolygon(other)
		if poly is not None:
			intersection = self.lineString & poly
			if (intersection.is_empty or
				not isinstance(intersection, (shapely.geometry.LineString,
											  shapely.geometry.MultiLineString))):
				# TODO handle points!
				return nowhere
			return PolylineRegion(polyline=intersection)
		return super().intersect(other, triedReversed)

	def intersects(self, other, triedReversed=False):
		poly = toPolygon(other)
		if poly is not None:
			intersection = self.lineString & poly
			return not intersection.is_empty
		return super().intersects(other, triedReversed)

	def difference(self, other):
		poly = toPolygon(other)
		if poly is not None:
			diff = self.lineString - poly
			if (diff.is_empty or
				not isinstance(diff, (shapely.geometry.LineString,
									  shapely.geometry.MultiLineString))):
				# TODO handle points!
				return nowhere
			return PolylineRegion(polyline=diff)
		return super().difference(other)

	@staticmethod
	def unionAll(regions):
		regions = tuple(regions)
		if not regions:
			return nowhere
		if any(not isinstance(region, PolylineRegion) for region in regions):
			raise RuntimeError(f'cannot take Polyline union of regions {regions}')
		# take union by collecting LineStrings, to preserve the order of points
		strings = []
		for region in regions:
			string = region.lineString
			if isinstance(string, shapely.geometry.MultiLineString):
				strings.extend(string.geoms)
			else:
				strings.append(string)
		newString = shapely.geometry.MultiLineString(strings)
		return PolylineRegion(polyline=newString)

	def containsPoint(self, point):
		return self.lineString.intersects(shapely.geometry.Point(point))

	def containsObject(self, obj):
		return False

	@distributionMethod
	def distanceTo(self, point) -> float:
		return self.lineString.distance(shapely.geometry.Point(point))

	@distributionMethod
	def signedDistanceTo(self, point) -> float:
		"""Compute the signed distance of the PolylineRegion to a point.

		The distance is positive if the point is left of the nearest segment,
		and negative otherwise.
		"""
		dist = self.distanceTo(point)
		start, end = self.nearestSegmentTo(point)
		rp = point - start
		tangent = end - start
		return dist if tangent.angleWith(rp) >= 0 else -dist

	@distributionMethod
	def project(self, point):
		return shapely.ops.nearest_points(self.lineString, shapely.geometry.Point(point))[0]

	@distributionMethod
	def nearestSegmentTo(self, point):
		dist = self.lineString.project(shapely.geometry.Point(point))
		# TODO optimize?
		for segment, cumLen in zip(self.segments, self.cumulativeLengths):
			if dist <= cumLen:
				break
		# FYI, could also get here if loop runs to completion due to rounding error
		return (Vector(*segment[0]), Vector(*segment[1]))

	def pointAlongBy(self, distance, normalized=False) -> Vector:
		"""Find the point a given distance along the polyline from its start.

		If **normalized** is true, then distance should be between 0 and 1, and
		is interpreted as a fraction of the length of the polyline. So for example
		``pointAlongBy(0.5, normalized=True)`` returns the polyline's midpoint.
		"""
		pt = self.lineString.interpolate(distance, normalized=normalized)
		return Vector(pt.x, pt.y)

	def equallySpacedPoints(self, spacing, normalized=False):
		if normalized:
			spacing *= self.length
		return [self.pointAlongBy(d) for d in numpy.arange(0, self.length, spacing)]

	@property
	def length(self):
		return self.lineString.length

	def getAABB(self):
		xmin, ymin, xmax, ymax = self.lineString.bounds
		return ((xmin, ymin), (xmax, ymax))

	def show(self, plt, style='r-', **kwargs):
		plotPolygon(self.lineString, plt, style=style, **kwargs)

	def __getitem__(self, i) -> Vector:
		"""Get the ith point along this polyline.

		If the region consists of multiple polylines, this order is linear
		along each polyline but arbitrary across different polylines.
		"""
		return Vector(*self.points[i])

	def __add__(self, other):
		if not isinstance(other, PolylineRegion):
			return NotImplemented
		# take union by collecting LineStrings, to preserve the order of points
		strings = []
		for region in (self, other):
			string = region.lineString
			if isinstance(string, shapely.geometry.MultiLineString):
				strings.extend(string.geoms)
			else:
				strings.append(string)
		newString = shapely.geometry.MultiLineString(strings)
		return PolylineRegion(polyline=newString)

	def __len__(self) -> int:
		"""Get the number of vertices of the polyline."""
		return len(self.points)

	def __repr__(self):
		return f'PolylineRegion({self.lineString})'

	def __eq__(self, other):
		if type(other) is not PolylineRegion:
			return NotImplemented
		return (other.lineString == self.lineString)

	@cached
	def __hash__(self):
		return hash(str(self.lineString))

class PolygonalRegion(Region):
	"""Region given by one or more polygons (possibly with holes).

	The region may be specified by giving either a sequence of points defining the
	boundary of the polygon, or a collection of ``shapely`` polygons (a ``Polygon``
	or ``MultiPolygon``).

	Args:
		points: sequence of points making up the boundary of the polygon (or `None` if
			using the **polygon** argument instead).
		polygon: ``shapely`` polygon or collection of polygons (or `None` if using
			the **points** argument instead).
		orientation (`VectorField`; optional): :term:`preferred orientation` to use.
		name (str; optional): name for debugging.
	"""
	def __init__(self, points=None, polygon=None, orientation=None, name=None):
		super().__init__(name, orientation=orientation)
		if polygon is None and points is None:
			raise RuntimeError('must specify points or polygon for PolygonalRegion')
		if polygon is None:
			points = tuple(pt[:2] for pt in points)
			if len(points) == 0:
				raise RuntimeError('tried to create PolygonalRegion from empty point list!')
			for point in points:
				if needsSampling(point):
					raise RuntimeError('only fixed PolygonalRegions are supported')
			self.points = points
			polygon = shapely.geometry.Polygon(points)

		if isinstance(polygon, shapely.geometry.Polygon):
			self.polygons = shapely.geometry.MultiPolygon([polygon])
		elif isinstance(polygon, shapely.geometry.MultiPolygon):
			self.polygons = polygon
		else:
			raise RuntimeError(f'tried to create PolygonalRegion from non-polygon {polygon}')
		if not self.polygons.is_valid:
			raise RuntimeError('tried to create PolygonalRegion with '
							   f'invalid polygon {self.polygons}')

		if (points is None and len(self.polygons.geoms) == 1
		    and len(self.polygons.geoms[0].interiors) == 0):
			self.points = tuple(self.polygons.geoms[0].exterior.coords[:-1])

		if self.polygons.is_empty:
			raise RuntimeError('tried to create empty PolygonalRegion')

		triangles = []
		for polygon in self.polygons.geoms:
			triangles.extend(triangulatePolygon(polygon))
		assert len(triangles) > 0, self.polygons
		self.trianglesAndBounds = tuple((tri, tri.bounds) for tri in triangles)
		areas = (triangle.area for triangle in triangles)
		self.cumulativeTriangleAreas = tuple(itertools.accumulate(areas))

	def uniformPointInner(self):
		triangle, bounds = random.choices(
			self.trianglesAndBounds,
			cum_weights=self.cumulativeTriangleAreas)[0]
		minx, miny, maxx, maxy = bounds
		# TODO improve?
		while True:
			x, y = random.uniform(minx, maxx), random.uniform(miny, maxy)
			if triangle.intersects(shapely.geometry.Point(x, y)):
				return self.orient(Vector(x, y))

	def difference(self, other):
		poly = toPolygon(other)
		if poly is not None:
			diff = self.polygons - poly
			if diff.is_empty:
				return nowhere
			elif isinstance(diff, (shapely.geometry.Polygon,
								   shapely.geometry.MultiPolygon)):
				return PolygonalRegion(polygon=diff, orientation=self.orientation)
			elif isinstance(diff, shapely.geometry.GeometryCollection):
				polys = []
				for geom in diff.geoms:
					if isinstance(geom, shapely.geometry.Polygon):
						polys.append(geom)
				if len(polys) == 0:
					# TODO handle points, lines
					raise RuntimeError('unhandled type of polygon difference')
				diff = shapely.geometry.MultiPolygon(polys)
				return PolygonalRegion(polygon=diff, orientation=self.orientation)
			else:
				# TODO handle points, lines
				raise RuntimeError('unhandled type of polygon difference')
		return super().difference(other)

	def intersect(self, other, triedReversed=False):
		poly = toPolygon(other)
		orientation = other.orientation if self.orientation is None else self.orientation
		if poly is not None:
			intersection = self.polygons & poly
			if intersection.is_empty:
				return nowhere
			elif isinstance(intersection, (shapely.geometry.Polygon,
										 shapely.geometry.MultiPolygon)):
				return PolygonalRegion(polygon=intersection, orientation=orientation)
			elif isinstance(intersection, shapely.geometry.GeometryCollection):
				polys = []
				for geom in intersection.geoms:
					if isinstance(geom, shapely.geometry.Polygon):
						polys.append(geom)
				if len(polys) == 0:
					# TODO handle points, lines
					raise RuntimeError('unhandled type of polygon intersection')
				intersection = shapely.geometry.MultiPolygon(polys)
				return PolygonalRegion(polygon=intersection, orientation=orientation)
			else:
				# TODO handle points, lines
				raise RuntimeError('unhandled type of polygon intersection')
		return super().intersect(other, triedReversed)

	def intersects(self, other, triedReversed=False):
		poly = toPolygon(other)
		if poly is not None:
			intersection = self.polygons & poly
			return not intersection.is_empty
		return super().intersects(other, triedReversed)

	def union(self, other, triedReversed=False, buf=0):
		poly = toPolygon(other)
		if not poly:
			return super().union(other, triedReversed)
		union = polygonUnion((self.polygons, poly), buf=buf)
		orientation = VectorField.forUnionOf((self, other))
		return PolygonalRegion(polygon=union, orientation=orientation)

	@staticmethod
	def unionAll(regions, buf=0):
		regs, polys = [], []
		for reg in regions:
			if reg != nowhere:
				regs.append(reg)
				polys.append(toPolygon(reg))
		if not polys:
			return nowhere
		if any(not poly for poly in polys):
			raise RuntimeError(f'cannot take union of regions {regions}')
		union = polygonUnion(polys, buf=buf)
		orientation = VectorField.forUnionOf(regs)
		return PolygonalRegion(polygon=union, orientation=orientation)

	@property
	def boundary(self) -> PolylineRegion:
		"""Get the boundary of this region as a `PolylineRegion`."""
		return PolylineRegion(polyline=self.polygons.boundary)

	@cached_property
	def prepared(self):
		return shapely.prepared.prep(self.polygons)

	def containsPoint(self, point):
		return self.prepared.intersects(shapely.geometry.Point(point))

	def containsObject(self, obj):
		objPoly = obj.polygon
		if objPoly is None:
			raise RuntimeError('tried to test containment of symbolic Object!')
		# TODO improve boundary handling?
		return self.prepared.contains(objPoly)

	def containsRegion(self, other, tolerance=0):
		poly = toPolygon(other)
		if poly is None:
			raise RuntimeError('cannot test inclusion of {other} in PolygonalRegion')
		return self.polygons.buffer(tolerance).contains(poly)

	@distributionMethod
	def distanceTo(self, point):
		return self.polygons.distance(shapely.geometry.Point(point))

	def getAABB(self):
		xmin, ymin, xmax, ymax = self.polygons.bounds
		return ((xmin, ymin), (xmax, ymax))

	def show(self, plt, style='r-', **kwargs):
		plotPolygon(self.polygons, plt, style=style, **kwargs)

	def __repr__(self):
		return f'PolygonalRegion({self.polygons})'

	def __eq__(self, other):
		if type(other) is not PolygonalRegion:
			return NotImplemented
		return (other.polygons == self.polygons
				and other.orientation == self.orientation)

	@cached
	def __hash__(self):
		# TODO better way to hash mutable Shapely geometries? (also for PolylineRegion)
		return hash((str(self.polygons), self.orientation))

	def __getstate__(self):
		state = self.__dict__.copy()
		state.pop('_cached_prepared', None)		# prepared geometries are not picklable
		return state

class PointSetRegion(Region):
	"""Region consisting of a set of discrete points.

	No `Object` can be contained in a `PointSetRegion`, since the latter is discrete.
	(This may not be true for subclasses, e.g. `GridRegion`.)

	Args:
		name (str): name for debugging
		points (iterable): set of points comprising the region
		kdTree (`scipy.spatial.KDTree`, optional): k-D tree for the points (one will
		  be computed if none is provided)
		orientation (`VectorField`; optional): :term:`preferred orientation` for the
			region
		tolerance (float; optional): distance tolerance for checking whether a point lies
		  in the region
	"""

	def __init__(self, name, points, kdTree=None, orientation=None, tolerance=1e-6):
		super().__init__(name, orientation=orientation)
		self.points = tuple(points)
		for point in self.points:
			if needsSampling(point):
				raise RuntimeError('only fixed PointSetRegions are supported')
		import scipy.spatial	# slow import not often needed
		self.kdTree = scipy.spatial.KDTree(self.points) if kdTree is None else kdTree
		self.orientation = orientation
		self.tolerance = tolerance

	def uniformPointInner(self):
		return self.orient(Vector(*random.choice(self.points)))

	def intersect(self, other, triedReversed=False):
		def sampler(intRegion):
			o = intRegion.regions[1]
			center, radius = o.circumcircle
			# TODO: @Matthew ValueError: Searching for 3d point in 2d KDTree
			# Better way to fix this? 
			possibles = (Vector(*self.kdTree.data[i])
						 for i in self.kdTree.query_ball_point(center[:2], radius))
			intersection = [p for p in possibles if o.containsPoint(p)]
			if len(intersection) == 0:
				raise RejectionException(f'empty intersection of Regions {self} and {o}')
			return self.orient(random.choice(intersection))
		return IntersectionRegion(self, other, sampler=sampler, orientation=self.orientation)

	def containsPoint(self, point):
		distance, location = self.kdTree.query(point)
		return (distance <= self.tolerance)

	def containsObject(self, obj):
		raise NotImplementedError()

	@distributionMethod
	def distanceTo(self, point):
		distance, _ = self.kdTree.query(point)
		return distance

	def __eq__(self, other):
		if type(other) is not PointSetRegion:
			return NotImplemented
		return (self.name == other.name
		        and self.points == other.points
		        and self.orientation == other.orientation)

	@cached
	def __hash__(self):
		return hash((self.name, self.points, self.orientation))

class GridRegion(PointSetRegion):
	"""A Region given by an obstacle grid.

	A point is considered to be in a `GridRegion` if the nearest grid point is
	not an obstacle.

	Args:
		name (str): name for debugging
		grid: 2D list, tuple, or NumPy array of 0s and 1s, where 1 indicates an obstacle
		  and 0 indicates free space
		Ax (float): spacing between grid points along X axis
		Ay (float): spacing between grid points along Y axis
		Bx (float): X coordinate of leftmost grid column
		By (float): Y coordinate of lowest grid row
		orientation (`VectorField`; optional): orientation of region
	"""
	def __init__(self, name, grid, Ax, Ay, Bx, By, orientation=None):
		self.grid = numpy.array(grid)
		self.sizeY, self.sizeX = self.grid.shape
		self.Ax, self.Ay = Ax, Ay
		self.Bx, self.By = Bx, By
		y, x = numpy.where(self.grid == 0)
		points = [self.gridToPoint(point) for point in zip(x, y)]
		super().__init__(name, points, orientation=orientation)

	def gridToPoint(self, gp):
		x, y = gp
		return ((self.Ax * x) + self.Bx, (self.Ay * y) + self.By)

	def pointToGrid(self, point):
		x, y, z = point
		x = (x - self.Bx) / self.Ax
		y = (y - self.By) / self.Ay
		nx = int(round(x))
		if nx < 0 or nx >= self.sizeX:
			return None
		ny = int(round(y))
		if ny < 0 or ny >= self.sizeY:
			return None
		return (nx, ny)

	def containsPoint(self, point):
		gp = self.pointToGrid(point)
		if gp is None:
			return False
		x, y = gp
		return (self.grid[y, x] == 0)

	def containsObject(self, obj):
		# TODO improve this procedure!
		# Fast check
		for c in obj.corners:
			if not self.containsPoint(c):
				return False
		# Slow check
		gps = [self.pointToGrid(corner) for corner in obj.corners]
		x, y = zip(*gps)
		minx, maxx = findMinMax(x)
		miny, maxy = findMinMax(y)
		for x in range(minx, maxx+1):
			for y in range(miny, maxy+1):
				p = self.gridToPoint((x, y))
				if self.grid[y, x] == 1 and obj.containsPoint(p):
					return False
		return True
