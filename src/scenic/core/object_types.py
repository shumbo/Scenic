"""Implementations of the built-in Scenic classes."""

import collections
import math
import random
import numpy as np
import trimesh
from abc import ABC, abstractmethod

from scenic.core.distributions import Samplable, needsSampling, distributionMethod, distributionFunction
from scenic.core.specifiers import Specifier, PropertyDefault, ModifyingSpecifier
from scenic.core.vectors import Vector, Orientation, alwaysGlobalOrientation
from scenic.core.geometry import (_RotatedRectangle, averageVectors, hypot, min,
                                  pointIsInCone)
from scenic.core.regions import (Region, CircularRegion, SectorRegion, MeshVolumeRegion, MeshSurfaceRegion, 
								  BoxRegion, SpheroidRegion, DefaultViewRegion, EmptyRegion)
from scenic.core.type_support import toVector, toHeading, toType, toScalar
from scenic.core.lazy_eval import needsLazyEvaluation
from scenic.core.utils import DefaultIdentityDict, areEquivalent, cached_property
from scenic.core.errors import RuntimeParseError
from scenic.core.shapes import Shape, BoxShape, MeshShape
from scenic.core.regions import IntersectionRegion

## Abstract base class

class _Constructible(Samplable):
	"""Abstract base class for Scenic objects.

	Scenic objects, which are constructed using specifiers, are implemented
	internally as instances of ordinary Python classes. This abstract class
	implements the procedure to resolve specifiers and determine values for
	the properties of an object, as well as several common methods supported
	by objects.
	"""

	def __init_subclass__(cls):
		super().__init_subclass__()
		# find all defaults provided by the class or its superclasses
		allDefs = collections.defaultdict(list)
		for sc in cls.__mro__:
			if issubclass(sc, _Constructible) and hasattr(sc, '__annotations__'):
				for prop, value in sc.__annotations__.items():
					allDefs[prop].append(PropertyDefault.forValue(value))

		# resolve conflicting defaults and gather dynamic properties
		resolvedDefs = {}
		dyns = []
		finals = []
		for prop, defs in allDefs.items():
			primary, rest = defs[0], defs[1:]
			spec = primary.resolveFor(prop, rest)
			resolvedDefs[prop] = spec

			if any(defn.isDynamic for defn in defs):
				dyns.append(prop)
			if primary.isFinal:
				finals.append(prop)
		cls._defaults = resolvedDefs
		cls._dynamicProperties = tuple(dyns)
		cls._finalProperties = tuple(finals)

	@classmethod
	def withProperties(cls, props):
		assert all(reqProp in props for reqProp in cls._defaults)
		assert all(not needsLazyEvaluation(val) for val in props.values())
		return cls(_internal=True, **props)

	def __init__(self, *args, _internal=False, **kwargs):
		if _internal:	# Object is being constructed internally; use fast path
			assert not args
			for prop, value in kwargs.items():
				assert not needsLazyEvaluation(value), (prop, value)
				object.__setattr__(self, prop, value)
			super().__init__(kwargs.values())
			self.properties = set(kwargs.keys())
			return

		# Validate specifiers
		name = self.__class__.__name__
		specifiers = list(args)
		for prop, val in kwargs.items():	# kwargs supported for internal use
			specifiers.append(Specifier({prop: 1}, val, internal=True))

		# Declare properties dictionary which maps properties to the specifier
		# that will specify that property.
		properties = dict()

		# Declare modifying dictionary, which maps properties to a specifier
		# that will modify that property.
		modifying = dict()

		# Dictionary mapping properties set so far to the priority with which they have
		# been set.
		priorities = dict()

		# Extract default property values dictionary and set of final properties
		defs = self.__class__._defaults
		finals = self.__class__._finalProperties

		# TODO: @Matthew Check for incompatible specifiers used with modifying specifier (itself or `at`)

		# Split the specifiers into two groups, normal and modifying. Normal specifiers set all relevant properties
		# first. Then modifying specifiers can modify or set additional properties
		normal_specifiers = [spec for spec in specifiers if not isinstance(spec, ModifyingSpecifier)]
		modifying_specifiers = [spec for spec in specifiers if isinstance(spec, ModifyingSpecifier)]

		'''
		For each property specified by a normal specifier:
			- If not in properties specified, properties[p] = specifier
			- Otherwise, if property specified, check if specifier's priority is higher. If so, replace it with specifier

		Priorties are inversed: A lower priority number means semantically that it has a higher priority level
		'''
		for spec in normal_specifiers:
			assert isinstance(spec, Specifier), (name, spec)

			# Iterate over each property.
			for prop in spec.priorities:
				# Check if this is a final property has been specified. If so, throw an assertion or error,
				# depending on whether or not this object is internal.
				if prop in finals:
					assert not _internal
					raise RuntimeParseError(f'property "{prop}" of {name} cannot be directly specified')


				if prop in properties:
					# This property already exists. Check that it has not already been specified
					# at equal priority level. Then if it was previously specified at a lower priority
					# level, override it with the value that this specifier sets.
					if spec.priorities[prop] == priorities[prop]:
						raise RuntimeParseError(f'property "{prop}" of {name} specified twice with the same priority')
					if spec.priorities[prop] < priorities[prop]:
						properties[prop] = spec
						priorities[prop] = spec.properties[prop]
				else:
					# This property has not already been specified, so we should initialize it.
					properties[prop] = spec
					priorities[prop] = spec.priorities[prop]

		'''
		If a modifying specifier specifies the property with a higher priority,
		set the object's property to be specified by the modifying specifier. Otherwise,
		if the property exists and the priorities match, object needs to be specified
		by the original specifier then the resulting value is modified by the
		modifying specifier. 

		If the property is not yet being specified, the modifying specifier will 
		act as a normal specifier for that property. 
		'''
		deprecate = []
		for spec in modifying_specifiers:
			for prop in spec.priorities:
				# If it has already been modified, which also implies this property has already been specified.


				# Now we check if the propert has already been specified
				if prop in properties:
					# This property has already been specified, so we should either modify
					# it or specify it.

					if spec.priorities[prop] < priorities[prop]:
						# Higher priority level, so it specifies
						properties[prop] = spec
						priorities[prop] = spec.priorities[prop]
						deprecate.append(prop)
					elif prop in spec.modifiable_props:
						# This specifer can modify this prop, so we set it to do so after
						# first checking it has not already been modified.
						if prop in modifying:
							raise RuntimeParseError(f'property "{prop}" of {name} modified twice.')

						modifying[prop] = spec
				else:
					# This property has not been specified, so we should specify it.
					properties[prop] = spec
					priorities[prop] = spec.priorities[prop]
					deprecate.append(prop)

		# Add any default specifiers needed
		for prop, default_spec in defs.items():
			if prop not in priorities:
				specifiers.append(default_spec)
				properties[prop] = default_spec

		# Create the actual_props dictionary, which maps each specifier to a set of properties
		# it is actually specifying or modifying.
		actual_props = {spec: set() for spec in specifiers}
		for prop in properties:
			# Extract the specifier that is specifying this prop and add it to the
			# specifier's entry in actual_props
			specifying_spec = properties[prop]
			actual_props[specifying_spec].add(prop)

			# If a specifier modifies this property, add this prop to the specifiers
			# actual_props list.
			if prop in modifying:
				modifying_spec = modifying[prop]
				actual_props[modifying_spec].add(prop)

		# Create an inversed modifying dictionary that specifiers to the properties they
		# are modifying.
		modifying_inv = {spec:prop for prop, spec in modifying.items()}

		# Topologically sort specifiers. Specifiers become vertices and the properties
		# those specifiers depend on become the in-edges of each vertex. The specifiers
		# are then sorted topologically according to this graph.
		order = []
		seen, done = set(), set()

		def dfs(spec):
			if spec in done:
				return
			elif spec in seen:
				raise RuntimeParseError(f'specifier for property {spec.priorities} '
										'depends on itself')
			seen.add(spec)

			# Recurse on dependencies
			for dep in spec.requiredProperties:
				child = properties.get(dep)
				if child is None:
					raise RuntimeParseError(f'property {dep} required by '
											f'specifier {spec} is not specified')
				else:
					dfs(child)

			# If this is a modifying specifier, recurse on the specifier
			# that specifies the property being modified.
			if spec in modifying_inv:
				specifying_spec = properties[modifying_inv[spec]]
				dfs(specifying_spec)

			order.append(spec)
			done.add(spec)

		for spec in specifiers:
			dfs(spec)
		assert len(order) == len(specifiers)

		# Establish a boolean array tracking which properties will be modified.
		self._mod_tracker = {prop: True for prop in modifying}

		# Evaluate and apply specifiers, using actual_props to indicate which properties
		# it should actually specify.
		self.properties = set()		# will be filled by calls to _specify below
		self._evaluated = DefaultIdentityDict()		# temporary cache for lazily-evaluated values
		for spec in order:
			spec.applyTo(self, actual_props[spec])
		del self._evaluated

		# Check that all modifications have been applied and then delete tracker
		assert all(self._mod_tracker)
		del self._mod_tracker

		# Set up dependencies
		deps = []
		for prop in properties:
			assert hasattr(self, prop)
			val = getattr(self, prop)
			deps.append(val)
		super().__init__(deps)

		# Possibly register this object
		self._register()

	def _specify(self, prop, value):
		if prop in self.properties:
			# We have already specified this property. Check if we can modify it and otherwise
			# raise an assert.
			if prop not in self._mod_tracker:
				assert prop not in self.properties, ("Resetting (not modifying) " + str(prop))
			else:
				# We can modify this prop. Ensure it hasn't already been modified and then mark
				# it so it can't be modified down the line.
				assert self._mod_tracker[prop] == True
				self._mod_tracker[prop] = False

		# Normalize types of some built-in properties
		if prop == 'position':
			value = toVector(value, f'"position" of {self} not a vector')
		elif prop in ('yaw', 'pitch', 'roll'):
			value = toScalar(value, f'"{prop}" of {self} not a scalar')

		# Check if this property is already an attribute
		if hasattr(self, prop) and prop not in self.properties:
			raise RuntimeParseError(f"Property {prop} would overwrite an attribute with the same name.")

		self.properties.add(prop)
		object.__setattr__(self, prop, value)

	def _register(self):
		pass	# do nothing by default; may be overridden by subclasses

	def sampleGiven(self, value):
		if not needsSampling(self):
			return self
		return self.withProperties({ prop: value[getattr(self, prop)]
								   for prop in self.properties })

	def allProperties(self):
		return { prop: getattr(self, prop) for prop in self.properties }

	def copyWith(self, **overrides):
		props = self.allProperties()
		props.update(overrides)
		return self.withProperties(props)

	def isEquivalentTo(self, other):
		if type(other) is not type(self):
			return False
		return areEquivalent(self.allProperties(), other.allProperties())

	def __str__(self):
		if hasattr(self, 'properties') and 'name' in self.properties:
			return self.name
		else:
			return f'unnamed {self.__class__.__name__} ({id(self)})'

	def __repr__(self):
		if hasattr(self, 'properties'):
			allProps = { prop: getattr(self, prop) for prop in self.properties }
		else:
			allProps = '<under construction>'
		return f'{type(self).__name__}({allProps})'

## Mutators

class Mutator:
	"""An object controlling how the ``mutate`` statement affects an `Object`.

	A `Mutator` can be assigned to the ``mutator`` property of an `Object` to
	control the effect of the ``mutate`` statement. When mutation is enabled
	for such an object using that statement, the mutator's `appliedTo` method
	is called to compute a mutated version.
	"""

	def appliedTo(self, obj):
		"""Return a mutated copy of the object. Implemented by subclasses."""
		raise NotImplementedError

class PositionMutator(Mutator):
	"""Mutator adding Gaussian noise to ``position``. Used by `Point`.

	Attributes:
		stddev (float): standard deviation of noise
	"""
	def __init__(self, stddev):
		self.stddev = stddev

	def appliedTo(self, obj):
		noise = Vector(random.gauss(0, self.stddev), random.gauss(0, self.stddev))
		pos = obj.position + noise
		return (obj.copyWith(position=pos), True)		# allow further mutation

	def __eq__(self, other):
		if type(other) is not type(self):
			return NotImplemented
		return (other.stddev == self.stddev)

	def __hash__(self):
		return hash(self.stddev)

class HeadingMutator(Mutator):
	"""Mutator adding Gaussian noise to ``heading``. Used by `OrientedPoint`.

	Attributes:
		stddev (float): standard deviation of noise
	"""
	def __init__(self, stddev):
		self.stddev = stddev

	def appliedTo(self, obj):
		noise = random.gauss(0, self.stddev)
		h = obj.heading + noise
		return (obj.copyWith(heading=h), True)		# allow further mutation

	def __eq__(self, other):
		if type(other) is not type(self):
			return NotImplemented
		return (other.stddev == self.stddev)

	def __hash__(self):
		return hash(self.stddev)

## Point

class Point(_Constructible):
	"""Implementation of the Scenic base class ``Point``.

	The default mutator for `Point` adds Gaussian noise to ``position`` with
	a standard deviation given by the ``positionStdDev`` property.

	Properties:
		position (`Vector`; dynamic): Position of the point. Default value is the origin.
		visibleDistance (float): Distance for ``can see`` operator. Default value 50.
		width (float): Default value zero (only provided for compatibility with
		  operators that expect an `Object`).
		length (float): Default value zero.

	.. note::

		If you're looking into Scenic's internals, note that `Point` is actually a
		subclass of the internal Python class `_Constructible`.
	"""
	position: PropertyDefault((), {'dynamic'}, lambda self: Vector(0, 0, 0))
	width: 0
	length: 0
	visibleDistance: 50
	# Density of rays per degree in one dimension. Number of rays sent will be
	# this value squared per 1 degree x 1 degree portion of the visible region
	rayDensity: 30

	mutationEnabled: False
	mutator: PropertyDefault({'positionStdDev'}, {'additive'},
							 lambda self: PositionMutator(self.positionStdDev))
	positionStdDev: 1


	@cached_property
	def visibleRegion(self):
		dimensions = (self.visibleDistance, self.visibleDistance, self.visibleDistance)
		return SpheroidRegion(position=self.position, dimensions=dimensions)

	@cached_property
	def corners(self):
		return (self.position,)

	def toVector(self) -> Vector:
		return self.position

	def canSee(self, other, occludingObjects=list()) -> bool:
		return canSee(position=self.position, orientation=None, visibleDistance=self.visibleDistance, \
			viewAngle=(math.tau, math.tau), viewRays=self.viewRays, visibleRegion=self.visibleRegion, \
			target=other, occludingObjects=occludingObjects)

	def sampleGiven(self, value):
		sample = super().sampleGiven(value)
		if self.mutationEnabled:
			for mutator in self.mutator:
				if mutator is None:
					continue
				sample, proceed = mutator.appliedTo(sample)
				if not proceed:
					break
		return sample

	# Points automatically convert to Vectors when needed
	def __getattr__(self, attr):
		if hasattr(Vector, attr):
			return getattr(self.toVector(), attr)
		else:
			raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

## OrientedPoint

class OrientedPoint(Point):
	"""Implementation of the Scenic class ``OrientedPoint``.

	The default mutator for `OrientedPoint` adds Gaussian noise to ``heading``
	with a standard deviation given by the ``headingStdDev`` property, then
	applies the mutator for `Point`.

	Properties:
		heading (float; dynamic): Heading of the `OrientedPoint`. Default value 0
			(North).
		viewAngle (float): View cone angle for ``can see`` operator. Default
		  value :math:`2\\pi`.
	"""
	# primitive orientation properties
	yaw: PropertyDefault((), {'dynamic'}, lambda self: 0)
	pitch: PropertyDefault((), {'dynamic'}, lambda self: 0)
	roll: PropertyDefault((), {'dynamic'}, lambda self: 0)
	parentOrientation: Orientation.fromEuler(0, 0, 0)

	# derived orientation properties that cannot be overwritten
	orientation: PropertyDefault(
	    {'yaw', 'pitch', 'roll', 'parentOrientation'},
	    {'final'},
	    lambda self: (Orientation.fromEuler(self.yaw, self.pitch, self.roll)
	                  * self.parentOrientation)
	)
	heading: PropertyDefault({'orientation'}, {'final'},
	    lambda self: self.yaw if alwaysGlobalOrientation(self.parentOrientation) else self.orientation.yaw)

	# The view angle in the horizontal and vertical direction
	viewAngle: (math.tau, math.tau)

	mutator: PropertyDefault({'headingStdDev'}, {'additive'},
		lambda self: HeadingMutator(self.headingStdDev))
	headingStdDev: math.radians(5)

	@cached_property
	def visibleRegion(self):
		return DefaultViewRegion(visibleDistance=self.visibleDistance, viewAngle=self.viewAngle,\
			position=self.position, rotation=self.orientation)

	def relativize(self, vec):
		pos = self.relativePosition(vec)
		return OrientedPoint(position=pos, parentOrientation=self.orientation)

	def relativePosition(self, vec):
		return self.position.offsetRotated(self.orientation, vec)

	def toHeading(self) -> float:
		return self.heading

	def canSee(self, other, occludingObjects=list()) -> bool:
		return canSee(position=self.position, orientation=self.orientation, visibleDistance=self.visibleDistance,
			viewAngle=(math.tau, math.tau), viewRays=self.viewRays, visibleRegion=self.visibleRegion, \
			target=other, occludingObjects=occludingObjects)

## Object

class Object(OrientedPoint, _RotatedRectangle):
	"""Implementation of the Scenic class ``Object``.

	This is the default base class for Scenic classes.

	Properties:
		width (float): Width of the object, i.e. extent along its X axis.
		  Default value 1.
		length (float): Length of the object, i.e. extent along its Y axis.
		  Default value 1.
		height (float): Height of the object, i.e. extent along its Z axis.
		  Default value 1.
		allowCollisions (bool): Whether the object is allowed to intersect
		  other objects. Default value ``False``.
		requireVisible (bool): Whether the object is required to be visible
		  from the ``ego`` object. Default value ``True``.
		regionContainedIn (`Region` or ``None``): A `Region` the object is
		  required to be contained in. If ``None``, the object need only be
		  contained in the scenario's workspace.
		shape: A Shape object to be used internally to handle collision detection,
		  amongst other boolean operators for geometry.
		cameraOffset (`Vector`): Position of the camera for the ``can see``
		  operator, relative to the object's ``position``. Default ``0 @ 0``.

		speed (float; dynamic): Speed in dynamic simulations. Default value 0.
		velocity (`Vector`; *dynamic*): Velocity in dynamic simulations. Default value is
			the velocity determined by ``self.speed`` and ``self.orientation``.
		angularSpeed (float; *dynamic*): Angular speed in dynamic simulations. Default
			value 0.

		behavior: Behavior for dynamic agents, if any (see :ref:`dynamics`). Default
			value ``None``.
	"""
	width: PropertyDefault(('shape',), {}, lambda self: self.shape.dimensions[0])
	length: PropertyDefault(('shape',), {}, lambda self: self.shape.dimensions[1])
	height: PropertyDefault(('shape',), {}, lambda self: self.shape.dimensions[2])

	allowCollisions: False
	requireVisible: False
	regionContainedIn: None
	cameraOffset: Vector(0, 0, 0)
	# Whether or not this object can occlude other objects
	occluding: True

	shape: BoxShape()

	baseOffset: PropertyDefault(('height',), {}, lambda self: Vector(0, 0, -self.height/2))
	contactTolerance: 0.00001

	velocity: PropertyDefault(('speed', 'orientation'), {'dynamic'},
	                          lambda self: Vector(0, self.speed).rotatedBy(self.orientation))
	speed: PropertyDefault((), {'dynamic'}, lambda self: 0)
	angularSpeed: PropertyDefault((), {'dynamic'}, lambda self: 0)

	min_top_z: 0.4

	occupiedSpace: PropertyDefault(('shape', 'width', 'length', 'height', 'position', 'orientation'), \
		{'final'}, lambda self: MeshVolumeRegion(mesh=self.shape.mesh, \
			dimensions=(self.width, self.length, self.height), \
			position=self.position, rotation=self.orientation))

	boundingBox: PropertyDefault(('occupiedSpace',), {'final'},  \
		lambda self: lazyBoundingBox(self.occupiedSpace))

	topSurface: PropertyDefault(('occupiedSpace', 'min_top_z'), \
		{}, lambda self: defaultTopSurface(self.occupiedSpace, self.min_top_z))

	behavior: None
	lastActions: None

	def __new__(cls, *args, **kwargs):
		obj = super().__new__(cls)
		# The _dynamicProxy attribute stores a mutable copy of the object used during
		# simulations, intercepting all attribute accesses to the original object;
		# we set this attribute very early to prevent problems during unpickling.
		object.__setattr__(obj, '_dynamicProxy', obj)
		return obj

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.hw = hw = self.width / 2
		self.hl = hl = self.length / 2
		self.hh = hh = self.height / 2
		self.radius = hypot(hw, hl, hh)	# circumcircle; for collision detection
		self.inradius = lazyInradius(self.occupiedSpace, self.position)	# incircle; for collision detection

		self._relations = []

	def _specify(self, prop, value):
		# Normalize types of some built-in properties
		if prop == 'behavior':
			import scenic.syntax.veneer as veneer	# TODO improve?
			value = toType(value, veneer.Behavior,
			               f'"behavior" of {self} not a behavior')
		super()._specify(prop, value)

	def _register(self):
		import scenic.syntax.veneer as veneer	# TODO improve?
		veneer.registerObject(self)

	def __getattribute__(self, name):
		proxy = object.__getattribute__(self, '_dynamicProxy')
		return object.__getattribute__(proxy, name)

	def __setattr__(self, name, value):
		proxy = object.__getattribute__(self, '_dynamicProxy')
		object.__setattr__(proxy, name, value)

	def __delattr__(self, name):
		proxy = object.__getattribute__(self, '_dynamicProxy')
		object.__delattr__(proxy, name)

	def containsPoint(self, point):
		return self.occupiedSpace.containsPoint(point)

	def intersects(self, other):
		return self.occupiedSpace.intersects(other.occupiedSpace)

	@cached_property
	def left(self):
		return self.relativize(Vector(-self.hw, 0))

	@cached_property
	def right(self):
		return self.relativize(Vector(self.hw, 0))

	@cached_property
	def front(self):
		return self.relativize(Vector(0, self.hl))

	@cached_property
	def back(self):
		return self.relativize(Vector(0, -self.hl))

	@cached_property
	def frontLeft(self):
		return self.relativize(Vector(-self.hw, self.hl))

	@cached_property
	def frontRight(self):
		return self.relativize(Vector(self.hw, self.hl))

	@cached_property
	def backLeft(self):
		return self.relativize(Vector(-self.hw, -self.hl))

	@cached_property
	def backRight(self):
		return self.relativize(Vector(self.hw, -self.hl))

	@cached_property
	def top(self):
		return self.relativize(Vector(0, 0, self.hh))

	@cached_property
	def bottom(self):
		return self.relativize(Vector(0, 0, -self.hh))

	@cached_property
	def topFrontLeft(self):
		return self.relativize(Vector(-self.hw, self.hl, self.hh))

	@cached_property
	def topFrontRight(self):
		return self.relativize(Vector(self.hw, self.hl, self.hh))

	@cached_property
	def topBackLeft(self):
		return self.relativize(Vector(-self.hw, -self.hl, self.hh))

	@cached_property
	def topBackRight(self):
		return self.relativize(Vector(self.hw, -self.hl, self.hh))

	@cached_property
	def bottomFrontLeft(self):
		return self.relativize(Vector(-self.hw, self.hl, -self.hh))

	@cached_property
	def bottomFrontRight(self):
		return self.relativize(Vector(self.hw, self.hl, -self.hh))

	@cached_property
	def bottomBackLeft(self):
		return self.relativize(Vector(-self.hw, -self.hl, -self.hh))

	@cached_property
	def bottomBackRight(self):
		return self.relativize(Vector(self.hw, -self.hl, -self.hh))

	@cached_property
	def visibleRegion(self):
		true_position = self.position.offsetRotated(self.orientation, toVector(self.cameraOffset))
		return DefaultViewRegion(visibleDistance=self.visibleDistance, viewAngle=self.viewAngle,\
			position=true_position, rotation=self.orientation)

	def canSee(self, other, occludingObjects=list()) -> bool:
		true_position = self.position.offsetRotated(self.orientation, toVector(self.cameraOffset))
		return canSee(position=true_position, orientation=self.orientation, visibleDistance=self.visibleDistance, \
			viewAngle=self.viewAngle, viewRays=self.viewRays, visibleRegion=self.visibleRegion, \
			target=other, occludingObjects=occludingObjects)

	@cached_property
	def corners(self):
		hw, hl = self.hw, self.hl
		return (
			self.relativePosition(Vector(hw, hl)),
			self.relativePosition(Vector(-hw, hl)),
			self.relativePosition(Vector(-hw, -hl)),
			self.relativePosition(Vector(hw, -hl))
		)

	def show_3d(self, viewer, highlight=False):
		if needsSampling(self):
			raise RuntimeError('tried to show() symbolic Object')

		# Render the object
		object_mesh = self.occupiedSpace.mesh.copy()

		if highlight:
			object_mesh.visual.face_colors = [30, 179, 0, 255]

		viewer.add_geometry(object_mesh)

		# If the camera is not a sphere, render the visible pyramid as a blue wireframe
		if self.viewAngle != (math.tau, math.tau) or self.visibleDistance != 50:
			camera_pyramid_mesh = self.visibleRegion.mesh.copy()

			edges = camera_pyramid_mesh.face_adjacency_edges[camera_pyramid_mesh.face_adjacency_angles > np.radians(0.1)].copy()
			vertices = camera_pyramid_mesh.vertices.copy()

			edge_path = trimesh.path.Path3D(**trimesh.path.exchange.misc.edges_to_path(edges, vertices))

			edge_path.colors = [[30, 30, 150, 255] for _ in range(len(edge_path.entities))]

			viewer.add_geometry(edge_path)

	def show_2d(self, workspace, plt, highlight=False):
		if needsSampling(self):
			raise RuntimeError('tried to show() symbolic Object')
		pos = self.position
		spos = workspace.scenicToSchematicCoords(pos)

		if highlight:
			# Circle around object
			rad = 1.5 * max(self.width, self.length)
			c = plt.Circle(spos, rad, color='g', fill=False)
			plt.gca().add_artist(c)
			# View cone
			ha = self.viewAngle / 2.0
			camera = self.position.offsetRotated(self.heading, self.cameraOffset)
			cpos = workspace.scenicToSchematicCoords(camera)
			for angle in (-ha, ha):
				p = camera.offsetRadially(20, self.heading + angle)
				edge = [cpos, workspace.scenicToSchematicCoords(p)]
				x, y = zip(*edge)
				plt.plot(x, y, 'b:')

		corners = [workspace.scenicToSchematicCoords(corner) for corner in self.corners]
		x, y = zip(*corners)
		color = self.color if hasattr(self, 'color') else (1, 0, 0)
		plt.fill(x, y, color=color)

		frontMid = averageVectors(corners[0], corners[1])
		baseTriangle = [frontMid, corners[2], corners[3]]
		triangle = [averageVectors(p, spos, weight=0.5) for p in baseTriangle]
		x, y = zip(*triangle)
		plt.fill(x, y, "w")
		plt.plot(x + (x[0],), y + (y[0],), color="k", linewidth=1)

@distributionFunction
def lazyBoundingBox(occupiedSpace):
	return MeshVolumeRegion(occupiedSpace.mesh.bounding_box, center_mesh=False)

@distributionFunction
def lazyInradius(occupiedSpace, position):
	if not occupiedSpace.containsPoint(position):
		return 0

	pq = trimesh.proximity.ProximityQuery(occupiedSpace.mesh)
	dist = abs(pq.signed_distance([position])[0])

	return dist

@distributionFunction
def defaultTopSurface(occupiedSpace, min_top_z):
	# Extract mesh from object
	obj_mesh = occupiedSpace.mesh.copy()

	# Drop all faces whose normal vector do not have a sufficiently
	# large z component.
	face_mask = obj_mesh.face_normals[:, 2] >= min_top_z
	obj_mesh.faces = obj_mesh.faces[face_mask]
	obj_mesh.remove_unreferenced_vertices()

	# Check if the resulting surface is empty and return an appropriate region.
	if not obj_mesh.is_empty:
		return MeshSurfaceRegion(mesh=obj_mesh, center_mesh=False)
	else:
		return EmptyRegion(name="EmptyTopSurface")

@distributionFunction
def defaultEmptySpace(occupiedSpace):
	# Extract the bounding box mesh, center it around the origin, scale it down slightly, and
	# then move it back to it's original position.
	bb_mesh = occupiedSpace.mesh.bounding_box.copy()
	bb_pos = bb_mesh.center_mass
	bb_extents = bb_mesh.extents*.99

	# Take the difference of the objects bounding box and its mesh.
	bb_region = BoxRegion(position=Vector(*bb_pos), dimensions=bb_extents)
	empty_space_region = bb_region.difference(occupiedSpace)

	# If empty_space_region forms a volume, it is meaningful. Else set it to EmptyRegion.
	if isinstance(empty_space_region, MeshVolumeRegion):
		return empty_space_region
	else:
		return EmptyRegion(name="EmptyEmptySpace")

def canSee(position, orientation, visibleDistance, viewAngle, viewRays, \
		visibleRegion, target, occludingObjects):
	# First check if the target is visible even without occlusion.
	if not visibleRegion.intersects(target.occupiedSpace):
		return False

	# Now generate candidate rays to check for actual visibility
	if isinstance(target, (Region, Object)):
		# First extract the target region from the object or region.
		if isinstance(target, Region):
			target_region = target
		elif isinstance(target, (Object)):
			target_region = target.occupiedSpace

		# Generate candidate rays
		h_range = (-viewAngle[0]/2, viewAngle[0]/2)
		v_range = (-viewAngle[1]/2, viewAngle[1]/2)

		h_angles = np.linspace(h_range[0],h_range[1],math.ceil(viewRays[0]))
		v_angles = np.linspace(v_range[0],v_range[1],math.ceil(viewRays[1]))

		angle_matrix = np.transpose([np.tile(h_angles, len(v_angles)), np.repeat(v_angles, len(h_angles))])

		ray_vectors = np.zeros((len(angle_matrix[:,0]), 3))

		ray_vectors[:,0] = np.cos(-angle_matrix[:,0]+math.pi/2)*np.sin(-angle_matrix[:,1]+math.pi/2)
		ray_vectors[:,1] = np.sin(-angle_matrix[:,0]+math.pi/2)*np.sin(-angle_matrix[:,1]+math.pi/2)
		ray_vectors[:,2] = np.cos(-angle_matrix[:,1]+math.pi/2)

		ray_vectors = orientation.getRotation().apply(ray_vectors)

		# Check if candidate rays hit target
		raw_target_hit_info = target_region.mesh.ray.intersects_location(
			ray_origins=[position.coordinates for ray in ray_vectors],
			ray_directions=ray_vectors)

		# Extract rays that are within visibleDistance, mapping the vector
		# to the distance at which they hit the target
		feasible_vectors = {}

		for hit_iter in range(len(raw_target_hit_info[0])):
			hit_vector = Vector(*raw_target_hit_info[0][hit_iter])
			hit_distance = position.distanceTo(hit_vector)

			if hit_distance <= visibleDistance:
				feasible_vectors[tuple(ray_vectors[raw_target_hit_info[1][hit_iter],:])] = hit_distance

		if len(feasible_vectors) > 0:
			naively_visible = True
		else:
			naively_visible = False

		# vertices = [vec for vec in ray_vectors]
		# vertices = [position.coordinates] + vertices
		# lines = [trimesh.path.entities.Line([0,v]) for v in range(1,len(vertices))]
		# colors =[(255,0,0,255) for line in lines]

		# render_scene = trimesh.scene.Scene()
		# render_scene.add_geometry(trimesh.path.Path3D(entities=lines, vertices=vertices, process=False, colors=colors))
		# render_scene.add_geometry(list(occludingObjects)[0].occupiedSpace.mesh)
		# render_scene.show()

		# assert False

	elif isinstance(target, (Point, OrientedPoint)):
		raise NotImplementedError()
	else:
		raise NotImplementedError("Cannot check if " + str(target) + " of type " + type(target) + " can be seen.")

	if not naively_visible:
		return False
	else:
		return True

	# Now check if occluded objects block sight to target
	for occ_obj in occludingObjects:
		candidate_vectors = np.array(list(feasible_vectors.keys()))

		object_hit_info = occ_obj.occupiedSpace.mesh.ray.intersects_location(
			ray_origins=[position.coordinates for ray in candidate_vectors],
			ray_directions=candidate_vectors)

		print(object_hit_info)

	return len(feasible_vectors) > 0

def enableDynamicProxyFor(obj):
	object.__setattr__(obj, '_dynamicProxy', obj.copyWith())

def setDynamicProxyFor(obj, proxy):
	object.__setattr__(obj, '_dynamicProxy', proxy)

def disableDynamicProxyFor(obj):
	object.__setattr__(obj, '_dynamicProxy', obj)
