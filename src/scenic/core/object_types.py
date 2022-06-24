"""Implementations of the built-in Scenic classes."""

import collections
import math
import random
from abc import ABC, abstractmethod

from scenic.core.distributions import Samplable, needsSampling, distributionMethod
from scenic.core.specifiers import Specifier, PropertyDefault, ModifyingSpecifier
from scenic.core.vectors import Vector, Orientation, alwaysGlobalOrientation
from scenic.core.geometry import (_RotatedRectangle, averageVectors, hypot, min,
                                  pointIsInCone)
from scenic.core.regions import CircularRegion, SectorRegion, MeshVolumeRegion, MeshSurfaceRegion, DefaultTopSurface
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
		properties = dict()
		modified = dict()
		priorities = dict()
		optionals = collections.defaultdict(list)
		defs = self.__class__._defaults
		finals = self.__class__._finalProperties

		# TODO: @Matthew Check for incompatible specifiers used with modifying specifier (itself or `at`)

		'''
		For each specifier:
			* If a modifying specifier, modifying[p] = specifier
			* If a specifier, and not in properties specified, properties[p] = specifier
				- Otherwise, if property specified, check if specifier's priority is higher.
				- If so, replace it with specifier

		Priorties are inversed: A lower priority number means semantically that it has a higher priority level
		'''
		for spec in specifiers:
			assert isinstance(spec, Specifier), (name, spec)
			# Extract dictionary mapping from properties to priorities
			props = spec.priorities

			# Iterate over each property.
			for p in props:
				# Check if this is a final property has been specified. If so, throw an assertion or error,
				# depending on whether or not this object is internal.
				if p in finals:
					assert not _internal
					raise RuntimeParseError(f'property "{p}" of {name} cannot be directly specified')
				# Check if this is a modifying specifier, and if so if this property has already been modified.
				# If so throw an error, as no property can be modified twice. Otherwise, note that it is a
				# modifying specifier and deal with it later.
				if isinstance(spec, ModifyingSpecifier):
					if p in modified:
						raise RuntimeParseError(f'property "{p}" of {name} modified twice')
					modified[p] = spec
				else:
					# Otherwise we need to apply the changes this specifier makes to the property.
					if p in properties:
						# This property already exists. Check that it has not already been specified
						# at equal priority level. Then if it was previously specified at a lower priority
						# level, override it with the value that this specifier sets.
						if spec.priorities[p] == priorities[p]:
							raise RuntimeParseError(f'property "{p}" of {name} specified twice with the same priority')
						if spec.priorities[p] < priorities[p]:
							properties[p] = spec
							priorities[p] = spec.properties[p]
							spec.modifying[p] = False
					else:
						# This property does not already exist, so we should initialize it.
						properties[p] = spec
						priorities[p] = spec.priorities[p]
						spec.modifying[p] = False

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
		for prop, spec in modified.items():
			if prop in properties:
				if spec.priorities[prop] < priorities[prop]:   # Higher priority level, so it specifies
					properties[prop] = spec
					priorities[prop] = spec.priorities[prop]
					spec.modifying[prop] = False
					deprecate.append(prop)
			else:                                              # Not specified, so specify it
				properties[prop] = spec
				priorities[prop] = spec.priorities[prop]
				spec.modifying[prop] = False
				deprecate.append(prop)

		# Delete all deprecated modifiers. Any remaining will modify a specified property later.
		for d in deprecate:
			assert d in modified
			del modified[d]

		# Add any default specifiers needed
		for prop in defs:
			if prop not in properties:
				spec = defs[prop]
				specifiers.append(spec)
				properties[prop] = spec

		# Topologically sort specifiers
		order = []
		seen, done = set(), set()

		def dfs(spec):
			if spec in done:
				return
			elif spec in seen:
				raise RuntimeParseError(f'specifier for property {spec.priorities} '
										'depends on itself')
			seen.add(spec)
			for dep in spec.requiredProperties:
				child = properties.get(dep)
				if child is None:
					raise RuntimeParseError(f'property {dep} required by '
											f'specifier {spec} is not specified')
				else:
					dfs(child)
			order.append(spec)
			done.add(spec)

		for spec in specifiers:
			dfs(spec)
		assert len(order) == len(specifiers)

		# 
		for spec in specifiers:
			if isinstance(spec, ModifyingSpecifier):
				for mod in modified:
					spec.modifying[mod] = True

		# Evaluate and apply specifiers
		self.properties = set()		# will be filled by calls to _specify below
		self._evaluated = DefaultIdentityDict()		# temporary cache for lazily-evaluated values
		for spec in order:
			spec.applyTo(self, spec.modifying)
		del self._evaluated

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
		assert prop not in self.properties

		# Normalize types of some built-in properties
		if prop == 'position':
			value = toVector(value, f'"position" of {self} not a vector')
		elif prop in ('yaw', 'pitch', 'roll'):
			value = toScalar(value, f'"{prop}" of {self} not a scalar')

		self.properties.add(prop)
		object.__setattr__(self, prop, value)

	def _register(self):
		pass	# do nothing by default; may be overridden by subclasses

	def sampleGiven(self, value):
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

	mutationEnabled: False
	mutator: PropertyDefault({'positionStdDev'}, {'additive'},
							 lambda self: PositionMutator(self.positionStdDev))
	positionStdDev: 1

	@cached_property
	def visibleRegion(self):
		return CircularRegion(self.position, self.visibleDistance)

	@cached_property
	def corners(self):
		return (self.position,)

	def toVector(self) -> Vector:
		return self.position

	# TODO: @Matthew Does this work for 3D space?
	def canSee(self, other) -> bool:	# TODO improve approximation?
		for corner in other.corners:
			if self.visibleRegion.containsPoint(corner):
				return True
		return False

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

	viewAngle: math.tau # TODO: @Matthew Implement 2-tuple view angle for 3D views

	mutator: PropertyDefault({'headingStdDev'}, {'additive'},
		lambda self: HeadingMutator(self.headingStdDev))
	headingStdDev: math.radians(5)

	@cached_property
	def visibleRegion(self):
		return SectorRegion(self.position, self.visibleDistance,
		                    self.heading, self.viewAngle)

	def relativize(self, vec):
		pos = self.relativePosition(vec)
		return OrientedPoint(position=pos, parentOrientation=self.orientation)

	def relativePosition(self, vec):
		return self.position.offsetRotated(self.orientation, vec)

	def toHeading(self) -> float:
		return self.heading

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
	requireVisible: True
	regionContainedIn: None
	cameraOffset: Vector(0, 0)

	shape: BoxShape()
	centerOffset: PropertyDefault(('height',), {}, lambda self: Vector(0, 0, -self.height/2-0.00001))

	velocity: PropertyDefault(('speed', 'orientation'), {'dynamic'},
	                          lambda self: Vector(0, self.speed).rotatedBy(self.orientation))
	speed: PropertyDefault((), {'dynamic'}, lambda self: 0)
	angularSpeed: PropertyDefault((), {'dynamic'}, lambda self: 0)

	min_top_z: 0.4
	topSurface: PropertyDefault(('shape', 'min_top_z', 'width', 'length', 'height', 'position','orientation'), \
		{'dynamic'}, lambda self: DefaultTopSurface(self.shape.mesh, self.min_top_z, \
			dimensions=(self.width, self.length, self.height), position=self.position, rotation=self.orientation))

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
		self.inradius = 0 #min(hw, hl, hh)	# incircle; for collision detection

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
		return self.region.containsPoint(point)

	def intersects(self, other):
		other_region = other.region
		return self.region.intersects(other_region)

	@property
	def region(self):
		return MeshVolumeRegion(mesh=self.shape.mesh, dimensions=(self.width, self.length, self.height), position=self.position, rotation=self.orientation)

	# def _topSurface(self):
	# 	# Copy surface mesh, and remove all faces who's normal vectors
	# 	# have a z component less than self.min_top_z
	# 	surface_mesh = self.region.mesh.copy()
	# 	face_mask = surface_mesh.face_normals[:, 2] > self.min_top_z
	# 	surface_mesh.faces = surface_mesh.faces[face_mask]
	# 	surface_mesh.remove_unreferenced_vertices()

	# 	# Return MeshSurfaceRegion representing top surface of this object.
	# 	return MeshSurfaceRegion(mesh=self.shape.mesh, dimensions=(self.width, self.length, self.height), position=self.position, rotation=self.orientation)

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
		camera = self.position.offsetRotated(self.heading, self.cameraOffset)
		return SectorRegion(camera, self.visibleDistance, self.heading, self.viewAngle)

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

		viewer_mesh = self.region.mesh.copy()

		if highlight:
			viewer_mesh.visual.face_colors = [30, 179, 0, 255]

		viewer.add_geometry(viewer_mesh)

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


def enableDynamicProxyFor(obj):
	object.__setattr__(obj, '_dynamicProxy', obj.copyWith())

def setDynamicProxyFor(obj, proxy):
	object.__setattr__(obj, '_dynamicProxy', proxy)

def disableDynamicProxyFor(obj):
	object.__setattr__(obj, '_dynamicProxy', obj)
