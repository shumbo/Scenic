"""Implementations of the built-in Scenic classes.

Defines the 3 Scenic classes `Point`, `OrientedPoint`, and `Object`, and associated
helper code (notably their base class `Constructible`, which implements the handling of
property definitions and :ref:`specifier resolution`).
"""

import warnings
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
                                  pointIsInCone, normalizeAngle)
from scenic.core.regions import (Region, CircularRegion, SectorRegion, MeshVolumeRegion, MeshSurfaceRegion, 
								  BoxRegion, SpheroidRegion, DefaultViewRegion, EmptyRegion)
from scenic.core.type_support import toVector, toHeading, toType, toScalar
from scenic.core.lazy_eval import needsLazyEvaluation
from scenic.core.serialization import dumpAsScenicCode
from scenic.core.utils import DefaultIdentityDict, cached_property
from scenic.core.errors import RuntimeParseError
from scenic.core.shapes import Shape, BoxShape, MeshShape
from scenic.core.regions import IntersectionRegion

## Abstract base class

class Constructible(Samplable):
	"""Abstract base class for Scenic objects.

	Scenic objects, which are constructed using specifiers, are implemented
	internally as instances of ordinary Python classes. This abstract class
	implements the procedure to resolve specifiers and determine values for
	the properties of an object, as well as several common methods supported
	by objects.

	.. warning::

		This class is an implementation detail, and none of its methods should be
		called directly from a Scenic program.
	"""

	def __init_subclass__(cls):
		super().__init_subclass__()
		# find all defaults provided by the class or its superclasses
		allDefs = collections.defaultdict(list)

		for sc in cls.__mro__:
			if issubclass(sc, Constructible) and hasattr(sc, '_scenic_properties'):
				for prop, value in sc._scenic_properties.items():
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
		cls._dynamicProperties = frozenset(dyns)
		cls._finalProperties = tuple(finals)

	@classmethod
	def _withProperties(cls, props, constProps=frozenset()):
		assert all(reqProp in props for reqProp in cls._defaults)
		assert all(not needsLazyEvaluation(val) for val in props.values())
		return cls(_internal=True, _constProps=constProps, **props)

	def __init__(self, *args, _internal=False, _constProps=frozenset(), **kwargs):
		if _internal:	# Object is being constructed internally; use fast path
			assert not args
			for prop, value in kwargs.items():
				assert not needsLazyEvaluation(value), (prop, value)
				object.__setattr__(self, prop, value)
			super().__init__(kwargs.values())
			self.properties = set(kwargs.keys())
			self._constProps = _constProps
			return

		# Resolve and apply specifiers
		specifiers = list(args)
		for prop, val in kwargs.items():	# kwargs supported for internal use
			specifiers.append(Specifier("Internal(Kwargs)", {prop: 1}, {prop: val}))

		# Apply specifiers
		self._applySpecifiers(specifiers)

		# Set up dependencies
		deps = []
		for prop in self.properties:
			assert hasattr(self, prop)
			val = getattr(self, prop)
			deps.append(val)
		super().__init__(deps)

		# Possibly register this object
		self._register()

	def _applySpecifiers(self, specifiers, defs=None, overriding=False):
		# Declare properties dictionary which maps properties to the specifier
		# that will specify that property.
		properties = dict()

		# Declare modifying dictionary, which maps properties to a specifier
		# that will modify that property.
		modifying = dict()

		# Dictionary mapping properties set so far to the priority with which they have
		# been set.
		priorities = dict()

		# Extract default property values dictionary and set of final properties,
		# unless defs is overriden.
		if defs is None:
			defs = self.__class__._defaults

		finals = self.__class__._finalProperties

		# Check for incompatible specifier combinations
		specifier_names = [spec.name for spec in specifiers]

		if "On" in specifier_names:
			if "At" in specifier_names:
				raise RuntimeParseError(f'Cannot use "On" specifier to modify "At" specifier')

			if collections.Counter(specifier_names)["On"] > 1:
				raise RuntimeParseError(f'Cannot use "On" specifier to modify "On" specifier')

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
					raise RuntimeParseError(f'property "{prop}" cannot be directly specified')


				if prop in properties:
					# This property already exists. Check that it has not already been specified
					# at equal priority level. Then if it was previously specified at a lower priority
					# level, override it with the value that this specifier sets.
					if spec.priorities[prop] == priorities[prop]:
						raise RuntimeParseError(f'property "{prop}" specified twice with the same priority')
					if spec.priorities[prop] < priorities[prop]:
						properties[prop] = spec
						priorities[prop] = spec.priorities[prop]
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
		_defaultedProperties = set()
		for prop, default_spec in defs.items():
			if prop not in priorities:
				specifiers.append(default_spec)
				properties[prop] = default_spec
				_defaultedProperties.add(prop)

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
			spec.applyTo(self, actual_props[spec], overriding=overriding)
		del self._evaluated
		self._constProps = frozenset({
			prop for prop in _defaultedProperties
			if not needsSampling(getattr(self, prop))
		})

		# Check that all modifications have been applied and then delete tracker
		assert all(self._mod_tracker)
		del self._mod_tracker

	def _specify(self, prop, value, overriding=False):
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
		if prop in ('position', 'velocity', 'cameraOffset'):
			value = toVector(value, f'"{prop}" of {self} not a vector')
		elif prop in ('width', 'length', 'visibleDistance', 'positionStdDev',
		              'viewAngle', 'headingStdDev', 'speed', 'angularSpeed',
		              'yaw', 'pitch', 'roll'):
			value = toScalar(value, f'"{prop}" of {self} not a scalar')

		if prop in ['yaw', 'pitch', 'roll']:
			value = normalizeAngle(value)

		# Check if this property is already an attribute, unless we are overriding
		if hasattr(self, prop) and prop not in self.properties and not overriding:
			raise RuntimeParseError(f"Property {prop} would overwrite an attribute with the same name.")

		self.properties.add(prop)
		object.__setattr__(self, prop, value)

	def _register(self):
		pass	# do nothing by default; may be overridden by subclasses

	def _override(self, specifiers):
		assert not needsSampling(self)
		oldVals = {}
		for spec in specifiers:
			for prop in spec.priorities:
				if prop in self._dynamicProperties:
					raise RuntimeParseError(f'cannot override dynamic property "{prop}"')
				if prop not in self.properties:
					raise RuntimeParseError(f'object has no property "{prop}" to override')
				oldVals[prop] = getattr(self, prop)
		defs = { prop: Specifier("OverrideDefault", {prop: -1}, {prop: getattr(self, prop)}) for prop in self.properties }
		self._applySpecifiers(list(specifiers), defs=defs, overriding=True)
		return oldVals

	def _revert(self, oldVals):
		for prop, val in oldVals.items():
			object.__setattr__(self, prop, val)

	def sampleGiven(self, value):
		if not needsSampling(self):
			return self
		return self._withProperties({ prop: value[getattr(self, prop)]
								    for prop in self.properties },
								    constProps=self._constProps)

	def _allProperties(self):
		return { prop: getattr(self, prop) for prop in self.properties }

	def _copyWith(self, **overrides):
		"""Copy this object, possibly overriding some of its properties."""
		props = self._allProperties()
		props.update(overrides)
		constProps = self._constProps.difference(overrides)
		return self._withProperties(props, constProps=constProps)

	def dumpAsScenicCode(self, stream, skipConstProperties=True):
		stream.write(f"new {self.__class__.__name__}")
		first = True
		for prop in sorted(self.properties):
			if skipConstProperties and prop in self._constProps:
				continue
			if prop == 'position':
				spec = 'at'
			elif prop == 'heading':
				spec = 'facing'
			else:
				spec = f'with {prop}'
			if first:
				stream.write(' ')
				first = False
			else:
				stream.write(',\n    ')
			stream.write(f'{spec} ')
			dumpAsScenicCode(getattr(self, prop), stream)

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
	"""An object controlling how the :keyword:`mutate` statement affects an `Object`.

	A `Mutator` can be assigned to the ``mutator`` property of an `Object` to
	control the effect of the :keyword:`mutate` statement. When mutation is enabled
	for such an object using that statement, the mutator's `appliedTo` method
	is called to compute a mutated version. The `appliedTo` method can also decide
	whether to apply mutators inherited from superclasses.
	"""

	def appliedTo(self, obj):
		"""Return a mutated copy of the given object. Implemented by subclasses.

		The mutator may inspect the ``mutationScale`` attribute of the given object
		to scale its effect according to the scale given in ``mutate O by S``.

		Returns:
			A pair consisting of the mutated copy of the object (which is most easily
			created using `_copyWith`) together with a Boolean indicating whether the
			mutator inherited from the superclass (if any) should also be applied.
		"""
		raise NotImplementedError

class PositionMutator(Mutator):
	"""Mutator adding Gaussian noise to ``position``. Used by `Point`.

	Attributes:
		stddev (float): standard deviation of noise
	"""
	def __init__(self, stddev):
		self.stddev = stddev

	def appliedTo(self, obj):
		stddev = self.stddev * obj.mutationScale
		noise = Vector(random.gauss(0, stddev), random.gauss(0, stddev))
		pos = obj.position + noise
		return (obj._copyWith(position=pos), True)		# allow further mutation

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
		noise = random.gauss(0, obj.mutationScale * self.stddev)
		h = obj.heading + noise
		return (obj._copyWith(heading=h), True)		# allow further mutation

	def __eq__(self, other):
		if type(other) is not type(self):
			return NotImplemented
		return (other.stddev == self.stddev)

	def __hash__(self):
		return hash(self.stddev)

## Point

class Point(Constructible):
	"""The Scenic base class ``Point``.

	The default mutator for `Point` adds Gaussian noise to ``position`` with
	a standard deviation given by the ``positionStdDev`` property.

	Properties:
		position (`Vector`; dynamic): Position of the point. Default value is the origin.
		visibleDistance (float): Distance for ``can see`` operator. Default value 50.
		width (float): Default value zero (only provided for compatibility with
		  operators that expect an `Object`).
		length (float): Default value zero.
		mutationScale (float): Overall scale of mutations, as set by the
		  :keyword:`mutate` statement. Default value zero (mutations disabled).
		positionStdDev (float): Standard deviation of Gaussian noise to add to this
		  object's ``position`` when mutation is enabled with scale 1. Default value 1.
	"""
	_scenic_properties = {
		"position": PropertyDefault((), {'dynamic'}, lambda self: Vector(0, 0)),
		"width": 0,
		"length": 0,
		"baseOffset": Vector(0,0,0),
		"contactTolerance": 0,
		"visibleDistance": 50,
		# Density of rays per degree in one dimension. Number of rays sent will be
		# this value squared per 1 degree x 1 degree portion of the visible region
		"rayDensity": 1,

		"mutationScale": 0,
		"mutator": PropertyDefault({'positionStdDev'}, {'additive'},
								lambda self: PositionMutator(self.positionStdDev)),
		"positionStdDev": 1,

		# This property is defined in Object, but we provide a default empty value
		# for Points for implementation convenience.
		"regionContainedIn": None,
		# These properties are used internally to store entities that must be able to
		# or must be unable to observe this entity.
		"observingEntity": None,
		"nonObservingEntity": None,
	}

	@cached_property
	def visibleRegion(self):
		"""The :term:`visible region` of this object.

		The visible region of a `Point` is a sphere centered at its ``position`` with
		radius ``visibleDistance``.
		"""
		dimensions = (self.visibleDistance, self.visibleDistance, self.visibleDistance)
		return SpheroidRegion(position=self.position, dimensions=dimensions)

	@cached_property
	def corners(self):
		return (self.position,)

	def toVector(self) -> Vector:
		return self.position

	def canSee(self, other, occludingObjects=list()) -> bool:
		return canSee(position=self.position, orientation=None, visibleDistance=self.visibleDistance, \
			viewAngles=(math.tau, math.pi), rayDensity=self.rayDensity, visibleRegion=self.visibleRegion, \
			target=other, occludingObjects=occludingObjects)

	def sampleGiven(self, value):
		sample = super().sampleGiven(value)
		if self.mutationScale != 0:
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
	"""The Scenic class ``OrientedPoint``.

	The default mutator for `OrientedPoint` adds Gaussian noise to ``heading``
	with a standard deviation given by the ``headingStdDev`` property, then
	applies the mutator for `Point`.

	Properties:
		heading (float; dynamic): Heading of the `OrientedPoint`. Default value 0
			(North).
		viewAngle (float): View cone angle for ``can see`` operator. Default
		  value 2π.
		headingStdDev (float): Standard deviation of Gaussian noise to add to this
		  object's ``heading`` when mutation is enabled with scale 1. Default value 5°.
	"""
	_scenic_properties = {
		# primitive orientation properties
		'yaw': PropertyDefault((), {'dynamic'}, lambda self: 0),
		'pitch': PropertyDefault((), {'dynamic'}, lambda self: 0),
		'roll': PropertyDefault((), {'dynamic'}, lambda self: 0),
		'parentOrientation': Orientation.fromEuler(0, 0, 0),

		# derived orientation properties that cannot be overwritten
		'orientation': PropertyDefault(
		    {'yaw', 'pitch', 'roll', 'parentOrientation'},
		    {'final'},
		    lambda self: (Orientation.fromEuler(self.yaw, self.pitch, self.roll)
			          * self.parentOrientation)
		),
		'heading': PropertyDefault({'orientation'}, {'final'},
		    lambda self: self.yaw if alwaysGlobalOrientation(self.parentOrientation) else self.orientation.yaw),

		# The view angle in the horizontal and vertical direction
		'viewAngle': math.tau,
		'viewAngles': PropertyDefault(('viewAngle',), set(), lambda self: (self.viewAngle, math.pi)),

		'mutator': PropertyDefault({'headingStdDev'}, {'additive'},
			lambda self: HeadingMutator(self.headingStdDev)),
		'headingStdDev': math.radians(5),
	}

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		if self.viewAngles[0] > math.tau or self.viewAngles[1] > math.pi:
			warnings.warn("ViewAngles can not have values greater than (math.tau, math.pi). Truncating values...")
			self.viewAngles = (min(self.viewAngles[0], math.tau), min(self.viewAngles[1], math.pi))

	@cached_property
	def visibleRegion(self):
		return DefaultViewRegion(visibleDistance=self.visibleDistance, viewAngles=self.viewAngles,\
			position=self.position, rotation=self.orientation)

	def relativize(self, vec):
		pos = self.relativePosition(vec)
		return OrientedPoint(position=pos, parentOrientation=self.orientation)

	def relativePosition(self, vec):
		return self.position.offsetRotated(self.orientation, vec)

	def distancePast(self, vec):
		"""Distance past a given point, assuming we've been moving in a straight line."""
		diff = self.position - vec
		return diff.rotatedBy(-self.heading).y

	def toHeading(self) -> float:
		return self.heading

	def toOrientation(self) -> Orientation:
		return self.orientation

	def canSee(self, other, occludingObjects=list()) -> bool:
		return canSee(position=self.position, orientation=self.orientation, visibleDistance=self.visibleDistance,
			viewAngles=self.viewAngles, rayDensity=self.rayDensity, visibleRegion=self.visibleRegion, \
			target=other, occludingObjects=occludingObjects)

## Object

class Object(OrientedPoint, _RotatedRectangle):
	"""The Scenic class ``Object``.

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
		  operator, relative to the object's ``position``. Default ``(0, 0)``.

		speed (float; dynamic): Speed in dynamic simulations. Default value 0.
		velocity (`Vector`; *dynamic*): Velocity in dynamic simulations. Default value is
			the velocity determined by ``self.speed`` and ``self.orientation``.
		angularSpeed (float; *dynamic*): Angular speed in dynamic simulations. Default
			value 0.

		behavior: Behavior for dynamic agents, if any (see :ref:`dynamics`). Default
			value ``None``.
	"""
	_scenic_properties = {
		'width': PropertyDefault(('shape',), {}, lambda self: self.shape.dimensions[0]),
		'length': PropertyDefault(('shape',), {}, lambda self: self.shape.dimensions[1]),
		'height': PropertyDefault(('shape',), {}, lambda self: self.shape.dimensions[2]),
		'allowCollisions': False,
		'requireVisible': False,
		'regionContainedIn': None,
		'cameraOffset': Vector(0, 0, 0),
		# Whether or not this object can occlude other objects
		'occluding': True,
		'shape': BoxShape(),
		'baseOffset': PropertyDefault(('height',), {}, lambda self: Vector(0, 0, -self.height/2)),
		'contactTolerance': 1e-4,
		'onDirection': Vector(0,0,1),
		'velocity': PropertyDefault((), {'dynamic'},
			                  lambda self: Vector(0, 0, 0)),
		'speed': PropertyDefault((), {'dynamic'}, lambda self: 0),
		'angularVelocity': PropertyDefault((), {'dynamic'}, lambda self: Vector(0, 0, 0)),
		'angularSpeed': PropertyDefault((), {'dynamic'}, lambda self: 0),
		'min_top_z': 0.4,
		'occupiedSpace': PropertyDefault(('shape', 'width', 'length', 'height', 'position', 'orientation', 'onDirection'), \
			{'final'}, lambda self: MeshVolumeRegion(mesh=self.shape.mesh, \
				dimensions=(self.width, self.length, self.height), \
				position=self.position, rotation=self.orientation, on_direction=self.onDirection)),
		'boundingBox': PropertyDefault(('occupiedSpace',), {'final'},  \
			lambda self: lazyBoundingBox(self.occupiedSpace)),
		'topSurface': PropertyDefault(('occupiedSpace', 'min_top_z'), \
			{}, lambda self: defaultTopSurface(self.occupiedSpace, self.min_top_z)),
		"behavior": None,
		"lastActions": None,
	}

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

	def _specify(self, prop, value, overriding=False):
		# Normalize types of some built-in properties
		if prop == 'behavior':
			import scenic.syntax.veneer as veneer	# TODO improve?
			value = toType(value, veneer.Behavior,
			               f'"behavior" of {self} not a behavior')
		super()._specify(prop, value, overriding=overriding)

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

	def startDynamicSimulation(self):
		"""Hook called at the beginning of each dynamic simulation.

		Does nothing by default; provided for objects to do simulator-specific
		initialization as needed.
		"""
		pass

	def containsPoint(self, point):
		return self.occupiedSpace.containsPoint(point)

	def distanceTo(self, other):
		return self.occupiedSpace.distanceTo(other)

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
		return DefaultViewRegion(visibleDistance=self.visibleDistance, viewAngles=self.viewAngles,\
			position=true_position, rotation=self.orientation)

	def canSee(self, other, occludingObjects=list()) -> bool:
		true_position = self.position.offsetRotated(self.orientation, toVector(self.cameraOffset))
		return canSee(position=true_position, orientation=self.orientation, visibleDistance=self.visibleDistance, \
			viewAngles=self.viewAngles, rayDensity=self.rayDensity, visibleRegion=self.visibleRegion, \
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
		if self.viewAngles != (math.tau, math.pi) or self.visibleDistance != 50:
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

def canSee(position, orientation, visibleDistance, viewAngles, rayDensity, \
		visibleRegion, target, occludingObjects):
	"""
	Perform visibility checks using ray tracing.

	For visibilty of Objects/Regions:
		1. Check if the region crosses the back and/or front of the viewing
		object. 
		2. Compute the spherical coordinates of all vertices in the mesh of
		the region we are trying to view, with the goal of using this to
		send rays only where they have a chance of hitting the region.
		3. Compute 2 ranges of angles (horizontal/vertical) in which rays have
		a chance of hitting the object.
			- If the object does not cross behind the object, take the min and
			max of the the spherical coordinate angles, while noting that this range
			is centered on the front of the object.
			- If the object crosses behind the object but not in front, transform 
			the spherical angles so they are coming from the back of the object,
			while noting that this range is centered on the back of the object.
			- If it crosses both, we do not optimize the amount of rays sent.
		4. Compute the intersection of the optimization range from step 3 and
		the viewAngles range, accounting for where the optimization range is centered. 
		If it is empty, the object is not visible. If it is not empty, shoot rays at 
		the desired density in the intersection region. Keep all rays that intersect
		the object (candidate rays).
		5. If there are no candidate rays, the object is not visible.
		6. For each object in occludingObjects:
			Check if any candidate rays interesect the object at a distance less than
			the distance they intersected the target region. If they do, remove
			them from the candidate rays.
		7. If any candidate rays remain, the object is visible. If not, it is occluded.
	"""
	if isinstance(target, (Region, Object)):
		# Extract the target region from the object or region.
		if isinstance(target, Region):
			raise NotImplementedError()
		elif isinstance(target, (Object)):
			target_region = target.occupiedSpace

		# Check that the distance to the target is not greater than visibleDistance, and
		# see if position is in the target.
		if target.containsPoint(position):
			return True

		if target.distanceTo(position) > visibleDistance:
			return False

		# Orient the object so that it has the same relative position and orientation to the
		# origin as it did to the viewer
		target_vertices = target_region.mesh.vertices - np.array(position.coordinates)
		if orientation is not None:
			target_vertices = orientation.invertRotation().getRotation().apply(target_vertices)

		## Check if the object crosses the y axis ahead and/or behind the viewer

		# Extract the two vectors that are part of each edge crossing the y axis.
		vec_0s = target_vertices[target_region.mesh.edges[:,0],:]
		vec_1s = target_vertices[target_region.mesh.edges[:,1],:]
		y_cross_edges = (vec_0s[:,0]/vec_1s[:,0]) < 0
		vec_0s = vec_0s[y_cross_edges]
		vec_1s = vec_1s[y_cross_edges]

		# Figure out for which t value the vectors cross the y axis
		t = (-vec_0s[:,0])/(vec_1s[:,0]-vec_0s[:,0])

		# Figure out what the y value is when the y axis is crossed
		y_intercept_points = t*(vec_1s[:,1]-vec_0s[:,1]) + vec_0s[:,1]

		# If the object crosses ahead and behind the object, or through 0,
		# we will not optimize ray casting.
		target_crosses_ahead = np.any(y_intercept_points >= 0)
		target_crosses_behind = np.any(y_intercept_points <= 0)

		## Compute the horizontal/vertical angle ranges which bound the object
		## (from the origin facing forwards)
		spherical_angles = np.zeros((len(target_vertices[:,0]), 2))

		spherical_angles[:,0] = np.arctan2(target_vertices[:,1], target_vertices[:,0])
		spherical_angles[:,1] = np.arcsin(target_vertices[:,2]/ \
								(np.linalg.norm(target_vertices, axis=1)))

		# Change from polar based coords to axis based coords
		spherical_angles[:,0] = spherical_angles[:,0] - math.pi/2

		# Normalize angles between (-Pi,Pi)
		spherical_angles[:,0] = np.mod(spherical_angles[:,0] + np.pi, 2*np.pi) - np.pi
		spherical_angles[:,1] = np.mod(spherical_angles[:,1] + np.pi, 2*np.pi) - np.pi

		# First we check if the vertical angles overlap with the vertical view angles.
		# If not, then the object cannot be visible.
		if np.min(spherical_angles[:,1]) > viewAngles[1]/2 or \
		   np.max(spherical_angles[:,1]) < -viewAngles[1]/2:
			return False 

		## Compute which horizontal/vertical angle ranges to cast rays in
		if target_crosses_ahead and target_crosses_behind:
			# No optimizations feasible here. Just send all rays.
			h_range = (-viewAngles[0]/2, viewAngles[0]/2)
			v_range = (-viewAngles[1]/2, viewAngles[1]/2)

			view_ranges = [(h_range, v_range)]

		elif target_crosses_behind:
			# We can keep the view angles oriented around the front of the object and
			# consider the spherical angles oriented around the back of the object.
			# We can then check for impossible visibility/optimize which rays will be cast.

			# Extract the viewAngle ranges
			va_h_range = (-viewAngles[0]/2, viewAngles[0]/2)
			va_v_range = (-viewAngles[1]/2, viewAngles[1]/2)

			# Convert spherical angles to be centered around the back of the viewing object.
			left_points  = spherical_angles[:,0] >= 0
			right_points = spherical_angles[:,0] < 0

			spherical_angles[:,0][left_points]  = spherical_angles[:,0][left_points] - np.pi
			spherical_angles[:,0][right_points] = spherical_angles[:,0][right_points] + np.pi

			sphere_h_range = (np.min(spherical_angles[:,0]), np.max(spherical_angles[:,0]))
			sphere_v_range = (np.min(spherical_angles[:,1]), np.max(spherical_angles[:,1]))

			# Extract the overlapping ranges in the horizontal and vertical view angles.
			# Note that the spherical range must cross the back plane and the view angles 
			# must cross the front plane (and are centered on these points),
			# which means we can just add up each side of the ranges and see if they add up to
			# greater than or equal to Pi. If none do, then it's impossible for object to overlap
			# with the viewAngle range.

			# Otherwise we can extract the overlapping v_ranges and use those going forwards.
			overlapping_v_range = (np.clip(sphere_v_range[0], va_v_range[0], va_v_range[1]),
								   np.clip(sphere_v_range[1], va_v_range[0], va_v_range[1]))
			view_ranges = []

			if (abs(va_h_range[0]) + abs(sphere_h_range[1]) > math.pi):
				h_range = (va_h_range[0], -math.pi+sphere_h_range[1])
				view_ranges.append((h_range, overlapping_v_range))

			if (abs(va_h_range[1]) + abs(sphere_h_range[0]) > math.pi):
				h_range = (math.pi+sphere_h_range[0], va_h_range[1])
				view_ranges.append((h_range, overlapping_v_range))

			if len(view_ranges) == 0:
				return False

		else:
			# We can immediately check for impossible visbility/optimize which rays
			# will be cast.

			# Check if view range and spherical angles overlap in horizontal or
			# vertical dimensions. If not, return False
			if 	(np.max(spherical_angles[:,0]) < -viewAngles[0]/2) or \
				(np.min(spherical_angles[:,0]) >  viewAngles[0]/2) or \
				(np.max(spherical_angles[:,1]) < -viewAngles[1]/2) or \
				(np.min(spherical_angles[:,1]) >  viewAngles[1]/2):
				return False

			# Compute trimmed view angles
			h_min = np.clip(np.min(spherical_angles[:,0]), -viewAngles[0]/2, viewAngles[0]/2)
			h_max = np.clip(np.max(spherical_angles[:,0]), -viewAngles[0]/2, viewAngles[0]/2)
			v_min = np.clip(np.min(spherical_angles[:,1]), -viewAngles[1]/2, viewAngles[1]/2)
			v_max = np.clip(np.max(spherical_angles[:,1]), -viewAngles[1]/2, viewAngles[1]/2)

			h_range = (h_min, h_max)
			v_range = (v_min, v_max)

			view_ranges = [(h_range, v_range)]

		## Generate candidate rays
		candidate_ray_list = []

		for h_range, v_range in view_ranges:
			h_size = h_range[1] - h_range[0]
			v_size = v_range[1] - v_range[0]

			assert h_size > 0
			assert v_size > 0

			# TODO This gives a non-uniform ray density. We could scale the number of rays in a single 
			# row by the cosine of the altitude angle to fix that.
			h_angles = np.linspace(h_range[0],h_range[1],math.ceil(math.degrees(h_size)*rayDensity))
			v_angles = np.linspace(v_range[0],v_range[1],math.ceil(math.degrees(v_size)*rayDensity))

			angle_matrix = np.transpose([np.tile(h_angles, len(v_angles)), np.repeat(v_angles, len(h_angles))])

			ray_vectors = np.zeros((len(angle_matrix[:,0]), 3))

			ray_vectors[:,0] = -np.sin(angle_matrix[:,0])
			ray_vectors[:,1] = np.cos(angle_matrix[:,0])
			ray_vectors[:,2] = np.tan(angle_matrix[:,1])

			ray_vectors /= np.linalg.norm(ray_vectors, axis=1)[:, np.newaxis]
			candidate_ray_list.append(ray_vectors)

		ray_vectors = np.concatenate(candidate_ray_list, axis=0)

		if orientation is not None:
			ray_vectors = orientation.getRotation().apply(ray_vectors)
		
		## DEBUG ##
		#Show all original candidate rays
		
		# vertices = [visibleDistance*vec + position.coordinates for vec in ray_vectors]
		# vertices = [position.coordinates] + vertices
		# lines = [trimesh.path.entities.Line([0,v]) for v in range(1,len(vertices))]
		# colors =[(0,0,255,255) for line in lines]

		# render_scene = trimesh.scene.Scene()
		# render_scene.add_geometry(trimesh.path.Path3D(entities=lines, vertices=vertices, process=False, colors=colors))
		# render_scene.add_geometry(target.occupiedSpace.mesh)
		# render_scene.add_geometry(list(occludingObjects)[0].occupiedSpace.mesh)
		# render_scene.show()

		# Check if candidate rays hit target
		raw_target_hit_info = target_region.mesh.ray.intersects_location(
			ray_origins=[position.coordinates for ray in ray_vectors],
			ray_directions=ray_vectors)


		# If no hits, this object can't be visible
		if len(raw_target_hit_info[0]) == 0:
			return False

		# Extract rays that are within visibleDistance, mapping the vector
		# to the closest distance at which they hit the target
		hit_locs = raw_target_hit_info[0]
		hit_distances = np.linalg.norm(hit_locs - np.array(position), axis=1)

		target_dist_map = {}

		for hit_iter in range(len(raw_target_hit_info[0])):
			hit_ray = tuple(ray_vectors[raw_target_hit_info[1][hit_iter]])
			hit_dist = hit_distances[hit_iter]

			# If the hit was out of visible distance, don't consider it.
			if hit_dist > visibleDistance:
				continue

			# If we don't already have a hit distance for this vector, or if
			# this hit was closer, update the target distance mapping.
			if hit_ray not in target_dist_map or hit_dist < target_dist_map[hit_ray]:
				target_dist_map[hit_ray] = hit_dist

		if len(target_dist_map) == 0:
			return False

		## DEBUG ##
		#Show all candidate vertices that hit target
		
		# vertices = [visibleDistance*np.array(vec) + position.coordinates for vec in candidate_rays]
		# vertices = [position.coordinates] + vertices
		# lines = [trimesh.path.entities.Line([0,v]) for v in range(1,len(vertices))]
		# colors =[(0,0,255,255) for line in lines]

		# render_scene = trimesh.scene.Scene()
		# render_scene.add_geometry(trimesh.path.Path3D(entities=lines, vertices=vertices, process=False, colors=colors))
		# render_scene.add_geometry(target.occupiedSpace.mesh)
		# render_scene.add_geometry(list(occludingObjects)[0].occupiedSpace.mesh)
		# render_scene.show()

		# Now check if occluded objects block sight to target

		candidate_rays = set(target_dist_map.keys())

		for occ_obj in occludingObjects:
			# If no more rays are candidates, then object is no longer visible.
			if len(candidate_rays) == 0:
				continue

			candidate_ray_list = np.array(list(candidate_rays))

			# Test all candidate rays against this occluding object
			object_hit_info = occ_obj.occupiedSpace.mesh.ray.intersects_location(
				ray_origins=[position.coordinates for ray in candidate_ray_list],
				ray_directions=candidate_ray_list)

			# If no hits, this object doesn't occlude.
			if len(object_hit_info[0]) == 0:
				continue

			# Check if any candidate ray hits the occluding object with a smaller
			# distance than the target.
			object_distances = np.linalg.norm(object_hit_info[0] - np.array(position), axis=1)

			occluded_rays = set()

			for hit_iter in range(len(object_hit_info[0])):
				hit_ray = tuple(candidate_ray_list[object_hit_info[1][hit_iter]])
				hit_dist = object_distances[hit_iter]

				# If this ray hit the object earlier than it hit the target, reject the ray.
				if hit_dist <= target_dist_map[hit_ray]:
					occluded_rays.add(hit_ray)

			candidate_rays = candidate_rays - occluded_rays

			## DEBUG ##
			# Show occluded and non occluded rays from this object
			
			# occluded_vertices = [visibleDistance*np.array(vec) + position.coordinates for vec in occluded_rays]
			# clear_vertices = [visibleDistance*np.array(vec) + position.coordinates for vec in candidate_rays]
			# vertices = occluded_vertices + clear_vertices
			# vertices = [position.coordinates] + vertices
			# lines = [trimesh.path.entities.Line([0,v]) for v in range(1,len(vertices))]
			# occluded_colors = [(255,0,0,255) for line in occluded_vertices]
			# clear_colors = [(0,255,0,255) for line in clear_vertices]
			# colors = occluded_colors + clear_colors
			# render_scene = trimesh.scene.Scene()
			# render_scene.add_geometry(trimesh.path.Path3D(entities=lines, vertices=vertices, process=False, colors=colors))
			# render_scene.add_geometry(target.occupiedSpace.mesh)
			# render_scene.add_geometry(list(occludingObjects)[0].occupiedSpace.mesh)
			# render_scene.show()


		return len(candidate_rays) > 0

	elif isinstance(target, (Point, OrientedPoint, Vector)):
		if isinstance(target, (Point, OrientedPoint)):
			target_loc = target.position
		else:
			target_loc = target

		# First check if the distance to the point is less than or equal to the visible distance. If not, the object cannot
		# be visible.
		target_distance = position.distanceTo(target_loc)
		if target_distance > visibleDistance:
			return False

		# Create the single candidate ray and check that it's within viewAngles.
		target_vertex = target_loc - position
		if orientation is not None:
			target_vertex = orientation.invertRotation().getRotation().apply(target_vertex)	
		candidate_ray = target_vertex/np.linalg.norm(target_vertex)

		azimuth = np.mod(np.arctan2(candidate_ray[1], candidate_ray[0]) - math.pi/2 + np.pi, 2*np.pi) - np.pi
		altitude = np.arcsin(candidate_ray[2]/(np.linalg.norm(candidate_ray)))

		# Check if this ray is within our view cone.
		if (not (-viewAngles[0]/2 <= azimuth <= viewAngles[0]/2)) or \
			(not (-viewAngles[1]/2 <= altitude <= viewAngles[1]/2)):
			return False

		candidate_ray_list = np.array([candidate_ray])

		# Now check if occluding objects block sight to target
		for occ_obj in occludingObjects:
			# Test all candidate rays against this occluding object
			object_hit_info = occ_obj.occupiedSpace.mesh.ray.intersects_location(
				ray_origins=[position.coordinates for ray in candidate_ray_list],
				ray_directions=candidate_ray_list)

			# Check the candidate ray hits the occluding object with a smaller
			# distance than the target.
			occluded_rays = set()

			for hit_iter in range(len(object_hit_info[0])):
				ray = tuple(candidate_ray_list[object_hit_info[1][hit_iter]])
				occ_distance = position.distanceTo(Vector(*object_hit_info[0][hit_iter,:]))

				if occ_distance <= target_distance:
					# The ray is occluded
					return False

		return True
	else:
		raise NotImplementedError("Cannot check if " + str(target) + " of type " + type(target) + " can be seen.")

def enableDynamicProxyFor(obj):
	object.__setattr__(obj, '_dynamicProxy', obj._copyWith())

def setDynamicProxyFor(obj, proxy):
	object.__setattr__(obj, '_dynamicProxy', proxy)

def disableDynamicProxyFor(obj):
	object.__setattr__(obj, '_dynamicProxy', obj)
