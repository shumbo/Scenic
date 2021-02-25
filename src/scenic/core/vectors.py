"""Scenic vectors and vector fields."""

import math
from math import sin, cos, sqrt, atan
import random
import collections
import itertools
import functools

import shapely.geometry
from scipy.spatial.transform import Rotation

from scenic.core.distributions import (Samplable, Distribution, MethodDistribution,
									   needsSampling, makeOperatorHandler, distributionMethod,
									   distributionFunction)
from scenic.core.lazy_eval import valueInContext, needsLazyEvaluation, makeDelayedFunctionCall
import scenic.core.utils as utils
from scenic.core.geometry import normalizeAngle, dotProduct, norm

class VectorDistribution(Distribution):
	"""A distribution over Vectors."""
	defaultValueType = None		# will be set after Vector is defined

	def toVector(self):
		return self

class CustomVectorDistribution(VectorDistribution):
	"""Distribution with a custom sampler given by an arbitrary function."""
	def __init__(self, sampler, *dependencies, name='CustomVectorDistribution', evaluator=None):
		super().__init__(*dependencies)
		self.sampler = sampler
		self.name = name
		self.evaluator = evaluator

	def sampleGiven(self, value):
		return self.sampler(value)

	def evaluateInner(self, context):
		if self.evaluator is None:
			raise NotImplementedError('evaluateIn() not supported by this distribution')
		return self.evaluator(self, context)

	def __str__(self):
		deps = utils.argsToString(self.dependencies)
		return f'{self.name}{deps}'

class VectorOperatorDistribution(VectorDistribution):
	"""Vector version of OperatorDistribution."""
	def __init__(self, operator, obj, operands):
		super().__init__(obj, *operands)
		self.operator = operator
		self.object = obj
		self.operands = operands

	def sampleGiven(self, value):
		first = value[self.object]
		rest = (value[child] for child in self.operands)
		op = getattr(first, self.operator)
		return op(*rest)

	def evaluateInner(self, context):
		obj = valueInContext(self.object, context)
		operands = tuple(valueInContext(arg, context) for arg in self.operands)
		return VectorOperatorDistribution(self.operator, obj, operands)

	def __str__(self):
		ops = utils.argsToString(self.operands)
		return f'{self.object}.{self.operator}{ops}'

class VectorMethodDistribution(VectorDistribution):
	"""Vector version of MethodDistribution."""
	def __init__(self, method, obj, args, kwargs):
		super().__init__(*args, *kwargs.values())
		self.method = method
		self.object = obj
		self.arguments = args
		self.kwargs = kwargs

	def sampleGiven(self, value):
		args = (value[arg] for arg in self.arguments)
		kwargs = { name: value[arg] for name, arg in self.kwargs.items() }
		return self.method(self.object, *args, **kwargs)

	def evaluateInner(self, context, modifying):
		obj = valueInContext(self.object, context)
		arguments = tuple(valueInContext(arg, context) for arg in self.arguments)
		kwargs = { name: valueInContext(arg, context) for name, arg in self.kwargs.items() }
		return VectorMethodDistribution(self.method, obj, arguments, kwargs)

	def __str__(self):
		args = utils.argsToString(itertools.chain(self.arguments, self.kwargs.values()))
		return f'{self.object}.{self.method.__name__}{args}'

def scalarOperator(method):
	"""Decorator for vector operators that yield scalars."""
	op = method.__name__
	setattr(VectorDistribution, op, makeOperatorHandler(op))

	@functools.wraps(method)
	def handler2(self, *args, **kwargs):
		if any(needsSampling(arg) for arg in itertools.chain(args, kwargs.values())):
			return MethodDistribution(method, self, args, kwargs)
		else:
			return method(self, *args, **kwargs)
	return handler2

def makeVectorOperatorHandler(op):
	def handler(self, *args):
		return VectorOperatorDistribution(op, self, args)
	return handler
def vectorOperator(method):
	"""Decorator for vector operators that yield vectors."""
	op = method.__name__
	setattr(VectorDistribution, op, makeVectorOperatorHandler(op))

	@functools.wraps(method)
	def handler2(self, *args):
		if needsSampling(self):
			return VectorOperatorDistribution(op, self, args)
		elif any(needsSampling(arg) for arg in args):
			return VectorMethodDistribution(method, self, args, {})
		elif any(needsLazyEvaluation(arg) for arg in args):
			# see analogous comment in distributionFunction
			return makeDelayedFunctionCall(handler2, args, {})
		else:
			return method(self, *args)
	return handler2

def vectorDistributionMethod(method):
	"""Decorator for methods that produce vectors. See distributionMethod."""
	@functools.wraps(method)
	def helper(self, *args, **kwargs):
		if any(needsSampling(arg) for arg in itertools.chain(args, kwargs.values())):
			return VectorMethodDistribution(method, self, args, kwargs)
		elif any(needsLazyEvaluation(arg) for arg in itertools.chain(args, kwargs.values())):
			# see analogous comment in distributionFunction
			return makeDelayedFunctionCall(helper, (self,) + args, kwargs)
		else:
			return method(self, *args, **kwargs)
	return helper

class Orientation():
	"""A quaternion representation of an orientation."""
	def __init__(self, quaternion):
		# TODO: @Matthew SciPy typecheck 'quaternion' 
		r = Rotation.from_quat(quaternion)
		self.q = r.as_quat()

	@property
	def w(self):
		return self.q[3]

	@property
	def x(self):
		return self.q[0]

	@property
	def y(self):  
		return self.q[1]

	@property
	def z(self): 
		return self.q[2]

	@classmethod 
	@distributionFunction
	def fromEuler(cls, yaw, pitch, roll):
		return Orientation(Rotation.from_euler('ZXY', [yaw, pitch, roll], degrees=False).as_quat())

	def getEuler(self):
		r = Rotation.from_quat(self.q)
		return r.as_euler('ZXY', degrees=False)

	def getRotation(self):
		return Rotation.from_quat(self.q)

	def invertRotation(self):
		r = Rotation.from_quat(self.q)
		return Orientation(r.inv().as_quat())

	def __mul__(self, other):
		if type(other) is not Orientation:
			return NotImplemented
		r = Rotation.from_quat(self.q)
		r2 = Rotation.from_quat(other.q)
		return Orientation((r * r2).as_quat())
	
	@distributionFunction
	def __add__(self, other):
		if isinstance(other, (float, int)):
			other = Orientation.fromEuler(other, 0, 0)
		elif type(other) is not Orientation:
			return NotImplemented
		return other * self

	@distributionFunction
	def __radd__(self, other):
		if isinstance(other, (float, int)):
			other = Orientation.fromEuler(other, 0, 0)
		elif type(other) is not Orientation:
			return NotImplemented
		return self * other

	def __getitem__(self, index):
		return self.q[index]

	def __repr__(self):
		return f'Orientation({self.q!r})'

	@distributionFunction
	def globalToLocalAngles(self, yaw, pitch, roll):
		"""Find Euler angles w.r.t. a given parent orientation."""
		orientation = Orientation.fromEuler(yaw, pitch, roll)
		inverseQuat = self.invertRotation()
		desiredQuat = inverseQuat * orientation 
		euler = desiredQuat.getEuler()
		return euler 


class Vector(Samplable, collections.abc.Sequence):
	"""A 2D vector, whose coordinates can be distributions."""
	def __init__(self, x, y=0, z=0):
		self.coordinates = (x, y, z)
		super().__init__(self.coordinates)

	@property
	def x(self):
		return self.coordinates[0]

	@property
	def y(self):
		return self.coordinates[1]

	@property
	def z(self):
		return self.coordinates[2]

	def toVector(self):
		return self

	def sampleGiven(self, value):
		return Vector(*(value[coord] for coord in self.coordinates))

	def evaluateInner(self, context, modifying):
		return Vector(*(valueInContext(coord, context) for coord in self.coordinates))

	@vectorOperator
	def applyRotation(self, rotation):
		if not isinstance(rotation, Orientation):
			return NotImplemented
		r = rotation.getRotation()
		return Vector(*r.apply(list(self.coordinates)))

	@vectorOperator
	def cartesianToSpherical(self):
		"""Returns this vector in spherical coordinates"""
		rho = math.hypot(self.x, self.y, self.z) 
		theta = math.atan2(self.y, self.x) - math.pi/2
		phi = math.atan2(self.z, math.hypot(self.x,self.y))
		return Vector(rho, theta, phi)

	@vectorOperator
	def rotatedBy(self, angle):
		"""Return a vector equal to this one rotated counterclockwise by the given angle."""
		x, y, z = self.x, self.y, self.z
		if isinstance(angle, tuple):
			angle = angle[0]
		c, s = cos(angle), sin(angle)
		return Vector((c * x) - (s * y), (s * x) + (c * y))

	@vectorOperator
	def offsetRotated(self, heading, offset):
		ro = offset.rotatedBy(heading)
		return self + ro

	@vectorOperator
	def offsetRadially(self, radius, heading):
		return self.offsetRotated(heading, Vector(0, radius))

	@scalarOperator
	def distanceTo(self, other):
		dx, dy, dz = other.toVector() - self
		return math.hypot(dx, dy, dz)

	@vectorOperator
	def __add__(self, other):
		return Vector(self[0] + other[0], self[1] + other[1], self[2] + other[2])

	@vectorOperator
	def __radd__(self, other):
		return Vector(self[0] + other[0], self[1] + other[1], self[2] + other[2])

	@vectorOperator
	def __sub__(self, other):
		return Vector(self[0] - other[0], self[1] - other[1], self[2] - other[2])

	@vectorOperator
	def __rsub__(self, other):
		return Vector(other[0] - self[0], other[1] - self[1], self[2] - other[2])
	
	@scalarOperator
	def angleTo(self, other):
		dx, dy, dz = other.toVector() - self
		return normalizeAngle(math.atan2(dy, dx) - (math.pi / 2))

	@scalarOperator
	def angleBetween(self, other):
		return normalizeAngle(math.acos(dotProduct(self, other) / norm(self) * norm(other)) - (math.pi / 2))

	def __len__(self):
		return len(self.coordinates)

	def __getitem__(self, index):
		return self.coordinates[index]

	def __repr__(self):
		return f'({self.x}, {self.y}, {self.z})'

	def __eq__(self, other):
		if type(other) is not Vector:
			return NotImplemented
		return other.coordinates == self.coordinates

	def __hash__(self):
		return hash(self.coordinates)

VectorDistribution.defaultValueType = Vector

class OrientedVector(Vector):
	def __init__(self, x, y, heading, z=0):
		super().__init__(x, y, z)
		# TODO: @Matthew OrientedVector initializer should use Orientation
		self.heading = heading

	def toHeading(self):
		return self.heading

	def __eq__(self, other):
		if type(other) is not OrientedVector:
			return NotImplemented
		return (other.coordinates == self.coordinates
			and other.heading == self.heading)

	def __hash__(self):
		return hash((self.coordinates, self.heading))

class VectorField:
	def __init__(self, name, value):
		# TODO: @Matthew needs to return an orientation, not just 3 angles 
		self.name = name
		self.value = value
		self.valueType = float

	@distributionMethod
	def __getitem__(self, pos):
		val = self.value(pos)
		if isinstance(val, (int, float)):
			val = Orientation.fromEuler(val, 0, 0)
		return val

	@vectorDistributionMethod
	def followFrom(self, pos, dist, steps=4):
		step = dist / steps
		for i in range(steps):
			pos = pos.offsetRadially(step, self[pos])
		return pos

	def __str__(self):
		return f'<{type(self).__name__} {self.name}>'

class PolygonalVectorField(VectorField):
	def __init__(self, name, cells, headingFunction=None, defaultHeading=None):
		self.cells = tuple(cells)
		if headingFunction is None and defaultHeading is not None:
			headingFunction = lambda pos, specifier: defaultHeading
		self.headingFunction = headingFunction
		for cell, heading in self.cells:
			if heading is None and headingFunction is None and defaultHeading is None:
				raise RuntimeError(f'missing heading for cell of PolygonalVectorField')
		self.defaultHeading = defaultHeading
		super().__init__(name, self.valueAt)

	def valueAt(self, pos):
		point = shapely.geometry.Point(pos)
		for cell, heading in self.cells:
			if cell.intersects(point):
				return self.headingFunction(pos) if heading is None else heading
		if self.defaultHeading is not None:
			return self.defaultHeading
		raise RuntimeError(f'evaluated PolygonalVectorField at undefined point {pos}')

class PolyhedronVectorField(VectorField):
	pass