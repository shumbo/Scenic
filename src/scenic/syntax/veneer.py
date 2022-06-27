
"""Python implementations of Scenic language constructs.

This module is automatically imported by all Scenic programs. In addition to
defining the built-in functions, operators, specifiers, etc., it also stores
global state such as the list of all created Scenic objects.
"""

__all__ = (
	# Primitive statements and functions
	'ego', 'require', 'resample', 'param', 'globalParameters', 'mutate', 'verbosePrint',
	'localPath', 'model', 'simulator', 'simulation', 'require_always', 'terminate_when',
	'terminate_simulation_when', 'terminate_after', 'in_initial_scenario',
	'sin', 'cos', 'hypot', 'max', 'min',
	'filter',
	# Prefix operators
	'Visible', 'NotVisible',
	'Front', 'Back', 'Left', 'Right',
	'FrontLeft', 'FrontRight', 'BackLeft', 'BackRight',
	'Top', 'Bottom', 'TopFrontLeft', 'TopFrontRight', 'TopBackLeft',
	'TopBackRight', 'BottomFrontLeft', 'BottomFrontRight', 'BottomBackLeft',
	'BottomBackRight',
	# Infix operators
	'FieldAt', 'RelativeTo', 'OffsetAlong', 'RelativePosition',
	'RelativeHeading', 'ApparentHeading',
	'DistanceFrom', 'AngleTo', 'AngleFrom', 'Follow', 'CanSee',
	# Primitive types
	'Vector', 'VectorField', 'PolygonalVectorField',
	'MeshShape', 'BoxShape',
	'MeshVolumeRegion', 'MeshSurfaceRegion', 
	'BoxRegion', 'SpheroidRegion',
	'Region', 'PointSetRegion', 'RectangularRegion', 'CircularRegion', 'SectorRegion',
	'PolygonalRegion', 'PolylineRegion',
	'Workspace', 'Mutator',
	'Range', 'DiscreteRange', 'Options', 'Uniform', 'Discrete', 'Normal',
	'TruncatedNormal',
	'VerifaiParameter', 'VerifaiRange', 'VerifaiDiscreteRange', 'VerifaiOptions',
	# Constructible types
	'Point', 'OrientedPoint', 'Object',
	# Specifiers
	'With',
	'At', 'In', 'On', 'Beyond', 'VisibleFrom', 'VisibleSpec', 'OffsetBy', 'OffsetAlongSpec',
	'Facing', 'FacingToward', 'ApparentlyFacing', 'FacingDirectlyToward', 'FacingAwayFrom',
	'LeftSpec', 'RightSpec', 'Ahead', 'Behind', 'Above', 'Below',
	'Following',
	# Constants
	'everywhere', 'nowhere',
	# Exceptions
	'GuardViolation', 'PreconditionViolation', 'InvariantViolation',
	# Internal APIs 	# TODO remove?
	'PropertyDefault', 'Behavior', 'Monitor', 'makeTerminationAction',
	'BlockConclusion', 'runTryInterrupt', 'wrapStarredValue', 'callWithStarArgs',
	'Modifier', 'DynamicScenario'
)

# various Python types and functions used in the language but defined elsewhere
from scenic.core.geometry import sin, cos, hypot, max, min
from scenic.core.vectors import Vector, VectorField, PolygonalVectorField
from scenic.core.shapes import MeshShape, BoxShape
from scenic.core.regions import (Region, PointSetRegion, RectangularRegion,
	CircularRegion, SectorRegion, PolygonalRegion, PolylineRegion,
	everywhere, nowhere,
	MeshVolumeRegion, MeshSurfaceRegion,
	BoxRegion, SpheroidRegion)
from scenic.core.workspaces import Workspace
from scenic.core.distributions import (Range, DiscreteRange, Options, Uniform, Normal,
	TruncatedNormal)
Discrete = Options
from scenic.core.external_params import (VerifaiParameter, VerifaiRange, VerifaiDiscreteRange,
										 VerifaiOptions)
from scenic.core.object_types import Mutator, Point, OrientedPoint, Object
from scenic.core.specifiers import PropertyDefault	# TODO remove
from scenic.core.dynamics import (Behavior, Monitor, DynamicScenario, BlockConclusion,
                                  GuardViolation, PreconditionViolation, InvariantViolation,
                                  makeTerminationAction, runTryInterrupt)

# everything that should not be directly accessible from the language is imported here:
import builtins
import collections.abc
from contextlib import contextmanager
import importlib
import sys
import random
import os.path
import traceback
import typing
from scenic.core.distributions import (RejectionException, Distribution,
									   TupleDistribution, StarredDistribution, toDistribution,
									   needsSampling, canUnpackDistributions, distributionFunction)
from scenic.core.type_support import (isA, toType, toTypes, toScalar, toHeading, toVector,
									  evaluateRequiringEqualTypes, underlyingType, toOrientation,
									  canCoerce, coerce)
from scenic.core.geometry import normalizeAngle, apparentHeadingAtPoint
from scenic.core.object_types import _Constructible
from scenic.core.specifiers import Specifier, ModifyingSpecifier
from scenic.core.lazy_eval import (DelayedArgument, needsLazyEvaluation, requiredProperties,
                                   valueInContext)
from scenic.core.errors import RuntimeParseError, InvalidScenarioError
from scenic.core.vectors import Orientation, alwaysGlobalOrientation
from scenic.core.external_params import ExternalParameter
import scenic.core.requirements as requirements
from scenic.core.simulators import RejectSimulationException

### Internals

activity = 0
currentScenario = None
scenarioStack = []
scenarios = []
evaluatingRequirement = False
_globalParameters = {}
lockedParameters = set()
lockedModel = None
loadingModel = False
currentSimulation = None
inInitialScenario = True
runningScenarios = set()
currentBehavior = None
simulatorFactory = None
evaluatingGuard = False

## APIs used internally by the rest of Scenic

# Scenic compilation

def isActive():
	"""Are we in the middle of compiling a Scenic module?

	The 'activity' global can be >1 when Scenic modules in turn import other
	Scenic modules."""
	return activity > 0

def activate(paramOverrides={}, modelOverride=None, filename=None, namespace=None):
	"""Activate the veneer when beginning to compile a Scenic module."""
	global activity, _globalParameters, lockedParameters, lockedModel, currentScenario
	if paramOverrides or modelOverride:
		assert activity == 0
		_globalParameters.update(paramOverrides)
		lockedParameters = set(paramOverrides)
		lockedModel = modelOverride

	activity += 1
	assert not evaluatingRequirement
	assert not evaluatingGuard
	assert currentSimulation is None
	# placeholder scenario for top-level code
	newScenario = DynamicScenario._dummy(filename, namespace)
	scenarioStack.append(newScenario)
	currentScenario = newScenario

def deactivate():
	"""Deactivate the veneer after compiling a Scenic module."""
	global activity, _globalParameters, lockedParameters, lockedModel
	global currentScenario, scenarios, scenarioStack, simulatorFactory
	activity -= 1
	assert activity >= 0
	assert not evaluatingRequirement
	assert not evaluatingGuard
	assert currentSimulation is None
	scenarioStack.pop()
	assert len(scenarioStack) == activity
	scenarios = []

	if activity == 0:
		lockedParameters = set()
		lockedModel = None
		currentScenario = None
		simulatorFactory = None
		_globalParameters = {}
	else:
		currentScenario = scenarioStack[-1]

# Object creation

def registerObject(obj):
	"""Add a Scenic object to the global list of created objects.

	This is called by the Object constructor.
	"""
	if evaluatingRequirement:
		raise RuntimeParseError('tried to create an object inside a requirement')
	elif currentBehavior is not None:
		raise RuntimeParseError('tried to create an object inside a behavior')
	elif activity > 0 or currentScenario:
		assert not evaluatingRequirement
		assert isinstance(obj, _Constructible)
		currentScenario._registerObject(obj)
		if currentSimulation:
			currentSimulation.createObject(obj)

# External parameter creation

def registerExternalParameter(value):
	"""Register a parameter whose value is given by an external sampler."""
	if activity > 0:
		assert isinstance(value, ExternalParameter)
		currentScenario._externalParameters.append(value)

# Function call support

def wrapStarredValue(value, lineno):
	if isinstance(value, TupleDistribution) or not needsSampling(value):
		return value
	elif isinstance(value, Distribution):
		return [StarredDistribution(value, lineno)]
	else:
		raise RuntimeParseError(f'iterable unpacking cannot be applied to {value}')

def callWithStarArgs(_func_to_call, *args, **kwargs):
	if not canUnpackDistributions(_func_to_call):
		# wrap function to delay evaluation until starred distributions are sampled
		_func_to_call = distributionFunction(_func_to_call)
	return _func_to_call(*args, **kwargs)

# Simulations

def instantiateSimulator(factory, params):
	global _globalParameters
	assert not _globalParameters		# TODO improve hack?
	_globalParameters = dict(params)
	try:
		return factory()
	finally:
		_globalParameters = {}

def beginSimulation(sim):
	global currentSimulation, currentScenario, inInitialScenario, runningScenarios
	global _globalParameters
	if isActive():
		raise RuntimeError('tried to start simulation during Scenic compilation!')
	assert currentSimulation is None
	assert currentScenario is None
	assert not scenarioStack
	currentSimulation = sim
	inInitialScenario = True
	currentScenario = sim.scene.dynamicScenario
	runningScenarios = {currentScenario}
	currentScenario._bindTo(sim.scene)
	_globalParameters = dict(sim.scene.params)

	# rebind globals that could be referenced by behaviors to their sampled values
	for modName, (namespace, sampledNS, originalNS) in sim.scene.behaviorNamespaces.items():
		namespace.clear()
		namespace.update(sampledNS)

def endSimulation(sim):
	global currentSimulation, currentScenario, currentBehavior, runningScenarios
	global _globalParameters
	currentSimulation = None
	currentScenario = None
	runningScenarios = set()
	currentBehavior = None
	_globalParameters = {}

	for modName, (namespace, sampledNS, originalNS) in sim.scene.behaviorNamespaces.items():
		namespace.clear()
		namespace.update(originalNS)

def simulationInProgress():
	return currentSimulation is not None

# Requirements

@contextmanager
def executeInRequirement(scenario, boundEgo):
	global evaluatingRequirement, currentScenario
	assert not evaluatingRequirement
	evaluatingRequirement = True
	if currentScenario is None:
		currentScenario = scenario
		clearScenario = True
	else:
		assert currentScenario is scenario
		clearScenario = False
	oldEgo = currentScenario._ego
	if boundEgo:
		currentScenario._ego = boundEgo
	try:
		yield
	finally:
		evaluatingRequirement = False
		currentScenario._ego = oldEgo
		if clearScenario:
			currentScenario = None

# Dynamic scenarios

def registerDynamicScenarioClass(cls):
	scenarios.append(cls)

@contextmanager
def executeInScenario(scenario, inheritEgo=False):
	global currentScenario
	oldScenario = currentScenario
	if inheritEgo and oldScenario is not None:
		scenario._ego = oldScenario._ego 	# inherit ego from parent
	currentScenario = scenario
	try:
		yield
	finally:
		currentScenario = oldScenario

def prepareScenario(scenario):
	if currentSimulation:
		verbosePrint(f'Starting scenario {scenario}', level=3)

def finishScenarioSetup(scenario):
	global inInitialScenario
	inInitialScenario = False

def startScenario(scenario):
	runningScenarios.add(scenario)

def endScenario(scenario, reason):
	runningScenarios.remove(scenario)
	verbosePrint(f'Stopping scenario {scenario} because of: {reason}', level=3)

# Dynamic behaviors

@contextmanager
def executeInBehavior(behavior):
	global currentBehavior
	oldBehavior = currentBehavior
	currentBehavior = behavior
	try:
		yield
	finally:
		currentBehavior = oldBehavior

@contextmanager
def executeInGuard():
	global evaluatingGuard
	assert not evaluatingGuard
	evaluatingGuard = True
	try:
		yield
	finally:
		evaluatingGuard = False

### Parsing support

class Modifier(typing.NamedTuple):
	name: str
	value: typing.Any
	terminator: typing.Optional[str] = None

### Primitive statements and functions

def ego(obj=None):
	"""Function implementing loads and stores to the 'ego' pseudo-variable.

	The translator calls this with no arguments for loads, and with the source
	value for stores.
	"""
	egoObject = currentScenario._ego
	if obj is None:
		if egoObject is None:
			raise RuntimeParseError('referred to ego object not yet assigned')
	elif not isinstance(obj, Object):
		raise RuntimeParseError('tried to make non-object the ego object')
	else:
		currentScenario._ego = obj
		for scenario in runningScenarios:
			if scenario._ego is None:
				scenario._ego = obj
	return egoObject

def require(reqID, req, line, prob=1):
	"""Function implementing the require statement."""
	if evaluatingRequirement:
		raise RuntimeParseError('tried to create a requirement inside a requirement')
	if currentSimulation is not None:	# requirement being evaluated at runtime
		if prob >= 1 or random.random() <= prob:
			result = req()
			assert not needsSampling(result)
			if needsLazyEvaluation(result):
				raise RuntimeParseError(f'requirement on line {line} uses value'
										' undefined outside of object definition')
			if not result:
				raise RejectSimulationException(f'requirement on line {line}')
	else:	# requirement being defined at compile time
		currentScenario._addRequirement(requirements.RequirementType.require,
                                        reqID, req, line, prob)

def require_always(reqID, req, line):
	"""Function implementing the 'require always' statement."""
	makeRequirement(requirements.RequirementType.requireAlways, reqID, req, line)

def terminate_when(reqID, req, line):
	"""Function implementing the 'terminate when' statement."""
	makeRequirement(requirements.RequirementType.terminateWhen, reqID, req, line)

def terminate_simulation_when(reqID, req, line):
	"""Function implementing the 'terminate simulation when' statement."""
	makeRequirement(requirements.RequirementType.terminateSimulationWhen,
                    reqID, req, line)

def makeRequirement(ty, reqID, req, line):
	if evaluatingRequirement:
		raise RuntimeParseError(f'tried to use "{ty.value}" inside a requirement')
	elif currentBehavior is not None:
		raise RuntimeParseError(f'"{ty.value}" inside a behavior on line {line}')
	elif currentSimulation is not None:
		currentScenario._addDynamicRequirement(ty, req, line)
	else:	# requirement being defined at compile time
		currentScenario._addRequirement(ty, reqID, req, line, 1)

def terminate_after(timeLimit, terminator=None):
	if not isinstance(timeLimit, (float, int)):
		raise RuntimeParseError('"terminate after N" with N not a number')
	assert terminator in (None, 'seconds', 'steps')
	inSeconds = (terminator != 'steps')
	currentScenario._setTimeLimit(timeLimit, inSeconds=inSeconds)

def resample(dist):
	"""The built-in resample function."""
	return dist.clone() if isinstance(dist, Distribution) else dist

def verbosePrint(msg, file=sys.stdout, level=1):
	"""Built-in function printing a message when the verbosity is >0.

	(Or when the verbosity exceeds the specified level.)
	"""
	import scenic.syntax.translator as translator
	if translator.verbosity >= level:
		if currentSimulation:
			indent = '      ' if translator.verbosity >= 3 else '  '
		else:
			indent = '  ' * activity if translator.verbosity >= 2 else '  '
		print(indent + msg, file=file)

def localPath(relpath):
	filename = traceback.extract_stack(limit=2)[0].filename
	base = os.path.dirname(filename)
	return os.path.join(base, relpath)

def simulation():
	if isActive():
		raise RuntimeParseError('used simulation() outside a behavior')
	assert currentSimulation is not None
	return currentSimulation

def simulator(sim):
	global simulatorFactory
	simulatorFactory = sim

def in_initial_scenario():
	return inInitialScenario

def model(namespace, modelName):
	global loadingModel
	if loadingModel:
		raise RuntimeParseError(f'Scenic world model itself uses the "model" statement')
	if lockedModel is not None:
		modelName = lockedModel
	try:
		loadingModel = True
		module = importlib.import_module(modelName)
	except ModuleNotFoundError as e:
		if e.name == modelName:
			raise InvalidScenarioError(f'could not import world model {modelName}') from None
		else:
			raise
	finally:
		loadingModel = False
	names = module.__dict__.get('__all__', None)
	if names is not None:
		for name in names:
			namespace[name] = getattr(module, name)
	else:
		for name, value in module.__dict__.items():
			if not name.startswith('_'):
				namespace[name] = value

@distributionFunction
def filter(function, iterable):
	return list(builtins.filter(function, iterable))

def param(*quotedParams, **params):
	"""Function implementing the param statement."""
	global loadingModel
	if evaluatingRequirement:
		raise RuntimeParseError('tried to create a global parameter inside a requirement')
	elif currentSimulation is not None:
		raise RuntimeParseError('tried to create a global parameter during a simulation')
	for name, value in params.items():
		if name not in lockedParameters and (not loadingModel or name not in _globalParameters):
			_globalParameters[name] = toDistribution(value)
	assert len(quotedParams) % 2 == 0, quotedParams
	it = iter(quotedParams)
	for name, value in zip(it, it):
		if name not in lockedParameters:
			_globalParameters[name] = toDistribution(value)

class ParameterTableProxy(collections.abc.Mapping):
	def __init__(self, map):
		self._internal_map = map

	def __getitem__(self, name):
		return self._internal_map[name]

	def __iter__(self):
		return iter(self._internal_map)

	def __len__(self):
		return len(self._internal_map)

	def __getattr__(self, name):
		return self.__getitem__(name)	# allow namedtuple-like access

	def _clone_table(self):
		return ParameterTableProxy(self._internal_map.copy())

def globalParameters():
	return ParameterTableProxy(_globalParameters)

def mutate(*objects):		# TODO update syntax
	"""Function implementing the mutate statement."""
	if evaluatingRequirement:
		raise RuntimeParseError('used mutate statement inside a requirement')
	if len(objects) == 0:
		objects = currentScenario._objects
	for obj in objects:
		if not isinstance(obj, Object):
			raise RuntimeParseError('"mutate X" with X not an object')
		obj.mutationEnabled = True

### Prefix operators

def Visible(region):
	"""The 'visible <region>' operator."""
	region = toType(region, Region, '"visible X" with X not a Region')
	return region.intersect(ego().visibleRegion)

def NotVisible(region):
	"""The 'not visible <region>' operator."""
	region = toType(region, Region, '"not visible X" with X not a Region')
	return region.difference(ego().visibleRegion)

# front of <object>, etc.
ops = (
	'front', 'back', 'left', 'right',
	'front left', 'front right',
	'back left', 'back right', 'top',
	'bottom', 'top front left', 'top front right',
	'top back left', 'top back right',
	'bottom front left', 'bottom front right',
	'bottom back left', 'bottom back right'
)
template = '''\
def {function}(X):
	"""The '{syntax} of <object>' operator."""
	if not isinstance(X, Object):
		raise RuntimeParseError('"{syntax} of X" with X not an Object')
	return X.{property}
'''
for op in ops:
	func = ''.join(word.capitalize() for word in op.split(' '))
	prop = func[0].lower() + func[1:]
	definition = template.format(function=func, syntax=op, property=prop)
	exec(definition)

### Infix operators

def FieldAt(X, Y):
	"""The '<VectorField> at <vector>' operator."""
	if not isinstance(X, VectorField):
		raise RuntimeParseError('"X at Y" with X not a vector field')
	Y = toVector(Y, '"X at Y" with Y not a vector')
	return X[Y]

def RelativeTo(X, Y):
	"""The 'X relative to Y' polymorphic operator.

	Allowed forms:
		F relative to G (with at least one a field, the other a field or heading)
		<vector> relative to <oriented point> (and vice versa)
		<vector> relative to <vector>
		<heading> relative to <heading>
		with <property> <value> relative to <oriented point | heading> 
	"""
	xf, yf = isA(X, VectorField), isA(Y, VectorField)
	if xf or yf:
		if xf and yf and X.valueType != Y.valueType:
			raise RuntimeParseError('"X relative to Y" with X, Y fields of different types')
		fieldType = X.valueType if xf else Y.valueType
		error = '"X relative to Y" with field and value of different types'
		def helper(context, spec):
			pos = context.position.toVector()
			xp = X[pos] if xf else toType(X, fieldType, error)
			yp = Y[pos] if yf else toType(Y, fieldType, error)
			return yp + xp
		return DelayedArgument({'position'}, helper)
	# elif isinstance(Y, Object) or isinstance(Y, OrientedPoint):
	# 	if not isinstance(X, float):
	# 		raise RuntimeParseError('"X relative to Y" with Y an object or oriented point, but X not a float')
	# 	# return X + Y.prop # TODO: @Matthew Need context for which property?
	# 	def helper(context, spec):
	# 		breakpoint()
	# 		print("hi")
	# 	return DelayedArgument({'parentOrientation'}, value=helper)
	else:
		if isinstance(X, OrientedPoint):	# TODO too strict?
			if isinstance(Y, OrientedPoint):
				raise RuntimeParseError('"X relative to Y" with X, Y both oriented points')
			Y = toVector(Y, '"X relative to Y" with X an oriented point but Y not a vector')
			return X.relativize(Y)
		elif isinstance(Y, OrientedPoint):
			X = toVector(X, '"X relative to Y" with Y an oriented point but X not a vector')
			return Y.relativize(X)
		else:
			X = toTypes(X, (Vector, Orientation), '"X relative to Y" with X neither a vector nor orientation')
			Y = toTypes(Y, (Vector, Orientation), '"X relative to Y" with Y neither a vector nor orientation')
			return evaluateRequiringEqualTypes(lambda: Y + X, X, Y,
											   '"X relative to Y" with vector and orientation')

def OffsetAlong(X, H, Y, specs=None):
	"""The 'X offset along H by Y' polymorphic operator.

	Allowed forms:
		<vector> offset along <heading> by <vector>
		<vector> offset along <field> by <vector>
	"""
	X = toVector(X, '"X offset along H by Y" with X not a vector')
	Y = toVector(Y, '"X offset along H by Y" with Y not a vector')
	if isinstance(H, VectorField):
		H = H[X]
	H = toOrientation(H, '"X offset along H by Y" with H not an orientation or vector field')
	return X.offsetRotated(H, Y)

def RelativePosition(X, Y=None):
	"""The 'relative position of <vector> [from <vector>]' operator.

	If the 'from <vector>' is omitted, the position of ego is used.
	"""
	X = toVector(X, '"relative position of X from Y" with X not a vector')
	if Y is None:
		Y = ego()
	Y = toVector(Y, '"relative position of X from Y" with Y not a vector')
	return X - Y

def RelativeHeading(X, Y=None):
	"""The 'relative heading of <heading> [from <heading>]' operator.

	If the 'from <heading>' is omitted, the heading of ego is used.
	"""
	X = toHeading(X, '"relative heading of X from Y" with X not a heading')
	if Y is None:
		Y = (ego().heading, 0, 0)
	else:
		Y = toHeading(Y, '"relative heading of X from Y" with Y not a heading')
	# TODO: should we actually return a tripple of 3 angles?
	return normalizeAngle(X[0] - Y[0]) 

def ApparentHeading(X, Y=None):
	"""The 'apparent heading of <oriented point> [from <vector>]' operator.

	If the 'from <vector>' is omitted, the position of ego is used.
	"""
	if not isinstance(X, OrientedPoint):
		raise RuntimeParseError('"apparent heading of X from Y" with X not an OrientedPoint')
	if Y is None:
		Y = ego()
	Y = toVector(Y, '"relative heading of X from Y" with Y not a vector')
	return apparentHeadingAtPoint(X.position, X.heading, Y)

def DistanceFrom(X, Y=None):
	"""The ``distance from {X} to {Y}`` polymorphic operator.

	Allowed forms:

	* ``distance from`` <vector> [``to`` <vector>]
	* ``distance from`` <region> [``to`` <vector>]
	* ``distance from`` <vector> ``to`` <region>

	If the ``to <vector>`` is omitted, the position of ego is used.
	"""
	X = toTypes(X, (Vector, Region), '"distance from X to Y" with X neither a vector nor region')
	if Y is None:
		Y = ego()
	Y = toTypes(Y, (Vector, Region), '"distance from X to Y" with Y neither a vector nor region')
	return X.distanceTo(Y)

def AngleTo(X):
	"""The 'angle to <vector>' operator (using the position of ego as the reference)."""
	X = toVector(X, '"angle to X" with X not a vector')
	return ego().angleTo(X)

def AngleFrom(X, Y):
	"""The 'angle from <vector> to <vector>' operator."""
	X = toVector(X, '"angle from X to Y" with X not a vector')
	Y = toVector(Y, '"angle from X to Y" with Y not a vector')
	return X.angleTo(Y)

def Follow(F, X, D):
	"""The 'follow <field> from <vector> for <number>' operator."""
	if not isinstance(F, VectorField):
		raise RuntimeParseError('"follow F from X for D" with F not a vector field')
	X = toVector(X, '"follow F from X for D" with X not a vector')
	D = toScalar(D, '"follow F from X for D" with D not a number')
	pos = F.followFrom(X, D)
	orientation = F[pos]
	return OrientedPoint(position=pos, parentOrientation=orientation)

def CanSee(X, Y):
	"""The 'X can see Y' polymorphic operator.

	Allowed forms:
		<point> can see <object>
		<point> can see <vector>
	"""
	if not isinstance(X, Point):
		raise RuntimeParseError('"X can see Y" with X not a Point')
	if isinstance(Y, Point):
		return X.canSee(Y)
	else:
		Y = toVector(Y, '"X can see Y" with Y not a vector')
		return X.visibleRegion.containsPoint(Y)

### Specifiers

def With(prop, val):
	"""The 'with <property> <value>' specifier.

	Specifies the given property, with no dependencies. If composed with 'relative to Y',
	the given property is specified with respect to the same property of Y (i.e., value
	is added to the value of the property of Y).
	"""
	return Specifier({prop: 1}, {prop: val})

def At(pos):
	"""The 'at <vector>' specifier.

	Specifies 'position', with no dependencies."""
	pos = toVector(pos, 'specifier "at X" with X not a vector')
	return Specifier({'position': 1}, {'position': pos})

def In(region):
	"""The 'in <region>' specifier.

	Specifies 'position', with no dependencies. Optionally specifies 'heading'
	if the given Region has a preferred orientation.
	
	If composed with 'on <X>', X must intersect the region. This specifies
	position and orientation. 
	"""
	region = toType(region, Region, 'specifier "in R" with R not a Region')
	pos = Region.uniformPointIn(region)
	props = {'position': 1}
	values = {'position': pos}
	if alwaysProvidesOrientation(region):
		props['parentOrientation'] = 3
		values['parentOrientation'] = region.orientation[pos]
	return Specifier(props, values)

def On(thing):
	"""The 'on <X>' specifier.

	Specifies 'position' and 'parentOrientation' with no dependencies.

	May be used to modify an already specified 'position' property if a compatible
	specifier has already done so.

	Allowed forms:
		on <region>
		on <object> 
	"""
	# TODO: @Matthew Helper function for delayed argument checks if modifying or not

	if isinstance(thing, Object):
		region = toType(thing.topSurface, Region, 'Cannot coax topSurface of Object to Region')
	else:
		region = toType(thing, Region, 'specifier "on R" with R not a Region')

	props = {'position': 1}

	if alwaysProvidesOrientation(region):
		props['parentOrientation'] = 2

	def helper(context, spec):
		pos = Region.uniformPointIn(region)

		values = {}

		contactOffset = Vector(0,0,context.contactTolerance) - context.centerOffset

		if 'parentOrientation' in props:
			values['parentOrientation'] = region.orientation[pos]
			contactOffset = contactOffset.rotatedBy(values['parentOrientation'])

		values['position'] = (pos + contactOffset)

		return values

	def mod_helper(context, spec):
		new_pos = findOnHelper(region, context.position)
		contactOffset = Vector(0,0,context.contactTolerance) - context.centerOffset

		modifying_values = {'position': new_pos + contactOffset}

		if 'parentOrientation' in props:
			modifying_values['parentOrientation'] = region.orientation[new_pos]

		return modifying_values

	return ModifyingSpecifier(props, DelayedArgument({'centerOffset', 'contactTolerance'}, helper), DelayedArgument({'position'}, mod_helper))

@distributionFunction
def findOnHelper(region, pos):
	on_pos = region.findOn(pos)

	if on_pos is None:
		raise RejectionException("Unable to place object on surface.")
	else:
		return on_pos

def alwaysProvidesOrientation(region):
	"""Whether a Region or distribution over Regions always provides an orientation."""
	if isinstance(region, Region):
		return region.orientation is not None
	elif (isinstance(region, Options)
		  and all(alwaysProvidesOrientation(opt) for opt in region.options)):
		return True
	else:	# TODO improve somehow!
		try:
			sample = region.sample()
			return sample.orientation is not None or sample is nowhere
		except RejectionException:
			return False

def Beyond(pos, offset, fromPt=None):
	"""The 'beyond X by Y [from Z]' polymorphic specifier.

	Specifies 'position', with no dependencies.

	Allowed forms:
		beyond <vector> by <number> [from <vector>]
		beyond <vector> by <vector> [from <vector>]

	If the 'from <vector>' is omitted, the position of ego is used.
	"""
	# Ensure X can be coaxed into  vector form
	pos = toVector(pos, 'specifier "beyond X by Y" with X not a vector')

	# If no from vector is specified, assume ego
	if fromPt is None:
		fromPt = ego()

	fromPt = toVector(fromPt, 'specifier "beyond X by Y from Z" with Z not a vector')

	dType = underlyingType(offset)

	if dType is float or dType is int:
		offset = Vector(0, offset, 0)
	else:
		# offset is not float or int, so try to coax it into vector form.
		offset = toVector(offset, 'specifier "beyond X by Y" with X not a number or vector')

	# If the from vector is oriented, set that to orientation. Else assume global coords.
	if isinstance(fromPt, OrientedPoint):
		orientation = fromPt.orientation
	else:
		orientation = Orientation.fromEuler(0,0,0)

	# TODO: @Matthew Compute orientation along line of sight
	# TODO: @Matthew `val` pos.offsetRotated() should be helper function defining both position and parent orientation
	# as dictionary of values

	print("pos", pos)
	print("frompt", fromPt)
	print("offset", offset)

	direction = pos - fromPt
	sphericalCoords = direction.cartesianToSpherical()
	offsetRotation = Orientation.fromEuler(sphericalCoords[1], sphericalCoords[2], 0)

	new_direction = pos + offset.applyRotation(offsetRotation)

	# def helper(context, spec):
	# 	direction = pos - context.position
	# 	inverseQuat = context.parentOrientation.invertRotation()
	# 	rotated = direction.applyRotation(inverseQuat)
	# 	sphericalCoords = rotated.cartesianToSpherical() # Ignore the rho, sphericalCoords[0]
	# 	return {'yaw': sphericalCoords[1], 'pitch': sphericalCoords[2]}
	# return Specifier({'yaw': 1, 'pitch': 3}, DelayedArgument({'position', 'parentOrientation'}, helper))

	return Specifier({'position': 1, 'parentOrientation': 3},
	   				 {'position': new_direction, 'parentOrientation': orientation})

def VisibleFrom(base):
	"""The 'visible from <Point>' specifier.

	Specifies 'position', with no dependencies.

	This uses the given object's 'visibleRegion' property, and so correctly
	handles the view regions of Points, OrientedPoints, and Objects.
	"""
	if not isinstance(base, Point):
		raise RuntimeParseError('specifier "visible from O" with O not a Point')
	# TODO: @Matthew Generalize uniformPointIn() for 3D regions
	return Specifier({'position': 1}, {'position': Region.uniformPointIn(base.visibleRegion)})

def VisibleSpec():
	"""The 'visible' specifier (equivalent to 'visible from ego').

	Specifies 'position', with no dependencies.
	"""
	return VisibleFrom(ego())

def OffsetBy(offset):
	"""The 'offset by <vector>' specifier.

	Specifies 'position', with no dependencies.
	"""
	offset = toVector(offset, 'specifier "offset by X" with X not a vector')
	value = {'position': RelativeTo(offset, ego()).toVector(), 'parentOrientation': ego().parentOrientation}
	return Specifier({'position': 1, 'parentOrientation': 3}, value)

def OffsetAlongSpec(direction, offset):
	"""The 'offset along X by Y' polymorphic specifier.

	Specifies 'position', with no dependencies.

	Allowed forms:
		offset along <heading> by <vector>
		offset along <field> by <vector>
	"""
	pos = OffsetAlong(ego(), direction, offset)
	parentOrientation = ego().parentOrientation
	return Specifier({'position': 1, 'parentOrientation': 3},  {'position': pos, 'parentOrientation': parentOrientation})

def Facing(heading):
	"""The 'facing X' polymorphic specifier.

	Specifies yaw and pitch angles of 'heading', with dependencies depending on the form:
		facing <number> -- depends on 'parentOrientation'
		facing <field> -- depends on 'position', 'parentOrientation'
		facing <vector> -- depends on 'parentOrientation'
	"""
	# TODO: @Matthew Type check 'heading' to aovid IndexError 
	if isinstance(heading, VectorField):
		def helper(context, spec):
			headingAtPos = heading[context.position]
			if alwaysGlobalOrientation(context.parentOrientation):
				orientation = headingAtPos	# simplify expr tree in common case
			else:
				inverseQuat = context.parentOrientation.invertRotation()
				orientation = inverseQuat * headingAtPos
			return {'yaw': orientation.yaw, 'pitch': orientation.pitch, 'roll': orientation.roll}
			# return heading[context.position]
		return Specifier({'yaw': 1, 'pitch': 1, 'roll': 1}, DelayedArgument({'position', 'parentOrientation'}, helper))
	else:
		heading = toHeading(heading, "facing x with x not a heading")
		headingDeps = requiredProperties(heading)
		def helper(context, spec):
			nonlocal heading
			heading = valueInContext(heading, context)
			euler = context.parentOrientation.globalToLocalAngles(heading[0], heading[1], heading[2])
			return {'yaw': euler[0], 'pitch': euler[1], 'roll': euler[2]}
			# return toHeading(heading, 'specifier "facing X" with X not a heading or vector field')

		return Specifier({'yaw': 1, 'pitch': 1, 'roll': 1}, DelayedArgument({'parentOrientation'}|headingDeps, helper))

def FacingToward(pos):
	"""The 'facing toward <vector>' specifier.

	Specifies the yaw and pitch angle with priorities 1 and 3 respectively, depends on position.
	and 'pitch'.
	"""
	pos = toVector(pos, 'specifier "facing toward X" with X not a vector')
	def helper(context, spec):
		direction = pos - context.position
		inverseQuat = context.parentOrientation.invertRotation()
		rotated = direction.applyRotation(inverseQuat)
		sphericalCoords = rotated.cartesianToSpherical() # Ignore the rho, sphericalCoords[0]
		return {'yaw': sphericalCoords[1], 'pitch': sphericalCoords[2]}
	return Specifier({'yaw': 1, 'pitch': 3}, DelayedArgument({'position', 'parentOrientation'}, helper))

def FacingDirectlyToward(pos):
	"""The 'facing directly toward <vector>' specifier.

	Specifies yaw and pitch with priority 1, depends on position.
	"""
	pos = toVector(pos, 'specifier "facing directly toward X" with X not a vector')
	def helper(context, spec):
		'''
		Same process as above, except by default also specify the pitch euler angle 
		'''
		direction = pos - context.position
		inverseQuat = context.parentOrientation.invertRotation()
		rotated = direction.applyRotation(inverseQuat)
		sphericalCoords = rotated.cartestianToSpherical()
		return {'yaw': sphericalCoords[1], 'pitch': sphericalCoords[2]}
	return Specifier({'yaw': 1, 'pitch': 1}, DelayedArgument({'position', 'parentOrientation'}, helper))

def FacingAwayFrom(pos):
	""" The 'facing away from <vector>' specifier.

	Specifies yaw angle of 'heading', depending on 'position', 'roll',
	and 'pitch'.
	"""
	pos = toVector(pos, 'specifier "facing away from X" with X not a vector')
	def helper(context, spec):
		'''
		As in FacingToward, except invert the resulting rotation axis 
		'''
		direction = context.position - pos
		inverseQuat = context.parentOrientation.invertRotation()
		rotated = direction.applyRotation(inverseQuat)
		sphericalCoords = rotated.cartestianToSpherical()
		return {'yaw': sphericalCoords[1], 'pitch': sphericalCoords[2]}
		# return {'heading': pos.angleTo(context.position)}
	return Specifier({'yaw': 1, 'pitch': 3}, DelayedArgument({'position', 'parentOrientation'}, helper))

def FacingDirectlyAwayFrom(pos):
	"""The 'facing directly away from <vector>' specifier. 

	Specifies yaw and pitch angles of 'heading', depending on 'position' and 'roll'.
	"""
	pos = toVector(pos, 'specifier "facing away from X" with X not a vector')
	def helper(context, spec):
		direction = context.position - pos
		inverseQuat = context.parentOrientation.invertRotation()
		rotated = direction.applyRotation(inverseQuat)
		sphericalCoords = rotated.cartestianToSpherical()
		return {'yaw': sphericalCoords[1], 'pitch': sphericalCoords[2]}
	return Specifier({'yaw': 1, 'pitch': 1}, DelayedArgument({'position', 'parentOrientation'}, helper))

def ApparentlyFacing(heading, fromPt=None):
	"""The 'apparently facing <heading> [from <vector>]' specifier.

	Specifies 'pitch', 'roll', and 'yaw' of 'orientation', 'heading', depending 
	on 'position'.

	If the 'from <vector>' is omitted, the position of ego is used.
	"""
	def helper(context, spec):
		heading = toHeading(heading, 'specifier "apparently facing X" with X not a heading')
		if fromPt is None:
			fromPt = ego()
		fromPt = toVector(fromPt, 'specifier "apparently facing X from Y" with Y not a vector')
		value = lambda self: fromPt.angleTo(self.position) + heading # TODO: @Matthew Should not return lambda function
		return value 
	return Specifier({'yaw': 1, 'pitch': 1}, {'heading': DelayedArgument({'position', 'parentOrientation'}, helper)})

def LeftSpec(pos, dist=0, specs=None):
	"""The 'left of X [by Y]' polymorphic specifier.

	Specifies 'position', depending on 'width'. See other dependencies below.

	Allowed forms:
		left of <oriented point> [by <scalar/vector>] -- optionally specifies 'heading';
		left of <vector> [by <scalar/vector>] -- depends on 'heading'.

	If the 'by <scalar/vector>' is omitted, zero is used.
	"""
	return leftSpecHelper('left of', pos, dist, 'width', lambda dist: (dist, 0, 0),
						  lambda self, dims, tol, dx, dy, dz: Vector(-self.width/2 - dx - dims[0]/2 - tol, dy, dz))

def RightSpec(pos, dist=0):
	"""The 'right of X [by Y]' polymorphic specifier.

	Specifies 'position', depending on 'width'. See other dependencies below.

	Allowed forms:
		right of <oriented point> [by <scalar/vector>] -- optionally specifies 'heading';
		right of <vector> [by <scalar/vector>] -- depends on 'heading'.

	If the 'by <scalar/vector>' is omitted, zero is used.
	"""
	return leftSpecHelper('right of', pos, dist, 'width', lambda dist: (dist, 0, 0),
						  lambda self, dims, tol, dx, dy, dz: Vector(self.width / 2 + dx + dims[0]/2 + tol, dy, dz))

def Ahead(pos, dist=0):
	"""The 'ahead of X [by Y]' polymorphic specifier.

	Specifies 'position', depending on 'length'. See other dependencies below.

	Allowed forms:

	* ``ahead of`` <oriented point> [``by`` <scalar/vector>] -- optionally specifies 'heading';
	* ``ahead of`` <vector> [``by`` <scalar/vector>] -- depends on 'heading'.

	If the 'by <scalar/vector>' is omitted, zero is used.
	"""
	return leftSpecHelper('ahead of', pos, dist, 'length', lambda dist: (0, dist, 0),
						  lambda self, dims, tol, dx, dy, dz: Vector(dx, self.length / 2 + dy + dims[1]/2 + tol, dz))

def Behind(pos, dist=0):
	"""The 'behind X [by Y]' polymorphic specifier.

	Specifies 'position', depending on 'length'. See other dependencies below.

	Allowed forms:
		behind <oriented point> [by <scalar/vector>] -- optionally specifies 'heading';
		behind <vector> [by <scalar/vector>] -- depends on 'heading'.

	If the 'by <scalar/vector>' is omitted, zero is used.
	"""
	return leftSpecHelper('behind', pos, dist, 'length', lambda dist: (0, dist, 0),
						  lambda self, dims, tol, dx, dy, dz: Vector(dx, -self.length / 2 - dy - dims[1]/2 - tol, dz))

def Above(pos, dist=0):
	"""The 'above X [by Y]' polymorphic specifier.

	Specifies 'position', depending on 'height'. 

	Allowed forms:
		above <oriented point> [by <scalar/vector>] -- optionally specifies 'heading;
		above <vector> [by <scalar/vector>] -- depends on 'heading'.

	If the 'by <scalar/vector>' is omitted, zero is used.
	"""
	return leftSpecHelper('above', pos, dist, 'height', lambda dist: (0, 0, dist),
						  lambda self, dims, tol, dx, dy, dz: Vector(dx, dy, self.height / 2 + dz + dims[2]/2 + tol))

def Below(pos, dist=0):
	"""The 'below X [by Y]' polymorphic specifier.

	Specifies 'position', depending on 'height'.

	Allowed forms:
		below <oriented point> [by <scalar/vector>] -- optionally specifies 'heading;
		below <vector> [by <scalar/vector>] -- depends on 'heading'.

	If the 'by <scalar/vector>' is omitted, zero is used.
	"""
	return leftSpecHelper('above', pos, dist, 'height', lambda dist: (0, 0, dist),
						  lambda self, dims, tol, dx, dy, dz: Vector(dx, dy, -self.height / 2 - dz - dims[2]/2 - tol))

def leftSpecHelper(syntax, pos, dist, axis, toComponents, makeOffset):
	prop = {'position': 1}
	if canCoerce(dist, float):
		dx, dy, dz = toComponents(coerce(dist, float))
	elif canCoerce(dist, Vector):
		dx, dy, dz = coerce(dist, Vector)
	else:
		raise RuntimeParseError(f'"{syntax} X by D" with D not a number or vector')

	if isinstance(pos, Object):
		prop['parentOrientation'] = 3
		obj_dims = (pos.width, pos.length, pos.height)
		val = lambda self, spec: {
			'position': pos.relativize(makeOffset(self, obj_dims, self.contactTolerance, dx, dy, dz)),
			'parentOrientation': pos.orientation
		}
		new = DelayedArgument({axis, "contactTolerance"}, val)
	elif isinstance(pos, OrientedPoint):		# TODO too strict?
		prop['parentOrientation'] = 3
		val = lambda self, spec: {
			'position': pos.relativize(makeOffset(self, (0,0,0), 0, dx, dy, dz)),
			'parentOrientation': pos.orientation
		}
		new = DelayedArgument({axis}, val)
	else:
		pos = toVector(pos, f'specifier "{syntax} X" with X not a vector')
		val = lambda self, spec: {'position': pos.offsetRotated(self.orientation, makeOffset(self, (0,0,0), 0, dx, dy, dz))}
		new = DelayedArgument({axis, 'orientation'}, val)
	return Specifier(prop, new)

def Following(field, dist, fromPt=None):
	"""The 'following F [from X] for D' specifier.

	Specifies 'position', and optionally 'heading', with no dependencies.

	Allowed forms:
		following <field> [from <vector>] for <number>

	If the 'from <vector>' is omitted, the position of ego is used.
	"""
	if fromPt is None:
		fromPt = ego()
	else:
		dist, fromPt = fromPt, dist
	if not isinstance(field, VectorField):
		raise RuntimeParseError('"following F" specifier with F not a vector field')
	fromPt = toVector(fromPt, '"following F from X for D" with X not a vector')
	dist = toScalar(dist, '"following F for D" with D not a number')
	pos = field.followFrom(fromPt, dist)
	orientation = field[pos]
	return Specifier({'position': 1, 'parentOrientation': 3},
	                 {'position': pos, 'parentOrientation': orientation})
