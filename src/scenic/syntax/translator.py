
"""Translator turning Scenic programs into Scenario objects.

The top-level interface to Scenic is provided by two functions:

* `scenarioFromString` -- compile a string of Scenic code;
* `scenarioFromFile` -- compile a Scenic file.

These output a `Scenario` object, from which scenes can be generated.
See the documentation for `Scenario` for details.

When imported, this module hooks the Python import system so that Scenic
modules can be imported using the ``import`` statement. This is primarily for the
translator's own use, but you could import Scenic modules from Python to
inspect them. [#import]_ Because Scenic uses Python's import system, the latter's
rules for finding modules apply, including the handling of packages.

Scenic is compiled in two main steps: translating the code into Python, and
executing the resulting Python module to generate a Scenario object encoding
the objects, distributions, etc. in the scenario. For details, see the function
`compileStream` below.

.. rubric:: Footnotes

.. [#import] Note however that care must be taken when importing Scenic modules
	which will later be used when compiling multiple Scenic scenarios. Because
	Python caches modules, there is the possibility of one version of a Scenic
	module persisting even when it should be recompiled during the compilation
	of another module that imports it.
	Scenic handles the most common case, that of Scenic modules which refer to
	other Scenic modules at the top level; but it is not practical to catch all
	possible cases. In particular, importing a Python package which contains
	Scenic modules as submodules and then later compiling those modules more
	than once within the same Python process may lead to errors or unexpected
	behavior. See the **cacheImports** argument of `scenarioFromFile`.
"""

import sys
import os
import io
import builtins
import time
import types
import importlib
import importlib.abc
import importlib.util
from contextlib import contextmanager

import ast

from scenic.core.distributions import RejectionException, toDistribution
from scenic.core.lazy_eval import needsLazyEvaluation
import scenic.core.errors as errors
from scenic.core.errors import (PythonParseError, InvalidScenarioError)
import scenic.core.dynamics as dynamics
import scenic.core.pruning as pruning
import scenic.syntax.veneer as veneer
from scenic.syntax.parser import parse_string
from scenic.syntax.compiler import compileScenicAST

### THE TOP LEVEL: compiling a Scenic program

def scenarioFromString(string, params={}, model=None, scenario=None,
					   filename='<string>', cacheImports=False):
	"""Compile a string of Scenic code into a `Scenario`.

	The optional **filename** is used for error messages.
	Other arguments are as in `scenarioFromFile`.
	"""
	stream = io.BytesIO(string.encode())
	return scenarioFromStream(stream, params=params, model=model, scenario=scenario,
							  filename=filename, cacheImports=cacheImports)

def scenarioFromFile(path, params={}, model=None, scenario=None, cacheImports=False):
	"""Compile a Scenic file into a `Scenario`.

	Args:
		path (str): Path to a Scenic file.
		params (dict): Global parameters to override, as a dictionary mapping
		  parameter names to their desired values.
		model (str): Scenic module to use as :term:`world model`.
		scenario (str): If there are multiple :term:`modular scenarios` in the
		  file, which one to compile; if not specified, a scenario called 'Main'
		  is used if it exists.
		cacheImports (bool): Whether to cache any imported Scenic modules.
		  The default behavior is to not do this, so that subsequent attempts
		  to import such modules will cause them to be recompiled. If it is
		  safe to cache Scenic modules across multiple compilations, set this
		  argument to True. Then importing a Scenic module will have the same
		  behavior as importing a Python module. See `purgeModulesUnsafeToCache`
		  for a more detailed discussion of the internals behind this.

	Returns:
		A `Scenario` object representing the Scenic scenario.
	"""
	if not os.path.exists(path):
		raise FileNotFoundError(path)
	fullpath = os.path.realpath(path)
	head, extension = os.path.splitext(fullpath)
	if not extension or extension[1:] not in scenicExtensions:
		ok = ', '.join(scenicExtensions)
		err = f'Scenic scenario does not have valid extension ({ok})'
		raise RuntimeError(err)
	directory, name = os.path.split(head)

	with open(path, 'rb') as stream:
		return scenarioFromStream(stream, params=params, model=model, scenario=scenario,
								  filename=fullpath, path=path, cacheImports=cacheImports)

def scenarioFromStream(stream, params={}, model=None, scenario=None,
					   filename='<stream>', path=None, cacheImports=False):
	"""Compile a stream of Scenic code into a `Scenario`."""
	# Compile the code as if it were a top-level module
	oldModules = list(sys.modules.keys())
	try:
		with topLevelNamespace(path) as namespace:
			compileStream(stream, namespace, params=params, model=model, filename=filename)
	finally:
		if not cacheImports:
			purgeModulesUnsafeToCache(oldModules)
	# Construct a Scenario from the resulting namespace
	return constructScenarioFrom(namespace, scenario)

@contextmanager
def topLevelNamespace(path=None):
	"""Creates an environment like that of a Python script being run directly.

	Specifically, __name__ is '__main__', __file__ is the path used to invoke
	the script (not necessarily its absolute path), and the parent directory is
	added to the path so that 'import blobbo' will import blobbo from that
	directory if it exists there.
	"""
	directory = os.getcwd() if path is None else os.path.dirname(path)
	namespace = { '__name__': '__main__' }
	if path is not None:
		namespace['__file__'] = path
	sys.path.insert(0, directory)
	try:
		yield namespace
	finally:
		# Remove directory from sys.path, being a little careful in case the
		# Scenic program modified it (unlikely but possible).
		try:
			sys.path.remove(directory)
		except ValueError:
			pass

def purgeModulesUnsafeToCache(oldModules):
	"""Uncache loaded modules which should not be kept after compilation.

	Keeping Scenic modules in `sys.modules` after compilation will cause
	subsequent attempts at compiling the same module to reuse the compiled
	scenario: this is usually not what is desired, since compilation can depend
	on external state (in particular overridden global parameters, used e.g. to
	specify the map for driving domain scenarios).

	Args:
		oldModules: List of names of modules loaded before compilation. These
			will be skipped.
	"""
	toRemove = []
	# copy sys.modules in case it mutates during iteration (actually happens!)
	for name, module in sys.modules.copy().items():
		if isinstance(module, ScenicModule) and name not in oldModules:
			toRemove.append(name)
	for name in toRemove:
		parent, _, child = name.rpartition('.')
		parent = sys.modules.get(parent)
		if parent:
			# Remove reference to purged module from parent module. This is necessary
			# so that future imports of the purged module will properly refer to the
			# newly-loaded version of it. See below for a long disquisition on this.
			del parent.__dict__[child]

			# Here are details on why the above line is necessary and the sorry history
			# of my attempts to fix this type of bug (hopefully this note will prevent
			# further self-sabotage). Suppose we have a Python package 'package'
			# with a Scenic submodule 'submodule'. A Scenic program with the line
			#	from package import submodule
			# will import 2 packages, namely package and package.submodule, when first
			# compiled. We will then purge package.submodule from sys.modules, but not
			# package, since it is an ordinary module. So if the program is compiled a
			# second time, the line above will NOT import package.submodule, but simply
			# access the attribute 'submodule' of the existing package 'package'. So the
			# reference to the old version of package.submodule will leak out.
			# (An alternative approach, which I used to use, would be to purge all
			# modules containing even indirect references to Scenic modules, but this
			# opens a can of worms: the implementation of
			#	import parent.child
			# does not set the 'child' attribute of 'parent' if 'parent.child' is already
			# in sys.modules, violating an invariant that Python expects [see
			# https://docs.python.org/3/reference/import.html#submodules] and leading to
			# confusing errors. So if parent is purged because it has some child which is
			# a Scenic module, *all* of its children must then be purged. Since the
			# scenic module itself can contain indirect references to Scenic modules (the
			# world models), this means we have to purge the entire scenic package. But
			# then whoever did 'import scenic' at the top level will be left with a
			# reference to the old version of the Scenic module.)
		del sys.modules[name]

def compileStream(stream, namespace, params={}, model=None, filename='<stream>'):
	"""Compile a stream of Scenic code and execute it in a namespace.

	The compilation procedure consists of the following main steps:

		1. Tokenize the input using the Python tokenizer.
		2. Partition the tokens into blocks separated by import statements.
		   This is done by the `partitionByImports` function.
		3. Translate Scenic constructions into valid Python syntax.
		   This is done by the `TokenTranslator`.
		4. Parse the resulting Python code into an AST using the Python parser.
		5. Modify the AST to achieve the desired semantics for Scenic.
		   This is done by the `translateParseTree` function.
		6. Compile and execute the modified AST.
		7. After executing all blocks, extract the global state (e.g. objects).
		   This is done by the `storeScenarioStateIn` function.
	"""
	if verbosity >= 2:
		veneer.verbosePrint(f'  Compiling Scenic module from {filename}...')
		startTime = time.time()
	veneer.activate(params, model, filename, namespace)
	newSourceBlocks = []
	try:
		# Execute preamble
		exec(compile(preamble, '<veneer>', 'exec'), namespace)
		namespace[namespaceReference] = namespace

		# Parse the translated source
		source = stream.read().decode('utf-8')
		scenic_tree = parse_string(source, "exec", filename=filename)

		if dumpScenicAST:
			print(f'### Begin Scenic AST of {filename}')
			print(ast.dump(scenic_tree, include_attributes=False, indent=4))
			print('### End Scenic AST')

		tree, requirements = compileScenicAST(scenic_tree, filename=filename)

		if dumpFinalAST:
			print(f'### Begin final AST of {filename}')
			print(ast.dump(tree, include_attributes=True, indent=4))
			print('### End final AST')

		if dumpASTPython:
			try:
				import astor
			except ModuleNotFoundError as e:
				raise RuntimeError('dumping the Python equivalent of the AST'
									'requires the astor package')
			print(f'### Begin Python equivalent of final AST of {filename}')
			print(astor.to_source(tree, add_line_information=True))
			print('### End Python equivalent of final AST')

		# Compile the modified tree
		code = compileTranslatedTree(tree, filename)

		# Execute it
		executeCodeIn(code, namespace)

		# Extract scenario state from veneer and store it
		storeScenarioStateIn(namespace, requirements)
	finally:
		veneer.deactivate()
	if verbosity >= 2:
		totalTime = time.time() - startTime
		veneer.verbosePrint(f'  Compiled Scenic module in {totalTime:.4g} seconds.')
	allNewSource = ''.join(newSourceBlocks)
	return code, allNewSource

### TRANSLATION PHASE ZERO: definitions of language elements not already in Python

## Options

dumpScenicAST = False
dumpFinalAST = False
dumpASTPython = False
verbosity = 0
usePruning = True

## Preamble
# (included at the beginning of every module to be translated;
# imports the implementations of the public language features)
preamble = """\
from scenic.syntax.veneer import *
"""

## Get Python names of various elements
## (for checking consistency between the translator and the veneer)

api = set(veneer.__all__)

namespaceReference = '_Scenic_module_namespace'		# used in the implementation of 'model'

### TRANSLATION PHASE ONE: handling imports

## Meta path finder and loader for Scenic files

scenicExtensions = ('scenic', 'sc')

class ScenicMetaFinder(importlib.abc.MetaPathFinder):
	def find_spec(self, name, paths, target=None):
		if paths is None:
			paths = sys.path
			modname = name
		else:
			modname = name.rpartition('.')[2]
		for path in paths:
			for extension in scenicExtensions:
				filename = modname + '.' + extension
				filepath = os.path.join(path, filename)
				if os.path.exists(filepath):
					filepath = os.path.abspath(filepath)
					spec = importlib.util.spec_from_file_location(name, filepath,
						loader=ScenicLoader(filepath, filename))
					return spec
		return None

class ScenicLoader(importlib.abc.InspectLoader):
	def __init__(self, filepath, filename):
		self.filepath = filepath
		self.filename = filename

	def create_module(self, spec):
		return ScenicModule(spec.name)

	def exec_module(self, module):
		# Read source file and compile it
		with open(self.filepath, 'r') as stream:
			source = stream.read()
		with open(self.filepath, 'rb') as stream:
			code, pythonSource = compileStream(stream, module.__dict__, filename=self.filepath)
		# Save code, source, and translated source for later inspection
		module._code = code
		module._source = source
		module._pythonSource = pythonSource

		# If we're in the process of compiling another Scenic module, inherit
		# objects, parameters, etc. from this one
		if veneer.isActive():
			veneer.currentScenario._inherit(module._scenario)

	def is_package(self, fullname):
		return False

	def get_code(self, fullname):
		module = importlib.import_module(fullname)
		assert isinstance(module, ScenicModule), module
		return module._code

	def get_source(self, fullname):
		module = importlib.import_module(fullname)
		assert isinstance(module, ScenicModule), module
		return module._pythonSource

class ScenicModule(types.ModuleType):
	def __getstate__(self):
		state = self.__dict__.copy()
		del state['__builtins__']
		return (self.__name__, state)

	def __setstate__(self, state):
		name, state = state
		self.__init__(name)		# needed to create __dict__
		self.__dict__.update(state)
		self.__builtins__ = builtins.__dict__

# register the meta path finder
sys.meta_path.insert(0, ScenicMetaFinder())

### TRANSLATION PHASE FIVE: AST compilation

def compileTranslatedTree(tree, filename):
	try:
		return compile(tree, filename, 'exec')
	except SyntaxError as e:
		raise PythonParseError(e) from None

### TRANSLATION PHASE SIX: Python execution

def executeCodeIn(code, namespace):
	"""Execute the final translated Python code in the given namespace."""
	try:
		exec(code, namespace)
	except RejectionException as e:
		# Determined statically that the scenario has probability zero.
		errors.optionallyDebugRejection(e)
		if errors.showInternalBacktrace:
			raise InvalidScenarioError(e.args[0]) from e
		else:
			raise InvalidScenarioError(e.args[0]).with_traceback(e.__traceback__) from None

### TRANSLATION PHASE SEVEN: scenario construction

def storeScenarioStateIn(namespace, requirementSyntax):
	"""Post-process an executed Scenic module, extracting state from the veneer."""

	# Save requirement syntax and other module-level information
	moduleScenario = veneer.currentScenario
	factory = veneer.simulatorFactory
	bns = gatherBehaviorNamespacesFrom(moduleScenario._behaviors)
	def handle(scenario):
		scenario._requirementSyntax = requirementSyntax
		if isinstance(scenario, type):
			scenario._simulatorFactory = staticmethod(factory)
		else:
			scenario._simulatorFactory = factory
		scenario._behaviorNamespaces = bns
	handle(moduleScenario)
	namespace['_scenarios'] = tuple(veneer.scenarios)
	for scenarioClass in veneer.scenarios:
		handle(scenarioClass)

	# Extract requirements, scan for relations used for pruning, and create closures
	# (only for top-level scenario; modular scenarios will be handled when instantiated)
	moduleScenario._compileRequirements()

	# Save global parameters
	for name, value in veneer._globalParameters.items():
		if needsLazyEvaluation(value):
			raise InvalidScenarioError(f'parameter {name} uses value {value}'
									   ' undefined outside of object definition')
	for scenario in veneer.scenarios:
		scenario._bindGlobals(veneer._globalParameters)
	moduleScenario._bindGlobals(veneer._globalParameters)

	# Save workspace
	namespace['_workspace'] = veneer._workspace

	namespace['_scenario'] = moduleScenario

def gatherBehaviorNamespacesFrom(behaviors):
	"""Gather any global namespaces which could be referred to by behaviors.

	We'll need to rebind any sampled values in them at runtime.
	"""
	behaviorNamespaces = {}
	def registerNamespace(modName, ns):
		oldNS = behaviorNamespaces.get(modName)
		if oldNS:
			# Already registered; just do a consistency check to avoid bizarre
			# bugs from having multiple versions of the same module around.
			if oldNS is not ns:
				raise RuntimeError(
				    f'scenario refers to multiple versions of module {modName}; '
				    'perhaps you imported it before you started compilation?')
			return
		behaviorNamespaces[modName] = ns
		for name, value in ns.items():
			if isinstance(value, ScenicModule):
				registerNamespace(value.__name__, value.__dict__)
			else:
				# Convert values requiring sampling to Distributions
				dval = toDistribution(value)
				if dval is not value:
					ns[name] = dval
	for behavior in behaviors:
		modName = behavior.__module__
		globalNamespace = behavior.makeGenerator.__globals__
		registerNamespace(modName, globalNamespace)
	return behaviorNamespaces

def constructScenarioFrom(namespace, scenarioName=None):
	"""Build a Scenario object from an executed Scenic module."""
	modularScenarios = namespace['_scenarios']
	def isModularScenario(thing):
		return isinstance(thing, type) and issubclass(thing, dynamics.DynamicScenario)
	if not scenarioName and isModularScenario(namespace.get('Main', None)):
		scenarioName = 'Main'
	if scenarioName:
		ty = namespace.get(scenarioName, None)
		if not isModularScenario(ty):
			raise RuntimeError(f'no scenario "{scenarioName}" found')
		if ty._requiresArguments():
			raise RuntimeError(f'cannot instantiate scenario "{scenarioName}"'
			                   ' with no arguments') from None

		dynScenario = ty()
	elif len(modularScenarios) > 1:
		raise RuntimeError('multiple choices for scenario to run '
		                   '(specify using the --scenario option)')
	elif modularScenarios and not modularScenarios[0]._requiresArguments():
		dynScenario = modularScenarios[0]()
	else:
		dynScenario = namespace['_scenario']

	if not dynScenario._prepared:	# true for all except top-level scenarios
		# Execute setup block (if any) to create objects and requirements;
		# extract any requirements and scan for relations used for pruning
		dynScenario._prepare(delayPreconditionCheck=True)
	scenario = dynScenario._toScenario(namespace)

	# Prune infeasible parts of the space
	if usePruning:
		pruning.prune(scenario, verbosity=verbosity)

	return scenario
