"""Specifiers and associated objects."""

import itertools

from scenic.core.lazy_eval import (DelayedArgument, valueInContext, requiredProperties,
                                   needsLazyEvaluation)
from scenic.core.distributions import toDistribution
from scenic.core.errors import RuntimeParseError

## Specifiers themselves

class Specifier:
	"""Specifier providing a value for a property given dependencies.

	Any optionally-specified properties are evaluated as attributes of the primary value.
	"""
	def __init__(self, priorities, value, deps=None, internal=False):
		if not isinstance(priorities, dict):
			priorities = {priorities: -1}
		self.priorities = priorities
		self.value = value
		self.modifying = dict()
		if deps is None:
			deps = set()
		deps |= requiredProperties(self.value)
		for p in priorities:
			if p in deps:
				raise RuntimeParseError(f'specifier for property {p} depends on itself')
		self.requiredProperties = deps

	def applyTo(self, obj, modifying):
		"""Apply specifier to an object"""
		val = valueInContext(self.value, obj, modifying)

		if isinstance(val, dict):
			for prop in modifying:
				print(modifying)
				distV = toDistribution(val[prop])
				assert not needsLazyEvaluation(distV)
				obj._specify(prop, distV)
		else:
			assert len(self.priorities) == 1
			val = toDistribution(val)
			assert not needsLazyEvaluation(val)
			for prop in self.priorities:
				obj._specify(prop, val)

	def __str__(self):
		return f'<Specifier of {self.priorities}>'

class ModifyingSpecifier(Specifier):
	pass

## Support for property defaults

class PropertyDefault:
	"""A default value, possibly with dependencies."""
	def __init__(self, requiredProperties, attributes, value):
		self.requiredProperties = set(requiredProperties)
		self.value = value

		def enabled(thing, default):
			if thing in attributes:
				attributes.remove(thing)
				return True
			else:
				return default
		self.isAdditive = enabled('additive', False)
		self.isDynamic = enabled('dynamic', False)
		self.isFinal = enabled('final', False)
		for attr in attributes:
			raise RuntimeParseError(f'unknown property attribute "{attr}"')

	@staticmethod
	def forValue(value):
		if isinstance(value, PropertyDefault):
			return value
		else:
			return PropertyDefault(set(), set(), lambda self, modifying: value)

	def resolveFor(self, prop, overriddenDefs):
		"""Create a Specifier for a property from this default and any superclass defaults."""
		for other in overriddenDefs:
			if other.isFinal:
				raise RuntimeParseError(f'"{prop}" property cannot be overridden')
		if self.isAdditive:
			allReqs = self.requiredProperties
			for other in overriddenDefs:
				allReqs |= other.requiredProperties
			def concatenator(context):
				allVals = [self.value(context)]
				for other in overriddenDefs:
					allVals.append(other.value(context))
				return tuple(allVals)
			val = DelayedArgument(allReqs, concatenator, _internal=True) # TODO: @Matthew Change to dicts
		else:
			val = DelayedArgument(self.requiredProperties, self.value, _internal=True)
		return Specifier({prop: -1}, val)
