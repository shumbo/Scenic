"""Objects representing propositions that can be used to specify conditions"""

from functools import reduce
import operator


class PropositionNode:
	"""Base class for temporal and non-temporal propositions"""
	def __init__(self, requirementSyntaxId = None) -> None:
		self.requirementSyntaxId = requirementSyntaxId

	@property
	def is_temporal(self) -> bool:
		"""checks if the proposition is temporal or not

		Returns:
			bool: True if the proposition is temporal, False otherwise
		"""
		return False

	@property
	def constrains_sampling(self):
		"""checks if the proposition can be used for pruning

		A requirement can be used for pruning if it is evaluated on the scene generation phase before simulation, and
		violation in that phase immediately results in discarding the scene and regenerating a new one.

		For simplicity, we currently check two special cases:
		1. requirements with no temporal requirement
		2. requirements with only one `always` operator on top-level

		Returns:
			bool: True if the requirement can be used for pruning, False otherwise
		"""
		node = self

		# if `always` is on top-level, check what's inside
		if isinstance(node, Always):
			node = node.req

		eligible = True
		for node in self.flatten():
			# turns false if any one of the node is temporal
			eligible = eligible and (not node.is_temporal)
		return eligible
	
	@property
	def children(self) -> list["PropositionNode"]:
		"""returns all children of proposition tree

		Returns:
			list: proposition nodes that are directly under this node
		"""
		return []

	def flatten(self):
		"""flattens the tree and return the list of nodes

		Returns:
			list: list of all children nodes
		"""
		return [self] + reduce(
			operator.concat, [node.flatten() for node in self.children], []
		)

class Atomic(PropositionNode):
	def __init__(self, closure, requirementSyntaxId = None):
		super().__init__(requirementSyntaxId)
		self.closure = closure
	def __str__(self):
		return f"(AP)"

class UnaryProposition(PropositionNode):
	"""Base class for temporal unary operators"""
	def __init__(self, req, requirementSyntaxId = None):
		super().__init__(requirementSyntaxId)
		self.req = req

	@property
	def children(self):
		return [self.req]

class Always(UnaryProposition):
	def __str__(self):
		return f"(Always {str(self.req)})"

class Eventually(UnaryProposition):
	def __str__(self):
		return f"(Eventually {str(self.req)})"

class Next(UnaryProposition):
	def __str__(self):
		return f"(Next {str(self.req)})"

class Not(UnaryProposition):
	def __str__(self):
		return f"(Not {str(self.req)})"

class And(PropositionNode):
	def __init__(self, reqs, requirementSyntaxId):
		super().__init__(requirementSyntaxId)
		self.reqs = reqs
	def __str__(self):
		return " and ".join([f"{str(req)}" for req in self.reqs])
	@property
	def children(self):
		return self.reqs

class Or(PropositionNode):
	def __init__(self, reqs, requirementSyntaxId):
		super().__init__(requirementSyntaxId)
		self.reqs = reqs
	def __str__(self):
		return " or ".join([f"{str(req)}" for req in self.reqs])
	@property
	def children(self):
		return self.reqs