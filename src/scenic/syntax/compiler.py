import ast
from typing import Optional, Tuple, List

import scenic.syntax.ast as s

# exposed functions


def compileScenicAST(scenicAST: ast.AST) -> Tuple[ast.AST, List[ast.AST]]:
    """Compiles Scenic AST to Python AST"""
    compiler = ScenicToPythonTransformer()
    node = ast.fix_missing_locations(compiler.visit(scenicAST))
    return node, compiler.requirements


# shorthands for convenience

loadCtx = ast.Load()
ego = ast.Name("ego")

noArgs = ast.arguments(
    posonlyargs=[],
    args=[],
    vararg=None,
    kwonlyargs=[],
    kw_defaults=[],
    kwarg=None,
    defaults=[],
)
selfArg = ast.arguments(
    posonlyargs=[],
    args=[ast.arg(arg="self", annotation=None)],
    vararg=None,
    kwonlyargs=[],
    kw_defaults=[],
    kwarg=None,
    defaults=[],
)

# helpers


class AttributeFinder(ast.NodeVisitor):
    """Utility class for finding all referenced attributes of a given name."""

    @staticmethod
    def find(target, node):
        af = AttributeFinder(target)
        af.visit(node)
        return af.attributes

    def __init__(self, target):
        super().__init__()
        self.target = target
        self.attributes = set()

    def visit_Attribute(self, node):
        val = node.value
        if isinstance(val, ast.Name) and val.id == self.target:
            self.attributes.add(node.attr)
        self.visit(val)


# transformer


class ScenicToPythonTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.requirements = []

    def generic_visit(self, node):
        if isinstance(node, s.AST):
            raise Exception(
                f'Scenic AST node "{node.__class__.__name__}" needs visitor in compiler'
            )
        return super().generic_visit(node)

    # helper functions
    def _register_requirement_syntax(self, syntax: ast.AST) -> int:
        self.requirements.append(syntax)
        return len(self.requirements) - 1

    def visit_Name(self, node: ast.Name) -> any:
        from scenic.syntax.translator import builtinNames, trackedNames

        if node.id in builtinNames:
            if not isinstance(node.ctx, ast.Load):
                raise SyntaxError(f'unexpected keyword "f{node.id}"')
        elif node.id in trackedNames:
            if not isinstance(node.ctx, ast.Load):
                raise SyntaxError(f'only simple assignments to "{node.id}" are allowed')
            node = ast.copy_location(ast.Call(ast.Name(node.id, loadCtx), [], []), node)
        # TODO(shun): Add handling for behavior locals
        return node

    # Special Case

    def visit_TrackedAssign(self, node: s.TrackedAssign):
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id=node.target.functionName, ctx=loadCtx),
                args=[self.visit(node.value)],
                keywords=[],
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> any:
        # use `Object` as base if none is specified
        if not node.bases:
            node.bases = [ast.Name("Object", loadCtx)]

        # extract all property definitions
        propertyDefs: list[s.PropertyDef] = []
        newBody = []
        for stmt in node.body:
            if isinstance(stmt, s.PropertyDef):
                propertyDefs.append(stmt)
            else:
                newBody.append(stmt)

        # create dictionary from property name (str) to default values
        propertyDict = {}
        for propertyDef in propertyDefs:
            if propertyDef.property in propertyDict:
                raise SyntaxError(f'duplicated property "{propertyDef.property}"')
            propertyDict[propertyDef.property] = propertyDef

        newBody.insert(
            0,
            ast.Assign(
                targets=[ast.Name(id="_scenic_properties", ctx=ast.Store())],
                value=ast.Dict(
                    keys=[ast.Constant(value=p) for p in propertyDict.keys()],
                    values=[
                        self.transformPropertyDef(v) for v in propertyDict.values()
                    ],
                ),
            ),
        )

        node.body = newBody
        return self.generic_visit(node)

    def transformPropertyDef(self, node: s.PropertyDef):
        properties = AttributeFinder.find("self", node.value)
        return ast.Call(
            func=ast.Name(id="PropertyDefault", ctx=ast.Load()),
            args=[
                ast.Set(elts=[ast.Constant(value=p) for p in properties]),
                ast.Set(
                    elts=[ast.Constant(value=attr.keyword) for attr in node.attributes]
                ),
                ast.Lambda(
                    args=selfArg,
                    body=self.visit(node.value),
                ),
            ],
            keywords=[],
        )

    def visit_PropertyDef(self, _: s.PropertyDef) -> any:
        assert False, "PropertyDef should be handled in `visit_ClassDef`"

    def visit_Model(self, node: s.Model):
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id="model", ctx=loadCtx),
                args=[
                    ast.Name(id="_Scenic_module_namespace", ctx=loadCtx),
                    ast.Constant(value=node.name),
                ],
                keywords=[],
            )
        )

    def visit_Mutate(self, node: s.Mutate):
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id="mutate", ctx=loadCtx),
                args=[self.visit(el) for el in node.elts],
                keywords=[ast.keyword(arg="scale", value=self.visit(node.scale))]
                if node.scale is not None
                else [],
            )
        )

    def visit_Param(self, node: s.Param):
        d = dict()
        for parameter in node.elts:
            if parameter.identifier in d:
                raise SyntaxError(f'Duplicated param "{parameter.identifier}"')
            d[parameter.identifier] = self.visit(parameter.value)
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id="param", ctx=loadCtx),
                args=[
                    ast.Dict(
                        [ast.Constant(k) for k in d.keys()],
                        list(d.values()),
                    )
                ],
                keywords=[],
            )
        )

    def visit_Require(self, node: s.Require):
        condition = self.visit(node.cond)
        syntax_id = self._register_requirement_syntax(condition)

        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id="require", ctx=loadCtx),
                args=[
                    ast.Constant(syntax_id),
                    ast.Lambda(
                        args=noArgs,
                        body=condition,
                    ),
                    ast.Constant(value=node.lineno),
                    ast.Constant(value=node.name),
                ],
                keywords=[ast.keyword(arg="prob", value=ast.Constant(node.prob))]
                if node.prob is not None
                else [],
            )
        )

    # Instance & Specifier

    def visit_New(self, node: s.New):
        return ast.Call(
            func=ast.Name(id="new", ctx=loadCtx),
            args=[
                ast.Name(id=node.className, ctx=loadCtx),
                ast.List(elts=[self.visit(s) for s in node.specifiers], ctx=ast.Load()),
            ],
            keywords=[],
        )

    def visit_WithSpecifier(self, node: s.WithSpecifier):
        return ast.Call(
            func=ast.Name(id="With", ctx=loadCtx),
            args=[
                ast.Constant(value=node.prop),
                self.visit(node.value),
            ],
            keywords=[],
        )

    def visit_AtSpecifier(self, node: s.AtSpecifier):
        return ast.Call(
            func=ast.Name(id="At", ctx=loadCtx),
            args=[self.visit(node.position)],
            keywords=[],
        )

    def visit_OffsetBySpecifier(self, node: s.OffsetBySpecifier):
        return ast.Call(
            func=ast.Name(id="OffsetBy", ctx=loadCtx),
            args=[
                self.visit(node.offset),
            ],
            keywords=[],
        )

    def visit_OffsetAlongSpecifier(self, node: s.OffsetAlongSpecifier):
        return ast.Call(
            func=ast.Name(id="OffsetAlongSpec", ctx=loadCtx),
            args=[
                self.visit(node.direction),
                self.visit(node.offset),
            ],
            keywords=[],
        )

    def visit_DirectionOfSpecifier(self, node: s.DirectionOfSpecifier):
        if isinstance(node.direction, s.LeftOf):
            fn = "LeftSpec"
        elif isinstance(node.direction, s.RightOf):
            fn = "RightSpec"
        elif isinstance(node.direction, s.AheadOf):
            fn = "Ahead"
        elif isinstance(node.direction, s.Behind):
            fn = "Behind"
        else:
            assert False, f"impossible direction {node.direction} in PositionSpecifier"
        return ast.Call(
            func=ast.Name(id=fn, ctx=loadCtx),
            args=[
                self.visit(node.position),
            ],
            keywords=(
                []
                if node.distance is None
                else [ast.keyword(arg="dist", value=self.visit(node.distance))]
            ),
        )

    def visit_BeyondSpecifier(self, node: s.BeyondSpecifier):
        return ast.Call(
            func=ast.Name(id="Beyond", ctx=loadCtx),
            args=[self.visit(node.position), self.visit(node.offset)],
            keywords=[ast.keyword(arg="fromPt", value=self.visit(node.base))]
            if node.base is not None
            else [],
        )

    def visit_VisibleSpecifier(self, node: s.VisibleSpecifier):
        if node.base is not None:
            return ast.Call(
                func=ast.Name(id="VisibleFrom", ctx=loadCtx),
                args=[self.visit(node.base)],
                keywords=[],
            )
        return ast.Call(
            func=ast.Name(id="VisibleSpec", ctx=loadCtx),
            args=[],
            keywords=[],
        )

    def visit_NotVisibleSpecifier(self, node: s.NotVisibleSpecifier):
        if node.base is not None:
            return ast.Call(
                func=ast.Name(id="NotVisibleFrom", ctx=loadCtx),
                args=[self.visit(node.base)],
                keywords=[],
            )
        return ast.Call(
            func=ast.Name(id="NotVisibleSpec", ctx=loadCtx),
            args=[],
            keywords=[],
        )

    def visit_InSpecifier(self, node: s.InSpecifier):
        return ast.Call(
            func=ast.Name(id="In", ctx=loadCtx),
            args=[self.visit(node.region)],
            keywords=[],
        )

    def visit_FollowingSpecifier(self, node: s.FollowingSpecifier):
        return ast.Call(
            func=ast.Name(id="Following", ctx=loadCtx),
            args=[self.visit(node.field), self.visit(node.distance)],
            keywords=[ast.keyword(arg="fromPt", value=self.visit(node.base))]
            if node.base is not None
            else [],
        )

    def visit_FacingSpecifier(self, node: s.FacingSpecifier):
        return ast.Call(
            func=ast.Name(id="Facing", ctx=loadCtx),
            args=[self.visit(node.heading)],
            keywords=[],
        )

    def visit_FacingTowardSpecifier(self, node: s.FacingTowardSpecifier):
        return ast.Call(
            func=ast.Name(id="FacingToward", ctx=loadCtx),
            args=[self.visit(node.position)],
            keywords=[],
        )

    def visit_ApparentlyFacingSpecifier(self, node: s.ApparentlyFacingSpecifier):
        return ast.Call(
            func=ast.Name(id="ApparentlyFacing", ctx=loadCtx),
            args=[self.visit(node.heading)],
            keywords=[ast.keyword(arg="fromPt", value=self.visit(node.base))]
            if node.base is not None
            else [],
        )

    # Operators

    def visit_RelativePositionOp(self, node: s.RelativePositionOp):
        return ast.Call(
            func=ast.Name(id="RelativePosition", ctx=loadCtx),
            args=[self.visit(node.target)],
            keywords=[]
            if node.base is None
            else [ast.keyword(arg="Y", value=self.visit(node.base))],
        )

    def visit_RelativeHeadingOp(self, node: s.RelativeHeadingOp):
        return ast.Call(
            func=ast.Name(id="RelativeHeading", ctx=loadCtx),
            args=[self.visit(node.target)],
            keywords=[]
            if node.base is None
            else [ast.keyword(arg="Y", value=self.visit(node.base))],
        )

    def visit_ApparentHeadingOp(self, node: s.ApparentHeadingOp):
        return ast.Call(
            func=ast.Name(id="ApparentHeading", ctx=loadCtx),
            args=[self.visit(node.target)],
            keywords=[]
            if node.base is None
            else [ast.keyword(arg="Y", value=self.visit(node.base))],
        )

    def visit_DistanceFromOp(self, node: s.DistanceFromOp):
        return ast.Call(
            func=ast.Name(id="DistanceFrom", ctx=loadCtx),
            args=[self.visit(node.target)],
            keywords=[ast.keyword(arg="Y", value=self.visit(node.base))]
            if node.base is not None
            else [],
        )

    def visit_DistancePastOp(self, node: s.DistancePastOp):
        return ast.Call(
            func=ast.Name(id="DistancePast", ctx=loadCtx),
            args=[self.visit(node.target)],
            keywords=[]
            if node.base is None
            else [ast.keyword(arg="Y", value=self.visit(node.base))],
        )

    def visit_AngleFromOp(self, node: s.AngleFromOp):
        assert (
            node.base is not None or node.target is not None
        ), "neither target nor base were specified in AngleFromOp"
        keywords = []
        if node.base is not None:
            keywords.append(ast.keyword("X", self.visit(node.base)))
        if node.target is not None:
            keywords.append(ast.keyword("Y", self.visit(node.target)))
        return ast.Call(
            func=ast.Name(id="AngleFrom", ctx=loadCtx),
            args=[],
            keywords=keywords,
        )

    def visit_FollowOp(self, node: s.FollowOp):
        return ast.Call(
            func=ast.Name(id="Follow", ctx=loadCtx),
            args=[
                self.visit(node.target),
                self.visit(node.base),
                self.visit(node.distance),
            ],
            keywords=[],
        )

    def visit_VisibleOp(self, node: s.VisibleOp):
        return ast.Call(
            func=ast.Name(id="Visible", ctx=loadCtx),
            args=[self.visit(node.region)],
            keywords=[],
        )

    def visit_NotVisibleOp(self, node: s.VisibleOp):
        return ast.Call(
            func=ast.Name(id="NotVisible", ctx=loadCtx),
            args=[self.visit(node.region)],
            keywords=[],
        )

    def visit_PositionOfOp(self, node: s.PositionOfOp):
        return ast.Call(
            func=ast.Name(id=node.position.functionName, ctx=loadCtx),
            args=[self.visit(node.target)],
            keywords=[],
        )

    def visit_DegOp(self, node: s.DegOp):
        return ast.BinOp(
            left=self.visit(node.operand), op=ast.Mult(), right=ast.Constant(0.01745329)
        )

    def visit_VectorOp(self, node: s.VectorOp):
        return ast.Call(
            func=ast.Name(id="Vector", ctx=loadCtx),
            args=[self.visit(node.left), self.visit(node.right)],
            keywords=[],
        )

    def visit_FieldAtOp(self, node: s.FieldAtOp):
        return ast.Call(
            func=ast.Name(id="FieldAt", ctx=loadCtx),
            args=[self.visit(node.left), self.visit(node.right)],
            keywords=[],
        )

    def visit_RelativeToOp(self, node: s.RelativeToOp):
        return ast.Call(
            func=ast.Name(id="RelativeTo", ctx=loadCtx),
            args=[self.visit(node.left), self.visit(node.right)],
            keywords=[],
        )

    def visit_OffsetAlongOp(self, node: s.OffsetAlongOp):
        return ast.Call(
            func=ast.Name(id="OffsetAlong", ctx=loadCtx),
            args=[
                self.visit(node.base),
                self.visit(node.direction),
                self.visit(node.offset),
            ],
            keywords=[],
        )

    def visit_CanSeeOp(self, node: s.CanSeeOp):
        return ast.Call(
            func=ast.Name(id="CanSee", ctx=loadCtx),
            args=[
                self.visit(node.left),
                self.visit(node.right),
            ],
            keywords=[],
        )
