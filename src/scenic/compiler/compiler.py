import ast
import scenic.ast as s

loadCtx = ast.Load()


def toPythonAST(scenicAST):
    return ast.fix_missing_locations(ScenicToPythonTransformer().visit(scenicAST))

selfArg = ast.arguments(
	posonlyargs=[],
	args=[ast.arg(arg='self', annotation=None)], vararg=None,
	kwonlyargs=[], kw_defaults=[],
	kwarg=None, defaults=[])

class ScenicToPythonTransformer(ast.NodeTransformer):
    def generic_visit(self, node):
        if isinstance(node, s.AST):
            raise Exception(
                f'Scenic AST visitor is missing for class "{node.__class__.__name__}"'
            )
        return super().generic_visit(node)

    # statements
    def visit_Param(self, node: s.Param):
        return ast.Expr(value=ast.Call(
            func=ast.Name(id="param", ctx=loadCtx),
            args=[],
            keywords=[
                ast.keyword(arg=name, value=value) for (name, value) in node.elts
            ],
        ))

    def visit_EgoAssign(self, node: s.EgoAssign):
        return ast.Expr(value=ast.Call(
            func=ast.Name(id="ego", ctx=loadCtx),
            args=[self.visit(node.value)],
            keywords=[],
        ))

    # instance & specifiers
    def visit_New(self, node: s.New):
        return ast.Call(
            func=ast.Name(id=node.className, ctx=loadCtx),
            args=[self.visit(s) for s in node.specifiers],
            keywords=[],
        )

    def visit_With(self, node: s.With):
        return ast.Call(
            func=ast.Name(id="With", ctx=loadCtx),
            args=[
                ast.Constant(value=node.name),
                self.visit(node.expr),
            ],
            keywords=[],
        )

    def visit_At(self, node: s.At):
        return ast.Call(
            func=ast.Name(id="At", ctx=loadCtx),
            args=[
                self.visit(node.position),
            ],
            keywords=[],
        )

    def visit_Offset(self, node: s.Offset):
        if node.along is None:
            return ast.Call(
                func=ast.Name(id="OffsetBy", ctx=loadCtx),
                args=[
                    self.visit(node.amount),
                ],
                keywords=[],
            )
        return ast.Call(
            func=ast.Name(id="OffsetAlongSpec", ctx=loadCtx),
            args=[
                self.visit(node.along),
                self.visit(node.amount),
            ],
            keywords=[],
        )

    def visit_Position(self, node: s.Position):
        _id = ""
        if isinstance(node.direction, s.Left):
            _id = "LeftSpec"
        if isinstance(node.direction, s.Right):
            _id = "RightSpec"
        if isinstance(node.direction, s.Ahead):
            _id = "Ahead"
        if isinstance(node.direction, s.Behind):
            _id = "Behind"
        return ast.Call(
            func=ast.Name(id=_id, ctx=loadCtx),
            args=[
                self.visit(node.position),
            ],
            keywords=(
                []
                if node.distance is None
                else [ast.keyword(arg="dist", value=self.visit(node.distance))]
            ),
        )

    
    def visit_Visible(self, node: s.Visible):
        return ast.Call(
            func=ast.Name(id="VisibleSpec", ctx=loadCtx),
            args=[],
            keywords=[],
        )

    # operators
    def visit_Deg(self, node: s.Deg):
        return ast.BinOp(
            left=node.value,
            op=ast.Mult(),
            right=ast.Constant(value=0.01745329252)
        )

    def visit_FieldAt(self, node: s.FieldAt):
        return ast.Call(
            func=ast.Name(id="FieldAt", ctx=loadCtx),
            args=[self.visit(node.vectorField), self.visit(node.vector)],
            keywords=[],
        )
    
    def visit_Follow(self, node: s.Follow):
        return ast.Call(
            func=ast.Name(id="Follow", ctx=loadCtx),
            args=[self.visit(node.field), self.visit(node.vector), self.visit(node.number)],
            keywords=[],
        )
    
    def visit_In(self, node: s.In):
        return ast.Call(
            func=ast.Name(id="In", ctx=loadCtx),
            args=[self.visit(node.region)],
            keywords=[],
        )

    def visit_Following(self, node: s.Following):
        return ast.Call(
            func=ast.Name(id="Following", ctx=loadCtx),
            args=[self.visit(node.field), self.visit(node.distance)],
            keywords=(
                []
                if node.fromPoint is None
                else [ast.keyword(arg="fromPt", value=self.visit(node.fromPoint))]
            ),
        )
    
    def visit_FieldAt(self, node: s.FieldAt):
        return ast.Call(
            func=ast.Name(id="FieldAt", ctx=loadCtx),
            args=[self.visit(node.vectorField), self.visit(node.vector)],
            keywords=[],
        )

    def visit_Of(self, node: s.Of):
        _id = ""
        if isinstance(node.align, s.Front):
            _id = "Front"
        if isinstance(node.align, s.Back):
            _id = "Back"
        if isinstance(node.align, s.Left):
            _id = "Left"
        if isinstance(node.align, s.Right):
            _id = "Right"
        if isinstance(node.align, s.FrontLeft):
            _id = "FrontLeft"
        if isinstance(node.align, s.FrontRight):
            _id = "FrontRight"
        if isinstance(node.align, s.BackLeft):
            _id = "BackLeft"
        if isinstance(node.align, s.BackRight):
            _id = "BackRight"
        return ast.Call(
            func=ast.Name(id=_id, ctx=loadCtx),
            args=[self.visit(node.object)],
            keywords=[],
        )
    
    def visit_Facing(self, node: s.Facing):
        return ast.Call(
            func=ast.Name(id="Facing", ctx=loadCtx),
            args=[self.visit(node.heading)],
            keywords=[],
        )
    
    def visit_RelativeTo(self, node: s.RelativeTo):
        return ast.Call(
            func=ast.Name(id="RelativeTo", ctx=loadCtx),
            args=[self.visit(node.left), self.visit(node.right)],
            keywords=[],
        )

    def visit_Vector(self, node: s.Vector):
        return ast.Call(
            func=ast.Name(id="Vector", ctx=loadCtx),
            args=[self.visit(node.x), self.visit(node.y)],
            keywords=[],
        )

    # override default behavior
    def visit_ClassDef(self, node: ast.ClassDef):
        # if no base class is specified
        if not node.bases:
            # use `Object` as a base
            node.bases = [ast.Name(id='Object', ctx=ast.Load())]

        # property defaults
        newBody = []
        for child in node.body:
            child = self.visit(child)
            if isinstance(child, ast.AnnAssign):	# default value for property
                origValue = child.annotation
                target = child.target
                # extract any attributes for this property
                metaAttrs = []
                if isinstance(target, ast.Subscript):
                    sl = target.slice
                    if isinstance(sl, ast.Index):	# needed for compatibility with Python 3.8 and earlier
                        sl = sl.value
                    if isinstance(sl, ast.Name):
                        metaAttrs.append(sl.id)
                    elif isinstance(sl, ast.Tuple):
                        for elt in sl.elts:
                            if not isinstance(elt, ast.Name):
                                self.parseError(elt,
                                    'malformed attributes for property default')
                            metaAttrs.append(elt.id)
                    else:
                        self.parseError(sl, 'malformed attributes for property default')
                    newTarget = ast.Name(target.value.id, ast.Store())
                    ast.copy_location(newTarget, target)
                    target = newTarget
                # find dependencies of the default value
                from scenic.syntax.translator import AttributeFinder
                properties = AttributeFinder.find('self', origValue)
                # create default value object
                args = [
                    ast.Set([ast.Str(prop) for prop in properties]),
                    ast.Set([ast.Str(attr) for attr in metaAttrs]),
                    ast.Lambda(selfArg, origValue)
                ]
                value = ast.Call(ast.Name("PropertyDefault", ast.Load()), args, [])
                ast.copy_location(value, origValue)
                newChild = ast.AnnAssign(
                    target=target, annotation=value,
                    value=None, simple=True)
                child = ast.copy_location(newChild, child)
            newBody.append(child)
        node.body = newBody

        # TODO(shun): Is this right?
        return node
