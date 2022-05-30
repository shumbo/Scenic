import ast


class AST(ast.AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)


# statements
class Require(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.value = kwargs["value"]
        self.prob = kwargs["prob"]
        self._fields = ["value", "prob"]


class Param(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.elts = kwargs["elts"]

class EgoAssign(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.value = kwargs["value"]

class New(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.className = kwargs["className"]
        self.specifiers = kwargs["specifiers"]
        self._fields = ["specifiers"]


class Specifier(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)


class With(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.name = kwargs["name"]
        self.expr = kwargs["expr"]
        self._fields = ["expr"]


class At(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.position = kwargs["position"]
        self._fields = ["position"]


class Offset(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.amount = kwargs["amount"]
        self.along = kwargs.get("along")
        self._fields = ["amount"]


class Position(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.direction = kwargs["direction"]
        self.position = kwargs["position"]
        self.distance = kwargs.get("distance")
        self._fields = ["position", "distance"]


class Beyond(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.position = kwargs["position"]
        self.offset = kwargs["offset"]
        self.fromPosition = kwargs.get("fromPosition")
        self._fields = ["position", "offset", "fromPosition"]


class Visible(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.region = kwargs.get("region")
        self._fields = ["region"]


class NotVisible(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.region = kwargs.get("region")
        self._fields = ["region"]


class In(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.region = kwargs.get("region")
        self._fields = ["region"]


class Following(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.field = kwargs.get("field")
        self.fromPoint = kwargs.get("fromPoint")
        self.distance = kwargs.get("distance")

class Deg(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.value = kwargs["value"]

class FieldAt(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.vectorField = kwargs["vectorField"]
        self.vector = kwargs["vector"]

class Follow(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.field = kwargs["field"]
        self.vector = kwargs["vector"]
        self.number = kwargs["number"]

class Of(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.align = kwargs["align"]
        self.object = kwargs["object"]

class Front(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)

class Back(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)

class Left(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)

class Right(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)

class FrontLeft(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)

class FrontRight(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)

class BackLeft(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)

class BackRight(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)

class Ahead(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)

class Behind(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)


class Facing(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.heading = kwargs["heading"]

class RelativeTo(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.left = kwargs["left"]
        self.right = kwargs["right"]

class Vector(AST):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.x = kwargs["x"]
        self.y = kwargs["y"]
