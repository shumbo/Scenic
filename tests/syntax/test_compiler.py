from ast import *
from scenic.syntax.ast import *

from scenic.syntax.compiler import compileScenicAST


class TestCompiler:
    # Special Case
    def test_ego_assign(self):
        node, _ = compileScenicAST(EgoAssign(Constant(1)))
        match node:
            case Expr(Call(Name("ego"), [Constant(1)])):
                assert True
            case _:
                assert False

    # Instance & Specifiers
    def test_new_no_specifiers(self):
        node, _ = compileScenicAST(New("Object", []))
        match node:
            case Call(Name("Object")):
                assert True
            case _:
                assert False

    def test_with_specifier(self):
        node, _ = compileScenicAST(WithSpecifier("foo", Constant(1)))
        match node:
            case Call(Name("With"), [Constant(prop), Constant(value)]):
                assert prop == "foo"
                assert value == 1
            case _:
                assert False

    def test_at_specifier(self):
        node, _ = compileScenicAST(AtSpecifier(Name("x")))
        match node:
            case Call(Name("At"), [Name("x")]):
                assert True
            case _:
                assert False

    def test_offset_by_specifier(self):
        node, _ = compileScenicAST(OffsetBySpecifier(Name("x")))
        match node:
            case Call(Name("OffsetBy"), [Name("x")]):
                assert True
            case _:
                assert False

    def test_offset_along_specifier(self):
        node, _ = compileScenicAST(
            OffsetAlongSpecifier(Name("direction"), Name("offset"))
        )
        match node:
            case Call(Name("OffsetAlongSpec"), [Name("direction"), Name("offset")]):
                assert True
            case _:
                assert False
