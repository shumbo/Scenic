from ast import *
from typing import Any

import pytest

from scenic.syntax.ast import *
from scenic.syntax.parser import parse_string


def parse_string_helper(source: str) -> Any:
    "Parse string and return Scenic AST"
    return parse_string(source, "exec")


def assert_equal_source_ast(source: str, expected: ast.AST) -> bool:
    "Parse string and compare the resulting AST with the given AST"
    mod = parse_string_helper(source)
    stmt = mod.body[0].value
    assert dump(stmt, annotate_fields=False) == dump(expected, annotate_fields=False)


class TestTrackedNames:
    def test_ego_assign(self):
        mod = parse_string_helper("ego = 10")
        stmt = mod.body[0]
        match stmt:
            case TrackedAssign(Ego(), Constant(10)):
                assert True
            case _:
                assert False

    def test_ego_assign_with_new(self):
        mod = parse_string_helper("ego = new Object")
        stmt = mod.body[0]
        match stmt:
            case TrackedAssign(Ego(), New("Object")):
                assert True
            case _:
                assert False

    def test_workspace_assign(self):
        mod = parse_string_helper("workspace = Workspace()")
        stmt = mod.body[0]
        match stmt:
            case TrackedAssign(Workspace(), Call(Name("Workspace"))):
                assert True
            case _:
                assert False


class TestModel:
    def test_basic(self):
        mod = parse_string_helper("model some_model")
        stmt = mod.body[0]
        match stmt:
            case Model("some_model"):
                assert True
            case _:
                assert False

    def test_dotted(self):
        mod = parse_string_helper("model scenic.simulators.carla.model")
        stmt = mod.body[0]
        match stmt:
            case Model("scenic.simulators.carla.model"):
                assert True
            case _:
                assert False


class TestMutate:
    def test_basic(self):
        mod = parse_string_helper("mutate x")
        stmt = mod.body[0]
        match stmt:
            case Mutate([Name("x", Load())]):
                assert True
            case _:
                assert False

    def test_multiple(self):
        mod = parse_string_helper("mutate x, y, z")
        stmt = mod.body[0]
        match stmt:
            case Mutate([Name("x", Load()), Name("y", Load()), Name("z", Load())]):
                assert True
            case _:
                assert False

    def test_ego(self):
        mod = parse_string_helper("mutate ego")
        stmt = mod.body[0]
        match stmt:
            case Mutate([Name("ego", Load())]):
                assert True
            case _:
                assert False

    def test_empty(self):
        mod = parse_string_helper("mutate")
        stmt = mod.body[0]
        match stmt:
            case Mutate([]):
                assert True
            case _:
                assert False


class TestParam:
    def test_basic(self):
        mod = parse_string_helper("param i = v")
        stmt = mod.body[0]
        match stmt:
            case Param([parameter("i", Name("v"))]):
                assert True
            case _:
                assert False

    def test_quoted(self):
        mod = parse_string_helper("param 'i' = v")
        stmt = mod.body[0]
        match stmt:
            case Param([parameter("i", Name("v"))]):
                assert True
            case _:
                assert False

    def test_multiple(self):
        mod = parse_string_helper('param p1=v1, "p2"=v2, "p1"=v3')
        stmt = mod.body[0]
        match stmt:
            case Param(
                [
                    parameter("p1", Name("v1")),
                    parameter("p2", Name("v2")),
                    parameter("p1", Name("v3")),
                ]
            ):
                assert True
            case _:
                assert False

    def test_empty(self):
        with pytest.raises(SyntaxError):
            parse_string_helper("param")


class TestRequire:
    def test_basic(self):
        mod = parse_string_helper("require X")
        stmt = mod.body[0]
        match stmt:
            case Require(Name("X"), None, None):
                assert True
            case _:
                assert False

    def test_prob(self):
        mod = parse_string_helper("require[0.2] X")
        stmt = mod.body[0]
        match stmt:
            case Require(Name("X"), 0.2, None):
                assert True
            case _:
                assert False

    def test_name(self):
        mod = parse_string_helper("require X as name")
        stmt = mod.body[0]
        match stmt:
            case Require(Name("X"), None, "name"):
                assert True
            case _:
                assert False

    def test_prob_name(self):
        mod = parse_string_helper("require[0.3] X as name")
        stmt = mod.body[0]
        match stmt:
            case Require(Name("X"), 0.3, "name"):
                assert True
            case _:
                assert False

    def test_name_quoted(self):
        mod = parse_string_helper("require X as 'requirement name'")
        stmt = mod.body[0]
        match stmt:
            case Require(Name("X"), None, "requirement name"):
                assert True
            case _:
                assert False

    def test_name_number(self):
        mod = parse_string_helper("require X as 123")
        stmt = mod.body[0]
        match stmt:
            case Require(Name("X"), None, "123"):
                assert True
            case _:
                assert False


class TestNew:
    def test_basic(self):
        mod = parse_string_helper("new Object")
        stmt = mod.body[0]
        match stmt:
            case Expr(New("Object")):
                assert True
            case _:
                assert False

    def test_specifier_single(self):
        mod = parse_string_helper("new Object with foo 1")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New(
                    "Object",
                    [WithSpecifier("foo", Constant(1))],
                )
            ):
                assert True
            case _:
                assert False

    def test_specifier_multiple(self):
        mod = parse_string_helper("new Object with foo 1, with bar 2, with baz 3")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New(
                    "Object",
                    [
                        WithSpecifier("foo", Constant(1)),
                        WithSpecifier("bar", Constant(2)),
                        WithSpecifier("baz", Constant(3)),
                    ],
                )
            ):
                assert True
            case _:
                assert False

    def test_specifier_at(self):
        mod = parse_string_helper("new Object at x")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New(
                    "Object",
                    [AtSpecifier(Name("x"))],
                )
            ):
                assert True
            case _:
                assert False

    def test_specifier_offset_by(self):
        mod = parse_string_helper("new Object offset by x")
        stmt = mod.body[0]
        match stmt:
            case Expr(New("Object", [OffsetBySpecifier(Name("x"))])):
                assert True
            case _:
                assert False

    def test_specifier_offset_along(self):
        mod = parse_string_helper("new Object offset along x by y")
        stmt = mod.body[0]
        match stmt:
            case Expr(New("Object", [OffsetAlongSpecifier(Name("x"), Name("y"))])):
                assert True
            case _:
                assert False

    def test_specifier_position_left(self):
        mod = parse_string_helper("new Object left of left")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New("Object", [DirectionOfSpecifier(LeftOf(), Name("left"), None)])
            ):
                assert True
            case _:
                assert False

    def test_specifier_position_right(self):
        mod = parse_string_helper("new Object right of right")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New("Object", [DirectionOfSpecifier(RightOf(), Name("right"), None)])
            ):
                assert True
            case _:
                assert False

    def test_specifier_position_ahead(self):
        mod = parse_string_helper("new Object ahead of ahead")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New("Object", [DirectionOfSpecifier(AheadOf(), Name("ahead"), None)])
            ):
                assert True
            case _:
                assert False

    def test_specifier_position_behind(self):
        mod = parse_string_helper("new Object behind behind")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New("Object", [DirectionOfSpecifier(Behind(), Name("behind"), None)])
            ):
                assert True
            case _:
                assert False

    def test_specifier_position_by(self):
        mod = parse_string_helper("new Object left of left by distance")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New(
                    "Object",
                    [DirectionOfSpecifier(LeftOf(), Name("left"), Name("distance"))],
                )
            ):
                assert True
            case _:
                assert False

    def test_specifier_beyond(self):
        mod = parse_string_helper("new Object beyond position by distance")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New(
                    "Object",
                    [BeyondSpecifier(Name("position"), Name("distance"))],
                )
            ):
                assert True
            case _:
                assert False

    def test_specifier_beyond_from(self):
        mod = parse_string_helper("new Object beyond position by d from base")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New(
                    "Object",
                    [BeyondSpecifier(Name("position"), Name("d"), Name("base"))],
                )
            ):
                assert True
            case _:
                assert False

    def test_specifier_visible(self):
        mod = parse_string_helper("new Object visible")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New(
                    "Object",
                    [VisibleSpecifier(None)],
                )
            ):
                assert True
            case _:
                assert False

    def test_specifier_visible_from(self):
        mod = parse_string_helper("new Object visible from base")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New(
                    "Object",
                    [VisibleSpecifier(Name("base"))],
                )
            ):
                assert True
            case _:
                assert False

    def test_specifier_in(self):
        mod = parse_string_helper("new Object in region")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New(
                    "Object",
                    [InSpecifier(Name("region"))],
                )
            ):
                assert True
            case _:
                assert False

    def test_specifier_on(self):
        mod = parse_string_helper("new Object on region")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New(
                    "Object",
                    [InSpecifier(Name("region"))],
                )
            ):
                assert True
            case _:
                assert False

    def test_specifier_following(self):
        mod = parse_string_helper("new Object following field for distance")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New(
                    "Object",
                    [FollowingSpecifier(Name("field"), Name("distance"), None)],
                )
            ):
                assert True
            case _:
                assert False

    def test_specifier_following_from(self):
        mod = parse_string_helper("new Object following field from base for distance")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New(
                    "Object",
                    [FollowingSpecifier(Name("field"), Name("distance"), Name("base"))],
                )
            ):
                assert True
            case _:
                assert False

    def test_specifier_facing(self):
        mod = parse_string_helper("new Object facing heading")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New(
                    "Object",
                    [FacingSpecifier(Name("heading"))],
                )
            ):
                assert True
            case _:
                assert False

    def test_specifier_facing_toward(self):
        mod = parse_string_helper("new Object facing toward position")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New(
                    "Object",
                    [FacingTowardSpecifier(Name("position"))],
                )
            ):
                assert True
            case _:
                assert False

    def test_specifier_apparently_facing(self):
        mod = parse_string_helper("new Object apparently facing heading")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New(
                    "Object",
                    [ApparentlyFacingSpecifier(Name("heading"), None)],
                )
            ):
                assert True
            case _:
                assert False

    def test_specifier_apparently_facing_from(self):
        mod = parse_string_helper("new Object apparently facing heading from base")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                New(
                    "Object",
                    [ApparentlyFacingSpecifier(Name("heading"), Name("base"))],
                )
            ):
                assert True
            case _:
                assert False


class TestOperator:
    def test_relative_position(self):
        mod = parse_string_helper("relative position of x")
        stmt = mod.body[0]
        match stmt:
            case Expr(RelativePositionOp(Name("x"))):
                assert True
            case _:
                assert False

    def test_relative_position_base(self):
        mod = parse_string_helper("relative position of x from y")
        stmt = mod.body[0]
        match stmt:
            case Expr(RelativePositionOp(Name("x"), Name("y"))):
                assert True
            case _:
                assert False

    @pytest.mark.parametrize(
        "code,expected",
        [
            (
                "relative position of relative position of A from B",
                RelativePositionOp(
                    RelativePositionOp(Name("A", Load()), Name("B", Load()))
                ),
            ),
            (
                "relative position of relative position of A from B from C",
                RelativePositionOp(
                    RelativePositionOp(Name("A", Load()), Name("B", Load())),
                    Name("C", Load()),
                ),
            ),
            (
                "relative position of A from relative position of B from C",
                RelativePositionOp(
                    Name("A", Load()),
                    RelativePositionOp(Name("B", Load()), Name("C", Load())),
                ),
            ),
            (
                "relative position of A << B from C",
                RelativePositionOp(
                    BinOp(Name("A", Load()), LShift(), Name("B", Load())),
                    Name("C", Load()),
                ),
            ),
            (
                "relative position of A from B << C",
                BinOp(
                    RelativePositionOp(Name("A", Load()), Name("B", Load())),
                    LShift(),
                    Name("C", Load()),
                ),
            ),
            (
                "relative position of A + B from C",
                RelativePositionOp(
                    BinOp(Name("A", Load()), Add(), Name("B", Load())),
                    Name("C", Load()),
                ),
            ),
            (
                "relative position of A from B + C",
                RelativePositionOp(
                    Name("A", Load()),
                    BinOp(Name("B", Load()), Add(), Name("C", Load())),
                ),
            ),
            (
                "relative position of A << B",
                BinOp(
                    RelativePositionOp(Name("A", Load())), LShift(), Name("B", Load())
                ),
            ),
            (
                "relative position of A + B",
                RelativePositionOp(BinOp(Name("A", Load()), Add(), Name("B", Load()))),
            ),
        ],
    )
    def test_relative_position_precedence(self, code, expected):
        assert_equal_source_ast(code, expected)

    def test_relative_heading(self):
        mod = parse_string_helper("relative heading of x")
        stmt = mod.body[0]
        match stmt:
            case Expr(RelativeHeadingOp(Name("x"))):
                assert True
            case _:
                assert False

    def test_relative_heading_from(self):
        mod = parse_string_helper("relative heading of x from y")
        stmt = mod.body[0]
        match stmt:
            case Expr(RelativeHeadingOp(Name("x"), Name("y"))):
                assert True
            case _:
                assert False

    @pytest.mark.parametrize(
        "code,expected",
        [
            (
                "relative heading of relative heading of A from B",
                RelativeHeadingOp(
                    RelativeHeadingOp(Name("A", Load()), Name("B", Load()))
                ),
            ),
            (
                "relative heading of relative heading of A from B from C",
                RelativeHeadingOp(
                    RelativeHeadingOp(Name("A", Load()), Name("B", Load())),
                    Name("C", Load()),
                ),
            ),
            (
                "relative heading of A from relative heading of B from C",
                RelativeHeadingOp(
                    Name("A", Load()),
                    RelativeHeadingOp(Name("B", Load()), Name("C", Load())),
                ),
            ),
            (
                "relative heading of A << B from C",
                RelativeHeadingOp(
                    BinOp(Name("A", Load()), LShift(), Name("B", Load())),
                    Name("C", Load()),
                ),
            ),
            (
                "relative heading of A from B << C",
                BinOp(
                    RelativeHeadingOp(Name("A", Load()), Name("B", Load())),
                    LShift(),
                    Name("C", Load()),
                ),
            ),
            (
                "relative heading of A + B from C",
                RelativeHeadingOp(
                    BinOp(Name("A", Load()), Add(), Name("B", Load())),
                    Name("C", Load()),
                ),
            ),
            (
                "relative heading of A from B + C",
                RelativeHeadingOp(
                    Name("A", Load()),
                    BinOp(Name("B", Load()), Add(), Name("C", Load())),
                ),
            ),
            (
                "relative heading of A << B",
                BinOp(
                    RelativeHeadingOp(Name("A", Load())), LShift(), Name("B", Load())
                ),
            ),
            (
                "relative heading of A + B",
                RelativeHeadingOp(BinOp(Name("A", Load()), Add(), Name("B", Load()))),
            ),
        ],
    )
    def test_relative_heading_precedence(self, code, expected):
        assert_equal_source_ast(code, expected)

    def test_apparent_heading(self):
        mod = parse_string_helper("apparent heading of x")
        stmt = mod.body[0]
        match stmt:
            case Expr(ApparentHeadingOp(Name("x"))):
                assert True
            case _:
                assert False

    def test_apparent_heading_from(self):
        mod = parse_string_helper("apparent heading of x from y")
        stmt = mod.body[0]
        match stmt:
            case Expr(ApparentHeadingOp(Name("x"), Name("y"))):
                assert True
            case _:
                assert False

    @pytest.mark.parametrize(
        "code,expected",
        [
            (
                "apparent heading of apparent heading of A from B",
                ApparentHeadingOp(
                    ApparentHeadingOp(Name("A", Load()), Name("B", Load()))
                ),
            ),
            (
                "apparent heading of apparent heading of A from B from C",
                ApparentHeadingOp(
                    ApparentHeadingOp(Name("A", Load()), Name("B", Load())),
                    Name("C", Load()),
                ),
            ),
            (
                "apparent heading of A from apparent heading of B from C",
                ApparentHeadingOp(
                    Name("A", Load()),
                    ApparentHeadingOp(Name("B", Load()), Name("C", Load())),
                ),
            ),
            (
                "apparent heading of A << B from C",
                ApparentHeadingOp(
                    BinOp(Name("A", Load()), LShift(), Name("B", Load())),
                    Name("C", Load()),
                ),
            ),
            (
                "apparent heading of A from B << C",
                BinOp(
                    ApparentHeadingOp(Name("A", Load()), Name("B", Load())),
                    LShift(),
                    Name("C", Load()),
                ),
            ),
            (
                "apparent heading of A + B from C",
                ApparentHeadingOp(
                    BinOp(Name("A", Load()), Add(), Name("B", Load())),
                    Name("C", Load()),
                ),
            ),
            (
                "apparent heading of A from B + C",
                ApparentHeadingOp(
                    Name("A", Load()),
                    BinOp(Name("B", Load()), Add(), Name("C", Load())),
                ),
            ),
            (
                "apparent heading of A << B",
                BinOp(
                    ApparentHeadingOp(Name("A", Load())), LShift(), Name("B", Load())
                ),
            ),
            (
                "apparent heading of A + B",
                ApparentHeadingOp(BinOp(Name("A", Load()), Add(), Name("B", Load()))),
            ),
        ],
    )
    def test_apparent_heading_precedence(self, code, expected):
        assert_equal_source_ast(code, expected)

    def test_distance_from(self):
        mod = parse_string_helper("distance from x")
        stmt = mod.body[0]
        match stmt:
            case Expr(DistanceFromOp(None, Name("x"))):
                assert True
            case _:
                assert False

    def test_distance_to(self):
        mod = parse_string_helper("distance to x")
        stmt = mod.body[0]
        match stmt:
            case Expr(DistanceFromOp(Name("x"), None)):
                assert True
            case _:
                assert False

    def test_distance_from_to(self):
        mod = parse_string_helper("distance from x to y")
        stmt = mod.body[0]
        match stmt:
            case Expr(DistanceFromOp(Name("y"), Name("x"))):
                assert True
            case _:
                assert False

    def test_distance_to_from(self):
        mod = parse_string_helper("distance to x from y")
        stmt = mod.body[0]
        match stmt:
            case Expr(DistanceFromOp(Name("x"), Name("y"))):
                assert True
            case _:
                assert False

    @pytest.mark.parametrize(
        "code,expected",
        [
            (
                "distance to distance from A to B",
                DistanceFromOp(DistanceFromOp(Name("B", Load()), Name("A", Load()))),
            ),
            (
                "distance to distance from A from B",
                DistanceFromOp(
                    DistanceFromOp(None, Name("A", Load())), Name("B", Load())
                ),
            ),
            (
                "distance to A << B from C << D",
                BinOp(
                    DistanceFromOp(
                        BinOp(Name("A", Load()), LShift(), Name("B", Load())),
                        Name("C", Load()),
                    ),
                    LShift(),
                    Name("D", Load()),
                ),
            ),
            (
                "distance to A + B from C + D",
                DistanceFromOp(
                    BinOp(Name("A", Load()), Add(), Name("B", Load())),
                    BinOp(Name("C", Load()), Add(), Name("D", Load())),
                ),
            ),
            (
                "distance to A + B",
                DistanceFromOp(
                    BinOp(Name("A", Load()), Add(), Name("B", Load())),
                    None,
                ),
            ),
            (
                "distance from A + B",
                DistanceFromOp(
                    None,
                    BinOp(Name("A", Load()), Add(), Name("B", Load())),
                ),
            ),
            (
                "distance to A << B",
                BinOp(
                    DistanceFromOp(Name("A", Load()), None),
                    LShift(),
                    Name("B", Load()),
                ),
            ),
            (
                "distance from A << B",
                BinOp(
                    DistanceFromOp(None, Name("A", Load())),
                    LShift(),
                    Name("B", Load()),
                ),
            ),
        ],
    )
    def test_distance_from_precedence(self, code, expected):
        assert_equal_source_ast(code, expected)

    def test_distance_past(self):
        mod = parse_string_helper("distance past x")
        stmt = mod.body[0]
        match stmt:
            case Expr(DistancePastOp(Name("x"))):
                assert True
            case _:
                assert False

    def test_distance_past_of(self):
        mod = parse_string_helper("distance past x of y")
        stmt = mod.body[0]
        match stmt:
            case Expr(DistancePastOp(Name("x"), Name("y"))):
                assert True
            case _:
                assert False

    @pytest.mark.parametrize(
        "code,expected",
        [
            (
                "distance past distance past A of B",
                DistancePastOp(DistancePastOp(Name("A", Load()), Name("B", Load()))),
            ),
            (
                "distance past distance past A of B of C",
                DistancePastOp(
                    DistancePastOp(Name("A", Load()), Name("B", Load())),
                    Name("C", Load()),
                ),
            ),
            (
                "distance past A of distance past B of C",
                DistancePastOp(
                    Name("A", Load()),
                    DistancePastOp(Name("B", Load()), Name("C", Load())),
                ),
            ),
            (
                "distance past A << B of C << D",
                BinOp(
                    DistancePastOp(
                        BinOp(
                            Name("A", Load()),
                            LShift(),
                            Name("B", Load()),
                        ),
                        Name("C", Load()),
                    ),
                    LShift(),
                    Name("D", Load()),
                ),
            ),
            (
                "distance past A + B of C + D",
                DistancePastOp(
                    BinOp(Name("A", Load()), Add(), Name("B", Load())),
                    BinOp(Name("C", Load()), Add(), Name("D", Load())),
                ),
            ),
            (
                "distance past A << B",
                BinOp(
                    DistancePastOp(
                        Name("A", Load()),
                    ),
                    LShift(),
                    Name("B", Load()),
                ),
            ),
            (
                "distance past A + B",
                DistancePastOp(
                    BinOp(Name("A", Load()), Add(), Name("B", Load())),
                ),
            ),
        ],
    )
    def test_distance_past_precedence(self, code, expected):
        assert_equal_source_ast(code, expected)

    def test_angle_from(self):
        mod = parse_string_helper("angle from x")
        stmt = mod.body[0]
        match stmt:
            case Expr(AngleFromOp(None, Name("x"))):
                assert True
            case _:
                assert False

    def test_angle_to(self):
        mod = parse_string_helper("angle to x")
        stmt = mod.body[0]
        match stmt:
            case Expr(AngleFromOp(Name("x"), None)):
                assert True
            case _:
                assert False

    def test_angle_from_to(self):
        mod = parse_string_helper("angle from x to y")
        stmt = mod.body[0]
        match stmt:
            case Expr(AngleFromOp(Name("y"), Name("x"))):
                assert True
            case _:
                assert False

    def test_angle_to_from(self):
        mod = parse_string_helper("angle to x from y")
        stmt = mod.body[0]
        match stmt:
            case Expr(AngleFromOp(Name("x"), Name("y"))):
                assert True
            case _:
                assert False

    @pytest.mark.parametrize(
        "code,expected",
        [
            (
                "angle to angle from A to B",
                AngleFromOp(AngleFromOp(Name("B", Load()), Name("A", Load()))),
            ),
            (
                "angle to angle from A from B",
                AngleFromOp(AngleFromOp(None, Name("A", Load())), Name("B", Load())),
            ),
            (
                "angle to A << B from C << D",
                BinOp(
                    AngleFromOp(
                        BinOp(Name("A", Load()), LShift(), Name("B", Load())),
                        Name("C", Load()),
                    ),
                    LShift(),
                    Name("D", Load()),
                ),
            ),
            (
                "angle from A + B to C + D",
                AngleFromOp(
                    BinOp(Name("C", Load()), Add(), Name("D", Load())),
                    BinOp(Name("A", Load()), Add(), Name("B", Load())),
                ),
            ),
            (
                "angle to A + B",
                AngleFromOp(
                    BinOp(Name("A", Load()), Add(), Name("B", Load())),
                    None,
                ),
            ),
            (
                "angle from A + B",
                AngleFromOp(
                    None,
                    BinOp(Name("A", Load()), Add(), Name("B", Load())),
                ),
            ),
            (
                "angle to A << B",
                BinOp(
                    AngleFromOp(Name("A", Load()), None),
                    LShift(),
                    Name("B", Load()),
                ),
            ),
            (
                "angle from A << B",
                BinOp(
                    AngleFromOp(None, Name("A", Load())),
                    LShift(),
                    Name("B", Load()),
                ),
            ),
        ],
    )
    def test_angle_from_precedence(self, code, expected):
        assert_equal_source_ast(code, expected)

    def test_follow(self):
        mod = parse_string_helper("follow x from y for z")
        stmt = mod.body[0]
        match stmt:
            case Expr(FollowOp(Name("x"), Name("y"), Name("z"))):
                assert True
            case _:
                assert False

    @pytest.mark.parametrize(
        "code,expected",
        [
            (
                "follow A + B from C + D for E + F",
                FollowOp(
                    BinOp(Name("A", Load()), Add(), Name("B", Load())),
                    BinOp(Name("C", Load()), Add(), Name("D", Load())),
                    BinOp(Name("E", Load()), Add(), Name("F", Load())),
                ),
            ),
            (
                "follow A << B from C << D for E << F",
                BinOp(
                    FollowOp(
                        BinOp(Name("A", Load()), LShift(), Name("B", Load())),
                        BinOp(Name("C", Load()), LShift(), Name("D", Load())),
                        Name("E", Load()),
                    ),
                    LShift(),
                    Name("F", Load()),
                ),
            ),
        ],
    )
    def test_follow_precedence(self, code, expected):
        assert_equal_source_ast(code, expected)

    def test_visible(self):
        mod = parse_string_helper("visible x")
        stmt = mod.body[0]
        match stmt:
            case Expr(VisibleOp(Name("x"))):
                assert True
            case _:
                assert False

    @pytest.mark.parametrize(
        "code,expected",
        [
            (
                "visible A + B",
                VisibleOp(
                    BinOp(Name("A", Load()), Add(), Name("B", Load())),
                ),
            ),
            (
                "visible A << B",
                BinOp(
                    VisibleOp(Name("A", Load())),
                    LShift(),
                    Name("B", Load()),
                ),
            ),
        ],
    )
    def test_visible_precedence(self, code, expected):
        assert_equal_source_ast(code, expected)

    def test_not_visible(self):
        mod = parse_string_helper("not visible x")
        stmt = mod.body[0]
        match stmt:
            case Expr(NotVisibleOp(Name("x"))):
                assert True
            case _:
                assert False

    def test_not_visible_inversion(self):
        mod = parse_string_helper("not visible")
        stmt = mod.body[0]
        match stmt:
            case Expr(UnaryOp(Not(), Name("visible"))):
                assert True
            case _:
                assert False

    def test_not_visible_with_not(self):
        mod = parse_string_helper("not not visible x")
        stmt = mod.body[0]
        match stmt:
            case Expr(UnaryOp(Not(), NotVisibleOp(Name("x")))):
                assert True
            case _:
                assert False

    @pytest.mark.parametrize(
        "code,expected",
        [
            (
                "not visible A + B",
                NotVisibleOp(
                    BinOp(Name("A", Load()), Add(), Name("B", Load())),
                ),
            ),
            (
                "not visible A << B",
                BinOp(
                    NotVisibleOp(Name("A", Load())),
                    LShift(),
                    Name("B", Load()),
                ),
            ),
        ],
    )
    def test_visible_precedence(self, code, expected):
        assert_equal_source_ast(code, expected)

    @pytest.mark.parametrize(
        "position,node",
        [
            ("front", Front),
            ("back", Back),
            ("left", Left),
            ("right", Right),
            ("front left", FrontLeft),
            ("front right", FrontRight),
            ("back left", BackLeft),
            ("back right", BackRight),
        ],
    )
    def test_position_of(self, position, node):
        mod = parse_string_helper(f"{position} of x")
        stmt = mod.body[0]
        match stmt:
            case Expr(PositionOfOp(positionNode, Name("x"))):
                assert isinstance(positionNode, node)
            case _:
                assert False

    @pytest.mark.parametrize(
        "code,expected",
        [
            (
                "front of A + B",
                PositionOfOp(
                    Front(),
                    BinOp(Name("A", Load()), Add(), Name("B", Load())),
                ),
            ),
            (
                "left of A << B",
                BinOp(
                    PositionOfOp(Left(), Name("A", Load())),
                    LShift(),
                    Name("B", Load()),
                ),
            ),
        ],
    )
    def test_visible_precedence(self, code, expected):
        assert_equal_source_ast(code, expected)

    def test_deg_1(self):
        mod = parse_string_helper("1 + 2 deg")
        stmt = mod.body[0]
        match stmt:
            case Expr(BinOp(Constant(1), Add(), DegOp(Constant(2)))):
                assert True
            case _:
                assert False

    def test_deg_2(self):
        mod = parse_string_helper("6 * 2 deg")
        stmt = mod.body[0]
        match stmt:
            case Expr(DegOp(BinOp(Constant(6), Mult(), Constant(2)))):
                assert True
            case _:
                assert False

    def test_vector_1(self):
        mod = parse_string_helper("1 + 2 @ 3 * 4")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                BinOp(
                    Constant(1),
                    Add(),
                    BinOp(VectorOp(Constant(2), Constant(3)), Mult(), Constant(4)),
                )
            ):
                assert True
            case _:
                assert False

    def test_vector_2(self):
        mod = parse_string_helper("1 * 2 @ 3 + 4")
        stmt = mod.body[0]
        match stmt:
            case Expr(
                BinOp(
                    VectorOp(BinOp(Constant(1), Mult(), Constant(2)), Constant(3)),
                    Add(),
                    Constant(4),
                )
            ):
                assert True
            case _:
                assert False

    def test_vector_3(self):
        mod = parse_string_helper("1 @ 2 @ 3")
        stmt = mod.body[0]
        match stmt:
            case Expr(VectorOp(VectorOp(Constant(1), Constant(2)), Constant(3))):
                assert True
            case _:
                assert False

    def test_field_at(self):
        mod = parse_string_helper("x at y")
        stmt = mod.body[0]
        match stmt:
            case Expr(FieldAtOp(Name("x"), Name("y"))):
                assert True
            case _:
                assert False

    def test_relative_to(self):
        mod = parse_string_helper("x relative to y")
        stmt = mod.body[0]
        match stmt:
            case Expr(RelativeToOp(Name("x"), Name("y"))):
                assert True
            case _:
                assert False

    def test_offset_by(self):
        mod = parse_string_helper("x offset by y")
        stmt = mod.body[0]
        match stmt:
            case Expr(RelativeToOp(Name("x"), Name("y"))):
                assert True
            case _:
                assert False

    def test_offset_along(self):
        mod = parse_string_helper("x offset along y by z")
        stmt = mod.body[0]
        match stmt:
            case Expr(OffsetAlongOp(Name("x"), Name("y"), Name("z"))):
                assert True
            case _:
                assert False

    def test_can_see(self):
        mod = parse_string_helper("x can see y ")
        stmt = mod.body[0]
        match stmt:
            case Expr(CanSeeOp(Name("x"), Name("y"))):
                assert True
            case _:
                assert False
