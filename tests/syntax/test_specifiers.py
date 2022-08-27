import math
import pytest

from scenic.core.errors import RuntimeParseError, InvalidScenarioError
from scenic.core.vectors import Vector
from tests.utils import compileScenic, sampleScene, sampleEgo, sampleEgoFrom

## Dependencies and lazy evaluation

def test_double_specification():
    with pytest.raises(RuntimeParseError):
        compileScenic('ego = Object at 0 @ 0, at 1 @ 1')

def test_cyclic_dependency():
    with pytest.raises(RuntimeParseError):
        compileScenic('ego = Object left of 0 @ 0, facing toward 1 @ 1')

def test_lazy_cyclic_dependency():
    with pytest.raises(RuntimeParseError):
        compileScenic(
            'vf = VectorField("Foo", lambda pos: 3 * pos.x)\n'
            'ego = Object at 0 @ (0 relative to vf)'
        )

def test_default_dependency():
    ego = sampleEgoFrom('ego = Object facing toward -1 @ 1')
    assert tuple(ego.position) == (0, 0, 0)
    assert ego.heading == pytest.approx(math.radians(45))

def test_missing_dependency():
    with pytest.raises(RuntimeParseError):
        compileScenic('Point left of 0 @ 0 by 5\n' 'ego = Object')

def test_lazy_value_in_param():
    with pytest.raises(InvalidScenarioError):
        compileScenic(
            'vf = VectorField("Foo", lambda pos: 3 * pos.x)\n'
            'param X = 0 relative to vf\n'
            'ego = Object\n'
        )

def test_lazy_value_in_requirement():
    # Case where we can statically detect the use of a lazy value
    with pytest.raises(InvalidScenarioError):
        compileScenic(
            'vf = VectorField("Foo", lambda pos: 3 * pos.x)\n'
            'x = 0 relative to vf\n'
            'require x >= 0\n'
            'ego = Object\n'
        )

def test_lazy_value_in_requirement_2():
    # Case where the lazy value is detected during requirement evaluation
    scenario = compileScenic(
        'vf = VectorField("Foo", lambda pos: 3 * pos.x)\n'
        'require 0 relative to vf\n'
        'ego = Object\n'
    )
    with pytest.raises(InvalidScenarioError):
        sampleScene(scenario, maxIterations=1)

## Generic specifiers

def test_with():
    ego = sampleEgoFrom('ego = Object with flubber 37')
    assert ego.flubber == 37

## Position specifiers ##

# At
def test_at():
    ego = sampleEgoFrom('ego = Object at 149 @ 42')
    assert tuple(ego.position) == pytest.approx((149, 42, 0))

def test_at_3d():
    ego = sampleEgoFrom('ego = Object at (3, 4, 12)')
    assert tuple(ego.position) == pytest.approx((3, 4, 12))

# Offset By/Along
def test_offset_by():
    ego = sampleEgoFrom(
        'ego = Object at 10 @ 40, facing 90 deg\n'
        'ego = Object offset by 5 @ 15'
    )
    assert tuple(ego.position) == pytest.approx((-5, 45, 0))

def test_offset_by_3d():
    ego = sampleEgoFrom(
        'ego = Object at (10, 10, 0), facing (90 deg, 0, 90 deg)\n'
        'ego = Object offset by (5, 15, 20)'
    )
    assert tuple(ego.position) == pytest.approx((-5, 30, -5))

def test_offset_by_3d_2():
    ego = sampleEgoFrom(
        'ego = Object at (10, 10, 10), facing (90 deg, 90 deg, 90 deg)\n'
        'ego = Object offset by (5, 15, 20)'
    )
    assert tuple(ego.position) == pytest.approx((5, 30, 25))

def test_offset_by_3d_3():
    ego = sampleEgoFrom(
        'ego = Object at (10,40,35), facing (90 deg, 90 deg,90 deg)\n'
        'ego = Object offset by (5, 10, 15)'
    )
    assert tuple(ego.position) == pytest.approx((5, 55, 45))
    assert tuple(ego.orientation.eulerAngles) == pytest.approx(math.pi/2, math.pi/2, math.pi/2)

def test_offset_by_no_ego():
    with pytest.raises(RuntimeParseError):
        compileScenic('ego = Object offset by 10 @ 40')

def test_offset_along():
    ego = sampleEgoFrom(
        'ego = Object at 10 @ 40\n'
        'ego = Object offset along -90 deg by -10 @ 5'
    )
    assert tuple(ego.position) == pytest.approx((15, 50, 0))

def test_offset_along_3d():
    ego = sampleEgoFrom(
        'ego = Object at (10, 40, 20)\n'
        'ego = Object offset along (90 deg, 0, 90 deg) by (-10, 5, 20)'
    )
    assert tuple(ego.position) == pytest.approx((5, 60, 30))

    ego = sampleEgoFrom(
        'ego = Object at (10, 40, 20)\n'
        'ego = Object offset along (90 deg, 90 deg, 90 deg) by (-10, 5, 20)'
    )
    assert tuple(ego.position) == pytest.approx((20, 60, 25))

def test_offset_along_no_ego():
    with pytest.raises(RuntimeParseError):
        compileScenic('ego = Object offset along 0 by 10 @ 0')

# Left/Right Of 
def test_left_of_vector():
    ego = sampleEgoFrom('ego = Object left of 10 @ 20, facing 90 deg')
    assert tuple(ego.position) == pytest.approx((10, 19.5, 0))
    ego = sampleEgoFrom('ego = Object left of 10 @ 20, with width 10')
    assert tuple(ego.position) == pytest.approx((5, 20, 0))

def test_left_of_vector_3d():
    ego = sampleEgoFrom('ego = Object left of (10, 20, 15), facing (90 deg, 0, 90 deg)')
    assert tuple(ego.position) == pytest.approx((10, 20, 15.5))

def test_left_of_vector_by():
    ego = sampleEgoFrom('ego = Object left of 10 @ 20 by 20')
    assert tuple(ego.position) == pytest.approx((-10.5, 20, 0))
    ego = sampleEgoFrom('ego = Object left of 10 @ 20 by 20 @ 5')
    assert tuple(ego.position) == pytest.approx((-10.5, 25, 0))

def test_left_of_vector_by_3d():
    ego = sampleEgoFrom('ego = Object left of (10, 20, 15) by 20')
    assert tuple(ego.position) == pytest.approx((-10.5, 20, 15))

def test_left_of_oriented_point():
    ego = sampleEgoFrom(
                        'target = OrientedPoint facing (-90 deg, 0, 0)\n'
                        'ego = Object left of target'
                       )
    assert tuple(ego.position) == pytest.approx((0, -0.5, 0))
    assert ego.orientation = Orientation.fromEuler(-math.pi/2, 0, 0)

def test_left_of_object():
    ego = sampleEgoFrom(
                        'target = Object facing (-90 deg, 0, 0)\n'
                        'ego = Object left of target'
                       )
    assert tuple(ego.position) == pytest.approx((0, -1, 0))
    assert ego.orientation = Orientation.fromEuler(-math.pi/2, 0, 0)

def test_right_of_vector():
    ego = sampleEgoFrom('ego = Object right of 10 @ 20, facing 90 deg')
    assert tuple(ego.position) == pytest.approx((10, 20.5, 0))
    ego = sampleEgoFrom('ego = Object right of 10 @ 20, with width 10')
    assert tuple(ego.position) == pytest.approx((15, 20, 0))

def test_right_of_vector_3d():
    ego = sampleEgoFrom('ego = Object right of (10, 20, 15), facing (90 deg, 0, 90 deg)')
    assert tuple(ego.position) == pytest.approx((10, 20, 14.5))

def test_right_of_vector_by():
    ego = sampleEgoFrom('ego = Object right of 10 @ 20 by 20')
    assert tuple(ego.position) == pytest.approx((30.5, 20, 0))
    ego = sampleEgoFrom('ego = Object right of 10 @ 20 by 20 @ 5')
    assert tuple(ego.position) == pytest.approx((30.5, 25, 0))

def test_right_of_vector_by_3d():
    ego = sampleEgoFrom('ego = Object right of (10, 20, 15) by 20')
    assert tuple(ego.position) == pytest.approx((30.5, 20, 15))

def test_right_of_oriented_point():
    ego = sampleEgoFrom(
                        'target = OrientedPoint facing (-90 deg, 0, 0)\n'
                        'ego = Object left of target'
                       )
    assert tuple(ego.position) == pytest.approx((0, 0.5, 0))
    assert ego.orientation = Orientation.fromEuler(-math.pi/2, 0, 0)

def test_right_of_object():
    ego = sampleEgoFrom(
                        'target = Object facing (-90 deg, 0, 0)\n'
                        'ego = Object left of target'
                       )
    assert tuple(ego.position) == pytest.approx((0, 1, 0))
    assert ego.orientation = Orientation.fromEuler(-math.pi/2, 0, 0)

# Ahead Of/Behind
def test_ahead_of_vector():
    ego = sampleEgoFrom('ego = Object ahead of 10 @ 20, facing 90 deg')
    assert tuple(ego.position) == pytest.approx((9.5, 20, 0))
    ego = sampleEgoFrom('ego = Object ahead of 10 @ 20, with length 10')
    assert tuple(ego.position) == pytest.approx((10, 25, 0))

def test_ahead_of_vector_3d():
    ego = sampleEgoFrom('ego = Object ahead of (10, 20, 15), facing (90 deg, 0, 90 deg)')
    assert tuple(ego.position) == pytest.approx((9.5, 20, 15))

def test_ahead_of_vector_by():
    ego = sampleEgoFrom('ego = Object ahead of 10 @ 20 by 20')
    assert tuple(ego.position) == pytest.approx((10, 40.5, 0))
    ego = sampleEgoFrom('ego = Object ahead of 10 @ 20 by 20 @ 5')
    assert tuple(ego.position) == pytest.approx((30, 25.5, 0))

def test_ahead_of_vector_by_3d():
    ego = sampleEgoFrom('ego = Object ahead of (10, 20, 15) by 20')
    assert tuple(ego.position) == pytest.approx((10, 40.5, 15))
    ego = sampleEgoFrom('ego = Object ahead of (10, 20, 15), facing (90 deg, 0, 90 deg), with length 10')
    assert tuple(ego.position) == pytest.approx((5, 20, 15))

def test_ahead_of_oriented_point():
    ego = sampleEgoFrom(
                        'target = OrientedPoint facing (-90 deg, 0, 0)\n'
                        'ego = Object ahead of target'
                       )
    assert tuple(ego.position) == pytest.approx((0.5, 0, 0))
    assert ego.orientation = Orientation.fromEuler(-math.pi/2, 0, 0)

def test_ahead_of_object():
    ego = sampleEgoFrom(
                        'target = Object facing (-90 deg, 0, 0)\n'
                        'ego = Object ahead of target'
                       )
    assert tuple(ego.position) == pytest.approx((1, 0, 0))
    assert ego.orientation = Orientation.fromEuler(-math.pi/2, 0, 0)

def test_behind_vector():
    ego = sampleEgoFrom('ego = Object behind 10 @ 20, facing 90 deg')
    assert tuple(ego.position) == pytest.approx((10.5, 20, 0))
    ego = sampleEgoFrom('ego = Object behind 10 @ 20, with length 10')
    assert tuple(ego.position) == pytest.approx((10, 15, 0))

def test_behind_vector_3d():
    ego = sampleEgoFrom('ego = Object behind (10, 20, 15), facing (90 deg, 0, 90 deg)')
    assert tuple(ego.position) == pytest.approx((10.5, 20, 15))
    ego = sampleEgoFrom('ego = Object behind (10, 20, 15), facing (90 deg, 0, 90 deg), with length 10')
    assert tuple(ego.position) == pytest.approx((15, 20, 15))

def test_behind_vector_by():
    ego = sampleEgoFrom('ego = Object behind 10 @ 20 by 20')
    assert tuple(ego.position) == pytest.approx((10, -0.5, 0))
    ego = sampleEgoFrom('ego = Object behind 10 @ 20 by 20 @ 5')
    assert tuple(ego.position) == pytest.approx((30, 14.5, 0))

def test_behind_vector_by_3d():
    ego = sampleEgoFrom('ego = Object behind (10, 20, 15) by 20')
    assert tuple(ego.position) == pytest.approx((10, -0.5, 15))
    ego = sampleEgoFrom('ego = Object behind (10, 20, 15) by 20, facing (90 deg, 0, 90 deg), with length 10')
    assert tuple(ego.position) == pytest.approx((35, 20, 15))

def test_behind_oriented_point():
    ego = sampleEgoFrom(
                        'target = OrientedPoint facing (-90 deg, 0, 0)\n'
                        'ego = Object behind target'
                       )
    assert tuple(ego.position) == pytest.approx((-0.5, 0, 0))
    assert ego.orientation = Orientation.fromEuler(-math.pi/2, 0, 0)

def test_behind_object():
    ego = sampleEgoFrom(
                        'target = Object facing (-90 deg, 0, 0)\n'
                        'ego = Object behind target'
                       )
    assert tuple(ego.position) == pytest.approx((-1, 0, 0))
    assert ego.orientation = Orientation.fromEuler(-math.pi/2, 0, 0)

# Above/Below
def test_above_vector_3d():
    ego = sampleEgoFrom('ego = Object above (10, 20, 15)')
    assert tuple(ego.position) == pytest.approx((10, 20, 15.5))
    ego = sampleEgoFrom('ego = Object above (10, 20, 15), facing (90 deg, 0, 90 deg), with height 10')
    assert tuple(ego.position) == pytest.approx((10, 25, 15))

def test_above_vector_by_3d():
    ego = sampleEgoFrom('ego = Object above (10, 20, 15) by 20')
    assert tuple(ego.position) == pytest.approx((10, 20, 35.5))
    ego = sampleEgoFrom('ego = Object above (10, 20, 15) by 20, facing (90 deg, 0, 90 deg), with height 10')
    assert tuple(ego.position) == pytest.approx((10, 45, 15))

def test_above_oriented_point():
    ego = sampleEgoFrom(
                        'target = OrientedPoint facing (0, 90 deg, 0)\n'
                        'ego = Object above target'
                       )
    assert tuple(ego.position) == pytest.approx((0, -0.5, 0))
    assert ego.orientation = Orientation.fromEuler(0, math.pi/2, 0)

def test_above_object():
    ego = sampleEgoFrom(
                        'target = Object facing (0, 90 deg, 0)\n'
                        'ego = Object above target'
                       )
    assert tuple(ego.position) == pytest.approx((0, -1, 0))
    assert ego.orientation = Orientation.fromEuler(0, math.pi/2, 0)

def test_below_vector_3d():
    ego = sampleEgoFrom('ego = Object below (10, 20, 15)')
    assert tuple(ego.position) == pytest.approx((10, 20, 14.5))
    ego = sampleEgoFrom('ego = Object below (10, 20, 15), facing (90 deg, 0, 90 deg), with height 10')
    assert tuple(ego.position) == pytest.approx((10, 15, 15))

def test_below_vector_by_3d():
    ego = sampleEgoFrom('ego = Object below (10, 20, 15) by 20')
    assert tuple(ego.position) == pytest.approx((10, 20, -5.5))
    ego = sampleEgoFrom('ego = Object below (10, 20, 15) by 20, facing (90 deg, 0, 90 deg), with height 10')
    assert tuple(ego.position) == pytest.approx((10, -5, 15))

def test_above_oriented_point():
    ego = sampleEgoFrom(
                        'target = OrientedPoint facing (0, 90 deg, 0)\n'
                        'ego = Object below target'
                       )
    assert tuple(ego.position) == pytest.approx((0, 0.5, 0))
    assert ego.orientation = Orientation.fromEuler(0, math.pi/2, 0)

def test_above_object():
    ego = sampleEgoFrom(
                        'target = Object facing (0, 90 deg, 0)\n'
                        'ego = Object below target'
                       )
    assert tuple(ego.position) == pytest.approx((0, 1, 0))
    assert ego.orientation = Orientation.fromEuler(0, math.pi/2, 0)

# Beyond
def test_beyond():
    ego = sampleEgoFrom(
        'ego = Object at 10 @ 5\n'
        'ego = Object beyond 4 @ 13 by 5'
    )
    assert tuple(ego.position) == pytest.approx((1, 17, 0))
    ego = sampleEgoFrom(
        'ego = Object at 10 @ 5\n'
        'ego = Object beyond 4 @ 13 by 10 @ 5'
    )
    assert tuple(ego.position) == pytest.approx((9, 23, 0))

def test_beyond_3d():
    ego = sampleEgoFrom("""
        import math
        ego = Object at (10, 5, 15)
        ego = Object beyond (11, 6, 15 + math.sqrt(2)) by (0, 10, 0)
    """)
    assert tuple(ego.position) == pytest.approx((16, 11, 15 + math.sqrt(2) + (10 / math.sqrt(2))))

def test_beyond_no_ego():
    with pytest.raises(RuntimeParseError):
        compileScenic('ego = Object beyond 10 @ 10 by 5')

def test_beyond_from():
    ego = sampleEgoFrom('ego = Object beyond 5 @ 0 by 20 from 5 @ 10')
    assert tuple(ego.position) == pytest.approx((5, -20, 0))
    ego = sampleEgoFrom('ego = Object beyond 5 @ 0 by 15 @ 20 from 5 @ 10')
    assert tuple(ego.position) == pytest.approx((-10, -20, 0))

def test_beyond_from_3d():
    ego = sampleEgoFrom("""
        import math
        ego = Object at (10, 5, 15)
        ego = Object beyond (11, 6, 15 + math.sqrt(2)) by (15, 20, 5) from (5, 10, 20)
    """)
    assert tuple(ego.position) == pytest.approx((16, 11, 15 + math.sqrt(2) + (10 / math.sqrt(2))))

    #ego = sampleEgoFrom('ego = Object beyond (5, 15, 10) by (15, 20, 5) from (5, 10, 20)')
    #assert tuple(ego.position) == pytest.approx((-10, -20, 0))

# Visible
def test_visible():
    scenario = compileScenic(
        'ego = Object at 100 @ 200, facing -45 deg,\n'
        '             with visibleDistance 10, with viewAngle 90 deg\n'
        'ego = Object visible'
    )
    for i in range(30):
        scene = sampleScene(scenario, maxIterations=50)
        ego, base = scene.objects
        assert ego.position.distanceTo(base.position) <= 10
        assert ego.position.x >= base.position.x
        assert ego.position.y >= base.position.y
    
def test_visible_no_ego():
    with pytest.raises(RuntimeParseError):
        compileScenic('ego = Object visible')

def test_visible_from_point():
    scenario = compileScenic(
        'x = Point at 300@200, with visibleDistance 2\n'
        'ego = Object visible from x'
    )
    for i in range(30):
        scene = sampleScene(scenario, maxIterations=1)
        assert scene.egoObject.position.distanceTo(Vector(300, 200)) <= 2

def test_visible_from_point_3d():
    scenario = compileScenic(
        'x = Point at (300, 200, 500), with visibleDistance 2\n'
        'ego = Object visible from x'
    )
    for i in range(30):
        scene = sampleScene(scenario, maxIterations=1)
        assert scene.egoObject.position.distanceTo(Vector(300, 200, 500)) <= 2

def test_visible_from_oriented_point():
    scenario = compileScenic(
        'op = OrientedPoint at 100 @ 200, facing 45 deg,\n'
        '                   with visibleDistance 5, with viewAngle 90 deg\n'
        'ego = Object visible from op'
    )
    base = Vector(100, 200)
    for i in range(30):
        scene = sampleScene(scenario, maxIterations=1)
        pos = scene.egoObject.position
        assert pos.distanceTo(base) <= 5
        assert pos.x <= base.x
        assert pos.y >= base.y

def test_visible_from_oriented_point_3d():
    scenario = compileScenic(
        'op = OrientedPoint at (100, 200, 300), facing (90 deg, 0 deg, 90 deg),\n'
        '                   with visibleDistance 5, with viewAngle 90 deg\n'
        'ego = Object visible from op'
    )
    base = Vector(100, 200, 300)
    for i in range(30):
        scene = sampleScene(scenario, maxIterations=1)
        pos = scene.egoObject.position
        assert pos.distanceTo(base) <= 5
        assert pos.x <= base.x
        assert pos.y >= base.y
        assert pos.z >= base.z

# In
def test_in():
    scenario = compileScenic(
        'r = RectangularRegion(100 @ 200, 90 deg, 50, 10)\n'
        'ego = Object in r'
    )
    for i in range(30):
        scene = sampleScene(scenario, maxIterations=1)
        pos = scene.egoObject.position
        assert 95 <= pos.x <= 105
        assert 150 <= pos.y <= 250
        assert scene.egoObject.heading == 0

def test_in_3d():
    scenario = compileScenic(
        'region = BoxRegion(dimensions=(10,20,30))\n'
        'ego = Object in region'
    )
    for i in range(30):
        scene = sampleScene(scenario)
        pos = scene.egoObject.position
        assert -5 <= pos.x <= 5
        assert -10 <= pos.y <= 10
        assert -15 <= pos.z <= 15
        assert scene.egoObject.orientation == Orientation.fromEuler(0,0,0)

def test_in_3d_heading():
    scenario = compileScenic(
        'vf = VectorField("TestVF", lambda pos: (0.5, 0.6, 0.7))\n'
        'region = BoxRegion(dimensions=(10,20,30), orientation=vf)\n'
        'ego = Object in region'
    )
    for i in range(30):
        scene = sampleScene(scenario)
        pos = scene.egoObject.position
        assert -5 <= pos.x <= 5
        assert -10 <= pos.y <= 10
        assert -15 <= pos.z <= 15
        assert scene.egoObject.orientation.eulerAngles == pytest.approx(0.5,0.6,0.7)

def test_in_heading():
    scenario = compileScenic(
        'r = PolylineRegion([50 @ -50, -20 @ 20])\n'
        'ego = Object in r'
    )
    for i in range(30):
        scene = sampleScene(scenario, maxIterations=1)
        pos = scene.egoObject.position
        assert -20 <= pos.x <= 50
        assert -50 <= pos.y <= 50
        assert pos.x == pytest.approx(-pos.y)
        assert scene.egoObject.heading == pytest.approx(math.radians(45))

def test_in_mistyped():
    with pytest.raises(RuntimeParseError):
        compileScenic('ego = Object in 3@2')

def test_in_distribution():
    scenario = compileScenic(
        'ra = RectangularRegion(0@0, 0, 2, 2)\n'
        'rb = RectangularRegion(10@0, 0, 2, 2)\n'
        'ego = Object in Uniform(ra, rb)'
    )
    xs = [sampleEgo(scenario).position.x for i in range(60)]
    assert all(-1 <= x <= 1 or 9 <= x <= 11 for x in xs)
    assert any(x < 5 for x in xs)
    assert any(x > 5 for x in xs)

def test_in_heading_distribution():
    scenario = compileScenic(
        'ra = RectangularRegion(0@0, 0, 2, 2)\n'
        'ra.orientation = VectorField("foo", lambda pt: 1)\n'
        'rb = PolylineRegion([0 @ 0, 1 @ 1])\n'
        'ego = Object in Uniform(ra, rb)'
    )
    hs = [sampleEgo(scenario).heading for i in range(60)]
    h2 = pytest.approx(-math.pi/4)
    assert all(h == 1 or h == h2 for h in hs)
    assert any(h == 1 for h in hs)
    assert any(h == h2 for h in hs)

# Contained In
def test_contained_in_2d():
    scenario = compileScenic(
        'region = RectangularRegion(0@0, 0, 2, 2))\n'
        'ego = Object contained in region'
    )
    for _ in range(30):
        pos = scene.egoObject.position
        assert -0.5 <= pos.x <= 0.5
        assert -0.5 <= pos.y <= 0.5
        assert pos.z == 0

def test_contained_in_3d():
    scenario = compileScenic(
        'region = BoxRegion(dimensions=(1,1,1))\n'
        'ego = Object contained in region'
    )
    for _ in range(30):
        pos = scene.egoObject.position
        assert -0.5 <= pos.x <= 0.5
        assert -0.5 <= pos.y <= 0.5
        assert -0.5 <= pos.z <= 0.5

# On
def test_on_3d():
    scenario = compileScenic(
        'region = BoxRegion(dimensions=(10,20,30))\n'
        'ego = Object on region'
    )
    for i in range(30):
        scene = sampleScene(scenario)
        pos = scene.egoObject.position
        assert -5 <= pos.x <= 5
        assert -10 <= pos.y <= 10
        assert -15 <= pos.z <= 15
        assert scene.egoObject.orientation == Orientation.fromEuler(0,0,0)

def test_on_3d_heading():
    scenario = compileScenic(
        'vf = VectorField("TestVF", lambda pos: (0.5, 0.6, 0.7))\n'
        'region = BoxRegion(dimensions=(10,20,30), orientation=vf)\n'
        'ego = Object on region'
    )
    for i in range(30):
        scene = sampleScene(scenario)
        pos = scene.egoObject.position
        assert -5 <= pos.x <= 5
        assert -10 <= pos.y <= 10
        assert -15 <= pos.z <= 15
        assert scene.egoObject.orientation.eulerAngles == pytest.approx(0.5,0.6,0.7)

def test_on_modifying():
    scenario = compileScenic(
        'floor = Object at (0,0,0), with shape BoxShape(dimensions=(40,40,0.1))\n'
        'air_region = BoxRegion(dimensions=(30,30,30), position=(0,0,15))\n'
        'ego = Object in air_region, on floor'
    )
    for i in range(30):
        scene = sampleScene(scenario)
        pos = scene.egoObject.position
        assert -15 <= pos.x <= 15
        assert -15 <= pos.y <= 15
        assert pos.z == pytest.approx(0.5, abs=0.001)

def test_on_mistyped():
    with pytest.raises(RuntimeParseError):
        compileScenic('ego = Object on 3@2')

# Following
def test_following():
    ego = sampleEgoFrom("""
        vf = VectorField("Foo", lambda pos: 90 deg * (pos.x + pos.y - 1),
                         minSteps=4, defaultStepSize=1)
        ego = Object following vf from 1@1 for 4
    """)
    assert tuple(ego.position) == pytest.approx((-1, 3, 0))
    assert ego.heading == pytest.approx(math.radians(90))

def test_following_random():
    ego = sampleEgoFrom("""
        vf = VectorField('Foo', lambda pos: -90 deg)
        x = Range(1, 2)
        ego = Object following vf from 1@2 for x, facing x
    """)
    assert tuple(ego.position) == pytest.approx((1+ego.heading, 2, 0))

## Orientation Specifiers ##

# Facing
def test_facing_heading():
    ego = sampleEgoFrom('ego = Object facing 90 deg')
    assert ego.heading == math.radians(90)

def test_facing_orientation():
    ego = sampleEgoFrom('ego = Object facing (90 deg, 90 deg, 90 deg)')
    assert ego.orientation == Orientation.fromEuler(math.pi/2, math.pi/2, math.pi/2)

def test_facing_vf_2d():
    ego = sampleEgoFrom(        
            'vf = VectorField("Foo", lambda pos: (100 * pos.x + 10 * pos.y) deg\n'
            'ego = Object at 1@5'
            )
    assert ego.heading == math.radians(150)

def test_facing_vf_3d():
    ego = sampleEgoFrom(        
            'vf = VectorField("Foo", lambda pos: (pos.x deg, pos.y deg, pos.z deg)\n'
            'ego = Object at (1, 5, 3)'
            )
    assert ego.orientation == Orientation.fromEuler(math.radians(1), math.radians(5), math.radians(3))

# Facing Toward/Away From
def test_facing_toward():
    ego = sampleEgoFrom("ego = Object facing toward (math.sqrt(0.5),math.sqrt(0.5),1), with pitch 0")
    assert ego.heading == pytest.approx(math.radians(45))
    assert ego.pitch == 0
    assert ego.roll == 0

def test_facing_away_from():
    ego = sampleEgoFrom("ego = Object facing away from (math.sqrt(0.5),math.sqrt(0.5),1), with pitch 0")
    assert ego.heading == pytest.approx(math.radians(135))
    assert ego.pitch == 0
    assert ego.roll == 0

# Facing directly Toward/Away From
def test_facing_directly_toward():
    ego = sampleEgoFrom("ego = Object facing toward (math.sqrt(0.5),math.sqrt(0.5),1)")
    assert ego.yaw == pytest.approx(math.radians(45))
    assert ego.pitch == pytest.approx(math.radians(45))
    assert ego.roll == 0

def test_facing_directly_away_from():
    ego = sampleEgoFrom("ego = Object facing away from (math.sqrt(0.5),math.sqrt(0.5),1)")
    assert ego.yaw == pytest.approx(math.radians(225))
    assert ego.pitch == pytest.approx(math.radians(225))
    assert ego.roll == 0

# Apparently Facing
def test_apparently_facing():
    ego = sampleEgoFrom('ego = Object at (0,0,0)\n'
                        'ego = Object at (1,1,0), apparently facing 90 deg'
                        )
    assert ego.heading == pytest.approx(math.radians(135))

def test_apparently_facing_from():
    ego = sampleEgoFrom('ego = Object at (1,1,0), apparently facing 90 deg from (0,0,0)')
    assert ego.heading == pytest.approx(math.radians(135))

