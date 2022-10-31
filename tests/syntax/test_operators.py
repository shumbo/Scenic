
import math
import pytest

from scenic.core.errors import RuntimeParseError
from tests.utils import compileScenic, sampleEgoFrom, sampleParamP, sampleParamPFrom

## Scalar operators

# Relative Heading
def test_relative_heading():
    p = sampleParamPFrom("""
        ego = Object facing 30 deg
        other = Object facing 65 deg, at 10@10
        param p = relative heading of other
    """)
    assert p == pytest.approx(math.radians(65 - 30))

def test_relative_orientation():
    p = sampleParamPFrom("""
        ego = Object facing (30 deg, 40 deg, 50 deg)
        other = Object facing (60 deg, 80 deg, 100 deg)
        param p = relative orientation of other
    """)
    assert p.eulerAngles() == pytest.approx(math.radians(60 - 30), math.radians(80 - 40), math.radians(100 - 50))

def test_relative_heading_no_ego():
    with pytest.raises(RuntimeParseError):
        compileScenic("""
            other = Object
            ego = Object at 2@2, facing relative heading of other
        """)

def test_relative_heading_from():
    ego = sampleEgoFrom('ego = Object facing relative heading of 70 deg from -10 deg')
    assert ego.heading == pytest.approx(math.radians(70 + 10))

# Apparent heading
def test_apparent_heading():
    p = sampleParamPFrom("""
        ego = Object facing 30 deg
        other = Object facing 65 deg, at 10@10
        param p = apparent heading of other
    """)
    assert p == pytest.approx(math.radians(65 + 45))

def test_apparent_heading_no_ego():
    with pytest.raises(RuntimeParseError):
        compileScenic("""
            other = Object
            ego = Object at 2@2, facing apparent heading of other
        """)

def test_apparent_heading_from():
    ego = sampleEgoFrom("""
        OP = OrientedPoint at 10@15, facing -60 deg
        ego = Object facing apparent heading of OP from 15@10
    """)
    assert ego.heading == pytest.approx(math.radians(-60 - 45))

# Angle
def test_angle():
    p = sampleParamPFrom("""
        ego = Object facing 30 deg
        other = Object facing 65 deg, at 10@10
        param p = angle to other
    """)
    assert p == pytest.approx(math.radians(-45))

def test_angle_3d():
    p = sampleParamPFrom("""
        ego = Object facing (30 deg, 0 deg, 30 deg)
        other = Object facing (65 deg, 0 deg, 65 deg), at (10, 10, 10)
        param p = angle to other
    """)
    assert p == pytest.approx(math.radians(-45))

def test_angle_no_ego():
    with pytest.raises(RuntimeParseError):
        compileScenic("""
            other = Object
            ego = Object at 2@2, facing angle to other
        """)

def test_angle_from():
    ego = sampleEgoFrom('ego = Object facing angle from 2@4 to 3@5')
    assert ego.heading == pytest.approx(math.radians(-45))

def test_angle_from_3d():
    ego = sampleEgoFrom('ego = Object facing angle from (2, 4, 5) to (3, 5, 6)')
    assert ego.heading == pytest.approx(math.radians(-45))

# Azimuth/Altitude
def test_azimuth_to():
    ego = sampleEgoFrom('ego = Object at (0,0,0)\n'
                        'ego = Object facing azimuth to (1, 1, 0)'
                        )
    assert ego.heading == pytest.approx(math.radians(45))

def test_azimuth_from_to():
    ego = sampleEgoFrom('ego = Object facing azimuth from (0,0,0) to (1, 1, 0)'
                        )
    assert ego.heading == pytest.approx(math.radians(45))

def test_altitude_to():
    ego = sampleEgoFrom('ego = Object at (0,0,0)\n'
                        'ego = Object facing azimuth to (1, 0, 1)'
                        )
    assert ego.heading == pytest.approx(math.radians(45))

def test_azimuth_from_to():
    ego = sampleEgoFrom('ego = Object facing azimuth from (0,0,0) to (1, 0, 1)')
    assert ego.heading == pytest.approx(math.radians(45))

# Distance
def test_distance():
    p = sampleParamPFrom("""
        ego = Object at 5@10
        other = Object at 7@-4
        param p = distance to other
    """)
    assert p == pytest.approx(math.hypot(7 - 5, -4 - 10))

def test_distance_3d():
    p = sampleParamPFrom("""
        ego = Object at (5, 10, 20)
        other = Object at (7, -4, 15)
        param p = distance to other
    """)
    assert p == pytest.approx(math.hypot(7 - 5, -4 - 10, 15 - 20))

def test_distance_no_ego():
    with pytest.raises(RuntimeParseError):
        compileScenic("""
            other = Object
            ego = Object at 2@2, facing distance to other
        """)

def test_distance_from():
    ego = sampleEgoFrom('ego = Object with wobble distance from -3@2 to 4@5')
    assert ego.wobble == pytest.approx(math.hypot(4 - -3, 5 - 2))

def test_distance_from_3d():
    ego = sampleEgoFrom('ego = Object with wobble distance from (-3, 2, 1) to (4, 5, 6)')
    assert ego.wobble == pytest.approx(math.hypot(4 - -3, 5 - 2, 6 - 1))

def test_distance_to_region():
    p = sampleParamPFrom("""
        r = CircularRegion(10@5, 3)
        ego = Object at 13@9
        param p = distance to r
    """)
    assert p == pytest.approx(2)

# Distance past

def test_distance_past():
    p = sampleParamPFrom("""
        ego = Object at 5@10, facing 45 deg
        other = Object at 2@3
        param p = distance past other
        """)
    assert p == pytest.approx(2 * math.sqrt(2))

def test_distance_past_of():
    p = sampleParamPFrom("""
        ego = Object at 1@2, facing 30 deg
        op = OrientedPoint at 3@-6, facing -45 deg
        param p = distance past ego of op
        """)
    assert p == pytest.approx(-3 * math.sqrt(2))

## Boolean operators ##

# Can See
def test_point_can_see_vector():
    p = sampleParamPFrom("""
        ego = Object
        pt = Point at 10@20, with visibleDistance 5
        param p = tuple([pt can see 8@19, pt can see 10@26])
    """)
    assert p == (True, False)

def test_point_can_see_object():
    p = sampleParamPFrom("""
        ego = Object with width 10, with length 10
        other = Object at 35@10
        pt = Point at 15@10, with visibleDistance 15
        param p = tuple([pt can see ego, pt can see other])
    """)
    assert p == (True, False)

def test_oriented_point_can_see_vector():
    p = sampleParamPFrom("""
        ego = Object facing -45 deg, with visibleDistance 5, with viewAngle 20 deg
        param p = tuple([ego can see 2@2, ego can see 4@4, ego can see 1@0])
    """)
    assert p == (True, False, False)

def test_oriented_point_can_see_object():
    p = sampleParamPFrom("""
        ego = Object facing -45 deg, with visibleDistance 5, with viewAngle 20 deg
        other = Object at 4@4, with width 2, with length 2
        other2 = Object at 4@0, with requireVisible False
        param p = tuple([ego can see other, ego can see other2])
    """)
    assert p == (True, False)

def test_can_see_occlusion():
    p = sampleParamPFrom("""
        workspace_region = RectangularRegion(0 @ 0, 0, 40, 40)
        workspace = Workspace(workspace_region)

        ego = Object with visibleDistance 30,
            at (0,0,1),
            with width 5,
            with length 5,
            with height 5,
            with pitch 45 deg,
            with viewAngles (340 deg, 60 deg),
            with rayDensity 5

        seeing_obj = Object at (0,10,5),
            with width 2,
            with height 2,
            with length 2,
            with name "seeingObject"

        Object at (0,5,4),
            with width 10,
            with length 0.5,
            with height 6,
            with name "wall",
            with occluding False

        param p = ego can see seeing_obj
    """)
    assert p == True

    p = sampleParamPFrom("""
        workspace_region = RectangularRegion(0 @ 0, 0, 40, 40)
        workspace = Workspace(workspace_region)

        ego = Object with visibleDistance 30,
            at (0,0,1),
            with width 5,
            with length 5,
            with height 5,
            with pitch 45 deg,
            with viewAngles (340 deg, 60 deg),
            with rayDensity 5

        seeing_obj = Object at (0,10,5),
            with width 2,
            with height 2,
            with length 2,
            with name "seeingObject"

        Object at (0,5,4),
            with width 10,
            with length 0.5,
            with height 6,
            with name "wall"

        param p = ego can see seeing_obj
    """)
    assert p == False

# In
def test_point_in_region_2d():
    p = sampleParamPFrom("""
        ego = Object
        reg = RectangularRegion(10@5, 0, 4, 2)
        ptA = Point at 11@4.5
        ptB = Point at 11@3.5
        ptC = Point at (11, 4.5, 1)
        param p = tuple([9@5.5 in reg, 9@7 in reg, (11, 4.5, -1) in reg, ptA in reg, ptB in reg, ptC in reg])
    """)
    assert p == (True, False, True, True, False, True)

def test_object_in_region_2d():
    p = sampleParamPFrom("""
        reg = RectangularRegion(10@5, 0, 4, 2)
        ego = Object at 11.5@5.5, with width 0.25, with length 0.25
        other_1 = Object at 9@4.5, with width 2.5
        other_2 = Object at (11.5, 5.5, 1), with width 2.5
        param p = tuple([ego in reg, other_1 in reg], other_2 in reg)
    """)
    assert p == (True, False, True)

def test_point_in_region_3d():
    p = sampleParamPFrom("""
        ego = Object
        reg = BoxRegion()
        ptA = Point at (0.25,0.25,0.25)
        ptB = Point at (1,1,1)
        param p = tuple([(0,0,0) in reg, (0.5,0.5,0.5) in reg, ptA in reg, ptB in reg])
    """)
    assert p == (True, True, True, False)

def test_object_in_region_3d():
    p = sampleParamPFrom("""
        ego = Object
        reg = BoxRegion(dimensions=(2,2,2))
        obj_1 = Object at (0,0,0)
        obj_2 = Object at (0.5, 0.5, 0.5)
        obj_3 = Object at (0.75, 0.75, 0.75)
        obj_4 = Object at (3,3,3)
        param p = tuple([obj_1 in reg, obj_2 in reg, obj_3 in reg, obj_4 in reg])
    """)
    assert p == (True, True, False, False)

## Heading operators

# At
def test_field_at_vector():
    ego = sampleEgoFrom("""
        vf = VectorField("Foo", lambda pos: (3 * pos.x) + pos.y)
        ego = Object facing (vf at 0.02 @ -1)
    """)
    assert ego.heading == pytest.approx((3 * 0.02) - 1)

def test_field_at_vector_3d():
    ego = sampleEgoFrom(        
            'vf = VectorField("Foo", lambda pos: (pos.x deg, pos.y deg, pos.z deg)\n'
            'ego = Object facing (vf at (1, 5, 3))'
            )
    assert ego.heading == Orientation.fromEuler(math.radians(1), math.radians(5), math.radians(3))

# Relative To
def test_heading_relative_to_field():
    ego = sampleEgoFrom("""
        vf = VectorField("Foo", lambda pos: 3 * pos.x)
        ego = Object at 0.07 @ 0, facing 0.5 relative to vf
    """)
    assert ego.heading == pytest.approx(0.5 + (3 * 0.07))

def test_field_relative_to_heading():
    ego = sampleEgoFrom("""
        vf = VectorField("Foo", lambda pos: 3 * pos.x)
        ego = Object at 0.07 @ 0, facing vf relative to 0.5
    """)
    assert ego.heading == pytest.approx(0.5 + (3 * 0.07))

def test_field_relative_to_field():
    ego = sampleEgoFrom("""
        vf = VectorField("Foo", lambda pos: 3 * pos.x)
        ego = Object at 0.07 @ 0, facing vf relative to vf
    """)
    assert ego.heading == pytest.approx(2 * (3 * 0.07))

def test_heading_relative_to_heading():
    ego = sampleEgoFrom('ego = Object facing 0.5 relative to -0.3')
    assert ego.heading == pytest.approx(0.5 - 0.3)

def test_heading_relative_to_heading_lazy():
    ego = sampleEgoFrom("""
        vf = VectorField("Foo", lambda pos: 0.5)
        ego = Object facing 0.5 relative to (0.5 relative to vf)
    """)
    assert ego.heading == pytest.approx(1.5)

def test_mistyped_relative_to():
    with pytest.raises(RuntimeParseError):
        compileScenic('ego = Object facing 0 relative to 1@2')

def test_mistyped_relative_to_lazy():
    with pytest.raises(RuntimeParseError):
        compileScenic("""
            vf = VectorField("Foo", lambda pos: 0.5)
            ego = Object facing 1@2 relative to (0 relative to vf)
        """)

def test_orientation_relative_to_orientation():
    ego = sampleEgoFrom("ego = Object facing (0, -90 deg, 0) relative to (90 deg, 90 deg, 90 deg)")
    assert ego.orientation == Orientation.fromEuler(0,0,0)

## Vector operators

# Relative To
def test_relative_to_vector():
    ego = sampleEgoFrom('ego = Object at 3@2 relative to -4@10')
    assert tuple(ego.position) == pytest.approx((-1, 12, 0))

def test_relative_to_vector_3d():
    ego = sampleEgoFrom('ego = Object at (3, 2, 1) relative to (-4, 10, 5)')
    assert tuple(ego.position) == pytest.approx((-1, 12, 6))

def test_relative_to_oriented_point():
    ego = sampleEgoFrom('op = OrientedPoint at (12,13,14) facing (90 deg, 0, 0)'
                        'ego = Object at ((1,0,0) relative to op)'
                        )
    assert tuple(ego.position) == pytest.approx((0,1,0))

# Offset By
def test_offset_by():
    ego = sampleEgoFrom('ego = Object at 3@2 offset by -4@10')
    assert tuple(ego.position) == pytest.approx((-1, 12, 0))

def test_offset_by_3d():
    ego = sampleEgoFrom('ego = Object at (3, 2, 1) offset by (-4, 10, 5)')
    assert tuple(ego.position) == pytest.approx((-1, 12, 6))

# Offset Along
def test_offset_along_heading():
    ego = sampleEgoFrom('ego = Object at 3@2 offset along 45 deg by -4@10')
    d = 1 / math.sqrt(2)
    assert tuple(ego.position) == pytest.approx((3 - 10*d - 4*d, 2 + 10*d - 4*d, 0))

def test_offset_along_heading_3d():
    ego = sampleEgoFrom('ego = Object at (3, 2, 5) offset along (90 deg, 0 deg, 90 deg) by (4, 10, 5)')
    assert tuple(ego.position) == pytest.approx((-7, 7, 1))

def test_offset_along_field():
    ego = sampleEgoFrom("""
        vf = VectorField("Foo", lambda pos: 3 deg * pos.x)
        ego = Object at 15@7 offset along vf by 2@-3
    """)
    d = 1 / math.sqrt(2)
    assert tuple(ego.position) == pytest.approx((15 + 3*d + 2*d, 7 - 3*d + 2*d, 0))

def test_offset_along_field_3d():
    ego = sampleEgoFrom("""
        vf = VectorField("Foo", lambda pos: 3 deg * pos.x) 
        ego = Object at (15, 7, 5) offset along vf by (2, -3, 4) 
    """)
    d = 1 / math.sqrt(2)
    assert tuple(ego.position) == pytest.approx((15 + 3*d + 2*d, 7 - 3*d + 2*d, 9))

# Follow
def test_follow():
    ego = sampleEgoFrom("""
        vf = VectorField("Foo", lambda pos: 90 deg * (pos.x + pos.y - 1),
                         minSteps=4, defaultStepSize=1)
        p = follow vf from 1@1 for 4
        ego = Object at p, facing p.heading
    """)
    assert tuple(ego.position) == pytest.approx((-1, 3, 0))
    assert ego.heading == pytest.approx(math.radians(90))

def test_follow_3d():
    ego = sampleEgoFrom("""
        vf = VectorField("Foo", lambda pos: 90 deg * (pos.x + pos.y - 1),
                         minSteps=4, defaultStepSize=1)
        p = follow vf from (1, 1, 1) for 4
        ego = Object at p, facing p.heading
    """)
    assert tuple(ego.position) == pytest.approx((-1, 3, 1))
    assert ego.heading == pytest.approx(math.radians(90))

## Region operators

# Visible
def test_visible():
    scenario = compileScenic("""
        ego = Object at 100 @ 200, facing -45 deg,
                     with visibleDistance 10, with viewAngle 90 deg
        reg = RectangularRegion(100@205, 0, 10, 20)
        param p = Point in visible reg
    """)
    for i in range(30):
        p = sampleParamP(scenario, maxIterations=100)
        assert p.x >= 100
        assert p.y >= 200
        assert math.hypot(p.x - 100, p.y - 200) <= 10

def test_not_visible():
    scenario = compileScenic("""
        ego = Object at 100 @ 200, facing -45 deg,
                     with visibleDistance 30, with viewAngle 90 deg
        reg = RectangularRegion(100@200, 0, 10, 10)
        param p = Point in not visible reg
    """)
    ps = [sampleParamP(scenario, maxIterations=100) for i in range(50)]
    assert all(p.x <= 100 or p.y <= 200 for p in ps)
    assert any(p.x > 100 for p in ps)
    assert any(p.y > 200 for p in ps)
