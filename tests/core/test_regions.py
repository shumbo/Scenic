
import pytest
import math
import shapely.geometry

from scenic.core.regions import *

from tests.utils import sampleSceneFrom

def test_all_region():
    ar = AllRegion('all')
    assert ar == everywhere
    assert ar.containsPoint(Vector(347, -962.5))
    circ = CircularRegion(Vector(29, 34), 5)
    assert ar.intersect(circ) is circ
    assert circ.intersect(ar) is circ
    assert ar.intersects(circ)
    assert circ.intersects(ar)
    diff = ar.difference(circ)
    assert diff.containsPoint(Vector(0, 0))
    assert not diff.containsPoint(Vector(29, 34))
    assert circ.difference(ar) == nowhere
    assert ar.union(circ) is ar
    assert circ.union(ar) is ar
    assert ar.distanceTo(Vector(4, 12)) == 0

def test_empty_region():
    er = EmptyRegion('none')
    assert er == nowhere
    assert not er.containsPoint(Vector(0, 0))
    circ = CircularRegion(Vector(29, 34), 5)
    assert er.intersect(circ) is er
    assert circ.intersect(er) is er
    assert not er.intersects(circ)
    assert not circ.intersects(er)
    assert circ.difference(er) is circ
    assert er.difference(circ) is er
    assert er.union(circ) is circ
    assert circ.union(er) is circ
    assert er.distanceTo(Vector(4, 12)) == float('inf')

def test_circular_region():
    circ = CircularRegion(Vector(4, -3), 2)
    for pt in ((4, -3), (6, -3), (2, -3), (4, -5), (4, -1), (5, -2)):
        assert circ.containsPoint(Vector(*pt))
    for pt in ((6, -1), (6, -5), (2, -5), (2, -1), (6.1, -3)):
        assert not circ.containsPoint(Vector(*pt))
    circ2 = CircularRegion(Vector(1, -7), 3.1)
    assert circ.intersects(circ2)
    assert circ2.intersects(circ)
    circ3 = CircularRegion(Vector(1, -7), 2.9)
    assert not circ.intersects(circ3)
    assert not circ3.intersects(circ)
    assert circ.distanceTo(Vector(4, -3)) == 0
    assert circ.distanceTo(Vector(1, -7)) == pytest.approx(3)
    assert circ.getAABB() == ((2, -5), (6, -1))

def test_circular_sampling():
    center = Vector(4, -3)
    circ = CircularRegion(center, 2)
    pts = [circ.uniformPointInner() for i in range(3000)]
    dists = [pt.distanceTo(center) for pt in pts]
    assert all(dist <= 2 for dist in dists)
    assert sum(dist <= 1.414 for dist in dists) >= 1250
    assert sum(dist > 1.414 for dist in dists) >= 1250
    xs, ys, zs = zip(*pts)
    assert sum(x >= 4 for x in xs) >= 1250
    assert sum(y >= -3 for y in ys) >= 1250

def test_polygon_sampling():
    p = shapely.geometry.Polygon(
        [(0,0), (0,3), (3,3), (3,0)],
        holes=[[(1,1), (1,2), (2,2), (2,1)]]
    )
    r = PolygonalRegion(polygon=p)
    pts = [r.uniformPointInner() for i in range(3000)]
    for x, y, z in pts:
        assert 0 <= x <= 3 and 0 <= y <= 3 and z == 0
        assert not (1 < x < 2 and 1 < y < 2)
    xs, ys, zs = zip(*pts)
    assert sum(1 <= x <= 2 for x in xs) <= 870
    assert sum(1 <= y <= 2 for y in ys) <= 870
    assert sum(x >= 1.5 for x in xs) >= 1250
    assert sum(y >= 1.5 for y in ys) >= 1250

def test_mesh_operation_blender():
    r1 = BoxRegion(position=(0,0,0), dimensions=(1,1,1), engine="blender")
    r2 = BoxRegion(position=(0,0,0), dimensions=(2,2,2), engine="blender")

    r = r1.intersect(r2)

def test_mesh_operation_scad():
    r1 = BoxRegion(position=(0,0,0), dimensions=(1,1,1), engine="scad")
    r2 = BoxRegion(position=(0,0,0), dimensions=(2,2,2), engine="scad")

    r = r1.intersect(r2)

def test_mesh_volume_region_sampling():
    r = BoxRegion(position=(0,0,0), dimensions=(2,2,2))
    pts = [r.uniformPointInner() for _ in range(3000)]

    for x, y, z in pts:
        assert -1 <= x <= 1
        assert -1 <= y <= 1
        assert -1 <= z <= 1

def test_mesh_surface_region_sampling():
    r = BoxRegion(position=(0,0,0), dimensions=(2,2,2)).getSurfaceRegion()
    pts = [r.uniformPointInner() for _ in range(3000)]

    for x, y, z in pts:
        assert  x == 1 or x == -1 or \
                y == 1 or y == -1 or \
                z == 1 or z == -1

def test_mesh_region_distribution():
    sampleSceneFrom("""
        position = (Range(-5,5), Range(-5,5), Range(-5,5))
        radius = Range(1,5)
        dimensions = (2*radius, 2*radius, 2*radius)
        rotation = (Range(0,360), Range(0,360), Range(0,360))

        region = SpheroidRegion(position=position, dimensions=dimensions, rotation=rotation)

        ego = new Object in region
    """)

def test_mesh_polygon_intersection():
    r1 = BoxRegion(position=(0,0,0), dimensions=(3,3,2))
    r2 = CircularRegion((0,0), 1, resolution=64)

    r = r1.intersect(r2)

    assert isinstance(r, MeshVolumeRegion)

    v_pts = list(trimesh.sample.volume_mesh(r.mesh, 3000))
    s_pts = [r.getSurfaceRegion().uniformPointInner() for _ in range(3000)]

    for x, y, z in v_pts:
        assert math.hypot(x, y) <= 1
        assert -1 <= z <= 1
        assert r1.containsPoint((x,y,z))
        assert r2.containsPoint((x,y,z))

    for x, y, z in s_pts:
        on_side = math.hypot(x, y) == pytest.approx(1, abs=1e-4)
        on_top_bottom = z == -1 or z == 1

        assert on_side or on_top_bottom

def test_mesh_polygons_intersection():
    p1 = shapely.geometry.Polygon(
        [(1,1), (1,2), (2,2), (2,1)]
    )
    r1 = PolygonalRegion(polygon=p1)

    p2 = shapely.geometry.Polygon(
        [(-2,-2), (-2,-1), (-1,-1), (-1,-2)]
    )
    r2 = PolygonalRegion(polygon=p2)

    r3 = BoxRegion(dimensions=(5,3,5))

    r = r3.intersect(r1.union(r2))

    assert isinstance(r, MeshVolumeRegion)

def test_mesh_line_strings_intersection():
    point_lists = []

    for y in range(-5,6,2):
        point_lists.append([])
        
        for x in range(-5,6,2):
            target_list = point_lists[-1]

            target_list.append(numpy.array((x,y,0)))
            target_list.append(numpy.array((x,y+1,0)))
            target_list.append(numpy.array((x+1,y+1,0)))
            target_list.append(numpy.array((x+1,y,0)))

    r1 = PolylineRegion(polyline=shapely.ops.linemerge(point_lists))
    r2 = SpheroidRegion(dimensions=(5,5,5))

    r = r1.intersect(r2)

    for point in [r.uniformPointInner() for _ in range(3000)]:
        assert r.containsPoint(point)
        assert r1.containsPoint(point)
        assert r2.containsPoint(point)

def test_view_region_construction():
    sampleSceneFrom("""
        workspace_region = RectangularRegion(0 @ 0, 0, 40, 40)
        workspace = Workspace(workspace_region)

        sample_space = BoxRegion(dimensions=(30,30,30), position=(0,0,15))

        ego = new Object with visibleDistance 20,
            with width 5,
            with length 5,
            with height 5,
            with viewAngles (360 deg, 180 deg)

        new Object in sample_space,
            with width 1,
            with length 1,
            with height 1,
            facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
            with visibleDistance 5,
            with viewAngles (360 deg, 90 deg),
            with requireVisible True

        new Object in sample_space,
            with width 1,
            with length 1,
            with height 1,
            facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
            with viewAngles (180 deg, 180 deg),
            with requireVisible True,
            with cameraOffset (0,0,0.5)

        new Object in sample_space,
            with width 1,
            with length 1,
            with height 1,
            facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
            with visibleDistance 5,
            with viewAngles (180 deg, 90 deg),
            with requireVisible True

        new Object in sample_space,
            with width 1,
            with length 1,
            with height 1,
            facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
            with visibleDistance 5,
            with viewAngles (200 deg, 180 deg),
            with requireVisible True

        new Object in sample_space,
            with width 1,
            with length 1,
            with height 1,
            facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
            with visibleDistance 5,
            with viewAngles (20 deg, 180 deg),
            with requireVisible True

        new Object in sample_space,
            with width 1,
            with length 1,
            with height 1,
            facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
            with visibleDistance 5,
            with viewAngles (90 deg, 90 deg),
            with requireVisible True

        new Object in sample_space,
            with width 1,
            with length 1,
            with height 1,
            facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
            with visibleDistance 5,
            with viewAngles (200 deg, 40 deg),
            with requireVisible True
    """, maxIterations=1000)
