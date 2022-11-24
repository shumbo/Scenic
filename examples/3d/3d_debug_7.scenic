import trimesh

# Pick a workspace
workspace_region = RectangularRegion(0 @ 0, 0, 40, 40)
workspace = Workspace(workspace_region)

ego = new Object in workspace,
    with visibleDistance 20,
    with width 5,
    with length 5,
    with height 5,
    facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
    with viewAngles (120 deg, 90 deg)

obj_1 = new Object visible,
    with width 5,
    with length 5,
    with height 5,
    facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
    with viewAngles (120 deg, 90 deg)

obj_2 = new Object visible from obj_1,
    with width 5,
    with length 5,
    with height 5,
    facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
    with viewAngles (120 deg, 90 deg)

#obj_3 = new Object not visible from obj_2,
#    with width 5,
#    with length 5,
#    with height 5,
#    facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
#    with viewAngles (120 deg, 90 deg)

test_point = new Point visible from obj_2

obj_4 = new Object at test_point,
    with shape MeshShape(trimesh.creation.cone(radius=0.5, height=1)),
    with width 5,
    with length 5,
    with height 5,
    facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
    with viewAngles (120 deg, 90 deg)

require obj_2 can see test_point
