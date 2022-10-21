import trimesh

# Pick a workspace
workspace_region = RectangularRegion(0 @ 0, 0, 40, 40)
workspace = Workspace(workspace_region)

ego = Object in workspace,
    with visibleDistance 20,
    with width 5,
    with length 5,
    with height 5,
    facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
    with viewAngles (120 deg, 90 deg)

obj_1 = Object visible,
    with width 5,
    with length 5,
    with height 5,
    facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
    with viewAngles (120 deg, 90 deg)

obj_2 = Object visible from obj_1,
    with width 5,
    with length 5,
    with height 5,
    facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
    with viewAngles (120 deg, 90 deg)

#obj_3 = Object not visible from obj_2,
#    with width 5,
#    with length 5,
#    with height 5,
#    facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
#    with viewAngles (120 deg, 90 deg)


#test_point = Point visible from obj_2

#obj_4 = Object at test_point,
#    with shape MeshShape(trimesh.creation.cone(radius=0.5, height=1)),
#    with width 5,
#    with length 5,
#    with height 5,
#    facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
#    with viewAngles (120 deg, 90 deg)
