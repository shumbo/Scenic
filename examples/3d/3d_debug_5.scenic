import trimesh

# Pick a workspace
workspace_region = RectangularRegion(0 @ 0, 0, 40, 40)
workspace = Workspace(workspace_region)

sample_space = BoxRegion(dimensions=(30,30,30), position=(0,0,15))

# Create an object at the origin who's vision cone should extend exactly
# to the edges of the workspace.
ego = Object with visibleDistance 20,
    with width 5,
    with length 5,
    with height 5,
    with viewAngle (360 deg, 360 deg)

Object in sample_space,
    with width 1,
    with length 1,
    with height 1,
    facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
    with visibleDistance 5,
    with viewAngle (360 deg, 45 deg)

Object in sample_space,
    with width 1,
    with length 1,
    with height 1,
    facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
    with visibleDistance 5,
    with viewAngle (120 deg, 90 deg)

Object in sample_space,
    with width 1,
    with length 1,
    with height 1,
    facing (Range(0,360) deg, Range(0,360) deg, Range(0,360) deg),
    with visibleDistance 5,
    with viewAngle (180 deg, 180 deg)

