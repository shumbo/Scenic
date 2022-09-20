import trimesh

# Pick a workspace
workspace_region = RectangularRegion(0 @ 0, 0, 40, 40)
workspace = Workspace(workspace_region)

sample_space = BoxRegion(dimensions=(40,14,20), position=(0,12.5,10))

# Create an object at the origin who's vision cone should extend exactly
# to the edges of the workspace.
ego = Object with visibleDistance 30,
    at (0,0,1),
    with width 5,
    with length 5,
    with height 5,
    with pitch 45 deg,
    with viewAngle (45 deg, 90 deg),
    with viewRays (40, 40)

Object in sample_space,
    with width 2,
    with height 2,
    with length 2,
    with requireVisible True

Object at (0,5,4),
    with width 10,
    with length 0.5,
    with height 6
