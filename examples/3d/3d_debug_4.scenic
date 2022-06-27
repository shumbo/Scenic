import trimesh

# Pick a workspace
workspace_region = RectangularRegion(0 @ 0, 0, 40, 40)
workspace = Workspace(workspace_region)

sample_space = BoxRegion(dimensions=(30,30,30))

## Boxes around Oriented Point ##
center_point = Point in sample_space

# Create small boxes on all sides
Object left of center_point by 1,
    with width 1,
    with length 1,
    with height 1

Object right of center_point by 1,
    with width 1,
    with length 1,
    with height 1

Object above center_point by 1,
    with width 1,
    with length 1,
    with height 1

Object below center_point by 1,
    with width 1,
    with length 1,
    with height 1

Object ahead of center_point by 1,
    with width 1,
    with length 1,
    with height 1

Object behind center_point by 1,
    with width 1,
    with length 1,
    with height 1

## Boxes around Oriented Point ##
center_point = OrientedPoint in sample_space,
    facing (10 deg, 10 deg, 10 deg)

# Create small boxes on all sides
Object left of center_point by 1,
    with width 1,
    with length 1,
    with height 1

Object right of center_point by 1,
    with width 1,
    with length 1,
    with height 1

Object above center_point by 1,
    with width 1,
    with length 1,
    with height 1

Object below center_point by 1,
    with width 1,
    with length 1,
    with height 1

Object ahead of center_point by 1,
    with width 1,
    with length 1,
    with height 1

Object behind center_point by 1,
    with width 1,
    with length 1,
    with height 1

## Boxes around Object ##
# Create big ego box.
ego = Object in sample_space,
    with width 5,
    with length 5,
    with height 5,
    facing (10 deg, 10 deg, 10 deg)

center_point = ego

# Create small boxes on all sides
Object left of center_point,
    with width 1,
    with length 1,
    with height 1

Object right of center_point,
    with width 1,
    with length 1,
    with height 1

Object above center_point,
    with width 1,
    with length 1,
    with height 1

Object below center_point,
    with width 1,
    with length 1,
    with height 1

Object ahead of center_point,
    with width 1,
    with length 1,
    with height 1

Object behind center_point,
    with width 1,
    with length 1,
    with height 1

