import trimesh

# Pick a workspace
workspace_region = RectangularRegion(0 @ 0, 0, 40, 40)

workspace = Workspace(workspace_region)

# Create floor region
floor = Object at (0,0, 0),
    with shape BoxShape(dimensions=(30,30,0.1))

floor.region.mesh.visual.face_colors = [30, 30, 150, 255]

# Load chair mesh from file
with open(localPath("mesh.obj"), "r") as mesh_file:
    mesh = trimesh.load(mesh_file, file_type="obj")

# Create surface shape
chair_shape = MeshShape(mesh, dimensions=(5,5,5))

# Create large chair object
chair = Object on floor,
    with pitch 90 deg,
    with shape chair_shape

ego = chair

# Place a small cube on the large chair
top_cube = Object on chair,
    with requireVisible False

# Place a small cube in the empty space of the large chair
bottom_cube = Object in chair.emptySpace, on floor,
    with requireVisible False
