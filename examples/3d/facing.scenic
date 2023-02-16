import trimesh

# Load plane mesh from file and create plane shape from it
with open(localPath("plane.obj"), "r") as mesh_file:
    plane_mesh = trimesh.load(mesh_file, file_type="obj")

plane_shape = MeshShape(mesh=plane_mesh, dimensions=(2,2,1), initial_rotation=(-90 deg, 0, -10 deg))

class Plane:
	shape: plane_shape

class Ball:
	shape: SpheroidShape()

ego = new Ball at (0,0, 1.25)
new Plane at (2,0,0), facing toward ego
new Plane at (-2,0,0), facing directly toward ego
