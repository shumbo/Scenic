import trimesh

# Load chair mesh from file and create chair shape from it
with open(localPath("chair.obj"), "r") as mesh_file:
    chair_mesh = trimesh.load(mesh_file, file_type="obj")

chair_shape = MeshShape(chair_mesh, dimensions=(1,1,1), initial_rotation=(0,90 deg,0))

class Chair:
	shape: chair_shape

floor = new Object with width 10, with length 10, with height 0.1
ego = new Chair on floor
