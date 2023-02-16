import trimesh

# Load chair mesh from file and create chair shape from it
with open(localPath("chair.obj"), "r") as mesh_file:
    chair_mesh = trimesh.load(mesh_file, file_type="obj")

chair_shape = MeshShape(chair_mesh, dimensions=(1,1,1), initial_rotation=(0,90 deg,0))

class Chair:
	shape: chair_shape

floor = new Object with width 5, with length 5, with height 0.1
air_cube = new Object at (Range(-5,5), Range(-5,5), 3), 
	facing (Range(0,360 deg), Range(0,30 deg), 0)
new Chair below air_cube, with color (0,0,200)
ego = new Chair below air_cube, on floor
