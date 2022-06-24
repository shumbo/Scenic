import trimesh

# Pick a workspace
workspace_region = MeshVolumeRegion(trimesh.creation.box((1,1,1)), dimensions=(20,20,20))

workspace = Workspace(workspace_region)

# Load chair mesh from file
with open(localPath("mesh.obj"), "r") as mesh_file:
    mesh = trimesh.load(mesh_file, file_type="obj")

# Create surface shape
surface_shape = MeshShape(mesh, dimensions=(5,5,5))

# Create large chair object
ego = Object in MeshVolumeRegion(trimesh.creation.box((1,1,1)), dimensions=(14,14,14)),
    with pitch 90 deg,
    with shape surface_shape

# Place a small cube on the chair
Object on ego,
    with requireVisible False
