import trimesh

with open(localPath("mesh.obj"), "r") as mesh_file:
    mesh = trimesh.load(mesh_file, file_type="obj")

mesh_shape = MeshShape(mesh)

class TestObj:
    """Test object"""

workspace = Workspace(MeshVolumeRegion(trimesh.creation.box((100, 100, 100))))

ego = TestObj in workspace,
    with width 5,
    with length 5,
    with height 5

TestObj in workspace,
    with width 10,
    with length 10,
    with height 10,
    with requireVisible False

TestObj in workspace,
    with shape mesh_shape,
    with requireVisible False
