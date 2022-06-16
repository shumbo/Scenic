import trimesh

# Pick a workspace
workspace_region = MeshVolumeRegion(trimesh.creation.box((1,1,1)), dimensions=(20,20,20))

workspace = Workspace(workspace_region)

# Place an ego object at the origin
ego = Object at Vector(0,0,0),
        with width 1,
        with length 1,
        with height 1

# Place many small boxes centered on the surface of a sphere
sample_space = MeshSurfaceRegion(trimesh.creation.icosphere(radius=10))

for i in range(20):
    Object in sample_space,
        with width 0.1,
        with length 0.1,
        with height 0.1,
        with requireVisible False
