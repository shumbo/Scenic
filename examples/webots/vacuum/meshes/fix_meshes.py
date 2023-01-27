import trimesh
from scenic.core.regions import MeshVolumeRegion
import math

with open("plane.obj", "r") as mesh_file:
    dining_table_mesh = trimesh.load(mesh_file, file_type="obj")

print(trimesh.repair.broken_faces(dining_table_mesh))

trimesh.repair.fill_holes(dining_table_mesh)

print(dining_table_mesh.is_watertight)

dining_table_mesh = MeshVolumeRegion(mesh=dining_table_mesh, rotation=(-math.radians(90),0,-math.radians(10))).mesh

dining_table_mesh.show()
