"""
Generate a room for the iroomba create vacuum
"""

model scenic.simulators.webots.model

import numpy as np
import trimesh

class Vacuum(WebotsObject):
	webotsName: "IROBOT_CREATE"

class Floor(WebotsObject):
	webotsAdhoc: True
	height: 0.1
	shape: MeshShape(mesh=trimesh.creation.box((1,1,1)))

# Create floor and set it to a brown color
floor = new Floor at (0,0,0),
		with width 10,
		with length 10

floor.shape.mesh.visual.face_colors = [58,42,4,255]


ego = new Vacuum on floor.topSurface
