"""
Generate a room for the iroomba create vacuum
"""

model scenic.simulators.webots.model

class Vacuum(WebotsObject):
	webotsName: "IROBOT_CREATE"

class Floor(WebotsObject):
	webotsAdhoc: True
	height: 0.1
	shape: BoxShape()

floor = new Floor at (0,0,0),
		with width 10,
		with length 10

ego = new Vacuum on floor.topSurface
