"""
Generate a city intersection driving scenario, an intersection
of two 2-lane one way roads in a city.
"""

import shapely
import time

model scenic.simulators.webots.model

class EgoCar(WebotsObject):
	webotsName: "EGO"
	width: 2.3
	length: 5
	height: 1.9
	positionOffset: (-0.33, 0, -0.5)
	cameraOffset: (0.77, 0, 0.95)
	orientationOffset: (90 deg, 0, 0)
	viewAngles: (120 deg, 60 deg)
	visibleDistance: 100
	rayDensity: 10

class Car(WebotsObject):
	webotsName: "CAR"
	width: 2.3
	length: 5
	height: 1.9
	positionOffset: (-0.33, 0, -0.5)
	orientationOffset: (90 deg, 0, 0)
	viewAngles: (1, 60 deg)
	visibleDistance: 100

class CommercialBuilding(WebotsObject):
	webotsType: "BUILDING_COMMERCIAL"
	width: 22
	length: 22
	height: 100
	yaw: Uniform(1, 2, 3) * 90 deg
	positionOffset: (0, 0, -50)

class ResidentialBuilding(WebotsObject):
	webotsType: "BUILDING_RESIDENTIAL"
	width: 14.275
	length: 57.4
	height: 40
	yaw: 90 deg
	positionOffset: (0, 0, -20)

class GlassBuilding(WebotsObject):
	webotsType: "BUILDING_GLASS"
	width: 14.1
	length: 8.1
	height: 112
	yaw: Uniform(1, 2, 3) * 90 deg
	positionOffset: (0, 0, -56)

class LogImageAction(Action):
	def __init__(self, visible: bool, path: str, count: int):
		self.visible = visible
		self.path = path
		self.count = count

	def applyTo(self, obj, sim):
		print("Other Car Visible:", self.visible)
		
		target_path = self.path + "/"
		target_path += "visible" if self.visible else "invisible"
		target_path += "/" + str(self.count) + ".png"

		print("IMG Path:", target_path)

		camera_obj = simulation().supervisor.getFromDef("EGO").getCamera('camera')

		camera_obj.saveImage(target_path, quality=100)

behavior LogCamera(path):
	count = 0
	while True:
		count += 1
		# Log a picture every 50 ticks (half second)
		if count%50 == 0:
			visible = ego can see car
			take LogImageAction(visible, path, count)
		else:
			wait

# Create a region that represents both lanes of the crossing road.
crossing_road_lane = RectangularRegion(0@0, 0, 160, 5, defaultZ=0.02)

car = new Car facing 90 deg, on crossing_road_lane, with regionContainedIn crossing_road_lane
require car.x > 10

# Create a region that represents both lanes of the bottom road.
bottom_road_lane = RectangularRegion(0@-55, 0, 5, 80, defaultZ=0.02)

# Place the ego car in one of the lanes, and ensure it is fully contained.
ego = new EgoCar on bottom_road_lane, with regionContainedIn bottom_road_lane, with behavior LogCamera(localPath(f"images/{time.time_ns()}"))

# Create a region composed of all 4 quadrants around the road
top_right_quadrant = RectangularRegion(56@56, 0, 100, 100, defaultZ=0)
top_left_quadrant = RectangularRegion(-56@56, 0, 100, 100, defaultZ=0)
bottom_right_quadrant = RectangularRegion(56@-56, 0, 100, 100, defaultZ=0)
bottom_left_quadrant = RectangularRegion(-56@-56, 0, 100, 100, defaultZ=0)

quadrant_polygon = shapely.ops.unary_union([top_right_quadrant.polygon, top_left_quadrant.polygon])#, bottom_right_quadrant.polygon, bottom_left_quadrant.polygon])

building_region = PolygonalRegion(polygon=quadrant_polygon)

# Add buildings, some randomly, some designed to block visibility of the center road
for _ in range(1):
	new CommercialBuilding on building_region, with regionContainedIn building_region

for _ in range(2):
	new ResidentialBuilding on building_region, with regionContainedIn building_region

for _ in range(2):
	new GlassBuilding on building_region, with regionContainedIn building_region

new ResidentialBuilding at (-36, -21, 20)
new CommercialBuilding at (18, -20, 50), facing Range(-5,5) deg
new CommercialBuilding at (50, -22, 50), facing Range(-5,5) deg

# Require that the cars reach the intersection at relatively different times
require abs(ego.distanceTo(0@0) - car.distanceTo(0@0)) > 7.5

# Terminate the simulation after the ego has passed through the intersection
terminate when ego.position.y > 20
