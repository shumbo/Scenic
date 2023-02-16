"""
Generate a city intersection driving scenario, an intersection
of two 2-lane one way roads in a city.
"""

import shapely
import time
import shutil
import os

model scenic.simulators.webots.model

class EgoCar(WebotsObject):
	webotsName: "EGO"
	width: 2.3
	length: 5.25
	height: 1.9
	positionOffset: (-0.33, 0, -0.5)
	cameraOffset: (0.77, 0, 0.95)
	orientationOffset: (90 deg, 0, 0)
	viewAngles: (1.5, 60 deg)
	visibleDistance: 100
	rayDensity: 10

class Car(WebotsObject):
	webotsName: "CAR"
	width: 2.3
	length: 5.25
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

		if not os.path.exists(target_path):
			os.makedirs(target_path)

		target_path += "/" + str(self.count) + ".jpeg"

		print("IMG Path:", target_path)

		# Wait for other controller to write image
		time.sleep(0.001)
		attempts = 0
		while not os.path.exists(localPath("images/live_img.jpeg")):
			print("Waiting for image...")
			attempts += 1
			time.sleep(0.001)

			if attempts > 10:
				print("Could not move image...")
				return

		shutil.move(localPath("images/live_img.jpeg"), target_path)

behavior LogCamera(path):
	count = 0
	while True:
		visible = ego can see car
		take LogImageAction(visible, path, count)
		count += 1

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
new CommercialBuilding at (18 + Range(-1,1), -20 + Range(-1,1), 50), facing Range(-5,5) deg
new CommercialBuilding at (50 + Range(-1,1), -22 + Range(-1,1), 50), facing Range(-5,5) deg

# Terminate the simulation after the ego has passed through the intersection or a timeout is reached
terminate when ego.position.y > 0
terminate after 60 seconds

# Require that the ego can eventually see the crossing car, but not until it gets close.
require eventually (ego can see car)
require (not ego can see car) until (distance from ego to car < 75)

# Require that the cars do not crash
require always distance to car > 2
