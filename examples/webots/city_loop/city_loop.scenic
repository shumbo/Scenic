"""
Generate a city intersection driving scenario, an intersection
of two 2-lane one way roads in a city.
"""

import shapely

model scenic.simulators.webots.model

class EgoCar(WebotsObject):
	webotsName: "EGO"
	width: 2.3
	length: 5
	height: 1.9
	positionOffset: (-0.33, 0, -0.5)
	orientationOffset: (90 deg, 0, 0)
	yaw: Range(-5 deg, 5 deg)

class Prius(WebotsObject):
	webotsName: "PRIUS"
	width: 2.2
	length: 4.6
	height: 1.7
	positionOffset: (-1.4, 0, -0.5)
	orientationOffset: (90 deg, 0, 0)
	yaw: 90 deg + Range(-5 deg, 5 deg)

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

# Create a region that represents both lanes of the bottom road, 
# with each one slightly narrowed to ensure car is relatively centered.
bottom_road_left_lane = RectangularRegion(-2.5@-55, 0, 4, 80, defaultZ=0.02)
bottom_road_right_lane = RectangularRegion(2.5@-55, 0, 4, 80, defaultZ=0.02)

ego_lane = Uniform(bottom_road_left_lane, bottom_road_right_lane)

# Place the ego car in one of the lanes, and ensure it is fully contained.
ego = new EgoCar on ego_lane, with regionContainedIn ego_lane

# Create a region that represents both lanes of the crossing road.
crossing_road_left_lane = RectangularRegion(0@2.5, 0, 160, 4, defaultZ=0.02)
crossing_road_right_lane = RectangularRegion(0@-2.5, 0, 160, 4, defaultZ=0.02)

prius_lane = Uniform(crossing_road_left_lane, crossing_road_right_lane)

new Prius on prius_lane, with regionContainedIn prius_lane

# Create a region composed of all 4 quadrants around the road
top_right_quadrant = RectangularRegion(56@56, 0, 100, 100, defaultZ=0)
top_left_quadrant = RectangularRegion(-56@56, 0, 100, 100, defaultZ=0)
bottom_right_quadrant = RectangularRegion(56@-56, 0, 100, 100, defaultZ=0)
bottom_left_quadrant = RectangularRegion(-56@-56, 0, 100, 100, defaultZ=0)

quadrant_polygon = shapely.ops.unary_union([top_right_quadrant.polygon, top_left_quadrant.polygon, bottom_right_quadrant.polygon, bottom_left_quadrant.polygon])

building_region = PolygonalRegion(polygon=quadrant_polygon)

# TODO Add support interval support for lazy inradius so we can prune here!

for _ in range(3):
	new CommercialBuilding on building_region, with regionContainedIn building_region

for _ in range(1):
	new ResidentialBuilding on building_region, with regionContainedIn building_region

for _ in range(2):
	new GlassBuilding on building_region, with regionContainedIn building_region
