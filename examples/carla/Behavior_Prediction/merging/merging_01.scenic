"""
TITLE: Behavior Prediction - Merging 01
AUTHOR: Francis Indaheng, findaheng@berkeley.edu
DESCRIPTION: Ego vehicle attempts to merge between two vehicles in 
adjacent lane.
SOURCE: IEEE Autonomous Driving AI Test Challenge
"""

#################################
# MAP AND MODEL                 #
#################################

param map = localPath('../../../../tests/formats/opendrive/maps/CARLA/Town04.xodr')
param carla_map = 'Town04'
model scenic.simulators.carla.model

#################################
# CONSTANTS                     #
#################################

MODEL = 'vehicle.lincoln.mkz2017'

param EGO_SPEED = VerifaiRange(7, 10)

param LEAD_SPEED = VerifaiRange(7, 10)

param TRAIL_SPEED = VerifaiRange(3, 5)
param TRAIL_BUFFER = VerifaiRange(7, 10)
param TRAIL_BRAKE = VerifaiRange(0.5, 1)

BUFFER_DIST = 50
TERM_DIST = 70

#################################
# AGENT BEHAVIORS               #
#################################

behavior EgoBehavior():
	trajectory = [self.laneSection.lane, lead.laneSection.lane]
	do FollowTrajectoryBehavior(
		target_speed=globalParameters.EGO_SPEED,
		trajectory=trajectory)
	do FollowLaneBehavior(target_speed=globalParameters.EGO_SPEED)

behavior AllowMergeBehavior():
	try:
		do FollowLaneBehavior(target_speed=globalParameters.TRAIL_SPEED)
	interrupt when withinDistanceToAnyObjs(self, globalParameters.TRAIL_BUFFER):
		take SetBrakeAction(globalParameters.TRAIL_BRAKE)

#################################
# SPATIAL RELATIONS             #
#################################

initLane = Uniform(*network.lanes)
egoSpawnPt = OrientedPoint in initLane.centerline
trailSpawnPt = OrientedPoint right of egoSpawnPt by 3

#################################
# SCENARIO SPECIFICATION        #
#################################

ego = Car at egoSpawnPt,
	with blueprint MODEL,
	with behavior EgoBehavior()

trail = Car right of ego by 3,
	with blueprint MODEL,
	with AllowMergeBehavior()

lead = Car following roadDirection from trail for 20,
	with blueprint MODEL,
	with FollowLaneBehavior(target_speed=globalParameters.LEAD_SPEED)

require (distance to intersection) > BUFFER_DIST
require always (trail.laneSection._fasterLane is not None)
terminate when (distance to egoSpawnPt) > TERM_DIST
