"""
TITLE: Behavior Prediction - Intersection 01
AUTHOR: Francis Indaheng, findaheng@berkeley.edu
DESCRIPTION: N vehicles approach an intersection and take a random maneuver.
SOURCE: Kesav Viswanadha
"""

#################################
# MAP AND MODEL                 #
#################################

param map = localPath('../../../../tests/formats/opendrive/maps/CARLA/Town05.xodr')
param carla_map = 'Town05'
model scenic.simulators.carla.model

#################################
# CONSTANTS                     #
#################################

param N = 5  # number of additional vehicles

param EGO_SPEED = VerifaiRange(7, 10)
param OTHER_SPEEDS = [VerifaiRange(7, 10) for _ in range(globalParameters.N)]

#################################
# SPATIAL RELATIONS             #
#################################

intersection = Uniform(*filter(lambda i: i.is4Way, network.intersections))

egoInitLane = Uniform(*intersection.incomingLanes)
egoManeuver = Uniform(*egoInitLane.maneuvers)
egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]

#################################
# SCENARIO SPECIFICATION        #
#################################

ego = Car on egoInitLane,
    with behavior FollowTrajectoryBehavior(target_speed=globalParameters.EGO_SPEED, trajectory=egoTrajectory)

for i in range(globalParameters.N):
    tempInitLane = Uniform(*intersection.incomingLanes)
    tempManeuver = Uniform(*tempInitLane.maneuvers)
    tempTrajectory = [tempInitLane, tempManeuver.connectingLane, tempManeuver.endLane]
    Car on tempInitLane,
        with behavior FollowTrajectoryBehavior(target_speed=globalParameters.OTHER_SPEEDS[i], trajectory=tempTrajectory)
