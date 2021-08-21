param map = localPath('../../../tests/formats/opendrive/maps/CARLA/Town01.xodr')
param carla_map = 'Town01'
model scenic.simulators.carla.model

scenario LeftIntersectionScenario(intersection):
    setup:

        advInitLane = Uniform(*intersection.incomingLanes)
        advManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.LEFT_TURN, advInitLane.maneuvers))
        advTrajectory = [advInitLane, advManeuver.connectingLane, advManeuver.endLane]
        advSpawnPt = OrientedPoint in advInitLane.centerline

        adv = Car at advSpawnPt,
            with behavior FollowTrajectoryBehavior(
                target_speed=globalParameters.ADV_SPEED, 
                trajectory=advTrajectory)

scenario RightIntersectionScenario(intersection):
    setup:

        advInitLane = Uniform(*intersection.incomingLanes)
        advManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.RIGHT_TURN, advInitLane.maneuvers))
        advTrajectory = [advInitLane, advManeuver.connectingLane, advManeuver.endLane]
        advSpawnPt = OrientedPoint in advInitLane.centerline

        adv = Car at advSpawnPt,
            with behavior FollowTrajectoryBehavior(
                target_speed=globalParameters.ADV_SPEED, 
                trajectory=advTrajectory)

scenario StraightIntersectionScenario(intersection):
    setup:

        advInitLane = Uniform(*intersection.incomingLanes)
        advManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, advInitLane.maneuvers))
        advTrajectory = [advInitLane, advManeuver.connectingLane, advManeuver.endLane]
        advSpawnPt = OrientedPoint in advInitLane.centerline

        adv = Car at advSpawnPt,
            with behavior FollowTrajectoryBehavior(
                target_speed=globalParameters.ADV_SPEED, 
                trajectory=advTrajectory)

scenario Main():
    setup:
        ego = Car on road,
            with behavior FollowLaneBehavior(target_speed=5)
    compose:
        left, right, straight = LeftIntersectionScenario(inter), RightIntersectionScenario(inter), StraightIntersectionScenario(inter)
        while True:
            inter = network.intersectionAt(ego.position)
            if inter is not None:
                do choose left, right, straight
