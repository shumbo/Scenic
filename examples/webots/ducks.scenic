"""
Demonstrate orientation can be saved to/loaded from webots
"""

model scenic.simulators.webots.model

workspace = Workspace(RectangularRegion((0, 0, 0), 0, 10, 10))

behavior Push():
    while True:
        take ApplyForceAction((1, 0, 0), relative=True)
        print("ego is facing", ego.yaw, ego.pitch, ego.roll)
        print("ego's elevation", ego.elevation)
        print("ego's angular velocity", ego.angularVelocity)

class Duck(WebotsObject):
    webotsType: 'Duck'
    width: 0.1
    length: 0.1
    orientationOffset: (90 deg, 0, 0)

ego   = new Duck at (0, 0, 0), facing (0, 0, 0), with behavior Push
duck1 = new Duck at (0, 1, 0), facing (90 deg, 0, 0) # yaw
duck2 = new Duck at (0, 2, 0), facing (0, 90 deg, 0) # pitch
duck3 = new Duck at (0, 3, 0), facing (0, 0, 90 deg) # roll
