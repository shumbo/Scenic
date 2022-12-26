
model scenic.simulators.webots.model

workspace = Workspace(RectangularRegion((0, 0), 0, 1, 1))

class Box(WebotsObject):
    webotsType: 'box'
    orientation: (Range(0, 360) deg, 0, 0)
    width: 0.1
    length: 0.1

behavior Push():
    while True:
        take ApplyForceAction((0, 2), relative=True)

ego = new Box with behavior Push

new Box in workspace
new Box in workspace

terminate after 5 seconds
