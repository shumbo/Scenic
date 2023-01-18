
from controller import Supervisor

import scenic
from scenic.simulators.webots import WebotsSimulator

supervisor = Supervisor()
simulator = WebotsSimulator(supervisor)

path = supervisor.getCustomData()
print(f'Loading Scenic scenario {path}')
scenario = scenic.scenarioFromFile(path)

while True:
    scene, _ = scenario.generate()
    print('Starting new simulation...')
    sim_results = simulator.simulate(scene, verbosity=2).result

    print(sim_results.records)
    assert False
