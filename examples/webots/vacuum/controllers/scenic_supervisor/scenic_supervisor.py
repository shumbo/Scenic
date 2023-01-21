import json
from pathlib import Path
from datetime import datetime

from controller import Supervisor

import scenic
from scenic.simulators.webots import WebotsSimulator

# iteration
iteration = 10

# parameters
params = {
    "numToys": 16
}

# save logs to `logs`
output_dir = Path(__file__).resolve().parent.parent.parent / "logs"
output_dir.mkdir(parents=True, exist_ok=True)

supervisor = Supervisor()
simulator = WebotsSimulator(supervisor)

path = supervisor.getCustomData()
print(f'Loading Scenic scenario {path}')
scenario = scenic.scenarioFromFile(path, params=params)

for _ in range(10):
    ts = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

    scene, _ = scenario.generate()
    print('Starting new simulation...')
    sim_results = simulator.simulate(scene, verbosity=2).result

    s = json.dumps({"params": params, "results": sim_results.records}, indent=4)

    filename = f"{ts}.json"
    with open(output_dir / filename, "x") as f:
        f.write(s)
