# Behavior Prediction Scenario Descriptions

The following includes a library of SCENIC programs written for use with CARLA simulator.
Scenarios were created primarily for the testing of behavior prediction models, though they may be used in general cases.
Inspiration was drawn from multiple sources, including [NHSTA's Pre-Crash Scenarios](https://rosap.ntl.bts.gov/view/dot/41932/dot_41932_DS1.pdf), the [INTERACTION dataset](https://github.com/interaction-dataset/interaction-dataset), the [Argoverse Motion Forecasting dataset](https://www.argoverse.org/data.html#forecasting-link), and the [CARLA Autonomous Driving Challenge](https://leaderboard.carla.org/scenarios/).
</br>
</br>
For questions and concerns, please contact Francis Indaheng at findaheng@berkeley.edu or post an issue to this repo.
</br>
> Some notation definitions:
> - DIP => Development in-progress
> - TIP => Testing in-progress

## Intersection
01. Ego vehicle goes straight at 4-way intersection and must suddenly stop to avoid collision when adversary vehicle from opposite lane makes a left turn.
02. Ego vehicle makes a left turn at 4-way intersection and must suddenly stop to avoid collision when adversary vehicle from opposite lane goes straight.
03. Ego vehicle either goes straight or makes a left turn at 4-way intersection and must suddenly stop to avoid collision when adversary vehicle from lateral lane continues straight.
04. Ego vehicle either goes straight or makes a left turn at 4-way intersection and must suddenly stop to avoid collision when adversary vehicle from lateral lane makes a left turn.
05. Ego vehicle makes a right turn at 4-way intersection while adversary vehicle from opposite lane makes a left turn.
06. Ego vehicle makes a right turn at 4-way intersection while adversary vehicle from lateral lane goes straight.
07. Ego vehicle makes a left turn at 3-way intersection and must suddenly stop to avoid collision when adversary vehicle from lateral lane continues straight.
08. Ego vehicle goes straight at 3-way intersection and must suddenly stop to avoid collision when adversary vehicle makes a left turn.
09. Ego vehicle makes a right turn at 3-way intersection while adversary vehicle from lateral lane goes straight.
10. Ego Vehicle waits at 4-way intersection while adversary vehicle in adjacent lane passes before performing a lane change to bypass a stationary vehicle waiting to make a left turn.
11. N vehicles approach an intersection and take a random maneuver.

## Bypassing
01. Ego vehicle performs a lane change to bypass a slow adversary vehicle before returning to its original lane.
02. Advesary vehicle performs a lange change to bypass the ego vehicle before returning to its original lane.
03. Ego vehicle performs a lane change to bypass a slow adversary vehicle but cannot return to its original lane because the adversary accelerates. Ego vehicle must then slow down to avoid collision with leading vehicle in new lane.
04. Ego vehicle performs multiple lane changes to bypass two slow adversary vehicles.
05. Ego vehicle performs multiple lane changes to bypass three slow adversary vehicles.

## Roundabout
01. (TIP) N vehicles approach a roundabout and take a random maneuver.

## Merging
01. (TIP) Ego vehicle attempts to merge between two vehicles in adjacent lane.
02. (DIP) Ego vehicle merges into merging lanes off highway.

## Pedestrian
01. Ego vehicle must suddenly stop to avoid collision when pedestrian crosses the road unexpectedly.
02. Both ego and adversary vehicles must suddenly stop to avoid collision when pedestrian crosses the road unexpectedly.
03. Ego vehicle makes a left turn at an intersection and must suddenly stop to avoid collision when pedestrian crosses the crosswalk.
04. Ego vehicle makes a right turn at an intersection and must yield when pedestrian crosses the crosswalk.
05. Ego vehicle goes straight at an intersection and must yield when pedestrian crosses the crosswalk.

