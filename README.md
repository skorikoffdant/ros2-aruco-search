# ROS 2 ArUco Search

Autonomous search and localization of five ArUco-marked cubes using ROS 2 Jazzy.

## Requirements

- ROS 2 Jazzy
- Gazebo simulation with TurtleBot3 Waffle (`arm05_sim`)
- Nav2 and AMCL
- Python 3
- OpenCV with ArUco support
- NumPy, SciPy, `cv_bridge`, `py_trees`

The Gazebo environment and navigation system must be running before starting the search.

## Build

Clone the repository into your ROS 2 workspace:

```bash
cd /arm_ws/src
git clone https://github.com/skorikoffdant/ros2-aruco-search.git
```

Build the packages:

```bash
cd /arm_ws
colcon build --symlink-install --packages-select aruco_search_interfaces aruco_search
source install/setup.bash
```

## Run

**1. Start the simulation**

In the provided ROS 2 / Gazebo environment:

```bash
ros2 launch arm05_sim world.launch.py
```

**2. Start the search nodes**

In another terminal:

```bash
source /arm_ws/install/setup.bash
ros2 launch aruco_search search.launch.py
```

**3. Start the search**

In another terminal:

```bash
source /arm_ws/install/setup.bash
ros2 action send_goal /search_cubes aruco_search_interfaces/action/SearchCubes '{target_count: 5}' --feedback
```

## Output

During execution, the action provides feedback containing:

- `detected` — number of detected cubes.
- `localized` — number of successfully localized cubes.
- `phase` — current search or localization state.

After successful completion, the action returns the IDs and final positions of all five cubes in the `map` coordinate frame.

The action finishes with status `SUCCEEDED` when all five cubes have been successfully localized.
