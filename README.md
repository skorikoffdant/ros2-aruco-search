# ROS 2 ArUco Search

Autonomous search and localization of five ArUco-marked cubes using ROS 2 Jazzy, Nav2, OpenCV and Behavior Trees.

The robot explores the environment, detects ArUco markers, refines their positions and returns the results through a custom ROS 2 Action.

## Packages

- `aruco_search` — detection, navigation and search management.
- `aruco_search_interfaces` — custom `SearchCubes` action definition.

## License

MIT
