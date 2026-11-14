import math
import numpy as np
import py_trees
import rclpy
from scipy.spatial.transform import Rotation
from rclpy.node import Node
from rclpy.task import Future
from rclpy.time import Time
from rclpy.parameter import Parameter
from rclpy.action import ActionClient, ActionServer, GoalResponse, CancelResponse
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Point
from aruco_search_interfaces.action import SearchCubes
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose, Spin
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from std_msgs.msg import Int32, Float64MultiArray
from tf2_ros import Buffer, TransformListener, TransformException


RUNNING = py_trees.common.Status.RUNNING
SUCCESS = py_trees.common.Status.SUCCESS
FAILURE = py_trees.common.Status.FAILURE


class Task(py_trees.behaviour.Behaviour):
    def __init__(self, name, callback):
        super().__init__(name)
        self.callback = callback

    def update(self):
        return self.callback()


class SearchManager(Node):
    def __init__(self):
        super().__init__('search_manager')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self, spin_thread=True)
        self.nav = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.spin = ActionClient(self, Spin, '/spin')

        self.goal_future = None
        self.goal_handle = None
        self.result_future = None

        self.grid = None
        self.info = None
        self.origin = None
        self.map_yaw = 0.0
        self.seen = None
        self.waypoints = []
        self.cache = {}

        self.cubes = {}
        self.obs_seq = {}
        self.refined = set()
        self.retry_after = {}
        self.active_id = None
        self.arrivals = 0

        self.search_state = 'idle'
        self.state = 'idle'
        self.approaches = []
        self.view = None
        self.target = (0.0, 0.0)
        self.wait_start = None
        self.view_start_seq = 0
        self.finished = False
        self.active_goal = None
        self.cancel_requested = False
        self.finish_status = None
        self.search_done = None
        self.action_server = ActionServer(
            self, SearchCubes, '/search_cubes',
            execute_callback=self.action_result_callback,
            goal_callback=self.accept_goal,
            handle_accepted_callback=self.start_search,
            cancel_callback=self.cancel_search
        )

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, qos)
        self.create_subscription(Float64MultiArray, '/detected_cube_poses', self.on_cubes, 10)
        self.create_subscription(Float64MultiArray, '/refinement_observation', self.on_observation, 10)
        self.refine_pub = self.create_publisher(Int32, '/active_refinement', 10)
        self.seen_pub = self.create_publisher(OccupancyGrid, '/seen_map', qos)

        self.tree = py_trees.composites.Selector(name='Cube search', memory=False,
            children=[Task('Done', self.done), Task('Localize', self.localize),
                      Task('Search', self.search)])
        self.create_timer(0.2, self.tick)
        self.create_timer(0.5, self.update_seen)
        self.create_timer(1.0, self.publish_seen)
        self.get_logger().info('Search manager started')

    def accept_goal(self, request):
        if request.target_count != 5 or self.active_goal is not None or self.finished:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def start_search(self, goal_handle):
        self.active_goal = goal_handle
        self.search_done = Future()
        self.cancel_requested = False
        self.get_logger().info('SEARCH ACTION: goal accepted (5 cubes)')
        goal_handle.execute()

    def cancel_search(self, goal_handle):
        self.cancel_requested = True
        return CancelResponse.ACCEPT

    async def action_result_callback(self, goal_handle):
        # Await an rclpy Future; asyncio.sleep requires an asyncio event loop.
        await self.search_done
        result = SearchCubes.Result()
        for i in sorted(self.refined):
            x, y, z = self.cubes[i]
            result.ids.append(i)
            result.positions.append(Point(x=float(x), y=float(y), z=float(z)))
        self.active_goal = None
        return result

    def complete_search(self, status):
        if self.finished:
            return
        self.finished = True
        self.finish_status = status
        if self.active_goal is None:
            return
        if status == 'success':
            self.active_goal.succeed()
        elif status == 'cancel':
            self.active_goal.canceled()
        else:
            self.active_goal.abort()
        if self.search_done is not None and not self.search_done.done():
            self.search_done.set_result(True)

    def on_cubes(self, msg):
        if len(msg.data) % 4:
            return
        for i in range(0, len(msg.data), 4):
            cube_id, x, y, z = msg.data[i:i + 4]
            if not all(map(math.isfinite, (cube_id, x, y, z))):
                continue
            if cube_id != int(cube_id) or not 0 <= cube_id <= 4:
                continue
            cube_id = int(cube_id)
            if cube_id not in self.cubes:
                self.get_logger().info(f'Found cube {cube_id}')
            self.cubes[cube_id] = (x, y, z)

    def on_observation(self, msg):
        if len(msg.data) != 5 or not all(map(math.isfinite, msg.data)):
            return
        cube_id, x, y, z, seq = msg.data
        if cube_id != self.active_id:
            return
        if seq > self.obs_seq.get(cube_id, 0):
            self.obs_seq[cube_id] = int(seq)
            self.cubes[cube_id] = (x, y, z)

    def on_map(self, msg):
        if self.grid is not None:
            return
        self.info = msg.info
        self.grid = np.array(msg.data, dtype=np.int16).reshape(
            self.info.height, self.info.width)
        self.seen = np.zeros(self.grid.shape, dtype=bool)
        p, q = self.info.origin.position, self.info.origin.orientation
        self.origin = (p.x, p.y)
        self.map_yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]

        step = max(1, round(2.0 / self.info.resolution))
        margin = math.ceil(0.75 / self.info.resolution)
        for gy in range(margin, self.info.height - margin, step):
            for gx in range(margin, self.info.width - margin, step):
                if self.clear_cell(gx, gy, 0.75):
                    x, y = self.grid_to_world(gx + 0.5, gy + 0.5)
                    self.waypoints.append((x, y))
        self.get_logger().info(f'Map ready: {len(self.waypoints)} waypoints')

    def grid_to_world(self, gx, gy):
        x, y = gx * self.info.resolution, gy * self.info.resolution
        a = self.map_yaw
        return (self.origin[0] + x * math.cos(a) - y * math.sin(a),
                self.origin[1] + x * math.sin(a) + y * math.cos(a))

    def world_to_grid(self, x, y):
        dx, dy = x - self.origin[0], y - self.origin[1]
        a, r = self.map_yaw, self.info.resolution
        return (math.floor((dx * math.cos(a) + dy * math.sin(a)) / r),
                math.floor((-dx * math.sin(a) + dy * math.cos(a)) / r))

    def clear_cell(self, gx, gy, clearance):
        m = math.ceil(clearance / self.info.resolution)
        h, w = self.grid.shape
        if gx - m < 0 or gy - m < 0 or gx + m >= w or gy + m >= h:
            return False
        return bool(np.all(self.grid[gy-m:gy+m+1, gx-m:gx+m+1] == 0))

    def line_of_sight(self, x, y, cx, cy):
        length = math.hypot(cx - x, cy - y)
        if length < 0.1:
            return False
        # Stop just before cube: its own cell may be occupied
        distance = max(0, length - 0.08)
        steps = max(1, math.ceil(distance / (self.info.resolution / 2)))
        for i in range(steps + 1):
            k = (distance / length) * i / steps
            gx, gy = self.world_to_grid(x + k * (cx - x), y + k * (cy - y))
            if not (0 <= gx < self.info.width and 0 <= gy < self.info.height):
                return False
            if self.grid[gy, gx] != 0:
                return False
        return True

    def pose(self, frame='base_link'):
        try:
            tf = self.tf.lookup_transform('map', frame, Time())
            p, q = tf.transform.translation, tf.transform.rotation
            yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
            return p.x, p.y, yaw
        except TransformException:
            return None

    def visible_cells(self, x, y, yaw):
        cells = set()
        # Keep sub-cell precision (not only integer map coordinates)
        dx, dy = x - self.origin[0], y - self.origin[1]
        a, res = self.map_yaw, self.info.resolution
        lx = dx * math.cos(a) + dy * math.sin(a)
        ly = -dx * math.sin(a) + dy * math.cos(a)
        h, w = self.grid.shape

        for offset in np.linspace(-math.pi / 6, math.pi / 6, 15):
            angle = yaw + offset - a
            for d in np.arange(0, 3.0, 0.1):
                gx = math.floor((lx + d * math.cos(angle)) / res)
                gy = math.floor((ly + d * math.sin(angle)) / res)
                if not (0 <= gx < w and 0 <= gy < h) or self.grid[gy, gx] != 0:
                    break
                cells.add(gy * w + gx)
        return np.fromiter(cells, dtype=np.int32)

    def update_seen(self):
        if self.grid is None or self.finished:
            return
        pose = self.pose()
        if pose:
            self.seen.ravel()[self.visible_cells(*pose)] = True

    def publish_seen(self):
        if self.grid is None:
            return
        data = np.full(self.grid.shape, -1, dtype=np.int8)
        data[self.grid >= 50] = 100
        data[self.seen & (self.grid == 0)] = 0
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info = self.info
        msg.data = data.ravel().tolist()
        self.seen_pub.publish(msg)

    def choose_waypoint(self):
        best, best_score = None, -1
        robot = self.pose()
        robot_xy = robot[:2] if robot else self.target
        unseen = ~self.seen.ravel()

        for i, (x, y) in enumerate(self.waypoints):
            for j in range(4):
                key = (x, y, j)
                if key not in self.cache:
                    self.cache[key] = self.visible_cells(x, y, j * math.pi / 2)
                gain = np.count_nonzero(unseen[self.cache[key]])
                score = gain / (1 + 0.35 * math.dist(robot_xy, (x, y)))
                if gain >= 30 and score > best_score:
                    best, best_score = (i, j * math.pi / 2, gain), score
        return best

    def start_nav(self, x, y, yaw):
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)
        return self.start_action(self.nav, goal)

    def start_spin(self, angle):
        goal = Spin.Goal()
        goal.target_yaw = float(angle)
        goal.time_allowance = Duration(sec=30)
        return self.start_action(self.spin, goal)

    def start_action(self, client, goal):
        if not client.server_is_ready():
            return False
        self.goal_future = client.send_goal_async(goal)
        self.goal_handle = None
        self.result_future = None
        return True

    def action_result(self):
        if not self.goal_future.done():
            return None
        if self.goal_handle is None:
            try:
                self.goal_handle = self.goal_future.result()
                if not self.goal_handle.accepted:
                    return False
                self.result_future = self.goal_handle.get_result_async()
            except Exception as e:
                self.get_logger().warn(f'Action error: {e}')
                return False
        if not self.result_future.done():
            return None
        try:
            return self.result_future.result().status == GoalStatus.STATUS_SUCCEEDED
        except Exception as e:
            self.get_logger().warn(f'Action result error: {e}')
            return False

    def pending(self):
        return [i for i in sorted(self.cubes) if i not in self.refined
                and self.arrivals >= self.retry_after.get(i, 0)]

    def done(self):
        return SUCCESS if len(self.refined) == 5 else FAILURE

    def search(self):
        if self.search_state == 'idle':
            if self.pending():
                return FAILURE
            if self.grid is None or not self.nav.server_is_ready():
                return RUNNING
            choice = self.choose_waypoint()
            if choice is None:
                return FAILURE
            i, yaw, gain = choice
            x, y = self.waypoints[i]
            if not self.start_nav(x, y, yaw):
                return RUNNING
            self.waypoints.pop(i)
            self.target = (x, y)
            self.search_state = 'moving'
            self.get_logger().info(f'SEARCH: ({x:.2f}, {y:.2f}), unseen={gain}')

        result = self.action_result()
        if result is None:
            return RUNNING
        if result:
            self.arrivals += 1
            self.get_logger().info(f'SEARCH: arrived ({self.arrivals})')
        else:
            self.get_logger().warn('SEARCH: unreachable waypoint')
        self.search_state = 'idle'
        return SUCCESS

    def points_around(self, center, radius, limit):
        cx, cy = center
        points = []
        for j in range(8):
            angle = j * math.pi / 4
            x, y = cx + radius * math.cos(angle), cy + radius * math.sin(angle)
            gx, gy = self.world_to_grid(x, y)
            if self.clear_cell(gx, gy, 0.40) and self.line_of_sight(x, y, cx, cy):
                points.append((x, y, math.atan2(cy - y, cx - x)))
        pose = self.pose()
        robot = pose[:2] if pose else self.target
        points.sort(key=lambda p: math.dist(robot, p[:2]))
        return points[:limit]

    def alignment_error(self):
        camera = self.pose('camera_rgb_frame')
        if camera is None:
            return None
        x, y, yaw = camera
        cx, cy, _ = self.cubes[self.active_id]
        angle = math.atan2(cy - y, cx - x) - yaw
        return math.atan2(math.sin(angle), math.cos(angle))

    def observe(self):
        self.view_start_seq = self.obs_seq.get(self.active_id, 0)
        self.wait_start = self.get_clock().now()
        self.state = 'observe'
        self.get_logger().info(f'LOCALIZE [{self.active_id}]: observing')

    def finish_localize(self, success):
        i = self.active_id
        self.refine_pub.publish(Int32(data=-1 if success else -2))
        if success:
            self.refined.add(i)
            self.get_logger().info(f'LOCALIZE [{i}]: FROZEN')
        else:
            self.retry_after[i] = self.arrivals + 5
            self.get_logger().warn(f'LOCALIZE [{i}]: failed, returning to SEARCH')
        self.active_id = None
        self.state = 'idle'
        return SUCCESS

    def localize(self):
        if self.state == 'idle':
            if (self.search_state != 'idle' or self.grid is None
                    or not self.nav.server_is_ready()
                    or not self.refine_pub.get_subscription_count()
                    or not self.pending()):
                return FAILURE
            self.active_id = self.pending()[0]
            self.approaches = self.points_around(
                self.cubes[self.active_id][:2], 1.8, 4)
            self.obs_seq[self.active_id] = 0
            self.refine_pub.publish(Int32(data=self.active_id))
            self.state = 'approach'
            self.get_logger().info(
                f'LOCALIZE [{self.active_id}]: {len(self.approaches)} approaches')

        if self.state == 'approach':
            if self.obs_seq.get(self.active_id, 0) >= 5:
                views = self.points_around(self.cubes[self.active_id][:2], 0.85, 1)
                if not views:
                    return self.finish_localize(False)
                self.view = views[0]
                self.state = 'view'
                self.get_logger().info(
                    f'LOCALIZE [{self.active_id}]: new center={self.cubes[self.active_id][:2]}')
                return RUNNING
            if not self.approaches:
                return self.finish_localize(False)
            x, y, yaw = self.approaches[0]
            if not self.start_nav(x, y, yaw):
                return RUNNING
            self.approaches.pop(0)
            self.state = 'approach_nav'
            self.get_logger().info(f'APPROACH [{self.active_id}]: ({x:.2f}, {y:.2f})')

        if self.state == 'approach_nav':
            result = self.action_result()
            if result is None:
                return RUNNING
            if not result:
                self.state = 'approach'
                return RUNNING
            self.wait_start = self.get_clock().now()
            self.state = 'approach_wait'

        if self.state == 'approach_wait':
            if self.obs_seq.get(self.active_id, 0) >= 5:
                self.state = 'approach'
                return RUNNING
            if (self.get_clock().now() - self.wait_start).nanoseconds < 2_000_000_000:
                return RUNNING
            if not self.start_spin(2.2):
                return RUNNING
            self.state = 'approach_scan'
            self.get_logger().info(f'APPROACH [{self.active_id}]: scanning')

        if self.state == 'approach_scan':
            if self.action_result() is None:
                return RUNNING
            self.state = 'approach'
            return RUNNING

        if self.state == 'view':
            x, y, yaw = self.view
            if not self.start_nav(x, y, yaw):
                return RUNNING
            self.state = 'view_nav'
            self.get_logger().info(f'LOCALIZE [{self.active_id}]: viewpoint ({x:.2f}, {y:.2f})')

        if self.state == 'view_nav':
            result = self.action_result()
            if result is None:
                return RUNNING
            if not result:
                return self.finish_localize(False)
            self.state = 'align'
            return RUNNING

        if self.state == 'align':
            error = self.alignment_error()
            if error is None:
                return RUNNING
            degrees = math.degrees(error)
            self.get_logger().info(
                f'LOCALIZE [{self.active_id}]: camera yaw error = {degrees:+.1f} deg')
            if abs(error) <= math.radians(5):
                self.observe()
                return RUNNING
            if not self.start_spin(error):
                return RUNNING
            self.state = 'align_wait'
            self.get_logger().info(
                f'LOCALIZE [{self.active_id}]: correcting yaw by {degrees:+.1f} deg')
            return RUNNING

        if self.state == 'align_wait':
            result = self.action_result()
            if result is None:
                return RUNNING
            if not result:
                self.get_logger().warn(f'LOCALIZE [{self.active_id}]: yaw correction failed')
            self.observe()
            return RUNNING

        if self.state == 'observe':
            if (self.get_clock().now() - self.wait_start).nanoseconds < 2_000_000_000:
                return RUNNING
            n = self.obs_seq.get(self.active_id, 0) - self.view_start_seq
            self.get_logger().info(f'LOCALIZE [{self.active_id}]: {n} fresh observations')
            return self.finish_localize(n > 0)
        return RUNNING

    def tick(self):
        if self.finished or self.active_goal is None:
            return

        if self.cancel_requested:
            self.refine_pub.publish(Int32(data=-2))
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
            self.complete_search('cancel')
            self.get_logger().info('SEARCH ACTION: canceled')
            return

        self.tree.tick_once()

        feedback = SearchCubes.Feedback()
        feedback.detected = len(self.cubes)
        feedback.localized = len(self.refined)
        feedback.phase = self.state if self.state != 'idle' else self.search_state
        self.active_goal.publish_feedback(feedback)

        if len(self.refined) == 5:
            self.get_logger().info('ALL 5 CUBES FOUND AND LOCALIZED')
            self.complete_search('success')
        elif (self.grid is not None and self.search_state == 'idle'
              and self.state == 'idle' and not self.pending()
              and self.choose_waypoint() is None):
            # If some cubes are waiting for a retry, don't finish prematurely.
            if any(i not in self.refined for i in self.cubes):
                # No waypoints remain to increase the retry counter.
                self.get_logger().warn('No waypoints left for retry')
            self.get_logger().warn(f'Search ended: {len(self.refined)}/5 localized')
            self.complete_search('failed')


def main():
    rclpy.init()
    node = SearchManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.refine_pub.publish(Int32(data=-2))
        if (node.goal_handle is not None and node.result_future is not None
                and not node.result_future.done()):
            node.goal_handle.cancel_goal_async()
        node.action_server.destroy()
        node.listener.unregister()
        node.listener.executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
