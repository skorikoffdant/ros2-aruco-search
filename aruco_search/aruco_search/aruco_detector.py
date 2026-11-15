import cv2
import numpy as np
import rclpy

from collections import deque
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.time import Time
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Int32MultiArray, Float64MultiArray
from tf2_ros import Buffer, TransformListener, TransformException


class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.set_parameters([Parameter('use_sim_time', value=True)])

        self.bridge = CvBridge()
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self, spin_thread=True)

        self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.params = cv2.aruco.DetectorParameters_create()
        self.K = np.array([[1696.8028, 0, 960], [0, 1696.8027, 540], [0, 0, 1]])
        self.marker_size = 0.09 * 588 / 755

        self.cubes = {}
        self.active_id = -1
        self.refine_start = None
        self.observe_start = None
        self.frames = deque(maxlen=60)
        self.last_frame = None

        self.create_subscription(Image, '/camera/image_raw', self.on_image, 10)
        self.create_subscription(Int32, '/active_refinement', self.on_refine, 10)

        self.ids_pub = self.create_publisher(Int32MultiArray, '/detected_cubes', 10)
        self.poses_pub = self.create_publisher(Float64MultiArray, '/detected_cube_poses', 10)
        self.obs_pub = self.create_publisher(Float64MultiArray, '/refinement_observation', 10)
        self.final_pub = self.create_publisher(Float64MultiArray, '/final_observation', 10)

        self.create_timer(0.1, self.process_frames)
        self.create_timer(2.0, self.print_cubes)
        self.get_logger().info('ArUco detector started')

    def on_refine(self, msg):
        new_id = msg.data

        if new_id == -3:
            if self.active_id >= 0 and self.observe_start is None:
                self.observe_start = self.get_clock().now()
                cube = self.cubes[self.active_id]
                cube['final_samples'].clear()
                cube['state'] = 'OBSERVING'
                self.get_logger().info(f'OBSERVE STARTED: cube {self.active_id}')
            return

        if new_id == self.active_id:
            return

        if self.active_id >= 0:
            cube = self.cubes[self.active_id]

            if new_id == -1 and cube['final_samples']:
                cube['position'] = np.median(cube['final_samples'], axis=0)
                cube['state'] = 'REFINED'
            else:
                cube['state'] = 'DETECTED'
                if new_id == -1:
                    self.get_logger().warn('Freeze requested without stationary samples')

            self.get_logger().info(f'CUBE {self.active_id}: {cube["state"]}')
            self.active_id = -1
            self.refine_start = None
            self.observe_start = None

        if new_id < 0:
            self.publish()
        elif new_id in self.cubes and self.cubes[new_id]['state'] != 'REFINED':
            cube = self.cubes[new_id]
            self.active_id = new_id
            self.refine_start = self.get_clock().now()
            cube['state'] = 'REFINING'
            cube['samples'].clear()
            cube['final_samples'].clear()
            cube['refine_n'] = 0
            self.get_logger().info(f'REFINEMENT STARTED: cube {new_id}')

    def on_image(self, msg):
        stamp = Time.from_msg(msg.header.stamp)

        if self.last_frame is not None:
            dt = (stamp - self.last_frame).nanoseconds / 1e9
            if 0 <= dt < 0.2:
                return
        self.last_frame = stamp

        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        corners, ids, _ = cv2.aruco.detectMarkers(
            image, self.dictionary, parameters=self.params
        )
        if ids is None:
            return

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, self.K, np.zeros(5)
        )

        found = [(int(i), r[0], t[0]) for i, r, t in zip(ids.flatten(), rvecs, tvecs)
                 if 0 <= i <= 4]

        if found:
            self.frames.append((stamp, found))

    def process_frames(self):
        remaining = deque(maxlen=60)
        now = self.get_clock().now()

        for stamp, detections in self.frames:
            if (now - stamp).nanoseconds > 6_000_000_000:
                continue

            try:
                tf = self.tf.lookup_transform('map', 'camera_rgb_frame', stamp)
            except TransformException:
                remaining.append((stamp, detections))
                continue

            t, q = tf.transform.translation, tf.transform.rotation
            rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
            translation = np.array([t.x, t.y, t.z])

            for cube_id, rvec, tvec in detections:
                marker_rotation, _ = cv2.Rodrigues(rvec)
                p = tvec - 0.045 * marker_rotation[:, 2]
                camera_point = [p[2], -p[0], -p[1]]
                position = rotation.apply(camera_point) + translation

                if not np.all(np.isfinite(position)):
                    continue

                if cube_id not in self.cubes:
                    self.cubes[cube_id] = {
                        'position': position,
                        'samples': deque(maxlen=20),
                        'final_samples': deque(maxlen=20),
                        'state': 'DETECTED',
                        'refine_n': 0
                    }
                    self.get_logger().info(f'NEW CUBE {cube_id}: {position.round(3)}')
                    continue

                if cube_id != self.active_id or stamp < self.refine_start:
                    continue

                cube = self.cubes[cube_id]

                if self.observe_start is not None:
                    if stamp < self.observe_start:
                        continue

                    cube['final_samples'].append(position)
                    median = np.median(cube['final_samples'], axis=0)

                    self.final_pub.publish(Float64MultiArray(data=[
                        float(cube_id), *map(float, median),
                        float(len(cube['final_samples']))
                    ]))
                else:
                    cube['samples'].append(position)
                    cube['position'] = np.median(cube['samples'], axis=0)
                    cube['refine_n'] += 1

                    self.obs_pub.publish(Float64MultiArray(data=[
                        float(cube_id), *map(float, cube['position']),
                        float(cube['refine_n'])
                    ]))

        self.frames = remaining

    def publish(self):
        ids = sorted(self.cubes)
        self.ids_pub.publish(Int32MultiArray(data=ids))

        poses = [v for i in ids for v in
                 (float(i), *map(float, self.cubes[i]['position']))]
        self.poses_pub.publish(Float64MultiArray(data=poses))

    def print_cubes(self):
        self.publish()

        for i, cube in sorted(self.cubes.items()):
            x, y, z = cube['position']
            self.get_logger().info(
                f'ID {i}: x={x:.3f}, y={y:.3f}, z={z:.3f} | '
                f'{cube["state"]} | refine_n={cube["refine_n"]}'
            )


def main():
    rclpy.init()
    node = ArucoDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.listener.unregister()
        node.listener.executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()