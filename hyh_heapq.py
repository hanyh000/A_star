# scan
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from math import pow, atan2, sqrt, sin, cos, pi
import heapq
import numpy as np
from sensor_msgs.msg import LaserScan, CompressedImage
from ultralytics import YOLO
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

model = YOLO("/home/dev/ros-cv_ws/tumblr.pt")

class NodeAStar:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0
    def __eq__(self, other): return self.position == other.position
    def __lt__(self, other): return self.f < other.f

class IntegratedNavigation(Node):
    def __init__(self):
        super().__init__('integrated_navigation')

        qos_profile = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        map_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.tumblr_detected = False
        self.frame = None

        self.last_action = None

        self.lookahead_dist  = 1.0
        self.linear_vel  = 0.12
        self.stop_tolerance  = 0.25
        
        self.map_data = None
        self.map_resolution = 0.05
        self.map_origin = [0.0, 0.0]
        self.map_width = 0
        self.map_height = 0
        
        self.current_pose = None
        self.current_yaw = 0.0
        self.global_path = [] 
        self.path_index = 0

        self.scan_data = None

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_path = self.create_publisher(Path, '/planned_path', 10)

        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos_profile)
        self.sub_pose = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        self.sub_goal = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.scan_subscription = self.create_subscription(LaserScan, "/scan", self.scan_callback, qos_profile)
        self.img_subscriber = self.create_subscription(CompressedImage, '/image_raw/compressed', self.image_callback, qos_profile)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Let's Run!")
    
    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f"이미지 디코딩 에러: {e}")
            return

        # YOLO 추론 (conf 0.7 이상인 것만)
        results = model(self.frame, conf=0.7, verbose=False)
        
        # 감지된 박스가 있는지 확인
        is_detected = False
        for r in results:
            if len(r.boxes) > 0:  # 감지된 물체(박스)가 1개라도 있다면
                is_detected = True
                # 가장 신뢰도가 높은 첫 번째 물체의 정보를 로그로 출력
                conf = r.boxes[0].conf[0]
                self.get_logger().info(f"!!! 물체 감지됨 (확신도: {conf:.2f}) - 로봇 정지 !!!")
                break # 하나만 찾아도 정지하므로 루프 종료

        # 전역 변수에 저장 (control_loop에서 사용)
        self.tumblr_detected = is_detected

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), 3.5, ranges)
        ranges = np.where(np.isnan(ranges), 3.5, ranges)
        self.scan_data = ranges 

    def map_callback(self, msg):
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        self.map_data = np.array(msg.data).reshape((self.map_height, self.map_width))

    def pose_callback(self, msg):
        self.current_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        q = msg.pose.pose.orientation
        self.current_yaw = atan2(2.0*(q.w*q.z + q.x*q.y), 1.0-2.0*(q.y*q.y + q.z*q.z))

    def goal_callback(self, msg):
        if self.map_data is None:
            self.get_logger().warn("Waiting for Map...")
            return
        if self.current_pose is None:
            self.get_logger().warn("Waiting for Initial Pose (Set 2D Pose Estimate in RViz)")
            return

        self.get_logger().info(f"Goal Received: {msg.pose.position.x}, {msg.pose.position.y}")

        goal_pose = [msg.pose.position.x, msg.pose.position.y]
        start_grid = self.world_to_grid(self.current_pose)
        goal_grid = self.world_to_grid(goal_pose)
        
        self.get_logger().info("Calculating Path...")
        path_grid = self.run_astar(start_grid, goal_grid)
        
        if path_grid:
            self.global_path = [self.grid_to_world(p) for p in path_grid]
            self.path_index = 0
            self.publish_path_viz()
            self.get_logger().info("Path Found! Go!")
        else:
            self.get_logger().warn("No Path Found.")

    def run_astar(self, start, end):
        if not (0 <= start[0] < self.map_height and 0 <= start[1] < self.map_width): return None
        if not (0 <= end[0] < self.map_height and 0 <= end[1] < self.map_width): return None

        start_node = NodeAStar(None, start)
        end_node = NodeAStar(None, end)
        open_list = []
        heapq.heappush(open_list, start_node)
        visited = set()
        moves = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]

        while open_list:
            current_node = heapq.heappop(open_list)
            if current_node.position in visited: continue
            visited.add(current_node.position)

            if current_node.position == end_node.position:
                path = []
                current = current_node
                while current:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]

            for move in moves:
                ny, nx = current_node.position[0] + move[0], current_node.position[1] + move[1]
                if not (0 <= ny < self.map_height and 0 <= nx < self.map_width): continue
                if self.map_data[ny][nx] > 50 or self.map_data[ny][nx] == -1: continue
                
                new_node = NodeAStar(current_node, (ny, nx))
                new_node.g = current_node.g + 1
                new_node.h = sqrt((ny - end[0])**2 + (nx - end[1])**2)
                new_node.f = new_node.g + new_node.h
                heapq.heappush(open_list, new_node)
        return None

    def control_loop(self):
        if self.scan_data is None or not self.global_path: return

        final_goal = self.global_path[-1]
        
        dist_to_final = sqrt((final_goal[0]-self.current_pose[0])**2 + (final_goal[1]-self.current_pose[1])**2)

        if dist_to_final < self.stop_tolerance:
            self.global_path = []
            self.stop_robot()
            return

        target_x, target_y = final_goal

        for i in range(self.path_index, len(self.global_path)):
            px, py = self.global_path[i]
            dist = sqrt((px - self.current_pose[0])**2 + (py - self.current_pose[1])**2)

            if dist >= self.lookahead_dist:
                target_x, target_y = px, py
                self.path_index = i
                break

        dx = target_x - self.current_pose[0]
        dy = target_y - self.current_pose[1]
        alpha = atan2(dy, dx) - self.current_yaw
        
        if alpha > pi: alpha -= 2*pi
        elif alpha < -pi: alpha += 2*pi
        
        angular_velocity = self.linear_vel * (2.0 * sin(alpha)) / self.lookahead_dist

        ranges = self.scan_data
        front_ranges = np.concatenate((ranges[:30], ranges[-30:]))
        left_ranges = ranges[45:130]
        right_ranges = ranges[230:315]
        back_ranges = ranges[145:215]

        front_min_dist = np.min(front_ranges)
        left_min_dist = np.min(left_ranges)
        right_min_dist = np.min(right_ranges)
        back_min_dist = np.min(back_ranges)

        safe_dist = 0.4

        cmd = Twist()

        diff_threshold = 0.15 

        if self.tumblr_detected:
            action = "stop"
        elif front_min_dist < 0.2 or left_min_dist < 0.16 or right_min_dist < 0.16 or back_min_dist < 0.2:
            action = "stop"
        elif front_min_dist < safe_dist:
            
            better_direction = "turn_left" if left_min_dist >= right_min_dist else "turn_right"
        
            if self.last_action in ["turn_left", "turn_right"]:
                if abs(left_min_dist - right_min_dist) < diff_threshold:
                    action = self.last_action
                else:
                    action = better_direction
            else:
                action = better_direction
        
        elif front_min_dist < safe_dist and left_min_dist < safe_dist and right_min_dist < safe_dist:
            action = "go_back"
        else:
            action = "go_forward"

        self.last_action = action
            
        if action == "go_forward":
            cmd.linear.x = self.linear_vel
            cmd.angular.z = angular_velocity
        elif action == "go_back":
            cmd.linear.x = -0.2
            cmd.angular.z = 0.0
        elif action == "turn_left":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.4
        elif action == "turn_right":
            cmd.linear.x = 0.0
            cmd.angular.z = -0.4
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        
        if cmd.angular.z > 1.0: cmd.angular.z = 1.0
        if cmd.angular.z < -1.0: cmd.angular.z = -1.0
        
        self.pub_cmd.publish(cmd)
        
    def world_to_grid(self, world):
        return (int((world[1]-self.map_origin[1])/self.map_resolution), int((world[0]-self.map_origin[0])/self.map_resolution))

    def grid_to_world(self, grid):
        return [(grid[1]*self.map_resolution)+self.map_origin[0], (grid[0]*self.map_resolution)+self.map_origin[1]]

    def publish_path_viz(self):
        msg = Path()
        msg.header.frame_id = 'map'
        for p in self.global_path:
            ps = PoseStamped()
            ps.pose.position.x, ps.pose.position.y = p[0], p[1]
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def stop_robot(self):
        self.pub_cmd.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = IntegratedNavigation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()