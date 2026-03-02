# mtrl/env/kuka_real_env.py
"""
KUKA 真实机器人环境

该环境提供与 Metaworld 仿真环境相同的接口，用于 sim-to-real 部署。
实现了 reset(), step(), 观测处理等核心方法。

使用方法:
    from mtrl.env.kuka_real_env import KukaRealEnv
    
    env = KukaRealEnv(task_name="reach-v2")
    env.set_goal(np.array([0.7, 0.0, 0.15]))
    
    obs = env.reset()
    for _ in range(200):
        action = model.select_action(obs)  # 从模型获取动作
        obs, reward, done, info = env.step(action)
        if done or info['success']:
            break
"""

import numpy as np
import math
import threading
from typing import Tuple, Dict, Any, Optional

# 尝试导入 ROS
try:
    import rospy
    from iiwa_msgs.msg import CartesianPose
    from geometry_msgs.msg import PoseStamped
    from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg
    ROS_AVAILABLE = True
except ImportError:
    print("[KukaRealEnv] Warning: ROS not available. Some features will be disabled.")
    ROS_AVAILABLE = False
    rospy = None
    CartesianPose = None
    PoseStamped = None
    outputMsg = None


class KukaRealEnv:
    """
    真实 KUKA 机器人环境
    
    该环境与 Metaworld 仿真环境接口兼容，提供：
    - reset(): 重置环境到初始状态
    - step(action): 执行动作并返回新观测
    - get_obs(): 获取当前观测
    - set_goal()/set_object_position(): 设置任务参数
    
    观测空间: 39维（与 Metaworld 兼容）
    动作空间: 4维 [delta_x, delta_y, delta_z, gripper]，范围 [-1, 1]
    
    坐标系说明:
        - 仿真坐标系 (Sim): Metaworld 使用的坐标系
        - 真实坐标系 (Real): KUKA 机器人使用的坐标系
        - 两个坐标系相差 90° 绕 Z 轴旋转
    """
    
    def __init__(
        self, 
        task_name: str = "reach-v2",
        init_ros: bool = True,
        max_episode_steps: int = 200
    ):
        """
        初始化真实机器人环境
        
        Args:
            task_name: 任务名称 (e.g., "reach-v2", "push-v2")
            init_ros: 是否初始化 ROS 节点
            max_episode_steps: 每个 episode 的最大步数
        """
        self.task_name = task_name
        self.max_episode_steps = max_episode_steps
        self.curr_path_length = 0
        
        # 动作缩放（与 Metaworld 相同）
        self.action_scale = 1.0 / 100
        
        # 工作空间边界（真实机器人坐标系）
        self.hand_low = np.array([0.4, -0.6, 0.01])
        self.hand_high = np.array([0.9, 0.6, 0.6])
        
        # 初始位置（真实坐标系）
        self.ee_init_position = np.array([0.6, 0.0, 0.2])
        self.ee_current_position = self.ee_init_position.copy()
        
        # 目标和物体位置（真实坐标系）
        self.goal = np.array([0.7, 0.0, 0.15])
        self.object_position = np.zeros(3)
        
        # 末端执行器偏移
        self._ee_offset = 0.16
        
        # ROS 初始化
        self._ros_initialized = False
        self._robot_pub = None
        self._gripper_pub = None
        self._pose_sub = None
        self._rate = None
        self._thread = None
        self.command = None
        
        if init_ros and ROS_AVAILABLE:
            self._init_ros()
        
        # 相机（可选）
        self.camera = None
        
        print(f"[KukaRealEnv] Initialized for task: {task_name}")
        print(f"[KukaRealEnv] ROS available: {ROS_AVAILABLE}, initialized: {self._ros_initialized}")
    
    def _init_ros(self):
        """初始化 ROS 通信"""
        try:
            rospy.init_node('kuka_real_env', anonymous=False)
            self._robot_pub = rospy.Publisher(
                '/iiwa/command/CartesianPose', 
                PoseStamped, 
                queue_size=1
            )
            self._gripper_pub = rospy.Publisher(
                "Robotiq2FGripperRobotOutput", 
                outputMsg.Robotiq2FGripper_robot_output, 
                queue_size=10
            )
            self._pose_sub = rospy.Subscriber(
                '/iiwa/state/CartesianPose', 
                CartesianPose, 
                self._get_ee_pose_callback
            )
            self._rate = rospy.Rate(20)  # 20 Hz
            
            # ROS 线程
            self._thread = threading.Thread(target=self._thread_job)
            self._thread.daemon = True
            self._thread.start()
            
            self._ros_initialized = True
            print("[KukaRealEnv] ROS initialized successfully")
        except Exception as e:
            print(f"[KukaRealEnv] ROS initialization failed: {e}")
            self._ros_initialized = False
    
    # ==================== 坐标变换 ====================
    
    def coordinate_from_sim_to_real(self, pos_sim: np.ndarray) -> np.ndarray:
        """
        仿真坐标系 -> 真实机器人坐标系
        
        变换矩阵（90°绕Z轴旋转）:
        R = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        """
        R_real_sim = np.array([
            [0.0, 1.0, 0.0], 
            [-1.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0]
        ])
        return np.matmul(R_real_sim, pos_sim)

    def coordinate_from_real_to_sim(self, pos_real: np.ndarray) -> np.ndarray:
        """
        真实机器人坐标系 -> 仿真坐标系
        
        逆变换矩阵:
        R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        """
        R_sim_real = np.array([
            [0.0, -1.0, 0.0], 
            [1.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0]
        ])
        return np.matmul(R_sim_real, pos_real)
    
    # ==================== 核心接口 ====================
    
    def reset(self) -> np.ndarray:
        """
        重置环境到初始状态
        
        Returns:
            observation: 39维观测向量（仿真坐标系）
        """
        if self._ros_initialized:
            # 重置机械臂到初始位置
            self._reset_manipulator()
            
            # 重置夹爪
            self.command = self._reset_gripper()
            
            if self._rate:
                rospy.sleep(0.1)
        else:
            # 模拟模式：直接设置位置
            self.ee_current_position = self.ee_init_position.copy()
        
        # 重置计数器
        self.curr_path_length = 0
        
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行动作
        
        Args:
            action: [4] 动作向量，范围 [-1, 1]
                   [delta_x, delta_y, delta_z, gripper] (仿真坐标系)
        
        Returns:
            observation: 39维观测向量
            reward: 奖励值
            done: 是否结束
            info: 额外信息 {'success': bool, 'unscaled_reward': float}
        """
        # 限制动作范围
        action = np.clip(action, -1, 1)
        
        # 执行 XYZ 动作（输入为仿真坐标系）
        self._set_xyz_action(action[:3])
        
        # 执行夹爪动作
        self._set_gripper_action(action[3])
        
        # 等待执行完成
        if self._ros_initialized and self._rate:
            self._rate.sleep()
        
        # 获取新观测
        obs = self._get_obs()
        
        # 计算奖励和成功
        reward = self._compute_reward()
        success = self._check_success()
        done = self.curr_path_length >= self.max_episode_steps
        
        self.curr_path_length += 1
        
        info = {
            'success': success,
            'unscaled_reward': reward
        }
        
        return obs, reward, done, info
    
    def _get_obs(self) -> np.ndarray:
        """
        获取观测（39维，仿真坐标系）
        
        观测格式与 Metaworld 兼容:
            [pos_hand(3), obs_obj(14), pos_hand(3), goal(3), obs_obj(14), gripper(2)]
        
        Returns:
            39维观测向量
        """
        # 转换到仿真坐标系
        pos_hand_sim = self.coordinate_from_real_to_sim(self.ee_current_position)
        pos_obj_sim = self.coordinate_from_real_to_sim(self.object_position)
        pos_goal_sim = self.coordinate_from_real_to_sim(self.goal)
        
        # 2. 构造当前帧 18 维向量 [EE(4), Obj(7), Relative(7)]
        curr_frame = np.zeros(18)
        curr_frame[0:3] = pos_hand_sim
        curr_frame[3] = 0.0 # 假设夹爪位置，如果有传感器可填入
        curr_frame[4:7] = pos_obj_sim # 物体位置！必须有相机数据或手动填入
        # curr_frame[7:11] 可以填物体姿态(四元数)，如果没相机就留0
        
        # 3. 构造上一帧 18 维向量 (使用缓存的 prev 数据)
        prev_frame = np.zeros(18)
        prev_frame[0:3] = self.prev_ee_pos_sim
        prev_frame[4:7] = self.prev_obj_pos_sim
        
        # 4. 拼接成 39 维 [Curr_18, Prev_18, Goal_3]
        obs = np.hstack([
            curr_frame,     # 0-17
            prev_frame,     # 18-35
            pos_goal_sim    # 36-38
        ])
        
        # 5. 更新缓存，供下一帧使用
        self.prev_ee_pos_sim = pos_hand_sim.copy()
        self.prev_obj_pos_sim = pos_obj_sim.copy()
        return obs
        
    def get_state_for_transformer(self) -> np.ndarray:
        """
        获取用于 Transformer 的 21 维状态
        
        这是模型实际使用的简化状态格式:
            [pos_hand(3), pos_obj(6), pos_goal(3), additional(9)]
        
        Returns:
            21维状态向量
        """
        pos_hand_sim = self.coordinate_from_real_to_sim(self.ee_current_position)
        pos_obj_sim = self.coordinate_from_real_to_sim(self.object_position)
        pos_goal_sim = self.coordinate_from_real_to_sim(self.goal)
        
        # 补齐物体位置到6维
        pos_obj_padded = np.zeros(6)
        pos_obj_padded[:3] = pos_obj_sim
        
        # 额外状态维度（可用于任务特定信息）
        additional = np.zeros(9)
        
        state = np.hstack([
            pos_hand_sim,       # [3]
            pos_obj_padded,     # [6]
            pos_goal_sim,       # [3]
            additional,         # [9]
        ])
        
        assert state.shape[0] == 21, f"Expected 21-dim state, got {state.shape[0]}"
        return state
    
    # ==================== 动作执行 ====================
    
    def _set_xyz_action(self, xyz_action_sim: np.ndarray):
        """
        设置 XYZ 动作
        
        Args:
            xyz_action_sim: [3] 仿真坐标系中的动作增量
        """
        # 转换到真实坐标系
        xyz_action_real = self.coordinate_from_sim_to_real(xyz_action_sim)
        
        # 应用动作缩放
        pos_delta = xyz_action_real * self.action_scale
        new_pos = self.ee_current_position + pos_delta
        
        # 限制在工作空间内
        new_pos = np.clip(new_pos, self.hand_low, self.hand_high)
        
        if self._ros_initialized and self._robot_pub:
            # 发送 ROS 命令
            goal = self._create_ee_pose_msg(new_pos[0], new_pos[1], new_pos[2])
            self._robot_pub.publish(goal)
        else:
            # 模拟模式：直接更新位置
            self.ee_current_position = new_pos
    
    def _set_gripper_action(self, gripper_state: float):
        """
        设置夹爪动作
        
        Args:
            gripper_state: [-1, 1] 夹爪状态 (-1=开, 1=闭)
        """
        if not self._ros_initialized or self.command is None:
            return
        
        # 转换 [-1, 1] 到 [0, 255]
        # -1 -> 255 (开), 1 -> 0 (闭)
        position = int(128 - 127.5 * gripper_state)
        position = np.clip(position, 0, 255)
        
        self.command.rPR = position
        if self._gripper_pub:
            self._gripper_pub.publish(self.command)
    
    # ==================== 奖励和成功检测 ====================
    
    def _compute_reward(self) -> float:
        """
        计算奖励
        
        默认实现：基于距离的奖励
        可以在子类中根据任务覆盖此方法
        """
        dist = np.linalg.norm(self.ee_current_position - self.goal)
        return -dist
    
    def _check_success(self) -> bool:
        """
        检查任务是否成功
        
        默认实现：末端到达目标附近
        可以在子类中根据任务覆盖此方法
        """
        dist = np.linalg.norm(self.ee_current_position - self.goal)
        return dist < 0.05  # 5cm 阈值
    
    # ==================== 任务设置 ====================
    
    def set_goal(self, goal: np.ndarray):
        """
        设置目标位置
        
        Args:
            goal: [3] 目标位置 (真实坐标系)
        """
        assert goal.shape[0] == 3
        self.goal = goal.copy()
        print(f"[KukaRealEnv] Goal set to: {self.goal}")
    
    def set_object_position(self, position: np.ndarray):
        """
        设置物体位置
        
        Args:
            position: [3] 物体位置 (真实坐标系)
        """
        assert position.shape[0] == 3
        self.object_position = position.copy()
        print(f"[KukaRealEnv] Object position set to: {self.object_position}")
    
    def set_init_position(self, position: np.ndarray):
        """
        设置初始末端位置
        
        Args:
            position: [3] 初始位置 (真实坐标系)
        """
        assert position.shape[0] == 3
        self.ee_init_position = position.copy()
        print(f"[KukaRealEnv] Init position set to: {self.ee_init_position}")
    
    # ==================== ROS 通信 ====================
    
    def _reset_manipulator(self):
        """重置机械臂到初始位置"""
        if not self._ros_initialized:
            return
        
        max_attempts = 100
        attempts = 0
        
        while np.linalg.norm(self.ee_init_position - self.ee_current_position) > 0.01:
            goal = self._create_ee_pose_msg(
                self.ee_init_position[0],
                self.ee_init_position[1],
                self.ee_init_position[2]
            )
            if self._robot_pub:
                self._robot_pub.publish(goal)
            if self._rate:
                self._rate.sleep()
            
            attempts += 1
            if attempts > max_attempts:
                print("[KukaRealEnv] Warning: Failed to reach init position")
                break
        
        print(f"[KukaRealEnv] Manipulator reset to: {self.ee_current_position}")
    
    def _reset_gripper(self):
        """重置并激活夹爪"""
        if not self._ros_initialized or not self._gripper_pub:
            return None
        
        # 关闭
        command = outputMsg.Robotiq2FGripper_robot_output()
        self._gripper_pub.publish(command)
        rospy.sleep(0.2)
        
        # 激活
        command = outputMsg.Robotiq2FGripper_robot_output()
        command.rPR = 255
        command.rACT = 1
        command.rGTO = 1
        command.rSP = 255
        command.rFR = 50
        self._gripper_pub.publish(command)
        rospy.sleep(0.2)
        
        print("[KukaRealEnv] Gripper reset and activated")
        return command
    
    def _get_ee_pose_callback(self, msg):
        """ROS 回调：接收末端执行器位置"""
        self.ee_current_position = np.array([
            msg.poseStamped.pose.position.x,
            msg.poseStamped.pose.position.y,
            msg.poseStamped.pose.position.z - self._ee_offset
        ])
    
    def _create_ee_pose_msg(self, x: float, y: float, z: float):
        """创建末端位置 ROS 消息"""
        goal = PoseStamped()
        goal.header.seq = 1
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "ee_link"
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = z + self._ee_offset
        goal.pose.orientation.x = 0
        goal.pose.orientation.y = 1
        goal.pose.orientation.z = 0
        goal.pose.orientation.w = 0
        return goal
    
    def _thread_job(self):
        """ROS 线程函数"""
        if ROS_AVAILABLE:
            rospy.spin()
    
    # ==================== 清理 ====================
    
    def close(self):
        """关闭环境，释放资源"""
        if self._ros_initialized and ROS_AVAILABLE:
            rospy.signal_shutdown("KukaRealEnv closed")
        print("[KukaRealEnv] Environment closed")
    
    # ==================== 属性 ====================
    
    @property
    def observation_space_dim(self) -> int:
        """观测空间维度"""
        return 39
    
    @property
    def action_space_dim(self) -> int:
        """动作空间维度"""
        return 4
    
    @property
    def state_dim(self) -> int:
        """Transformer 使用的状态维度"""
        return 21


# ==================== 便捷函数 ====================

def create_kuka_real_env(task_name: str = "reach-v2") -> KukaRealEnv:
    """
    创建真实机器人环境的便捷函数
    
    Args:
        task_name: 任务名称
    
    Returns:
        KukaRealEnv 实例
    """
    return KukaRealEnv(task_name=task_name)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test KUKA Real Environment")
    parser.add_argument("--task", type=str, default="reach-v2", help="Task name")
    parser.add_argument("--goal", type=float, nargs=3, default=[0.7, 0.0, 0.15], 
                        help="Goal position [x, y, z]")
    parser.add_argument("--steps", type=int, default=50, help="Number of steps")
    args = parser.parse_args()
    
    # 创建环境
    env = KukaRealEnv(task_name=args.task)
    env.set_goal(np.array(args.goal))
    
    # 重置
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial state for transformer: {env.get_state_for_transformer().shape}")
    
    # 运行几步
    for i in range(args.steps):
        # 随机动作
        action = np.random.uniform(-0.1, 0.1, size=4)
        obs, reward, done, info = env.step(action)
        
        print(f"Step {i+1}: reward={reward:.3f}, success={info['success']}")
        
        if done or info['success']:
            break
    
    env.close()