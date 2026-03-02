
# mtrl/env/real_envs/base_env.py
"""
KUKA 真实机器人基础环境类

所有任务环境的基类，提供：
- ROS 通信（发布者、订阅者）
- 坐标系转换（仿真 ↔ 真实）
- 基础观测和动作处理
- 两种目标设置方法：手动设置和相机检测
"""
import os
import sys
import numpy as np
import math
import threading
from typing import Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod

# 尝试导入 ROS
try:
    import rospy
    from iiwa_msgs.msg import CartesianPose
    from geometry_msgs.msg import PoseStamped
    from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg
    ROS_AVAILABLE = True
except ImportError:
    print("[KukaBaseRealEnv] Warning: ROS not available.")
    ROS_AVAILABLE = False
    rospy = None
    CartesianPose = None
    PoseStamped = None
    outputMsg = None


class KukaBaseRealEnv(ABC):
    """
    KUKA 真实机器人基础环境类
    
    所有具体任务环境都继承自此类。提供：
    - ROS 通信接口
    - 坐标系转换
    - 观测和动作处理
    - 两种目标设置方法
    
    子类需要实现：
    - compute_reward(): 计算奖励
    - _check_success(): 判断任务是否成功
    - 任务特定的默认目标和初始位置
    """
    
    # 成功阈值（可被子类覆盖）
    SUCCESS_THRESHOLD = 0.05
    
    # Apriltag 未检测到标记时返回的默认位置（原始读数，未加任何偏移）
    # 当机械臂遮挡标记时会返回此值，需要过滤掉
    APRILTAG_DEFAULT_POSITION = np.array([0.63942897, 0.03250715, 0.07286723])
    APRILTAG_DEFAULT_TOLERANCE = 0.01  # 判定为默认值的容差 (m)
    
    def __init__(
        self,
        task_name: str = "base",
        init_ros: bool = True,
        max_episode_steps: int = 200,
        use_camera: bool = False,
        use_camera_apriltag: bool = False,
        observation_offset: Optional[np.ndarray] = None,
        camera_offset: Optional[np.ndarray] = None,
        apriltag_offset: Optional[np.ndarray] = None,
        track_object_realtime: bool = False, # 新增：是否实时追踪物体
        track_goal_realtime: bool = False,   # 新增：是否实时追踪目标
        object_color: str = 'red',           # 新增：物体颜色
        goal_color: str = 'green',           # 新增：目标颜色
        object_tag_id: int = 0,              # 新增：物体 Apriltag ID
        goal_tag_id: int = 1,                # 新增：目标 Apriltag ID
    ):
        """
        初始化基础环境
        
        Args:
            task_name: 任务名称
            init_ros: 是否初始化 ROS
            max_episode_steps: 每个 episode 最大步数
            use_camera: 是否使用颜色相机获取目标/物体位置
            use_camera_apriltag: 是否使用 Apriltag 相机获取目标/物体位置
            observation_offset: 观测坐标偏移量 [x, y, z]
                               将所有位置观测加上此偏移，使agent以为自己在不同位置
                               例如: [0, 0.1, 0] 使agent以为Y坐标大0.1
                               用例: 机器人从Y=0.5移动到Y=0.75，但agent看到Y=0.6到Y=0.85
            camera_offset: 相机系统性校正偏移 [x, y, z] (applied to all camera readings)
            apriltag_offset: Apriltag 标记到实际物体位置的偏移 [x, y, z] (真实坐标系)
                            例如: Apriltag 贴在抽屉把手右侧 3cm, 则 apriltag_offset = [-0.03, 0, 0]
                            此偏移在坐标系转换之前应用到 Apriltag 原始读数上
            object_tag_id: Apriltag ID for object tracking (used when use_camera_apriltag=True)
            goal_tag_id: Apriltag ID for goal tracking (used when use_camera_apriltag=True)
        """
        self.task_name = task_name
        self.max_episode_steps = max_episode_steps
        self.curr_path_length = 0
        self.use_camera = use_camera
        self.use_camera_apriltag = use_camera_apriltag
        self.track_object_realtime = track_object_realtime
        self.track_goal_realtime = track_goal_realtime
        self.object_color = object_color
        self.goal_color = goal_color
        self.object_tag_id = object_tag_id
        self.goal_tag_id = goal_tag_id
        
        # 动作缩放（与 Metaworld 相同）
        self.action_scale = 1.0 / 100
        
        # ==================== 坐标偏移功能 ====================
        # observation_offset: 加到所有观测位置上的偏移量
        # 机器人实际在 Y=0.5，但偏移 [0, 0.1, 0] 后，agent看到 Y=0.6
        if observation_offset is not None:
            self.observation_offset = np.array(observation_offset, dtype=np.float64)
        else:
            self.observation_offset = np.zeros(3, dtype=np.float64)

        if camera_offset is not None:
            self.camera_offset = np.array(camera_offset, dtype=np.float64)
        else:
            self.camera_offset = np.zeros(3, dtype=np.float64)
        
        # apriltag_offset: Apriltag 标记到实际物体位置的偏移
        # 在真实坐标系中, applied to apriltag raw reading BEFORE coordinate conversion
        # 例如: Apriltag 贴在窗户把手右侧 3cm, 设置 apriltag_offset = [-0.03, 0, 0]
        if apriltag_offset is not None:
            self.apriltag_offset = np.array(apriltag_offset, dtype=np.float64)
        else:
            self.apriltag_offset = np.zeros(3, dtype=np.float64)
        # =====================================================
        
        # 工作空间边界（真实机器人坐标系）
        self.hand_low = np.array([0.4, -0.6, 0.01])
        self.hand_high = np.array([0.9, 0.6, 0.6])
        
        # 初始位置（真实坐标系）- 子类可覆盖
        self.ee_init_position = np.array([0.6, 0.0, 0.2])
        self.ee_current_position = self.ee_init_position.copy()
        
        # 目标位置（真实坐标系）- 子类应设置默认值
        self.goal = np.array([0.7, 0.0, 0.15])
        
        # 物体位置（真实坐标系）
        self.object_position = np.zeros(3)
        self.obj_init_pos = np.zeros(3)
        
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
        self.camera_apriltag = None
        if use_camera:
            self._init_camera()
        if use_camera_apriltag:
            self._init_camera_apriltag()
        self.prev_ee_pos_sim = self.coordinate_from_real_to_sim(self.ee_current_position)
        self.prev_obj_pos_sim = self.coordinate_from_real_to_sim(self.object_position)
        self.prev_gripper_state = 1.0
        
        print(f"[{self.__class__.__name__}] Initialized for task: {task_name}")
        if not np.allclose(self.observation_offset, np.zeros(3)):
            print(f"[{self.__class__.__name__}] Observation offset: {self.observation_offset}")
        if not np.allclose(self.apriltag_offset, np.zeros(3)):
            print(f"[{self.__class__.__name__}] Apriltag offset: {self.apriltag_offset}")
    
    def _init_ros(self):
        """初始化 ROS 通信"""
        try:
            rospy.init_node('kuka_real_env', anonymous=True)
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
            self._rate = rospy.Rate(20)
            
            self._thread = threading.Thread(target=self._thread_job)
            self._thread.daemon = True
            self._thread.start()
            
            self._ros_initialized = True
            print(f"[{self.__class__.__name__}] ROS initialized")
        except Exception as e:
            print(f"[{self.__class__.__name__}] ROS init failed: {e}")
            self._ros_initialized = False
    
    def _init_camera(self):
        """初始化相机（如果可用）"""
        try:
            # 尝试导入相机模块
            # 注意：需要根据实际相机模块路径调整
            current_file_path = os.path.abspath(__file__)

            # 2. 向上跳转 4 级到达 liu 目录
            # 第一级: real_envs, 第二级: env, 第三级: mtrl, 第四级: RLCE, 第五级才是 liu
            # 注意：根据你提供的路径，RLCE 在 liu 下面，所以需要向上跳 5 层到达 liu 所在的层级
            liu_path = os.path.abspath(os.path.join(current_file_path, "../../../../../"))

            # 3. 将 liu 目录加入到环境变量
            if liu_path not in sys.path:
                sys.path.append(liu_path)
            from camera import Camera
            self.camera = Camera(150)  # 150ms 曝光时间
            print(f"[{self.__class__.__name__}] Camera initialized")
        except ImportError:
            print(f"[{self.__class__.__name__}] Camera module not available")
            self.camera = None
    
    def _init_camera_apriltag(self):
        """初始化 Apriltag 相机（如果可用）"""
        try:
            current_file_path = os.path.abspath(__file__)
            liu_path = os.path.abspath(os.path.join(current_file_path, "../../../../../"))
            if liu_path not in sys.path:
                sys.path.append(liu_path)
            from camera import Camera_Apriltag
            self.camera_apriltag = Camera_Apriltag(warmup_loop=60, families='tag36h11')
            print(f"[{self.__class__.__name__}] Camera_Apriltag initialized")
        except ImportError:
            print(f"[{self.__class__.__name__}] Camera_Apriltag module not available")
            self.camera_apriltag = None
    
    def _is_apriltag_default_reading(self, raw_position: np.ndarray) -> bool:
        """
        检查 Apriltag 原始读数是否为未检测到标记时的默认值。
        
        当机械臂遮挡 Apriltag 标记时，相机会返回一个固定的默认位置。
        此方法通过与已知默认值比较来检测这种情况。
        
        Args:
            raw_position: Apriltag 原始读数 [3]（未加偏移）
        
        Returns:
            bool: True 表示该读数是默认值（无效），应被忽略
        """
        distance = np.linalg.norm(np.array(raw_position) - self.APRILTAG_DEFAULT_POSITION)
        return distance < self.APRILTAG_DEFAULT_TOLERANCE

    def set_object_from_apriltag(self, tag_id: Optional[int] = None):
        """
        使用 Apriltag 相机检测物体位置
        
        偏移应用顺序:
        1. Apriltag 原始读数 (相机坐标系 → 机器人 base frame)
        2. + apriltag_offset (标记到物体的物理偏移，真实坐标系)
        3. + camera_offset (相机系统性校正)
        
        Args:
            tag_id: Apriltag ID (default: self.object_tag_id)
        
        Returns:
            np.ndarray: 检测到的物体位置 (真实坐标系)
        """
        if self.camera_apriltag is None:
            print(f"[{self.__class__.__name__}] Warning: Camera_Apriltag not available")
            return self.object_position
        
        if tag_id is None:
            tag_id = self.object_tag_id
        
        try:
            raw_position = self.camera_apriltag.getPosition_inBaseFrame(id=tag_id)
            
            # 检查是否为默认读数（标记被遮挡时的无效值）
            if self._is_apriltag_default_reading(raw_position):
                print(f"[{self.__class__.__name__}] Object Apriltag reading rejected (default/occluded): raw={np.array(raw_position)}, keeping last valid position")
                return self.object_position
            
            # 1. apriltag_offset: 标记到物体的物理偏移 (在真实坐标系中)
            corrected = np.array(raw_position) + self.apriltag_offset
            # 2. camera_offset: 相机系统性校正
            corrected = corrected + self.camera_offset
            
            self.object_position = corrected
            self.obj_init_pos = self.object_position.copy()
            print(f"[{self.__class__.__name__}] Object from Apriltag (tag_id={tag_id}): raw={np.array(raw_position)} → final={corrected}")
            return self.object_position
        except Exception as e:
            print(f"[{self.__class__.__name__}] Apriltag detection failed: {e}")
            return self.object_position
    
    def set_goal_from_apriltag(self, tag_id: Optional[int] = None):
        """
        使用 Apriltag 相机检测目标位置
        
        偏移应用顺序: raw + apriltag_offset + camera_offset
        
        Args:
            tag_id: Apriltag ID (default: self.goal_tag_id)
        
        Returns:
            np.ndarray: 检测到的目标位置 (真实坐标系)
        """
        if self.camera_apriltag is None:
            print(f"[{self.__class__.__name__}] Warning: Camera_Apriltag not available")
            return self.goal
        
        if tag_id is None:
            tag_id = self.goal_tag_id
        
        try:
            raw_position = self.camera_apriltag.getPosition_inBaseFrame(id=tag_id)
            
            # 检查是否为默认读数（标记被遮挡时的无效值）
            if self._is_apriltag_default_reading(raw_position):
                print(f"[{self.__class__.__name__}] Goal Apriltag reading rejected (default/occluded): raw={np.array(raw_position)}, keeping last valid position")
                return self.goal
            
            corrected = np.array(raw_position) + self.apriltag_offset + self.camera_offset
            self.goal = corrected
            print(f"[{self.__class__.__name__}] Goal from Apriltag (tag_id={tag_id}): raw={np.array(raw_position)} → final={corrected}")
            return self.goal
        except Exception as e:
            print(f"[{self.__class__.__name__}] Apriltag goal detection failed: {e}")
            return self.goal
    
    # ==================== 坐标变换 ====================
    
    def coordinate_from_sim_to_real(self, pos_sim: np.ndarray) -> np.ndarray:
        """仿真坐标系 -> 真实机器人坐标系（90°绕Z轴旋转）"""
        R_real_sim = np.array([
            [0.0, 1.0, 0.0], 
            [-1.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0]
        ])
        return np.matmul(R_real_sim, pos_sim)

    def coordinate_from_real_to_sim(self, pos_real: np.ndarray) -> np.ndarray:
        """真实机器人坐标系 -> 仿真坐标系"""
        R_sim_real = np.array([
            [0.0, -1.0, 0.0], 
            [1.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0]
        ])
        return np.matmul(R_sim_real, pos_real)
    
    # ==================== 目标设置方法 ====================
    
    def set_goal(self, goal: np.ndarray):
        """
        方法1: 手动设置目标位置
        
        Args:
            goal: [3] 目标位置 (真实坐标系)
        """
        assert goal.shape[0] == 3
        self.goal = goal.copy()
        print(f"[{self.__class__.__name__}] Goal set to: {self.goal}")
    
    def set_goal_from_camera(self, color: str = 'red'):
        """
        方法2: 使用相机检测目标位置 (带异常值过滤)
        
        Args:
            color: 目标颜色 ('green', 'red', 'blue', etc.)
        
        Returns:
            np.ndarray: 检测到的目标位置 (真实坐标系，带camera_offset校正)
        """
        if self.camera is None:
            print(f"[{self.__class__.__name__}] Warning: Camera not available, using default goal")
            return self.goal
        
        try:
            raw_position = self.camera.getPosition_inBaseFrame(color=color)
            corrected = np.array(raw_position) + self.camera_offset
            
            # 验证读数是否合理
            if not self._validate_camera_reading(corrected, self.goal):
                return self.goal  # 保留上次有效读数
            
            self.goal = corrected
            return self.goal
        except Exception as e:
            print(f"[{self.__class__.__name__}] Camera detection failed: {e}")
            return self.goal
    
    def set_object_position(self, position: np.ndarray):
        """手动设置物体位置"""
        assert position.shape[0] == 3
        self.object_position = position.copy()
        self.obj_init_pos = position.copy()
        print(f"[{self.__class__.__name__}] Object position set to: {self.object_position}")
    
    def set_object_from_camera(self, color: str = 'red'):
        """
        使用相机检测物体位置 (带异常值过滤)
        
        Args:
            color: 物体颜色 ('green', 'red', 'blue', etc.)
        
        Returns:
            np.ndarray: 检测到的物体位置 (真实坐标系，带camera_offset校正)
        """
        if self.camera is None:
            print(f"[{self.__class__.__name__}] Warning: Camera not available")
            return self.object_position
        
        try:
            raw_position = self.camera.getPosition_inBaseFrame(color=color)
            corrected = np.array(raw_position) + self.camera_offset
            
            # 验证读数是否合理
            if not self._validate_camera_reading(corrected, self.object_position):
                return self.object_position  # 保留上次有效读数
            
            self.object_position = corrected
            self.obj_init_pos = self.object_position.copy()
            return self.object_position
        except Exception as e:
            print(f"[{self.__class__.__name__}] Camera detection failed: {e}")
            return self.object_position
    
    def set_observation_offset(self, offset: np.ndarray):
        """
        设置观测坐标偏移量
        
        将此偏移加到所有观测位置上，使agent以为自己在不同位置。
        
        用例: 机器人从 Y=0.5 移动到 Y=0.75，但偏移 [0, 0.1, 0] 后，
        agent看到 Y=0.6 到 Y=0.85（agent的训练范围）
        
        Args:
            offset: [3] 坐标偏移量 [x, y, z]
        """
        assert offset.shape[0] == 3
        self.observation_offset = offset.copy()
        print(f"[{self.__class__.__name__}] Observation offset set to: {self.observation_offset}")

    def set_camera_offset(self, offset: np.ndarray):
        """
        设置相机检测位置的校正偏移量
        
        如果相机检测到的位置有系统性偏移，可以用此参数校正。
        例如: 相机检测到的Y坐标总是偏大0.02，设置 [0, -0.02, 0]
        
        Args:
            offset: [3] 校正偏移量 [x, y, z]
        
        Raises:
            ValueError: 如果偏移量不是3维向量
        """
        if offset.shape[0] != 3:
            raise ValueError(f"Camera offset must be 3D, got shape {offset.shape}")
        self.camera_offset = offset.copy()
        print(f"[{self.__class__.__name__}] Camera offset set to: {self.camera_offset}")
    
    def set_apriltag_offset(self, offset: np.ndarray):
        """
        设置 Apriltag 标记到实际物体位置的偏移量（真实坐标系）
        
        这个偏移在坐标系转换之前应用。
        例如: Apriltag 贴在窗户把手右侧 3cm, 设置 apriltag_offset = [-0.03, 0, 0]
        
        Args:
            offset: [3] 偏移量 [x, y, z] (真实坐标系)
        """
        if offset.shape[0] != 3:
            raise ValueError(f"Apriltag offset must be 3D, got shape {offset.shape}")
        self.apriltag_offset = offset.copy()
        print(f"[{self.__class__.__name__}] Apriltag offset set to: {self.apriltag_offset}")
    
    def _validate_camera_reading(self, position: np.ndarray, last_position: np.ndarray) -> bool:
        """
        验证相机读数是否合理，过滤异常值
        
        Color filter 有时会产生大范围波动。这个方法检查:
        1. 任何轴的绝对值不应超过 1.0m (机器人工作空间之外)
        2. 如果有上次读数，单次跳变不应超过 0.15m (物理上不可能的位移)
        
        Args:
            position: 新的相机读数 [3]
            last_position: 上次有效读数 [3]
        
        Returns:
            bool: True 如果读数有效
        """
        # 检查 1: 绝对范围检查 - 任何轴超过 1.0m 认为无效
        if np.any(np.abs(position) > 1.0):
            print(f"[{self.__class__.__name__}] Camera reading rejected (out of range): {position}")
            return False
        
        # 检查 2: 工作空间边界检查 - 必须在合理范围内
        # 真实机器人工作空间: X=[0.4, 0.9], Y=[-0.6, 0.6], Z=[0.0, 0.6]
        if position[0] < 0.2 or position[0] > 1.0:
            print(f"[{self.__class__.__name__}] Camera reading rejected (X out of workspace): {position}")
            return False
        if position[1]>0.4:
            print(f"[{self.__class__.__name__}] Camera reading rejected (Y out of workspace): {position}")
            return False
        if position[2] < -0.05 or position[2] > 0.3:
            print(f"[{self.__class__.__name__}] Camera reading rejected (Z out of workspace): {position}")
            return False
        
        # 检查 3: 跳变检查 - 单步位移过大认为无效
        if last_position is not None:
            displacement = np.linalg.norm(position - last_position)
            if displacement > 0.15:
                print(f"[{self.__class__.__name__}] Camera reading rejected (jump={displacement:.3f}m): {position}")
                return False
        
        return True
    
    def set_ee_init_position(self, position: np.ndarray):
        """
        设置机械臂初始位置
        
        Args:
            position: [3] 初始位置 (真实坐标系)
        """
        assert position.shape[0] == 3
        self.ee_init_position = position.copy()
        print(f"[{self.__class__.__name__}] EE init position set to: {self.ee_init_position}")
    
    # ==================== 核心接口 ====================
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        while not rospy.is_shutdown():
            if self.reset_manipulator():
                break
        rospy.sleep(0.1)
        # reset gripper
        self.command = self.reset_gripper()
        rospy.sleep(0.1)
        self.set_gripper_action(1)
        rospy.sleep(2)
        print("gripper initialized")
        if self.use_camera:
            
            # --- 1. 追踪物体 (Object) ---
            if self.track_object_realtime:
                # 1. 保存旧位置 (作为备份)
                last_object_pos = self.object_position.copy()
                
                # 2. 获取新位置 (注意：这步操作已经修改了 self.object_position)
                self.set_object_from_camera(color=self.object_color)
                
            # --- 2. 追踪目标 (Goal) ---
            if self.track_goal_realtime:
                # 1. 保存旧位置
                last_goal_pos = self.goal.copy()
                
                # 2. 获取新位置
                self.set_goal_from_camera(color=self.goal_color)
        
        elif self.use_camera_apriltag:
            
            # --- 1. 追踪物体 (Object) via Apriltag ---
            if self.track_object_realtime:
                self.set_object_from_apriltag(tag_id=self.object_tag_id)
                
            # --- 2. 追踪目标 (Goal) via Apriltag ---
            if self.track_goal_realtime:
                self.set_goal_from_apriltag(tag_id=self.goal_tag_id)
        
        try:
            obs = self._get_obs() # get obs in {Sim}
        except:
            obs = self.ee_current_position
        rospy.sleep(0.1)
        self.prev_ee_pos_sim = self.coordinate_from_real_to_sim(self.ee_current_position)
        # 注意：如果你在 reset 外面修改了 object_position，这里也能正确同步
        self.prev_obj_pos_sim = self.coordinate_from_real_to_sim(self.object_position)
        self.curr_path_length = 0
        return obs
    
    def reset_manipulator(self):
        '''
        reset the manipulator to its initial position & orientation
        '''
        while np.linalg.norm(self.ee_init_position - self.ee_current_position) > 0.01:
            
            step_goal = self._set_ee_pose(
                self.ee_init_position[0],
                self.ee_init_position[1],
                self.ee_init_position[2])
            print(step_goal)
            self._robot_pub.publish(step_goal)
            print("goal published")
            self._rate.sleep()
        print(f"KUKA ee reset to init position: {self.ee_current_position}")
        return True
    
    def reset_gripper(self):
        count = 0
        while not rospy.is_shutdown():
            command = self._deactivate_gripper()
            self._rate.sleep()
            count+=1
            if count > 10:
                break
        rospy.sleep(0.2)
        command = self._activate_gripper()
        rospy.sleep(0.2)
        print("Robotiq gripper reset")
        return command
    
    def _deactivate_gripper(self):
        
        command = outputMsg.Robotiq2FGripper_robot_output()
        self._gripper_pub.publish(command)
        return command

    def _activate_gripper(self):
        
        command = outputMsg.Robotiq2FGripper_robot_output()
        command.rPR = 255 #0-255. open:0  close:255
        command.rACT = 1
        command.rGTO = 1
        command.rSP = 255
        command.rFR = 50
        self._gripper_pub.publish(command)
        return command
    
    def reset_after_success(self) -> np.ndarray:
        """
        任务成功后重置（不重置夹爪）
        用于连续执行多次任务
        """
        if self._ros_initialized:
            self._reset_manipulator()
            rospy.sleep(0.2)
        else:
            self.ee_current_position = self.ee_init_position.copy()
        
        self.curr_path_length = 0
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行动作
        
        Args:
            action: [4] 动作向量，范围 [-1, 1]
                   [delta_x, delta_y, delta_z, gripper] (仿真坐标系)
        
        Returns:
            observation, reward, done, info
        """
        assert action.shape[0] == 4
        action = np.clip(action, -1, 1)
        
        # 执行动作
        self.set_xyz_action(action[:3])
        self.set_gripper_action(action[3])
        
        if self._ros_initialized and self._rate:
            self._rate.sleep()
        
        if self.use_camera:
            
            # --- 1. 追踪物体 (Object) ---
            if self.track_object_realtime:
                # 1. 保存旧位置 (作为备份)
                last_object_pos = self.object_position.copy()
                
                # 2. 获取新位置 (注意：这步操作已经修改了 self.object_position)
                self.set_object_from_camera(color=self.object_color)
                
            # --- 2. 追踪目标 (Goal) ---
            if self.track_goal_realtime:
                # 1. 保存旧位置
                last_goal_pos = self.goal.copy()
                
                # 2. 获取新位置
                self.set_goal_from_camera(color=self.goal_color)
        
        elif self.use_camera_apriltag:
            
            # --- 1. 追踪物体 (Object) via Apriltag ---
            if self.track_object_realtime:
                self.set_object_from_apriltag(tag_id=self.object_tag_id)
                
            # --- 2. 追踪目标 (Goal) via Apriltag ---
            if self.track_goal_realtime:
                self.set_goal_from_apriltag(tag_id=self.goal_tag_id)

        # 获取观测
        obs = self._get_obs()
        obs_dict = self._get_obs_dict()
        
        # 计算奖励
        reward, reward_info = self.compute_reward(action, obs_dict)
        
        # 检查成功
        success = self._check_success(obs_dict)
        done = self.curr_path_length >= self.max_episode_steps
        
        self.curr_path_length += 1
        
        info = {
            'success': float(success),
            'goal': self.goal.copy(),
            **reward_info
        }
        
        return obs, reward, done, info
    
    def set_xyz_action(self, xyz_action_sim: np.ndarray):
        """设置 XYZ 动作（输入为仿真坐标系）"""
        xyz_action_real = self.coordinate_from_sim_to_real(xyz_action_sim)
        pos_delta = xyz_action_real * self.action_scale
        new_pos = self.ee_current_position + pos_delta
        new_pos = np.clip(new_pos, self.hand_low, self.hand_high)
        
        if self._ros_initialized and self._robot_pub:
            goal = self._create_ee_pose_msg(new_pos[0], new_pos[1], new_pos[2])
            self._robot_pub.publish(goal)
        else:
            self.ee_current_position = new_pos
    
    def set_gripper_action(self, gripper_state: float):
        """设置夹爪动作 (-1=开, 1=闭)"""
        if not self._ros_initialized or self.command is None:
            return
        
        # 离散化夹爪状态
        state = math.floor(gripper_state / 0.4) * 0.4 + 0.5
        position = int(128 - 127.5 * state)
        position = np.clip(position, 0, 255)
        
        self.command.rPR = position
        if self._gripper_pub:
            self._gripper_pub.publish(self.command)
    
    # ==================== 观测 ====================

    def _get_gripper_state(self) -> float:
        """
        获取当前夹爪状态（用于观测），匹配 MetaWorld obs[3] 的范围 [0, 1]
        
        MetaWorld 中 obs[3] = clip(gripper_distance / 0.1, 0, 1):
          - 0.0 = 完全关闭 (手指接触)
          - 1.0 = 完全打开 (手指距离 ≈ 0.1m)
        
        Robotiq 夹爪: rPR=0 表示完全打开, rPR=255 表示完全关闭
        映射: obs_state = (255 - rPR) / 255 ∈ [0, 1]
        
        注意: 这与 set_gripper_action 的输入范围 [-1, 1] 不同！
              set_gripper_action 使用 MetaWorld 动作空间 [-1, 1]
              _get_gripper_state 使用 MetaWorld 观测空间 [0, 1]
        
        Returns:
            float: 夹爪开合程度 (0.0 = 完全关闭, 1.0 = 完全打开)
                   与 MetaWorld 观测空间 obs[3] 的范围一致
        """
        if self._ros_initialized and self.command is not None:
            # 从 Robotiq 夹爪命令读取当前位置
            # rPR: 0 = 完全打开, 255 = 完全关闭
            try:
                position = self.command.rPR
                gripper_state = (255 - position) / 255.0
                return np.clip(gripper_state, 0.0, 1.0)
            except AttributeError as e:
                print(f"[{self.__class__.__name__}] Warning: Could not read gripper state: {e}, using default 1.0")
                return 1.0  # 默认打开
        else:
            # 无 ROS 时使用缓存的状态
            return self.prev_gripper_state
    
    def _get_object_quat(self) -> np.ndarray:
        """
        获取物体四元数，子类可覆盖
        
        与 MetaWorld 的 _get_quat_objects() 对应:
        - reach/push/pick_place: [0, 0, 0, 1] (无旋转，xyzw 格式)
        - window-close/window-open: [0, 0, 0, 0] (MetaWorld 约定)
        - drawer-open: [1, 0, 0, 0] (MuJoCo 默认，wxyz 格式)
        
        Returns:
            np.ndarray: 四元数 [4]
        """
        return np.array([0.0, 0.0, 0.0, 1.0])
    
    def _get_obs(self) -> np.ndarray:
        """
        获取39维观测（仿真坐标系）
        
        NOTE: 如果设置了 observation_offset，所有位置会加上此偏移
        使 agent 以为自己在不同位置
        """
        # 获取实际位置（仿真坐标系）
        pos_hand_actual = self.coordinate_from_real_to_sim(self.ee_current_position)
        pos_obj_actual = self.coordinate_from_real_to_sim(self.object_position)
        pos_goal_actual = self.coordinate_from_real_to_sim(self.goal)
        
        # 应用观测偏移 - agent 看到偏移后的位置
        pos_hand_agent = pos_hand_actual + self.observation_offset
        pos_obj_agent = pos_obj_actual + self.observation_offset
        pos_goal_agent = pos_goal_actual + self.observation_offset
        
        # 获取夹爪状态
        curr_gripper = self._get_gripper_state()
        
        # 获取任务对应的物体四元数
        obj_quat = self._get_object_quat()
        
        # --- 构造 Current Frame (使用偏移后的位置) ---
        curr_frame = np.zeros(18)
        curr_frame[0:3] = pos_hand_agent
        curr_frame[3] = curr_gripper
        curr_frame[4:7] = pos_obj_agent
        curr_frame[7:11] = obj_quat
        
        # Previous frame 也需要应用偏移
        prev_hand_agent = self.prev_ee_pos_sim + self.observation_offset
        prev_obj_agent = self.prev_obj_pos_sim + self.observation_offset
        
        # --- 构造 Previous Frame (使用偏移后的位置) ---
        prev_frame = np.zeros(18)
        prev_frame[0:3] = prev_hand_agent
        prev_frame[3] = self.prev_gripper_state 
        prev_frame[4:7] = prev_obj_agent
        prev_frame[7:11] = obj_quat

        # 构造 Obs
        obs = np.hstack([
            curr_frame,       # 0-17
            prev_frame,       # 18-35
            pos_goal_agent    # 36-38
        ])
        
        # --- 更新缓存 (存储实际位置，不是偏移后的) ---
        self.prev_ee_pos_sim = pos_hand_actual.copy()
        self.prev_obj_pos_sim = pos_obj_actual.copy()
        self.prev_gripper_state = curr_gripper # 更新夹爪缓存

        return obs
    
    def _get_obs_dict(self) -> Dict[str, Any]:
        """
        获取观测字典
        
        NOTE: 此字典用于奖励计算，所以返回的是真实位置（不带偏移）
        observation_offset 只影响 _get_obs() 返回的观测向量
        """
        return {
            'state_observation': self._get_obs(),  # 这里包含了偏移
            'state_desired_goal': self.goal.copy(),  # 真实目标位置（用于奖励计算）
            'state_achieved_goal': self.object_position.copy(),
            'ee_position': self.ee_current_position.copy(),
            'object_position': self.object_position.copy(),
        }
    
    def get_state_for_transformer(self) -> np.ndarray:
        """获取21维 Transformer 状态 (带偏移修复版)"""
        # 1. 坐标转换 (Real -> Sim) 并应用偏移
        pos_hand_agent = self.coordinate_from_real_to_sim(self.ee_current_position) + self.observation_offset
        pos_obj_agent = self.coordinate_from_real_to_sim(self.object_position) + self.observation_offset
        pos_goal_agent = self.coordinate_from_real_to_sim(self.goal) + self.observation_offset
        
        # 2. 构造关键缺失信息
        
        # [Index 3] 夹爪状态
        gripper_state = np.array([self._get_gripper_state()])
        
        # [Index 7-10] 物体姿态 (四元数): 使用任务对应的四元数
        obj_quat_sim = self._get_object_quat()

        # 3. 计算 Padding
        # 目前已有维度: 3(Hand) + 1(Grip) + 3(Obj) + 4(Quat) + 3(Goal) = 14 维
        # 目标总维度是 21，所以还需要 7 个 0
        padding = np.zeros(21 - 14)
        
        # 4. 拼接 (使用带偏移的位置)
        state = np.hstack([
            pos_hand_agent,   # 0-2:  Hand Pos (with offset)
            gripper_state,    # 3:    Gripper State
            pos_obj_agent,    # 4-6:  Object Pos (with offset)
            obj_quat_sim,     # 7-10: Object Quat
            padding,
            pos_goal_agent,   # Goal (with offset)
        ])
        
        return state
    
    # ==================== 抽象方法 ====================
    
    @abstractmethod
    def compute_reward(self, action: np.ndarray, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        计算奖励（子类必须实现）
        
        Args:
            action: 动作
            obs_dict: 观测字典
        
        Returns:
            (reward, info_dict)
        """
        raise NotImplementedError
    
    @abstractmethod
    def _check_success(self, obs_dict: Dict[str, Any]) -> bool:
        """
        检查任务是否成功（子类必须实现）
        
        Args:
            obs_dict: 观测字典
        
        Returns:
            是否成功
        """
        raise NotImplementedError
    
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
                break
        
        print(f"[{self.__class__.__name__}] Manipulator reset to: {self.ee_current_position}")
    
    def _reset_gripper(self):
        """重置夹爪"""
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
        
        return command
    
    def _get_ee_pose_callback(self, msg):
        """ROS 回调"""
        self.ee_current_position = np.array([
            msg.poseStamped.pose.position.x,
            msg.poseStamped.pose.position.y,
            msg.poseStamped.pose.position.z - self._ee_offset
        ])
    def _set_ee_pose(self, x,y,z):
        '''
        Input: 
            action: array(x,y,z) -- the position of gripper center according to the base frame
        Notation:
            The offset between gripper center and joint 7 (ee) is 0.16 m in z axis
            Set ee posistion and orientation, the orientation is fixed 
            (corresponding to the euler orientation: x:-180, y:0, z:-180)
        '''
        
        goal = PoseStamped()

        goal.header.seq = 1
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "ee_link"
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = z + self._ee_offset # there is 0.16 m offset between gripper center and joint 7 ee
        goal.pose.orientation.x = 0
        goal.pose.orientation.y = 1
        goal.pose.orientation.z = 0
        goal.pose.orientation.w = 0
        
        return goal
    
    def _create_ee_pose_msg(self, x: float, y: float, z: float):
        """创建 ROS 消息"""
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
        """ROS 线程"""
        if ROS_AVAILABLE:
            rospy.spin()
    
    def close(self):
        """关闭环境"""
        if self._ros_initialized and ROS_AVAILABLE:
            rospy.signal_shutdown("Environment closed")
        print(f"[{self.__class__.__name__}] Closed")