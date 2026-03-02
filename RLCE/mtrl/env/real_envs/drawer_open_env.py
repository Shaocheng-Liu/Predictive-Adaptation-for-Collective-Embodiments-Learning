# mtrl/env/real_envs/drawer_open_env.py
"""
KUKA Drawer Open 任务环境

任务: 打开抽屉
成功条件: 抽屉把手与目标位置距离 < 0.03m

基于 MetaWorld sawyer_drawer_open_v2.py 的 Reward 逻辑，
适配 real_envs/base_env.py 的真机接口风格。
"""

import numpy as np
import sys
import os
from typing import Dict, Any, Tuple, Optional
from mtrl.env.real_envs.base_env import KukaBaseRealEnv

# Import MetaWorld's reward_utils for accurate reward computation
metaworld_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'Metaworld')
if os.path.exists(metaworld_path) and metaworld_path not in sys.path:
    sys.path.insert(0, metaworld_path)

try:
    from metaworld.envs.mujoco.utils import reward_utils
    tolerance = reward_utils.tolerance
except ImportError:
    print("[drawer_open_env] Warning: MetaWorld not available, using local fallback")
    def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='long_tail', value_at_margin=0.1):
        """Fallback tolerance matching MetaWorld's implementation."""
        lower, upper = bounds
        if lower <= x <= upper:
            return 1.0
        if margin == 0:
            return 0.0
        if x < lower:
            d = (lower - x) / margin
        else:
            d = (x - upper) / margin
        if sigmoid == 'long_tail':
            scale = np.sqrt(1 / value_at_margin - 1)
            return 1 / ((d * scale) ** 2 + 1)
        elif sigmoid == 'gaussian':
            scale = np.sqrt(-2 * np.log(value_at_margin))
            return np.exp(-0.5 * (d * scale) ** 2)
        else:
            return max(0, 1 - d)


class KukaDrawerOpenEnv(KukaBaseRealEnv):
    """
    Drawer Open 任务环境
    
    目标: 拉开抽屉
    
    使用方法:
        env = KukaDrawerOpenEnv()
        env.set_goal(np.array([0.7, -0.1, 0.15]))      # 设置目标位置（打开后的把手位置）
        env.set_object_position(np.array([0.7, 0.0, 0.15]))  # 设置把手初始位置
        # 或使用相机
        env.set_goal_from_camera('red')
        env.set_object_from_camera('green')
    """
    
    SUCCESS_THRESHOLD = 0.06  # 3cm
    
    def __init__(self, **kwargs):
        # Drawer Open 任务默认配置：需要实时追踪物体（抽屉把手）
        kwargs.setdefault('track_object_realtime', True)
        kwargs.setdefault('track_goal_realtime', False)
        kwargs.setdefault('object_color', 'red')
        kwargs.setdefault('goal_color', 'green')
        super().__init__(task_name="drawer-open-v2", **kwargs)
        
        # 最大拉出距离
        self.maxDist = 0.2
        
        # 默认把手位置（真实坐标系，关闭状态）
        self.object_position = np.array([0.7, 0.0, 0.15])  # 把手位置
        self.obj_init_pos = self.object_position.copy()
        
        # 基于把手位置自动计算目标位置
        # MetaWorld: target = body + [0, -0.36, 0.09]
        #            body = handle - [0, -0.16, 0] = handle + [0, 0.16, 0]
        #            target = handle + [0, 0.16, 0] + [0, -0.36, 0.09] = handle + [0, -0.20, 0.09]
        self.goal = self.object_position + np.array([0.0, -0.20, 0.09])
        
        # 把手初始位置（用于计算 caging 奖励）
        self.handle_pos_init = self.object_position.copy()
        
        # 初始化 TCP 位置
        self.init_tcp = self.ee_init_position.copy()
        
        # 目标奖励值（用于归一化）
        self.target_reward = 1000 * self.maxDist + 1000 * 2
    
    def set_handle_position(self, handle_pos: np.ndarray):
        """
        设置把手位置，自动计算目标位置
        
        Args:
            handle_pos: 把手位置 [x, y, z]（真实坐标系）
        """
        self.object_position = np.array(handle_pos, dtype=np.float64)
        self.obj_init_pos = self.object_position.copy()
        self.handle_pos_init = self.object_position.copy()
        # 自动计算目标: handle + [0, -0.20, 0.09]
        self.goal = self.object_position + np.array([-0.19, 0.0, 0.0])
    
    def set_handle_from_apriltag(self, tag_id: Optional[int] = None):
        """
        使用 Apriltag 检测把手位置, 自动计算 goal
        
        Apriltag 读数 + apriltag_offset + camera_offset → handle 位置 → auto-compute goal
        
        Args:
            tag_id: Apriltag ID (default: self.object_tag_id)
        """
        detected = self.set_object_from_apriltag(tag_id=tag_id)
        self.set_handle_position(detected)

        
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        obs = super().reset()
        # 保存初始 TCP 位置
        self.init_tcp = self.ee_current_position.copy()
        # 保存把手初始位置
        self.handle_pos_init = self.object_position.copy()
        return obs
    
    def compute_reward(self, action: np.ndarray, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        计算奖励 - 对齐 MetaWorld V2 Drawer Open
        
        奖励组成:
        1. reward_for_caging: 夹爪接近把手的奖励
        2. reward_for_opening: 把手接近目标（打开）位置的奖励
        """
        # 获取位置（真实坐标系）
        gripper = self.ee_current_position
        handle = self.object_position  # 把手当前位置
        target = self.goal             # 把手目标位置（打开后）
        
        # 计算把手到目标的距离
        handle_error = float(np.linalg.norm(handle - target))
        
        # Opening reward: 把手接近目标位置
        reward_for_opening = tolerance(
            handle_error, 
            bounds=(0, 0.02), 
            margin=self.maxDist, 
            sigmoid="long_tail"
        )
        
        # 把手初始位置（关闭状态）
        handle_pos_init = target + np.array([0.0, self.maxDist, 0.0])
        
        # Caging reward: 夹爪接近把手
        # 强调 XY 误差，让夹爪能够下降并 cage 把手
        scale = np.array([3.0, 3.0, 1.0])
        gripper_error = (handle - gripper) * scale
        gripper_error_init = (handle_pos_init - self.init_tcp) * scale
        
        reward_for_caging = tolerance(
            float(np.linalg.norm(gripper_error)),
            bounds=(0, 0.01),
            margin=float(np.linalg.norm(gripper_error_init)),
            sigmoid="long_tail",
        )
        
        # 组合奖励
        reward = reward_for_caging + reward_for_opening
        reward *= 5.0
        
        # 夹爪开闭状态
        tcp_opened = action[3] if action is not None else 1.0
        
        # 构造 info
        info = {
            'handle_error': handle_error,
            'gripper_to_handle': float(np.linalg.norm(handle - gripper)),
            'gripped': tcp_opened,
            'caging_reward': reward_for_caging,
            'opening_reward': reward_for_opening,
            'reward': reward,
            'success': float(handle_error <= self.SUCCESS_THRESHOLD)
        }
        
        return reward, info
    
    def _check_success(self, obs_dict: Dict[str, Any]) -> bool:
        """检查抽屉是否打开"""
        handle_error = np.linalg.norm(self.object_position - self.goal)
        return handle_error <= self.SUCCESS_THRESHOLD
    
    def _get_pos_objects(self, color: str = 'green') -> np.ndarray:
        """获取把手位置"""
        if self.camera is not None:
            try:
                return self.camera.getPosition_inBaseFrame(color=color)
            except Exception:
                pass
        return self.object_position.copy()
    
    def _get_object_quat(self) -> np.ndarray:
        """Drawer Open 四元数 - 使用 MuJoCo 默认单位四元数 [w,x,y,z]=[1,0,0,0]"""
        return np.array([1.0, 0.0, 0.0, 0.0])