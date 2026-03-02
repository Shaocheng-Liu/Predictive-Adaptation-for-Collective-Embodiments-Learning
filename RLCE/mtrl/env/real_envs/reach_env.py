# mtrl/env/real_envs/reach_env.py
"""
KUKA Reach 任务环境

任务: 移动末端执行器到目标位置
成功条件: 末端与目标距离 < 0.05m
"""

import numpy as np
import sys
import os
from typing import Dict, Any, Tuple
from mtrl.env.real_envs.base_env import KukaBaseRealEnv

# Import MetaWorld's reward_utils for accurate reward computation
metaworld_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'Metaworld')
if os.path.exists(metaworld_path) and metaworld_path not in sys.path:
    sys.path.insert(0, metaworld_path)

try:
    from metaworld.envs.mujoco.utils import reward_utils
    tolerance = reward_utils.tolerance
except ImportError:
    print("[reach_env] Warning: MetaWorld not available, using local tolerance fallback")
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
class KukaReachEnv(KukaBaseRealEnv):
    """
    Reach 任务环境
    
    目标: 将机械臂末端移动到指定目标位置
    
    使用方法:
        env = KukaReachEnv()
        env.set_goal(np.array([0.65, -0.1, 0.2]))  # 手动设置
        # 或
        env.set_goal_from_camera('green')  # 相机检测
    """
    
    SUCCESS_THRESHOLD = 0.15  # 5cm
    
    def __init__(self, **kwargs):
        # Reach 任务默认配置：只需追踪目标
        kwargs.setdefault('track_object_realtime', False)
        kwargs.setdefault('track_goal_realtime', True)
        kwargs.setdefault('object_color', 'red')
        kwargs.setdefault('goal_color', 'green')
        super().__init__(task_name="reach-v2", **kwargs)
        
        # 默认目标位置（真实坐标系）
        self.goal = np.array([0.75, 0.0, 0.15])
        
        # Reach 任务不需要物体
        self.object_position = np.array([0.5, 0, 0.02])

    
    
    def compute_reward(self, action: np.ndarray, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        计算奖励 - 使用 MetaWorld V2 Reach 的 reward_utils.tolerance
        
        完全匹配 MetaWorld SawyerReachEnvV2.compute_reward 的逻辑:
        - sigmoid="long_tail"
        - bounds=(0, 0.05)
        - margin = ||hand_init_pos - target||
        """
        tcp = self.ee_current_position
        goal = self.goal
        
        tcp_to_target = float(np.linalg.norm(tcp - goal))
        
        in_place_margin = float(np.linalg.norm(self.ee_init_position - goal))
        
        _TARGET_RADIUS = 0.05
        
        in_place = tolerance(
            tcp_to_target,
            bounds=(0, _TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid="long_tail",
        )
        
        reward = 10.0 * in_place
        
        info = {
            'reach_dist': tcp_to_target,
            'in_place': in_place,
            'reward': reward,
            'success': float(tcp_to_target <= _TARGET_RADIUS)
        }
        return reward, info
    
    def _check_success(self, obs_dict: Dict[str, Any]) -> bool:
        """检查是否到达目标"""
        reach_dist = np.linalg.norm(self.ee_current_position - self.goal)
        return reach_dist <= self.SUCCESS_THRESHOLD
    
    def _get_pos_objects(self, color: str = 'green') -> np.ndarray:
        """Reach 任务中物体位置等于目标位置"""
        return self.goal.copy()