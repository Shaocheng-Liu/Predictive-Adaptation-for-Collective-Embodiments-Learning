# mtrl/env/real_envs/pick_place_env.py
"""
KUKA Pick-Place 任务环境

任务: 抓取物体并放置到目标位置
成功条件: 物体与目标距离 < 0.07m

基于 MetaWorld sawyer_pick_place_v2.py 的 Reward 逻辑，
适配 real_envs/base_env.py 的真机接口风格。
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
    hamacher_product = reward_utils.hamacher_product
except ImportError:
    print("[pick_place_env] Warning: MetaWorld not available, using local fallback")
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

    def hamacher_product(a: float, b: float) -> float:
        """Hamacher product for reward composition."""
        a = max(0, min(1, a))
        b = max(0, min(1, b))
        denominator = a + b - a * b
        return (a * b / denominator) if denominator > 0 else 0


class KukaPickPlaceEnv(KukaBaseRealEnv):
    """
    Pick-Place 任务环境
    
    目标: 抓取物体并放置到指定目标位置
    
    使用方法:
        env = KukaPickPlaceEnv()
        env.set_goal(np.array([0.7, 0.1, 0.2]))         # 设置目标位置（空中）
        env.set_object_position(np.array([0.6, 0.0, 0.02]))  # 设置物体初始位置
        # 或使用相机
        env.set_goal_from_camera('red')
        env.set_object_from_camera('green')
    """
    
    SUCCESS_THRESHOLD = 0.1  # 7cm - MetaWorld uses 0.07 for pick-place
    TARGET_RADIUS = 0.05
    LIFT_THRESHOLD = 0.01  # Minimum lift height to be considered "lifted"
    
    def __init__(self, **kwargs):
        # Pick-Place 任务默认配置：需要实时追踪物体
        kwargs.setdefault('track_object_realtime', True)
        kwargs.setdefault('track_goal_realtime', True)
        kwargs.setdefault('object_color', 'red')
        kwargs.setdefault('goal_color', 'green')
        super().__init__(task_name="pick-place-v2", **kwargs)
        
        # 默认目标位置（真实坐标系）
        # MetaWorld 默认: goal_low=(-0.1, 0.8, 0.05), goal_high=(0.1, 0.9, 0.3)
        self.goal = np.array([0.7, 0.1, 0.2])  # 注意 Z 轴较高，需要抬起物体
        
        # 默认物体位置（真实坐标系）
        # MetaWorld 默认: obj_low=(-0.1, 0.6, 0.02), obj_high=(0.1, 0.7, 0.02)
        self.object_position = np.array([0.6, 0.0, 0.02])
        self.obj_init_pos = self.object_position.copy()
        
        # 初始化 TCP 和 pad 位置（用于 gripper caging reward）
        self.init_tcp = self.ee_init_position.copy()
        self.init_left_pad = self.ee_init_position.copy()
        self.init_right_pad = self.ee_init_position.copy()
        
        # 当前夹爪状态
        self.gripper_state = 1.0  # 1.0 = 开, -1.0 = 闭
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        obs = super().reset()
        # 保存初始 TCP 位置
        self.init_tcp = self.ee_current_position.copy()
        self.gripper_state = 1.0
        self._gripper_auto_closed = False
        return obs
    
    def step(self, action: np.ndarray) -> Tuple:
        """
        Pick-Place 任务的 step 覆写:
        当夹爪足够接近物体后，自动设定夹爪为闭合状态（减少 agent 抖动）
        """
        action = np.clip(action, -1, 1).copy()
        
        if self.ee_current_position[2] <= 0.03:
            action[2] = max(action[2], 0)  # 只允许向上或不动
        # 当夹爪接近物体时自动闭合
        tcp_to_obj = np.linalg.norm(self.ee_current_position - self.object_position)
        if tcp_to_obj < 0.02:
            self._gripper_auto_closed = True
        if self._gripper_auto_closed:
            action[3] =-1.0  # MetaWorld: positive = close gripper
        else:
            print("set tool to open")
            action[3]=1.0
        
        return super().step(action)
    
    def set_gripper_action(self, gripper_state: float):
        """设置夹爪动作并跟踪状态"""
        super().set_gripper_action(gripper_state)
        self.gripper_state = gripper_state
    
    def compute_reward(self, action: np.ndarray, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        计算奖励 - 对齐 MetaWorld V2 Pick-Place
        
        奖励组成:
        1. object_grasped: 夹爪 caging 和抓取奖励
        2. in_place: 物体接近目标位置的奖励
        3. 抬起物体的额外奖励
        4. 成功奖励
        """
        # 获取位置（真实坐标系）
        tcp = self.ee_current_position
        obj = self.object_position
        target = self.goal
        
        # 计算距离
        obj_to_target = float(np.linalg.norm(obj - target))
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        in_place_margin = float(np.linalg.norm(self.obj_init_pos - target))
        
        # 夹爪开闭状态（从动作或跟踪状态获取）
        tcp_opened = max(0, action[3]) if action is not None else self.gripper_state
        
        # In-place reward: 物体接近目标
        in_place = tolerance(
            obj_to_target,
            bounds=(0, self.TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid="long_tail",
        )
        
        # Gripper caging reward
        object_grasped = self._gripper_caging_reward(action, obj, tcp)
        
        # 组合 in_place 和 object_grasped
        in_place_and_object_grasped = hamacher_product(object_grasped, in_place)
        reward = in_place_and_object_grasped
        
        # 额外奖励：抬起物体
        lifted = obj[2] - self.LIFT_THRESHOLD > self.obj_init_pos[2]
        if tcp_to_obj < 0.02 and tcp_opened > 0 and lifted:
            reward += 1.0 + 5.0 * in_place
        
        # 成功奖励
        if obj_to_target < self.TARGET_RADIUS:
            reward = 10.0
        
        # 构造 info
        info = {
            'obj_to_target': obj_to_target,
            'tcp_to_obj': tcp_to_obj,
            'in_place': in_place,
            'object_grasped': object_grasped,
            'lifted': lifted,
            'reward': reward,
            'success': float(obj_to_target <= self.SUCCESS_THRESHOLD)
        }
        
        return reward, info
    
    def _gripper_caging_reward(
        self, 
        action: np.ndarray, 
        obj_pos: np.ndarray, 
        tcp_pos: np.ndarray
    ) -> float:
        """
        计算夹爪 caging 奖励
        
        基于 MetaWorld 的 _gripper_caging_reward 简化实现
        """
        pad_success_margin = 0.05
        x_z_success_margin = 0.005
        obj_radius = 0.015
        
        # TCP 到物体的 XZ 平面距离
        tcp_xz = np.array([tcp_pos[0], 0.0, tcp_pos[2]])
        obj_xz = np.array([obj_pos[0], 0.0, obj_pos[2]])
        tcp_obj_norm_x_z = float(np.linalg.norm(tcp_xz - obj_xz))
        
        # 初始距离
        init_tcp_xz = np.array([self.init_tcp[0], 0.0, self.init_tcp[2]])
        init_obj_xz = np.array([self.obj_init_pos[0], 0.0, self.obj_init_pos[2]])
        tcp_obj_x_z_margin = float(np.linalg.norm(init_obj_xz - init_tcp_xz)) - x_z_success_margin
        tcp_obj_x_z_margin = max(tcp_obj_x_z_margin, 0.01)
        
        # XZ 平面接近奖励
        x_z_caging = tolerance(
            tcp_obj_norm_x_z,
            bounds=(0, x_z_success_margin),
            margin=tcp_obj_x_z_margin,
            sigmoid="long_tail",
        )
        
        # Y 方向接近奖励（简化版本）
        y_diff = abs(tcp_pos[1] - obj_pos[1])
        y_caging = tolerance(
            y_diff,
            bounds=(0, obj_radius),
            margin=pad_success_margin,
            sigmoid="long_tail",
        )
        
        # 组合 caging
        caging = hamacher_product(y_caging, x_z_caging)
        
        # 抓取奖励：当 caging 足够好且夹爪闭合时
        # MetaWorld: action[-1] positive = close gripper, so gripper_closed = clip(action[-1], 0, 1)
        gripper_closed = min(max(0, action[3] if action is not None else 0), 1)
        gripping = gripper_closed if caging > 0.97 else 0.0
        
        # 组合 caging 和 gripping
        caging_and_gripping = hamacher_product(caging, gripping)
        result = (caging_and_gripping + caging) / 2
        
        return result
    
    def _check_success(self, obs_dict: Dict[str, Any]) -> bool:
        """检查物体是否到达目标"""
        obj_to_target = np.linalg.norm(self.object_position - self.goal)
        return obj_to_target <= self.SUCCESS_THRESHOLD
    
    def _get_pos_objects(self, color: str = 'green') -> np.ndarray:
        """获取物体位置"""
        if self.camera is not None:
            try:
                return self.camera.getPosition_inBaseFrame(color=color)
            except Exception:
                pass
        return self.object_position.copy()