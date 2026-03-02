# mtrl/env/real_envs/push_env.py
"""
KUKA Push 任务环境

任务: 推动物体到目标位置
成功条件: 物体与目标距离 < 0.05m

基于 MetaWorld sawyer_push_v2.py 的 Reward 逻辑，
完全匹配 MetaWorld 的奖励计算。
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
    print("[push_env] Warning: MetaWorld not available, using local fallback")
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
        denominator = a + b - (a * b)
        if denominator > 0:
            return (a * b) / denominator
        return 0.0


class KukaPushEnv(KukaBaseRealEnv):
    """
    Push 任务环境 - 完全匹配 MetaWorld sawyer_push_v2 的奖励计算
    
    目标: 推动物体到指定目标位置
    
    使用方法:
        env = KukaPushEnv()
        env.set_goal(np.array([0.7, 0.1, 0.02]))        # 设置目标位置
        env.set_object_position(np.array([0.6, 0.0, 0.02]))  # 设置物体初始位置
        
        # 或使用相机（例如用红色标记目标，绿色标记物体）:
        env.set_goal_from_camera('red')
        env.set_object_from_camera('green')
    """
    
    SUCCESS_THRESHOLD = 0.05  # 5cm
    TARGET_RADIUS = 0.05
    
    def __init__(self, **kwargs):
        # Push 任务默认配置：需要实时追踪物体
        kwargs.setdefault('track_object_realtime', True)
        kwargs.setdefault('track_goal_realtime', False)
        kwargs.setdefault('object_color', 'red')
        kwargs.setdefault('goal_color', 'green')
        super().__init__(task_name="push-v2", **kwargs)
        
        # 默认目标位置（真实坐标系）
        # MetaWorld 默认: goal_low=(-0.1, 0.8, 0.01), goal_high=(0.1, 0.9, 0.02)
        self.goal = np.array([0.7, 0.1, 0.02])
        
        # 默认物体位置（真实坐标系）
        # MetaWorld 默认: obj_low=(-0.1, 0.6, 0.02), obj_high=(0.1, 0.7, 0.02)
        self.object_position = np.array([0.6, 0.0, 0.02])
        self.obj_init_pos = self.object_position.copy()
        
        # 初始化 TCP 位置（用于 gripper caging reward）
        self.init_tcp = self.ee_init_position.copy()
        
        # 夹爪 pad 位置估计
        # 真机上没有 left_pad/right_pad 传感器，用估计值
        # 假设 pad 在 TCP 两侧，Y 方向偏移约 0.02m
        self.pad_y_offset = 0.02
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        obs = super().reset()
        # 保存初始 TCP 位置
        self.init_tcp = self.ee_current_position.copy()
        self._gripper_auto_closed = False
        return obs
    
    def step(self, action: np.ndarray) -> Tuple:
        """
        Push 任务的 step 覆写:
        1. 当夹爪 Z 位置 < 0.02 时，阻止继续下移（防止触发安全锁定）
        2. 当夹爪足够接近物体后，自动设定夹爪为闭合状态（减少 agent 抖动）
        """
        action = np.clip(action, -1, 1).copy()
        
        # 安全限制: 夹爪 Z 位置低于 0.02 时不让继续向下
        # Z 轴在真实坐标系和仿真坐标系中一致（旋转矩阵不改变 Z）
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
    
    def compute_reward(self, action: np.ndarray, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        计算奖励 - 完全匹配 MetaWorld V2 Push
        
        参考: metaworld/envs/mujoco/sawyer_xyz/v2/sawyer_push_v2.py
        
        奖励组成:
        1. object_grasped: 夹爪 caging 奖励 (high_density=True)
        2. in_place: 物体接近目标位置的奖励
        3. 成功奖励: 物体到达目标位置时 reward=10
        """
        # 获取位置（真实坐标系）
        tcp = self.ee_current_position
        obj = self.object_position
        target = self.goal
        
        # 从动作获取夹爪开合状态
        tcp_opened = 1.0
        if action is not None and len(action) >= 4:
            tcp_opened = max(0, action[3])
        
        # 计算距离
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        target_to_obj = float(np.linalg.norm(obj - target))
        target_to_obj_init = float(np.linalg.norm(self.obj_init_pos - target))
        
        # In-place reward: 物体接近目标
        in_place = tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid="long_tail",
        )
        
        # Gripper caging reward: 完全匹配 MetaWorld
        # 使用 high_density=True 模式
        object_grasped = self._gripper_caging_reward(
            action=action,
            obj_pos=obj,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_thresh=0.05,
            xz_thresh=0.005,
            high_density=True,
        )
        
        # 组合奖励 - 精确匹配 MetaWorld
        reward = 2 * object_grasped
        
        if tcp_to_obj < 0.02 and tcp_opened > 0:
            reward += 1.0 + reward + 5.0 * in_place
        
        if target_to_obj < self.TARGET_RADIUS:
            reward = 10.0
        
        # 构造 info - 匹配 MetaWorld evaluate_state 返回
        info = {
            'tcp_to_obj': tcp_to_obj,
            'tcp_opened': tcp_opened,
            'target_to_obj': target_to_obj,
            'object_grasped': object_grasped,
            'in_place': in_place,
            'near_object': float(tcp_to_obj <= 0.03),
            'grasp_success': float(tcp_to_obj < 0.02),
            'grasp_reward': object_grasped,
            'in_place_reward': in_place,
            'obj_to_target': target_to_obj,
            'unscaled_reward': reward,
            'success': float(target_to_obj <= self.TARGET_RADIUS)
        }
        
        return reward, info
    
    def _gripper_caging_reward(
        self, 
        action: np.ndarray, 
        obj_pos: np.ndarray,
        object_reach_radius: float = 0.01,
        obj_radius: float = 0.015,
        pad_success_thresh: float = 0.05,
        xz_thresh: float = 0.005,
        desired_gripper_effort: float = 1.0,
        high_density: bool = False,
        medium_density: bool = False,
    ) -> float:
        """
        计算夹爪 caging 奖励 - 完全匹配 MetaWorld 实现
        
        参考: metaworld/envs/mujoco/sawyer_xyz/sawyer_xyz_env.py _gripper_caging_reward
        
        由于真机没有 left_pad/right_pad 传感器，使用 TCP 位置估计 pad 位置
        """
        tcp = self.ee_current_position
        
        # ===== Left-right gripper (Y axis) caging reward =====
        # 估计 left_pad 和 right_pad 的 Y 坐标
        # 假设 pad 在 TCP 两侧对称
        left_pad_y = tcp[1] + self.pad_y_offset
        right_pad_y = tcp[1] - self.pad_y_offset
        pad_y_lr = np.array([left_pad_y, right_pad_y])
        
        # 当前 pad 到物体的 Y 距离
        pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
        # 当前 pad 到初始物体位置的 Y 距离
        pad_to_objinit_lr = np.abs(pad_y_lr - self.obj_init_pos[1])
        
        # 计算 caging margin
        caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
        
        # 左右 pad 的 caging reward
        caging_lr = [
            tolerance(
                pad_to_obj_lr[i],
                bounds=(obj_radius, pad_success_thresh),
                margin=caging_lr_margin[i],
                sigmoid="long_tail",
            )
            for i in range(2)
        ]
        caging_y = hamacher_product(caging_lr[0], caging_lr[1])
        
        # ===== X-Z gripper caging reward =====
        xz = [0, 2]  # X and Z indices
        MIN_MARGIN = 0.001  # Minimum margin to prevent zero division
        
        caging_xz_margin = np.linalg.norm(self.obj_init_pos[xz] - self.init_tcp[xz])
        caging_xz_margin -= xz_thresh
        caging_xz_margin = max(caging_xz_margin, MIN_MARGIN)  # Prevent zero/negative margin
        
        caging_xz = tolerance(
            np.linalg.norm(tcp[xz] - obj_pos[xz]),
            bounds=(0, xz_thresh),
            margin=caging_xz_margin,
            sigmoid="long_tail",
        )
        
        # ===== Closed-extent gripper reward =====
        gripper_closed = 0.0
        if action is not None and len(action) > 0:
            gripper_closed = min(max(0, action[-1]), desired_gripper_effort) / desired_gripper_effort
        
        # ===== Combine components =====
        caging = hamacher_product(caging_y, float(caging_xz))
        gripping = gripper_closed if caging > 0.97 else 0.0
        caging_and_gripping = hamacher_product(caging, gripping)
        
        if high_density:
            caging_and_gripping = (caging_and_gripping + caging) / 2
        
        if medium_density:
            tcp_to_obj = np.linalg.norm(obj_pos - tcp)
            tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.init_tcp)
            reach_margin = abs(tcp_to_obj_init - object_reach_radius)
            reach_margin = max(reach_margin, MIN_MARGIN)
            reach = tolerance(
                tcp_to_obj,
                bounds=(0, object_reach_radius),
                margin=reach_margin,
                sigmoid="long_tail",
            )
            caging_and_gripping = (caging_and_gripping + float(reach)) / 2
        
        return caging_and_gripping
    
    def _check_success(self, obs_dict: Dict[str, Any]) -> bool:
        """检查物体是否到达目标"""
        target_to_obj = np.linalg.norm(self.object_position - self.goal)
        return target_to_obj <= self.SUCCESS_THRESHOLD
    
    def _get_pos_objects(self, color: str = 'green') -> np.ndarray:
        """
        获取物体位置
        
        如果相机可用，使用相机检测；否则返回当前存储的位置。
        
        Args:
            color: 物体颜色 ('green', 'red', 'blue', etc.)
        
        Returns:
            物体位置 [3]
        """
        if self.camera is not None:
            try:
                return self.camera.getPosition_inBaseFrame(color=color)
            except Exception:
                pass
        return self.object_position.copy()