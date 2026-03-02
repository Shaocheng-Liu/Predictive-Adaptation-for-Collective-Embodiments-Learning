# mtrl/env/real_envs/button_press_topdown_env.py
"""
KUKA Button Press Topdown 任务环境 (Real Robot Version)

任务: 从上方按下按钮
真机物理特性: 按钮表面高度约 0.22m, 最大行程 0.04m (4cm)
Agent 训练特性: 按钮表面高度约 0.31m, 最大行程 ~0.09m

适配策略:
1. Environment: 设定物理 Goal 为 "当前高度 - 0.04m"。
2. Observation: 使用虚拟按压逻辑 (Virtual Press)，让按钮位置跟随 TCP 下降，
   使 Agent 看到和仿真中一致的按钮运动模式。
3. Success: 只要 TCP 到达物理 Goal，即视为成功 (防止过度按压)。
4. External: 在 evaluate_real 中设置 observation_offset=[0, 0, 0.09] 来欺骗 Agent。

核心问题分析:
    在 MetaWorld 仿真中，按钮会跟随 TCP (手) 向下运动 ~9.5cm。
    Agent 的策略依赖于看到按钮位置随手的下压而下降。
    在真机上按钮只能下压 4cm，观测中的按钮位置几乎不变，
    这会导致 Agent 策略混乱（它期望看到按钮跟着手动）。
    
    解决方案: 在观测中让虚拟按钮 Z = min(按钮顶部, TCP Z)，
    这样 Agent 看到按钮跟随手向下运动，与仿真行为一致。
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
    hamacher_product = reward_utils.hamacher_product
except ImportError:
    print("[button_press_topdown_env] Warning: MetaWorld not available, using local fallback")
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


class KukaButtonPressTopdownEnv(KukaBaseRealEnv):
    """
    Button Press Topdown (适配 Sim-to-Real 行程不匹配问题)
    
    核心思路:
    - 真机按钮只能下压 4cm，但仿真中按钮下压 ~9.5cm
    - 在观测中，让虚拟按钮位置跟随 TCP (手) 下降，模拟仿真中按钮被按下的效果
    - 使用物理 Goal (4cm) 判断成功，防止过度按压
    """
    
    SUCCESS_THRESHOLD = 0.02
    
    # 【物理行程】真机只能按 4cm
    REAL_PRESS_DEPTH = 0.035
    
    # 【训练行程】Agent 在仿真里习惯看到按钮下降 ~9.5cm
    SIM_PRESS_DEPTH = 0.095
    
    # AprilTag 检测的就是按钮顶部，无需额外偏移
    BUTTON_TOP_OFFSET = np.array([0.0, 0.0, 0.0])
    
    def __init__(self, **kwargs):
        kwargs.setdefault('track_object_realtime', True)
        kwargs.setdefault('track_goal_realtime', False)
        super().__init__(task_name="button-press-topdown-v2", **kwargs)
        
        # 1. 物理位置初始化
        self.object_position = np.array([0.7, 0.0, 0.22])
        self.obj_init_pos = self.object_position.copy()
        
        # 2. 物理 Goal (用于判断 Success，仅 4cm)
        self.goal = self.object_position.copy()
        self.goal[2] = self.object_position[2] - self.REAL_PRESS_DEPTH
        
        self.init_tcp = self.ee_init_position.copy()
        self._button_top_offset = self.BUTTON_TOP_OFFSET.copy()
        self._gripper_auto_closed=False
        
        # 3. 计算 Agent 坐标系中的按钮顶部 Z
        self._update_agent_button_top()
        
        # 4. _obj_to_target_init: 虚拟按钮初始到物理目标的 Z 距离
        #    用作 button_pressed tolerance 的 margin
        self._obj_to_target_init = self.REAL_PRESS_DEPTH
    
    def _update_agent_button_top(self):
        """计算按钮顶部在 Agent 坐标系中的 Z 位置
        
        Z 轴在 real<->sim 坐标转换中保持不变，加上 observation_offset[2]
        """
        self._agent_button_top_z = self.obj_init_pos[2] + self.observation_offset[2]
    
    def set_button_position(self, button_pos: np.ndarray):
        """设置按钮位置，并更新物理 Goal"""
        self.object_position = np.array(button_pos, dtype=np.float64)
        self.obj_init_pos = self.object_position.copy()
        
        # 物理 Goal: 真实位置 - 0.04m
        self.goal = self.object_position.copy()
        self.goal[2] = self.object_position[2] - self.REAL_PRESS_DEPTH
        
        # 更新 Agent 坐标系中的按钮顶部 Z
        self._update_agent_button_top()
        
        self._obj_to_target_init = self.REAL_PRESS_DEPTH
    
    def set_button_from_apriltag(self, tag_id: Optional[int] = None):
        detected = self.set_object_from_apriltag(tag_id=tag_id)
        self.set_button_position(detected)
    
    def reset(self) -> np.ndarray:
        if self.use_camera_apriltag:
            self.set_button_from_apriltag(tag_id=self.object_tag_id)
        
        obs = super().reset()
        self.init_tcp = self.ee_current_position.copy()
        self._update_agent_button_top()
        return obs

    def step(self, action: np.ndarray) -> Tuple:
        """
        Push 任务的 step 覆写:
        1. 当夹爪 Z 位置 < 0.02 时，阻止继续下移（防止触发安全锁定）
        2. 当夹爪足够接近物体后，自动设定夹爪为闭合状态（减少 agent 抖动）
        """
        action = np.clip(action, -1, 1).copy()
        
        # 当夹爪接近物体时自动闭合
        tcp_to_obj = np.linalg.norm(self.ee_current_position[:2] - self.object_position[:2])
        if tcp_to_obj < 0.12:
            self._gripper_auto_closed = True
        if self._gripper_auto_closed:
            action[3] =-1.0  # MetaWorld: positive = close gripper
        else:
            print("set tool to open")
            action[3]=1.0
        
        return super().step(action)
    # =================================================================
    # ★★★ 核心: 虚拟按钮位置，让按钮跟随 TCP 下降 ★★★
    # =================================================================
    
    def _get_virtual_button_z(self, tcp_z_agent: float) -> float:
        """
        计算虚拟按钮 Z 位置 (Agent 坐标系)
        
        在仿真中，按钮会跟随 TCP 向下运动 (物理引擎驱动)。
        在真机上，我们让虚拟按钮 Z = min(按钮顶部, TCP Z)，
        这样 Agent 看到的效果和仿真一致：手向下压，按钮跟着下降。
        
        Args:
            tcp_z_agent: TCP 的 Z 坐标 (Agent 坐标系, 即仿真坐标+偏移)
        
        Returns:
            虚拟按钮的 Z 坐标 (Agent 坐标系)
        """
        return min(self._agent_button_top_z, tcp_z_agent)

    def _get_obs(self) -> np.ndarray:
        """
        覆盖父类的 _get_obs。
        修改观测中的按钮位置和目标位置，使 Agent 看到仿真一致的场景。
        
        观测结构 (39维):
        Current frame [0:18]:  Hand(3)[0:3], Grip(1)[3], Obj(3)[4:7], Quat(4)[7:11], Pad(7)[11:18]
        Previous frame [18:36]: Hand(3)[18:21], Grip(1)[21], Obj(3)[22:25], Quat(4)[25:29], Pad(7)[29:36]
        Goal [36:39]: Goal(3)
        
        关键修改:
        - obs[6]: 当前帧按钮 Z → 虚拟按钮 Z (跟随 TCP)
        - obs[24]: 上一帧按钮 Z → 虚拟按钮 Z (跟随上一步 TCP)
        - obs[38]: Goal Z → 固定为仿真目标 (按钮顶部 - SIM_PRESS_DEPTH)
        """
        obs = super()._get_obs()
        
        # --- 当前帧: 虚拟按钮 Z ---
        tcp_z_agent = obs[2]   # 当前 TCP Z (Agent 坐标系)
        obs[6] = self._get_virtual_button_z(tcp_z_agent)
        
        # --- 上一帧: 虚拟按钮 Z ---
        prev_tcp_z_agent = obs[20]  # 上一步 TCP Z (Agent 坐标系)
        obs[24] = self._get_virtual_button_z(prev_tcp_z_agent)
        
        # --- Goal Z: 固定为仿真目标 ---
        obs[38] = self._agent_button_top_z - self.SIM_PRESS_DEPTH
        
        return obs

    def get_state_for_transformer(self) -> np.ndarray:
        """
        覆盖父类的 Transformer 状态生成函数。
        使用虚拟按钮位置和仿真目标。
        """
        # 坐标转换 Real -> Sim 并加 Offset
        pos_hand_agent = self.coordinate_from_real_to_sim(self.ee_current_position) + self.observation_offset
        pos_obj_agent = self.coordinate_from_real_to_sim(self.object_position) + self.observation_offset
        
        # 虚拟按钮 Z: 跟随 TCP (与 _get_obs 一致)
        pos_obj_agent[2] = self._get_virtual_button_z(pos_hand_agent[2])
        
        # 虚拟 Goal: 固定仿真目标
        pos_goal_agent = pos_obj_agent.copy()
        pos_goal_agent[2] = self._agent_button_top_z - self.SIM_PRESS_DEPTH
        
        gripper_state = np.array([self._get_gripper_state()])
        obj_quat_sim = self._get_object_quat()
        padding = np.zeros(7)
        
        state = np.hstack([
            pos_hand_agent,   # 0-2
            gripper_state,    # 3
            pos_obj_agent,    # 4-6
            obj_quat_sim,     # 7-10
            padding,
            pos_goal_agent,   # 18-20
        ])
        
        return state

    # =================================================================
    
    def compute_reward(self, action: np.ndarray, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        计算奖励 - 使用虚拟按钮位置 (按钮跟随 TCP)
        
        与仿真对齐的 Reward 逻辑:
        1. 虚拟按钮 Z = min(按钮顶部, TCP Z) → 按钮跟随 TCP 下降
        2. near_button: TCP 接近虚拟按钮的奖励
        3. button_pressed: 虚拟按钮到达物理目标的奖励
        4. 使用 REAL_PRESS_DEPTH 作为 button_pressed 的 margin
        """
        tcp = self.ee_current_position
        obj_top_init_z = self.obj_init_pos[2]
        
        # 虚拟按钮 Z: 跟随 TCP (真实坐标系)
        # 与仿真一致: 按钮不会高于初始位置，且跟随 TCP 下降
        virtual_obj_z = min(obj_top_init_z, tcp[2])
        
        virtual_obj = self.obj_init_pos.copy()
        virtual_obj[2] = virtual_obj_z
        
        # 使用物理目标计算 obj_to_target
        target = self.goal
        print("object position:", virtual_obj)
        print("tcp position: ", tcp)
        tcp_to_obj = float(np.linalg.norm(virtual_obj - tcp))
        tcp_to_obj_init = float(np.linalg.norm(virtual_obj - self.init_tcp))
        obj_to_target = abs(target[2] - virtual_obj_z)
        
        gripper_state = self._get_gripper_state()
        tcp_closed = 1.0 - gripper_state
        
        near_button = tolerance(
            tcp_to_obj, bounds=(0, 0.01),
            margin=tcp_to_obj_init, sigmoid="long_tail"
        )
        button_pressed = tolerance(
            obj_to_target, bounds=(0, 0.005),
            margin=self._obj_to_target_init, sigmoid="long_tail"
        )
        
        reward = 5 * hamacher_product(tcp_closed, near_button)
        print("hamacher_product:",reward)
        if tcp_to_obj <= 0.03:
            print("start pressing")
            reward += 5 * button_pressed
        
        success = self._check_success(obs_dict)
        
        info = {
            'tcp_to_obj': tcp_to_obj,
            'obj_to_target': obj_to_target,
            'near_button': near_button,
            'button_pressed': button_pressed,
            'reward': reward,
            'success': float(success)
        }
        return reward, info
    
    def _check_success(self, obs_dict: Dict[str, Any]) -> bool:
        """
        判定成功: 使用【物理 Goal】(4cm)
        当 TCP 到达物理 Goal 位置即成功
        """
        tcp_pos = self.ee_current_position
        goal_z = self.goal[2]
        
        # 1. 高度达标
        z_success = tcp_pos[2] <= (goal_z + self.SUCCESS_THRESHOLD)
        
        # 2. 水平距离达标 (例如必须在按钮中心 2cm 范围内)
        # 按钮位置在 self.object_position
        xy_dist = np.linalg.norm(tcp_pos[:2] - self.object_position[:2])
        xy_success = xy_dist < 0.02 

        return z_success and xy_success

    def _get_pos_objects(self, color: str = 'green') -> np.ndarray:
        if self.camera is not None:
            try:
                pos = self.camera.getPosition_inBaseFrame(color=color)
                self.object_position = pos.copy()
                return pos
            except Exception:
                pass
        return self.object_position.copy()

    def _get_object_quat(self) -> np.ndarray:
        return np.array([0.7074, -0.7068, 0.0, 0.0])