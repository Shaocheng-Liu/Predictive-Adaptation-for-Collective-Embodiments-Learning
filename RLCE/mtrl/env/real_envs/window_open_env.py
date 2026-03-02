# mtrl/env/real_envs/window_open_env.py
"""
KUKA Window Open 任务环境

任务: 打开滑动窗户
成功条件: 窗户把手 X 位置与目标距离 < 0.05m

基于 MetaWorld sawyer_window_open_v2.py 的 Reward 逻辑，
适配 real_envs/base_env.py 的真机接口风格。

Apriltag 遮挡策略 (2024):
    由于推窗过程中机械臂会遮挡 Apriltag，采用以下策略:
    1. reset 时读取一次 Apriltag 作为把手初始位置（object_position），之后不再实时读取
    2. 分两阶段计算 reward:
       - Phase 1 (接近阶段): TCP 还没接触把手，object_position 保持初始值不变
       - Phase 2 (推窗阶段): TCP 足够靠近把手后，根据 TCP 移动量估算窗户位置
         （因为此时 TCP 正在推窗，TCP 的位移 ≈ 窗户的位移）
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
    print("[window_open_env] Warning: MetaWorld not available, using local fallback")
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


class KukaWindowOpenEnv(KukaBaseRealEnv):
    """
    Window Open 任务环境
    
    目标: 打开滑动窗户
    
    MetaWorld 中 state[4:7] 的含义:
        _get_pos_objects() = _get_site_pos("handleOpenStart")
        = body("window").pos + [-0.04, -0.095, 0]  (sim 坐标, qpos=0)
        
        handle_position 在此任务中 ≈ body position (qpos=0 时窗户关闭)
    
    偏移说明:
        apriltag_offset: Apriltag → 把手位置（由用户配置，保持不变）
        OBJ_POS_OFFSET:  把手位置 → state[4:7]（MetaWorld _get_pos_objects 等价位置）
        GOAL_POS_OFFSET: 把手位置 → goal position（MetaWorld _target_pos 等价位置）
    
    Apriltag 遮挡处理:
        - track_object_realtime 默认设为 False，不再每步读取 Apriltag
        - reset() 时读取一次 Apriltag 确定把手初始位置
        - step() 中根据 TCP 是否靠近把手来切换 reward 计算模式:
          Phase 1: TCP 远离把手 → 使用固定的初始 object_position
          Phase 2: TCP 靠近把手 → 根据 TCP 位移估算窗户位置
    
    使用方法:
        env = KukaWindowOpenEnv()
        env.set_handle_position(np.arry([0.7, 0.0, 0.2]))
        # 或使用 Apriltag (只在 reset 时读取一次)
        env.set_handle_from_apriltag()
    """
    
    SUCCESS_THRESHOLD = 0.05  # 5cm
    TARGET_RADIUS = 0.05
    
    # TCP 距离把手小于此值时，认为已接触把手，进入推窗阶段
    CONTACT_THRESHOLD = 0.03  # 4cm
    
    # MetaWorld: _get_pos_objects = _get_site_pos("handleOpenStart")
    #            = body("window").pos + [-0.04, -0.095, 0]  (sim coords)
    #            handle ≈ body (window-open, qpos=0)
    # 转换到真实坐标系: R_real_sim * [-0.04, -0.095, 0] = [-0.095, 0.04, 0]
    OBJ_POS_OFFSET_SIM = np.array([0.0, 0.0, 0.0])
    OBJ_POS_OFFSET_REAL = np.array([0.0, 0.0, 0.0])
    
    # MetaWorld: goal = body + [0.2, 0, 0] = handle + [0.2, 0, 0]  (sim coords)
    # 转换到真实坐标系: R_real_sim * [0.2, 0, 0] = [0, -0.2, 0]
    GOAL_POS_OFFSET_SIM = np.array([0.2, 0.095, 0.0])
    GOAL_POS_OFFSET_REAL = np.array([0.095, -0.2, 0.0])
    
    def __init__(self, **kwargs):
        # ============================================================
        # 关键修改: 禁用实时 Apriltag 追踪
        # 原因: 推窗过程中机械臂会遮挡 Apriltag，导致读到随机噪声值
        # 策略: 仅在 reset 时读取一次初始位置，之后不再更新
        # ============================================================
        kwargs.setdefault('track_object_realtime', False)  # 改为 False
        kwargs.setdefault('track_goal_realtime', False)
        kwargs.setdefault('object_color', 'red')
        kwargs.setdefault('goal_color', 'green')
        super().__init__(task_name="window-open-v2", **kwargs)
        
        # 最大滑动距离
        self.maxPullDist = 0.2
        
        # 默认把手位置（真实坐标系，关闭状态）
        self._handle_position = np.array([0.7, 0.0, 0.2])
        
        # 应用偏移: handle → state[4:7] 等价位置
        self.object_position = self._handle_position + self.OBJ_POS_OFFSET_REAL
        self.obj_init_pos = self.object_position.copy()
        
        # 基于把手位置自动计算目标位置
        self.goal = self._handle_position + self.GOAL_POS_OFFSET_REAL
        
        # 窗户把手初始位置（使用偏移后的位置，匹配 MetaWorld）
        self.window_handle_pos_init = self.object_position.copy()
        
        # 初始化 TCP 位置
        self.init_tcp = self.ee_init_position.copy()
        
        # 目标奖励值
        self.target_reward = 1000 * self.maxPullDist + 1000 * 2
        
        # ============================================================
        # 推窗阶段追踪状态
        # ============================================================
        # 是否已进入推窗阶段 (TCP 足够靠近把手)
        self._contact_made = False
        # TCP 接触把手时的位置 (用于计算位移)
        self._tcp_at_contact = None
    
    def set_handle_position(self, handle_pos: np.ndarray):
        """
        设置把手位置，自动计算 state[4:7] 和目标位置
        
        偏移链: handle_pos → (+OBJ_POS_OFFSET_REAL) → object_position (state[4:7])
                handle_pos → (+GOAL_POS_OFFSET_REAL) → goal
        
        Args:
            handle_pos: 把手位置 [x, y, z]（真实坐标系）
        """
        self._handle_position = np.array(handle_pos, dtype=np.float64)
        self.object_position = self._handle_position + self.OBJ_POS_OFFSET_REAL
        self.obj_init_pos = self.object_position.copy()
        self.window_handle_pos_init = self.object_position.copy()
        self.goal = self._handle_position + self.GOAL_POS_OFFSET_REAL
    
    def set_handle_from_apriltag(self, tag_id: Optional[int] = None):
        """
        使用 Apriltag 检测把手位置, 自动计算 goal 和 object_position
        
        注意: 此方法仅应在 reset 时或任务开始前调用一次。
        之后不再实时读取 Apriltag（因为推窗时会被遮挡）。
        
        偏移链: Apriltag → (+apriltag_offset) → (+camera_offset) → handle → (+task offset) → state
        
        Args:
            tag_id: Apriltag ID (default: self.object_tag_id)
        """
        detected = super().set_object_from_apriltag(tag_id=tag_id)
        self.set_handle_position(detected)
    
    def set_object_from_apriltag(self, tag_id: Optional[int] = None):
        """Override: 实时追踪时，检测把手位置并应用 task offset 到 object_position
        
        注意: 由于 track_object_realtime=False，此方法在 step 中不会被自动调用。
        仅在手动调用或 reset 中使用。
        """
        handle_pos = super().set_object_from_apriltag(tag_id=tag_id)
        self._handle_position = handle_pos.copy()
        self.object_position = handle_pos + self.OBJ_POS_OFFSET_REAL
        return handle_pos
    
    def set_object_from_camera(self, color: str = 'red'):
        """Override: RGB 追踪时，检测把手位置并应用 task offset"""
        handle_pos = super().set_object_from_camera(color=color)
        self._handle_position = handle_pos.copy()
        self.object_position = handle_pos + self.OBJ_POS_OFFSET_REAL
        return handle_pos
    
    def reset(self) -> np.ndarray:
        """重置环境
        
        reset 时通过 base_env 读取一次 Apriltag（如果 use_camera_apriltag=True），
        确定把手初始位置。之后 step 中不再读取。
        """
        # ============================================================
        # 重置推窗阶段状态
        # ============================================================
        self._contact_made = False
        self._tcp_at_contact = None
        
        obs = super().reset()
        # 保存初始 TCP 位置
        self.init_tcp = self.ee_current_position.copy()
        # 保存窗户把手初始位置（reset 中已通过 base_env 读取了 Apriltag）
        self.window_handle_pos_init = self.object_position.copy()
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行动作
        
        在 base_env.step() 基础上，添加基于 TCP 位移的窗户位置估算逻辑。
        
        推窗位置估算策略:
            - Phase 1 (接近): TCP 距离把手 > CONTACT_THRESHOLD
              → object_position 保持初始值不变
            - Phase 2 (推窗): TCP 距离把手 <= CONTACT_THRESHOLD
              → object_position = 初始位置 + TCP位移量（从接触点开始算）
              → 这样 reward 中的 "in_place" 就能反映窗户是否被推到目标位置
        """
        # 调用父类 step（执行动作、获取观测、计算reward等）
        # 注意: 由于 track_object_realtime=False，父类不会读取 Apriltag
        
        
        # ============================================================
        # 基于 TCP 位移估算窗户位置
        # ============================================================
        tcp = self.ee_current_position.copy()
        tcp_to_obj = float(np.linalg.norm(self.obj_init_pos[0] - tcp[0]))
        print("tcp pos:", tcp)
        print("obj init pos:", self.obj_init_pos)
        
        if not self._contact_made:
            # Phase 1: 检测是否已接触把手
            if tcp_to_obj <= self.CONTACT_THRESHOLD:
                self._contact_made = True
                self._tcp_at_contact = tcp.copy()
                print(f"[KukaWindowOpenEnv] Contact made! TCP={tcp}, obj_init={self.obj_init_pos}")
        
        if self._contact_made and self._tcp_at_contact is not None:
            # Phase 2: 根据 TCP 位移估算窗户当前位置
            tcp_displacement = tcp - self._tcp_at_contact
            self.object_position = self.obj_init_pos + tcp_displacement
            action[1] = 0+np.random.uniform(-0.05,0.05)
            action[0] = 0.7+np.random.uniform(-0.2,0.2)

        obs, reward, done, info = super().step(action)
        
        return obs, reward, done, info
    

    def compute_reward(self, action: np.ndarray, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        计算奖励 - 对齐 MetaWorld V2 Window Open
        
        奖励组成:
        1. reach: 夹爪接近把手的奖励
        2. in_place: 把手接近目标（打开）位置的奖励
        
        Apriltag 遮挡适配:
        - Phase 1 (接近): object_position 是固定的初始值，in_place ≈ 0，
          reward 主要由 reach 驱动（引导 TCP 靠近把手）
        - Phase 2 (推窗): object_position 基于 TCP 位移估算，
          in_place 开始增长（反映窗户是否被推向目标）
        
        特点：Window Open 主要关注 X 轴方向的移动（与 Close 方向相反）
        """
        # 获取位置（真实坐标系）
        tcp = self.ee_current_position
        obj = self.object_position  # 把手当前位置（固定值或 TCP 估算值）
        target = self.goal.copy()   # 把手目标位置（打开后）
        
        # 计算把手到目标的距离（主要看 X 轴）
        target_to_obj = float(abs(obj[1] - target[1]))
        
        # 初始距离
        target_to_obj_init = float(abs(self.obj_init_pos[1] - target[1]))
        
        # In-place reward: 把手接近目标位置
        in_place = tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid="long_tail",
        )
        
        # Reach reward: 夹爪接近把手（使用初始位置，因为把手没推之前不会动）
        handle_radius = 0.02
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        tcp_to_obj_init = float(np.linalg.norm(self.window_handle_pos_init - self.init_tcp))
        
        reach = tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid="long_tail",
        )
        
        # 组合奖励
        object_grasped = reach
        reward = 10 * hamacher_product(reach, in_place)
        
        # 构造 info
        info = {
            'target_to_obj': target_to_obj,
            'tcp_to_obj': tcp_to_obj,
            'reach': reach,
            'in_place': in_place,
            'grasp_reward': object_grasped,
            'reward': reward,
            'success': float(target_to_obj <= self.TARGET_RADIUS),
            'contact_made': self._contact_made,  # 调试信息: 是否已接触把手
        }
        
        return reward, info
    
    def _check_success(self, obs_dict: Dict[str, Any]) -> bool:
        """检查窗户是否打开
        
        Phase 2 阶段通过 TCP 位移估算的 object_position 来判断
        """
        target_to_obj = abs(self.object_position[1] - self.goal[1])
        return target_to_obj <= self.SUCCESS_THRESHOLD
    
    def _get_pos_objects(self, color: str = 'green') -> np.ndarray:
        """获取窗户把手位置
        
        不再从相机读取（会被遮挡），直接返回当前估算的 object_position
        """
        return self.object_position.copy()
    
    def _get_object_quat(self) -> np.ndarray:
        """Window Open 四元数 - 与 MetaWorld 一致返回 zeros(4)"""
        return np.zeros(4)