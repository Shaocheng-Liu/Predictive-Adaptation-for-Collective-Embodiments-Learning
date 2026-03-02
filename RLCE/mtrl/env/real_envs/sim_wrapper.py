# mtrl/env/real_envs/sim_wrapper.py
"""
Simulation Wrapper for Real Environments

This module provides a mock interface that allows real_envs classes to run
in MuJoCo simulation instead of on physical hardware. The wrapper "tricks"
the real environment into thinking it's controlling real hardware, but actually
routes all move commands to MetaWorld's MuJoCo engine and gets all state data
from MuJoCo's sim.data.

Key Features:
- Overrides hardware-dependent functions (ROS, camera drivers)
- Uses MetaWorld simulation backend for physics
- Preserves the reward function and task logic from real_envs
- Supports hardcoded goal_position and object_position
- Enables real-time MuJoCo rendering

Usage:
    from mtrl.env.real_envs.sim_wrapper import SimulatedRealEnv
    
    # Create simulated environment with hardcoded positions
    env = SimulatedRealEnv(
        task_name="reach-v2",
        goal_position=np.array([0.0, 0.8, 0.15]),
        object_position=np.array([0.0, 0.6, 0.02]),
        render_mode="human"  # Enable MuJoCo rendering
    )
    
    obs = env.reset()
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from gymnasium import spaces

# Import MetaWorld environments
import sys
import os

# Add MetaWorld to path
metaworld_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'Metaworld')
if os.path.exists(metaworld_path) and metaworld_path not in sys.path:
    sys.path.insert(0, metaworld_path)

try:
    from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_reach_v2 import SawyerReachEnvV2
    from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_push_v2 import SawyerPushEnvV2
    from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import SawyerPickPlaceEnvV2
    from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_drawer_open_v2 import SawyerDrawerOpenEnvV2
    from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_window_close_v2 import SawyerWindowCloseEnvV2
    from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_window_open_v2 import SawyerWindowOpenEnvV2
    from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_button_press_topdown_v2 import SawyerButtonPressTopdownEnvV2
    from metaworld.envs.mujoco.utils import reward_utils
    METAWORLD_AVAILABLE = True
except ImportError as e:
    print(f"[SimWrapper] Warning: MetaWorld not available: {e}")
    METAWORLD_AVAILABLE = False


# ========================= HARDCODED CONFIGURATION =========================
# Modify these values to set custom positions for simulation.
# All coordinates are in the MetaWorld/simulation coordinate system.
#
# For push-v2 / pick-place-v2:
#   User sets object_position (ball) and goal_position (target) separately.
#
# For drawer-open-v2 / window-open-v2 / window-close-v2:
#   User only needs to set the HANDLE position.
#   The code auto-computes body position and goal position from the handle.
#
# Handle-position ranges (MetaWorld defaults):
#   drawer-open-v2:  handle X ∈ [-0.1, 0.1], handle Y ≈ body_Y - 0.16 (body_Y = 0.9)
#                    handle Z ≈ 0.0
#                    Default handle: [0.0, 0.74, 0.0]   (body=[0,0.9,0], handle=body+[0,-0.16,0])
#   window-close-v2: handle X ≈ body_X + qpos (initial qpos=0.2, so handle_X = body_X + 0.2)
#                    handle Y ∈ [0.75, 0.9], handle Z ≈ 0.16
#                    Default handle: [0.2, 0.785, 0.16]  (body=[0,0.785,0.16], qpos=0.2)
#   window-open-v2:  handle X ≈ body_X + qpos (initial qpos=0, so handle_X ≈ body_X)
#                    handle Y ∈ [0.7, 0.9], handle Z ≈ 0.16
#                    Default handle: [-0.1, 0.785, 0.16] (body=[-0.1,0.785,0.16], qpos=0)

# Default goal positions for reach/push/pick-place
DEFAULT_GOAL_POSITIONS = {
    "reach-v2": np.array([0.0, 0.75, 0.15]),
    "push-v2": np.array([0.0, 0.75, 0.02]),
    "pick-place-v2": np.array([0.05, 0.75, 0.1]),
}

# Default object positions for reach/push/pick-place
DEFAULT_OBJECT_POSITIONS = {
    "reach-v2": np.array([0.0, 0.5, 0.02]),
    "push-v2": np.array([0.0, 0.5, 0.02]),
    "pick-place-v2": np.array([0.0, 0.5, 0.02]),
}

# Default HANDLE positions for fixture tasks
DEFAULT_HANDLE_POSITIONS = {
    # drawer handle = drawer_body + [0, -0.16, 0], body default = [0, 0.9, 0]
    "drawer-open-v2": np.array([0.0, 0.7, 0.0]),
    # window-close: body=[0, 0.785, 0.16], initial qpos=0.2 → handle at body_X + 0.2
    "window-close-v2": np.array([0.2, 0.762, 0.2]),
    # window-open: body=[-0.1, 0.785, 0.16], initial qpos=0 → handle at body_X
    "window-open-v2": np.array([0, 0.7, 0.16]),
    # button-press-topdown: button box body position (MetaWorld default: [0, 0.8, 0.115])
    # Note: for button-press, "handle_position" actually means the button box body position.
    # Unlike drawer/window where handle and body are different, button-press uses body pos directly.
    "button-press-topdown-v2": np.array([0.0, 0.65, 0.115]),
}


def compute_fixture_positions(task_name: str, handle_position: np.ndarray):
    """
    Given a handle position, compute body position and goal position for fixture tasks.
    
    For drawer-open-v2:
        body = handle + [0, 0.16, 0]
        goal = body + [0, -0.36, 0.09]
    For window-close-v2:
        body = handle - [0.2, 0, 0]  (initial qpos=0.2)
        goal = body.copy()
    For window-open-v2:
        body = handle.copy()  (initial qpos=0)
        goal = body + [0.2, 0, 0]
    
    Args:
        task_name: Task identifier
        handle_position: 3D handle position [x, y, z]
    
    Returns:
        (body_position, goal_position) tuple of np.ndarray
    """
    handle = np.array(handle_position, dtype=np.float64)
    
    if task_name == "drawer-open-v2":
        # handle = body + [0, -0.16, 0] → body = handle - [0, -0.16, 0]
        body = handle + np.array([0.0, 0.16, 0.0])
        # goal = body + [0, -0.36, 0.09]
        goal = body + np.array([0.0, -0.36, 0.09])
        return body, goal
    
    elif task_name == "window-close-v2":
        # Window starts open, qpos=0.2, handle at body_X + 0.2
        # body_X = handle_X - 0.2
        body = handle.copy()
        body[0] = handle[0] - 0.2
        # goal = body (window closes to body position)
        goal = body.copy()
        return body, goal
    
    elif task_name == "window-open-v2":
        # Window starts closed, qpos=0, handle at body_X
        body = handle.copy()
        # goal = body + [0.2, 0, 0] (slide right to open)
        goal = body + np.array([0.2, 0.0, 0.0])
        return body, goal
    
    elif task_name == "button-press-topdown-v2":
        # For button-press-topdown, the "handle" is the button box body position.
        # MetaWorld's reset_model sets model.body("box").pos = obj_init_pos,
        # then _target_pos = _get_site_pos("hole") which depends on the box position.
        # The goal will be read back from sim after reset, so we use a placeholder here.
        body = handle.copy()
        # Placeholder goal — will be overridden by sim's _target_pos after reset
        goal = body.copy()
        return body, goal
    
    else:
        raise ValueError(f"compute_fixture_positions: not a fixture task: {task_name}")

# ============================================================================


# Mapping from task names to MetaWorld environment classes
TASK_TO_METAWORLD_ENV = {
    "reach-v2": SawyerReachEnvV2 if METAWORLD_AVAILABLE else None,
    "push-v2": SawyerPushEnvV2 if METAWORLD_AVAILABLE else None,
    "pick-place-v2": SawyerPickPlaceEnvV2 if METAWORLD_AVAILABLE else None,
    "drawer-open-v2": SawyerDrawerOpenEnvV2 if METAWORLD_AVAILABLE else None,
    "window-close-v2": SawyerWindowCloseEnvV2 if METAWORLD_AVAILABLE else None,
    "window-open-v2": SawyerWindowOpenEnvV2 if METAWORLD_AVAILABLE else None,
    "button-press-topdown-v2": SawyerButtonPressTopdownEnvV2 if METAWORLD_AVAILABLE else None,
}

# Tasks that have furniture/fixture objects instead of movable puck objects
# These tasks should NOT use _set_obj_xyz() which is for movable pucks
# button-press-topdown's _set_obj_xyz expects a SCALAR (qpos), not a 3D position,
# and its body is positioned via model.body("box").pos in reset_model.
TASKS_WITH_FIXTURE_OBJECTS = {
    "drawer-open-v2",              # Object is the drawer body, positioned via model.body("drawer").pos
    "window-close-v2",             # Object is the window body, positioned via model.body("window").pos
    "window-open-v2",              # Object is the window body, positioned via model.body("window").pos
    "button-press-topdown-v2",     # Button box body, positioned via model.body("box").pos
}


class SimulatedRealEnv:
    """
    Simulated Real Environment - Wrapper that runs real_envs logic on MuJoCo simulator.
    
    This class acts as a bridge between the real environment interface and the
    MetaWorld MuJoCo simulation. It:
    - Initializes a MetaWorld simulation environment as the physics backend
    - Overrides hardware-dependent methods to use simulation data
    - Preserves the real environment's reward function and task logic
    - Supports hardcoded goal and object positions
    - Enables real-time rendering of the MuJoCo simulation
    
    Attributes:
        task_name (str): Name of the task (e.g., "reach-v2")
        sim_env: MetaWorld simulation environment (physics backend)
        goal_position (np.ndarray): Goal position in simulation coordinates [3]
        object_position (np.ndarray): Object position in simulation coordinates [3]
        render_enabled (bool): Whether to render the MuJoCo simulation
    
    Virtual Goal Feature:
        For tasks where the trained model's goal range (e.g., Y=0.8-0.9) doesn't match
        the real robot's workspace (e.g., max Y=0.8), you can use "virtual goal":
        - Pass `goal_position` that the model was trained on (e.g., Y=0.85)
        - Pass `real_goal_position` where you actually want the robot to go (e.g., Y=0.75)
        - The observation/reward uses `goal_position` (what model expects)
        - Success check uses `real_goal_position` (when to stop in real world)
    """
    
    # Success threshold (meters)
    SUCCESS_THRESHOLD = 0.05
    LIFT_THRESHOLD = 0.01  # Minimum lift height to be considered "lifted"
    
    def __init__(
        self,
        task_name: str = "reach-v2",
        goal_position: Optional[np.ndarray] = None,
        object_position: Optional[np.ndarray] = None,
        handle_position: Optional[np.ndarray] = None,
        real_goal_position: Optional[np.ndarray] = None,
        ee_start_position: Optional[np.ndarray] = None,
        observation_offset: Optional[np.ndarray] = None,
        max_episode_steps: int = 200,
        render_mode: str = "human",
    ):
        """
        Initialize the simulated real environment.
        
        Args:
            task_name: Task name (reach-v2, push-v2, pick-place-v2, etc.)
            goal_position: Goal position [x, y, z] for reach/push/pick-place.
                          Ignored for fixture tasks if handle_position is provided.
            object_position: Object position [x, y, z] for reach/push/pick-place.
                            Ignored for fixture tasks if handle_position is provided.
            handle_position: Handle position [x, y, z] for fixture tasks 
                            (drawer-open-v2, window-close-v2, window-open-v2).
                            The code auto-computes body_position and goal_position.
                            If None, uses DEFAULT_HANDLE_POSITIONS.
            real_goal_position: Real goal position for success checking.
                               If None, uses goal_position.
            ee_start_position: Robot arm initial position [x, y, z].
                              If None, uses default [0.0, 0.6, 0.2].
            observation_offset: Offset added to all positions in observations [x, y, z].
            max_episode_steps: Maximum steps per episode.
            render_mode: "human" for window display, None to disable rendering.
        """
        if not METAWORLD_AVAILABLE:
            raise ImportError("MetaWorld is required for simulation mode")
        
        self.task_name = task_name
        self.max_episode_steps = max_episode_steps
        self.curr_path_length = 0
        self.render_enabled = render_mode == "human"
        
        # ==================== COORDINATE OFFSET FEATURE ====================
        if observation_offset is not None:
            self.observation_offset = np.array(observation_offset, dtype=np.float64)
        else:
            self.observation_offset = np.zeros(3, dtype=np.float64)
        
        if ee_start_position is not None:
            self.ee_start_position = np.array(ee_start_position, dtype=np.float64)
        else:
            self.ee_start_position = np.array([0.0, 0.6, 0.2], dtype=np.float64)
        # =================================================================
        
        # ==================== GOAL & OBJECT POSITIONS ====================
        if task_name in TASKS_WITH_FIXTURE_OBJECTS:
            # --- Fixture tasks: user provides handle_position only ---
            # Body position and goal are auto-computed from handle position
            if handle_position is not None:
                handle_pos = np.array(handle_position, dtype=np.float64)
            else:
                default_handle = DEFAULT_HANDLE_POSITIONS.get(task_name)
                if default_handle is None:
                    raise ValueError(
                        f"No default handle position for fixture task '{task_name}'. "
                        f"Provide handle_position explicitly."
                    )
                handle_pos = default_handle.copy()
            
            body_pos, goal_pos = compute_fixture_positions(task_name, handle_pos)
            
            # For fixture tasks, object_position stores the BODY position
            # (used for sim env setup: model.body("drawer").pos = body_pos)
            self.object_position = body_pos.copy()
            self.goal = goal_pos.copy()
            self._handle_position = handle_pos.copy()
            
            print(f"[SimulatedRealEnv] Fixture task '{task_name}':")
            print(f"  Handle position (input): {handle_pos}")
            print(f"  Body position (computed): {body_pos}")
            print(f"  Goal position (computed): {goal_pos}")
        else:
            # --- Standard tasks: user provides goal and object separately ---
            if goal_position is not None:
                self.goal = np.array(goal_position, dtype=np.float64)
            else:
                self.goal = DEFAULT_GOAL_POSITIONS.get(
                    task_name, np.array([0.0, 0.8, 0.15])
                ).copy()
            
            if object_position is not None:
                self.object_position = np.array(object_position, dtype=np.float64)
            else:
                self.object_position = DEFAULT_OBJECT_POSITIONS.get(
                    task_name, np.array([0.0, 0.6, 0.02])
                ).copy()
            
            self._handle_position = None
        
        # Virtual goal feature: use different goal for success checking
        if real_goal_position is not None:
            self.real_goal = np.array(real_goal_position, dtype=np.float64)
        else:
            self.real_goal = self.goal.copy()
        
        # Store FIXED positions - reset() restores these at the start of each episode
        self._fixed_goal = self.goal.copy()
        self._fixed_object_position = self.object_position.copy()
        self._fixed_real_goal = self.real_goal.copy()
        self._fixed_ee_start_position = self.ee_start_position.copy()
        self._fixed_observation_offset = self.observation_offset.copy()
        
        self.obj_init_pos = self.object_position.copy()
        # =================================================================
        
        # Initialize MetaWorld simulation environment
        self._init_sim_env(render_mode)
        
        # Initialize action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(39,),
            dtype=np.float64
        )
        
        # State tracking for observation
        self.ee_current_position = np.zeros(3)
        self.prev_ee_pos_sim = np.zeros(3)
        self.prev_obj_pos_sim = np.zeros(3)
        self.prev_gripper_state = 1.0
        
        # Action scaling (same as real env)
        self.action_scale = 1.0 / 100
        
        # EE init position (will be set in reset based on ee_start_position)
        self.ee_init_position = self.ee_start_position.copy()
        
        # init_tcp: actual TCP position after reset (set in reset())
        self.init_tcp = self.ee_start_position.copy()
        
        # window_handle_pos_init: stored during reset for correct reward margins
        self.window_handle_pos_init = self.object_position.copy()
        
        print(f"[SimulatedRealEnv] Initialized for task: {task_name}")
        print(f"[SimulatedRealEnv] EE start position: {self.ee_start_position}")
        print(f"[SimulatedRealEnv] Goal position: {self.goal}")
        if not np.allclose(self.real_goal, self.goal):
            print(f"[SimulatedRealEnv] Real goal position (for success check): {self.real_goal}")
        print(f"[SimulatedRealEnv] Object position: {self.object_position}")
        if not np.allclose(self.observation_offset, np.zeros(3)):
            print(f"[SimulatedRealEnv] Observation offset: {self.observation_offset}")
            print(f"[SimulatedRealEnv]   Agent sees: EE at {self.ee_start_position + self.observation_offset}, "
                  f"goal at {self.goal + self.observation_offset}")
        print(f"[SimulatedRealEnv] Rendering: {self.render_enabled}")
    
    def _init_sim_env(self, render_mode: str):
        """Initialize the MetaWorld simulation environment."""
        env_class = TASK_TO_METAWORLD_ENV.get(self.task_name)
        if env_class is None:
            raise ValueError(f"Unknown task: {self.task_name}. "
                           f"Available tasks: {list(TASK_TO_METAWORLD_ENV.keys())}")
        
        # Create simulation environment with rendering
        self.sim_env = env_class(render_mode=render_mode if self.render_enabled else None)
        
        # Set the task (required for MetaWorld)
        self.sim_env._partially_observable = False
        self.sim_env._set_task_called = True
        
        # Set custom start position for robot arm (hand_init_pos)
        # This determines where the robot arm starts in the simulation
        if hasattr(self.sim_env, 'hand_init_pos'):
            self.sim_env.hand_init_pos = self.ee_start_position.copy().astype(np.float32)
        if hasattr(self.sim_env, 'init_config'):
            self.sim_env.init_config['hand_init_pos'] = self.ee_start_position.copy().astype(np.float32)
        
        # Initialize _last_rand_vec before setting _freeze_rand_vec = True
        # This is required because MetaWorld's reset() calls _get_state_rand_vec()
        # which asserts _last_rand_vec is not None when _freeze_rand_vec is True
        if hasattr(self.sim_env, '_random_reset_space') and self.sim_env._random_reset_space is not None:
            # Generate initial random vector from the reset space
            rand_vec = np.random.uniform(
                self.sim_env._random_reset_space.low,
                self.sim_env._random_reset_space.high,
                size=self.sim_env._random_reset_space.low.size,
            ).astype(np.float64)
            self.sim_env._last_rand_vec = rand_vec
        
        # Now we can safely freeze the random vector
        self.sim_env._freeze_rand_vec = True
        
    def set_goal(self, goal: np.ndarray):
        """
        Set goal position (simulation coordinates).
        
        Args:
            goal: [3] Goal position in simulation coordinates
        """
        assert goal.shape[0] == 3
        self.goal = goal.copy()
        self._fixed_goal = goal.copy()  # Update fixed position too
        print(f"[SimulatedRealEnv] Goal set to: {self.goal}")
    
    def set_real_goal(self, real_goal: np.ndarray):
        """
        Set real goal position for success checking (virtual goal feature).
        
        Use this when the trained model's goal range doesn't match your robot's workspace.
        The model sees `goal` in observations, but success is checked against `real_goal`.
        
        Example:
            env.set_goal(np.array([0.0, 0.85, 0.15]))  # Model sees this (training range)
            env.set_real_goal(np.array([0.0, 0.75, 0.15]))  # Success checked here (robot workspace)
        
        Args:
            real_goal: [3] Real goal position for success checking
        """
        assert real_goal.shape[0] == 3
        self.real_goal = real_goal.copy()
        self._fixed_real_goal = real_goal.copy()
        print(f"[SimulatedRealEnv] Real goal set to: {self.real_goal}")
    
    def set_object_position(self, position: np.ndarray):
        """
        Set object position (simulation coordinates).
        
        Args:
            position: [3] Object position in simulation coordinates
        """
        assert position.shape[0] == 3
        self.object_position = position.copy()
        self._fixed_object_position = position.copy()  # Update fixed position too
        self.obj_init_pos = position.copy()
        print(f"[SimulatedRealEnv] Object position set to: {self.object_position}")
    
    def set_ee_start_position(self, position: np.ndarray):
        """
        Set robot arm starting position.
        
        Args:
            position: [3] Starting position [x, y, z] in simulation coordinates
        """
        assert position.shape[0] == 3
        self.ee_start_position = position.copy()
        self._fixed_ee_start_position = position.copy()
        self.ee_init_position = position.copy()
        
        # Update MetaWorld's hand_init_pos
        if hasattr(self.sim_env, 'hand_init_pos'):
            self.sim_env.hand_init_pos = position.copy().astype(np.float32)
        if hasattr(self.sim_env, 'init_config'):
            self.sim_env.init_config['hand_init_pos'] = position.copy().astype(np.float32)
        
        print(f"[SimulatedRealEnv] EE start position set to: {self.ee_start_position}")
    
    def set_observation_offset(self, offset: np.ndarray):
        """
        Set observation offset (coordinate transformation).
        
        This offset is added to all positions in observations, making the agent
        think it's in a different location than it actually is.
        
        Use case: Robot moves from Y=0.5 to Y=0.7, but with offset [0, 0.1, 0],
        agent sees Y=0.6 to Y=0.8 (agent's training range).
        
        Args:
            offset: [3] Offset to add to observations [x, y, z]
        """
        assert offset.shape[0] == 3
        self.observation_offset = offset.copy()
        self._fixed_observation_offset = offset.copy()
        print(f"[SimulatedRealEnv] Observation offset set to: {self.observation_offset}")
        print(f"[SimulatedRealEnv]   Agent now sees: EE at {self.ee_start_position + offset}, "
              f"goal at {self.goal + offset}")
    
    def _validate_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate goal/object positions to satisfy MetaWorld constraints.
        
        For non-fixture tasks (reach, push, pick-place):
            MetaWorld requires: distance between object XY and goal XY >= 0.15
            If the constraint is not satisfied, adjust goal position to be valid.
        
        For fixture tasks (drawer, window):
            Skip the XY distance check. In MetaWorld, the goal position for
            fixture tasks is computed from the body position in reset_model()
            (e.g., window-close: goal = body position, window-open: goal = body + [0.2,0,0]).
            The body position and goal are naturally close or identical.
        
        Returns:
            Tuple of (valid_object_position, valid_goal_position)
        """
        obj_pos = self.object_position.copy()
        goal_pos = self.goal.copy()
        
        # Skip XY distance check for fixture tasks - their goal is computed
        # from body position and is naturally close to it
        if self.task_name in TASKS_WITH_FIXTURE_OBJECTS:
            return obj_pos, goal_pos
        
        # Check XY distance constraint (required by MetaWorld reset_model)
        xy_dist = np.linalg.norm(obj_pos[:2] - goal_pos[:2])
        min_required_dist = 0.15
        
        if xy_dist < min_required_dist:
            print(f"[SimulatedRealEnv] WARNING: Object-Goal XY distance ({xy_dist:.3f}) < {min_required_dist}")
            print(f"[SimulatedRealEnv] Adjusting goal position to satisfy constraint...")
            
            # Adjust goal position to be at least min_required_dist away
            direction = goal_pos[:2] - obj_pos[:2]
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 0.001:
                direction = direction / dir_norm
            else:
                direction = np.array([0.0, 1.0])  # Default: move goal in +Y
            
            goal_pos[:2] = obj_pos[:2] + direction * min_required_dist
            print(f"[SimulatedRealEnv] Adjusted goal position: {goal_pos}")
        
        return obj_pos, goal_pos
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment.
        
        Returns:
            Initial observation [39]
        """
        # IMPORTANT: Restore fixed positions at start of each episode
        # This ensures consistent starting positions across all episodes
        self.goal = self._fixed_goal.copy()
        self.object_position = self._fixed_object_position.copy()
        self.obj_init_pos = self._fixed_object_position.copy()
        self.real_goal = self._fixed_real_goal.copy()
        self.ee_start_position = self._fixed_ee_start_position.copy()
        self.observation_offset = self._fixed_observation_offset.copy()
        self.ee_init_position = self._fixed_ee_start_position.copy()
        
        # Reset task-specific state
        self._gripper_auto_closed = False
        
        # Update MetaWorld's hand_init_pos for custom start position
        if hasattr(self.sim_env, 'hand_init_pos'):
            self.sim_env.hand_init_pos = self.ee_start_position.copy().astype(np.float32)
        if hasattr(self.sim_env, 'init_config'):
            self.sim_env.init_config['hand_init_pos'] = self.ee_start_position.copy().astype(np.float32)
        
        # For fixture tasks (drawer, window), the _fixed_object_position stores
        # the BODY position, not the handle position
        if self.task_name in TASKS_WITH_FIXTURE_OBJECTS:
            body_position = self._fixed_object_position.copy()
        
        # Validate positions to avoid infinite loop in MetaWorld's reset_model
        valid_obj_pos, valid_goal_pos = self._validate_positions()
        
        # Update _last_rand_vec to use validated goal and object positions
        # The random vector typically contains [obj_x, obj_y, obj_z, goal_x, goal_y, goal_z]
        # for tasks that have both object and goal positions
        if hasattr(self.sim_env, '_random_reset_space') and self.sim_env._random_reset_space is not None:
            rand_vec_size = self.sim_env._random_reset_space.low.size
            
            # For fixture tasks, rand_vec controls the body position
            fixture_obj_pos = body_position if self.task_name in TASKS_WITH_FIXTURE_OBJECTS else valid_obj_pos
            
            # Create rand_vec based on validated goal and object positions
            if rand_vec_size == 6:
                # Format: [obj_x, obj_y, obj_z, goal_x, goal_y, goal_z]
                self.sim_env._last_rand_vec = np.concatenate([
                    fixture_obj_pos,
                    valid_goal_pos
                ]).astype(np.float64)
            elif rand_vec_size == 3:
                # Format: just object or goal position (fixture tasks use body pos)
                self.sim_env._last_rand_vec = fixture_obj_pos.copy().astype(np.float64)
            else:
                # For other sizes, use a mix or default
                rand_vec = np.zeros(rand_vec_size, dtype=np.float64)
                rand_vec[:min(3, rand_vec_size)] = fixture_obj_pos[:min(3, rand_vec_size)]
                if rand_vec_size > 3:
                    remaining_slots = rand_vec_size - 3
                    rand_vec[3:3 + remaining_slots] = valid_goal_pos[:remaining_slots]
                self.sim_env._last_rand_vec = rand_vec
        
        # Reset simulation environment
        self.sim_env.reset()
        
        # After reset, set up goal and object positions
        if self.task_name in TASKS_WITH_FIXTURE_OBJECTS:
            # For fixture tasks (drawer, window), MetaWorld's reset_model() computes
            # the correct _target_pos from the body position automatically:
            #   window-close: _target_pos = obj_init_pos (goal = body position)
            #   window-open: _target_pos = obj_init_pos + [0.2, 0, 0]
            #   drawer-open: _target_pos = obj_init_pos + [0, -0.36, 0.09]
            # Read it back rather than overriding it.
            if hasattr(self.sim_env, '_target_pos'):
                self.goal = self.sim_env._target_pos.copy()
                valid_goal_pos = self.goal.copy()
        else:
            # For non-fixture tasks, set goal as validated
            if hasattr(self.sim_env, '_target_pos'):
                self.sim_env._target_pos = valid_goal_pos.copy()
            try:
                if hasattr(self.sim_env, 'model') and hasattr(self.sim_env.model, 'site'):
                    self.sim_env.model.site("goal").pos = valid_goal_pos.copy()
            except (AttributeError, KeyError):
                pass  # Goal site might not exist for all tasks
        
        # Set object position in simulation
        # For fixture-based tasks (drawer, window), the object is part of furniture
        # and should be set via model.body() instead of _set_obj_xyz()
        if self.task_name not in TASKS_WITH_FIXTURE_OBJECTS:
            # Standard movable object tasks (reach, push, pick-place)
            if hasattr(self.sim_env, '_set_obj_xyz'):
                try:
                    self.sim_env._set_obj_xyz(valid_obj_pos)
                except (ValueError, IndexError) as e:
                    print(f"[SimulatedRealEnv] Warning: Could not set object position: {e}")
        
        if hasattr(self.sim_env, 'obj_init_pos'):
            self.sim_env.obj_init_pos = valid_obj_pos.copy()
        
        # Update internal state
        self.goal = valid_goal_pos.copy()
        
        # Reset state tracking
        self.curr_path_length = 0
        self._update_state_from_sim()
        
        # For fixture tasks, read the actual handle position from sim after reset
        # This is the HANDLE position, not the body position
        # For non-fixture tasks, _update_state_from_sim already set object_position
        self.obj_init_pos = self.object_position.copy()
        
        # Store init_tcp (EE position after reset) for reward margin computations
        self.init_tcp = self.ee_current_position.copy()
        
        # For window-close/open, store window_handle_pos_init matching MetaWorld's logic
        # MetaWorld window-close: window_handle_pos_init = _get_pos_objects() + [0.2, 0, 0]
        #   (handle position when window is fully open, with qpos=0.2)
        # MetaWorld window-open:  window_handle_pos_init = _get_pos_objects()
        #   (handle position when window is fully closed, with qpos=0)
        if self.task_name == "window-close-v2":
            if hasattr(self.sim_env, 'window_handle_pos_init'):
                self.window_handle_pos_init = self.sim_env.window_handle_pos_init.copy()
            else:
                # Fallback: handle starts at obj_init_pos + [0.2, 0, 0] 
                self.window_handle_pos_init = self.object_position.copy() + np.array([0.2, 0.0, 0.0])
        elif self.task_name == "window-open-v2":
            if hasattr(self.sim_env, 'window_handle_pos_init'):
                self.window_handle_pos_init = self.sim_env.window_handle_pos_init.copy()
            else:
                self.window_handle_pos_init = self.object_position.copy()
        
        # For button-press-topdown, compute initial obj-to-target Z distance
        # (matching MetaWorld's _obj_to_target_init)
        if self.task_name == "button-press-topdown-v2":
            if hasattr(self.sim_env, '_obj_to_target_init'):
                self._obj_to_target_init = self.sim_env._obj_to_target_init
            else:
                self._obj_to_target_init = abs(
                    self.goal[2] - self.object_position[2]
                )
                # Fallback: approximate button travel distance if computed value is near-zero
                if self._obj_to_target_init < 0.001:
                    self._obj_to_target_init = 0.015  # ~15mm button press depth
        
        self.prev_ee_pos_sim = self.ee_current_position.copy()
        self.prev_obj_pos_sim = self.object_position.copy()
        self.prev_gripper_state = 1.0
        
        # Render if enabled
        if self.render_enabled:
            self.sim_env.render()
        
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the simulation.
        
        Uses real_env reward functions instead of MetaWorld's built-in reward.
        Applies task-specific action overrides (push z-limit, auto-close gripper).
        
        Args:
            action: [4] Action vector [delta_x, delta_y, delta_z, gripper]
        
        Returns:
            observation, reward, done, info
        """
        assert action.shape[0] == 4
        action = np.clip(action, -1, 1).copy()
        
        # Task-specific action overrides (matching real_env behavior)
        if self.task_name == "push-v2":
            # 安全限制: 夹爪 Z 位置低于 0.02 时不让继续向下
            if self.ee_current_position[2] <= 0.02:
                action[2] = max(action[2], 0)
            # 当夹爪接近物体时自动闭合
            tcp_to_obj = np.linalg.norm(self.ee_current_position - self.object_position)
            if tcp_to_obj < 0.02:
                self._gripper_auto_closed = True
            if self._gripper_auto_closed:
                action[3] = 1.0  # MetaWorld: positive = close gripper
        
        elif self.task_name == "pick-place-v2":
            # 当夹爪接近物体时自动闭合
            tcp_to_obj = np.linalg.norm(self.ee_current_position - self.object_position)
            if tcp_to_obj < 0.02:
                self._gripper_auto_closed = True
            if self._gripper_auto_closed:
                action[3] = 1.0  # MetaWorld: positive = close gripper
        
        # Execute action in simulation
        sim_obs, sim_reward, sim_terminated, sim_truncated, sim_info = self.sim_env.step(action.astype(np.float32))
        
        # Update state from simulation
        self._update_state_from_sim()
        
        # Get observation in real_env format
        obs = self._get_obs()
        obs_dict = self._get_obs_dict()
        
        # Use real_env reward functions instead of MetaWorld's sim_reward
        reward, reward_info = self.compute_reward(action, obs_dict)
        
        # Check success
        success = self._check_success(obs_dict)
        
        self.curr_path_length += 1
        done = self.curr_path_length >= self.max_episode_steps
        
        info = {
            'success': float(success),
            'goal': self.goal.copy(),
            **reward_info
        }
        
        # Render if enabled
        if self.render_enabled:
            self.sim_env.render()
        
        return obs, reward, done, info
    
    def _update_state_from_sim(self):
        """Update internal state from MuJoCo simulation data."""
        # Get end-effector position from simulation
        if hasattr(self.sim_env, 'tcp_center'):
            self.ee_current_position = self.sim_env.tcp_center.copy()
        elif hasattr(self.sim_env, 'get_endeff_pos'):
            self.ee_current_position = self.sim_env.get_endeff_pos().copy()
        else:
            # Fallback to reading from sim data
            self.ee_current_position = self.sim_env.data.site("grip_site").xpos.copy()
        
        # Get object position from simulation for ALL tasks
        # For fixture tasks (drawer, window), _get_pos_objects returns the HANDLE
        # position which changes dynamically during the task:
        #   - drawer: drawer_link COM + [0, -0.16, 0]
        #   - window-close: handleCloseStart site position
        #   - window-open: handleOpenStart site position
        if hasattr(self.sim_env, '_get_pos_objects'):
            self.object_position = self.sim_env._get_pos_objects().copy()
    
    def _get_object_quat(self) -> np.ndarray:
        """
        获取物体四元数 - 与 real_env 各任务保持一致
        
        - reach/push/pick_place: [0, 0, 0, 1] (无旋转)
        - window-close/window-open: [0, 0, 0, 0] (MetaWorld 约定)
        - drawer-open/button-press-topdown: [1, 0, 0, 0] (MuJoCo 默认)
        """
        if self.task_name in ["window-close-v2", "window-open-v2"]:
            return np.zeros(4)
        elif self.task_name in ["drawer-open-v2"]:
            return np.array([1.0, 0.0, 0.0, 0.0])
        elif self.task_name in ["button-press-topdown-v2"]:
            return np.array([0.7074, -0.7068,  0.0000,  0.0000])
        else:
            return np.array([0.0, 0.0, 0.0, 1.0])
    
    def _get_obs(self) -> np.ndarray:
        """
        Get 39-dim observation (simulation coordinates).
        
        Matches the format from real_envs/base_env.py, including task-specific quaternions.
        
        NOTE: Applies observation_offset to all positions, so the agent sees
        shifted coordinates. Robot is actually at Y=0.5, but with offset [0, 0.1, 0],
        agent sees Y=0.6.
        """
        # Get actual positions
        pos_hand_actual = self.ee_current_position.copy()
        pos_obj_actual = self.object_position.copy()
        pos_goal_actual = self.goal.copy()
        
        # Apply observation offset - agent sees shifted positions
        # This allows reusing trained policies with different workspace limits
        pos_hand_agent = pos_hand_actual + self.observation_offset
        pos_obj_agent = pos_obj_actual + self.observation_offset
        pos_goal_agent = pos_goal_actual + self.observation_offset
        
        # Current gripper state from simulation
        # In sim we can get the actual gripper opening distance
        curr_gripper = self._get_gripper_state_from_sim()
        
        # Task-specific quaternion (matching real_env)
        obj_quat = self._get_object_quat()
        
        # Construct current frame (with offset applied)
        curr_frame = np.zeros(18)
        curr_frame[0:3] = pos_hand_agent
        curr_frame[3] = curr_gripper
        curr_frame[4:7] = pos_obj_agent
        curr_frame[7:11] = obj_quat
        
        # Previous frame also needs offset applied
        prev_hand_agent = self.prev_ee_pos_sim + self.observation_offset
        prev_obj_agent = self.prev_obj_pos_sim + self.observation_offset
        
        # Construct previous frame (with offset applied)
        prev_frame = np.zeros(18)
        prev_frame[0:3] = prev_hand_agent
        prev_frame[3] = self.prev_gripper_state
        prev_frame[4:7] = prev_obj_agent
        prev_frame[7:11] = obj_quat
        
        # Construct observation
        obs = np.hstack([
            curr_frame,     # 0-17
            prev_frame,     # 18-35
            pos_goal_agent  # 36-38
        ])
        
        # Update caches (store actual positions, not offset ones)
        self.prev_ee_pos_sim = pos_hand_actual.copy()
        self.prev_obj_pos_sim = pos_obj_actual.copy()
        self.prev_gripper_state = curr_gripper
        
        return obs
    
    def _get_obs_dict(self) -> Dict[str, Any]:
        """Get observation dictionary (with offset applied)."""
        return {
            'state_observation': self._get_obs(),
            'state_desired_goal': self.goal.copy() + self.observation_offset,
            'state_achieved_goal': self.object_position.copy() + self.observation_offset,
            'ee_position': self.ee_current_position.copy() + self.observation_offset,
            'object_position': self.object_position.copy() + self.observation_offset,
        }
    
    def _get_gripper_state_from_sim(self) -> float:
        """
        从仿真中获取夹爪状态（用于观测），匹配 MetaWorld obs[3] 的范围 [0, 1]
        
        MetaWorld obs[3] = clip(gripper_distance / 0.1, 0, 1):
          - 0.0 = 完全关闭
          - 1.0 = 完全打开
        
        注意: 这与 action 空间的 [-1, 1] 不同！obs 是 [0, 1]。
        """
        try:
            finger_right = self.sim_env.data.body("rightclaw").xpos
            finger_left = self.sim_env.data.body("leftclaw").xpos
            gripper_distance = np.linalg.norm(finger_right - finger_left)
            # 与 MetaWorld 一致: 归一化到 [0, 1]
            normalized = np.clip(gripper_distance / 0.1, 0.0, 1.0)
            return float(normalized)
        except (AttributeError, KeyError):
            print("still wrong")
            return 1.0  # Default open
    
    def get_state_for_transformer(self) -> np.ndarray:
        """Get 21-dim state for transformer model (with offset applied)."""
        # Apply observation offset so agent sees shifted coordinates
        pos_hand_agent = self.ee_current_position.copy() + self.observation_offset
        pos_obj_agent = self.object_position.copy() + self.observation_offset
        pos_goal_agent = self.goal.copy() + self.observation_offset
        
        gripper_state = np.array([self._get_gripper_state_from_sim()])
        obj_quat_sim = self._get_object_quat()
        padding = np.zeros(21 - 14)
        
        state = np.hstack([
            pos_hand_agent,   # 0-2
            gripper_state,    # 3
            pos_obj_agent,    # 4-6
            obj_quat_sim,     # 7-10
            padding,          # 11-17
            pos_goal_agent,   # 18-20
        ])
        
        return state
    
    def compute_reward(self, action: np.ndarray, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute reward based on task type.
        
        Uses the same reward logic as the real environment.
        """
        if self.task_name == "reach-v2":
            return self._compute_reach_reward(action, obs_dict)
        elif self.task_name == "push-v2":
            return self._compute_push_reward(action, obs_dict)
        elif self.task_name == "pick-place-v2":
            return self._compute_pick_place_reward(action, obs_dict)
        elif self.task_name == "drawer-open-v2":
            return self._compute_drawer_open_reward(action, obs_dict)
        elif self.task_name in ["window-close-v2", "window-open-v2"]:
            return self._compute_window_reward(action, obs_dict)
        elif self.task_name == "button-press-topdown-v2":
            return self._compute_button_press_topdown_reward(action, obs_dict)
        else:
            # Default to reach reward
            return self._compute_reach_reward(action, obs_dict)
    
    def _compute_reach_reward(self, action: np.ndarray, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Reach task reward."""
        tcp = self.ee_current_position
        goal = self.goal
        
        tcp_to_target = np.linalg.norm(tcp - goal)
        in_place_margin = np.linalg.norm(self.ee_init_position - goal)
        
        TARGET_RADIUS = 0.05
        in_place = self._tolerance(
            tcp_to_target,
            bounds=(0, TARGET_RADIUS),
            margin=in_place_margin,
            value_at_margin=0.1
        )
        
        reward = 10.0 * in_place
        
        info = {
            'reach_dist': tcp_to_target,
            'reward': reward,
            'success': float(tcp_to_target <= TARGET_RADIUS)
        }
        return reward, info
    
    def _compute_push_reward(self, action: np.ndarray, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Push task reward - matching real_env push_env.py compute_reward."""
        tcp = self.ee_current_position
        obj = self.object_position
        target = self.goal
        
        # 夹爪开闭状态: 使用实际夹爪状态 (匹配 MetaWorld 的 obs[3])
        # obs[3] = gripper distance apart, [0, 1], 0=closed, 1=open
        tcp_opened = self._get_gripper_state_from_sim()
        
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        target_to_obj = float(np.linalg.norm(obj - target))
        target_to_obj_init = float(np.linalg.norm(self.obj_init_pos - target))
        
        TARGET_RADIUS = 0.05
        
        in_place = self._tolerance(
            target_to_obj,
            bounds=(0, TARGET_RADIUS),
            margin=target_to_obj_init,
        )
        
        # 在仿真中使用 MetaWorld 的 _gripper_caging_reward 以获得准确的 gripper reward
        if hasattr(self.sim_env, '_gripper_caging_reward'):
            object_grasped = self.sim_env._gripper_caging_reward(
                action,
                obj,
                object_reach_radius=0.01,
                obj_radius=0.015,
                pad_success_thresh=0.05,
                xz_thresh=0.005,
                high_density=True,
            )
        else:
            object_grasped = max(0, 1.0 - tcp_to_obj / 0.1)
        
        reward = 2 * object_grasped
        
        if tcp_to_obj < 0.02 and tcp_opened > 0:
            reward += 1.0 + reward + 5.0 * in_place
        if target_to_obj < TARGET_RADIUS:
            reward = 10.0
        
        info = {
            'tcp_to_obj': tcp_to_obj,
            'tcp_opened': tcp_opened,
            'target_to_obj': target_to_obj,
            'object_grasped': object_grasped,
            'in_place': in_place,
            'success': float(target_to_obj <= TARGET_RADIUS)
        }
        
        return reward, info
    
    def _compute_pick_place_reward(self, action: np.ndarray, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Pick-place task reward — delegates to MetaWorld's native compute_reward.
        
        MetaWorld's SawyerPickPlaceEnvV2.compute_reward() uses internal state
        (init_left_pad, init_right_pad, init_tcp, obj_init_pos) that is set during
        reset_model(). The _gripper_caging_reward override in pick-place uses these
        internal pad positions which cannot be accurately reproduced externally.
        
        By delegating to the sim_env, we get the exact same reward computation
        that MetaWorld uses, including the correct caging+gripping logic.
        """
        try:
            # Build an obs vector matching MetaWorld's format (actual sim positions, no offset).
            # MetaWorld observation format: [hand(3), gripper(1), obj(3), obj_quat(4), ...(7), prev_hand(3), prev_gripper(1), prev_obj(3), prev_quat(4), ...(7), goal(3)]
            obs_for_metaworld = np.zeros(39)
            obs_for_metaworld[0:3] = self.ee_current_position.copy()   # hand position
            obs_for_metaworld[3] = self._get_gripper_state_from_sim()  # gripper state [0,1]
            obs_for_metaworld[4:7] = self.object_position.copy()       # object position
            obs_for_metaworld[7:11] = self._get_object_quat()          # object quaternion
            # prev frame
            obs_for_metaworld[18:21] = self.prev_ee_pos_sim.copy()
            obs_for_metaworld[21] = self.prev_gripper_state
            obs_for_metaworld[22:25] = self.prev_obj_pos_sim.copy()
            obs_for_metaworld[25:29] = self._get_object_quat()
            # goal
            obs_for_metaworld[36:39] = self.goal.copy()
            
            # Delegate to MetaWorld's compute_reward
            reward_tuple = self.sim_env.compute_reward(action.astype(np.float32), obs_for_metaworld)
            reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place = reward_tuple
            
            info = {
                'obj_to_target': obj_to_target,
                'tcp_to_obj': tcp_to_obj,
                'in_place': in_place,
                'object_grasped': object_grasped,
                'lifted': self.object_position[2] - 0.01 > self.obj_init_pos[2],
                'success': float(obj_to_target <= 0.07)
            }
            return float(reward), info
        except Exception as e:
            # Fallback: manual computation if MetaWorld delegation fails
            print(f"[SimulatedRealEnv] Warning: MetaWorld compute_reward failed, using fallback: {e}")
            return self._compute_pick_place_reward_fallback(action, obs_dict)
    
    def _compute_pick_place_reward_fallback(self, action: np.ndarray, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Fallback pick-place reward computation."""
        tcp = self.ee_current_position
        obj = self.object_position
        target = self.goal
        
        obj_to_target = float(np.linalg.norm(obj - target))
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        in_place_margin = float(np.linalg.norm(self.obj_init_pos - target))
        tcp_opened = self._get_gripper_state_from_sim()
        
        TARGET_RADIUS = 0.05
        
        in_place = self._tolerance(
            obj_to_target,
            bounds=(0, TARGET_RADIUS),
            margin=in_place_margin,
        )
        
        if hasattr(self.sim_env, '_gripper_caging_reward'):
            object_grasped = self.sim_env._gripper_caging_reward(action, obj)
        else:
            object_grasped = max(0, 1.0 - tcp_to_obj / 0.1)
        
        in_place_and_object_grasped = self._hamacher_product(object_grasped, in_place)
        reward = in_place_and_object_grasped
        
        lifted = obj[2] - self.LIFT_THRESHOLD > self.obj_init_pos[2]
        if tcp_to_obj < 0.02 and tcp_opened > 0 and lifted:
            reward += 1.0 + 5.0 * in_place
        if obj_to_target < TARGET_RADIUS:
            reward = 10.0
        
        info = {
            'obj_to_target': obj_to_target,
            'tcp_to_obj': tcp_to_obj,
            'in_place': in_place,
            'object_grasped': object_grasped,
            'lifted': lifted,
            'success': float(obj_to_target <= 0.07)
        }
        
        return reward, info
    
    def _compute_drawer_open_reward(self, action: np.ndarray, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Drawer open task reward - matching real_env drawer_open_env.py compute_reward."""
        gripper = self.ee_current_position
        handle = self.object_position
        target = self.goal
        
        maxDist = 0.2
        handle_error = float(np.linalg.norm(handle - target))
        
        reward_for_opening = self._tolerance(
            handle_error, bounds=(0, 0.02), margin=maxDist
        )
        
        # Caging reward matching MetaWorld drawer_open_v2
        handle_pos_init = target + np.array([0.0, maxDist, 0.0])
        scale = np.array([3.0, 3.0, 1.0])
        gripper_error = (handle - gripper) * scale
        gripper_error_init = (handle_pos_init - self.ee_init_position) * scale
        
        reward_for_caging = self._tolerance(
            float(np.linalg.norm(gripper_error)),
            bounds=(0, 0.01),
            margin=float(np.linalg.norm(gripper_error_init)),
        )
        
        reward = (reward_for_caging + reward_for_opening) * 5.0
        
        info = {
            'handle_error': handle_error,
            'gripper_to_handle': float(np.linalg.norm(handle - gripper)),
            'caging_reward': reward_for_caging,
            'opening_reward': reward_for_opening,
            'success': float(handle_error <= 0.03)
        }
        
        return reward, info
    
    def _compute_window_reward(self, action: np.ndarray, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Window open/close task reward — matching MetaWorld's compute_reward exactly.
        
        MetaWorld window-close compute_reward uses:
          - target_to_obj_init = window_handle_pos_init[0] - target[0]
          - tcp_to_obj_init = ||window_handle_pos_init - init_tcp||
          - reach sigmoid = "gaussian"
        
        MetaWorld window-open compute_reward uses:
          - target_to_obj_init = obj_init_pos[0] - target[0]  (≈ window_handle_pos_init[0] since qpos=0)
          - tcp_to_obj_init = ||window_handle_pos_init - init_tcp||
          - reach sigmoid = "long_tail"
        """
        tcp = self.ee_current_position
        obj = self.object_position  # current handle position (from _get_pos_objects)
        target = self.goal
        
        TARGET_RADIUS = 0.05
        
        # in_place: how close the handle X is to the target X
        target_to_obj = float(abs(obj[0] - target[0]))
        
        # For the margin, use window_handle_pos_init (stored during reset)
        if self.task_name == "window-close-v2":
            # MetaWorld: target_to_obj_init = window_handle_pos_init[0] - target[0]
            target_to_obj_init = float(abs(self.window_handle_pos_init[0] - target[0]))
        else:
            # window-open: target_to_obj_init = obj_init_pos[0] - target[0]
            # (obj_init_pos = body position, window_handle_pos_init ≈ handle at qpos=0)
            target_to_obj_init = float(abs(self.window_handle_pos_init[0] - target[0]))
        
        in_place = self._tolerance(
            target_to_obj,
            bounds=(0, TARGET_RADIUS),
            margin=abs(target_to_obj_init - TARGET_RADIUS),
        )
        
        handle_radius = 0.02
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        # MetaWorld: tcp_to_obj_init = ||window_handle_pos_init - init_tcp||
        tcp_to_obj_init = float(np.linalg.norm(self.window_handle_pos_init - self.init_tcp))
        
        # MetaWorld: window-close uses "gaussian", window-open uses "long_tail"
        reach_sigmoid = "gaussian" if self.task_name == "window-close-v2" else "long_tail"
        
        reach = self._tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid=reach_sigmoid,
        )
        
        object_grasped = reach
        reward = 10 * self._hamacher_product(reach, in_place)
        
        info = {
            'target_to_obj': target_to_obj,
            'tcp_to_obj': tcp_to_obj,
            'reach': reach,
            'in_place': in_place,
            'grasp_reward': object_grasped,
            'success': float(target_to_obj <= TARGET_RADIUS)
        }
        
        return reward, info
    
    def _compute_button_press_topdown_reward(self, action: np.ndarray, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Button press topdown task reward - matching MetaWorld sawyer_button_press_topdown_v2.
        
        MetaWorld's compute_reward uses:
          tcp_to_obj_init = ||obj - init_tcp||  (not ee_init_position)
        """
        tcp = self.ee_current_position
        obj = self.object_position  # button top position (from _get_pos_objects)
        target = self.goal
        print("object position:", obj)
        print("tcp position: ", tcp)
        
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        # MetaWorld: tcp_to_obj_init = ||obj - self.init_tcp||
        tcp_to_obj_init = float(np.linalg.norm(obj - self.init_tcp))
        obj_to_target = abs(target[2] - obj[2])
        
        # tcp_closed = 1 - obs[3], where obs[3] is gripper opening [0,1]
        tcp_closed = 1.0 - self._get_gripper_state_from_sim()
        
        near_button = self._tolerance(
            tcp_to_obj,
            bounds=(0, 0.01),
            margin=tcp_to_obj_init,
            sigmoid="long_tail",
        )
        
        # _obj_to_target_init: initial Z distance from button top to target
        if not hasattr(self, '_obj_to_target_init') or self._obj_to_target_init is None:
            self._obj_to_target_init = abs(target[2] - self.obj_init_pos[2])
            if self._obj_to_target_init < 0.001:
                self._obj_to_target_init = 0.015  # fallback
        
        button_pressed = self._tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=self._obj_to_target_init,
            sigmoid="long_tail",
        )
        
        reward = 5 * self._hamacher_product(tcp_closed, near_button)
        print("_hamacher_product:", reward)
        if tcp_to_obj <= 0.03:
            reward += 5 * button_pressed
        
        info = {
            'tcp_to_obj': tcp_to_obj,
            'obj_to_target': obj_to_target,
            'near_button': near_button,
            'button_pressed': button_pressed,
            'tcp_closed': tcp_closed,
            'success': float(obj_to_target <= 0.024)
        }
        
        return reward, info
    
    def _check_success(self, obs_dict: Dict[str, Any]) -> bool:
        """
        Check if task is successful.
        
        NOTE: Uses `real_goal` for success checking, not `goal`.
        This supports the "virtual goal" feature where the model sees one goal
        but success is measured against a different (real) goal position.
        """
        if self.task_name == "reach-v2":
            # For reach, check distance from end-effector to real_goal
            reach_dist = np.linalg.norm(self.ee_current_position - self.real_goal)
            return reach_dist <= self.SUCCESS_THRESHOLD
        elif self.task_name in ["push-v2", "pick-place-v2"]:
            # For push/pick-place, check distance from object to real_goal
            obj_dist = np.linalg.norm(self.object_position - self.real_goal)
            return obj_dist <= self.SUCCESS_THRESHOLD
        elif self.task_name == "drawer-open-v2":
            # For drawer, check handle distance to real_goal
            handle_error = np.linalg.norm(self.object_position - self.real_goal)
            return handle_error <= 0.03
        elif self.task_name in ["window-close-v2", "window-open-v2"]:
            # For window, check X distance from handle to real_goal X
            target_to_obj = abs(self.object_position[0] - self.real_goal[0])
            return target_to_obj <= self.SUCCESS_THRESHOLD
        elif self.task_name == "button-press-topdown-v2":
            # For button press, check Z distance from button to real_goal
            # 0.024 matches MetaWorld's evaluate_state success threshold
            obj_to_target = abs(self.object_position[2] - self.real_goal[2])
            return obj_to_target <= 0.024
        return False
    
    @staticmethod
    def _tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='long_tail', value_at_margin=0.1):
        """Tolerance function for reward computation - matches MetaWorld reward_utils.tolerance."""
        lower, upper = bounds
        if lower <= x <= upper:
            return 1.0
        
        if margin == 0:
            return 0.0
        
        if x < lower:
            d = (lower - x) / margin
        else:
            d = (x - upper) / margin
        
        if sigmoid == 'gaussian':
            scale = np.sqrt(-2 * np.log(value_at_margin))
            return np.exp(-0.5 * (d * scale) ** 2)
        elif sigmoid == 'long_tail':
            scale = np.sqrt(1 / value_at_margin - 1)
            return 1 / ((d * scale) ** 2 + 1)
        else:
            # Linear fallback
            return max(0, 1 - d)
    
    @staticmethod
    def _hamacher_product(a: float, b: float) -> float:
        """Hamacher product for reward computation."""
        a = max(0, min(1, a))
        b = max(0, min(1, b))
        denominator = a + b - a * b
        return (a * b / denominator) if denominator > 0 else 0
    
    def close(self):
        """Close the environment."""
        if hasattr(self, 'sim_env') and self.sim_env is not None:
            self.sim_env.close()
        print(f"[SimulatedRealEnv] Closed")


def create_simulated_real_env(
    task_name: str,
    goal_position: Optional[np.ndarray] = None,
    object_position: Optional[np.ndarray] = None,
    handle_position: Optional[np.ndarray] = None,
    max_episode_steps: int = 200,
    render_mode: str = "human",
    observation_offset: Optional[np.ndarray] = None
) -> SimulatedRealEnv:
    """
    Factory function to create a simulated real environment.
    
    For push-v2 / pick-place-v2: set object_position and goal_position.
    For drawer-open-v2 / window-open-v2 / window-close-v2: set handle_position only.
    
    Args:
        task_name: Task name (reach-v2, push-v2, etc.)
        goal_position: Goal position for reach/push/pick-place tasks
        object_position: Object position for reach/push/pick-place tasks
        handle_position: Handle position for fixture tasks (drawer/window).
                        Body and goal are auto-computed.
        max_episode_steps: Maximum steps per episode
        render_mode: Rendering mode ("human" or None)
        observation_offset: Coordinate offset for observations
    
    Returns:
        SimulatedRealEnv instance
    """
    return SimulatedRealEnv(
        task_name=task_name,
        goal_position=goal_position,
        object_position=object_position,
        handle_position=handle_position,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
        observation_offset=observation_offset
    )


def list_available_sim_tasks():
    """List all available tasks for simulation."""
    return list(TASK_TO_METAWORLD_ENV.keys())