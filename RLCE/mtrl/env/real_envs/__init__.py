# mtrl/env/real_envs/__init__.py
"""
真实 KUKA 机器人任务环境

本模块提供了 10 个与 Metaworld 仿真环境接口兼容的真实机器人环境：

1. KukaReachEnv - reach-v2
2. KukaPushEnv - push-v2
3. KukaPickPlaceEnv - pick-place-v2
4. KukaDoorOpenEnv - door-open-v2
5. KukaWindowOpenEnv - window-open-v2
6. KukaWindowCloseEnv - window-close-v2
7. KukaDrawerOpenEnv - drawer-open-v2
8. KukaButtonPressTopdownEnv - button-press-topdown-v2
9. KukaFaucetOpenEnv - faucet-open-v2
10. KukaPegInsertSideEnv - peg-insert-side-v2

使用方法:
    from mtrl.env.real_envs import KukaReachEnv, create_real_env
    
    # 方法1: 直接使用类
    env = KukaReachEnv()
    env.set_goal(np.array([0.7, 0.0, 0.15]))  # 手动设置目标
    
    # 方法2: 使用工厂函数
    env = create_real_env("reach-v2")
    env.set_goal_from_camera()  # 使用相机获取目标
"""

from mtrl.env.real_envs.base_env import KukaBaseRealEnv
from mtrl.env.real_envs.reach_env import KukaReachEnv
from mtrl.env.real_envs.push_env import KukaPushEnv
from mtrl.env.real_envs.pick_place_env import KukaPickPlaceEnv
from mtrl.env.real_envs.drawer_open_env import KukaDrawerOpenEnv
from mtrl.env.real_envs.window_close_env import KukaWindowCloseEnv
from mtrl.env.real_envs.window_open_env import KukaWindowOpenEnv
from mtrl.env.real_envs.button_press_topdown_env import KukaButtonPressTopdownEnv
# from mtrl.env.real_envs.door_open_env import KukaDoorOpenEnv
# from mtrl.env.real_envs.faucet_open_env import KukaFaucetOpenEnv
# from mtrl.env.real_envs.peg_insert_side_env import KukaPegInsertSideEnv

# 任务名称到环境类的映射
TASK_ENV_MAP = {
    "reach-v2": KukaReachEnv,
    "push-v2": KukaPushEnv,
    "pick-place-v2": KukaPickPlaceEnv,
    "drawer-open-v2": KukaDrawerOpenEnv,
    "window-close-v2": KukaWindowCloseEnv,
    "window-open-v2": KukaWindowOpenEnv,
    "button-press-topdown-v2": KukaButtonPressTopdownEnv,
    # "door-open-v2": KukaDoorOpenEnv,
    # "faucet-open-v2": KukaFaucetOpenEnv,
    # "peg-insert-side-v2": KukaPegInsertSideEnv,
}


def create_real_env(task_name: str, **kwargs):
    """
    根据任务名称创建真实机器人环境
    
    Args:
        task_name: 任务名称 (e.g., "reach-v2", "push-v2")
        **kwargs: 传递给环境构造函数的参数
    
    Returns:
        对应的真实机器人环境实例
    
    Example:
        env = create_real_env("reach-v2")
        env = create_real_env("push-v2", use_camera=True)
    """
    if task_name not in TASK_ENV_MAP:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {list(TASK_ENV_MAP.keys())}"
        )
    
    env_class = TASK_ENV_MAP[task_name]
    return env_class(**kwargs)


def list_available_tasks():
    """列出所有可用的任务名称"""
    return list(TASK_ENV_MAP.keys())


__all__ = [
    # 基类
    "KukaBaseRealEnv",
    # 具体任务环境
    "KukaReachEnv",
    "KukaPushEnv", 
    "KukaPickPlaceEnv",
    "KukaDoorOpenEnv",
    "KukaWindowOpenEnv",
    "KukaWindowCloseEnv",
    "KukaDrawerOpenEnv",
    "KukaButtonPressTopdownEnv",
    "KukaFaucetOpenEnv",
    "KukaPegInsertSideEnv",
    # 工具函数
    "create_real_env",
    "list_available_tasks",
    "TASK_ENV_MAP",
]