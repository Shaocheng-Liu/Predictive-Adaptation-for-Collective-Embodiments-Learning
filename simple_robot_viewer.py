#!/usr/bin/env python3
"""
Simple robot arm visualizer with Object Base Position tracking.
"""

import argparse
import sys
import os
import time
import traceback

import numpy as np

# Add metaworld to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Metaworld'))

import metaworld
from metaworld.envs.robot_utils import set_robot_type, patch_all_metaworld_envs


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Simple robot arm viewer')
    parser.add_argument('--robot', type=str, default='panda', 
                        choices=['sawyer', 'panda', 'kuka', 'ur5e', 'ur10e','unitree_z1','viperx', 'gen3','xarm7'],
                        help='Robot arm to visualize')
    parser.add_argument('--env', type=str, default='reach-v2', help='Environment to load')
    parser.add_argument('--camera', type=str, default='corner3', help='Camera view')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps')
    parser.add_argument('--demo', action='store_true', help='Run a simple movement demo')
    # 默认初始位置设为 None，方便后续逻辑处理
    parser.add_argument('--init-pos', type=float, nargs=3, default=[0, 0.6, 0.2], help='Initial hand pos (x y z)')
    return parser.parse_args()


def create_env(robot_type, env_name, camera_name):
    """创建环境"""
    set_robot_type(robot_type)
    patch_all_metaworld_envs()
    print(f"\nCreating {robot_type} in {env_name}...")
    
    ml1 = metaworld.ML1(env_name, seed=42)
    env_cls = ml1.train_classes[env_name]
    env = env_cls(render_mode='human', camera_name=camera_name)
    
    task = ml1.train_tasks[0]
    env.set_task(task)
    
    return env


def get_base_body_pos(env, env_name):
    """
    根据环境名称，智能推断主要物体的 Body 名称并获取其坐标。
    MetaWorld 中不同任务的基座 Body 名字不同。
    """
    body_name = None
    env_name = env_name.lower()
    
    # 1. 简单的名称映射逻辑
    if 'window' in env_name:
        body_name = 'window'      # 窗户任务的基座叫 window
    elif 'button' in env_name:
        body_name = 'box'         # 按钮任务的基座叫 box
    elif 'drawer' in env_name:
        body_name = 'drawer'      # 抽屉任务的基座叫 drawer
    elif 'door' in env_name:
        body_name = 'door'        # 门任务的基座叫 door
    elif 'peg' in env_name:
        body_name = 'box'         # 插销任务的插孔盒子叫 box
    elif 'pick' in env_name or 'assembly' in env_name:
        # Pick Place 只有物体，没有基座，或者基座是桌子(table)
        return None

    # 2. 尝试从 MuJoCo data 中读取
    if body_name:
        try:
            # env.unwrapped.data.body(name).xpos 返回的是实时的世界坐标
            return env.unwrapped.data.body(body_name).xpos
        except Exception:
            return None
    return None


def run_simple_viewer(args):
    """主循环逻辑"""
    env = create_env(args.robot, args.env, args.camera)
    
    # 注入初始位置
    if args.init_pos is not None:
        target_pos = np.array(args.init_pos)
        print(f"[Config] Overwriting initial hand position to: {target_pos}")
        
        if hasattr(env, 'init_config'):
            env.init_config['hand_init_pos'] = target_pos
        
        raw_env = env.unwrapped
        if hasattr(raw_env, 'hand_init_pos'):
            raw_env.hand_init_pos = target_pos

    print("\nInitializing environment...")
    env.reset()
    
    print("-" * 115)
    # 三列数据对比：机械臂手 | 交互点(把手/按钮) | 物体基座(窗框/盒子)
    print(f"{'Step':<6} | {'TCP (Hand)':<30} | {'Interact Site (Handle/Btn)':<30} | {'Base Body (Frame/Box)':<30}")
    print("-" * 115)
    
    for step in range(args.steps):
        if args.demo:
            t = step / 100.0
            action = np.array([0.1 * np.sin(t), 0.1 * np.sin(t * 1.5), 0.05 * np.sin(t * 0.5), 0.0], dtype=np.float32)
        else:
            action = np.zeros(4, dtype=np.float32)
        
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        if done or truncated:
            print(f"[Reset] at step {step}")
            env.reset()
        
        if step % 50 == 0:
            try:
                # 1. 机械臂 TCP 位置
                tcp = env.unwrapped.tcp_center
                
                # 2. 交互点位置 (Handle/Button Surface)
                # 这是 MetaWorld 任务逻辑里认为的“目标物体”
                site = env.unwrapped._get_pos_objects()
                
                # 3. 基座 Body 位置 (Window Frame/Button Box)
                # 这是我们新加的，读取刚体基座
                base = get_base_body_pos(env, args.env)
                
                # 格式化
                tcp_str = f"[{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}]"
                site_str = f"[{site[0]:.3f}, {site[1]:.3f}, {site[2]:.3f}]"
                
                if base is not None:
                    base_str = f"[{base[0]:.3f}, {base[1]:.3f}, {base[2]:.3f}]"
                else:
                    base_str = "N/A" # 有些任务没有基座（如 PickPlace）
                
                print(f"{step:<6} | {tcp_str:<30} | {site_str:<30} | {base_str:<30}")
                
            except AttributeError:
                pass

        time.sleep(0.01)

    print("\nVisualization complete.")
    env.close()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        run_simple_viewer(args)
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()