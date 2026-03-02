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
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Simple robot arm viewer')
    parser.add_argument('--robot', type=str, default='panda', 
                        choices=['sawyer', 'panda', 'kuka', 'ur5e', 'ur10e','unitree_z1','viperx', 'gen3','xarm7'],
                        help='Robot arm to visualize')
    parser.add_argument('--env', type=str, default='reach-v2', help='Environment to load')
    parser.add_argument('--camera', type=str, default='corner3', help='Camera view')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps')
    parser.add_argument('--demo', action='store_true', help='Run a simple movement demo')
    # Default initial position set to None for downstream logic
    parser.add_argument('--init-pos', type=float, nargs=3, default=[0, 0.6, 0.2], help='Initial hand pos (x y z)')
    return parser.parse_args()


def create_env(robot_type, env_name, camera_name):
    """Create the environment"""
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
    Infer the main object's Body name based on the environment name and get its coordinates.
    Different MetaWorld tasks have different base Body names.
    """
    body_name = None
    env_name = env_name.lower()
    
    # 1. Simple name mapping logic
    if 'window' in env_name:
        body_name = 'window'      # Window task base is called window
    elif 'button' in env_name:
        body_name = 'box'         # Button task base is called box
    elif 'drawer' in env_name:
        body_name = 'drawer'      # Drawer task base is called drawer
    elif 'door' in env_name:
        body_name = 'door'        # Door task base is called door
    elif 'peg' in env_name:
        body_name = 'box'         # Peg insertion task hole box is called box
    elif 'pick' in env_name or 'assembly' in env_name:
        # Pick Place has only an object, no base, or the base is the table
        return None

    # 2. Try reading from MuJoCo data
    if body_name:
        try:
            # env.unwrapped.data.body(name).xpos returns real-time world coordinates
            return env.unwrapped.data.body(body_name).xpos
        except Exception:
            return None
    return None


def run_simple_viewer(args):
    """Main loop logic"""
    env = create_env(args.robot, args.env, args.camera)
    
    # Inject initial position
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
    # Three-column data comparison: Robot hand | Interaction point (handle/button) | Object base (frame/box)
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
                # 1. Robot arm TCP position
                tcp = env.unwrapped.tcp_center
                
                # 2. Interaction point position (Handle/Button Surface)
                # This is the "target object" as defined by the MetaWorld task logic
                site = env.unwrapped._get_pos_objects()
                
                # 3. Base Body position (Window Frame/Button Box)
                # Newly added: read rigid body base position
                base = get_base_body_pos(env, args.env)
                
                # Format output
                tcp_str = f"[{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}]"
                site_str = f"[{site[0]:.3f}, {site[1]:.3f}, {site[2]:.3f}]"
                
                if base is not None:
                    base_str = f"[{base[0]:.3f}, {base[1]:.3f}, {base[2]:.3f}]"
                else:
                    base_str = "N/A" # Some tasks have no base (e.g. PickPlace)
                
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