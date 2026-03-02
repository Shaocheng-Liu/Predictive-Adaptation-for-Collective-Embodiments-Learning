#!/usr/bin/env python3
"""
2K High-Definition Robot Renderer (No Sites / Red Dots)
修复：除了隐藏环境物体，额外强制隐藏所有 Site（红点、目标标记、辅助点）。
"""

import os
import numpy as np
import imageio
import metaworld
from metaworld.envs.robot_utils import set_robot_type, patch_all_metaworld_envs
import mujoco 

# === 核心配置 ===
OUTPUT_DIR = "paper_assets_clean_final"
RENDER_WIDTH = 2560
RENDER_HEIGHT = 1440
CAMERA_NAME = "corner2"
ROBOT_LIST = ['sawyer', 'gen3', 'unitree_z1', 'ur10e', 'xarm7', 'panda', 'kuka', 'ur5e', 'viperx']

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def hide_visuals(raw_env):
    """
    全方位隐身术：
    1. 黑名单隐藏环境 Geom (桌子、地板)
    2. 无差别隐藏所有 Site (红点、目标点)
    """
    model = raw_env.model
    
    # -------------------------------------------------
    # 步骤 1: 隐藏所有 Site (标记点/红点)
    # -------------------------------------------------
    # Reach 任务里的红点通常是 Site，直接全部设为透明最安全
    # 机械臂本体通常由 Geom 构成，隐藏 Site 不会影响机械臂外观
    if model.nsite > 0:
        model.site_rgba[:, 3] = 0.0
        print(f"   [标记清理] 已隐藏 {model.nsite} 个 Sites (包含红点/目标)")

    # -------------------------------------------------
    # 步骤 2: 黑名单隐藏环境 Geom (物理实体)
    # -------------------------------------------------
    env_keywords = [
        'table', 'desk', 'floor', 'ground', 'wall',
        'target', 'goal', 'site', 'obj', 'box', 'sphere', 'cube', 'cylinder', 'mocap'
    ]

    hidden_geom_count = 0
    
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        body_id = model.geom_bodyid[i]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        
        full_name = ""
        if geom_name: full_name += geom_name.lower()
        if body_name: full_name += " " + body_name.lower()

        should_hide = False
        for kw in env_keywords:
            if kw in full_name:
                should_hide = True
                break
        
        if should_hide:
            model.geom_rgba[i, 3] = 0.0
            hidden_geom_count += 1

    print(f"   [环境清理] 已隐藏 {hidden_geom_count} 个环境物体")

def get_high_res_image(env, camera_name):
    """渲染流程"""
    raw_env = env.unwrapped
    if hasattr(raw_env, 'model') and hasattr(raw_env, 'data'):
        try:
            # 执行清理
            hide_visuals(raw_env)
            
            # 设置分辨率
            raw_env.model.vis.global_.offwidth = RENDER_WIDTH
            raw_env.model.vis.global_.offheight = RENDER_HEIGHT
            
            # 渲染
            renderer = mujoco.Renderer(raw_env.model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
            renderer.update_scene(raw_env.data, camera=camera_name)
            img = renderer.render()
            img = np.flipud(img) 
            return img
        except Exception as e:
            print(f"渲染出错: {e}")
            return None
    return None

def main():
    ensure_dir(OUTPUT_DIR)
    print(f"开始生成无红点纯净版...\n")

    for robot_name in ROBOT_LIST:
        print(f"[{robot_name}] 正在处理...")
        try:
            set_robot_type(robot_name)
            patch_all_metaworld_envs()
            
            ml1 = metaworld.ML1('reach-v2', seed=42)
            env = ml1.train_classes['reach-v2'](camera_name=CAMERA_NAME)
            task = ml1.train_tasks[0]
            env.set_task(task)
            env.reset()
            
            # 动作预热
            for _ in range(15):
                env.step(np.array([0.1, 0.05, 0.05, 0.0], dtype=np.float32))
            
            img = get_high_res_image(env, CAMERA_NAME)
            
            if img is not None:
                save_path = os.path.join(OUTPUT_DIR, f"{robot_name}_clean.png")
                imageio.imwrite(save_path, img)
                print(f"   ---> 保存成功: {save_path} ✅")
            else:
                print("   ---> 渲染失败 ❌")
            env.close()
        except Exception as e:
            print(f"   ---> 处理异常: {e}")

if __name__ == '__main__':
    main()