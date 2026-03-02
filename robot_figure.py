#!/usr/bin/env python3
"""
2K High-Definition Robot Renderer (No Sites / Red Dots)
Fix: In addition to hiding environment objects, forcibly hide all Sites (red dots, target markers, auxiliary points).
"""

import os
import numpy as np
import imageio
import metaworld
from metaworld.envs.robot_utils import set_robot_type, patch_all_metaworld_envs
import mujoco 

# === Core Configuration ===
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
    Comprehensive visual cleanup:
    1. Blacklist-hide environment Geoms (table, floor)
    2. Indiscriminately hide all Sites (red dots, target points)
    """
    model = raw_env.model
    
    # -------------------------------------------------
    # Step 1: Hide all Sites (marker points / red dots)
    # -------------------------------------------------
    # Red dots in the Reach task are typically Sites; hiding them all is the safest approach.
    # Robot arms are usually composed of Geoms, so hiding Sites won't affect arm appearance.
    if model.nsite > 0:
        model.site_rgba[:, 3] = 0.0
        print(f"   [Marker cleanup] Hidden {model.nsite} Sites (including red dots/targets)")

    # -------------------------------------------------
    # Step 2: Blacklist-hide environment Geoms (physical entities)
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

    print(f"   [Environment cleanup] Hidden {hidden_geom_count} environment objects")

def get_high_res_image(env, camera_name):
    """Rendering pipeline"""
    raw_env = env.unwrapped
    if hasattr(raw_env, 'model') and hasattr(raw_env, 'data'):
        try:
            # Execute cleanup
            hide_visuals(raw_env)
            
            # Set resolution
            raw_env.model.vis.global_.offwidth = RENDER_WIDTH
            raw_env.model.vis.global_.offheight = RENDER_HEIGHT
            
            # Render
            renderer = mujoco.Renderer(raw_env.model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
            renderer.update_scene(raw_env.data, camera=camera_name)
            img = renderer.render()
            img = np.flipud(img) 
            return img
        except Exception as e:
            print(f"Rendering error: {e}")
            return None
    return None

def main():
    ensure_dir(OUTPUT_DIR)
    print(f"Starting clean render (no red dots)...\n")

    for robot_name in ROBOT_LIST:
        print(f"[{robot_name}] Processing...")
        try:
            set_robot_type(robot_name)
            patch_all_metaworld_envs()
            
            ml1 = metaworld.ML1('reach-v2', seed=42)
            env = ml1.train_classes['reach-v2'](camera_name=CAMERA_NAME)
            task = ml1.train_tasks[0]
            env.set_task(task)
            env.reset()
            
            # Warm-up actions
            for _ in range(15):
                env.step(np.array([0.1, 0.05, 0.05, 0.0], dtype=np.float32))
            
            img = get_high_res_image(env, CAMERA_NAME)
            
            if img is not None:
                save_path = os.path.join(OUTPUT_DIR, f"{robot_name}_clean.png")
                imageio.imwrite(save_path, img)
                print(f"   ---> Saved successfully: {save_path} ✅")
            else:
                print("   ---> Rendering failed ❌")
            env.close()
        except Exception as e:
            print(f"   ---> Processing error: {e}")

if __name__ == '__main__':
    main()