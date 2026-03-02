#!/usr/bin/env python3
"""
示例脚本：使用不同视角录制视频

这个脚本展示了如何使用新的视角参数来录制机械臂评估的视频。
可以直接在 RLCE 文件夹中运行 record_videos 和 record_videos_for_transformer 方法时调用。
"""

# 示例 1: 在 Python 代码中使用不同的视角

# 导入必要的模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mtrl.experiment.collective_learning import Experiment
import hydra
from omegaconf import OmegaConf

# 定义不同的视角配置
CAMERA_VIEWS = {
    "default": {
        "azimuth": 140.0,
        "elevation": -40.0,
        "lookat": [0, 0.5, 0.0],
        "distance": 2.0,
        "description": "右上方视角 (Upper-right)"
    },
    "left": {
        "azimuth": 90.0,
        "elevation": -40.0,
        "lookat": [0, 0.5, 0.0],
        "distance": 2.0,
        "description": "正左侧视角 (Left side)"
    },
    "right": {
        "azimuth": 270.0,
        "elevation": -40.0,
        "lookat": [0, 0.5, 0.0],
        "distance": 2.0,
        "description": "正右侧视角 (Right side)"
    },
    "front": {
        "azimuth": 0.0,
        "elevation": -40.0,
        "lookat": [0, 0.5, 0.0],
        "distance": 2.0,
        "description": "正前方视角 (Front)"
    },
    "back": {
        "azimuth": 180.0,
        "elevation": -40.0,
        "lookat": [0, 0.5, 0.0],
        "distance": 2.0,
        "description": "正后方视角 (Back)"
    },
    "top": {
        "azimuth": 0.0,
        "elevation": 60.0,
        "lookat": [0, 0.5, 0.0],
        "distance": 2.0,
        "description": "俯视图 (Top-down)"
    },
    "close": {
        "azimuth": 90.0,
        "elevation": -40.0,
        "lookat": [0, 0.5, 0.0],
        "distance": 1.0,
        "description": "更近的左侧视角 (Closer left)"
    },
}


def record_videos_with_multiple_views(experiment, agent_type="collective_network", num_samples=1):
    """
    使用多个视角录制视频的示例函数
    
    Args:
        experiment: Experiment 实例
        agent_type: 代理类型 ("collective_network", "agent", "scripted")
        num_samples: 每个视角的样本数
    """
    print("=" * 80)
    print("开始使用多个视角录制视频")
    print("=" * 80)
    
    for view_name, view_config in CAMERA_VIEWS.items():
        print(f"\n录制 {view_config['description']} ({view_name})...")
        print(f"  方位角: {view_config['azimuth']}°")
        print(f"  仰角: {view_config['elevation']}°")
        print(f"  距离: {view_config['distance']}")
        
        for i in range(num_samples):
            if agent_type == "collective_network":
                experiment.record_videos_for_transformer(
                    experiment.col_agent,
                    tag=f"{view_name}_{i}",
                    seq_len=experiment.seq_len,
                    camera_azimuth=view_config["azimuth"],
                    camera_elevation=view_config["elevation"],
                    camera_lookat=view_config["lookat"],
                    camera_distance=view_config["distance"]
                )
            else:
                experiment.record_videos(
                    eval_agent="worker" if agent_type == "agent" else "scripted",
                    tag=f"{view_name}_{i}",
                    camera_azimuth=view_config["azimuth"],
                    camera_elevation=view_config["elevation"],
                    camera_lookat=view_config["lookat"],
                    camera_distance=view_config["distance"]
                )
        
        print(f"✓ {view_config['description']} 录制完成")
    
    print("\n" + "=" * 80)
    print("所有视角录制完成！")
    print(f"视频保存位置: {experiment.video.dir_name}")
    print("=" * 80)


def record_custom_view(experiment, agent_type="collective_network", **camera_params):
    """
    使用自定义视角录制视频的示例函数
    
    Args:
        experiment: Experiment 实例
        agent_type: 代理类型
        **camera_params: 相机参数 (camera_azimuth, camera_elevation, camera_lookat, camera_distance)
    """
    print(f"使用自定义视角录制视频...")
    print(f"  相机参数: {camera_params}")
    
    if agent_type == "collective_network":
        experiment.record_videos_for_transformer(
            experiment.col_agent,
            tag="custom_view",
            seq_len=experiment.seq_len,
            **camera_params
        )
    else:
        experiment.record_videos(
            eval_agent="worker",
            tag="custom_view",
            **camera_params
        )
    
    print("✓ 自定义视角录制完成")


# 使用示例
if __name__ == "__main__":
    """
    使用这个脚本的方法:
    
    1. 直接运行 (使用默认配置):
       python record_with_custom_view.py
    
    2. 与 main.py 结合使用时，可以修改 evaluate_collective_transformer 方法:
       
       # 在 collective_learning.py 中的 evaluate_collective_transformer 方法中:
       if self.config.experiment.save_video:
           # 使用多个视角录制
           record_videos_with_multiple_views(
               self, 
               agent_type="collective_network",
               num_samples=1
           )
    
    3. 使用自定义视角:
       exp.record_videos_for_transformer(
           exp.col_agent,
           tag="my_view",
           camera_azimuth=90.0,           # 左侧
           camera_elevation=-30.0,        # 稍微向下
           camera_distance=1.5            # 更近一些
       )
    """
    
    print("这是一个示例脚本，展示如何使用不同的视角录制视频。")
    print("\n关键参数说明:")
    print("  camera_azimuth: 方位角 (0=前, 90=左, 180=后, 270=右)")
    print("  camera_elevation: 仰角 (-90=下, 0=水平, 90=上)")
    print("  camera_lookat: 相机看向的点 [x, y, z]")
    print("  camera_distance: 相机到中心的距离")
    print("\n查看 VIDEO_RECORDING_GUIDE.md 了解更多详情。")
