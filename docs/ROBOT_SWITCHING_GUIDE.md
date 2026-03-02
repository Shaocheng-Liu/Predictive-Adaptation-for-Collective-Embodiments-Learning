# 动态机械臂切换方案 - Robot Arm Switching Solution

## 问题 (Problem)

当前工作流程需要手动编辑XML文件来切换机械臂，每个XML包含多个被注释的机械臂配置。这个过程：
- ❌ 效率低下
- ❌ 容易出错
- ❌ 难以自动化
- ❌ 不便于批量处理

Currently, you need to manually edit XML files to switch robot arms, with each XML containing multiple commented robot configurations. This process is:
- ❌ Inefficient
- ❌ Error-prone
- ❌ Hard to automate
- ❌ Inconvenient for batch processing

## 解决方案 (Solution)

### 方案1：自动XML生成器（推荐）

我创建了一个**机械臂配置管理器**，可以自动从多机械臂XML中提取并生成特定机械臂的XML文件。

I created a **Robot Configuration Manager** that automatically extracts and generates robot-specific XML files from multi-robot XMLs.

#### 使用方法 (Usage)

**1. 生成特定机械臂的XML配置**
```bash
# 为push-v2生成ur5e配置
cd Metaworld
python -m metaworld.envs.robot_config_manager --env push-v2 --robot ur5e

# 为reach-v2生成ur10e配置
python -m metaworld.envs.robot_config_manager --env reach-v2 --robot ur10e

# 生成所有常用环境的所有机械臂配置
python -m metaworld.envs.robot_config_manager
```

生成的文件将保存在：
```
Metaworld/metaworld/envs/assets_v2/<robot_type>/
├── ur5e/
│   ├── sawyer_push_v2.xml
│   ├── sawyer_reach_v2.xml
│   └── ...
├── ur10e/
│   ├── sawyer_push_v2.xml
│   └── ...
└── panda/
    └── ...
```

**2. 在代码中使用生成的XML**
```python
from metaworld.envs.robot_config_manager import get_robot_xml_path
import mujoco

# 自动获取或生成ur5e的push-v2 XML
xml_path = get_robot_xml_path("push-v2", robot_type="ur5e")

# 加载模型
model = mujoco.MjModel.from_xml_path(str(xml_path))
```

### 方案2：通过实验配置切换

在运行train_task, online_distill, evaluate_task等命令时，通过配置自动切换：

```bash
# 使用run.sh中的函数
source run.sh

# 训练UR5e专家
train_task ur5e reach-v2 100000

# 在线蒸馏
online_distill ur5e push-v2

# 评估
evaluate_task ur5e reach-v2
```

系统会自动：
1. 检测`experiment.robot_type`参数
2. 从多机械臂XML中提取对应的配置
3. 生成干净的单机械臂XML文件
4. 使用正确的XML创建环境

### 方案3：命令行动态切换

不修改任何文件，直接通过命令行指定：

```bash
python main.py \
  experiment.robot_type=ur5e \
  experiment.mode=train_worker \
  env.benchmark.env_name=push-v2
```

## 完整工作流程 (Complete Workflow)

### 场景：为UR5e准备训练环境

```bash
# 步骤1: 生成UR5e的XML配置（一次性）
cd Metaworld
python -m metaworld.envs.robot_config_manager --robot ur5e

# 步骤2: 训练专家
cd ..
source run.sh
train_task ur5e push-v2 100000
train_task ur5e reach-v2 100000

# 步骤3: 在线蒸馏
online_distill ur5e push-v2

# 步骤4: 评估
evaluate_task ur5e push-v2
```

### 场景：批量为多个机械臂准备配置

```bash
# 一次性生成所有配置
cd Metaworld
python -m metaworld.envs.robot_config_manager

# 输出：
# ✓ Generated reach-v2 for sawyer: .../sawyer/sawyer_reach_v2.xml
# ✓ Generated reach-v2 for kuka: .../kuka/sawyer_reach_v2.xml
# ✓ Generated reach-v2 for panda: .../panda/sawyer_reach_v2.xml
# ✓ Generated reach-v2 for ur5e: .../ur5e/sawyer_reach_v2.xml
# ✓ Generated reach-v2 for ur10e: .../ur10e/sawyer_reach_v2.xml
# ...
# ✓ Generated 25 configurations
```

## 代码示例 (Code Examples)

### 在Python中动态切换机械臂

```python
from metaworld.envs.robot_config_manager import RobotConfigManager, get_robot_xml_path
from metaworld import MT1

# 方法1: 使用便捷函数
xml_path_ur5e = get_robot_xml_path("push-v2", "ur5e")
xml_path_ur10e = get_robot_xml_path("push-v2", "ur10e")

# 方法2: 使用管理器类
manager = RobotConfigManager()

# 为所有环境生成ur5e配置
envs = ["reach-v2", "push-v2", "pick_place-v2"]
for env in envs:
    xml_path = manager.create_robot_specific_xml(env, "ur5e")
    print(f"Generated: {xml_path}")
```

## 技术实现 (Technical Implementation)

### XML提取逻辑

```python
# robot_config_manager.py 会：
# 1. 读取包含多个机械臂的XML文件
# 2. 通过正则表达式找到所有 <mujoco>...</mujoco> 块
# 3. 识别包含特定机械臂依赖的块（如 xyz_base_dependencies_ur5e.xml）
# 4. 移除注释标记，提取纯净的XML
# 5. 保存到 assets_v2/<robot_type>/ 目录
```

### 自动集成

在collective_experiment.py中自动进行：
```python
# 在 collective_experiment.__init__() 中
from metaworld.envs.robot_utils import set_robot_type, patch_all_metaworld_envs

# 检测robot_type并设置
if robot_type:
    set_robot_type(robot_type)
    patch_all_metaworld_envs()
```

## 优势对比 (Advantages)

| 特性 | 手动编辑XML | 自动生成方案 |
|------|------------|------------|
| 效率 | ❌ 慢 | ✅ 快 |
| 易错性 | ❌ 高 | ✅ 低 |
| 自动化 | ❌ 难 | ✅ 易 |
| 批量处理 | ❌ 不便 | ✅ 简单 |
| 版本控制 | ❌ 冲突多 | ✅ 清晰 |
| 可重现性 | ❌ 差 | ✅ 好 |

## 常见问题 (FAQ)

**Q: 生成的XML文件应该提交到git吗？**
A: 机械臂专用的XML文件已经保存在 `assets_v2/<robot_type>/` 目录下，可以提交。

**Q: 如果XML结构改变了怎么办？**
A: 重新运行生成命令即可：`python -m metaworld.envs.robot_config_manager --robot ur5e`

**Q: 可以自定义输出目录吗？**
A: 可以：`python -m metaworld.envs.robot_config_manager --robot ur5e --output-dir /path/to/dir`

**Q: 支持哪些机械臂？**
A: 目前支持：sawyer, kuka, panda, ur5e, ur10e。可以通过扩展 `ROBOT_DEPENDENCIES` 字典添加新的机械臂。

## 迁移指南 (Migration Guide)

如果你之前手动编辑XML，现在可以这样迁移：

**之前的工作流程：**
```bash
# 1. 手动打开 sawyer_push_v2.xml
# 2. 注释掉其他机械臂，取消注释ur5e部分
# 3. 保存
# 4. 运行训练
```

**现在的工作流程：**
```bash
# 1. 直接使用配置运行（XML会自动生成）
source run.sh
train_task ur5e push-v2 100000
```

**收益：**
- ⏱️ 节省时间：不需要每次手动编辑
- 🎯 减少错误：自动化避免人为失误
- 🔄 易于切换：改变命令行参数即可
- 📊 便于对比：可以快速测试不同机械臂

## 总结 (Summary)

这个解决方案让你可以：
1. ✅ **一键生成**所有机械臂配置
2. ✅ **自动选择**正确的XML文件
3. ✅ **轻松切换**不同机械臂
4. ✅ **批量处理**多个环境
5. ✅ **保持整洁**的版本控制

不再需要手动编辑XML文件！🎉
