[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/facebookresearch/mtrl/blob/main/LICENSE)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# PACE: Predictive Adaptation for Collective Embodiments Learning

## Contents

1. [Introduction](#introduction)
2. [Repository Structure](#repository-structure)
3. [Setup](#setup)
4. [Run the Code](#run-the-code)
5. [Acknowledgements](#acknowledgements)

## Introduction

Learning a unified manipulation policy that generalises across heterogeneous robotic embodiments is fundamentally difficult: lightweight collaborative arms and heavy industrial manipulators demand drastically different actuation efforts, yet standard behaviour cloning treats all data as i.i.d. and averages away these physical discrepancies. Collective learning—aggregating multi-robot data to discover shared structure—offers a promising direction, but still fails to reconcile the low-level dynamic gaps between embodiments unless the shared task semantics are explicitly separated from hardware-specific physics.

PACE addresses this challenge through a two-stage collective learning framework:

1. **Representation Layer** — A transformer encoder, pre-trained on offline trajectory data via masked reconstruction and contrastive learning, extracts an embodiment-agnostic task context.
2. **Decision Layer with Predictive Dynamics Adapter** — A predictive adapter forces the latent state to predict future transitions, acting as a latent regulariser that grounds the policy in the physical rules of the executing robot. This enables real-time adaptation without the computational overhead of explicit online planning.

Across simulated and real platforms, PACE prevents performance collapse on distinct embodiments, achieving over **93 % overall success** and **89 % on zero-shot robots** with out-of-distribution dynamic features. The trained policy further demonstrates robust zero-shot sim-to-real transfer when deployed on a physical KUKA robot.

### Framework
![img](/imgs/framework.png "PACE Framework")

### Datasets
![img](/imgs/robots.png "Datasets")

## Repository Structure

```
.
├── mtrl/                       # Core multi-task RL framework (PACE implementation)
│   ├── agent/                  # Policy networks and transformer-based agents
│   │   ├── components/         # Actor, critic, encoder, predictive adapter, etc.
│   │   └── transformer_agent.py  # TransformerAgent with predictive adapter integration
│   ├── experiment/             # Experiment lifecycle management
│   │   ├── collective_experiment.py  # Base experiment class (env setup, model init)
│   │   ├── collective_learning.py    # Training loops: worker, distillation, predictive adapter
│   │   └── collective_metaworld.py   # MetaWorld-specific evaluation logic
│   ├── env/                    # Environment wrappers
│   ├── utils/                  # Logging, checkpointing, config utilities
│   ├── replay_buffer.py        # Standard replay buffer
│   ├── col_replay_buffer.py    # Collective replay buffer for cross-embodiment data
│   └── transformer_replay_buffer.py  # Sequence-based buffer for transformer training
│
├── Metaworld/                  # Modified Meta-World benchmark
│   └── metaworld/              # Multi-task manipulation environments with support for
│                               # multiple robot embodiments (Sawyer, Panda, Kuka, UR5e,
│                               # UR10e, xArm7, Unitree Z1, Gen3, ViperX)
│
├── Transformer_RNN/            # Representation learning modules
│   ├── RepresentationTransformerWithCLS.py  # Transformer encoder with CLS token training
│   ├── dataset_tf.py           # Torch dataset creation from collected trajectories
│   ├── RNNEncoder.py           # RNN-based sequence encoder (baseline)
│   ├── InfoNceLoss.py          # Contrastive loss implementation (InfoNCE)
│   ├── cluster_properties_eval.py  # Latent space clustering evaluation
│   └── subsample_dataset.py    # Dataset subsampling utilities
│
├── config/                     # Hydra configuration files
│   ├── experiment/             # Experiment modes and hyperparameters
│   ├── transformer_collective_network/  # Collective network architecture configs
│   │   ├── components/         # Actor, critic, transformer encoder, predictive adapter
│   │   └── optimizers/         # Per-component optimizer settings
│   ├── worker/                 # Single-task worker agent configs
│   ├── env/                    # Environment configs (mt1, mt10)
│   └── metrics/                # Logging metric definitions
│
├── main.py                     # Entry point for all experiment modes
├── train.sh                    # Training pipeline script (source or execute)
├── eval.sh                     # Evaluation script (all/per-robot/per-task)
└── setup.py                    # Package installation
```

## Setup

### Requirements

- **Python 3.9**
- **CUDA 12.x** (for GPU training)
- **MuJoCo 2.3.7**

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Shaocheng-Liu/Robot-manipulation-learning-via-cross-embodiments.git
cd Robot-manipulation-learning-via-cross-embodiments

# 2. Create a Python 3.9 virtual environment
python3.9 -m venv venv
echo "export PROJECT_ROOT=$(pwd)" >> venv/bin/activate
source venv/bin/activate

# 3. Install core dependencies
pip install torch torchvision torchaudio
pip install mujoco==2.3.7
pip install numpy==1.23.5
pip install hydra-core==1.0.7 omegaconf==2.0.6
pip install gym==0.21.0 gymnasium==1.0.0a2
pip install opencv-python==4.5.5.64 Pillow==9.0.1
pip install tensorboard wandb
pip install scikit-learn==1.0.2 scipy==1.10.1 pandas
pip install PyYAML termcolor psutil imageio imageio-ffmpeg
pip install bnpy joblib==0.14.1
pip install glfw PyOpenGL pygame

# 4. Install the modified Meta-World environment
cd Metaworld
pip install -e .
cd ..

# 5. Install the mtenv environment manager
cd mtenv_repo
pip install -e .
pip install -e .[all]
cd ..

# 6. Install the mtrl package
pip install -e .
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.8+ | Deep learning backend |
| mujoco | 2.3.7 | Physics simulation |
| hydra-core | 1.0.7 | Configuration management |
| gym | 0.21.0 | Environment interface |
| gymnasium | 1.0.0a2 | Updated environment interface |
| numpy | 1.23.5 | Numerical computation |
| tensorboard / wandb | latest | Experiment logging |
| scikit-learn | 1.0.2 | Clustering and evaluation |

## Run the Code

### Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management. Key config files:

- **`config/experiment/collective_metaworld.yaml`** — Experiment modes and training hyperparameters. Supported modes:
  `train_worker` | `online_distill_collective_transformer` | `distill_collective_transformer` | `evaluate_collective_transformer` | `train_predictive_adapter` | `evaluate_predictive_adapter` | `generate_distill_data` | `record`

- **`config/env/metaworld-mt1.yaml`** or **`metaworld-mt10.yaml`** — Task selection.

- **`config/transformer_collective_network/`** — Architecture and optimizer settings for the collective network and predictive adapter.

#### Experiment Variables

Both `train.sh` and `eval.sh` support the following environment variables to control experiment-specific paths:

| Variable | Default | Description |
|----------|---------|-------------|
| `EXPERIMENT_NAME` | `"none"` | Experiment identifier. Controls model directory layout and auto-derives transformer checkpoint paths. |
| `SEED` | `1` | Random seed for reproducibility. |
| `TRANSFORMER_PATH` | (auto) | Override the representation transformer checkpoint path. When not set and `EXPERIMENT_NAME` is not `"none"`, it is automatically derived as `Transformer_RNN/checkpoints_${EXPERIMENT_NAME}_seed_${SEED}/representation_cls_transformer_checkpoint.pth`. |

When `EXPERIMENT_NAME` is set (e.g. `"my_exp"`), all model directories become experiment-aware:
- **Collective agent models**: `model_dir/${EXPERIMENT_NAME}/model_col_seed_${SEED}/`
- **Predictive adapter models**: `model_dir/${EXPERIMENT_NAME}/model_predictive_adapter_seed_${SEED}/`
- **Transformer checkpoints**: `Transformer_RNN/checkpoints_${EXPERIMENT_NAME}_seed_${SEED}/`

### Pipeline Overview

The full training pipeline follows these steps:

```
Step 1: Train Expert Workers
       ↓
Step 2: Online Distillation  (or)  Generate Noise-Injected Data
       ↓
Step 3: Train Predictive Adapter  &  Train Representation Transformer
       ↓
Step 4: Distill Collective Transformer
       ↓
Step 5: Evaluate
```

### Step-by-Step Execution

You can either **source `train.sh`** to load helper functions and call them individually, or run each step directly. To run a named experiment, set `EXPERIMENT_NAME` and `SEED` before calling:

```bash
export EXPERIMENT_NAME="my_exp"
export SEED=3
```

#### Step 1: Train Expert Workers

Train a single-task expert on each robot–task pair to collect experience data:

```bash
source train.sh

# Train an expert worker (robot_type, task_name, num_steps)
train_task sawyer reach-v2 200000
train_task sawyer push-v2 900000
train_task sawyer pick-place-v2 2400000
```

#### Step 2: Online Distillation (or Generate Noise-Injected Data)

Create the offline dataset for collective learning by distilling expert knowledge:

```bash
# Option A: Online distillation
online_distill sawyer reach-v2
online_distill sawyer push-v2

# Option B: Directly generate noise-injected data
generate_distill_data sawyer pick-place-v2 240000
```

After distillation, split the collected buffers for transformer training:

```bash
split_buffer sawyer reach-v2
split_online_buffer sawyer reach-v2
```

#### Step 3: Train Predictive Adapter & Representation Transformer

These two components can be trained independently:

```bash
# Train the predictive adapter on offline data
train_predictive_adapter

# (Or directly:)
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 \
    experiment.mode=train_predictive_adapter \
    experiment.experiment="${EXPERIMENT_NAME}" \
    setup.seed="${SEED}" \
    transformer_collective_network.predictive_adapter.load_on_init=False

# Train the representation transformer
python3 Transformer_RNN/dataset_tf.py          # Create torch dataset
python3 Transformer_RNN/RepresentationTransformerWithCLS.py  # Train transformer
```

#### Step 4: Distill Collective Transformer

Train the collective policy network using all collected data:

```bash
distill_collective

# (Or directly:)
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 \
    experiment.mode=distill_collective_transformer \
    experiment.experiment="${EXPERIMENT_NAME}" \
    setup.seed="${SEED}"
```

#### Step 5: Evaluate

Evaluate the trained collective policy on different robots and tasks:

```bash
# Evaluate all robots on all tasks (using default experiment)
bash eval.sh

# Evaluate a specific robot
bash eval.sh sawyer

# Evaluate a specific robot on a specific task
bash eval.sh sawyer reach-v2

# Evaluate with a named experiment and seed
EXPERIMENT_NAME="my_exp" SEED=3 bash eval.sh sawyer reach-v2

# Pass extra Hydra overrides
bash eval.sh sawyer reach-v2 setup.seed=5

# Override transformer checkpoint path explicitly
TRANSFORMER_PATH="/path/to/checkpoint.pth" bash eval.sh sawyer reach-v2
```


Supported robots: `sawyer`, `panda`, `kuka`, `ur5e`, `ur10e`, `xarm7`, `unitree_z1`, `gen3`, `viperx`.

## Acknowledgements

* Project structure inherited from [MTRL](https://mtrl.readthedocs.io/en/latest/index.html) library.

* Meta-World environments based on the [Meta-World](https://meta-world.github.io/) benchmark.

* Configuration management powered by [Hydra](https://github.com/facebookresearch/hydra).
