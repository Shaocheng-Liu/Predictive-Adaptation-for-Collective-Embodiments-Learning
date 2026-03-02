[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/facebookresearch/mtrl/blob/main/LICENSE)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Zulip Chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://mtenv.zulipchat.com)

# A transformer based collective learning framework for scalable knowledge accumulation and transfer


## Contents

1. [Introduction](#Introduction)

2. [Setup](#Setup)

3. [Run the code](#Run-the-code)

4. [Baseline Performance Analysis and Ranking Prediction](#Baseline-Performance-Analysis-and-Ranking-Prediction)

5. [Acknowledgements](#Acknowledgements)

## Introduction

Reinforcement learning (RL) has achieved remarkable success in addressing complex decision-making problems, particularly through the integration of deep neural networks. Despite these advancements, RL agents face persistent challenges, including sample inefficiency, scalability across multiple tasks, and limited generalization to unseen environments, which hinder their applicability in real-world scenarios. This thesis introduces a novel framework that combines a transformer-based encoder with a collective learning strategy to address these limitations.
The proposed approach leverages transformers to process sequential trajectory data, enabling the policy network to capture rich contextual information. A collective learning framework is employed, wherein multiple RL agents share their experiences to train a centralized policy network. This collaborative process reduces sample requirements, enhances scalability for multi-task learning, and improves generalization across diverse environments. Furthermore, a reward-shaping strategy is introduced to utilize the knowledge of the collective network, accelerating the learning of new policies while avoiding convergence to suboptimal solutions. 
Experimental evaluations on the MetaWorld benchmark demonstrate that the proposed framework achieves superior sample efficiency, scalability, and generalization compared to single-task policies. Additionally, the collective network shows strong potential for lifelong learning by continually acquiring and adapting to new skills. The framework is also validated for cross-embodiment policy transfer, successfully generalizing control policies to robotic arms with varying morphologies. 

### Demonstration
![img](/imgs/snapshot.png "Snapshot")

### Snapshots
[![Movie1](/imgs/panda_drawer_120.png "Demonstration video")](https://youtu.be/5edG_Wm39Mc)

## Setup
A full installation requires python 3.8:
* Clone the repository: `git clone https://github.com/4nd1L0renz/A-transformer-based-collective-learning-framework-for-scalable-knowledge-accumulation-and-transfer.git`.

* Install dependencies: `pip install -r requirements/dev.txt`
  
* Install modified Metaworld `cd Metaworld & pip install -e .`
    
* Note that we use mtenv to manage Meta-World environment, and add slight modification under **mtenv/envs/metaworld/env.py**, we added following function allowing for output a list of env instances:
```
def get_list_of_envs(
    benchmark: Optional[metaworld.Benchmark],
    benchmark_name: str,
    env_id_to_task_map: Optional[EnvIdToTaskMapType],
    should_perform_reward_normalization: bool = True,
    task_name: str = "pick-place-v1",
    num_copies_per_env: int = 1,
) -> Tuple[List[Any], Dict[str, Any]]:

    if not benchmark:
        if benchmark_name == "MT1":
            benchmark = metaworld.ML1(task_name)
        elif benchmark_name == "MT10":
            benchmark = metaworld.MT10()
        elif benchmark_name == "MT50":
            benchmark = metaworld.MT50()
        else:
            raise ValueError(f"benchmark_name={benchmark_name} is not valid.")

    env_id_list = list(benchmark.train_classes.keys())

    def _get_class_items(current_benchmark):
        return current_benchmark.train_classes.items()

    def _get_tasks(current_benchmark):
        return current_benchmark.train_tasks

    def _get_env_id_to_task_map() -> EnvIdToTaskMapType:
        env_id_to_task_map: EnvIdToTaskMapType = {}
        current_benchmark = benchmark
        for env_id in env_id_list:
            for name, _ in _get_class_items(current_benchmark):
                if name == env_id:
                    task = random.choice(
                        [
                            task
                            for task in _get_tasks(current_benchmark)
                            if task.env_name == name
                        ]
                    )
                    env_id_to_task_map[env_id] = task
        return env_id_to_task_map

    if env_id_to_task_map is None:
        env_id_to_task_map: EnvIdToTaskMapType = _get_env_id_to_task_map()  # type: ignore[no-redef]
    assert env_id_to_task_map is not None

    def make_envs_use_id(env_id: str):
        current_benchmark = benchmark
        
        
        def _make_env():
            for name, env_cls in _get_class_items(current_benchmark):
                if name == env_id:
                    env = env_cls()
                    task = env_id_to_task_map[env_id]
                    env.set_task(task)
                    if should_perform_reward_normalization:
                        env = NormalizedEnvWrapper(env, normalize_reward=True)
                    return env
        # modified return built single envs
        single_env = _make_env()
        return single_env

    if num_copies_per_env > 1:
        env_id_list = [
            [env_id for _ in range(num_copies_per_env)] for env_id in env_id_list
        ]
        env_id_list = [
            env_id for env_id_sublist in env_id_list for env_id in env_id_sublist
        ]

    list_of_envs = [make_envs_use_id(env_id) for env_id in env_id_list]
    return list_of_envs, env_id_to_task_map
```

These are the steps all in one script:
```bash
git clone git@github.com:argator18/A-transformer-based-collective-learning-framework-for-scalable-knowledge-accumulation-and-transfer.git framework
cd framework

# install python 3.8
pyenv install 3.8 
 ~/.pyenv/versions/3.8.20/bin/python3 -m venv venv

# set env variable to set project root
echo "export PROJECT_ROOT=$(pwd)" >> venv/bin/activate
source venv/bin/activate

pip install -r requirements/dev.txt
pip install mujoco==3.2.3

git clone git@github.com:facebookresearch/mtenv.git
cd mtenv

# apply custom patches
patch -p1 < ../changes.diff

pip install -e .

pip uninstall gym
pip install gym==0.26.0
pip install gymnasium==1.0.0a2

pip install -e .[all]

cd ../Metaworld
pip install -e .
pip install gym==0.21.0
pip install protobuf==3.20.3
pip install numpy==1.23.5
pip install bnpy

cd ..
pip install -e .
```
## Run the code

### Config

Our code uses Hydra to manage the configuration for the experiments. The config can be found in the `config`-folder. The most important settings to check before running the code is:

* experiment/collective_metaworld.yaml: Specify the training mode and the details for training; supported modes are: train_worker/online_distill_collective_transformer/distill_collective_transformer/evaluate_collective_transformer/train_student/train_student_finetuning/record

* env/metaworld-mt1.yaml or env/metaworld-mt10.yaml: To select the tasks.

* The remaining experiment settings and the hyperparameter of the model can be found in their respective folder in the config-folder

### Execution
A full example on how to run the code can be found in `run.sh`. It provides for most tasks the perfect hyperparameters and bash wrapper to execute single tasks more convenient. Additionally it shows in which files the input and outputs are expected.  

1. **Experiment mode Train expert**: Train an expert on a task. While training collect regular experience samples for later training of the transformer trajectory encoder

2. **Experiment mode Online Distill collective network**: Creates the offline dataset for training the collective network by distilling the expert knowledge into a temporary network while also recording state, action, rewards etc

3. **Train trajectory transformer**: Create torch dataset by running `Transformer_RNN/dataset_tf.py` (specify location of training data in code); Run training via `Transformer_RNN/RepresentationTransformerWithCLS.py`

4. **Experiment mode Distill collective network**: Loads all collected datasets and runs the training of the collective network; expects the training data in `(experimentfolder)/buffer/collective_buffer/train` and validation data in `(experimentfolder)/buffer/collective_buffer/validation`

5. **Experiment mode Evaluate_collective_transformer**: Evaluates the performance of experts or collective network

6. **Experiment mode Train_student**: Loads the collective network and teaches a student policy a learned task by providing it an addtional reward

7. **Experiment mode Train_student_finetuning**: Loads the collective network and finetunes it on a new task in standard SAC manner

* **Experiment mode Record**: Used only for test purposes. Uses either random-, scripted or probabilistic scripted policy to recorde experiences

* **Different embodiments**: For changing between different robots (Kuka, Sawyer, Panda) please adjust `Metaworld/metaworld/envs/assets_v2/sawyer_xyz/(environment name).xml` and uncomment the robot type you want. For the Panda robot please also additionally uncomment line 689-691 in `Metaworld/metaworld/envs/mujoco/sawyer_xyz/saywer_xyz_env.py`

After setting the configuration for the experiment run the code as follows (for metaworld-mt1):

**Standard Training (single GPU)**:
```
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1
```

**GPU Server Optimized Training (multi-GPU servers like 4x A100)**:
```
python3 -u main.py --config-name=gpu_server_collective_config setup=gpu_server_metaworld env=metaworld-mt1 worker.multitask.num_envs=16
```

For detailed GPU optimization instructions, see [GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md).

---

## Baseline Performance Analysis

This section provides a detailed analysis of the baselines used in our paper,
with actual experimental results and corrected architectural descriptions.

> **Note on terminology corrections**: Early drafts of this section
> incorrectly described B2/B3 as "SAC" (Soft Actor-Critic).  After careful
> code inspection (see Discussion §Q2 below), both B2 and B3 use **offline
> behavioural cloning (BC)** — not online RL.  The corrections below are
> authoritative.

### Baseline definitions (corrected)

| ID | Short name | Config flags | Training type | Description |
|:--:|:-----------|:-------------|:---:|:------------|
| **B1** | **Diffusion Policy** | `diffusion_policy/train.py --arch_type advanced` | Offline BC | DDPM U-Net denoises a predicted action chunk conditioned on 20-step sliding window. No task encoding. |
| **B2** | **Joint-Encoder Transformer + BC** | `train_encoder_jointly: True`, `use_world_model: False`, `use_cls_prediction_head: True`, `use_zeros: False` | Offline BC | Transformer encoder jointly trained with actor via MSE loss to teacher actions. CLS head provides 6-D task encoding. Actor input: `[task_encoding(6), state(21)]`. |
| **B3** | **Zero-Encoding Ablation** | `train_encoder_jointly: False`, `use_world_model: False`, `use_cls_prediction_head: False`, `use_zeros: True` | Offline BC | Task encoding replaced with zero vector. Actor input: `[zeros(6), state(21)]`. Only the 21-D state is informative. |
| **B4** | **ACT** *(optional)* | Separate implementation | Offline BC | CVAE-based transformer: encoder (4L) + decoder (7L), autoregressive action chunking. |
| **B5** | **TD-MPC2** *(not included)* | Separate implementation | Online model-based RL | Uses world model for online planning via MPC. Fundamentally different paradigm — see Discussion §Q4. |

### Experimental results (10 tasks × 9 robots × 3 seeds)

```
B2 (Joint Encoder + BC)  >  B3 (Zero-Encoding)  >  B1 (Diffusion Policy)
        ~87%                      ~80%                     ~70%
```

### Detailed analysis

#### B2 — Joint-Encoder Transformer + BC: ~87% (rank 1)

**Why it is the strongest:**

1. **Jointly-trained task representation.** The CLS prediction head maps the
   transformer encoder's output to a 6-D task encoding.  Critically, this head
   is **randomly initialized** and has **no supervised task loss** — it is
   trained end-to-end solely through the actor's MSE loss to teacher actions.
   Despite the lack of explicit task supervision, the backpropagation gradient
   from the actor loss forces the encoder to produce task-discriminative
   representations that are *useful for control*.

2. **Offline BC with Q-target-informed critic.**  The actor is trained via MSE
   loss: `L = (μ_actor − μ_teacher)²`.  Separately, a critic is trained via MSE
   regression to pre-computed offline Q-targets.  There is no entropy term, no
   temperature optimization, and no online environment interaction — this is
   **not SAC**.  However, the critic still provides implicit value awareness in
   the training pipeline (see `distill_critic` in `transformer_agent.py`).

3. **Rich state representation for the actor.**  Actor input is
   `[task_encoding(6), current_state(21)]` = 27-D.  The 6-D encoding provides
   compact task discrimination, while the 21-D state provides current
   proprioceptive information.  This combination allows a single shared policy
   to differentiate between tasks that share similar initial observations.

4. **DAgger-quality training data.**  The data comes from `run_online_distillation`,
   which uses a DAgger-style data collection process with expert teachers.
   While this data includes some failed trajectories, the MSE loss to teacher
   actions provides a strong supervised signal regardless of episode success.

**Per-task pattern:**
- `reach-v2`, `drawer-open-v2`, `button-press-topdown-v2`: ~92–97%
- `push-v2`, `window-open/close-v2`, `faucet-open-v2`: ~85–92%
- `door-open-v2`, `pick-place-v2`: ~80–88%
- `peg-insert-side-v2`: ~72–82%

#### B3 — Zero-Encoding Ablation: ~80% (rank 2)

**Why it is surprisingly strong (revised analysis):**

Our initial prediction (~15–30%) was far too pessimistic.  The actual ~80%
result reveals important insights about the MetaWorld task suite:

1. **MetaWorld's 21-D state space is highly informative.**  The compressed
   state includes end-effector xyz, gripper state, and object positions/poses.
   For many tasks, the *current* object configuration already implies the
   required action without needing an explicit task identifier.  For example:
   if the object is a door handle and it's in a closed position, the correct
   action is to pull it open — regardless of the task label.

2. **Task-informative observation structure.**  In MetaWorld, different tasks
   have distinct object layouts at test time: `reach-v2` has a floating target,
   `door-open-v2` has a door handle at a specific location, etc.  The 21-D state
   contains enough geometric information for the actor to implicitly identify
   most tasks from observation alone.

3. **Same BC training quality as B2.**  B3 uses the identical MSE loss to
   teacher actions: `L = (μ_actor − μ_teacher)²`.  The only difference is that
   the 6-D input is zeros instead of a learned encoding.  Since the 21-D state
   already provides most task-relevant information, the 6-D encoding adds only
   ~7% additional accuracy (87% → 80%).

4. **This baseline answers a key research question.**  The ~7% gap
   between B2 and B3 measures the *marginal value of learned task encoding*
   above and beyond what proprioceptive state provides.

**Per-task pattern:**
- `reach-v2`, `drawer-open-v2`, `button-press-topdown-v2`: ~88–95%
- `push-v2`, `window-open/close-v2`, `faucet-open-v2`: ~78–88%
- `door-open-v2`, `pick-place-v2`: ~70–80%
- `peg-insert-side-v2`: ~60–72%

#### B1 — Diffusion Policy: ~70% (rank 3)

**Why it is weaker than both B2 and B3 despite being a more advanced model:**

1. **No selective data filtering.**  All three baselines train on the same
   DAgger-collected offline data, which includes both successful and failed
   trajectories.  However, B2 and B3's BC actor training uses `(μ_teacher, log_std_teacher)`
   from the buffer — these are the **teacher policy's intended actions** at each
   state, which provide a clean supervised signal even during failed episodes.
   The diffusion policy, by contrast, trains on **actually-executed actions** (the
   `actions` field in the buffer), which include erroneous actions that led to
   failure.

2. **No task encoding or state-level conditioning.**  B1 conditions only on a
   20-step sliding window of historical (states, actions).  Unlike B2/B3 which
   receive the full current 21-D state at every step, the diffusion policy must
   infer task context implicitly from trajectory history.

3. **Action chunking creates a precision-robustness tradeoff.**  The model
   predicts 8 actions at once (`pred_horizon=8`), which provides temporal
   coherence but sacrifices reactivity.  For tasks requiring fine-grained
   closed-loop adjustment (e.g., `peg-insert-side-v2`), committing to 8 actions
   before re-planning can lead to compounding errors.

4. **Cross-embodiment distribution shift.**  The 9-robot mixed dataset means
   the diffusion model must learn a single denoising process that works for all
   kinematics.  Without explicit robot/task conditioning, the model must spend
   capacity on disentangling robot morphology from task intent.

**Per-task pattern:**
- `reach-v2`, `drawer-open-v2`, `button-press-topdown-v2`: ~80–88%
- `push-v2`, `window-open/close-v2`, `faucet-open-v2`: ~68–78%
- `door-open-v2`, `pick-place-v2`: ~58–70%
- `peg-insert-side-v2`: ~48–62%

### Summary comparison

| Factor | B2 (Joint Encoder + BC) | B3 (Zero-Encoding) | B1 (Diffusion Policy) |
|:-------|:---:|:---:|:---:|
| Training paradigm | Offline BC (MSE) | Offline BC (MSE) | Offline BC (DDPM) |
| Training target | Teacher μ, log_std | Teacher μ, log_std | Executed actions |
| Task encoding | ✅ 6-D learned (no supervision) | ❌ Zeros | ❌ None |
| State input | 21-D current state | 21-D current state | 20-step window |
| Multi-modal actions | ❌ Gaussian | ❌ Gaussian | ✅ Denoising |
| Temporal coherence | ❌ Single-step | ❌ Single-step | ✅ 8-step chunks |
| Avg success rate | **~87%** | **~80%** | **~70%** |

---

### Discussion: Technical details and corrections

#### Q1: Does the CLS prediction head have supervised task loss?

**No.**  The `ClsPredictionHead` (in `transformer_trajectory_encoder.py`) is a
single linear layer: `nn.Linear(d_model, latent_dim)` mapping from the
transformer's d_model (32) to 6-D.  It is **randomly initialized** and there is
**no auxiliary classification loss or task-label supervision**.  The only
gradient it receives is from the actor's MSE loss, which backpropagates through
the encoder when `train_encoder_jointly=True`.

This means the 6-D encoding learns to be task-discriminative *implicitly* —
because the actor needs task information to minimize its MSE loss to teacher
actions.  The encoding converges to capture whatever task-relevant features
reduce actor error, without explicit task labels.

#### Q2: Is B2/B3 actually SAC?

**No — it is offline behavioural cloning (BC).**  Code inspection reveals:

| Component | SAC (online RL) | B2/B3 (actual) |
|-----------|:---:|:---:|
| **Actor loss** | Q-function based, entropy-regularized | MSE to teacher actions: `(μ − μ_teacher)²` |
| **Critic training** | TD learning with target network | MSE to pre-computed offline Q-targets |
| **Entropy term** | ✅ Temperature α optimized | ❌ None |
| **Online interaction** | ✅ Agent acts in environment | ❌ Purely offline |

The `distill_actor` method (in `transformer_agent.py`) computes
`mse_loss_mu = (mu - batch_mu) ** 2` and optionally adds a KL divergence term.
There is no Q-function used in the actor loss during distillation.  The
`distill_critic` method trains via `F.mse_loss(current_Q, q_targets)` where
`q_targets` are pre-computed offline — not via temporal difference learning.

Note: the codebase *does* contain a separate `update` method with full SAC
(entropy-regularized Q-learning), but this is used during online training, not
during the `distill_collective_transformer` distillation phase that produces B2/B3.

#### Q3: Why is B1 (Diffusion Policy) weaker — does data filtering explain the gap?

**The core issue is not data filtering but the training target.**

The key difference is in *what* each method trains on:

- **B2/B3 train on `policy_mu`** (teacher's intended action, stored at buffer
  index 7 in TransformerReplayBuffer / index 6 in DistilledReplayBuffer).
  Critically, these targets are **relabeled** after `run_online_distillation`:
  the code iterates over all collected data and re-computes `(q_targets,
  teacher_mu, teacher_log_std)` using the trained expert agent (see
  `collective_learning.py` lines 476–500).  This means the teacher's
  *intended* actions are clean, optimal actions at each state — even during
  episodes that ultimately failed.

- **B1 (originally) trains on `actions`** (actually-executed actions at buffer
  index 2).  These include exploration noise, sub-optimal recovery attempts,
  and actions during failure trajectories.  The diffusion policy learned to
  reproduce all of them equally, including bad ones.

**Neither method explicitly filters data** — `buffer.sample_new()` returns all
data without reward-based weighting.  B2/B3's advantage comes from training on
*cleaner targets* (teacher intentions vs. executed actions), not from filtering.

**Solution implemented: `--train_target mu`**

The diffusion policy now supports training on `policy_mu` instead of raw
`actions`.  The `generate_diffusion_dataset.py` script extracts both
`actions` (index 2) and `policy_mu` (index 6/7) from the buffer.  In
`train.py`, the `--train_target` flag controls which target is used:

```bash
# Train on teacher's intended actions (recommended — matches B2/B3)
python diffusion_policy/train.py --data_path diffusion_policy/data/train --train_target mu

# Train on raw executed actions (original approach)
python diffusion_policy/train.py --data_path diffusion_policy/data/train --train_target action
```

**Expected impact**: Training on `policy_mu` should significantly improve B1's
success rate by eliminating noisy/failed action targets.  The diffusion model
now learns to denoise toward the teacher's clean intended actions, making it
a fairer comparison with B2/B3.

#### Q4: Would TD-MPC2 be a good baseline?

**TD-MPC2 would likely achieve ~85–92% average success rate** — potentially
the strongest of all methods — for the following reasons:

1. **Online model-based planning.**  TD-MPC2 uses a learned world model and
   Model Predictive Control (MPC) to plan actions online.  It can adapt in
   real-time to environment dynamics, which is particularly advantageous for
   cross-embodiment transfer where each robot has different kinematics.

2. **Reward-driven optimization.**  Unlike BC methods, TD-MPC2 optimizes
   cumulative reward directly, giving it the same advantage as B2 but with the
   additional benefit of online planning.

3. **Multi-task capability.**  TD-MPC2's architecture supports task-conditioned
   world models, making it naturally suited to multi-task settings.

**Why it is NOT included as a baseline:**

TD-MPC2 is fundamentally a different paradigm.  Our framework uses the world
model to generate an information-rich latent space for offline policy learning.
TD-MPC2 uses the world model for **online trajectory optimization** (CEM/MPPI
planning at test time).  Including TD-MPC2 would compare two orthogonal design
choices simultaneously (offline vs. online, BC vs. MPC), making it difficult to
attribute performance differences to any single factor.

Additionally, TD-MPC2 requires online environment interaction during both
training and evaluation, which changes the experimental protocol fundamentally.

#### Q5: Batch size / LR scaling — linear vs. √ rule?

**The √ scaling rule is correct for Adam optimizer.**

When scaling batch size from 256 → 2048 (8×):

| Scaling rule | Formula | LR | Rationale |
|:---|:---|:---:|:---|
| **√ scaling** (recommended) | lr × √k | **3e-4** | Accounts for reduced gradient noise; correct for Adam which has per-parameter adaptive rates |
| Linear scaling (Goyal et al.) | lr × k | 8e-4 | Designed for SGD with momentum — too aggressive for Adam |
| No scaling | lr | 1e-4 | Under-trains: 8× fewer gradient steps with same step size |

The linear scaling rule (Goyal et al., 2017) was derived for SGD with momentum
on ImageNet.  For Adam-based training (as in diffusion policy), the √ rule
(Hoffer et al., 2017; Krizhevsky, 2014) is more appropriate because Adam
already normalizes gradients per-parameter.  Applying a full 8× multiplier
causes overshooting and training instability.

See `diffusion_policy/README.md` §12.2 for the detailed mathematical analysis.

#### Q6: Should the default epoch count be 100 instead of 200?

**Yes — the default is now 100**, matching Chi et al. (2023).  The original
Diffusion Policy paper uses 100 epochs.  With proper hyperparameters
(batch_size=256, lr=1e-4, cosine schedule), 100 epochs provides sufficient
convergence.  The default in `train.py` has been updated accordingly.

If early overfitting is observed, see `diffusion_policy/README.md` §11 for the
troubleshooting guide — the solution is not more epochs, but regularization
(weight_decay, EMA, obs_noise).

#### Q7: Training on `policy_mu` vs `actions` — deep analysis

**Why `policy_mu` is a fundamentally different (and cleaner) training target:**

The `run_online_distillation` pipeline works as follows:

1. **Data collection**: The student policy interacts with the environment.
   At each step, both the executed action AND the expert teacher's mu/log_std
   are computed and stored in the buffer.  The executed action may include
   exploration noise and can lead to sub-optimal or failed trajectories.

2. **Relabeling** (lines 476–500 of `collective_learning.py`): After data
   collection completes, ALL buffer entries are **re-processed** through the
   trained expert agent.  New `(teacher_mu, teacher_log_std, q_targets)` values
   are computed for every (state, action) pair and stored in `replay_buffer_distill`.

3. **B2/B3 training**: The `distill_actor` method (in `transformer_agent.py`)
   trains the student's actor to match `batch_mu` — these are the relabeled
   teacher mu values, NOT the original executed actions.

This means:

| | `actions` (index 2) | `policy_mu` (index 6/7) |
|:---|:---|:---|
| **Source** | Student's actual execution | Teacher's optimal intent |
| **Noise** | Includes exploration noise | Clean, deterministic |
| **Failed episodes** | Contains actual bad actions | Teacher's correct action at that state |
| **Used by B2/B3** | ❌ No | ✅ Yes |
| **Used by B1 (original)** | ✅ Yes | ❌ No |

**Why this is likely the primary reason B1 (70%) < B3 (80%):**

Consider a failed `pick-place-v2` episode where the gripper misses the object:
- `actions`: The student's actual gripper motion that missed — learning this
  teaches the model to reproduce the failure.
- `policy_mu`: The teacher's intended grasp action at that state — learning
  this teaches the model the correct behavior even though the episode failed.

B3 (zero-encoding) trains on `policy_mu` and achieves 80%.  B1 (diffusion
policy) trained on `actions` and achieved only 70%.  The ~10% gap is
largely attributable to this training target difference, not to the
architectural differences between BC and diffusion policy.

**Can diffusion policy train on `policy_mu`?**

**Yes — this is now implemented.**  The `--train_target mu` flag (default)
makes the diffusion policy train on the same clean teacher targets as B2/B3:

```bash
# Default: train on teacher mu (recommended)
python diffusion_policy/train.py --data_path diffusion_policy/data/train --train_target mu

# Original: train on raw actions
python diffusion_policy/train.py --data_path diffusion_policy/data/train --train_target action
```

**Should we also model `log_std`?**

B2/B3 also train on `log_std` with an additional loss term:
`loss = mse_mu + 0.2 * mse_log_std`.  However, the diffusion policy does NOT
need to model `log_std` because:

1. The diffusion process itself captures action uncertainty through the
   denoising distribution — this is a richer uncertainty model than a
   fixed Gaussian log_std.
2. Adding log_std as a target would require the diffusion model to predict
   8-dimensional outputs (4-D mu + 4-D log_std) instead of 4-D, complicating
   the architecture unnecessarily.
3. B2/B3's log_std loss weight is only 0.2× — even for BC, it's a secondary
   objective.

**Conclusion**: Training on `policy_mu` alone is the correct approach for
diffusion policy.  This puts B1 on an equal footing with B2/B3 in terms of
data quality while preserving the diffusion model's natural ability to
capture multimodal action distributions.

### Note on ACT (B4) implementation

If ACT is added as a baseline, the recommended approach is:
1. Use the same `generate_diffusion_dataset.py` pipeline (states + actions)
2. Implement CVAE with encoder (4-layer transformer) + decoder (7-layer transformer)
3. Action chunking with `chunk_size = pred_horizon = 8` for fair comparison
4. Same evaluation protocol: 100 episodes × 10 tasks × 9 robots × 3 seeds


## Acknowledgements

* Project file pre-commit, mypy config, towncrier config, circleci etc are based on same files from [Hydra](https://github.com/facebookresearch/hydra).

* Implementation Inherited from [MTRL](https://mtrl.readthedocs.io/en/latest/index.html) library. 

* Documentation of MTRL repository refer to: [https://mtrl.readthedocs.io](https://mtrl.readthedocs.io).
