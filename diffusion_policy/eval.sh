#!/usr/bin/env bash
# ============================================================================
# Diffusion Policy — Parallel Evaluation Script
#
# Evaluates the diffusion policy across multiple robots and tasks.
# Each (robot, task, seed) combination runs as a separate background process,
# following the same parallelization pattern as eval_kuka.sh.
#
# Usage:
#   # Evaluate all 9 robots in parallel (one process per robot):
#   bash diffusion_policy/eval.sh
#
#   # Evaluate a subset of robots:
#   bash diffusion_policy/eval.sh sawyer panda ur5e
#
#   # Override checkpoint dir, arch type, or seed list via env vars:
#   CKPT_DIR=diffusion_policy/checkpoints ARCH=vanilla SEEDS="3 4 5" \
#       bash diffusion_policy/eval.sh sawyer
# ============================================================================

set -euo pipefail

# ── Configurable defaults (override via environment variables) ─────────────
CKPT_DIR="${CKPT_DIR:-diffusion_policy/checkpoints}"
RESULT_DIR="${RESULT_DIR:-diffusion_policy/results}"
ARCH="${ARCH:-advanced}"
SEEDS="${SEEDS:-3 4 5}"
NUM_EPISODES="${NUM_EPISODES:-100}"
DEVICE="${DEVICE:-cuda:0}"
SAMPLER="${SAMPLER:-ddim}"
DDIM_STEPS="${DDIM_STEPS:-10}"
ACTION_HORIZON="${ACTION_HORIZON:-}"

# Extra model args for vanilla arch (ignored for advanced)
VANILLA_ARGS="${VANILLA_ARGS:---cond_dim 64 --diffusion_step_embed_dim 64 --vanilla_hidden_dim 256 --vanilla_n_layers 3 --num_diffusion_steps 50}"

ALL_ROBOTS=(sawyer xarm7 panda ur10e gen3 unitree_z1 viperx ur5e kuka)
ALL_TASKS=(reach-v2 push-v2 pick-place-v2 door-open-v2 drawer-open-v2 button-press-topdown-v2 peg-insert-side-v2 window-open-v2 window-close-v2 faucet-open-v2)

# Use CLI args as robot list, or default to all robots
if [[ $# -gt 0 ]]; then
    ROBOTS=("$@")
else
    ROBOTS=("${ALL_ROBOTS[@]}")
fi

mkdir -p "${RESULT_DIR}"

# ── Build the extra args string for the architecture ───────────────────────
EXTRA_ARGS="--sampler ${SAMPLER} --ddim_steps ${DDIM_STEPS}"
if [[ -n "$ACTION_HORIZON" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --action_horizon ${ACTION_HORIZON}"
fi
if [[ "$ARCH" == "vanilla" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} ${VANILLA_ARGS}"
fi

# ── Evaluate one robot (all tasks, all seeds) ──────────────────────────────
evaluate_robot() {
    local robot="$1"
    local log_file="${RESULT_DIR}/eval_${robot}.log"

    echo "[$(date +%H:%M:%S)] Starting evaluation: robot=${robot}"

    python3 -u diffusion_policy/evaluate.py \
        --checkpoint "${CKPT_DIR}" \
        --arch_type "${ARCH}" \
        --robot_type "${robot}" \
        --tasks "${ALL_TASKS[@]}" \
        --seeds ${SEEDS} \
        --num_episodes "${NUM_EPISODES}" \
        --device "${DEVICE}" \
        --result_dir "${RESULT_DIR}" \
        ${EXTRA_ARGS} \
        > "${log_file}" 2>&1

    echo "[$(date +%H:%M:%S)] Finished evaluation: robot=${robot}  →  ${log_file}"
}

# ── Launch all robots in parallel ──────────────────────────────────────────
echo "============================================================"
echo "  Diffusion Policy — Parallel Evaluation"
echo "============================================================"
echo "  Robots    : ${ROBOTS[*]}"
echo "  Seeds     : ${SEEDS}"
echo "  Arch      : ${ARCH}"
echo "  Sampler   : ${SAMPLER} (ddim_steps=${DDIM_STEPS})"
echo "  Action Hz : ${ACTION_HORIZON:-all (pred_horizon)}"
echo "  Checkpoint: ${CKPT_DIR}"
echo "  Results   : ${RESULT_DIR}"
echo "  Device    : ${DEVICE}"
echo "============================================================"
echo ""

for robot in "${ROBOTS[@]}"; do
    evaluate_robot "${robot}" &
done

wait
echo ""
echo "✅  All robot evaluations finished.  Results in ${RESULT_DIR}/"
