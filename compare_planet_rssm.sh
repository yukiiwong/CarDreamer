#!/bin/bash
# Comprehensive comparison script for PlaNet vs RSSM
# This runs both models on the same task and compares performance

set -e

echo "=========================================="
echo "PlaNet vs RSSM Comparison Test"
echo "=========================================="
echo ""

# Configuration
CARLA_PORT_BASE=${1:-2000}
GPU_ID=${2:-0}
TASK=${3:-carla_four_lane}
STEPS=${4:-1000000}  # 1M steps for meaningful comparison

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Test configurations
declare -A tests=(
    ["rssm_small"]="small"
    ["planet_small"]="planet_small"
    ["rssm_medium"]="medium"
    ["planet_medium"]="planet_medium"
)

echo "Will test the following configurations:"
for name in "${!tests[@]}"; do
    echo "  - $name (config: ${tests[$name]})"
done
echo ""

# Run each test
for name in "${!tests[@]}"; do
    config="${tests[$name]}"
    port=$((CARLA_PORT_BASE++))

    echo "=========================================="
    echo "Testing: $name"
    echo "Config: $config"
    echo "Port: $port"
    echo "=========================================="

    LOGDIR="./logdir/comparison_${TIMESTAMP}/${name}"

    echo "Starting training..."
    echo "  Task: $TASK"
    echo "  Steps: $STEPS"
    echo "  Logdir: $LOGDIR"
    echo ""

    # Run training in background if multiple GPUs available
    bash train_dm3.sh $port $GPU_ID \
        --configs $config \
        --task $TASK \
        --dreamerv3.logdir $LOGDIR \
        --dreamerv3.run.steps $STEPS \
        --dreamerv3.run.eval_every 10000 \
        --dreamerv3.run.save_every 50000 &

    PID=$!
    echo "Training started with PID: $PID"
    echo ""

    # Wait a bit before starting next training
    sleep 10
done

echo "=========================================="
echo "All training jobs started!"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  - TensorBoard: tensorboard --logdir ./logdir/comparison_${TIMESTAMP}"
echo "  - Visualization: http://localhost:<port+7000>"
echo ""
echo "Wait for all jobs to complete, then run:"
echo "  python analyze_comparison.py ./logdir/comparison_${TIMESTAMP}"
