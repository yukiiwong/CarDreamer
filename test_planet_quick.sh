#!/bin/bash
# Quick test script for PlaNet integration
# This script runs a short training session to verify PlaNet works correctly

set -e  # Exit on error

echo "=========================================="
echo "PlaNet Integration Quick Test"
echo "=========================================="
echo ""

# Configuration
CARLA_PORT=${1:-2000}
GPU_ID=${2:-0}
TASK=${3:-carla_four_lane}
STEPS=${4:-1000}  # Short test, just 1000 steps

# Test different PlaNet configurations
configs=("planet_small" "planet_medium")

for config in "${configs[@]}"; do
    echo "----------------------------------------"
    echo "Testing configuration: $config"
    echo "----------------------------------------"

    LOGDIR="./logdir/test_${config}_$(date +%Y%m%d_%H%M%S)"

    echo "Starting training with:"
    echo "  - Config: $config"
    echo "  - Task: $TASK"
    echo "  - Steps: $STEPS"
    echo "  - Logdir: $LOGDIR"
    echo ""

    # Run training
    bash train_dm3.sh $CARLA_PORT $GPU_ID \
        --configs $config \
        --task $TASK \
        --dreamerv3.logdir $LOGDIR \
        --dreamerv3.run.steps $STEPS \
        --dreamerv3.run.eval_every 500 \
        --dreamerv3.run.save_every 500 || {
            echo "❌ Test failed for $config"
            exit 1
        }

    echo "✅ Test passed for $config"
    echo ""

    # Check if checkpoint was created
    if [ -d "$LOGDIR" ]; then
        echo "Log directory created: $LOGDIR"
        ls -lh $LOGDIR/
    else
        echo "⚠️  Warning: Log directory not found"
    fi

    echo ""
done

echo "=========================================="
echo "✅ All PlaNet tests completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Check the log directories for training curves"
echo "2. Run longer training sessions for performance comparison"
echo "3. Compare results with RSSM baseline"
