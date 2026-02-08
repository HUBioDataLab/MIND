#!/bin/bash

# Script to train QM9 transfer learning models for all 19 properties
# Each model will be trained for 100 epochs
#
# Usage:
#   ./train_all_qm9_properties.sh
#   or
#   bash train_all_qm9_properties.sh
#
# The script will:
# - Train models for all 19 QM9 properties sequentially
# - Save outputs to ./outputs/qm9_transfer/<property>/
# - Log progress to ./outputs/qm9_transfer/training_log.txt
# - Each property's training output is saved to <property>/training_output.log

# Configuration
CONFIG_FILE="core/downstream/qm9/qm9_transfer_config.yaml"
BASE_OUTPUT_DIR="./outputs/qm9_transfer"
WANDB_PROJECT="qm9-transfer-learning"
MAX_EPOCHS=100
GPU_ID=0
BATCH_SIZE=7936  # You can adjust this if needed, or set to "" to use config default

# All 19 QM9 properties
PROPERTIES=(
    "mu"
    "alpha"
    "homo"
    "lumo"
    "gap"
    "r2"
    "zpve"
    "u0"
    "u298"
    "h298"
    "g298"
    "cv"
    "u0_atom"
    "u_atom"
    "h_atom"
    "g_atom"
    "A"
    "B"
    "C"
)

# Create base output directory if it doesn't exist
mkdir -p "$BASE_OUTPUT_DIR"

# Log file for tracking progress
LOG_FILE="$BASE_OUTPUT_DIR/training_log.txt"
echo "=== QM9 Transfer Learning - All Properties Training ===" > "$LOG_FILE"
echo "Started at: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Counter for tracking
TOTAL=${#PROPERTIES[@]}
CURRENT=0
SUCCESSFUL=0
FAILED=0

# Function to train a single property
train_property() {
    local property=$1
    local current=$2
    local total=$3
    
    echo ""
    echo "=========================================="
    echo "[$current/$total] Training property: $property"
    echo "=========================================="
    echo "[$current/$total] Training property: $property" >> "$LOG_FILE"
    
    # Create property-specific output directory
    PROP_OUTPUT_DIR="$BASE_OUTPUT_DIR/${property}"
    mkdir -p "$PROP_OUTPUT_DIR"
    
    # Set CUDA device and run training
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    # Run training and capture output
    python core/downstream/qm9/train_qm9_transfer.py \
        --config "$CONFIG_FILE" \
        --target-name "$property" \
        --max-epochs $MAX_EPOCHS \
        --gpu $GPU_ID \
        --batch-size $BATCH_SIZE \
        > "$PROP_OUTPUT_DIR/training_output.log" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Successfully completed training for $property"
        echo "✓ Successfully completed training for $property" >> "$LOG_FILE"
        ((SUCCESSFUL++))
        return 0
    else
        echo "✗ Failed to train $property (exit code: $exit_code)"
        echo "✗ Failed to train $property (exit code: $exit_code)" >> "$LOG_FILE"
        ((FAILED++))
        return 1
    fi
}

# Train all properties
for property in "${PROPERTIES[@]}"; do
    ((CURRENT++))
    train_property "$property" "$CURRENT" "$TOTAL"
    
    # Optional: Add a small delay between trainings to avoid resource conflicts
    sleep 2
done

# Summary
echo ""
echo "=========================================="
echo "Training Summary"
echo "=========================================="
echo "Total properties: $TOTAL"
echo "Successful: $SUCCESSFUL"
echo "Failed: $FAILED"
echo "Completed at: $(date)"
echo "" >> "$LOG_FILE"
echo "=== Training Summary ===" >> "$LOG_FILE"
echo "Total properties: $TOTAL" >> "$LOG_FILE"
echo "Successful: $SUCCESSFUL" >> "$LOG_FILE"
echo "Failed: $FAILED" >> "$LOG_FILE"
echo "Completed at: $(date)" >> "$LOG_FILE"

# List failed properties if any
if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed properties:"
    echo "Check individual log files in: $BASE_OUTPUT_DIR/<property>/training_output.log"
fi

echo ""
echo "All training logs saved to: $LOG_FILE"
echo "Individual property outputs in: $BASE_OUTPUT_DIR/<property>/"
