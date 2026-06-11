#!/bin/bash

# Script to submit all generated learning rate sweep jobs

JOB_DIR="jobs/sweep/generated_lr_sweep"

if [ ! -d "$JOB_DIR" ]; then
    echo "Error: Job directory $JOB_DIR does not exist."
    echo "Please run generate_sweep_jobs.sh first."
    exit 1
fi

job_count=$(ls ${JOB_DIR}/*.sh 2>/dev/null | wc -l)

if [ "$job_count" -eq 0 ]; then
    echo "Error: No job files found in $JOB_DIR"
    echo "Please run generate_sweep_jobs.sh first."
    exit 1
fi

echo "=============================================="
echo "Submitting ${job_count} jobs from ${JOB_DIR}"
echo "=============================================="
echo ""

submitted=0
failed=0

for job in ${JOB_DIR}/*.sh; do
    job_name=$(basename "$job")
    echo "Submitting: ${job_name}"
    
    if sbatch "$job"; then
        submitted=$((submitted + 1))
    else
        echo "  ERROR: Failed to submit ${job_name}"
        failed=$((failed + 1))
    fi
done

echo ""
echo "=============================================="
echo "Submission complete!"
echo "Successfully submitted: ${submitted}"
echo "Failed: ${failed}"
echo "=============================================="
echo ""
echo "To check job status, run:"
echo "  squeue -u $USER"

