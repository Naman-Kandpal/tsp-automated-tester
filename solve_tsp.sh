#!/bin/bash
export PATH="/d/Programming-Tools/Python/Python312/:$PATH"
export PYTHONIOENCODING="utf-8"
export PYTHONUNBUFFERED=1

# --- Dynamic Run Configuration ---
# Generate a timestamp to uniquely identify this experiment batch
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Algorithms to test
ALGORITHMS="ga nn 2opt"

# Experiment constraints
NUM_RUNS=10
MIN_CITIES=10
MAX_CITIES=100

# File Paths (Timestamped to prevent accidental overwriting of valuable data)
OUTPUT_FILENAME="runs_${TIMESTAMP}.csv"
OUTPUT_FILE="results/${OUTPUT_FILENAME}"
LOG_FILE="results/log_${TIMESTAMP}.txt"

# --- Initialization ---
mkdir -p results

# Start the overall timer
START_TIME=$(date +%s)

echo "==================================================" | tee -a "$LOG_FILE"
echo " STARTING TSP EXPERIMENT SUITE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo "Timestamp:      $TIMESTAMP" | tee -a "$LOG_FILE"
echo "Algorithms:     $ALGORITHMS" | tee -a "$LOG_FILE"
echo "Runs per Test:  $NUM_RUNS" | tee -a "$LOG_FILE"
echo "City Range:     $MIN_CITIES - $MAX_CITIES" | tee -a "$LOG_FILE"
echo "Output Data:    $OUTPUT_FILE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# --- Execution Phase ---
# Run main.py and pipe the output to both the console and the log file simultaneously
python main.py \
    --algorithms $ALGORITHMS \
    --runs $NUM_RUNS \
    --min-cities $MIN_CITIES \
    --max-cities $MAX_CITIES \
    --output-file "$OUTPUT_FILE" | tee -a "$LOG_FILE"

# Check if main.py executed successfully
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "\n❌ ERROR: main.py encountered an issue and stopped." | tee -a "$LOG_FILE"
    echo "Aborting analysis phase." | tee -a "$LOG_FILE"
    exit 1
fi

# --- Analysis Phase (The Icing on the Cake) ---
echo "" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo " GENERATING ANALYTICS & PLOTS" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

# Pass the newly generated timestamped filename directly to the analyzer
python src/analyser.py --file "$OUTPUT_FILENAME" | tee -a "$LOG_FILE"

# --- Completion ---
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo "" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo "EXPERIMENT SUITE FINISHED" | tee -a "$LOG_FILE"
echo "Total Time:     ${MINUTES}m ${SECONDS}s" | tee -a "$LOG_FILE"
echo "Results Saved:  $OUTPUT_FILE" | tee -a "$LOG_FILE"
echo "Console Log:    $LOG_FILE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"