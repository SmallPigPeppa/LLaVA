#!/usr/bin/env bash

# Goals:
# 1) Run up to 100 times.
# 2) After each execution, sleep for 30 seconds.
# 3) If the total time (execution + sleep) is less than 60 seconds, stop.

MAX_RUNS=1000         # Maximum number of runs
MIN_TOTAL_TIME=80    # If execution + sleep is less than this, stop early
SLEEP_TIME=30        # Sleep time after each run

for ((i=1; i<=MAX_RUNS; i++))
do
    echo "===> Starting run #$i..."

    # Record the start time
    START_TIME=$(date +%s)

    # ----------------------------
    # Adjust the Python command as needed
    python main_new_qwen.py \
      --api_key dcf59e46-a998-4a69-adfd-31aeabac1760 \
      --input_file input/part2.json \
      --output_file output-qwen72b/part2.json \
      --base_url https://api.ppinfra.com/v3/openai \
      --model qwen/qwen-2.5-72b-instruct \
      --max_tokens 2048 \
      --max_workers 60
    # ----------------------------

    # Record the end time
    END_TIME=$(date +%s)

    # Calculate execution time
    ELAPSED_EXEC=$(( END_TIME - START_TIME ))
    echo "===> Run #$i took $ELAPSED_EXEC seconds to complete."

    # Sleep for 30 seconds
    echo "===> Sleeping for $SLEEP_TIME seconds..."
    sleep $SLEEP_TIME

    # Calculate total time = execution time + sleep
    TOTAL_TIME=$(( ELAPSED_EXEC + SLEEP_TIME ))
    echo "===> Total time for run #$i (execution + sleep): $TOTAL_TIME seconds."

    # If the total time is less than 60 seconds, stop early
    if [ "$TOTAL_TIME" -lt "$MIN_TOTAL_TIME" ]; then
        echo "Total time < $MIN_TOTAL_TIME seconds. Stopping early."
        break
    fi

    echo
done

echo "All done."