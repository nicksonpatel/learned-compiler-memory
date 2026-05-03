#!/usr/bin/env bash
# Monitors generate_data.py — kills it if quota is exhausted or stalled.
# Checks every 3 minutes.

CHECKPOINT="/Users/nickson/Documents/GitHub/learned-compiler-memory/data/synthetic/checkpoint.jsonl"
LOG="/tmp/mcm_gen.log"
INTERVAL=180   # 3 minutes
STALL_LIMIT=2  # kill after 2 consecutive stall checks (~6 min no progress)

prev_count=-1
stall_count=0

echo "[watcher] Started. Checking every ${INTERVAL}s."

while true; do
    sleep "$INTERVAL"

    # --- Check if generation process is still alive ---
    if ! pgrep -f "generate_data.py" > /dev/null 2>&1; then
        echo "[watcher] generate_data.py is no longer running. Exiting."
        break
    fi

    # --- Current checkpoint count ---
    cur_count=$(wc -l < "$CHECKPOINT" 2>/dev/null || echo 0)
    echo "[watcher] $(date '+%H:%M:%S')  checkpoint=$cur_count"

    # --- Detect quota / budget exhaustion in log ---
    if tail -40 "$LOG" 2>/dev/null | grep -qiE "429|quota|budget|rate.?limit|too.?many.?request"; then
        recent_errors=$(tail -40 "$LOG" | grep -ciE "429|quota|budget|rate.?limit|too.?many.?request")
        echo "[watcher] Detected $recent_errors quota/rate-limit errors in recent log."
        if [ "$recent_errors" -ge 5 ]; then
            echo "[watcher] Quota exhausted — killing generation process."
            pkill -f "generate_data.py" && echo "[watcher] Killed generate_data.py." || echo "[watcher] pkill returned non-zero (may already be dead)."
            pkill -f "caffeinate" 2>/dev/null
            break
        fi
    fi

    # --- Detect stall (no new examples) ---
    if [ "$prev_count" -eq "$cur_count" ] && [ "$prev_count" -ne -1 ]; then
        stall_count=$((stall_count + 1))
        echo "[watcher] No progress for ${stall_count}/${STALL_LIMIT} checks."
        if [ "$stall_count" -ge "$STALL_LIMIT" ]; then
            echo "[watcher] Stalled too long — killing generation process."
            pkill -f "generate_data.py" && echo "[watcher] Killed generate_data.py." || echo "[watcher] pkill returned non-zero."
            pkill -f "caffeinate" 2>/dev/null
            break
        fi
    else
        stall_count=0
    fi

    prev_count="$cur_count"
done

echo "[watcher] Done. Final checkpoint count: $(wc -l < "$CHECKPOINT" 2>/dev/null)"
