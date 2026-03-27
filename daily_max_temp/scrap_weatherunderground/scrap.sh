#!/bin/bash

START="2018-01-01"
END="2026-03-27"
STATION="LEMD"
CITY="madrid"
ROOT="/home/camarada/Documents/projects/temp-grss-nasa/data_"

ATTEMPT=0

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "========================================"
    echo "Attempt #$ATTEMPT — $(date)"
    echo "========================================"

    python days_scraper.py \
        --start "$START" \
        --end "$END" \
        --station-id "$STATION" \
        --city "$CITY" \
        --root "$ROOT"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ All data collected! Exiting."
        break  # exits the while loop cleanly
    else
        echo "⚠️  Incomplete (exit code $EXIT_CODE). Retrying in 60s..."
        sleep 60
    fi
done

