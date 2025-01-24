#!/bin/bash

# Default parameters
CSV_FILE="data/batch_files.csv"
SPEAKER_PROMPT_AUDIO_FOLDER="data"
OUTPUT_AUDIO_FOLDER="results"

# Run the Python script with default parameters
python batch_inference.py \
    --csv_file "$CSV_FILE" \
    --speaker_prompt_audio_folder "$SPEAKER_PROMPT_AUDIO_FOLDER" \
    --output_audio_folder "$OUTPUT_AUDIO_FOLDER"