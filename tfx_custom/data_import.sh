#!/bin/bash

# Script: data_import.sh
# Description: Imports MP3 or WAV files for overt sounds, processes them, and organizes them for the TFX pipeline.
# Usage: ./data_import.sh <input_directory> <output_directory>

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Function to process audio files
process_audio_files() {
    local input_file=$1
    local output_file=$2

    # Check file format and convert if needed
    if [[ "$input_file" == *.mp3 ]]; then
        echo "Converting MP3 to WAV: $input_file -> $output_file"
        ffmpeg -i "$input_file" -ar 16000 -ac 1 "$output_file" >/dev/null 2>&1
    elif [[ "$input_file" == *.wav ]]; then
        echo "Copying WAV file: $input_file -> $output_file"
        cp "$input_file" "$output_file"
    else
        echo "Skipping unsupported file: $input_file"
    fi
}

# Iterate through all audio files in the input directory
for file in "$INPUT_DIR"/*; do
    if [[ -f "$file" ]]; then
        # Define output file name
        base_name=$(basename "$file")
        output_file="$OUTPUT_DIR/${base_name%.*}.wav"

        # Process the file
        process_audio_files "$file" "$output_file"
    fi
done

echo "Audio import complete. Processed files are in $OUTPUT_DIR."

