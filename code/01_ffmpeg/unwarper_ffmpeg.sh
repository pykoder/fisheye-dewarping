#!/bin/bash

# Fisheye Unwarper using FFmpeg
# 
# A bash script that uses FFmpeg's v360 filter to dewarp fisheye images
# into multiple perspective views.
#
# Usage: ./unwarper_ffmpeg.sh [input_image] [output_dir]

# Default values
INPUT_IMAGE="../images/fisheye1024.mp4"
OUTPUT_DIR="."

# Parse command line arguments
if [ $# -ge 1 ]; then
    INPUT_IMAGE="$1"
fi
if [ $# -ge 2 ]; then
    OUTPUT_DIR="$2"
fi

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Get base filename without extension
BASENAME=$(basename "$INPUT_IMAGE" | cut -d. -f1)
INTERP=near

echo "Processing image: $INPUT_IMAGE"
echo "Output directory: $OUTPUT_DIR"
echo "Base filename: $BASENAME"

# Repeat the entire batch of dewarping operations
    
# Process each zone
ffmpeg -y -threads 0 -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i "$INPUT_IMAGE"  \
-vf "crop=1920:1920,v360=input=fisheye:output=flat:interp=$INTERP:yaw=0:pitch=45:roll=0:v_fov=90:w=960:h=960" "$OUTPUT_DIR/${BASENAME}_1_ffmpeg.mp4" \
-vf "rotate=4*72*PI/180,crop=1920:1920,v360=input=fisheye:output=flat:interp=$INTERP:yaw=0:pitch=45:roll=0:v_fov=90:w=960:h=960" "$OUTPUT_DIR/${BASENAME}_2_ffmpeg.mp4" \
-vf "rotate=3*72*PI/180,crop=1920:1920,v360=input=fisheye:output=flat:interp=$INTERP:yaw=0:pitch=45:roll=0:v_fov=90:w=960:h=960" "$OUTPUT_DIR/${BASENAME}_3_ffmpeg.mp4" \
-vf "rotate=2*72*PI/180,crop=1920:1920,v360=input=fisheye:output=flat:interp=$INTERP:yaw=0:pitch=45:roll=0:v_fov=90:w=960:h=960" "$OUTPUT_DIR/${BASENAME}_4_ffmpeg.mp4" \
-vf "rotate=72*PI/180,crop=1920:1920,v360=input=fisheye:output=flat:interp=$INTERP:yaw=0:pitch=45:roll=0:v_fov=90:w=960:h=960" "$OUTPUT_DIR/${BASENAME}_5_ffmpeg.mp4"

echo "Dewarping complete!"
echo "Generated 5 perspective views using FFmpeg"
