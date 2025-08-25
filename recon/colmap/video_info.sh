#!/bin/bash
# Usage: ./video_info.sh your_video.avi

video="$1"

if [ -z "$video" ]; then
    echo "Usage: $0 <video_file>"
    exit 1
fi

# Frame count
frames=$(ffprobe -v error \
    -select_streams v:0 \
    -count_packets \
    -show_entries stream=nb_read_packets \
    -of csv=p=0 "$video")

# Duration in seconds
duration=$(ffprobe -v error \
    -select_streams v:0 \
    -show_entries format=duration \
    -of csv=p=0 "$video")

echo "Frames: $frames"
printf "Duration: %.2f seconds\n" "$duration"
