#!/bin/bash

# Run shape_matching.py for folders 100-125 in Debug_Dec23 (all in parallel)

for i in $(seq 100 125); do
    folder="/media/raid/cloth/capture_data/Debug_Dec23/$i"
    
    if [ -d "$folder" ]; then
        echo "Starting folder $i in background..."
        
        python ./recon/calibration/shape_matching.py \
            shape_matching.folder_path="$folder" \
            shape_matching.num_workers=64 \
            shape_matching.z_outlier_percentile=0.0 \
            > "shape_matching_${i}.log" 2>&1 &
    else
        echo "Folder $i does not exist, skipping..."
    fi
done

echo ""
echo "All jobs started! Waiting for completion..."
wait

echo "All jobs completed!"
