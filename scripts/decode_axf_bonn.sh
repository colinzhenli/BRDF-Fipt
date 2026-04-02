#!/bin/bash
set -e

SDK=/media/raid/cloth/downloads_software_developer_AxF_AxF-Decoding-SDK-Package-1.9.2/AxF-Decoding-SDK
AXF_DECODE=$SDK/AxFDecode
INPUT_DIR=/media/raid/cloth/Bonn_train
OUTPUT_DIR=/media/raid/cloth/Bonn_svfresnel

export LD_LIBRARY_PATH=$SDK/Linux.x64/lib:$LD_LIBRARY_PATH

mkdir -p $OUTPUT_DIR

total=$(ls $INPUT_DIR/*.axf | wc -l)
count=0

for axf_file in $INPUT_DIR/*.axf; do
    filename=$(basename "$axf_file" .axf)   # e.g. mat0005_svfresnel
    mat_dir=$OUTPUT_DIR/$filename
    mkdir -p $mat_dir

    count=$((count + 1))
    echo "[$count/$total] Decoding $filename ..."

    $AXF_DECODE --in "$axf_file" --out "$mat_dir" > /dev/null 2>&1
    echo "  -> saved to $mat_dir"
done

echo ""
echo "Done. All $total materials decoded to $OUTPUT_DIR"
