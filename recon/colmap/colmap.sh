PROJECT="${1:?usage: $0 <project_dir>}"
gpu_id="${2:?usage: $0 <gpu_id>}"
IMG_DIR="${PROJECT}/ldr"
DB="${PROJECT}/database.db"
OUT_SPARSE="${PROJECT}/sparse"         # triangulated sparse model goes here
UNDIST_OUT="${PROJECT}/undistorted"          # undistorted workspace (optional)

echo "== Feature extraction =="
CUDA_VISIBLE_DEVICES=$gpu_id colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$IMG_DIR" \
    --ImageReader.single_camera=true \

echo "== Sequential matching =="
CUDA_VISIBLE_DEVICES=$gpu_id colmap sequential_matcher --database_path "$DB"

echo "== Mapping =="
mkdir -p ${PROJECT}/sparse
CUDA_VISIBLE_DEVICES=$gpu_id colmap mapper \
    --database_path=${PROJECT}/database.db \
    --image_path=${IMG_DIR} \
    --output_path=${PROJECT}/sparse

echo "== Bundle adjust =="
cp ${PROJECT}/sparse/0/*.bin ${PROJECT}/sparse/
for path in ${PROJECT}/sparse/*/; do
    m=$(basename ${path})
    if [ ${m} != "0" ]; then
        colmap model_merger \
            --input_path1=${PROJECT}/sparse \
            --input_path2=${PROJECT}/sparse/${m} \
            --output_path=${PROJECT}/sparse
        colmap bundle_adjuster \
            --input_path=${PROJECT}/sparse \
            --output_path=${PROJECT}/sparse
    fi
done

# echo "== Undistort images =="
# colmap image_undistorter \
#     --image_path=${IMG_DIR} \
#     --input_path=${PROJECT}/sparse \
#     --output_path=${UNDIST_OUT} \
#     --output_type=COLMAP

echo "== Convert sparse model to text =="
# convert sparse/0 from binary → text in place
colmap model_converter \
    --input_path   ${PROJECT}/sparse \
    --output_path  ${PROJECT}/sparse \
    --output_type  TXT

# echo "== Convert undistorted sparse model to text =="
# colmap model_converter \
#     --input_path   ${UNDIST_OUT}/sparse \
#     --output_path  ${UNDIST_OUT}/sparse \
#     --output_type  TXT

echo "== Convert sparse model to ply =="
CUDA_VISIBLE_DEVICES=$gpu_id colmap model_converter \
    --input_path "${PROJECT}/sparse" \
    --output_path "${PROJECT}/sparse/points3D.ply" \
    --output_type PLY

echo "== Finished COLMAP reconstruction =="
