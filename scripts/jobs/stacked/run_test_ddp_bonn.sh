#!/bin/bash
# Smoke-test DDP on Stage-1 Bonn: runs single-GPU then DDP-2 with the same
# debug materials and rays_num, checks that DDP ranks end with identical weights,
# and prints elapsed time for each. Uses GPUs 1 and 2 by default (GPU 0 is in
# use by another tenant on this workstation).

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-material-capture}
SINGLE_GPU=${SINGLE_GPU:-1}
DDP_GPUS=${DDP_GPUS:-1,2}
DEBUG_NUM=${DEBUG_NUM:-2}
RAYS_NUM=${RAYS_NUM:-512}
MAX_STEPS=${MAX_STEPS:-5}
DUMP_DIR=${DUMP_DIR:-/tmp/ddp_bonn_test}

cd "$(dirname "$0")/../.."

rm -f ${DUMP_DIR}/state_devices*.pt 2>/dev/null || true
mkdir -p ${DUMP_DIR}

echo "############################################################"
echo "# Phase 1: single-GPU baseline (GPU=${SINGLE_GPU})"
echo "############################################################"
CUDA_VISIBLE_DEVICES=${SINGLE_GPU} conda run -n ${CONDA_ENV} --live-stream \
    python scripts/test_ddp_bonn.py \
        devices=1 \
        debug_num=${DEBUG_NUM} \
        rays_num=${RAYS_NUM} \
        max_steps=${MAX_STEPS} \
        dump_dir=${DUMP_DIR}

echo ""
echo "############################################################"
echo "# Phase 2: DDP-2 (GPUs=${DDP_GPUS})"
echo "############################################################"
CUDA_VISIBLE_DEVICES=${DDP_GPUS} conda run -n ${CONDA_ENV} --live-stream \
    python scripts/test_ddp_bonn.py \
        devices=2 \
        debug_num=${DEBUG_NUM} \
        rays_num=${RAYS_NUM} \
        max_steps=${MAX_STEPS} \
        dump_dir=${DUMP_DIR}

echo ""
echo "############################################################"
echo "# Phase 3: single-GPU vs DDP-2 weight-shape sanity"
echo "############################################################"
conda run -n ${CONDA_ENV} --live-stream python -c "
import torch
from pathlib import Path
d = Path('${DUMP_DIR}')
single = torch.load(d/'state_devices1_rank0.pt', map_location='cpu')['state_dict']
ddp = torch.load(d/'state_devices2_rank0.pt', map_location='cpu')['state_dict']
keys_single, keys_ddp = set(single), set(ddp)
print(f'single-GPU keys: {len(keys_single)},  DDP rank-0 keys: {len(keys_ddp)}')
print(f'symmetric diff:  {sorted(keys_single ^ keys_ddp)[:10]}')
shape_mm = []
for k in keys_single & keys_ddp:
    if single[k].shape != ddp[k].shape:
        shape_mm.append((k, tuple(single[k].shape), tuple(ddp[k].shape)))
if shape_mm:
    print(f'SHAPE MISMATCHES ({len(shape_mm)}):')
    for k, a, b in shape_mm[:10]:
        print(f'  {k}: single={a}  ddp={b}')
else:
    print('All shared keys have matching shapes.')
"
