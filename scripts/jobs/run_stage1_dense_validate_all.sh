#!/bin/bash

# Launch Stage-1 dense (Ours) validation across all epoch=*.ckpt in a checkpoint
# dir, launching every job concurrently and round-robin assigning them across
# two GPUs (so multiple jobs share a GPU). experiment_name is derived from the
# epoch number. Ray count and learning rates are aligned with the Bonn
# validate-all setting.

set -u

CKPT_DIR="/media/raid/cloth/output/BRDF/Bonn_VML/output/Stage1_Fir_Ours-300-all_Subsample-0.1_Batch-500K_run_1/training/model_0.20_0.20"
GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}
RUN_TAG=${RUN_TAG:-run_1}
LOG_DIR=${LOG_DIR:-logs/validate_stage1_dense}

mkdir -p "${LOG_DIR}"

# Collect epoch_*.ckpt sorted numerically (rename `epoch=N.ckpt` -> `epoch_N.ckpt`
# so the path can be passed as a Hydra override; `=` breaks command-line parsing).
mapfile -t CKPTS < <(ls "${CKPT_DIR}"/epoch_*.ckpt 2>/dev/null \
    | awk -F'epoch_|\\.ckpt' '{print $2"\t"$0}' \
    | sort -n \
    | cut -f2-)

if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "No epoch_*.ckpt found in ${CKPT_DIR}" >&2
    exit 1
fi

echo "Found ${#CKPTS[@]} checkpoints in ${CKPT_DIR}:"
for c in "${CKPTS[@]}"; do echo "  $c"; done

run_one() {
    local gpu="$1"
    local ckpt="$2"
    local epoch
    epoch=$(basename "${ckpt}" .ckpt | sed 's/^epoch_//')
    local exp="Validate_Stage1_Ours-420_Epoch-${epoch}_Subsample-0.02_${RUN_TAG}"
    local log="${LOG_DIR}/${exp}.log"
    echo "[GPU ${gpu}] -> epoch ${epoch}  (log: ${log})"
    CUDA_VISIBLE_DEVICES="${gpu}" python main.py \
        dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11 \
        data.training_list_path=/media/raid/cloth/capture_data/Dataset_Nov11/test_list_442.txt \
        data=points_dense \
        data.rays_num=200000 \
        data.point_subsample_ratio=0.001 \
        renderer=multiarea_emitter \
        material=multi_material_latent \
        experiment_name="${exp}" \
        model.optimizer.name=Adam8bit \
        model.continue_training=False \
        model.validate_on_stage1=True \
        model.freeze_decoder=True \
        model.optimizer.reset_latent_momentum_on_chunk_switch=False \
        model.optimizer.lr=1e-3 \
        model.optimizer.decoder_lr=1e-4 \
        model.loss.recon_loss.name=logrel \
        model.loss.reg_loss.weight=0.0 \
        model.stage=1 \
        model.test=False \
        model.trainer.max_epochs=50 \
        model.trainer.check_val_every_n_epoch=10000 \
        model.trainer.num_sanity_val_steps=0 \
        model.trainer.limit_train_batches=2000 \
        model.trainer.log_every_n_steps=20 \
        model.trainer.enable_checkpointing=False \
        material.decoder.use_skip_connection=True \
        material.latent_dim=24 \
        material.decoder.degree=3 \
        material.decoder.smooth_reg=False \
        material.different_decoder=False \
        data.filter_observations=False \
        data.switch_iters=100 \
        data.chunk_size=2 \
        renderer.spp.train=4 \
        model.ckpt_path="${ckpt}" \
        > "${log}" 2>&1
    local rc=$?
    echo "[GPU ${gpu}] epoch ${epoch} done (rc=${rc})"
    return ${rc}
}

# Launch every job concurrently, round-robin across the two GPUs.
PIDS=()
for i in "${!CKPTS[@]}"; do
    if [ $((i % 2)) -eq 0 ]; then
        gpu="${GPU_A}"
    else
        gpu="${GPU_B}"
    fi
    run_one "${gpu}" "${CKPTS[$i]}" &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} jobs in parallel across GPUs ${GPU_A} and ${GPU_B}."

RC=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || RC=$?
done

echo "All jobs finished. aggregate rc=${RC}"
exit ${RC}
