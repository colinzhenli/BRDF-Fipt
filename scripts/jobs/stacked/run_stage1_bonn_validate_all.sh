#!/bin/bash

# Launch Stage-1 Bonn validation across all epoch_*.ckpt in a checkpoint dir,
# launching every job concurrently and round-robin assigning them across two
# GPUs (so multiple jobs share a GPU). experiment_name is derived from the
# epoch number.

set -u

CKPT_DIR=${CKPT_DIR:-"/media/raid/cloth/output/BRDF/Bonn_VML/output/Bonn-Theia2/Stage-1_VML_Bonn_Subsample-0.1_Batch-500K_run_1/training/model_0.20_0.20"}
GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}
RUN_TAG=${RUN_TAG:-run_1}
LOG_DIR=${LOG_DIR:-logs/validate_stage1_bonn}
# EPOCHS: optional space-separated subset (e.g. EPOCHS="39 59 79"). Empty = all.
EPOCHS=${EPOCHS:-}

mkdir -p "${LOG_DIR}"

# Collect epochs sorted numerically (epoch_9 before epoch_39, etc.)
mapfile -t CKPTS < <(ls "${CKPT_DIR}"/epoch_*.ckpt 2>/dev/null \
    | awk -F'epoch_|\\.ckpt' '{print $2"\t"$0}' \
    | sort -n \
    | cut -f2-)

if [ -n "${EPOCHS}" ]; then
    FILTERED=()
    for ckpt in "${CKPTS[@]}"; do
        e=$(basename "${ckpt}" .ckpt | sed 's/^epoch_//')
        for want in ${EPOCHS}; do
            if [ "${e}" = "${want}" ]; then FILTERED+=("${ckpt}"); break; fi
        done
    done
    CKPTS=("${FILTERED[@]}")
fi

if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "No matching epoch_*.ckpt found in ${CKPT_DIR} (EPOCHS='${EPOCHS}')" >&2
    exit 1
fi

echo "Found ${#CKPTS[@]} checkpoints in ${CKPT_DIR}:"
for c in "${CKPTS[@]}"; do echo "  $c"; done

run_one() {
    local gpu="$1"
    local ckpt="$2"
    local epoch
    epoch=$(basename "${ckpt}" .ckpt | sed 's/^epoch_//')
    local exp="Validate_Stage-1_Bonn_Epoch-${epoch}_Batch-500K_${RUN_TAG}"
    local log="${LOG_DIR}/${exp}.log"
    echo "[GPU ${gpu}] -> epoch ${epoch}  (log: ${log})"
    CUDA_VISIBLE_DEVICES="${gpu}" python main.py \
        dataset_folder=/media/raid/cloth/Bonn_val \
        data=bonn \
        data.num_load_workers=4 \
        data.use_pan=True \
        data.use_lls=True \
        data.debug=False \
        data.debug_num=200 \
        data.rays_num=200000 \
        data.point_subsample_ratio=0.001 \
        renderer=multiarea_emitter \
        material=bonn_latent \
        experiment_name="${exp}" \
        model.optimizer.reset_latent_momentum_on_chunk_switch=False \
        model.optimizer.name=Adam8bit \
        model.loss.recon_loss.name=logrel \
        # model.loss.recon_loss.log_space.logrel_ref=0.05 \
        model.loss.recon_loss.log_space.logrel_ref=0.02 \
        model.loss.reg_loss.weight=0.0 \
        model.loss.pan_weight=1.0 \
        model.loss.lls_weight=1.0 \
        model.lls_spp=4 \
        model.stage=1 \
    model.apply_cosine_weight=False \
        model.test=False \
        model.continue_training=False \
        model.validate_on_stage1=True \
        model.freeze_decoder=True \
        model.trainer.max_epochs=20 \
        model.optimizer.lr=1e-3 \
        model.optimizer.decoder_lr=1e-4 \
        model.trainer.limit_train_batches=2000 \
        model.trainer.check_val_every_n_epoch=10000 \
        model.trainer.num_sanity_val_steps=0 \
        model.trainer.enable_checkpointing=False \
        material.decoder.use_skip_connection=True \
        material.decoder.use_film=False \
        material.decoder.use_color_decomp=False \
        material.latent_dim=24 \
        material.decoder.degree=3 \
        material.decoder.smooth_reg=False \
        material.different_decoder=False \
        data.filter_observations=False \
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
