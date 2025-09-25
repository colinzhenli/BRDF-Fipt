Here’s a clean `README.md` version of your instructions, properly formatted in Markdown:

````
#Data Processing

Replace `/media/raid/cloth/No_cable_capture_Sep22/Axis_estimation_Sep22` in the following commands with your actual data folder.  
The folder should have the following structure:

```bash
/images
   scan-0000_{}.png
   scan-0001_{}.png
scan_log.json
````

---

## 0. Correct indices or 

You can do this during saving the log file.

```bash
# python process_json.py <input_json_path> <output_json_path>
python process_json.py \
    /media/raid/cloth/No_cable_capture_Sep22/Axis_estimation_Sep22/scan_fixed_light_log.json \
    /media/raid/cloth/No_cable_capture_Sep22/Axis_estimation_Sep22/scan_log_reindexed.json
```


## 1. Add Scan ID and Camera ID to JSON Entries

You can do this during saving the log file.

```bash
# python process_json.py <input_json_path> <output_json_path>
python process_json.py \
    /media/raid/cloth/No_cable_capture_Sep22/Axis_estimation_Sep22/scan_fixed_light_log.json \
    /media/raid/cloth/No_cable_capture_Sep22/Axis_estimation_Sep22/scan_log_reindexed.json
```

---

## 2. Mask the Rectangle

Update parameters in `config/renderer/realcapture_area_emitter.yaml`:

* `renderer.emitter.turntable.center`

Then run:

```bash
python recon/preprocess/background_mask_multi_thread.py \
    exp_folder=/media/raid/cloth/No_cable_capture_Sep22/Axis_estimation_Sep22
```

---

## 3. Debayering the Images

This step saves both HDR images (for training) and LDR images (for COLMAP).

```bash
python recon/preprocess/debayering_mask_multi_thread.py \
    --input /media/raid/cloth/No_cable_capture_Sep22/Axis_estimation_Sep22/masks/masked_images \
    --output /media/raid/cloth/No_cable_capture_Sep22/Axis_estimation_Sep22/masks/masked_images_debayered \
    --pattern RGGB \
    --jobs 16 \
    --format png  # options: int16 png or float32 exr
```

---

## 4. Copy LDR Images to Filtered Folder

```bash
cp /media/raid/cloth/No_cable_capture_Sep22/Axis_estimation_Sep22/masks/ldr \
   /media/raid/cloth/No_cable_capture_Sep22/Axis_estimation_Sep22/masks/filtered_purple_ldr
```

---

## 5. Filter Purple Images

> ⚠️ This code is sensitive to data format and lighting conditions.
> Tune it before running.

```bash
python recon/preprocess/purple_filter.py \
    exp_folder=/media/raid/cloth/No_cable_capture_Sep22/Axis_estimation_Sep22
```

---

## 6. Run COLMAP

```bash
bash ./recon/colmap/colmap.sh \
    /media/raid/cloth/No_cable_capture_Sep22/Axis_estimation_Sep22 0
# 0 = GPU ID
```

```

Do you want me to also add a **Quick Pipeline Summary section** at the top (like a numbered workflow overview) so someone new can quickly see the whole process before diving into each command?
```
