#!/usr/bin/env python3
"""Create a minimal Bonn debug dataset from mat0001 only.

Selects the first 4 poly views (cv01+il026 in rot000/045/090/135),
crops every image to 2×2 pixels, and writes a self-contained
Bonn_debug/ folder that the BonnDataset / BonnValDataset code can
load directly.

Output files
------------
Bonn_debug/
  mat0001_poly.exr         -- (2, 2, 12)  4 views × 3 RGB channels
  mat0001_xyz_rot000.exr   -- (2, 2, 3)   x/y/z
  mat0001_calibration.mat  -- trimmed to cv01 + il26 only
  bonn_point_metadata.json -- H=2, W=2, num_points=4
  mat0001.txt              -- copied as-is

Usage
-----
  python scripts/create_bonn_debug_dataset.py \
      --src  /media/raid/cloth/Bonn_train \
      --dst  /media/raid/cloth/Bonn_debug \
      --mat  1 \
      --n_views 4 \
      --crop_h 2 \
      --crop_w 2
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import OpenEXR
import Imath
import pyexr
import scipy.io as spio


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_poly_channels(channel_names):
    """Return list of dicts with camera/led/rotation/ch_start for each image."""
    pattern = re.compile(r'poly_(cv\d+)_(il\d+)_(rot\d+)_[BGR]')
    images = []
    for i in range(0, len(channel_names), 3):
        if i + 2 >= len(channel_names):
            break
        m = pattern.match(channel_names[i])
        if m:
            cam, led, rot = m.groups()
            images.append(dict(camera=cam, led=led, rotation=rot, ch_start=i))
    return images


def _read_exr_as_dict(filepath):
    """Read an EXR file.  Returns (channel_dict, H, W).

    channel_dict maps channel_name → (H, W) float32 array.
    """
    exr = pyexr.open(str(filepath))
    ch_names = exr.channel_map['all']
    data = exr.get(group='all', precision=pyexr.HALF).astype(np.float32)  # (H,W,C)
    H, W = data.shape[:2]
    ch_dict = {name: data[:, :, i] for i, name in enumerate(ch_names)}
    return ch_dict, H, W


def _write_exr(filepath, ch_dict):
    """Write a multi-channel EXR with exact channel names using OpenEXR.

    pyexr.write appends a '.Z' suffix to 2-D arrays (treats them as
    grayscale images), which would break the channel-name parser in
    bonn.py.  Writing via OpenEXR directly preserves the names exactly.
    """
    first = next(iter(ch_dict.values()))
    H, W = first.shape
    header = OpenEXR.Header(W, H)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    header['channels'] = {name: half_chan for name in ch_dict}
    out = OpenEXR.OutputFile(str(filepath), header)
    out.writePixels({name: arr.astype(np.float16).tobytes()
                     for name, arr in ch_dict.items()})
    out.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create Bonn debug dataset")
    parser.add_argument('--src', default='/media/raid/cloth/Bonn_train',
                        help='Source Bonn_train folder')
    parser.add_argument('--dst', default='/media/raid/cloth/Bonn_debug',
                        help='Destination debug folder')
    parser.add_argument('--mat', type=int, default=1,
                        help='Material ID (default: 1 → mat0001)')
    parser.add_argument('--n_views', type=int, default=4,
                        help='Number of poly views to keep (default: 4)')
    parser.add_argument('--crop_h', type=int, default=2,
                        help='Crop height in pixels (default: 2)')
    parser.add_argument('--crop_w', type=int, default=2,
                        help='Crop width in pixels (default: 2)')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    mat_id  = args.mat
    prefix  = f'mat{mat_id:04d}'
    H_out   = args.crop_h
    W_out   = args.crop_w
    N_views = args.n_views

    print(f"Source      : {src}")
    print(f"Destination : {dst}")
    print(f"Material    : {prefix}")
    print(f"Crop size   : {H_out}×{W_out} pixels")
    print(f"Views kept  : {N_views}")
    print()

    # -----------------------------------------------------------------------
    # 1.  xyz_rot000.exr  →  crop to (H_out, W_out, 3)
    # -----------------------------------------------------------------------
    xyz_src = src / f'{prefix}_xyz_rot000.exr'
    print(f"[1/4] Reading {xyz_src.name} …")
    xyz_dict, H_src, W_src = _read_exr_as_dict(xyz_src)
    print(f"      Original size : {H_src}×{W_src}  channels: {list(xyz_dict.keys())}")

    xyz_cropped = {name: arr[:H_out, :W_out] for name, arr in xyz_dict.items()}

    xyz_dst = dst / f'{prefix}_xyz_rot000.exr'
    _write_exr(xyz_dst, xyz_cropped)
    print(f"      Written  {xyz_dst}  ({H_out}×{W_out}×{len(xyz_cropped)})")

    # -----------------------------------------------------------------------
    # 2.  poly.exr  →  select first N_views images, crop to (H_out, W_out)
    # -----------------------------------------------------------------------
    poly_src = src / f'{prefix}_poly.exr'
    print(f"\n[2/4] Reading {poly_src.name} …  (this may take a moment)")
    poly_dict, pH, pW = _read_exr_as_dict(poly_src)
    ch_names = list(poly_dict.keys())   # alphabetical order preserved by pyexr
    print(f"      Original size : {pH}×{pW}  channels: {len(ch_names)}")

    # Parse all images and pick the first N_views
    images = _parse_poly_channels(ch_names)
    print(f"      Total images  : {len(images)}")

    if N_views > len(images):
        raise ValueError(
            f"Requested {N_views} views but only {len(images)} available.")

    selected = images[:N_views]
    print(f"      Selected views:")
    for im in selected:
        print(f"        {im['camera']}  {im['led']}  {im['rotation']}")

    # Build output channel dict for the selected views (cropped)
    poly_out = {}
    for im in selected:
        i = im['ch_start']
        # ch_names[i], [i+1], [i+2] are _B, _G, _R for this image
        for offset in range(3):
            name = ch_names[i + offset]
            poly_out[name] = poly_dict[name][:H_out, :W_out]

    poly_dst = dst / f'{prefix}_poly.exr'
    _write_exr(poly_dst, poly_out)
    print(f"      Written  {poly_dst}  ({H_out}×{W_out}×{len(poly_out)})")

    # -----------------------------------------------------------------------
    # 3.  calibration.mat  →  keep only cameras & LEDs used by selected views
    # -----------------------------------------------------------------------
    calib_src = src / f'{prefix}_calibration.mat'
    print(f"\n[3/4] Processing {calib_src.name} …")
    raw = spio.loadmat(str(calib_src))

    # Determine unique cameras & 2-digit LED names used by selected views
    used_cameras = sorted({im['camera'] for im in selected})    # e.g. ['cv01']
    used_leds_3d = sorted({im['led']    for im in selected})    # e.g. ['il026']

    # Convert 3-digit EXR names → 2-digit calib names  (il026 → il26, il032 → il32)
    def _to_2digit(led3):
        m = re.match(r'(il)(\d+)', led3)
        if m:
            return f"{m.group(1)}{int(m.group(2))}"
        return led3

    used_leds_2d = [_to_2digit(l) for l in used_leds_3d]

    wanted_fields = used_cameras + used_leds_2d  # fields to keep per rotation
    print(f"      Cameras kept : {used_cameras}")
    print(f"      LEDs kept    : {used_leds_2d}")

    rot_keys = ['rot000', 'rot045', 'rot090', 'rot135', 'rot180']

    # Build trimmed calibration dict suitable for scipy.io.savemat
    # scipy.io writes nested dicts as MATLAB structs; loadmat reads them
    # back as (1,1) structured arrays — exactly what _load_calibration expects.
    new_calib = {}
    for rot_key in rot_keys:
        rd = raw[rot_key][0, 0]
        rot_out = {}
        for field in wanted_fields:
            if field in rd.dtype.names:
                rot_out[field] = np.array(rd[field], dtype=np.float32).flatten()
        # Always keep llsCorners (needed by lls data; shape (3,4,14))
        if 'llsCorners' in rd.dtype.names:
            rot_out['llsCorners'] = np.array(rd['llsCorners'], dtype=np.float32)
        new_calib[rot_key] = rot_out

    new_calib['llsAnglesDegrees'] = raw['llsAnglesDegrees'].flatten().astype(np.float64)

    # poly2panIndices: original (100, 1) mapping poly→pan image index.
    # Trim to the N_views we kept.
    if 'poly2panIndices' in raw:
        orig_idx = raw['poly2panIndices']              # (100, 1)
        view_rows = [im['ch_start'] // 3 for im in selected]  # row index per image
        new_calib['poly2panIndices'] = orig_idx[view_rows, :]  # (N_views, 1)
    if 'poly2panWeights' in raw:
        new_calib['poly2panWeights'] = raw['poly2panWeights']

    calib_dst = dst / f'{prefix}_calibration.mat'
    spio.savemat(str(calib_dst), new_calib)
    print(f"      Written  {calib_dst}")

    # -----------------------------------------------------------------------
    # 4.  bonn_point_metadata.json  →  update entry for mat_id
    # -----------------------------------------------------------------------
    meta_src = src / 'bonn_point_metadata.json'
    print(f"\n[4/4] Processing {meta_src.name} …")
    if meta_src.exists():
        with open(meta_src) as f:
            meta = json.load(f)
    else:
        meta = {}

    meta[str(mat_id)] = {
        'H': H_out,
        'W': W_out,
        'num_points': H_out * W_out,
    }

    # Only keep the entry for the material we are exporting
    meta_trimmed = {str(mat_id): meta[str(mat_id)]}

    meta_dst = dst / 'bonn_point_metadata.json'
    with open(meta_dst, 'w') as f:
        json.dump(meta_trimmed, f, indent=2)
    print(f"      Written  {meta_dst}  → {meta_trimmed}")

    # -----------------------------------------------------------------------
    # 5.  Copy mat0001.txt as-is
    # -----------------------------------------------------------------------
    txt_src = src / f'{prefix}.txt'
    if txt_src.exists():
        shutil.copy2(txt_src, dst / f'{prefix}.txt')
        print(f"\n[+]  Copied {txt_src.name}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Debug dataset written to: {dst}")
    print(f"  mat0001_xyz_rot000.exr   {H_out}×{W_out}×3")
    print(f"  mat0001_poly.exr         {H_out}×{W_out}×{N_views * 3}  ({N_views} views)")
    print(f"  mat0001_calibration.mat  {len(wanted_fields)} fields/rotation")
    print(f"  bonn_point_metadata.json H={H_out} W={W_out} pts={H_out*W_out}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
