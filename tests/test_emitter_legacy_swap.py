"""
Unit test for MultiAreaEmitter legacy_swap_indexing.

Validates that when the flag is on, the emitter reads scan_log.json from the
swap partner's folder (matching MultiMaterialDenseDataset and
_load_point_metadata behavior) — fixing the silent inconsistency where the
emitter would otherwise read post-swap lights for a slot while dataset/material
read pre-swap lights via the partner remap.

Strategy: build a tiny on-disk fixture with two folders (slot 23 and slot 500)
holding scan_log.json files of *different* lengths, then construct
MultiAreaEmitter with legacy off vs on and assert the per-material emitter
count differs.

Run from repo root:
    python tests/test_emitter_legacy_swap.py
"""

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.emitter import MultiAreaEmitter  # noqa: E402


def _make_pose(light_id: int, turn_angle: float = 0.0):
    """One scan_log entry with the minimal fields read_light_transforms uses."""
    return {
        "light_id": light_id,
        "turn_angle": turn_angle,
        "rotation_matrix_light": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "position_light": [0.0, 0.0, 0.5],
    }


def _build_emitter_cfg(folder_path: str, training_list_path: str, *, legacy: bool, replace_list_path: str | None):
    """Minimal cfg matching the fields MultiAreaEmitter actually reads."""
    cfg_dict = {
        "folder_path": folder_path,
        "training_list_path": training_list_path,
        "radius": 0.007,
        "fwhm_deg": 115.0,
        "radiance": [1.0, 1.0, 1.0],
        "turntable": {
            "center": [0.16084722, -0.11011424, -0.021],
            "axis": [-0.00876202, -0.01346449, 0.99987096],
        },
        "R_l2g": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "t_l2g": [0.0, -0.102, 0.02112],
        "base2_to_base1": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    if legacy:
        cfg_dict["legacy_swap_indexing"] = True
        if replace_list_path is not None:
            cfg_dict["replace_list_path"] = replace_list_path
    return OmegaConf.create(cfg_dict)


class TestEmitterLegacySwap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("MultiAreaEmitter hard-codes device='cuda'; skipping on CPU-only host.")

    def setUp(self):
        # Fresh fixture per test so failures are isolated.
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        # Slot 23: post-swap holds D500's good data → 1 pose.
        # Slot 500: post-swap holds D23's parked data → 3 poses (different length).
        # We use length to distinguish which folder was actually read.
        slot_23 = self.root / "23"
        slot_500 = self.root / "500"
        slot_23.mkdir()
        slot_500.mkdir()
        (slot_23 / "scan_log.json").write_text(json.dumps([_make_pose(0)]))
        (slot_500 / "scan_log.json").write_text(json.dumps([
            _make_pose(0), _make_pose(1, turn_angle=10.0), _make_pose(2, turn_angle=20.0),
        ]))

        self.replace_list_path = self.root / "replace_list.json"
        self.replace_list_path.write_text(json.dumps({
            "records": [
                {"backup_id": 500, "replaces": 23, "reason": "test"},
            ],
        }))

        self.training_list_path = self.root / "training_list.txt"
        self.training_list_path.write_text("23\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _build(self, *, legacy: bool, replace_list_path: str | None = None):
        cfg = _build_emitter_cfg(
            folder_path=str(self.root),
            training_list_path=str(self.training_list_path),
            legacy=legacy,
            replace_list_path=replace_list_path,
        )
        return MultiAreaEmitter(cfg)

    def test_legacy_off_reads_slot_folder(self):
        """Default behavior: emitter reads root/23/scan_log.json → 1 pose."""
        emitter = self._build(legacy=False)
        self.assertEqual(emitter.num_emitters_per_material.tolist(), [1])
        # mat_id_to_idx[23] should index the one (and only) material.
        self.assertEqual(int(emitter.mat_id_to_idx[23].item()), 0)

    def test_legacy_on_reads_partner_folder(self):
        """With legacy=True, emitter reads root/500/scan_log.json → 3 poses."""
        emitter = self._build(legacy=True)
        self.assertEqual(emitter.num_emitters_per_material.tolist(), [3])
        # mat_id (downstream indexing) is unchanged: still slot 23.
        self.assertEqual(int(emitter.mat_id_to_idx[23].item()), 0)

    def test_legacy_on_no_overlap_is_noop(self):
        """Slot not in replace_list → legacy mode falls through to slot folder.

        Mirrors training_list_442 case: zero swapped IDs ⇒ legacy=True is a no-op."""
        # Add an unswapped slot 7 to fixture and use a list of just 7.
        slot_7 = self.root / "7"
        slot_7.mkdir()
        (slot_7 / "scan_log.json").write_text(json.dumps([_make_pose(0), _make_pose(1)]))
        tl = self.root / "tl_unswapped.txt"
        tl.write_text("7\n")
        cfg = _build_emitter_cfg(
            folder_path=str(self.root),
            training_list_path=str(tl),
            legacy=True,
            replace_list_path=None,
        )
        emitter = MultiAreaEmitter(cfg)
        self.assertEqual(emitter.num_emitters_per_material.tolist(), [2])

    def test_legacy_on_missing_replace_list_warns_and_falls_through(self):
        """If replace_list.json is missing, behave as if no remap."""
        # Point to a nonexistent path explicitly; legacy stays True.
        bogus = str(self.root / "does_not_exist.json")
        emitter = self._build(legacy=True, replace_list_path=bogus)
        # Slot 23 read directly → 1 pose.
        self.assertEqual(emitter.num_emitters_per_material.tolist(), [1])

    def test_legacy_off_with_replace_list_present_does_not_remap(self):
        """replace_list.json on disk but flag off ⇒ no remap (Job 2 case)."""
        emitter = self._build(legacy=False)
        # Reads slot 23 directly even though replace_list.json sits next to it.
        self.assertEqual(emitter.num_emitters_per_material.tolist(), [1])

    def test_yaml_interpolation_path(self):
        """End-to-end: simulate Hydra resolving ${data.legacy_swap_indexing}.

        Verifies the same behavior the production yaml provides
        (cfg.legacy_swap_indexing = ${data.legacy_swap_indexing}) without
        booting Hydra — a parent cfg with data.* and a renderer.emitter that
        references it via interpolation gets resolved before construction."""
        emitter_template = OmegaConf.to_container(_build_emitter_cfg(
            folder_path=str(self.root),
            training_list_path=str(self.training_list_path),
            legacy=False,
            replace_list_path=None,
        ), resolve=True)
        # Replace the two fields with interpolation strings, mirroring yaml.
        emitter_template["legacy_swap_indexing"] = "${data.legacy_swap_indexing}"
        emitter_template["replace_list_path"] = "${data.replace_list_path}"

        parent_cfg = OmegaConf.create({
            "data": {
                "legacy_swap_indexing": True,
                "replace_list_path": str(self.replace_list_path),
            },
            "renderer": {"emitter": emitter_template},
        })

        emitter = MultiAreaEmitter(parent_cfg.renderer.emitter)
        # Slot 23 with legacy True ⇒ partner 500 read ⇒ 3 poses.
        self.assertEqual(emitter.num_emitters_per_material.tolist(), [3])


if __name__ == "__main__":
    # Reduce torch noise so test output is readable.
    torch.set_printoptions(precision=4)
    unittest.main(verbosity=2)
