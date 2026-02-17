#!/usr/bin/env python3
# rename_max_suffix.py
# Usage: python rename_max_suffix.py <folder>

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm: pip install tqdm")
    sys.exit(1)

# Match a suffix like "_max_65535.00" or "_max_123" at the END of the stem (before the extension)
SUFFIX_RE = re.compile(r'_max_\d+(?:\.\d+)?$')

def main():
    if len(sys.argv) != 2:
        print("Usage: python rename_max_suffix.py <folder>")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Not a directory: {folder}")
        sys.exit(1)

    # Build the plan quickly using os.scandir (low overhead)
    plans = []
    with os.scandir(folder) as it:
        for entry in it:
            if not entry.is_file():
                continue
            name = entry.name
            stem, ext = os.path.splitext(name)
            if not ext:  # skip extensionless files
                continue
            if not SUFFIX_RE.search(stem):
                continue
            new_name = SUFFIX_RE.sub('', stem) + ext
            if new_name == name:
                continue
            src = os.path.join(folder, name)
            dst = os.path.join(folder, new_name)
            if os.path.exists(dst):
                # Collision: skip to be safe
                continue
            plans.append((src, dst))

    if not plans:
        print("No files found matching the pattern '_max_XXXXX[.XX]'")
        return

    # I/O-bound -> more threads helps; cap to avoid oversubmission
    max_workers = min(64, (os.cpu_count() or 8) * 4)

    errors = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex, \
         tqdm(total=len(plans), desc="Renaming", unit="file", dynamic_ncols=True) as pbar:

        futures = [ex.submit(_rename_one, src, dst) for src, dst in plans]
        for fut in as_completed(futures):
            ok = fut.result()
            if ok is None:
                errors += 1
            elif ok is False:
                skipped += 1
            pbar.update(1)

    done = len(plans) - errors - skipped
    print(f"\nDone. Renamed: {done}, Skipped: {skipped}, Errors: {errors}")

def _rename_one(src: str, dst: str):
    """
    Returns:
      True  -> renamed
      False -> skipped due to existing destination (race)
      None -> error
    """
    try:
        # If a new file pops up mid-run, avoid overwriting: check again atomically-ish
        if os.path.exists(dst):
            return False
        os.rename(src, dst)
        return True
    except FileNotFoundError:
        # Source vanished between planning and execution
        return None
    except FileExistsError:
        # Destination appeared in the meantime
        return False
    except Exception:
        return None

if __name__ == "__main__":
    main()
