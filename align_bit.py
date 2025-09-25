import os
import re

folder = "/media/raid/cloth/No_cable_capture_Sep22/Axis_estimation_Sep22/images"  # change to your folder path if needed

for fname in os.listdir(folder):
    if fname.startswith("scan-") and fname.endswith(".png"):
        # extract the ID after "scan-"
        match = re.match(r"scan-(\d+)_(.*)", fname)
        if match:
            old_id, rest = match.groups()
            new_id = f"{int(old_id):04d}"  # zero-pad to 4 digits
            new_name = f"scan-{new_id}_{rest}"
            old_path = os.path.join(folder, fname)
            new_path = os.path.join(folder, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {fname} -> {new_name}")
