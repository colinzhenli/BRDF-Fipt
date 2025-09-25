import json
import re
import sys
from pathlib import Path

def process_json(input_path, output_path):
    """
    Process JSON file to:
    - Add scan_id (4-digit padded)
    - Add camera_id (kept as original, not padded)
    - Rewrite the filename field to use 4-digit scan ID only
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    scan_pattern = re.compile(r"(scan-)(\d+)(_.*)")
    camera_pattern = re.compile(r"(camera-)(\d+)")

    for entry in data:
        filename = entry.get("filename", "")

        # --- Fix scan ID ---
        scan_match = scan_pattern.search(filename)
        if scan_match:
            prefix, scan_raw, rest = scan_match.groups()
            scan_id = f"{int(scan_raw):04d}"
            filename = f"{prefix}{scan_id}{rest}"
            entry["scan_id"] = scan_id
        else:
            print(f"Warning: no scan id in {filename}")
            scan_id = None

        # --- Keep camera ID as-is ---
        cam_match = camera_pattern.search(filename)
        if cam_match:
            _, cam_raw = cam_match.groups()
            camera_id = cam_raw  # no padding
            entry["camera_id"] = camera_id
        else:
            print(f"Warning: no camera id in {filename}")
            camera_id = None

        # Update filename in JSON
        entry["filename"] = filename

        print(f"Updated: {filename} (scan_id={scan_id}, camera_id={camera_id})")

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Processed {len(data)} entries → saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_json.py <input_json_path> <output_json_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not Path(input_path).exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    process_json(input_path, output_path)
