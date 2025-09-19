import os
import cv2
import imageio
from glob import glob
from tqdm import tqdm
import argparse

def generate_video_and_gif(output_folder, video_name="results_video.mp4", gif_name="results_video.gif", fps=20):
    # Find matching PNGs
    # Sort by view number instead of string
    result_paths = sorted(glob(os.path.join(output_folder, "result_view_*.png")), 
                         key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    if len(result_paths) == 0:
        print(f"[!] Missing result images in {output_folder}")
        return

    # Read size from first images
    first_result = cv2.imread(result_paths[0])
    height, width, _ = first_result.shape

    # Create frame size
    frame_size = (width, height)

    # === Write MP4 ===
    mp4_path = os.path.join(output_folder, video_name)
    writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    # === Prepare GIF frames ===
    gif_path = os.path.join(output_folder, gif_name)
    gif_frames = []

    print(f"[*] Writing result video & gif for {output_folder}...")

    for result_path in tqdm(result_paths, desc="Processing frames"):
        # Read frame
        result_frame = cv2.imread(result_path)
        
        writer.write(result_frame)

        # Convert BGR to RGB for GIF
        rgb_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        gif_frames.append(rgb_frame)

    writer.release()
    imageio.mimsave(gif_path, gif_frames, fps=fps)
    print(f"[✓] Saved: {mp4_path}, {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, required=True,
                        help='Path to root folder containing material subfolders like 0.10_0.90')
    args = parser.parse_args()

    subfolders = [f.path for f in os.scandir(args.root_folder) if f.is_dir()]
    for folder in subfolders:
        print(f"Processing folder: {folder}")
        generate_video_and_gif(folder, video_name="comparison_video.mp4",
                          gif_name="comparison_video.gif", fps=20)
