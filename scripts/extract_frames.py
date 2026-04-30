"""
Frame extraction script.
Usage: python scripts/extract_frames.py --input data/raw_video/session1.mp4 --output data/frames/session1 --every 5
"""
import argparse
import subprocess
import os

def extract_frames(input_path, output_dir, every_n=5):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vf", f"select=not(mod(n\,{every_n}))",
        "-vsync", "vfr",
        os.path.join(output_dir, "frame_%04d.jpg")
    ]
    print(f"Extracting every {every_n}th frame from {input_path} -> {output_dir}")
    subprocess.run(cmd, check=True)
    frames = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
    print(f"Done. {len(frames)} frames extracted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Output directory for frames")
    parser.add_argument("--every", type=int, default=5, help="Extract every Nth frame (default: 5)")
    args = parser.parse_args()
    extract_frames(args.input, args.output, args.every)
