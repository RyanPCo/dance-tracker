import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Run AlphaPose on a video using demo_inference.py')
    parser.add_argument('--cfg', type=str, required=True, help='AlphaPose config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--outdir', type=str, default='output', help='Directory to save results')
    parser.add_argument('--format', type=str, default='openpose', help='Output format: coco/cmu/openpose')
    parser.add_argument('--pose_track', action='store_true', help='Enable tracking')
    parser.add_argument('--save_video', action='store_true', help='Save rendered video')
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.outdir, exist_ok=True)

    # Build the command to run demo_inference.py
    cmd = [
        "python", "scripts/demo_inference.py",
        "--cfg", args.cfg,
        "--checkpoint", args.checkpoint,
        "--video", args.video,
        "--outdir", args.outdir,
        "--format", args.format
    ]

    if args.pose_track:
        cmd.append("--pose_track")
    if args.save_video:
        cmd.append("--save_video")

    print("Running command:")
    print(" ".join(cmd))

    # Set the working directory to the AlphaPose folder,
    # so that all relative paths inside demo_inference.py resolve correctly.
    alpha_pose_dir = os.path.join(os.getcwd(), "AlphaPose")
    subprocess.call(cmd, cwd=alpha_pose_dir)

if __name__ == '__main__':
    main()
