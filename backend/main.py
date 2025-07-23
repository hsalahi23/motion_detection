
import os
import argparse

from frame_processor import process_video
from motion_detector import detect_motion
from viewport_tracker import track_viewport
from visualizer import visualize_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sports Motion Detection & Viewport Tracking"
    )
    parser.add_argument(
        "--video", type=str, required=True, help="Path to input video file"
    )
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--fps", type=int, default=5, help="Target frames per second")
    parser.add_argument(
        "--viewport_size",
        type=str,
        default="720x480",
        help="Size of viewport in format WIDTHxHEIGHT",
    )
    return parser.parse_args()


def main():
    """Main function to run the motion detection and viewport tracking pipeline."""
    args = parse_args()

    try:
        viewport_width, viewport_height = map(int, args.viewport_size.split("x"))
        viewport_size = (viewport_width, viewport_height)
    except ValueError:
        print(
            f"Invalid viewport size format: {args.viewport_size}. Using default 720x480."
        )
        viewport_size = (720, 480)

    os.makedirs(args.output, exist_ok=True)

    print(f"Processing video: {args.video}")

    frames = process_video(args.video, args.fps)
    print(f"Extracted {len(frames)} frames")

    motion_results = []
    for i, frame in enumerate(frames):
        print(f"Processing frame {i + 1}/{len(frames)}")

        # Pass the entire frames list and the current index to detect_motion
        motion_boxes = detect_motion(frames, i)
        motion_results.append(motion_boxes)

    viewport_positions = track_viewport(frames, motion_results, viewport_size)

    visualize_results(
        frames, motion_results, viewport_positions, viewport_size, args.output
    )

    print(f"Processing complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()