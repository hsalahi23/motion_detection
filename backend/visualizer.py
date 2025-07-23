import os
import cv2

def visualize_results(
    frames, motion_results, viewport_positions, viewport_size, output_dir
):
    """
    Create visualization of motion detection and viewport tracking results.

    Args:
        frames: List of video frames
        motion_results: List of motion detection results for each frame
        viewport_positions: List of viewport center positions for each frame
        viewport_size: Tuple (width, height) of the viewport
        output_dir: Directory to save visualization results
    """
    # Create output directory for frames
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    viewport_dir = os.path.join(output_dir, "viewport")
    os.makedirs(viewport_dir, exist_ok=True)

    height, width = frames[0].shape[:2]

    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(output_dir, "motion_detection.mp4")
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

    viewport_video_path = os.path.join(output_dir, "viewport_tracking.mp4")
    vp_width, vp_height = viewport_size
    viewport_writer = cv2.VideoWriter(
        viewport_video_path, fourcc, 30, (vp_width, vp_height)
    )

    for idx, (frame, motions, vp_center) in enumerate(
        zip(frames, motion_results, viewport_positions)
    ):
        vis_frame = frame.copy()

        # Draw motion bounding boxes (green)
        for (x, y, w, h) in motions:
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate viewport coordinates from center
        vp_x = int(vp_center[0] - vp_width / 2)
        vp_y = int(vp_center[1] - vp_height / 2)
        vp_x = max(0, min(vp_x, width - vp_width))
        vp_y = max(0, min(vp_y, height - vp_height))

        # Draw viewport rectangle (blue)
        cv2.rectangle(
            vis_frame,
            (vp_x, vp_y),
            (vp_x + vp_width, vp_y + vp_height),
            (255, 0, 0),
            2,
        )

        # Add frame number
        cv2.putText(
            vis_frame,
            f"Frame: {idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        # Extract viewport content
        viewport_content = frame[vp_y : vp_y + vp_height, vp_x : vp_x + vp_width]

        # Save frames as images
        frame_filename = os.path.join(frames_dir, f"frame_{idx:04d}.jpg")
        vp_filename = os.path.join(viewport_dir, f"viewport_{idx:04d}.jpg")
        cv2.imwrite(frame_filename, vis_frame)
        cv2.imwrite(vp_filename, viewport_content)

        # Write frames to videos
        video_writer.write(vis_frame)
        viewport_writer.write(viewport_content)

    # Release writers
    video_writer.release()
    viewport_writer.release()
