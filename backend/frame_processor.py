import cv2

def process_video(video_path, target_fps=5, resize_dim=(1280, 720)):
    """
    Extract frames from a video at a specified frame rate.

    Args:
        video_path: Path to the video file
        target_fps: Target frames per second to extract
        resize_dim: Dimensions to resize frames to (width, height)

    Returns:
        List of extracted frames
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame interval for the target FPS
    frame_interval = max(1, int(original_fps / target_fps))

    frames = []
    frame_index = 0

    for frame in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            # Resize the frame
            resized_frame = cv2.resize(frame, resize_dim)
            frames.append(resized_frame)

        frame_index += 1

    cap.release()

    return frames



