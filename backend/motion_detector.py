
import cv2

def detect_motion(frames, frame_idx, threshold=25, min_area=100):
    """
    Detect motion in the current frame by comparing with previous frame.

    Args:
        frames: List of video frames
        frame_idx: Index of the current frame
        threshold: Threshold for frame difference detection
        min_area: Minimum contour area to consider

    Returns:
        List of bounding boxes for detected motion regions
    """
    # We need at least 2 frames to detect motion
    if frame_idx < 1 or frame_idx >= len(frames):
        return []

    current_frame = frames[frame_idx]
    prev_frame = frames[frame_idx - 1]

    frame_to_show = current_frame.copy()

    motion_boxes = []

    current_frame, prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_frame = cv2.GaussianBlur(current_frame, (5, 5), 0)
    prev_frame = cv2.GaussianBlur(prev_frame, (5, 5), 0)
    frame_diff = cv2.absdiff(current_frame, prev_frame)
    _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    #thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            motion_boxes.append((x, y, w, h))
            cv2.rectangle(frame_to_show, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return motion_boxes
