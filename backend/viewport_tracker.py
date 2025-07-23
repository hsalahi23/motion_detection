"""
Viewport tracking functions for creating a smooth "virtual camera".
"""

# def calculate_region_of_interest(motion_boxes, frame_shape):
#     """
#     Calculate the primary region of interest based on motion boxes.
#
#     Args:
#         motion_boxes: List of motion detection bounding boxes
#         frame_shape: Shape of the video frame (height, width)
#
#     Returns:
#         Tuple (x, y, w, h) representing the region of interest center point and dimensions
#     """
#
#     if not motion_boxes:
#         # If no motion is detected, use the center of the frame
#         height, width = frame_shape[:2]
#         return (width // 2, height // 2, 0, 0)
#
#     total_area = 0
#     sum_cx = 0
#     sum_cy = 0
#     max_w = 0
#     max_h = 0
#
#     for (x, y, w, h) in motion_boxes:
#         area = w * h
#         cx = x + w // 2
#         cy = y + h // 2
#
#         sum_cx += cx * area
#         sum_cy += cy * area
#         total_area += area
#
#         max_w = max(max_w, w)
#         max_h = max(max_h, h)
#
#     # Compute weighted average center
#     avg_cx = int(sum_cx / total_area)
#     avg_cy = int(sum_cy / total_area)
#
#     return (avg_cx, avg_cy, max_w, max_h)


import numpy as np
from scipy.spatial.distance import cdist

def calculate_region_of_interest(motion_boxes, frame_shape, distance_threshold=50):
    """
    Calculate the primary region of interest based on motion boxes by combining nearby boxes.

    Args:
        motion_boxes: List of bounding boxes [(x, y, w, h), ...]
        frame_shape: Shape of the video frame (height, width)
        distance_threshold: Max distance between box centers to consider them as nearby

    Returns:
        Tuple (x, y, w, h) representing the region of interest
    """
    if not motion_boxes:
        height, width = frame_shape[:2]
        return (width // 2, height // 2, 0, 0)

    # Step 1: Compute box centers
    centers = np.array([[x + w / 2, y + h / 2] for x, y, w, h in motion_boxes])

    # Step 2: Group boxes based on distance between centers
    # Use a simple greedy clustering approach
    clusters = []
    unvisited = set(range(len(centers)))

    while unvisited:
        current = unvisited.pop()
        cluster = {current}
        to_visit = {current}

        while to_visit:
            idx = to_visit.pop()
            dists = cdist([centers[idx]], centers[list(unvisited)])
            nearby_indices = [list(unvisited)[i] for i, d in enumerate(dists[0]) if d < distance_threshold]
            for ni in nearby_indices:
                to_visit.add(ni)
                unvisited.remove(ni)
                cluster.add(ni)

        clusters.append(list(cluster))

    # Step 3: Find the largest cluster
    largest_cluster = max(clusters, key=len)

    # Step 4: Combine boxes in the largest cluster
    x_vals = []
    y_vals = []
    x2_vals = []
    y2_vals = []
    for idx in largest_cluster:
        x, y, w, h = motion_boxes[idx]
        x_vals.append(x)
        y_vals.append(y)
        x2_vals.append(x + w)
        y2_vals.append(y + h)

    x_min = min(x_vals)
    y_min = min(y_vals)
    x_max = max(x2_vals)
    y_max = max(y2_vals)

    roi_x = x_min
    roi_y = y_min
    roi_w = x_max - x_min
    roi_h = y_max - y_min

    return (roi_x, roi_y, roi_w, roi_h)


def track_viewport(frames, motion_results, viewport_size, smoothing_factor=0.3):
    """
    Track viewport position across frames with smoothing.

    Args:
        frames: List of video frames
        motion_results: List of motion detection results for each frame
        viewport_size: Tuple (width, height) of the viewport
        smoothing_factor: Factor for smoothing viewport movement (0-1)
                          Lower values create smoother movement

    Returns:
        List of viewport positions for each frame as (x, y) center coordinates
    """
    viewport_positions = []

    if frames:
        height, width = frames[0].shape[:2]
        prev_x, prev_y = width // 2, height // 2
    else:
        return []

    for i, frame in enumerate(frames):
        motion_boxes = motion_results[i]

        roi_x, roi_y, roi_w, roi_h = calculate_region_of_interest(motion_boxes, frame.shape)

        # If no motion detected, use previous viewport position
        if roi_w == 0 and roi_h == 0:
            viewport_x = prev_x
            viewport_y = prev_y
        else:
            viewport_x = roi_x
            viewport_y = roi_y

        # Apply smoothing
        viewport_x = int(smoothing_factor * viewport_x + (1 - smoothing_factor) * prev_x)
        viewport_y = int(smoothing_factor * viewport_y + (1 - smoothing_factor) * prev_y)

        # Ensure viewport stays within frame boundaries
        vp_width, vp_height = viewport_size
        viewport_x = max(vp_width // 2, min(viewport_x, frame.shape[1] - vp_width // 2))
        viewport_y = max(vp_height // 2, min(viewport_y, frame.shape[0] - vp_height // 2))

        prev_x, prev_y = viewport_x, viewport_y

        viewport_positions.append((viewport_x, viewport_y))

    return viewport_positions
