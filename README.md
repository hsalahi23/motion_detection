## Motion Detection & Viewport Tracking

### Overview
Implementing a motion detection and viewport tracking system for sports video that identifies areas of activity and creates a smooth "virtual camera" view that follows the action.


### Project Files

#### 1. Video Processing
* Loads and processes a sports video clip
* Extracts frames at a regular interval (e.g., 5 fps)
* Image preprocessing (resize to a standard resolution)

#### 2. Motion Detection
* Implements a simple frame differencing algorithm to detect motion
* Applies basic filtering to reduce noise
* Identifies and prioritize significant movement areas

#### 3. Viewport Tracking
* Creates a "virtual camera" viewport (a rectangle of fixed size, e.g., 720x480)
* Makes the viewport track the main action area based on motion detection
* Implements basic smoothing to prevent jerky camera movements

#### 4. Results Visualization
* Creates a visualization showing:
   * Original frame with detected motion areas (bounding boxes)
   * Viewport rectangle overlaid on the original frame
   * The cropped viewport view as a separate visualization
   * Save the output as a series of images and a video
