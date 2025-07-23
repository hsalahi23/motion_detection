import asyncio

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from backend.frame_processor import process_video
from backend.motion_detector import detect_motion
from backend.viewport_tracker import track_viewport
from backend.visualizer import visualize_results
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-video/")
async def process_video_endpoint(file: UploadFile = File(...)):
    temp_id = str(uuid.uuid4())
    upload_dir = f"uploads/{temp_id}"
    output_dir = f"outputs/{temp_id}"
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    # Save uploaded file
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process the video
    frames = process_video(file_path, target_fps=30)
    motion_results = [detect_motion(frames, i) for i in range(len(frames))]
    viewport_positions = track_viewport(frames, motion_results, (720, 480))
    visualize_results(frames, motion_results, viewport_positions, (720, 480), output_dir)

    # Return the generated video
    output_video_path = os.path.join(output_dir, "motion_detection.mp4")
    return FileResponse(output_video_path, media_type="video/mp4", filename="motion_detection.mp4")

if __name__ == "__main__":
    asyncio.run(process_video_endpoint())