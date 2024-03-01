from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import aiofiles
import os
from pathlib import Path
import json
from jinja2 import Environment, FileSystemLoader
import uvicorn
from fastapi.responses import JSONResponse

# Create a JSON file with tracking results
# Define the Jinja2 environment
jinja_env = Environment(loader=FileSystemLoader("templates"))

# Set the environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize FastAPI
app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates directory
templates = Jinja2Templates(directory="templates")

# Load the exported ONNX model
onnx_model = YOLO('yolov8n.onnx')

# Directory to save uploaded videos
video_directory = Path("uploaded_videos")

# Chunk size for reading video files
CHUNK_SIZE = 1024 * 1024  # 1 MB

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload/image", response_class=HTMLResponse)
async def upload_image_form(request: Request):
    return templates.TemplateResponse("upload_image.html", {"request": request})

@app.post("/upload/video", response_class=HTMLResponse)
async def upload_video(video: UploadFile = File(...)):
    # Ensure the directory exists, create it if it doesn't
    video_directory.mkdir(parents=True, exist_ok=True)

    # Define the path where the video will be saved
    video_path = video_directory / video.filename

    # Open the video file in binary write mode asynchronously and write the video data
    async with aiofiles.open(video_path, "wb") as buffer:
        while True:
            # Read the video data in chunks asynchronously
            chunk = await video.read(CHUNK_SIZE)
            if not chunk:
                break
            # Write the chunk to the file
            await buffer.write(chunk)

    # Perform object tracking on the uploaded video
    results = onnx_model.track(str(video_path), save=True)

    # Process tracking results
    processed_results = []
    for result in results:
        # Check if track method outputs are available and then process each detection
        if hasattr(result, 'boxes') and len(result.boxes):
            tracked_detections = result.boxes  # Get the detections with tracking

            for det in tracked_detections:
                cls_id = int(det.cls)  # Extract class ID
                conf = float(det.conf)  # Extract confidence score
                xyxy = det.xyxy[0].tolist()  # Extract bounding box coordinates

                # Placeholder depth estimation using basic geometric principles
                width = xyxy[2] - xyxy[0]
                height = xyxy[3] - xyxy[1]
                depth = 1000 / (width * height)  # Placeholder depth calculation

                # Get the class name from model.names
                class_name = result.names[cls_id]

                # Add extracted information to the processed results
                processed_results.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "bounding_box": xyxy,
                    "width": width,
                    "height": height,
                })
    
    # Create a JSON file with tracking results
    json_results_path = video_directory / f"{video.filename}.json"
    open_path = str(json_results_path)
    with open(open_path, "w") as json_file:
        json.dump(processed_results, json_file, indent=4)

    # Return the processed JSON data
    return JSONResponse(content=processed_results)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
