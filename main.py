from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
import uvicorn
from fastapi import UploadFile, File
from fastapi import Header
from fastapi import Response
import pathlib
from ultralytics import YOLO
import os
import json
# Set the environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


app = FastAPI()


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates directory
templates = Jinja2Templates(directory="templates")
video_path = pathlib.Path("static/video/out.mp4")
CHUNK_SIZE = 1024

# Directory to save uploaded videos
video_directory = pathlib.Path("static/video")

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Export the model to ONNX format
model.export(format='onnx')

# Load the exported ONNX model
onnx_model = YOLO('yolov8n.onnx')
#model = YOLO("models/best_seg_75.onnx")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload/image", response_class=HTMLResponse)
async def upload_image_form(request: Request):
    return templates.TemplateResponse("upload_image.html", {"request": request})


def process_tracking_results(results):
    json_results = []
    for result in results:
        json_result = result.__dict__

        # Check if track method outputs are available and then process each detection
        if hasattr(result, 'boxes') and len(result.boxes):
            tracked_detections = result.boxes  # Get the detections with tracking

            for det in tracked_detections:
                cls_id = int(det.cls)  # Extract class ID
                conf = float(det.conf)  # Extract confidence score
                # track_id = int(det.id)  # Extract track ID
                xyxy = det.xyxy[0].tolist()  # Extract bounding box coordinates

                # Placeholder depth estimation using basic geometric principles
                # Here, we assume the depth is inversely proportional to the area of the bounding box
                # You can replace this with a more accurate depth estimation method if available
                width = xyxy[2] - xyxy[0]
                height = xyxy[3] - xyxy[1]
                depth = 1000 / (width * height)  # Placeholder depth calculation

                # Get the class name from model.names
                class_name = result.names[cls_id]

                # Create or update the entry for the object in json_result['names']
                if class_name not in json_result['names']:
                    json_result['names'][class_name] = {'count': 0, 'max_width': 0, 'max_height': 0, 'depth': []}

                # Update the count, maximum width and height, and depth for the object
                json_result['names'][class_name]['count'] += 1
                json_result['names'][class_name]['max_width'] = max(json_result['names'][class_name]['max_width'], width)
                json_result['names'][class_name]['max_height'] = max(json_result['names'][class_name]['max_height'], height)
                json_result['names'][class_name]['depth'].append(depth)
        
        json_results.append(json_result)
    
    return json_results

@app.post("/upload/video")
async def upload_video(video: UploadFile = File(...)):
    # Ensure the directory exists, create it if it doesn't
    video_directory.mkdir(parents=True, exist_ok=True)
    
    # Define the path where the video will be saved
    video_path = video_directory / video.filename
    
    # Write the video data to the specified path
    with open(video_path, "wb") as buffer:
        buffer.write(await video.read())
    
    # Perform object tracking on the uploaded video
    results = onnx_model(video_path, save=True)
    
    # Process tracking results
    processed_results = []
    for result in results:
        # Check if track method outputs are available and then process each detection
        if hasattr(result, 'boxes') and len(result.boxes):
            tracked_detections = result.boxes  # Get the detections with tracking

            for det in tracked_detections:
                cls_id = int(det["cls"])  # Extract class ID
                conf = float(det["conf"])  # Extract confidence score
                # track_id = int(det["id"])  # Extract track ID
                xyxy = det["xyxy"][0].tolist()  # Extract bounding box coordinates
                
                # Add extracted information to the processed results
                processed_results.append({
                    "class_id": cls_id,
                    "confidence": conf,
                    # "track_id": track_id,
                    "bounding_box": xyxy
                })
    
    # Return the tracking results as JSON
    return processed_results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)