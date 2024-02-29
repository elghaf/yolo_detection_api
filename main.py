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


model = YOLO("models/best_seg_75.pt")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload/image", response_class=HTMLResponse)
async def upload_image_form(request: Request):
    return templates.TemplateResponse("upload_image.html", {"request": request})




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
    results = model.track(video_path, show=True, save=True)
    
    # Optionally, you can return the tracking results as JSON
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)