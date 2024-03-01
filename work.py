@app.post("/upload/video")
async def upload_video(video: UploadFile = File(...)):
    # Ensure the directory exists, create it if it doesn't
    VIDEO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define the path where the video will be saved
    video_path = VIDEO_UPLOAD_DIR / video.filename
    
    # Open the video file in binary write mode and write the video data
    with open(video_path, "wb") as buffer:
        buffer.write(await video.read())
    results = onnx_model(video_path)












    # Process tracking results
    processed_results = []
    unique_class_ids = set()  # Set to store unique class IDs encountered
    
    for result in results:
        # Check if track method outputs are available and then process each detection
        if hasattr(result, 'boxes') and len(result.boxes):
            tracked_detections = result.boxes  # Get the detections with tracking
    
            for det in tracked_detections:
                cls_id = int(det.cls)  # Extract class ID
    
                # Check if the class ID is already in the set
                if cls_id not in unique_class_ids:
                    unique_class_ids.add(cls_id)  # Add the class ID to the set
                    conf = float(det.conf)  # Extract confidence score
                    xyxy = det.xyxy[0].tolist()  # Extract bounding box coordinates
    
                    # Placeholder depth estimation using basic geometric principles
                    # Here, we assume the depth is inversely proportional to the area of the bounding box
                    # You can replace this with a more accurate depth estimation method if available
                    width = xyxy[2] - xyxy[0]
                    height = xyxy[3] - xyxy[1]
                    depth = 1000 / (width * height)  # Placeholder depth calculation
    
                    # Get the class name from model.names
                    class_name = result.names[cls_id]
    
                    # Add extracted information to the processed results
                    processed_results.append({
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": conf,
                        # "track_id": track_id,
                        "bounding_box": xyxy
                    })
    
    # Return the tracking results as JSON
    return {"message": processed_results}
    