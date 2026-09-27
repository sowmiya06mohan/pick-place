from ultralytics import YOLO

# Load YOLO pose model
model = YOLO("yolo11n-pose.pt")

# Process the recorded video
model.predict(
    source="input/pick_place.mp4",
    save=True,
    conf=0.5
)

print("Pose detection completed successfully.")