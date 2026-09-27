from ultralytics import YOLO

# Load YOLO11 object detection model
model = YOLO("yolo11n.pt")

# Detect objects in the recorded video
model.predict(
    source="input/pick_place.mp4",
    save=True,
    conf=0.25
)

print("Backpack detection test completed.")