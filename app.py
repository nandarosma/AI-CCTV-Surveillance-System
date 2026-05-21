from ultralytics import YOLO
import cv2
import os

# ==========================================
# LOAD YOLOv8 MODEL
# ==========================================
model = YOLO("yolov8n.pt")

# ==========================================
# VIDEO SOURCE
# ==========================================
video_path = "cctv.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video tidak bisa dibuka.")
    exit()

# ==========================================
# GET VIDEO PROPERTIES
# ==========================================
frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = int(
    cap.get(cv2.CAP_PROP_FPS)
)

# ==========================================
# OUTPUT VIDEO
# ==========================================
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(
    "output_ai_cctv.avi",
    fourcc,
    fps,
    (frame_width, frame_height)
)

# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================
os.makedirs(
    "screenshots",
    exist_ok=True
)

# ==========================================
# TARGET OBJECTS
# ==========================================
target_classes = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "train"
]

# ==========================================
# OBJECT COLORS
# ==========================================
colors = {
    "person": (0, 255, 0),
    "bicycle": (0, 255, 255),
    "car": (255, 0, 0),
    "motorcycle": (255, 255, 0),
    "bus": (255, 0, 255),
    "truck": (0, 165, 255),
    "train": (0, 0, 255)
}

# ==========================================
# TOTAL COUNTERS
# ==========================================
total_counts = {
    "person": 0,
    "bicycle": 0,
    "car": 0,
    "motorcycle": 0,
    "bus": 0,
    "truck": 0,
    "train": 0
}

print("AI CCTV Surveillance Running...")

# ==========================================
# MAIN LOOP
# ==========================================
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print("Video selesai.")
        break

    # ==========================================
    # YOLO DETECTION
    # ==========================================
    results = model(
        frame,
        conf=0.25
    )

    # Copy frame
    annotated_frame = frame.copy()

    # Current frame counter
    frame_counts = {
        "person": 0,
        "bicycle": 0,
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
        "train": 0
    }

    # ==========================================
    # DETECTION LOOP
    # ==========================================
    for result in results:

        for box in result.boxes:

            # Class ID
            cls = int(box.cls[0])

            # Confidence
            conf = float(box.conf[0])

            # Object Name
            class_name = model.names[cls]

            # Skip object
            if class_name not in target_classes:
                continue

            # Bounding Box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Box Color
            color = colors[class_name]

            # ==========================================
            # COUNT OBJECTS
            # ==========================================
            frame_counts[class_name] += 1
            total_counts[class_name] += 1

            # ==========================================
            # DRAW BOUNDING BOX
            # ==========================================
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # ==========================================
            # LABEL
            # ==========================================
            label = (
                f"{class_name} "
                f"{conf:.2f}"
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # ==========================================
    # DASHBOARD PANEL
    # ==========================================
    cv2.rectangle(
        annotated_frame,
        (10, 10),
        (430, 330),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        annotated_frame,
        "AI CCTV Object Detection",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    y_position = 80

    for object_name in frame_counts:

        text = (
            f"{object_name.capitalize()}: "
            f"{frame_counts[object_name]}"
        )

        cv2.putText(
            annotated_frame,
            text,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        y_position += 35

    # ==========================================
    # SAVE VIDEO OUTPUT
    # ==========================================
    out.write(annotated_frame)

    # ==========================================
    # DISPLAY VIDEO
    # ==========================================
    cv2.imshow(
        "AI CCTV Surveillance",
        annotated_frame
    )

    # ==========================================
    # PRESS Q TO EXIT
    # ==========================================
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Program dihentikan.")
        break

# ==========================================
# RELEASE RESOURCES
# ==========================================
cap.release()
out.release()

cv2.destroyAllWindows()

# ==========================================
# FINAL REPORT
# ==========================================
print("\n========== FINAL REPORT ==========")

for object_name in total_counts:

    print(
        f"Total {object_name.capitalize():12}: "
        f"{total_counts[object_name]}"
    )

print("\nOutput video saved:")
print("output_ai_cctv.avi")

print("==================================")