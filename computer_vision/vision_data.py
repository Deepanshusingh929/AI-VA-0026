import cv2
import json
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break

    results = model.track(frame, persist=True)

    objects = []

    for result in results:
        boxes = result.boxes

        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = None

                objects.append({
                    "class": class_name,
                    "track_id": track_id,
                    "confidence": round(confidence, 2)
                })

    vision_data = {
        "objects": objects
    }

    print(json.dumps(vision_data, indent=2))

    annotated_frame = results[0].plot()

    cv2.imshow("AI VA-0026 - Vision Data", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()