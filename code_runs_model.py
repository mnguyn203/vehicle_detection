from ultralytics import YOLO
import cv2, time

model = YOLO("best (1).pt")

model.to("cuda")

cap = cv2.VideoCapture("youtube_testing.mp4")

cv2.namedWindow("YOLOv8_detect", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8_detect", 650, 400)
prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(
        # frame,
        # conf=0.4,
        # iou=0.5,
        # imgsz=640,
        # device=0,
        # half=True,
        # # verbose=False

        frame,
        conf=0.5,
        iou=0.6,
        imgsz=640,
        device=0,
        half = True,
        persist=True,
        tracker="bytetrack.yaml"
    )

    annotated = results[0].plot()

    fps = 1 / (time.time() - prev)
    prev = time.time()

    cv2.putText(
        annotated,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0), 2
    )

    cv2.imshow("YOLOv8_detect", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
