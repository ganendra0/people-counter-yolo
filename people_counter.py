from ultralytics import YOLO
import cv2

# variabel global yang akan dibaca Flask
counts = {"masuk": 0, "keluar": 0, "inside": 0}

def start_counter():
    cap = cv2.VideoCapture(0)
    model = YOLO("yolov8n.pt")
    line_x = 320  # garis vertikal tengah (ubah sesuai resolusi)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if model.names[cls] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = int((x1 + x2) / 2)

                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, "person", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # logika line crossing
                    if cx < line_x - 10:
                        counts["masuk"] += 1
                        counts["inside"] += 1
                    elif cx > line_x + 10:
                        counts["keluar"] += 1
                        counts["inside"] -= 1

        # tampilkan garis dan data
        cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 2)
        cv2.putText(frame, f"Masuk: {counts['masuk']} Keluar: {counts['keluar']} Inside: {counts['inside']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("People Counter", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
