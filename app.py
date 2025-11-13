from flask import Flask, render_template, Response, jsonify
import cv2
import ultralytics
import threading
import math
import torch

# Monkey patch torch.load untuk mengatasi masalah weights_only
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from ultralytics import YOLO

app = Flask(__name__)

# Inisialisasi YOLO
model = YOLO("yolov8n.pt")

# Variabel global
people_in = 0
people_out = 0
inside = 0
lock = threading.Lock()

# Buka webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame_width = frame.shape[1] if ret else 640
line_x = frame_width // 2  # garis vertikal di tengah

# Simpan posisi sebelumnya
positions = {}

def generate_frames():
    global people_in, people_out, inside
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Deteksi orang
        results = model(frame, stream=False, classes=[0])

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf > 0.5:
                detections.append((x1, y1, x2, y2))

        # Gambar garis tengah
        cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 255), 2)

        # Simple tracking berdasarkan jarak terdekat
        current_positions = {}
        for (x1, y1, x2, y2) in detections:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Cari ID yang paling dekat dari frame sebelumnya
            best_id = None
            min_distance = float('inf')
            
            for prev_id, (prev_cx, prev_cy) in positions.items():
                distance = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                if distance < min_distance and distance < 100:  # threshold jarak
                    min_distance = distance
                    best_id = prev_id
            
            # Jika tidak ada ID yang cocok, buat ID baru
            if best_id is None:
                best_id = len(positions) + len(current_positions)
            
            current_positions[best_id] = (cx, cy)

        # Update posisi dan deteksi crossing
        detection_map = {}
        for i, (x1, y1, x2, y2) in enumerate(detections):
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            detection_map[(cx, cy)] = (x1, y1, x2, y2)
        
        for id, (cx, cy) in current_positions.items():
            if id in positions:
                prev_cx, prev_cy = positions[id]
                
                # Deteksi lintasan hanya jika ada perubahan posisi signifikan
                if abs(cx - prev_cx) > 5:  # threshold pergerakan minimum
                    if prev_cx < line_x and cx >= line_x:
                        with lock:
                            people_in += 1
                            inside += 1
                    elif prev_cx > line_x and cx <= line_x:
                        with lock:
                            people_out += 1
                            inside = max(0, inside - 1)

            # Gambar kotak dan ID
            if (cx, cy) in detection_map:
                x1, y1, x2, y2 = detection_map[(cx, cy)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"ID {id}", (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Update positions untuk frame berikutnya
        positions.clear()
        positions.update(current_positions)

        # Tampilkan counter
        cv2.putText(frame, f'In: {people_in}  Out: {people_out}  Inside: {inside}',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Detected: {len(current_positions)}', 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Kirim frame ke dashboard
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_counts')
def get_counts():
    with lock:
        return jsonify({'in': people_in, 'out': people_out, 'inside': inside})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
