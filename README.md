# People Counter Dashboard (YOLOv8 + Flask)

Sistem **People Counter Dashboard** ini menggunakan **YOLOv8** (Ultralytics) dan **Flask** untuk mendeteksi serta menghitung jumlah orang yang **masuk**, **keluar**, dan **berada di dalam** area secara real-time melalui kamera webcam.  

Aplikasi ini dilengkapi tampilan web (*dashboard*) yang menampilkan video streaming serta perhitungan secara dinamis.

---

## Fitur Utama
- Deteksi manusia secara real-time menggunakan model YOLOv8.
- Menghitung jumlah orang yang **masuk** dan **keluar** area.
- Menampilkan **jumlah total orang yang sedang berada di dalam**.
- Terdapat **garis vertikal pemisah area** di tampilan video.
- Dapat dijalankan secara lokal tanpa internet setelah model diunduh.

---

## ⚙️ Cara Setup & Menjalankan

```bash
# 1. Clone repository
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>

# 2. Buat dan aktifkan virtual environment
python3 -m venv venv
source venv/bin/activate   # (untuk macOS/Linux)
# venv\Scripts\activate    # (untuk Windows)

# 3. Install dependensi
pip install --upgrade pip
pip install ultralytics==8.0.196 opencv-python flask torch torchvision torchaudio

# 4. Jalankan aplikasi
python app.py
