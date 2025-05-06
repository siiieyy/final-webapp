import sys
from pathlib import Path
from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import threading
import time
import numpy as np
from smbus2 import SMBus, i2c_msg

# Add YOLOv5 directory to path
FILE = Path(__file__).resolve()
YOLOV5_DIR = FILE.parent / 'yolov5'
sys.path.append(str(YOLOV5_DIR))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.plots import Annotator, colors

# SHT20 constants
SHT20_I2C_ADDR = 0x40
TRIG_TEMP_NOHOLD = 0xF3
TRIG_HUMI_NOHOLD = 0xF5
SOFT_RESET = 0xFE

# Sensor globals
current_temp = 0.0
current_hum = 0.0
sensor_error = None

app = Flask(__name__)

def sht20_reset():
    try:
        with SMBus(1) as bus:
            bus.write_byte(SHT20_I2C_ADDR, SOFT_RESET)
        time.sleep(0.05)
        return True
    except Exception:
        return False

def read_sht20(command):
    try:
        with SMBus(1) as bus:
            bus.write_byte(SHT20_I2C_ADDR, command)
            time.sleep(0.085 if command == TRIG_TEMP_NOHOLD else 0.030)
            read = i2c_msg.read(SHT20_I2C_ADDR, 3)
            bus.i2c_rdwr(read)
            data = list(read)
            if len(data) < 2:
                raise IOError("Incomplete data received")
            raw = (data[0] << 8) + data[1]
            raw &= 0xFFFC
            if command == TRIG_TEMP_NOHOLD:
                return round(-46.85 + 175.72 * raw / 65536.0, 2), None
            return round(-6.0 + 125.0 * raw / 65536.0, 2), None
    except Exception as e:
        return None, str(e)

def sensor_loop():
    global current_temp, current_hum, sensor_error
    if not sht20_reset():
        sensor_error = "Sensor initialization failed"
        return
    time.sleep(0.2)
    while True:
        try:
            temp, temp_err = read_sht20(TRIG_TEMP_NOHOLD)
            if temp_err:
                raise IOError(temp_err)
            hum, hum_err = read_sht20(TRIG_HUMI_NOHOLD)
            if hum_err:
                raise IOError(hum_err)
            current_temp = temp
            current_hum = hum
            sensor_error = None
        except Exception as e:
            sensor_error = f"Sensor error: {str(e)}"
            print(sensor_error)
            sht20_reset()
        time.sleep(2)

sensor_thread = threading.Thread(target=sensor_loop, daemon=True)
sensor_thread.start()

def init_camera():
    for index in [0, 1, 2]:
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"Successfully opened camera at index {index} with backend {backend}")
                return cap
            cap.release()
    raise RuntimeError("Could not open any camera")

try:
    cap = init_camera()
except RuntimeError as e:
    print(f"Camera initialization failed: {e}")
    cap = None

device = select_device('')
weights = "/home/team1/Desktop/best (1).pt"

if cap:
    model_data = torch.load(weights, map_location=device)
    model = model_data['model'].float().fuse().eval()
else:
    model = None

stride = int(model.stride.max()) if model else 32
names = model.module.names if hasattr(model, 'module') else model.names

latest_frame = None
lock = threading.Lock()
detection_active = True
open_beak_count = 0

def detect(frame):
    annotator = Annotator(frame, line_width=2, example=str(names))

    img = letterbox(frame, 640, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    count = 0  # Local counter for this frame

    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                cls_int = int(cls)
                if cls_int == 0:  # Assuming class 0 = open beak
                    count += 1
                label = f'{names[cls_int]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(cls_int, True))

    return annotator.result(), count

def generate_frames():
    global latest_frame, open_beak_count
    while cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            time.sleep(0.1)
            continue

        if detection_active and model:
            with lock:
                try:
                    processed_frame, count = detect(frame.copy())
                    open_beak_count = count  # Update the shared variable
                    latest_frame = processed_frame
                except Exception as e:
                    print(f"Detection error: {e}")
                    latest_frame = frame
                    open_beak_count = 0
        else:
            with lock:
                latest_frame = frame
                open_beak_count = 0

        ret, buffer = cv2.imencode('.jpg', latest_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if not cap:
        return "Camera not available", 503
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_active
    detection_active = not detection_active
    return {'status': 'success', 'detection_active': detection_active}

@app.route('/get_beak_count')
def get_beak_count():
    return {'count': open_beak_count}

@app.route('/get_sensor_data')
def get_sensor_data():
    return {
        'temperature': current_temp,
        'humidity': current_hum,
        'error': sensor_error
    }

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        if cap:
            cap.release()
