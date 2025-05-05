from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import threading
import time
from smbus2 import SMBus, i2c_msg

app = Flask(__name__)

# Camera initialization
def init_camera():
    # Try different camera indices and backends
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

# Load YOLO model
model = YOLO("/home/team1/Desktop/chicken monitoring/detection/best.pt") if cap else None

# Global variables for camera/detection
latest_frame = None
lock = threading.Lock()
detection_active = True
open_beak_count = 0

# SHT20 Sensor configuration
SHT20_I2C_ADDR = 0x40
TRIG_TEMP_NOHOLD = 0xF3
TRIG_HUMI_NOHOLD = 0xF5
SOFT_RESET = 0xFE

# Global variables for sensor
current_temp = 0.0
current_hum = 0.0
sensor_error = None

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
                    results = model(frame, verbose=False)
                    # Count open beaks (assuming class 0 is 'open_beak')
                    current_count = sum(1 for box in results[0].boxes if box.cls == 0)
                    open_beak_count = current_count
                    
                    annotated_frame = results[0].plot()  # This keeps only the detection boxes
                    latest_frame = annotated_frame
                except Exception as e:
                    print(f"Detection error: {e}")
                    latest_frame = frame
        else:
            with lock:
                latest_frame = frame
                # Reset count when detection is off
                open_beak_count = 0
                
        ret, buffer = cv2.imencode('.jpg', latest_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# SHT20 Sensor functions
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

# Start sensor thread
sensor_thread = threading.Thread(target=sensor_loop, daemon=True)
sensor_thread.start()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if not cap:
        return "Camera not available", 503
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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
