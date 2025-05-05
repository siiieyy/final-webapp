from flask import Flask, render_template, jsonify, Response
from smbus2 import SMBus, i2c_msg
import time
import threading
import cv2
from detection.model import detect_open_beak_from_frame  # Assuming your model is here

app = Flask(__name__)

# ===================== Sensor Configuration =====================
SHT20_I2C_ADDR = 0x40
TRIG_TEMP_NOHOLD = 0xF3
TRIG_HUMI_NOHOLD = 0xF5
SOFT_RESET = 0xFE

current_temp = 0.0
current_hum = 0.0
sensor_error = None

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

# ===================== Open Beak Detection =====================
camera = cv2.VideoCapture(0)
open_beak_state = {"count": 0}

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        count, annotated_frame = detect_open_beak_from_frame(frame)
        open_beak_state["count"] = count
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ===================== Routes =====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_sensor_data():
    return jsonify({
        'temperature': current_temp,
        'humidity': current_hum,
        'error': sensor_error,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(open_beak_state)

# ===================== App Start =====================
if __name__ == '__main__':
    sensor_thread = threading.Thread(target=sensor_loop)
    sensor_thread.daemon = True
    sensor_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=True)
