import sys
import cv2
import torch
import time
import threading
import numpy as np
from smbus2 import SMBus, i2c_msg
import RPi.GPIO as GPIO
from pathlib import Path
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
from matplotlib.ticker import MaxNLocator
from queue import Queue
from itertools import cycle
import os
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer

# --- Model Setup ---
sys.path.append('/home/team1/Desktop/ssd')
class_names = ['background', 'head', 'open-mouth', 'closed-beak']
model = torch.jit.load("/home/team1/Desktop/ssd/ssd_lite_traced.pt").eval()
predictor = create_mobilenetv2_ssd_lite_predictor(model, candidate_size=200)
timer = Timer()

# --- GPIO Setup ---
RELAY_PINS = {'Exhaust Fan': 17, 'Water Pump': 27, 'Intake Fan': 23}
GPIO.setmode(GPIO.BCM)
for pin in RELAY_PINS.values():
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.HIGH)

# --- Globals ---
current_temp, current_hum, sensor_error = 0.0, 0.0, None
open_beak_history = []
open_beak_lock = threading.Lock()
sensor_readings = []
temp_data, hum_data, thi_data = [], [], []
frame_queue = Queue(maxsize=1)
detection_queue = Queue(maxsize=1)
open_beak_count = 0
previous_states = {
    'Exhaust Fan': None,
    'Water Pump': None,
    'Intake Fan': None
}
last_warning_time = 0 # To debounce warning popups
WARNING_COOLDOWN = 30 # seconds, only show warning once per minute
current_thi = 0.0
THI_ON_LEVEL  = 84.0
THI_OFF_LEVEL = 83.0
pump_active = False

# --- Sensor Setup ---
SHT20_I2C_ADDR = 0x40
TRIG_TEMP_NOHOLD = 0xF3
TRIG_HUMI_NOHOLD = 0xF5
SOFT_RESET = 0xFE

# --- Sensor Functions with Calibration ---
HUMIDITY_OFFSET = -7.0

def sht20_reset():
    try:
        with SMBus(1) as bus:
            bus.write_byte(SHT20_I2C_ADDR, SOFT_RESET)
        time.sleep(0.05)
        return True
    except:
        return False

def _check_crc(data):
    crc = 0x00
    polynomial = 0x131
    for byte in data[:2]:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ polynomial
            else:
                crc <<= 1
            crc &= 0xFF
    return crc == data[2]

def read_sht20(command):
    try:
        with SMBus(1) as bus:
            bus.write_byte(SHT20_I2C_ADDR, command)
            time.sleep(0.085 if command == TRIG_TEMP_NOHOLD else 0.03)
            read = i2c_msg.read(SHT20_I2C_ADDR, 3)
            bus.i2c_rdwr(read)
            data = list(read)

            if len(data) != 3:
                raise IOError("Incomplete data received")

            if not _check_crc(data):
                raise IOError("CRC check failed")

            raw = (data[0] << 8) + data[1]
            raw &= 0xFFFC  # Mask status bits

            if command == TRIG_TEMP_NOHOLD:
                temp = round(-46.85 + 175.72 * raw / 65536.0, 2)
                return temp, None
            else:
                raw_humidity = -6.0 + 125.0 * raw / 65536.0
                calibrated = raw_humidity + HUMIDITY_OFFSET
                calibrated = max(0.0, min(calibrated, 100.0))  # Clamp 0–100%
                print(f"[SHT20 DEBUG] Raw Humidity: {raw_humidity:.2f}% | Calibrated: {calibrated:.2f}%")
                return round(calibrated, 2), None

    except Exception as e:
        return None, str(e)

# Helper to avoid duplicate logs
def set_relay(relay_name, state, reason=" "):
    pin = RELAY_PINS[relay_name]
    desired_state_str = 'ON' if state == GPIO.LOW else 'OFF'

    if previous_states[relay_name] != desired_state_str:
        GPIO.output(pin, state)
        log_message = f"{relay_name} {desired_state_str}"
        if relay_name == 'Water Pump' and reason:
            log_message += f" - {reason}"

        add_log_with_gif(relay_name, log_message)
        log_event(log_message)  # THIS LINE WAS MISSING - adds to CSV events
        previous_states[relay_name] = desired_state_str

        if desired_state_str == 'ON':
            play_gif(relay_name)
        else:
            pause_gif(relay_name)

def compute_thi(temp, hum):
    # Standard poultry THI approximation
    thi = (1.8 * temp + 32) - ((0.55 - 0.0055 * hum) * (1.8 * temp - 26))
    return round(thi, 2)

# --- Sensor Thread ---
def sensor_loop():
    global current_temp, current_hum, current_thi, sensor_error, pump_active

    while True:
        try:
            if not sht20_reset():
                sensor_error = "Sensor init failed"
                time.sleep(1)
                continue
               
            current_thi = compute_thi(current_temp, current_hum)

            temp, err1 = read_sht20(TRIG_TEMP_NOHOLD)
            hum, err2 = read_sht20(TRIG_HUMI_NOHOLD)
           
            if err1 or err2 or temp is None or hum is None:
                raise IOError(err1 or err2 or "Invalid sensor data")

            current_temp = float(temp)
            current_hum = min(float(hum), 100.0)
            current_thi = compute_thi(current_temp, current_hum)
            sensor_error = None

            log_sensor_data()

            # Always ON: Exhaust Fan and Intake Fan
            set_relay('Exhaust Fan', GPIO.LOW, "ON")
            set_relay('Intake Fan', GPIO.LOW, "ON")

            water_pump_state = GPIO.HIGH
            reason_for_pump_state = "No Heat Stress"

            # --- STATE-BASED CONTROL ---
            if pump_active:
                if current_thi <= THI_OFF_LEVEL:
                    pump_active = False
                    water_pump_state = GPIO.HIGH
                    reason_for_pump_state = "THI Recovered(No Heat Stress)"
                else:
                    water_pump_state = GPIO.LOW
                    reason_for_pump_state = "Cooling (THI High)"

            else:
                if open_beak_count > 0 and current_thi >= THI_ON_LEVEL:
                    pump_active = True
                    water_pump_state = GPIO.LOW
                    reason_for_pump_state = "Open Beak + High THI"
                elif open_beak_count > 0 and current_thi < THI_ON_LEVEL:
                    show_warning()
                    reason_for_pump_state = "Open Beak (THI OK)"

            set_relay('Water Pump', water_pump_state, reason_for_pump_state)

        except Exception as e:
            sensor_error = str(e)
            sht20_reset()

        time.sleep(1)

def show_warning():
    global last_warning_time
    current_time = time.time()

    # Only show warning if enough time has passed since the last one
    if current_time - last_warning_time > WARNING_COOLDOWN:
        last_warning_time = current_time

        warning_popup = Toplevel(root)
        warning_popup.title("Warning")
        warning_popup.geometry("350x100") # Set a default size for the popup
        warning_popup.configure(bg="white")
       
        # Center the popup relative to the main window or screen
        root_x = root.winfo_x()
        root_y = root.winfo_y()
        root_width = root.winfo_width()
        root_height = root.winfo_height()
       
        popup_width = 350
        popup_height = 100
       
        center_x = root_x + (root_width // 2) - (popup_width // 2)
        center_y = root_y + (root_height // 2) - (popup_height // 2)
       
        warning_popup.geometry(f"{popup_width}x{popup_height}+{center_x}+{center_y}")


        Label(warning_popup,
              text="Warning: Open beak detected! Please check the chickens.",
              bg="white",
              font=("Helvetica", 12),
              wraplength=300).pack(padx=20, pady=10)
       
        warning_popup.after(5000, warning_popup.destroy)
       
        warning_popup.lift() # Bring to top of window stack
        warning_popup.attributes('-topmost', True) # Keep it on top
        warning_popup.focus_force() # Force focus to this window
        warning_popup.attributes('-topmost', False) # Allow other windows to come on top after gaining focus

# --- Log Sensor Data ---
def log_sensor_data():
    timestamp = datetime.datetime.now()
    formatted_time = timestamp.strftime("%H:%M:%S")
   
    reading = {
        'timestamp': timestamp,
        'formatted_time': formatted_time,
        'temp': current_temp,
        'hum': current_hum,
        'thi': current_thi,
        'beak_count': open_beak_count,
        'events': []
    }
    sensor_readings.append(reading)
   
    # Keep last 1000 readings (adjust as needed)
    if len(sensor_readings) > 1000:
        sensor_readings.pop(0)

    # Update graph data
    temp_data.append(current_temp)
    hum_data.append(current_hum)
    thi_data.append(current_thi)

# --- Log Event ---
def log_event(message):
    if not sensor_readings:
        log_sensor_data()  # Ensure we have at least one reading
   
    # Add event to the most recent sensor reading (for export)
    sensor_readings[-1]['events'].append(message)
   
    # Only show relay status in log box with timestamp
    #timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    #relay_status_message = f"{timestamp} - {message}"
    #log_box.insert(END, relay_status_message)
   
    # Keep only last 10 relay messages
    #if log_box.size() > 10:
    #    log_box.delete(0)
           
# --- GUI Setup ---
root = Tk()
root.title("Chicken Monitoring Dashboard")
root.geometry("1375x1000")
root.configure(bg="#1e1e2f")
root.attributes("-fullscreen", True)
root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

try:
    icon = PhotoImage(file="/home/team1/Documents/icon.png")
    root.iconphoto(True, icon)
except Exception as e:
    print(f"Icon load failed: {e}")

video_label = Label(root, bg="#1e1e2f")
video_label.place(x=20, y=20, width=800, height=500)

temp_label = Label(root, text="Temperature: -- °C", fg="cyan", bg="#1e1e2f", font=("Helvetica", 14))
temp_label.place(x=850, y=40)
hum_label = Label(root, text="Humidity: -- %", fg="lime", bg="#1e1e2f", font=("Helvetica", 14))
hum_label.place(x=850, y=80)

# --- MODIFIED: Replaced beak_label with status_label ---
status_label = Label(root, text="Status: Initializing...", fg="orange", bg="#1e1e2f", font=("Helvetica", 14, "bold"))
status_label.place(x=850, y=120)

thi_label = Label(
    root,
    text="THI: --",
    fg="red",
    bg="#1e1e2f",
    font=("Helvetica", 13, "bold")
)
thi_label.place(x=850, y=155)

timestamp_label = Label(root, text="Time: --:--", fg="white", bg="#1e1e2f", font=("Helvetica", 12))
timestamp_label.place(x=850, y=185)

log_frame = Frame(root, bg="#111")
log_frame.place(x=850, y=230, width=485, height=210)

Label(
    log_frame, text="Relay Logs", bg="#111", fg="white",
    font=("Helvetica", 13, "bold")
).pack(pady=5)

# This frame will contain GIFs + text logs (KEEP THIS)
log_container = Frame(log_frame, bg="#111")
log_container.pack(fill=BOTH, expand=True)

# Scrollable canvas setup (KEEP THIS)
canvas = Canvas(log_frame, bg="#111", highlightthickness=0, bd=0)
scrollbar = Scrollbar(
    log_frame,
    orient="vertical",
    command=canvas.yview,
    width=8,
)
scrollbar.config(
    troughcolor="#222",
    bg="#444",
    activebackground="#666",
    relief="flat"
)

scrollable_frame = Frame(canvas, bg="#111")

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side=LEFT, fill=BOTH, expand=True)
scrollbar.pack(side=RIGHT, fill=Y)

# --- Load GIF Frames ---
def load_gif_frames(path, size=(30, 30)):
    frames = []
    if not os.path.exists(path):
        print(f"❌ Missing GIF: {path}")
        return frames
    try:
        img = Image.open(path)
        while True:
            frames.append(ImageTk.PhotoImage(img.copy().resize(size)))
            img.seek(len(frames))
    except EOFError:
        pass
    return frames

# Load your animations
exhaust_frames = load_gif_frames("/home/team1/Documents/EFAN.gif")
intake_frames  = load_gif_frames("/home/team1/Documents/INTAKE1.gif")
pump_frames    = load_gif_frames("/home/team1/Documents/cent.gif")

gif_states = {
    'Exhaust Fan': {'frames': exhaust_frames, 'running': False, 'job': None, 'index': 0, 'last_frame': None},
    'Intake Fan':  {'frames': intake_frames,  'running': False, 'job': None, 'index': 0, 'last_frame': None},
    'Water Pump':  {'frames': pump_frames,    'running': False, 'job': None, 'index': 0, 'last_frame': None}
}

# Keep label refs so PhotoImages aren't GC'd
relay_log_rows = {}

def add_log_with_gif(relay_name, message):
    """Add a new scrolling log entry with GIF icon."""
    timestamp = time.strftime("%H:%M:%S")
    text = f"[{timestamp}] {message}"

    # Create new row in the scrollable frame
    row = Frame(scrollable_frame, bg="#111")
    row.pack(anchor='w', pady=2, padx=5, fill='x')

    gif_label = Label(row, bg="#111")
    gif_label.pack(side=LEFT, padx=5)

    text_label = Label(row, text=text, fg="white", bg="#111", font=("Courier", 10), anchor='w')
    text_label.pack(side=LEFT)

    # Set first frame as default image
    frames = gif_states[relay_name]['frames']
    if frames:
        first_frame = frames[0]
        gif_label.config(image=first_frame)
        gif_label.image = first_frame
        gif_states[relay_name]['last_frame'] = first_frame

    relay_log_rows[row] = {'gif_label': gif_label, 'relay_name': relay_name}

    # Auto-scroll to newest log
    root.after(100, lambda: canvas.yview_moveto(1.0))

def play_gif(relay_name):
    """Animate all GIFs for the given relay name (latest entry stays moving)."""
    state = gif_states[relay_name]
    if not state['frames']:
        return
    if not state['running']:
        state['running'] = True

        def animate():
            if state['running']:
                frame = state['frames'][state['index']]
                state['last_frame'] = frame
                # Update only the most recent log of this relay
                for row, data in relay_log_rows.items():
                    if data['relay_name'] == relay_name:
                        data['gif_label'].config(image=frame)
                        data['gif_label'].image = frame
                state['index'] = (state['index'] + 1) % len(state['frames'])
                state['job'] = root.after(100, animate)
        animate()

def pause_gif(relay_name):
    """Pause all GIFs of a relay and freeze last frame."""
    state = gif_states[relay_name]
    state['running'] = False
    if state['job']:
        root.after_cancel(state['job'])
        state['job'] = None
    if state['last_frame']:
        for row, data in relay_log_rows.items():
            if data['relay_name'] == relay_name:
                data['gif_label'].config(image=state['last_frame'])
                data['gif_label'].image = state['last_frame']

# --- Download Logs Function ---
def export_logs():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chicken_monitor_log_{timestamp}.csv"
    try:
        with open(filename, "w") as f:
            f.write("Timestamp,Temperature (°C),Humidity (%),THI,Open Beak Count,Events\n")
            for entry in sensor_readings:
                events = " | ".join(entry['events']) if entry['events'] else "No Events"
                f.write(f"{entry['formatted_time']},{entry['temp']:.1f},{entry['hum']:.1f},{entry.get('thi', 0):.1f},{entry['beak_count']},\"{events}\"\n")

        popup = Toplevel(root)
        popup.title("Export Complete")
        popup.configure(bg="white")
        Label(popup, text=f"Successfully exported all data to:\n{filename}", bg="white", font=("Helvetica", 10)).pack(padx=20, pady=10)
        Label(popup, text="✔ Export Completed", fg="green", bg="white", font=("Helvetica", 10, "bold")).pack()
       
        # --- ADD THIS BUTTON ---
        popup.after(2000, popup.destroy)

    except Exception as e:
        error_popup = Toplevel(root)
        error_popup.title("Export Failed")
        error_popup.configure(bg="white")
        Label(error_popup, text="Failed to export logs:", bg="white", font=("Helvetica", 10)).pack(padx=20, pady=5)
        Label(error_popup, text=str(e), fg="red", bg="white", font=("Courier", 8)).pack()
        Label(error_popup, text="✖ Export Failed", fg="red", bg="white", font=("Helvetica", 10, "bold")).pack()

        error_popup.after(3000, error_popup.destroy)
       
# --- Download Button ---
download_btn = Button(root,
                      text="Download Logs",
                      command=export_logs,
                      bg="#4CAF50",
                      fg="white",
                      font=("Helvetica", 10))
download_btn.place(x=850, y=455, width=150, height=32)

# --- Plotting Graphs ---
fig_temp, ax_temp = plt.subplots(figsize=(6, 4), dpi=100, facecolor='#1e1e2f')
fig_hum, ax_hum = plt.subplots(figsize=(6, 4), dpi=100, facecolor='#1e1e2f')
fig_beak, ax_thi = plt.subplots(figsize=(6, 4), dpi=100, facecolor='#1e1e2f')


canvas_temp = FigureCanvasTkAgg(fig_temp, master=root)
canvas_temp.get_tk_widget().place(x=20, y=600)
canvas_hum = FigureCanvasTkAgg(fig_hum, master=root)
canvas_hum.get_tk_widget().place(x=670, y=600)
canvas_beak = FigureCanvasTkAgg(fig_beak, master=root)
canvas_beak.get_tk_widget().place(x=1320, y=600)

def update_graphs():
    for ax, data, label, color, canvas, ylabel in [
        (ax_temp, temp_data, "Temperature (°C)", 'cyan', canvas_temp, "Temperature (°C)"),
        (ax_hum, hum_data, "Humidity (%)", 'lime', canvas_hum, "Humidity (%)"),
        (ax_thi, thi_data, "THI Index", 'red', canvas_beak, "THI")
    ]:
        if data:
            y_min_val = min(data[-30:]) if data[-30:] else 0
            y_max_val = max(data[-30:]) if data[-30:] else 1
            y_min = max(0, int(y_min_val) - 1)
            y_max = int(y_max_val) + 1
       
        ax.clear()
        ax.set_facecolor('#1e1e2f')
        ax.plot(range(len(data[-30:])), data[-30:], color=color, marker='o', linewidth=2.0, markersize=4.0, label=label)
        ax.set_title(label, color='white', fontsize=10)
        ax.set_xlabel("Time(s)", color='white', fontsize=8)
        ax.set_ylabel(ylabel, color='white', fontsize=8)
        ax.tick_params(axis='both', which='both', colors='white', labelsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

       
        num_samples = len(data[-30:])
        ax.set_xlim(0, max(1, num_samples - 1))

        if data:
            ax.set_ylim(y_min, y_max)
       
        ax.grid(True, color='#444444', linestyle='--', alpha=0.7)
        ax.legend(loc="upper right", facecolor="#1e1e2f", labelcolor='white', fontsize=8)
        canvas.draw()

    root.after(3000, update_graphs)

# --- Camera & Detection ---
cap = cv2.VideoCapture(0)

# --- NEW: Helper function to check if a mouth is inside a head's bounding box ---
def associate_mouth_to_head(mouth_box, head_box):
    """Check if the center of the mouth_box is inside the head_box."""
    mouth_center_x = (mouth_box[0] + mouth_box[2]) / 2
    mouth_center_y = (mouth_box[1] + mouth_box[3]) / 2
   
    is_inside = (head_box[0] < mouth_center_x < head_box[2] and
                 head_box[1] < mouth_center_y < head_box[3])
    return is_inside

def compute_iou(boxA, boxB):
    """Compute Intersection over Union between two boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def apply_nms_to_heads(head_boxes, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to remove overlapping head boxes"""
    if not head_boxes:
        return []
   
    # Sort by area (largest first)
    head_boxes.sort(key=lambda x: x['area'], reverse=True)
   
    keep = []
   
    while head_boxes:
        current = head_boxes.pop(0)
        keep.append(current)
       
        head_boxes = [box for box in head_boxes
                     if compute_iou(current['box'], box['box']) < iou_threshold]
   
    return keep

def split_merged_boxes(boxes, labels, probs, frame_shape):
    """Split large bounding boxes that might contain multiple chickens with adaptive thresholds"""
    new_boxes = []
    new_labels = []
    new_probs = []
   
    h, w = frame_shape[:2]
    frame_area = w * h
   
    # Adaptive thresholds based on frame size
    max_single_chicken_area = frame_area * 0.1  # 10% of frame area
    min_single_chicken_area = frame_area * 0.01  # 1% of frame area
   
    for i in range(boxes.size(0)):
        box = boxes[i].numpy()
       
        # Scale to original frame coordinates to check area
        scaled_box = box.copy()
        scaled_box[0] *= w / 300; scaled_box[2] *= w / 300
        scaled_box[1] *= h / 300; scaled_box[3] *= h / 300
       
        area = (scaled_box[2] - scaled_box[0]) * (scaled_box[3] - scaled_box[1])
        label = labels[i].item()
       
        # Check if this might be multiple chickens
        if (label == 1 and  # 'head' class
            area > max_single_chicken_area and
            area < frame_area * 0.4):  # Don't split boxes > 40% of frame
           
            box_width = scaled_box[2] - scaled_box[0]
            box_height = scaled_box[3] - scaled_box[1]
           
            # Calculate expected single chicken dimensions
            expected_single_width = box_width / 2
            expected_single_height = box_height
           
            # Only split if the split boxes would be reasonable size
            if (expected_single_width > w * 0.05 and  # At least 5% of frame width
                expected_single_height > h * 0.05 and  # At least 5% of frame height
                box_width > box_height * 1.2):  # Wider than tall
               
                # Split into two boxes
                split_point = box_width / 2
               
                # Create two boxes in original 300x300 coordinates
                box1_300 = [box[0], box[1], box[0] + split_point * (300/w), box[3]]
                box2_300 = [box[0] + split_point * (300/w), box[1], box[2], box[3]]
               
                new_boxes.extend([torch.tensor(box1_300), torch.tensor(box2_300)])
                new_labels.extend([labels[i], labels[i]])
                new_probs.extend([probs[i] * 0.7, probs[i] * 0.7])  # Reduce confidence
               
                print(f"Split large box: {area:.0f}px -> two boxes")
                continue
       
        # Keep original box
        new_boxes.append(boxes[i])
        new_labels.append(labels[i])
        new_probs.append(probs[i])
   
    if new_boxes:
        return torch.stack(new_boxes), torch.stack(new_labels), torch.stack(new_probs)
    return boxes, labels, probs

def adaptive_nms(head_boxes, frame_shape, base_iou_threshold=0.4):
    """Apply NMS with adaptive thresholds based on box size"""
    if not head_boxes:
        return []
   
    h, w = frame_shape[:2]
    frame_area = w * h
   
    # Sort by area (largest first)
    head_boxes.sort(key=lambda x: x['area'], reverse=True)
   
    keep = []
   
    while head_boxes:
        current = head_boxes.pop(0)
        keep.append(current)
       
        remaining_boxes = []
        for box in head_boxes:
            iou = compute_iou(current['box'], box['box'])
           
            # Use stricter NMS for small boxes (likely separate chickens)
            # Use looser NMS for large boxes (might be merged chickens)
            if current['area'] > frame_area * 0.15:  # Large box
                iou_threshold = base_iou_threshold * 0.7  # Stricter
            elif current['area'] < frame_area * 0.03:  # Small box  
                iou_threshold = base_iou_threshold * 1.3  # Looser
            else:  # Medium box
                iou_threshold = base_iou_threshold
               
            if iou < iou_threshold:
                remaining_boxes.append(box)
       
        head_boxes = remaining_boxes
   
    return keep
   
# --- MODIFIED: detection_loop with improved status display ---
# --- IMPROVED: Better multi-chicken tracking ---
def detection_loop():
    global open_beak_count

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
           
            # Real detection code
            resized = cv2.resize(frame, (300, 300))
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            with torch.no_grad():
                # Try different confidence levels for better detection
                boxes, labels, probs = predictor.predict(rgb_image, 200, 0.35)  # Even lower confidence

            h, w, _ = frame.shape
           
            # Apply merged box splitting FIRST
            boxes, labels, probs = split_merged_boxes(boxes, labels, probs, frame.shape)
           
            scale_x, scale_y = w / 300, h / 300
           
            head_boxes_current_frame = []
            open_mouth_boxes_current_frame = []
           
            for i in range(boxes.size(0)):
                box = boxes[i].numpy()
                box[0] *= scale_x; box[2] *= scale_x
                box[1] *= scale_y; box[3] *= scale_y
                box = box.astype(int)
               
                label = class_names[labels[i]]
                if label == 'head':
                    head_info = {
                        'box': box,
                        'center_x': (box[0] + box[2]) / 2,
                        'center_y': (box[1] + box[3]) / 2,
                        'width': box[2] - box[0],
                        'height': box[3] - box[1],
                        'area': (box[2] - box[0]) * (box[3] - box[1])
                    }
                   
                    # Filter out boxes that are too small to be real chickens
                    if head_info['area'] > (w * h * 0.005):  # At least 0.5% of frame area
                        head_boxes_current_frame.append(head_info)
                       
                elif label == 'open-mouth':
                    open_mouth_boxes_current_frame.append({
                        'box': box,
                        'center_x': (box[0] + box[2]) / 2,
                        'center_y': (box[1] + box[3]) / 2
                    })

            # Apply NMS with different thresholds based on box size
            head_boxes_current_frame = adaptive_nms(head_boxes_current_frame, frame.shape)
           
            # Continue with your existing tracking logic...
            used_mouths = set()
            closed_beak_count = 0
            open_beak_count_frame = 0

            for head_idx, head_info in enumerate(head_boxes_current_frame):
                head_box = head_info['box']
                head_center = (head_info['center_x'], head_info['center_y'])

                best_mouth_idx = None
                best_distance = float('inf')
                max_allowed_distance = min(head_info['width'], head_info['height']) * 0.6

                for mouth_idx, mouth_info in enumerate(open_mouth_boxes_current_frame):
                    if mouth_idx in used_mouths:
                        continue

                    mouth_center = (mouth_info['center_x'], mouth_info['center_y'])
                    distance = np.sqrt(
                        (head_center[0] - mouth_center[0])**2 +
                        (head_center[1] - mouth_center[1])**2
                    )

                    if (
                       distance < best_distance and
                       distance <= max_allowed_distance and
                       head_box[0] < mouth_center[0] < head_box[2] and
                       head_box[1] < mouth_center[1] < head_box[3]
                    ):
                       best_distance = distance
                       best_mouth_idx = mouth_idx

                if best_mouth_idx is not None:
                    used_mouths.add(best_mouth_idx)
                    open_beak_count_frame += 1

                    cv2.rectangle(frame,
                                 (head_box[0], head_box[1]),
                                 (head_box[2], head_box[3]),
                                 (0, 0, 255), 2)
                    cv2.putText(frame,
                              "Open Beak",
                              (head_box[0], head_box[1] - 5),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.6, (0, 0, 255), 2)
                else:
                    closed_beak_count += 1

                    cv2.rectangle(frame,
                                (head_box[0], head_box[1]),
                                (head_box[2], head_box[3]),
                                (0, 255, 0), 2)
                    cv2.putText(frame,
                               "Closed Beak",
                               (head_box[0], head_box[1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 255, 0), 2)
                   
            open_beak_count = open_beak_count_frame
           
            # Enhanced status message with detection info
            if len(head_boxes_current_frame) == 0:
                detection_status = "No Detection"
            else:
                total_chickens = closed_beak_count + open_beak_count
                avg_box_size = np.mean([box['area'] for box in head_boxes_current_frame]) if head_boxes_current_frame else 0
                detection_status = f"Chickens: {total_chickens} | Open: {open_beak_count} | Closed: {closed_beak_count}"

            detection_queue.put((frame, detection_status))
           
# --- MODIFIED: Update frame function with improved status display ---
def update_frame():
    # Capture and queue frame
    ret, frame = cap.read()
    if ret and frame_queue.empty():
        frame_queue.put(frame)

    # Process detection results
    if not detection_queue.empty():
        processed_frame, detection_status = detection_queue.get()
       
        # Image conversion and display
        img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
       
        display_width = video_label.winfo_width()
        display_height = video_label.winfo_height()
       
        if display_width > 0 and display_height > 0:
            img = img.resize((display_width, display_height), Image.LANCZOS)
           
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Label updates
        temp_label.config(text=f"Temperature: {current_temp:.1f} °C")
        hum_label.config(text=f"Humidity: {current_hum:.1f} %")
        timestamp_label.config(text=f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}")
        status_label.config(text=f"Status: {detection_status}")
        status_label.config(fg="orange")
        thi_label.config(text=f"THI: {current_thi:.1f}")

    root.after(100, update_frame)

# --- Start Threads ---
sensor_thread = threading.Thread(target=sensor_loop, daemon=True)
detection_thread = threading.Thread(target=detection_loop, daemon=True)
sensor_thread.start()
detection_thread.start()

root.after(0, update_frame)
root.after(3000, update_graphs)
root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), GPIO.cleanup(), root.destroy()))
root.mainloop()
