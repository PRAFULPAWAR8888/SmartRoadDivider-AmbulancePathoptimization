import os
import sys
import argparse
import glob
import time
import serial  # Added for Arduino communication

import cv2
import numpy as np
from ultralytics import YOLO

# Initialize serial communication with Arduino
def init_serial(port='COM11', baudrate=9600, timeout=1):
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # Give time for connection to establish
        print("Connected to Arduino")
        return ser
    except Exception as e:
        print(f"Warning: Could not connect to Arduino on {port}. Error: {e}")
        return None

arduino = init_serial()

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model file')
parser.add_argument('--source', required=True, help='Image or video source')
parser.add_argument('--thresh', default=0.5, help='Confidence threshold')
parser.add_argument('--resolution', default=None, help='Resolution WxH (example: "640x480")')
parser.add_argument('--record', action='store_true', help='Record video output')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Check if model exists
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or not found.')
    sys.exit(0)

# Load YOLO model
model = YOLO(model_path, task='detect')
labels = model.names

# Determine input type
img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    source_type = 'image' if ext in img_ext_list else 'video' if ext in vid_ext_list else sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
else:
    print(f'Invalid input: {img_source}')
    sys.exit(0)

# Handle resolution
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.replace('×', 'x').split('x'))

# Video recording setup
if record:
    if source_type not in ['video', 'usb']:
        print('Recording only works for video or camera sources.')
        sys.exit(0)
    if not user_res:
        print('Specify resolution for recording.')
        sys.exit(0)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

# Open image/video source
cap = None
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [file for file in glob.glob(img_source + '/*') if os.path.splitext(file)[1] in img_ext_list]
elif source_type in ['video', 'usb']:
    cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

# Define colors for bounding boxes
bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133)]

# Function to send signal to Arduino with retries
def send_signal_to_arduino(signal, retries=3):
    if arduino and arduino.is_open:
        for attempt in range(retries):
            try:
                arduino.write(signal.encode())
                print(f"Signal '{signal}' sent to Arduino.")
                return
            except Exception as e:
                print(f"Error sending signal to Arduino (attempt {attempt+1}): {e}")
                time.sleep(1)  # Wait before retrying
    else:
        print("Arduino connection not available.")

# Inference loop
while True:
    t_start = time.perf_counter()

    # Load frame
    if source_type in ['image', 'folder']:
        if not imgs_list:
            print('All images processed. Exiting.')
            sys.exit(0)
        frame = cv2.imread(imgs_list.pop(0))
    elif cap:
        ret, frame = cap.read()
        if not ret:
            print('End of video/camera stream.')
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run inference
    results = model(frame, verbose=False)
    detections = results[0].boxes

    ambulance_detected = False  # Flag to check if ambulance is detected

    # Process detections
    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, f'{classname}: {int(conf*100)}%', (xmin, ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # If an ambulance is detected, send a signal to Arduino
            if classname.lower() == "ambulance":
                ambulance_detected = True

    # Send signal to Arduino
    if ambulance_detected:
        send_signal_to_arduino('A')

    # Display frame
    cv2.imshow('YOLO Detection', frame)
    if record:
        recorder.write(frame)

    # Handle keypress
    key = cv2.waitKey(5)
    if key in [ord('q'), ord('Q')]:
        break

    # FPS Calculation
    t_stop = time.perf_counter()
    print(f"FPS: {1 / (t_stop - t_start):.2f}")

# Cleanup
if cap:
    cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()

