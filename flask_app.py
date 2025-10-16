import os
import sqlite3
import threading
import time
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_socketio import SocketIO

try:
    from flask_cors import CORS  # optional CORS; if not present we just continue without it
    _HAS_CORS = True
except Exception:
    _HAS_CORS = False

# --------------------- CONFIG ---------------------
DB_PATH = os.getenv("SC_DB", "smart_crosswalk.db")
JPEG_QUALITY = int(os.getenv("SC_JPEG_QUALITY", "70"))
STREAM_INTERVAL_S = float(os.getenv("SC_STREAM_INTERVAL", "0.03"))
DB_TAIL_INTERVAL_S = float(os.getenv("SC_DB_TAIL_INTERVAL", "0.5"))

# --------------------- SHARED STATE ---------------------
latest_frames: Dict[str, Optional[np.ndarray]] = {"ped": None, "veh": None, "tl": None}
latest_jpegs:  Dict[str, Optional[bytes]]     = {"ped": None, "veh": None, "tl": None}

# Current board & scenario state (controlled by /api/set_scenario)
board_state = {
    "board_veh": "OFF",      # "ON" or "OFF"
    "board_ped_l": "OFF",
    "board_ped_r": "OFF",
    "scenario": "baseline"   # baseline | scenario_1_night_ped | scenario_2_rush_hold | scenario_3_emergency
}

latest_status = {
    "ts": 0.0,
    "ped_count": 0,
    "veh_count": 0,
    "tl_color": "unknown",
    "nearest_vehicle_distance_m": 0.0,
    "avg_vehicle_speed_mps": 0.0,
    "action": "OFF",
    "scenario": "baseline",
    "board_veh": "OFF",
    "board_ped_l": "OFF",
    "board_ped_r": "OFF",
}

_state_lock = threading.Lock()

# --------------------- APP / SOCKET ---------------------
# Serve static files from ./static (place index.html, style.css, images/, etc. there)
app = Flask(__name__, static_folder="static", static_url_path="")
if _HAS_CORS:
    CORS(app, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

@app.get("/")
def root_index():
    return send_from_directory(app.static_folder, "index.html")

# --------------------- MJPEG STREAMING ---------------------
def mjpeg_stream(key: str):
    boundary = b"--frame"
    header   = b"Content-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n"
    while True:
        with _state_lock:
            jpg = latest_jpegs.get(key)
        if jpg is None:
            socketio.sleep(0.03)
            continue
        yield boundary + b"\r\n" + header + jpg + b"\r\n"
        socketio.sleep(STREAM_INTERVAL_S)

@app.get("/stream/<cam>")
def stream_cam(cam: str):
    if cam not in ("ped", "veh", "tl"):
        return "unknown cam", 404
    return Response(mjpeg_stream(cam), mimetype="multipart/x-mixed-replace; boundary=frame")

# --------------------- DB HELPERS ---------------------
def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=3.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=3000;")
    return conn

def _safe_query_rows(sql: str, args=()):
    try:
        conn = _db_connect()
        cur = conn.cursor()
        cur.execute(sql, args)
        rows = cur.fetchall()
        conn.close()
        return rows
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return []
        raise

# --------------------- DECISION / SCENARIO ---------------------
def decide_scenario(now_ts: float, ped_count: int, veh_count: int, tl_color: str, flags: Dict[str, bool]) -> Tuple[str, str]:
    if flags.get("ambulance", False):
        return ("STOP", "scenario_3_emergency")

    if flags.get("night", False) and ped_count >= 30 and veh_count <= 2 and tl_color == "green":
        return ("STOP", "scenario_1_night_ped")
    if flags.get("rush", False) and (5 <= ped_count <= 10) and veh_count >= 20 and tl_color == "red":
        return ("OFF", "scenario_2_rush_hold")

    action = "OFF"
    if tl_color == "red":
        action = "STOP"
    elif tl_color == "green":
        if ped_count > 0 and veh_count > 0:
            action = "STOP"
        elif ped_count > 0:
            action = "GO"
    elif tl_color == "yellow":
        action = "STOP" if ped_count > 0 else "OFF"
    return (action, "baseline")

# --------------------- VEHICLE AND PEDESTRIAN DETECTION ---------------------
def detect_pedestrians_and_vehicles(frame: np.ndarray) -> Tuple[List[Tuple[str, Tuple[int, int, int, int]]], List[Tuple[str, Tuple[int, int, int, int]]]]:
    # Placeholder YOLO model detection function (you can replace it with actual detection logic)
    # Detect pedestrians (ped) and vehicles (veh) from the frame
    ped_detections = []
    veh_detections = []
    
    # Assume these are detections from your YOLO model, for example:
    # ped_detections = [("person", (x1, y1, x2, y2), confidence)]
    # veh_detections = [("car", (x1, y1, x2, y2), confidence)]
    
    return ped_detections, veh_detections

# --------------------- MAIN ---------------------
def main():
    while True:
        time.sleep(0.005)
        # Simulate receiving frames
        ped_frame = None  # Replace with actual frame for pedestrian detection
        veh_frame = None  # Replace with actual frame for vehicle detection
        tl_frame = None   # Replace with actual frame for traffic light detection

        # Detect pedestrians and vehicles
        ped_detections, veh_detections = detect_pedestrians_and_vehicles(ped_frame)

        # Process detected objects
        ped_count = len(ped_detections)
        veh_count = len(veh_detections)
        nearest_m = float("inf")  # Placeholder
        avg_mps = 0.0  # Placeholder for average speed calculation

        # Process traffic light color
        tl_color = "green"  # Replace with actual logic for detecting traffic light color

        # Publish decision
        decision = decide_scenario(time.time(), ped_count, veh_count, tl_color, flags={})
        print(f"Pedestrian count: {ped_count}, Vehicle count: {veh_count}, Decision: {decision[0]}")

        # Publish annotated frames (visualizations)
        # Visualize pedestrian detections
        if ped_frame is not None:
            # Add bounding boxes for pedestrians
            for label, (x1, y1, x2, y2), _ in ped_detections:
                cv2.rectangle(ped_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Visualize vehicle detections
        if veh_frame is not None:
            for label, (x1, y1, x2, y2), _ in veh_detections:
                cv2.rectangle(veh_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Publish the frames (here we're just using a dummy publisher for the example)
        if ped_frame is not None:
            print("Publishing pedestrian frame...")
        if veh_frame is not None:
            print("Publishing vehicle frame...")

        # Sleep to maintain frame rate (if needed)
        time.sleep(1)

if __name__ == "__main__":
    main()
