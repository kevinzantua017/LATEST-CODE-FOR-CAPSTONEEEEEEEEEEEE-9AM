"""
Flask and SocketIO web application for the Smart Crosswalk dashboard.

This module exposes three key functions used by the Raspberry Pi pipeline:

  * ``publish_frame(key, frame_bgr)``:  Store the latest frame for the
    specified camera (``'ped'`` for pedestrian, ``'veh'`` for vehicles,
    ``'tl'`` for the traffic light) and notify connected web clients.

  * ``publish_status_from_loop(**kwargs)``:  Update the dashboard status
    (counts, traffic light colour, computed decision, scenario and board
    states) and broadcast it via Socket.IO.

  * ``start_http_server(host, port)``:  Launch the Flask-SocketIO
    server.  This is called by ``app.py`` on the Raspberry Pi to serve
    the web dashboard.  When this module is executed directly, it
    likewise starts the server for development or standalone usage.

In addition to these functions, this module includes endpoints for
streaming MJPEG camera feeds, simple database helpers, scenario logic,
and placeholders for integrating a YOLO-based object detector.  The
default detection and scenario logic here are minimal; the actual
heavy-lifting runs in the cloud worker.  The dashboard displays
whatever frames and status are passed in via ``publish_frame`` and
``publish_status_from_loop``.

Created for the Smart Crosswalk project.
"""

import os
import sqlite3
import threading
import time
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_socketio import SocketIO

# Optional CORS; if not present we just continue without it
try:
    from flask_cors import CORS
    _HAS_CORS = True
except Exception:
    _HAS_CORS = False

# --------------------- CONFIG ---------------------
DB_PATH = os.getenv("SC_DB", "smart_crosswalk.db")
JPEG_QUALITY = int(os.getenv("SC_JPEG_QUALITY", "70"))
STREAM_INTERVAL_S = float(os.getenv("SC_STREAM_INTERVAL", "0.03"))
DB_TAIL_INTERVAL_S = float(os.getenv("SC_DB_TAIL_INTERVAL", "0.5"))

# --------------------- SHARED STATE ---------------------
# Latest frames and JPEGs for each camera, keyed by 'ped', 'veh', 'tl'.
latest_frames: Dict[str, Optional[np.ndarray]] = {"ped": None, "veh": None, "tl": None}
latest_jpegs: Dict[str, Optional[bytes]] = {"ped": None, "veh": None, "tl": None}

# Current board & scenario state (controlled by /api/set_scenario)
board_state = {
    "board_veh": "OFF",      # "ON" or "OFF"
    "board_ped_l": "OFF",
    "board_ped_r": "OFF",
    "scenario": "baseline"   # baseline | scenario_1_night_ped | scenario_2_rush_hold | scenario_3_emergency
}

# Last status broadcast to the dashboard.  Keys correspond to fields
# displayed on the UI (see static/index.html).  The values are
# updated by publish_status_from_loop() and may be overridden by
# board_state.
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

# --------------------- FRAME & STATUS PUBLISHING ---------------------
def _encode_jpg(img: Optional[np.ndarray], q: int = JPEG_QUALITY) -> Optional[bytes]:
    """Encode a BGR frame to JPEG bytes using OpenCV.

    If encoding fails or img is None, returns None.
    """
    if img is None:
        return None
    try:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return buf.tobytes() if ok else None
    except Exception:
        return None


def publish_frame(key: str, frame_bgr: np.ndarray):
    """
    Update the latest frame and JPEG for the specified camera and emit a
    Socket.IO event so connected clients refresh their views.

    ``key`` should be one of 'ped', 'veh' or 'tl'.  This function
    stores a copy of the frame and its JPEG representation under a
    thread lock to avoid race conditions, then emits a lightweight
    timestamp message via Socket.IO.  The dashboard client uses this
    timestamp event to fetch the latest image from the MJPEG stream.
    """
    if frame_bgr is None:
        return
    with _state_lock:
        latest_frames[key] = frame_bgr.copy()
        latest_jpegs[key] = _encode_jpg(frame_bgr, q=JPEG_QUALITY)
    try:
        socketio.emit(f"frame_{key}", {"ts": time.time()}, namespace="/realtime")
    except Exception as e:
        print("[SocketIO frame] error:", repr(e))


def publish_status_from_loop(**kwargs):
    """
    Update the dashboard status with counts, traffic light colour, decision
    and scenario, then broadcast it to all connected clients via
    Socket.IO.

    The Raspberry Pi pipeline calls this function periodically and
    passes fields as keyword arguments.  Recognised keys include:

        ``now_ts``   (float): current timestamp
        ``ped_count`` (int): number of detected pedestrians
        ``veh_count`` (int): number of detected vehicles
        ``tl_color`` (str): current traffic light colour ('red', 'yellow', 'green' or 'unknown')
        ``nearest_m`` (float): nearest detected vehicle distance in metres
        ``avg_mps``  (float): average vehicle speed in m/s
        ``flags``    (dict): flag values such as night, rush, ambulance
        ``extra``    (dict): reserved for future use

    The decision and scenario are computed by calling ``decide_scenario``.
    Board state values (board_veh, board_ped_l, board_ped_r) are merged
    from the global ``board_state`` dict.  The updated status is stored
    and emitted via Socket.IO as a single JSON object.
    """
    now_ts = float(kwargs.get("now_ts", time.time()))
    ped_count = int(kwargs.get("ped_count", 0))
    veh_count = int(kwargs.get("veh_count", 0))
    tl_color = str(kwargs.get("tl_color", "unknown"))
    nearest_m = float(kwargs.get("nearest_m", 0.0))
    avg_mps = float(kwargs.get("avg_mps", 0.0))
    flags = kwargs.get("flags", {}) or {}
    # Compute decision and scenario
    action, scenario = decide_scenario(now_ts, ped_count, veh_count, tl_color, flags)
    with _state_lock:
        latest_status["ts"] = now_ts
        latest_status["ped_count"] = ped_count
        latest_status["veh_count"] = veh_count
        latest_status["tl_color"] = tl_color
        latest_status["nearest_vehicle_distance_m"] = nearest_m
        latest_status["avg_vehicle_speed_mps"] = avg_mps
        latest_status["action"] = action
        latest_status["scenario"] = scenario
        # Merge board state overrides from board_state dict
        latest_status["board_veh"] = board_state.get("board_veh", latest_status.get("board_veh", "OFF"))
        latest_status["board_ped_l"] = board_state.get("board_ped_l", latest_status.get("board_ped_l", "OFF"))
        latest_status["board_ped_r"] = board_state.get("board_ped_r", latest_status.get("board_ped_r", "OFF"))
    try:
        socketio.emit("status", latest_status, namespace="/realtime")
    except Exception as e:
        print("[SocketIO status] error:", repr(e))


def start_http_server(host: str = "0.0.0.0", port: int = 5000):
    """
    Start the Flask-SocketIO server.

    This function wraps ``socketio.run`` and should be used by the
    Raspberry Pi application to serve the dashboard.  When this
    module is run directly (``python3 flask_app.py``), it is also
    invoked from the ``__main__`` block.
    """
    print(f"[flask_app] Starting HTTP server on {host}:{port}")
    socketio.run(app, host=host, port=port)


# --------------------- APP / SOCKET ---------------------
# Serve static files from ./static (place index.html, style.css, images/, etc. there)
app = Flask(__name__, static_folder="static", static_url_path="")
if _HAS_CORS:
    CORS(app, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")


@app.get("/")
def root_index():
    """Serve the dashboard HTML."""
    return send_from_directory(app.static_folder, "index.html")


# --------------------- MJPEG STREAMING ---------------------
def mjpeg_stream(key: str):
    """Generator yielding MJPEG frames for a given camera key."""
    boundary = b"--frame"
    header = b"Content-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n"
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
    """
    Return an MJPEG streaming response for the requested camera.

    Valid values are 'ped', 'veh' and 'tl'.  If an invalid key is
    supplied, a 404 is returned.
    """
    if cam not in ("ped", "veh", "tl"):
        return "unknown cam", 404
    return Response(mjpeg_stream(cam), mimetype="multipart/x-mixed-replace; boundary=frame")


# --------------------- DB HELPERS ---------------------
def _db_connect() -> sqlite3.Connection:
    """Connect to the SQLite database with sensible defaults."""
    conn = sqlite3.connect(DB_PATH, timeout=3.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=3000;")
    return conn


def _safe_query_rows(sql: str, args=()):
    """Run a query and return rows, ignoring missing-table errors."""
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
    """
    Decide the action (GO/STOP/OFF) and scenario based on inputs.

    This mirrors the logic used on the Raspberry Pi but is simplified
    here.  If ``flags`` includes 'ambulance', emergency is always
    prioritised.  Otherwise, at night we prioritise pedestrians when
    many pedestrians are waiting and the light is green; during rush
    hour we hold pedestrians when there are many vehicles and few
    pedestrians.  Red lights always yield STOP.  Green lights yield
    GO if pedestrians have priority and there are no vehicles; else
    STOP.  Yellow yields STOP if pedestrians present, OFF otherwise.
    """
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
def detect_pedestrians_and_vehicles(frame: Optional[np.ndarray]) -> Tuple[List[Tuple[str, Tuple[int, int, int, int], float]], List[Tuple[str, Tuple[int, int, int, int], float]]]:
    """
    Placeholder detection function returning empty lists.

    In the production system detection happens in the cloud worker.  If
    you run this module directly, you can integrate a YOLOv8 model
    here to produce detections.  The return value should be two lists
    of tuples: (label, (x1, y1, x2, y2), confidence).
    """
    return [], []  # Replace with actual detection logic if needed


# --------------------- MAIN (for standalone testing) ---------------------
def main():
    """Optional test loop that simulates detection and publishes dummy frames."""
    # This loop is intentionally minimal.  In production, the cloud
    # worker performs inference and the Raspberry Pi calls publish_frame
    # and publish_status_from_loop() based on MQTT messages.
    # Here we simply increment counts and broadcast them for testing.
    count = 0
    while True:
        time.sleep(1)
        ped_count = count % 5
        veh_count = (count // 5) % 3
        tl_color = "green" if (count % 10) < 5 else "red"
        flags = {"night": False, "rush": False, "ambulance": False}
        now = time.time()
        publish_status_from_loop(now_ts=now, ped_count=ped_count, veh_count=veh_count, tl_color=tl_color, nearest_m=3.0, avg_mps=1.2, flags=flags, extra={})
        # No frames published here; you can add calls to publish_frame() if you
        # want to test MJPEG streaming with dummy data.
        count += 1


if __name__ == "__main__":
    # When run directly, start the server on the default port.  This
    # allows you to test the dashboard without the full Raspberry Pi
    # pipeline.  If you instead run ``python3 app.py`` on the Pi, the
    # pipeline will call start_http_server() directly.
    start_http_server(host="0.0.0.0", port=5000)
