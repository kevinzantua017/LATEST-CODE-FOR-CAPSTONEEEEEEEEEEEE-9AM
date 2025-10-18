"""
Flask and SocketIO web application for the Smart Crosswalk dashboard.
(… top-of-file comment unchanged …)
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

# Optional CORS
try:
    from flask_cors import CORS
    _HAS_CORS = True
except Exception:
    _HAS_CORS = False

# --------------------- CONFIG ---------------------
DB_PATH = os.getenv("SC_DB", "smart_crosswalk.db")
JPEG_QUALITY = int(os.getenv("SC_JPEG_QUALITY", "70"))
STREAM_INTERVAL_S = float(os.getenv("SC_STREAM_INTERVAL", "0.03"))

# --------------------- SHARED STATE ---------------------
latest_frames: Dict[str, Optional[np.ndarray]] = {"ped": None, "veh": None, "tl": None}
latest_jpegs: Dict[str, Optional[bytes]] = {"ped": None, "veh": None, "tl": None}

board_state = {
    "board_veh": "OFF",
    "board_ped_l": "OFF",
    "board_ped_r": "OFF",
    "scenario": "baseline"
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
app = Flask(__name__, static_folder="static", static_url_path="")
if _HAS_CORS:
    CORS(app, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# --------------------- FRAME & STATUS PUBLISHING ---------------------
def _encode_jpg(img: Optional[np.ndarray], q: int = JPEG_QUALITY) -> Optional[bytes]:
    if img is None:
        return None
    try:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return buf.tobytes() if ok else None
    except Exception:
        return None

def publish_frame(key: str, frame_bgr: np.ndarray):
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
    now_ts = float(kwargs.get("now_ts", time.time()))
    ped_count = int(kwargs.get("ped_count", 0))
    veh_count = int(kwargs.get("veh_count", 0))
    tl_color = str(kwargs.get("tl_color", "unknown"))
    nearest_m = float(kwargs.get("nearest_m", 0.0))
    avg_mps = float(kwargs.get("avg_mps", 0.0))
    flags = kwargs.get("flags", {}) or {}

    action = kwargs.get("action")
    scenario = kwargs.get("scenario")
    if action is None or scenario is None:
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
        board_state["scenario"] = scenario
        latest_status["board_veh"]   = board_state.get("board_veh", latest_status.get("board_veh", "OFF"))
        latest_status["board_ped_l"] = board_state.get("board_ped_l", latest_status.get("board_ped_l", "OFF"))
        latest_status["board_ped_r"] = board_state.get("board_ped_r", latest_status.get("board_ped_r", "OFF"))
    try:
        socketio.emit("status", latest_status, namespace="/realtime")
    except Exception as e:
        print("[SocketIO status] error:", repr(e))

def start_http_server(host: str = "0.0.0.0", port: int = 5000):
    print(f"[flask_app] Starting HTTP server on {host}:{port}")
    socketio.run(app, host=host, port=port)

# --------------------- ANALYTICS (REALTIME) ---------------------
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

def compute_analytics_data() -> List[Dict]:
    """
    Build last-10-min per-minute aggregates from DB.
    """
    now_ts = time.time()
    start_ts = now_ts - 10 * 60
    rows = _safe_query_rows(
        """
        SELECT strftime('%Y-%m-%d %H:%M', datetime(ts, 'unixepoch', 'localtime')) AS minute,
               AVG(ped_count) AS avg_ped,
               AVG(veh_count) AS avg_veh,
               SUM(CASE WHEN action = 'GO'   THEN 1 ELSE 0 END) AS go,
               SUM(CASE WHEN action = 'STOP' THEN 1 ELSE 0 END) AS stop,
               SUM(CASE WHEN action = 'OFF'  THEN 1 ELSE 0 END) AS off
          FROM events
         WHERE ts >= ?
         GROUP BY minute
         ORDER BY minute ASC
        """,
        (start_ts,),
    )
    data = []
    for minute, avg_ped, avg_veh, go, stop, off in rows:
        data.append({
            "minute": minute,
            "avg_ped": float(avg_ped) if avg_ped is not None else 0.0,
            "avg_veh": float(avg_veh) if avg_veh is not None else 0.0,
            "go": int(go) if go is not None else 0,
            "stop": int(stop) if stop is not None else 0,
            "off": int(off) if off is not None else 0,
        })
    return data

def publish_analytics_push() -> None:
    """
    Compute analytics now and broadcast over Socket.IO (real-time updates).
    Called by app.py right after writing a new DB row.
    """
    try:
        data = compute_analytics_data()
        socketio.emit("analytics", data, namespace="/realtime")
    except Exception as e:
        print("[SocketIO analytics] error:", repr(e))

# --------------------- ROUTES ---------------------
@app.get("/")
def root_index():
    return send_from_directory(app.static_folder, "index.html")

def mjpeg_stream(key: str):
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
    if cam not in ("ped", "veh", "tl"):
        return "unknown cam", 404
    return Response(mjpeg_stream(cam), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/status_now")
def api_status_now():
    with _state_lock:
        return jsonify(latest_status)

@app.post("/api/set_scenario")
def api_set_scenario():
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        data = {}
    valid_keys = {"board_veh", "board_ped_l", "board_ped_r", "scenario"}
    updated = False
    with _state_lock:
        for k, v in data.items():
            if k in valid_keys:
                board_state[k] = str(v).upper() if k.startswith("board_") else str(v)
                updated = True
    if "scenario" in data:
        latest_status["scenario"] = data["scenario"]
    if updated:
        publish_status_from_loop(
            now_ts=time.time(),
            ped_count=latest_status.get("ped_count", 0),
            veh_count=latest_status.get("veh_count", 0),
            tl_color=latest_status.get("tl_color", "unknown"),
            nearest_m=latest_status.get("nearest_vehicle_distance_m", 0.0),
            avg_mps=latest_status.get("avg_vehicle_speed_mps", 0.0),
            flags={},
            extra={},
        )
    return jsonify({"ok": True, "board_state": board_state})

@app.get("/api/analytics")
def api_analytics():
    # Same computation used for real-time push
    return jsonify(compute_analytics_data())

@app.get("/api/logs")
def api_logs():
    try:
        limit = int(request.args.get('limit', '200'))
    except Exception:
        limit = 200
    rows = _safe_query_rows(
        """
        SELECT id, ts, ped_count, veh_count, tl_color,
               nearest_vehicle_distance_m, avg_vehicle_speed_mps, action
          FROM events
         ORDER BY id DESC
         LIMIT ?
        """,
        (limit,),
    )
    result = []
    current_board = {
        "board_veh": latest_status.get("board_veh", "OFF"),
        "board_ped_l": latest_status.get("board_ped_l", "OFF"),
        "board_ped_r": latest_status.get("board_ped_r", "OFF"),
    }
    for row in rows:
        event_id, ts_val, ped_c, veh_c, tl_col, nearest_m, avg_mps, action_val = row
        result.append({
            "id": event_id,
            "ts": ts_val,
            "ped_count": ped_c,
            "veh_count": veh_c,
            "tl_color": tl_col,
            "nearest_vehicle_distance_m": nearest_m,
            "avg_vehicle_speed_mps": avg_mps,
            "action": action_val,
            "board_veh": current_board["board_veh"],
            "board_ped_l": current_board["board_ped_l"],
            "board_ped_r": current_board["board_ped_r"],
        })
    return jsonify(result)

@app.post("/api/logs/delete")
def api_logs_delete():
    try:
        data = request.get_json(force=True) or {}
        ids = data.get("ids", [])
        ids = [int(i) for i in ids if isinstance(i, (int, float, str))]
    except Exception:
        ids = []
    if not ids:
        return jsonify({"ok": True, "deleted": 0})
    placeholders = ",".join(["?"] * len(ids))
    conn = _db_connect()
    cur = conn.cursor()
    try:
        cur.execute(f"DELETE FROM events WHERE id IN ({placeholders})", ids)
        conn.commit()
        deleted = cur.rowcount
    except Exception:
        deleted = 0
    finally:
        conn.close()
    return jsonify({"ok": True, "deleted": deleted})

@app.post("/api/logs/clear")
def api_logs_clear():
    try:
        data = request.get_json(force=True) or {}
        do_clear = bool(data.get("all"))
    except Exception:
        do_clear = False
    if not do_clear:
        return jsonify({"ok": False, "deleted": 0})
    conn = _db_connect()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM events")
        conn.commit()
        deleted = cur.rowcount
    except Exception:
        deleted = 0
    finally:
        conn.close()
    return jsonify({"ok": True, "deleted": deleted})

# --------------------- DECISION / SCENARIO ---------------------
def decide_scenario(now_ts: float, ped_count: int, veh_count: int, tl_color: str, flags: Dict[str, bool]) -> Tuple[str, str]:
    if flags.get("ambulance", False):
        return ("STOP", "scenario_3_emergency")
    if flags.get("night", False) and ped_count >= 30 and veh_count <= 2 and tl_color == "green":
        return ("STOP", "scenario_1_night_ped")
    if flags.get("rush", False) and (5 <= ped_count <= 10) and veh_count >= 20 and tl_color == "red":
        return ("OFF", "scenario_2_rush_hold")
    if tl_color == "green" or tl_color == "yellow":
        return ("STOP", "baseline")
    if tl_color == "red":
        return ("GO", "baseline") if ped_count > 0 else ("OFF", "baseline")
    if ped_count > 0 and veh_count > 0:
        return ("STOP", "baseline")
    return ("OFF", "baseline")

# --------------------- MAIN (for standalone testing) ---------------------
if __name__ == "__main__":
    start_http_server(host="0.0.0.0", port=5000)
