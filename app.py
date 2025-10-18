"""
Raspberry Pi edge application for the AI-driven smart crosswalk.
"""

import os
import re
import ssl
import threading
import time
import json
from typing import Optional, Union, Tuple

import cv2
import numpy as np
import paho.mqtt.client as mqtt

def _noop_publish_frame(kind: str, frame_bgr: np.ndarray) -> None: pass
def _noop_publish_status_from_loop(**kwargs) -> None: pass
def _noop_start_http_server(*a, **k): print("[flask_app] not yet loaded")

publish_frame = _noop_publish_frame
publish_status_from_loop = _noop_publish_status_from_loop
start_http_server = _noop_start_http_server

os.environ.setdefault("EVENTLET_NO_GREENDNS", "YES")

# ---------- Environment ----------
SITE_ID = os.getenv("SC_SITE_ID", "adsmn-01")

FRAME_W = int(os.getenv("SC_FRAME_W", "640"))
FRAME_H = int(os.getenv("SC_FRAME_H", "360"))
FPS     = int(os.getenv("SC_FPS", "8"))
FRAME_TIME = 1.0 / max(1, FPS)
SKIP_FRAMES = int(os.getenv("SC_SKIP", "1")) if os.getenv("SC_SKIP", "1").isdigit() else 1
PUBLISH_HZ = float(os.getenv("SC_PUBLISH_HZ", "6"))
JPEG_QUALITY = int(os.getenv("SC_JPEG_QUALITY", "60"))
LANE_Y = int(os.getenv("SC_LANE_Y", "250"))

MQTT_HOST = os.getenv("MQTT_HOST")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_USER = os.getenv("MQTT_USERNAME")
MQTT_PASS = os.getenv("MQTT_PASSWORD")
MQTT_TLS = os.getenv("MQTT_TLS", "1") == "1"

TOPIC_PED = f"crosswalk/{SITE_ID}/frames/ped"
TOPIC_VEH = f"crosswalk/{SITE_ID}/frames/veh"
TOPIC_TL  = f"crosswalk/{SITE_ID}/frames/tl"
TOPIC_DEC = f"crosswalk/{SITE_ID}/decision"
TOPIC_VPED = f"crosswalk/{SITE_ID}/viz/ped"
TOPIC_VVEH = f"crosswalk/{SITE_ID}/viz/veh"
TOPIC_VTL  = f"crosswalk/{SITE_ID}/viz/tl"

DB_PATH = os.getenv("SC_DB", "smart_crosswalk.db")
# Log more frequently so Analytics has data to show promptly
LOG_EVERY_SEC = int(os.getenv("SC_LOG_SEC", "10"))
STATUS_MIN_PERIOD = float(os.getenv("SC_STATUS_PERIOD", "0.20"))

# ---------- JPEG encoding helpers ----------
try:
    from turbojpeg import TurboJPEG, TJPF_BGR  # type: ignore
    _tj = TurboJPEG()
    def _jpeg(bgr: np.ndarray, q: int = JPEG_QUALITY) -> Optional[bytes]:
        return _tj.encode(bgr, quality=int(q), pixel_format=TJPF_BGR)
    print("[JPEG] Using turbojpeg")
except Exception:
    def _jpeg(bgr: np.ndarray, q: int = JPEG_QUALITY) -> Optional[bytes]:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
        return buf.tobytes() if ok else None
    print("[JPEG] Using OpenCV JPEG")

cv2.setNumThreads(1)
try: cv2.ocl.setUseOpenCL(False)
except Exception: pass

# ---------- LED matrix ----------
def _init_led():
    try:
        from luma.core.interface.serial import spi, noop  # type: ignore
        from luma.led_matrix.device import max7219  # type: ignore
        from luma.core.render import canvas  # type: ignore
        from PIL import ImageFont  # type: ignore
        serial = spi(port=0, device=0, gpio=noop())
        device = max7219(serial, cascaded=int(os.getenv("SC_LED_CASCADE", "4")),
                         block_orientation=int(os.getenv("SC_LED_ORIENTATION", "-90")), rotate=0)
        font = ImageFont.load_default()
        def show_led(msg: str) -> None:
            with canvas(device) as draw:
                draw.text((1, -2), (msg or "")[:10], fill="white", font=font)
        print("[LED] MAX7219 initialized")
        return show_led
    except Exception as e:
        print("[LED] Fallback console:", repr(e))
        return lambda msg: print("[LED]", msg)

show_led = _init_led()

# ---------- Shared state ----------
viz_lock = threading.Lock()
# Keep last annotated frame AND timestamp; sticky for a longer window to avoid flicker
viz_frames: dict[str, Tuple[Optional[np.ndarray], float]] = {"ped": (None, 0.0), "veh": (None, 0.0), "tl": (None, 0.0)}
VIZ_MAX_AGE_S = float(os.getenv("SC_VIZ_MAX_AGE", "5.0"))  # was 1.0 — increased to stop oscillation

last_decision = {
    "ts": 0.0, "ped_count": 0, "veh_count": 0, "tl_color": "unknown",
    "nearest_m": 0.0, "avg_mps": 0.0, "action": "OFF", "scenario": "baseline",
}

SCENARIO_MESSAGES = {
    "scenario_2_rush_hold": "VEHICLES PRIORITY - WAIT",
    "scenario_1_night_ped": "PREPARE TO STOP - PED",
    "scenario_3_emergency": "EMERGENCY - PLEASE WAIT",
}

# ---------- MQTT callbacks ----------
def on_connect(client: mqtt.Client, userdata, flags, rc, properties=None) -> None:
    print(f"[MQTT] Connected rc={rc}")
    client.subscribe([(TOPIC_DEC, 1), (TOPIC_VPED, 0), (TOPIC_VVEH, 0), (TOPIC_VTL, 0)])

def on_message(client: mqtt.Client, userdata, msg: mqtt.MQTTMessage) -> None:
    global last_decision
    try:
        if msg.topic == TOPIC_DEC:
            d = json.loads(msg.payload.decode("utf-8", "ignore"))
            # Normalize field names for downstream
            last_decision["ts"] = float(d.get("ts", time.time()))
            last_decision["ped_count"] = int(d.get("ped_count", 0))
            last_decision["veh_count"] = int(d.get("veh_count", 0))
            last_decision["tl_color"] = str(d.get("tl_color", "unknown"))
            last_decision["nearest_m"] = float(d.get("nearest_m", 0.0))
            last_decision["avg_mps"] = float(d.get("avg_mps", 0.0))
            last_decision["action"] = str(d.get("action", "OFF"))
            last_decision["scenario"] = str(d.get("scenario", "baseline"))
            publish_status_from_loop(
                now_ts=last_decision["ts"],
                ped_count=last_decision["ped_count"],
                veh_count=last_decision["veh_count"],
                tl_color=last_decision["tl_color"],
                nearest_m=last_decision["nearest_m"],
                avg_mps=last_decision["avg_mps"],
                action=last_decision["action"],
                scenario=last_decision["scenario"],
                flags={"night": time.localtime().tm_hour >= 21, "rush": time.localtime().tm_hour == 7},
                extra={"ambulance": False},
            )
        else:
            arr = np.frombuffer(msg.payload, dtype=np.uint8)
            im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if im is None:
                return
            now_ts = time.time()
            with viz_lock:
                if msg.topic == TOPIC_VPED:
                    viz_frames["ped"] = (im, now_ts)
                elif msg.topic == TOPIC_VVEH:
                    viz_frames["veh"] = (im, now_ts)
                elif msg.topic == TOPIC_VTL:
                    viz_frames["tl"] = (im, now_ts)
    except Exception as e:
        print("[MQTT] on_message error:", repr(e))

def build_mqtt() -> mqtt.Client:
    if not MQTT_HOST:
        raise RuntimeError("MQTT_HOST not set. Load env first.")
    client = mqtt.Client(client_id=f"pi-io-{SITE_ID}", clean_session=True, protocol=mqtt.MQTTv311)
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASS or None)
    if MQTT_TLS:
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS_CLIENT)
        client.tls_insecure_set(False)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    client.loop_start()
    return client

# ---------- Camera handling ----------
def _normalize_cam(value: Union[str, int, None]) -> Union[int, str, None]:
    if value is None: return None
    if isinstance(value, int): return value
    s = str(value).strip()
    if s.isdigit(): return int(s)
    m = re.match(r"^/dev/video(\d+)$", s)
    return int(m.group(1)) if m else s

class CameraStream:
    def __init__(self, index: Union[int, str], width: int, height: int, fps: int) -> None:
        self.index = index; self.width = width; self.height = height; self.fps = fps
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.ret: bool = False
        self.stopped: bool = False
        self.last_update_ts: float = 0.0
        self.reopen_stale_sec = float(os.getenv("SC_CAM_REOPEN_STALE_SEC", "2.5"))
        self.read_interval = float(os.getenv("SC_CAM_READ_INTERVAL", "0.010"))
        self.cap: Optional[cv2.VideoCapture] = None
        self._open_camera()
        threading.Thread(target=self._update, daemon=True).start()
        threading.Thread(target=self._watchdog, daemon=True).start()

    def _open_camera(self) -> None:
        idx = _normalize_cam(self.index)
        try:
            if self.cap is not None: self.cap.release()
        except Exception: pass
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index/path {idx}")
        try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        try: self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception: pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, min(self.fps, 20))
        self.cap.read()
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self.cap.get(cv2.CAP_PROP_FPS)
        four = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        four_s = "".join([chr((four >> 8 * i) & 0xFF) for i in range(4)]).strip("\x00")
        print(f"[OPEN] {idx} -> {w}x{h}@{f:.1f} FOURCC={four_s}")

    def _update(self) -> None:
        target_interval = 1.0 / max(1, self.fps)
        last_t = 0.0
        skip = 0
        while not self.stopped:
            ok, frame = self.cap.read() if self.cap is not None else (False, None)
            now = time.time()
            if ok and frame is not None:
                if skip < (SKIP_FRAMES - 1):
                    skip += 1
                else:
                    skip = 0
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                    with self.lock:
                        self.ret = True
                        self.frame = frame
                        self.last_update_ts = now
            sleep_left = max(self.read_interval, target_interval - (now - last_t))
            if sleep_left > 0: time.sleep(sleep_left)
            last_t = now

    def _watchdog(self) -> None:
        while not self.stopped:
            time.sleep(0.5)
            if self.reopen_stale_sec <= 0: continue
            now = time.time()
            if (now - self.last_update_ts) > self.reopen_stale_sec:
                try:
                    print(f"[WATCHDOG] Reopening camera {self.index} (stale {now - self.last_update_ts:.1f}s)")
                    self._open_camera()
                except Exception as e:
                    print(f"[WATCHDOG] reopen failed for {self.index}: {e}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            return self.ret, None if self.frame is None else self.frame.copy()

    def stop(self) -> None:
        self.stopped = True
        try:
            if self.cap is not None: self.cap.release()
        except Exception: pass

# ---------- Database helpers ----------
def init_db(path: str) -> None:
    import sqlite3
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            ped_count INTEGER,
            veh_count INTEGER,
            tl_color TEXT,
            nearest_vehicle_distance_m REAL,
            avg_vehicle_speed_mps REAL,
            action TEXT,
            scenario TEXT
        );
        """
    )
    con.commit()
    con.close()

def log_event(path: str, ts: float, ped_count: int, veh_count: int, tl_color: str,
              nearest_m: float, avg_mps: float, action: str, scenario: str) -> None:
    import sqlite3
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO events (ts, ped_count, veh_count, tl_color, nearest_vehicle_distance_m, avg_vehicle_speed_mps, action, scenario)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (ts, ped_count, veh_count, tl_color, nearest_m, avg_mps, action, scenario),
    )
    con.commit()
    con.close()

# ---------- Main pipeline ----------
_last_pub: dict[str, float] = {}

def run_pipeline() -> None:
    init_db(DB_PATH)

    i_ped = int(os.getenv("SC_CAM_PED", "0"))
    i_veh = int(os.getenv("SC_CAM_VEH", "1"))
    i_tl  = int(os.getenv("SC_CAM_TL",  "2"))

    cam_ped = CameraStream(i_ped, FRAME_W, FRAME_H, FPS)
    cam_veh = CameraStream(i_veh, FRAME_W, FRAME_H, FPS)
    cam_tl  = CameraStream(i_tl,  FRAME_W, FRAME_H, FPS)

    print(f"[Pedestrian Cam] {i_ped}")
    print(f"[Vehicle Cam]    {i_veh}")
    print(f"[Traffic Light]  {i_tl}")

    mc = build_mqtt()
    last_status_ts = 0.0
    last_log_ts = 0.0

    while True:
        t0 = time.time()
        rp, fp = cam_ped.read()
        rv, fv = cam_veh.read()
        rt, ft = cam_tl.read()

        ok_ped = bool(rp and fp is not None)
        ok_veh = bool(rv and fv is not None)
        ok_tl  = bool(rt and ft is not None)

        # Base frames are the current camera frames (never blank them)
        fp_s = fp if ok_ped else None
        fv_s = fv if ok_veh else None
        ft_s = ft if ok_tl  else None

        # Prefer recent annotated frames; if not fresh, KEEP the last annotated (sticky) to avoid flicker.
        now_chk = time.time()
        with viz_lock:
            vp, tp = viz_frames.get("ped", (None, 0.0))
            vv, tv = viz_frames.get("veh", (None, 0.0))
            vt, tt = viz_frames.get("tl",  (None, 0.0))

        def choose(base, annotated, tstamp):
            if annotated is not None and (now_chk - tstamp) <= VIZ_MAX_AGE_S:
                return cv2.resize(annotated, (FRAME_W, FRAME_H))
            # If annotated exists but stale, still keep it (sticky) rather than jumping back to raw.
            if annotated is not None:
                return cv2.resize(annotated, (FRAME_W, FRAME_H))
            # No annotated yet — use base if present
            if base is not None:
                return cv2.resize(base, (FRAME_W, FRAME_H))
            # Fallback black
            return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

        fp_s = choose(fp_s, vp, tp)
        fv_s = choose(fv_s, vv, tv)
        ft_s = choose(ft_s, vt, tt)

        # Draw lane line on vehicles feed (top-most)
        cv2.line(fv_s, (0, LANE_Y), (FRAME_W, LANE_Y), (0, 255, 255), 2)

        # Publish frames to the web dashboard via SocketIO (throttled)
        now_ui = time.time()
        if now_ui - _last_pub.setdefault("ui", 0.0) >= (1.0 / max(1.0, PUBLISH_HZ)):
            publish_frame("ped", fp_s)
            publish_frame("veh", fv_s)
            publish_frame("tl",  ft_s)
            _last_pub["ui"] = now_ui

        # Publish raw JPEG frames to the cloud worker (throttled)
        now_mq = time.time()
        if now_mq - _last_pub.setdefault("mqtt", 0.0) >= (1.0 / max(1.0, PUBLISH_HZ)):
            jb = _jpeg(fp_s)
            jc = _jpeg(fv_s)
            jt = _jpeg(ft_s)
            if jb is not None: mc.publish(TOPIC_PED, jb, qos=0, retain=False)
            if jc is not None: mc.publish(TOPIC_VEH, jc, qos=0, retain=False)
            if jt is not None: mc.publish(TOPIC_TL,  jt, qos=0, retain=False)
            _last_pub["mqtt"] = now_mq

        # Status/LED
        now = time.time()
        if now - last_status_ts >= STATUS_MIN_PERIOD:
            publish_status_from_loop(
                now_ts=now,
                ped_count=int(last_decision.get("ped_count", 0)),
                veh_count=int(last_decision.get("veh_count", 0)),
                tl_color=str(last_decision.get("tl_color", "unknown")),
                nearest_m=float(last_decision.get("nearest_m", 0.0)),
                avg_mps=float(last_decision.get("avg_mps", 0.0)),
                action=last_decision.get("action", "OFF"),
                scenario=last_decision.get("scenario", "baseline"),
                flags={"night": time.localtime(now).tm_hour >= 21, "rush": time.localtime(now).tm_hour == 7},
                extra={"ambulance": False},
            )
            scen = str(last_decision.get("scenario", "baseline"))
            msg = SCENARIO_MESSAGES.get(scen) or str(last_decision.get("action", "OFF"))
            try: show_led(msg.upper())
            except Exception as e: print("[LED] update error:", repr(e))
            last_status_ts = now

        # DB logging
        if now - last_log_ts >= LOG_EVERY_SEC:
            log_event(
                DB_PATH, now,
                int(last_decision.get("ped_count", 0)),
                int(last_decision.get("veh_count", 0)),
                str(last_decision.get("tl_color", "unknown")),
                float(last_decision.get("nearest_m", 0.0)),
                float(last_decision.get("avg_mps", 0.0)),
                str(last_decision.get("action", "OFF")),
                str(last_decision.get("scenario", "baseline")),
            )
            last_log_ts = now

        # pacing
        elapsed = time.time() - t0
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)

def _late_import_flask_app() -> None:
    global publish_frame, publish_status_from_loop, start_http_server
    try:
        from flask_app import (
            publish_frame as _pf,
            publish_status_from_loop as _ps,
            start_http_server as _sh,
        )
        publish_frame = _pf
        publish_status_from_loop = _ps
        start_http_server = _sh
        print("[flask_app] module import completed")
    except Exception as e:
        print("[flask_app] import failed:", repr(e))

if __name__ == "__main__":
    _late_import_flask_app()
    threading.Thread(target=run_pipeline, daemon=True).start()
    start_http_server(host="0.0.0.0", port=5000)
