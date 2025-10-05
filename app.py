# app.py  (Pi I/O only) — keeps your same UI & LED, offloads YOLO/logic to cloud via MQTT

# Import the web app FIRST so eventlet is monkey-patched before anything else.
from flask_app import publish_frame, publish_status_from_loop, start_http_server

import os, re, cv2, time, math, sqlite3, threading, json
from queue import Queue, Empty, Full
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Union

import numpy as np
import paho.mqtt.client as mqtt
from urllib.parse import urlparse, unquote

# Keep OpenCV predictable/light on Pi
cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

SHOW_WINDOWS = False
PRINT_DEBUG = True

# ---------- CONFIG ----------
# Robustness / perf toggles (kept for compatibility, some now unused)
FALLBACK_SYNC_AFTER_S = float(os.getenv("SC_FALLBACK_SYNC_AFTER_S", "0.6"))

# Frames
FRAME_W = int(os.getenv("SC_FRAME_W", "640"))
FRAME_H = int(os.getenv("SC_FRAME_H", "360"))
FPS_TARGET = int(os.getenv("SC_FPS", "15"))
FRAME_TIME = 1.0 / max(1, FPS_TARGET)
SKIP_FRAMES = int(os.getenv("SC_SKIP", "1"))

# Throttle JPEG encodes only (keeps UI smooth)
PUBLISH_HZ = float(os.getenv("SC_PUBLISH_HZ", "10"))
_last_pub = {"ped": 0.0, "veh": 0.0, "tl": 0.0}
JPEG_QUALITY = int(os.getenv("SC_JPEG_QUALITY", "70"))

# Vehicle lane helper for UI overlay
PEDESTRIAN_LANE_Y = int(os.getenv("SC_LANE_Y", "250"))

# DB + status throttling
DB_PATH = os.getenv("SC_DB", "smart_crosswalk.db")
LOG_EVERY_SEC = int(os.getenv("SC_LOG_SEC", "30"))
STATUS_MIN_PERIOD = float(os.getenv("SC_STATUS_PERIOD", "0.25"))

# Colors
COLOR_GREEN  = (0,255,0)
COLOR_RED    = (0,0,255)
COLOR_YELLOW = (0,255,255)
COLOR_WHITE  = (255,255,255)
COLOR_BLUE   = (255,0,0)

# ---------- LED ----------
def _init_led():
    try:
        from luma.core.interface.serial import spi, noop
        from luma.led_matrix.device import max7219
        from luma.core.render import canvas
        from PIL import ImageFont

        serial = spi(port=0, device=0, gpio=noop())
        device = max7219(serial, cascaded=int(os.getenv("SC_LED_CASCADE","4")),
                         block_orientation=int(os.getenv("SC_LED_ORIENTATION","-90")), rotate=0)
        font = ImageFont.load_default()

        def show_led(msg: str):
            with canvas(device) as draw:
                draw.text((1, -2), (msg or "")[:10], fill="white", font=font)
        if PRINT_DEBUG: print("[LED] MAX7219 initialized")
        return show_led
    except Exception as e:
        if PRINT_DEBUG: print("[LED] Fallback console:", repr(e))
        return lambda msg: print("[LED]", msg)

show_led = _init_led()

# ---------- MQTT ----------
SITE_ID    = os.getenv("SC_SITE_ID", "adsmn-01")
BROKER_URL = os.getenv("SC_MQTT_URL", "mqtts://user:pass@host:8883")

u = urlparse(BROKER_URL)
MQTT_HOST = u.hostname or "localhost"
MQTT_PORT = u.port or (8883 if u.scheme == "mqtts" else 1883)
MQTT_TLS  = (u.scheme == "mqtts")
MQTT_USER = unquote(u.username) if u.username else None
MQTT_PASS = unquote(u.password) if u.password else None

TOPIC_PED = f"crosswalk/{SITE_ID}/frames/ped"
TOPIC_VEH = f"crosswalk/{SITE_ID}/frames/veh"
TOPIC_TL  = f"crosswalk/{SITE_ID}/frames/tl"
TOPIC_DEC = f"crosswalk/{SITE_ID}/decision"

# Last decision from cloud
last_decision = {
    "ts": 0.0,
    "ped_count": 0,
    "veh_count": 0,
    "tl_color": "unknown",
    "nearest_m": 0.0,
    "avg_mps": 0.0,
    "action": "OFF",
    "scenario": "baseline",
}

def _to_jpeg(frame: np.ndarray) -> Optional[bytes]:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok: return None
    return buf.tobytes()

def on_connect(client, userdata, flags, rc):
    print("[MQTT] Connected rc=", rc)
    client.subscribe(TOPIC_DEC, qos=1)

def on_message(client, userdata, msg):
    global last_decision
    if msg.topic == TOPIC_DEC:
        try:
            d = json.loads(msg.payload.decode("utf-8","ignore"))
            # merge with defaults
            for k in last_decision.keys():
                if k in d: last_decision[k] = d[k]
            # push into your existing UI/status path
            publish_status_from_loop(
                now_ts=float(last_decision.get("ts", time.time())),
                ped_count=int(last_decision.get("ped_count",0)),
                veh_count=int(last_decision.get("veh_count",0)),
                tl_color=str(last_decision.get("tl_color","unknown")),
                nearest_m=float(last_decision.get("nearest_m",0.0)),
                avg_mps=float(last_decision.get("avg_mps",0.0)),
                flags={"night": time.localtime().tm_hour >= 21, "rush": time.localtime().tm_hour == 7},
                extra={"ambulance": False},
            )
            # Drive LED
            show_led(str(last_decision.get("action","OFF")))
        except Exception as e:
            print("[MQTT] decision parse error:", repr(e))

def build_mqtt():
    c = mqtt.Client(client_id=f"pi-io-{SITE_ID}", clean_session=True)
    if MQTT_USER: c.username_pw_set(MQTT_USER, MQTT_PASS or "")
    if MQTT_TLS:  c.tls_set()  # system CA
    c.on_connect = on_connect
    c.on_message = on_message
    c.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    threading.Thread(target=c.loop_forever, daemon=True).start()
    return c

# ---------- DB ----------
def init_db(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            ped_count INTEGER,
            veh_count INTEGER,
            tl_color TEXT,
            nearest_vehicle_distance_m REAL,
            avg_vehicle_speed_mps REAL,
            action TEXT
        );
    """)
    con.commit(); con.close()

def log_event(path, ts, ped_count, veh_count, tl_color, nearest_m, avg_mps, action):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute(
        "INSERT INTO events (ts,ped_count,veh_count,tl_color,nearest_vehicle_distance_m,avg_vehicle_speed_mps,action) VALUES (?,?,?,?,?,?,?)",
        (ts, ped_count, veh_count, tl_color, nearest_m, avg_mps, action)
    )
    con.commit(); con.close()

# ---------- UTIL ----------
def _normalize_cam(value: Union[str,int,None]):
    if value is None: return None
    if isinstance(value,int): return value
    s = str(value).strip()
    if s.isdigit(): return int(s)
    try:
        if s.startswith("/dev/"):
            real = os.path.realpath(s)
            m = re.match(r"^/dev/video(\d+)$", real)
            if m: return int(m.group(1))
            if os.path.exists(real): return real
    except Exception: pass
    m = re.match(r"^/dev/video(\d+)$", s)
    return int(m.group(1)) if m else s

def publish_frame_throttled(key: str, frame: np.ndarray):
    now = time.time()
    if now - _last_pub.get(key,0.0) >= (1.0/max(1.0,PUBLISH_HZ)):
        publish_frame(key, frame)
        _last_pub[key] = now

# ---------- CAMERA ----------
class CameraStream:
    def __init__(self, index, width, height, fps):
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.stopped = False
        self._open_camera()
        threading.Thread(target=self._update, daemon=True).start()

    def _open_camera(self):
        idx = _normalize_cam(self.index)
        if isinstance(idx,int):
            self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not self.cap.isOpened(): self.cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if not self.cap.isOpened(): raise RuntimeError(f"Could not open camera index {idx}")
            pretty = f"/dev/video{idx}"
        else:
            self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not self.cap.isOpened(): self.cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if not self.cap.isOpened(): raise RuntimeError(f"Could not open camera path {idx}")
            pretty = str(idx)

        try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, min(self.fps, 20))

        def set_fourcc(code):
            f = cv2.VideoWriter_fourcc(*code)
            self.cap.set(cv2.CAP_PROP_FOURCC, f)
            return int(self.cap.get(cv2.CAP_PROP_FOURCC)) == f
        if not set_fourcc('MJPG'):
            set_fourcc('YUYV') or set_fourcc('YUY2')

        self.cap.read()

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self.cap.get(cv2.CAP_PROP_FPS)
        four = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        four_s = "".join([chr((four >> 8*i) & 0xFF) for i in range(4)])
        print(f"[OPEN] {pretty} -> {w}x{h}@{f:.1f} FOURCC={four_s}")

    def _reopen_once(self):
        try: self.cap.release()
        except Exception: pass
        time.sleep(0.2)
        self._open_camera()

    def _update(self):
        no_frame = 0
        frame_interval = 1.0 / max(1,self.fps)
        last_t = 0.0
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                no_frame += 1
                if no_frame == 20:
                    print("[INFO] Reopening camera due to stalled frames…")
                    self._reopen_once()
                    no_frame = 0
                time.sleep(0.02)
                continue
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width,self.height), interpolation=cv2.INTER_AREA)
            with self.lock:
                self.ret, self.frame = True, frame
            now = time.time()
            sleep_left = frame_interval - (now - last_t)
            if sleep_left > 0: time.sleep(sleep_left)
            last_t = now

    def read(self):
        with self.lock:
            return self.ret, None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        try: self.cap.release()
        except Exception: pass

# ---------- PIPELINE ----------
def run_pipeline():
    init_db(DB_PATH)

    # camera selection: keep your same behavior (env or auto 0/1/2)
    def pick_cameras():
        ped_env, veh_env, tl_env = os.getenv("SC_CAM_PED"), os.getenv("SC_CAM_VEH"), os.getenv("SC_CAM_TL")
        if ped_env and veh_env and tl_env:
            return _normalize_cam(ped_env), _normalize_cam(veh_env), _normalize_cam(tl_env)
        return 0, 1, 2

    i_ped, i_veh, i_tl = pick_cameras()
    cam_ped = CameraStream(i_ped, FRAME_W, FRAME_H, FPS_TARGET)
    cam_veh = CameraStream(i_veh, FRAME_W, FRAME_H, FPS_TARGET)
    cam_tl  = CameraStream(i_tl,  FRAME_W, FRAME_H, FPS_TARGET)
    print(f"[Pedestrian Cam] {i_ped}")
    print(f"[Vehicle Cam]    {i_veh}")
    print(f"[Traffic Light]  {i_tl}")

    # MQTT client (to publish frames & receive decision)
    mc = build_mqtt()

    # Status loop timing
    last_log_ts = 0.0
    last_status_ts = 0.0

    try:
        while True:
            loop_start = time.time()

            rp, fp = cam_ped.read()
            rv, fv = cam_veh.read()
            rt, ft = cam_tl.read()

            ok_ped = bool(rp and fp is not None)
            ok_veh = bool(rv and fv is not None)
            ok_tl  = bool(rt and ft is not None)

            # If no frame, show black
            blank_640x360 = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            fp_s = fp if ok_ped else blank_640x360
            fv_s = fv if ok_veh else blank_640x360
            ft_s = ft if ok_tl  else blank_640x360

            # simple vehicle lane overlay for context
            cv2.line(fv_s,(0,PEDESTRIAN_LANE_Y),(FRAME_W,PEDESTRIAN_LANE_Y),COLOR_YELLOW,2)

            # Publish to your existing UI (same as before)
            publish_frame_throttled("ped", fp_s)
            publish_frame_throttled("veh", fv_s)
            publish_frame_throttled("tl",  ft_s)

            # Publish JPEG frames to cloud (throttled)
            now_pub = time.time()
            if now_pub - _last_pub.get("mqtt", 0.0) >= (1.0/max(1.0,PUBLISH_HZ)):
                jb = _to_jpeg(fp_s);  jc = _to_jpeg(fv_s);  jt = _to_jpeg(ft_s)
                if jb is not None: mc.publish(TOPIC_PED, jb, qos=0, retain=False)
                if jc is not None: mc.publish(TOPIC_VEH, jc, qos=0, retain=False)
                if jt is not None: mc.publish(TOPIC_TL,  jt, qos=0, retain=False)
                _last_pub["mqtt"] = now_pub

            # Show the latest decision in UI at STATUS_MIN_PERIOD cadence
            now = time.time()
            if now - last_status_ts >= STATUS_MIN_PERIOD:
                publish_status_from_loop(
                    now_ts=now,
                    ped_count=int(last_decision.get("ped_count",0)),
                    veh_count=int(last_decision.get("veh_count",0)),
                    tl_color=str(last_decision.get("tl_color","unknown")),
                    nearest_m=float(last_decision.get("nearest_m",0.0)),
                    avg_mps=float(last_decision.get("avg_mps",0.0)),
                    flags={"night": time.localtime(now).tm_hour >= 21, "rush": time.localtime(now).tm_hour == 7},
                    extra={"ambulance": False},
                )
                last_status_ts = now
                # Also keep the LED consistent with the latest cloud action
                show_led(str(last_decision.get("action","OFF")))

            # DB logging at intervals (uses last decision)
            if now - last_log_ts >= LOG_EVERY_SEC:
                log_event(
                    DB_PATH, now,
                    int(last_decision.get("ped_count",0)),
                    int(last_decision.get("veh_count",0)),
                    str(last_decision.get("tl_color","unknown")),
                    float(last_decision.get("nearest_m",0.0)),
                    float(last_decision.get("avg_mps",0.0)),
                    str(last_decision.get("action","OFF")),
                )
                last_log_ts = now

            if SHOW_WINDOWS:
                cv2.imshow("Ped", fp_s); cv2.imshow("Veh", fv_s); cv2.imshow("TL", ft_s)
                if (cv2.waitKey(1)&0xFF)==27: break

            elapsed = time.time() - loop_start
            if elapsed < FRAME_TIME: time.sleep(FRAME_TIME - elapsed)

    finally:
        for c in (cam_ped, cam_veh, cam_tl):
            try: c.stop()
            except Exception: pass
        if SHOW_WINDOWS: cv2.destroyAllWindows()

if __name__ == "__main__":
    threading.Thread(target=run_pipeline, daemon=True).start()
    start_http_server(host="0.0.0.0", port=5000)
