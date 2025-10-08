# app.py — Raspberry Pi I/O + Dashboard
# Import the web app FIRST so eventlet is monkey-patched before anything else.
from flask_app import publish_frame, publish_status_from_loop, start_http_server

import os, re, cv2, time, math, sqlite3, threading, json, ssl
from queue import Queue
from urllib.parse import urlparse, unquote
from typing import Optional, Union

import numpy as np
import paho.mqtt.client as mqtt

cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

SHOW_WINDOWS = False
PRINT_DEBUG = True

# ---------- ENV ----------
SITE_ID  = os.getenv("SC_SITE_ID", "adsmn-01")

# Camera + UI perf
FRAME_W   = int(os.getenv("SC_FRAME_W", "640"))
FRAME_H   = int(os.getenv("SC_FRAME_H", "360"))
FPS       = int(os.getenv("SC_FPS", "8"))
FRAME_TIME = 1.0 / max(1, FPS)
SKIP_FRAMES = int(os.getenv("SC_SKIP", "1"))
PUBLISH_HZ  = float(os.getenv("SC_PUBLISH_HZ", "6"))
JPEG_QUALITY= int(os.getenv("SC_JPEG_QUALITY", "60"))

LANE_Y      = int(os.getenv("SC_LANE_Y", "250"))

# MQTT split fields
MQTT_HOST = os.getenv("MQTT_HOST")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_USER = os.getenv("MQTT_USERNAME")
MQTT_PASS = os.getenv("MQTT_PASSWORD")
MQTT_TLS  = os.getenv("MQTT_TLS", "1") == "1"

# Topics
TOPIC_PED = f"crosswalk/{SITE_ID}/frames/ped"
TOPIC_VEH = f"crosswalk/{SITE_ID}/frames/veh"
TOPIC_TL  = f"crosswalk/{SITE_ID}/frames/tl"
TOPIC_DEC = f"crosswalk/{SITE_ID}/decision"
TOPIC_VPED= f"crosswalk/{SITE_ID}/viz/ped"
TOPIC_VVEH= f"crosswalk/{SITE_ID}/viz/veh"
TOPIC_VTL = f"crosswalk/{SITE_ID}/viz/tl"

# DB
DB_PATH = os.getenv("SC_DB", "smart_crosswalk.db")
LOG_EVERY_SEC = int(os.getenv("SC_LOG_SEC", "30"))
STATUS_MIN_PERIOD = float(os.getenv("SC_STATUS_PERIOD", "0.20"))

# Colors
COLOR_YELLOW=(0,255,255)
COLOR_WHITE =(255,255,255)

_last_pub = {"mqtt": 0.0, "ui": 0.0}

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
        print("[LED] MAX7219 initialized")
        return show_led
    except Exception as e:
        print("[LED] Fallback console:", repr(e))
        return lambda msg: print("[LED]", msg)

show_led = _init_led()

# ---------- MQTT (Pi) ----------
viz_lock = threading.Lock()
viz_frames = {"ped": None, "veh": None, "tl": None}
last_decision = {
    "ts": 0.0, "ped_count": 0, "veh_count": 0, "tl_color": "unknown",
    "nearest_m": 0.0, "avg_mps": 0.0, "action": "OFF", "scenario": "baseline"
}

def _to_jpeg(img: np.ndarray) -> Optional[bytes]:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buf.tobytes() if ok else None

def on_connect(c, u, f, rc, p=None):
    print(f"[MQTT] Connected rc={rc}")
    c.subscribe([(TOPIC_DEC,1), (TOPIC_VPED,0), (TOPIC_VVEH,0), (TOPIC_VTL,0)])

def on_message(c, u, msg):
    global last_decision, viz_frames
    try:
        if msg.topic == TOPIC_DEC:
            d = json.loads(msg.payload.decode("utf-8","ignore"))
            for k in last_decision.keys():
                if k in d: last_decision[k] = d[k]
            # push to UI and LED
            publish_status_from_loop(
                now_ts=float(last_decision.get("ts", time.time())),
                ped_count=int(last_decision.get("ped_count",0)),
                veh_count=int(last_decision.get("veh_count",0)),
                tl_color=str(last_decision.get("tl_color","unknown")),
                nearest_m=float(last_decision.get("nearest_m",0.0)),
                avg_mps=float(last_decision.get("avg_mps",0.0)),
                flags={"night": time.localtime().tm_hour>=21, "rush": time.localtime().tm_hour==7},
                extra={"ambulance": False},
            )
            show_led(str(last_decision.get("action","OFF")))
        else:
            arr = np.frombuffer(msg.payload, dtype=np.uint8)
            im  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            with viz_lock:
                if msg.topic == TOPIC_VPED: viz_frames["ped"] = im
                elif msg.topic == TOPIC_VVEH: viz_frames["veh"] = im
                elif msg.topic == TOPIC_VTL:  viz_frames["tl"]  = im
    except Exception as e:
        print("[MQTT] on_message error:", repr(e))

def build_mqtt():
    if not MQTT_HOST:
        raise RuntimeError("MQTT_HOST not set. Source .env.pi first.")
    c = mqtt.Client(client_id=f"pi-io-{SITE_ID}", clean_session=True, protocol=mqtt.MQTTv311)
    if MQTT_USER:
        c.username_pw_set(MQTT_USER, MQTT_PASS or None)
    if MQTT_TLS:
        c.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS_CLIENT)
        c.tls_insecure_set(False)
    c.on_connect    = on_connect
    c.on_message    = on_message
    c.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    c.loop_start()
    return c

# ---------- Cameras ----------
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
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not self.cap.isOpened(): self.cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if not self.cap.isOpened(): raise RuntimeError(f"Could not open camera index/path {idx}")
        try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, min(self.fps, 20))
        # Prefer YUYV -> fewer CPU cycles to decode than MJPG on some sticks
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        self.cap.read()
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self.cap.get(cv2.CAP_PROP_FPS)
        four = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        four_s = "".join([chr((four >> 8*i) & 0xFF) for i in range(4)])
        print(f"[OPEN] {idx} -> {w}x{h}@{f:.1f} FOURCC={four_s}")

    def _update(self):
        frame_interval = 1.0 / max(1,self.fps)
        last_t = 0.0
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
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

# ---------- MAIN LOOP ----------
def run_pipeline():
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

        # Acquire latest local frames
        rp, fp = cam_ped.read()
        rv, fv = cam_veh.read()
        rt, ft = cam_tl.read()

        ok_ped = bool(rp and fp is not None)
        ok_veh = bool(rv and fv is not None)
        ok_tl  = bool(rt and ft is not None)

        blank = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        fp_s = fp if ok_ped else blank
        fv_s = fv if ok_veh else blank
        ft_s = ft if ok_tl  else blank

        # Try to use cloud "viz" frames if present (to show boxes)
        with viz_lock:
            vp = viz_frames["ped"]
            vv = viz_frames["veh"]
            vt = viz_frames["tl"]
        if vp is not None: fp_s = cv2.resize(vp, (FRAME_W, FRAME_H))
        if vv is not None:
            vv = cv2.resize(vv, (FRAME_W, FRAME_H))
            cv2.line(vv, (0, LANE_Y), (FRAME_W, LANE_Y), COLOR_YELLOW, 2)
            fv_s = vv
        if vt is not None: ft_s = cv2.resize(vt, (FRAME_W, FRAME_H))

        # Publish to dashboard (SocketIO)
        now_ui = time.time()
        if now_ui - _last_pub["ui"] >= (1.0 / max(1.0, PUBLISH_HZ)):
            publish_frame("ped", fp_s)
            publish_frame("veh", fv_s)
            publish_frame("tl",  ft_s)
            _last_pub["ui"] = now_ui

        # Publish JPEG frames to cloud (raw) — throttled
        now_mq = time.time()
        if now_mq - _last_pub["mqtt"] >= (1.0 / max(1.0, PUBLISH_HZ)):
            jb = _to_jpeg(fp_s);  jc = _to_jpeg(fv_s);  jt = _to_jpeg(ft_s)
            if jb is not None: mc.publish(TOPIC_PED, jb, qos=0, retain=False)
            if jc is not None: mc.publish(TOPIC_VEH, jc, qos=0, retain=False)
            if jt is not None: mc.publish(TOPIC_TL,  jt, qos=0, retain=False)
            _last_pub["mqtt"] = now_mq

        # Periodic status + LED synced to cloud decision
        now = time.time()
        if now - last_status_ts >= STATUS_MIN_PERIOD:
            publish_status_from_loop(
                now_ts=now,
                ped_count=int(last_decision.get("ped_count",0)),
                veh_count=int(last_decision.get("veh_count",0)),
                tl_color=str(last_decision.get("tl_color","unknown")),
                nearest_m=float(last_decision.get("nearest_m",0.0)),
                avg_mps=float(last_decision.get("avg_mps",0.0)),
                flags={"night": time.localtime(now).tm_hour>=21, "rush": time.localtime(now).tm_hour==7},
                extra={"ambulance": False},
            )
            show_led(str(last_decision.get("action","OFF")))
            last_status_ts = now

        # DB log every N sec
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

        # pacing
        elapsed = time.time() - t0
        if elapsed < FRAME_TIME: time.sleep(FRAME_TIME - elapsed)

if __name__ == "__main__":
    # Avoid slow greendns TLS on Pi
    os.environ.setdefault("EVENTLET_NO_GREENDNS", "YES")
    threading.Thread(target=run_pipeline, daemon=True).start()
    start_http_server(host="0.0.0.0", port=5000)
