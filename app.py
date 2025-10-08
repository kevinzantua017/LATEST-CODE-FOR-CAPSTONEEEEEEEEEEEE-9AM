# app.py — Pi UI + camera publisher + overlay from cloud detections
from flask_app import publish_frame, publish_status_from_loop, start_http_server

import os, re, cv2, time, math, sqlite3, threading, json, ssl
from queue import Queue
from dataclasses import dataclass
from typing import Union, Optional, Dict, Tuple, List

import numpy as np
import paho.mqtt.client as mqtt
from urllib.parse import urlparse, unquote

cv2.setNumThreads(1)
try: cv2.ocl.setUseOpenCL(False)
except Exception: pass

SHOW_WINDOWS = False
PRINT_DEBUG = True

# ---------- CONFIG ----------
FRAME_W = int(os.getenv("SC_FRAME_W", "424"))
FRAME_H = int(os.getenv("SC_FRAME_H", "240"))
FPS_TARGET = int(os.getenv("SC_FPS", "12"))
FRAME_TIME = 1.0 / max(1, FPS_TARGET)

PUBLISH_HZ = float(os.getenv("SC_PUBLISH_HZ", "6"))
JPEG_QUALITY = int(os.getenv("SC_JPEG_QUALITY", "55"))
_last_pub_ts = 0.0

PEDESTRIAN_LANE_Y = int(os.getenv("SC_LANE_Y", "250"))

DB_PATH = os.getenv("SC_DB", "smart_crosswalk.db")
LOG_EVERY_SEC = int(os.getenv("SC_LOG_SEC", "30"))
STATUS_MIN_PERIOD = float(os.getenv("SC_STATUS_PERIOD", "0.25"))

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
        print("[LED] MAX7219 initialized")
        return show_led
    except Exception as e:
        print("[LED] Fallback console:", repr(e))
        return lambda msg: print("[LED]", msg)

show_led = _init_led()

# ---------- MQTT ----------
SITE_ID = os.getenv("SC_SITE_ID", "adsmn-01")
BROKER_URL = os.getenv("SC_MQTT_URL")  # optional single URL

# split fields (preferred)
MQTT_HOST = os.getenv("MQTT_HOST") or (urlparse(BROKER_URL).hostname if BROKER_URL else None)
MQTT_PORT = int(os.getenv("MQTT_PORT", urlparse(BROKER_URL).port if BROKER_URL and urlparse(BROKER_URL).port else "8883"))
MQTT_TLS  = os.getenv("MQTT_TLS", "1") == "1" if not BROKER_URL else (urlparse(BROKER_URL).scheme == "mqtts")
MQTT_USER = os.getenv("MQTT_USERNAME") or (unquote(urlparse(BROKER_URL).username) if BROKER_URL else None)
MQTT_PASS = os.getenv("MQTT_PASSWORD") or (unquote(urlparse(BROKER_URL).password) if BROKER_URL else None)

TOPIC_PED = f"crosswalk/{SITE_ID}/frames/ped"
TOPIC_VEH = f"crosswalk/{SITE_ID}/frames/veh"
TOPIC_TL  = f"crosswalk/{SITE_ID}/frames/tl"
TOPIC_DET = f"crosswalk/{SITE_ID}/detections"
TOPIC_DEC = f"crosswalk/{SITE_ID}/decision"

latest_det = {
    "ts": 0.0,
    "tl_color": "unknown",
    "ped": [],  # [x1,y1,x2,y2,conf]
    "veh": [],  # [x1,y1,x2,y2,conf,speed_mps,dist_m]
}
latest_dec = {
    "ts": 0.0, "ped_count":0, "veh_count":0, "tl_color":"unknown",
    "nearest_m":0.0, "avg_mps":0.0, "action":"OFF", "scenario":"baseline"
}

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

# ---------- Utils ----------
def _normalize_cam(value: Union[str,int,None]):
    if value is None: return None
    if isinstance(value,int): return value
    s = str(value).strip()
    if s.isdigit(): return int(s)
    m = re.match(r"^/dev/video(\d+)$", s)
    return int(m.group(1)) if m else s

def _to_jpeg(frame: np.ndarray) -> Optional[bytes]:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buf.tobytes() if ok else None

def draw_overlays(ped_frame: np.ndarray, veh_frame: np.ndarray, tl_frame: np.ndarray):
    # TL label
    tlc = latest_det.get("tl_color","unknown")
    cv2.putText(tl_frame, f"TL: {tlc.upper()}", (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

    # Ped boxes
    for (x1,y1,x2,y2,cf) in latest_det.get("ped",[]):
        cv2.rectangle(ped_frame,(x1,y1),(x2,y2),COLOR_GREEN,2)
        cv2.putText(ped_frame, f"person {cf:.2f}", (x1,max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREEN,1)
    cv2.putText(ped_frame, f"Pedestrians: {len(latest_det.get('ped',[]))}", (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

    # Veh boxes + lane + speed/dist
    cv2.line(veh_frame,(0,PEDESTRIAN_LANE_Y),(FRAME_W,PEDESTRIAN_LANE_Y),COLOR_YELLOW,2)
    nearest = float('inf'); speeds=[]
    for (x1,y1,x2,y2,cf,spd,dist) in latest_det.get("veh",[]):
        cv2.rectangle(veh_frame,(x1,y1),(x2,y2),COLOR_RED,2)
        label = f"{cf:.2f} | {spd*3.6:.0f} km/h | {dist:.1f} m"
        cv2.putText(veh_frame, label, (x1,max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_RED,1)
        nearest = min(nearest, dist); speeds.append(spd)
    avg_mps = float(np.mean(speeds)) if speeds else 0.0
    cv2.putText(veh_frame, f"Vehicles: {len(latest_det.get('veh',[]))}", (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE,2)
    cv2.putText(veh_frame, f"Nearest: {0.0 if nearest==float('inf') else nearest:.1f} m", (8,44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE,2)
    cv2.putText(veh_frame, f"Avg: {avg_mps:.1f} m/s", (8,68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE,2)

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
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened(): cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if not cap.isOpened(): raise RuntimeError(f"Could not open camera index/path {idx}")

        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # fast if supported
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, min(self.fps, 15))
        self.cap = cap

        # prime
        self.cap.read()
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self.cap.get(cv2.CAP_PROP_FPS)
        four = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        four_s = "".join([chr((four >> 8*i) & 0xFF) for i in range(4)])
        print(f"[OPEN] /dev/video{idx if isinstance(idx,int) else idx} -> {w}x{h}@{f:.1f} FOURCC={four_s}")

    def _update(self):
        frame_interval = 1.0 / max(1,self.fps)
        last_t = 0.0
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01); continue
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

# ---------- MQTT handlers ----------
def on_connect(c, u, f, rc, p=None):
    print(f"[MQTT] Connected rc={rc}")
    c.subscribe([(TOPIC_DET,0),(TOPIC_DEC,1)])

def on_message(c, u, msg):
    global latest_det, latest_dec
    try:
        payload = json.loads(msg.payload.decode("utf-8","ignore"))
        if msg.topic == TOPIC_DET:
            latest_det = {
                "ts": float(payload.get("ts", time.time())),
                "tl_color": payload.get("tl_color","unknown"),
                "ped": payload.get("ped", []),
                "veh": payload.get("veh", []),
            }
        elif msg.topic == TOPIC_DEC:
            latest_dec.update(payload)
            # reflect on UI + LED
            publish_status_from_loop(
                now_ts=float(latest_dec.get("ts", time.time())),
                ped_count=int(latest_dec.get("ped_count",0)),
                veh_count=int(latest_dec.get("veh_count",0)),
                tl_color=str(latest_dec.get("tl_color","unknown")),
                nearest_m=float(latest_dec.get("nearest_m",0.0)),
                avg_mps=float(latest_dec.get("avg_mps",0.0)),
                flags={"night": time.localtime().tm_hour >= 21, "rush": time.localtime().tm_hour == 7},
                extra={"ambulance": False},
            )
            show_led(str(latest_dec.get("action","OFF")))
    except Exception as e:
        print("[MQTT] message parse error:", repr(e))

def build_mqtt():
    if not MQTT_HOST:
        raise RuntimeError("MQTT_HOST not set")
    c = mqtt.Client(client_id=f"pi-io-{SITE_ID}", clean_session=True)
    if MQTT_USER:
        c.username_pw_set(MQTT_USER, MQTT_PASS or None)
    if MQTT_TLS:
        c.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS_CLIENT)
        c.tls_insecure_set(False)
    c.on_connect = on_connect
    c.on_message = on_message
    c.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    c.loop_start()
    return c

# ---------- PIPELINE ----------
def run_pipeline():
    init_db(DB_PATH)

    # choose cameras (0/2/4 by your setup)
    i_ped = int(os.getenv("SC_CAM_PED", "0"))
    i_veh = int(os.getenv("SC_CAM_VEH", "2"))
    i_tl  = int(os.getenv("SC_CAM_TL",  "4"))

    cam_ped = CameraStream(i_ped, FRAME_W, FRAME_H, FPS_TARGET)
    cam_veh = CameraStream(i_veh, FRAME_W, FRAME_H, FPS_TARGET)
    cam_tl  = CameraStream(i_tl,  FRAME_W, FRAME_H, FPS_TARGET)
    print(f"[Pedestrian Cam] {i_ped}")
    print(f"[Vehicle Cam]    {i_veh}")
    print(f"[Traffic Light]  {i_tl}")

    mc = build_mqtt()

    last_status_ts = 0.0
    last_log_ts = 0.0

    while True:
        loop_t = time.time()
        rp, fp = cam_ped.read()
        rv, fv = cam_veh.read()
        rt, ft = cam_tl.read()

        if not rp: fp = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        if not rv: fv = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        if not rt: ft = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

        # draw cloud overlays into the frames we show
        draw_overlays(fp, fv, ft)

        # publish frames to the browser UI (throttled inside publish_frame)
        publish_frame("ped", fp)
        publish_frame("veh", fv)
        publish_frame("tl",  ft)

        # publish JPEG frames to the VM (throttled)
        global _last_pub_ts
        now_pub = time.time()
        if (now_pub - _last_pub_ts) >= (1.0/max(1.0, PUBLISH_HZ)):
            jb = _to_jpeg(fp); jc = _to_jpeg(fv); jt = _to_jpeg(ft)
            if jb is not None: mc.publish(TOPIC_PED, jb, qos=0, retain=False)
            if jc is not None: mc.publish(TOPIC_VEH, jc, qos=0, retain=False)
            if jt is not None: mc.publish(TOPIC_TL,  jt, qos=0, retain=False)
            _last_pub_ts = now_pub

        # mirror decision to status bar at UI cadence
        now = time.time()
        if now - last_status_ts >= STATUS_MIN_PERIOD:
            publish_status_from_loop(
                now_ts=float(latest_dec.get("ts", now)),
                ped_count=int(latest_dec.get("ped_count",0)),
                veh_count=int(latest_dec.get("veh_count",0)),
                tl_color=str(latest_dec.get("tl_color","unknown")),
                nearest_m=float(latest_dec.get("nearest_m",0.0)),
                avg_mps=float(latest_dec.get("avg_mps",0.0)),
                flags={"night": time.localtime(now).tm_hour >= 21, "rush": time.localtime(now).tm_hour == 7},
                extra={"ambulance": False},
            )
            show_led(str(latest_dec.get("action","OFF")))
            last_status_ts = now

        # DB logging for archive/analytics
        if now - last_log_ts >= LOG_EVERY_SEC:
            log_event(
                DB_PATH, now,
                int(latest_dec.get("ped_count",0)),
                int(latest_dec.get("veh_count",0)),
                str(latest_dec.get("tl_color","unknown")),
                float(latest_dec.get("nearest_m",0.0)),
                float(latest_dec.get("avg_mps",0.0)),
                str(latest_dec.get("action","OFF")),
            )
            last_log_ts = now

        if SHOW_WINDOWS:
            cv2.imshow("Ped", fp); cv2.imshow("Veh", fv); cv2.imshow("TL", ft)
            if (cv2.waitKey(1)&0xFF)==27: break

        # pace the loop to camera FPS
        elapsed = time.time() - loop_t
        if elapsed < FRAME_TIME: time.sleep(FRAME_TIME - elapsed)

if __name__ == "__main__":
    threading.Thread(target=run_pipeline, daemon=True).start()
    start_http_server(host="0.0.0.0", port=5000)
