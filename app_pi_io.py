# app_pi_io.py
# Raspberry Pi = I/O-only (cameras + LED + local website).
# Offloads YOLO + scenario logic to the cloud via MQTT (HiveMQ Cloud).
# Reuses your existing Flask-SocketIO app for streaming/metrics.
#
# ENV (examples):
#   SC_SITE_ID=adsmn-01
#   SC_MQTT_URL=mqtts://<USER>:<URL_ENCODED_PASS>@<host>:8883
#   SC_FRAME_W=640 SC_FRAME_H=360 SC_FPS=15 SC_PUBLISH_HZ=6
#   SC_DB=/home/pi/crosswalk/smart_crosswalk.db
#
# Topics:
#   Pub frames: crosswalk/{SITE_ID}/frames/{ped|veh|tl}   (binary JPEG)
#   Sub decision: crosswalk/{SITE_ID}/decision            (JSON)

from flask_app import publish_frame, publish_status_from_loop, start_http_server

import os
import re
import cv2
import time
import math
import sqlite3
import threading
import json
from queue import Queue, Empty, Full
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Union
from urllib.parse import urlparse, unquote

import numpy as np
import paho.mqtt.client as mqtt

# Keep OpenCV predictable/light on Pi
cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

SHOW_WINDOWS = False
PRINT_DEBUG = True

# ---------- CONFIG ----------
# Robustness / perf toggles
STATUS_MIN_PERIOD = float(os.getenv("SC_STATUS_PERIOD", "0.25"))

# Frames
FRAME_W = int(os.getenv("SC_FRAME_W", "640"))
FRAME_H = int(os.getenv("SC_FRAME_H", "360"))
FPS_TARGET = int(os.getenv("SC_FPS", "15"))
FRAME_TIME = 1.0 / max(1, FPS_TARGET)
SKIP_FRAMES = int(os.getenv("SC_SKIP", "1"))

# MQTT frame publish throttle (JPEG encode + publish only)
PUBLISH_HZ = float(os.getenv("SC_PUBLISH_HZ", "6"))
_last_pub = {"ped": 0.0, "veh": 0.0, "tl": 0.0}

# Vehicle distance calibration (for optional local overlays only)
PIXELS_PER_METER_VEH = float(os.getenv("SC_VEH_PPM", "40.0"))
PEDESTRIAN_LANE_Y    = int(os.getenv("SC_LANE_Y", "250"))

# Traffic light ROI (x,y,w,h) — only used to draw box / optional local heuristic
TRAFFIC_LIGHT_ROI = tuple(map(int, os.getenv("SC_TL_ROI", "100,60,120,160").split(",")))

# DB + status throttling
DB_PATH = os.getenv("SC_DB", "smart_crosswalk.db")
LOG_EVERY_SEC = int(os.getenv("SC_LOG_SEC", "30"))

# Colors
COLOR_GREEN  = (0,255,0)
COLOR_RED    = (0,0,255)
COLOR_YELLOW = (0,255,255)
COLOR_WHITE  = (255,255,255)
COLOR_BLUE   = (255,0,0)

# Site / MQTT
SITE_ID = os.getenv("SC_SITE_ID", "adsmn-01")
MQTT_URL = os.getenv("SC_MQTT_URL", "mqtts://user:pass@host:8883")

# Parse MQTT URL
_u = urlparse(MQTT_URL)
MQTT_HOST = _u.hostname or "localhost"
MQTT_PORT = _u.port or (8883 if _u.scheme == "mqtts" else 1883)
MQTT_TLS  = (_u.scheme == "mqtts")
MQTT_USER = unquote(_u.username) if _u.username else None
MQTT_PASS = unquote(_u.password) if _u.password else None

# MQTT topics
TOPIC_FRAME_PED = f"crosswalk/{SITE_ID}/frames/ped"
TOPIC_FRAME_VEH = f"crosswalk/{SITE_ID}/frames/veh"
TOPIC_FRAME_TL  = f"crosswalk/{SITE_ID}/frames/tl"
TOPIC_DECISION  = f"crosswalk/{SITE_ID}/decision"

# If we don't receive a decision for a while, go safe.
DECISION_STALE_S = float(os.getenv("SC_DECISION_STALE_S", "4.0"))

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
                draw.text((1, -2), msg, fill="white", font=font)
        if PRINT_DEBUG: print("[LED] MAX7219 initialized")
        return show_led
    except Exception as e:
        if PRINT_DEBUG: print("[LED] Fallback console:", repr(e))
        return lambda msg: print("[LED]", msg)

show_led = _init_led()

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

def q_replace_latest(q: Queue, item):
    try:
        while True:
            try: q.get_nowait()
            except Empty: break
        q.put_nowait(item)
    except Full:
        pass

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

# ---------- MQTT ----------
def mqtt_client():
    c = mqtt.Client(client_id=f"pi-{SITE_ID}", clean_session=True)
    if MQTT_USER: c.username_pw_set(MQTT_USER, MQTT_PASS or "")
    if MQTT_TLS:
        try:
            c.tls_set()  # use system CA bundle
        except Exception as e:
            print("[MQTT] TLS set error:", repr(e))
    c.on_connect = _on_connect
    c.on_message = _on_message
    c.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    return c

def _on_connect(client, userdata, flags, rc):
    print("[MQTT] Connected rc=", rc)
    client.subscribe(TOPIC_DECISION, qos=1)

latest_decision = {
    "ts": 0.0,
    "ped_count": 0,
    "veh_count": 0,
    "tl_color": "unknown",
    "nearest_m": 0.0,
    "avg_mps": 0.0,
    "action": "OFF",
    "scenario": "baseline",
}
_dec_lock = threading.Lock()

def _on_message(client, userdata, msg):
    if msg.topic == TOPIC_DECISION:
        try:
            d = json.loads(msg.payload.decode("utf-8"))
            with _dec_lock:
                latest_decision.update({
                    "ts": float(d.get("ts", time.time())),
                    "ped_count": int(d.get("ped_count",0)),
                    "veh_count": int(d.get("veh_count",0)),
                    "tl_color": str(d.get("tl_color","unknown")),
                    "nearest_m": float(d.get("nearest_m",0.0)),
                    "avg_mps": float(d.get("avg_mps",0.0)),
                    "action": str(d.get("action","OFF")),
                    "scenario": str(d.get("scenario","baseline")),
                })
        except Exception as e:
            if PRINT_DEBUG: print("[MQTT decision parse err]", repr(e))

def mqtt_pub_jpeg(client, topic, frame, quality=70):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if ok:
        client.publish(topic, buf.tobytes(), qos=0, retain=False)

# ---------- TL DETECTION (optional local display helper) ----------
def detect_traffic_light_color(frame):
    x,y,w,h = TRAFFIC_LIGHT_ROI
    roi = frame[y:y+h, x:x+w]
    if roi.size==0: return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    def mask(hsv_img, lo, hi):
        return cv2.inRange(hsv_img, np.array(lo,np.uint8), np.array(hi,np.uint8))
    HSV_RED_1 = ((0,120,120),(10,255,255))
    HSV_RED_2 = ((170,120,120),(180,255,255))
    HSV_YELLOW= ((15,120,120),(35,255,255))
    HSV_GREEN = ((40,70,70),(90,255,255))
    mask_red = cv2.bitwise_or(mask(hsv,*HSV_RED_1), mask(hsv,*HSV_RED_2))
    mask_y   = mask(hsv,*HSV_YELLOW)
    mask_g   = mask(hsv,*HSV_GREEN)
    r = int(np.sum(mask_red>0))
    yv= int(np.sum(mask_y>0))
    g = int(np.sum(mask_g>0))
    vals={"red":r,"yellow":yv,"green":g}
    best=max(vals,key=vals.get)
    return best if vals[best]>=50 else "unknown"

# ---------- PIPELINE ----------
def pick_cameras():
    ped_env, veh_env, tl_env = os.getenv("SC_CAM_PED"), os.getenv("SC_CAM_VEH"), os.getenv("SC_CAM_TL")
    if ped_env and veh_env and tl_env:
        ped,veh,tl = _normalize_cam(ped_env), _normalize_cam(veh_env), _normalize_cam(tl_env)
        ok = [safe_camera(ped), safe_camera(veh), safe_camera(tl)]
        if all(o is not None for o in ok): return ped,veh,tl
        raise RuntimeError(f"SC_CAM_* failed: {ok}")

    found=[]
    for i in range(10):
        ok = safe_camera(i)
        if ok is not None: found.append(ok)
        if len(found)>=3: break
    if len(found)<3: raise RuntimeError(f"Found only {len(found)} working cams: {found}")
    return found[0], found[1], found[2]

def safe_camera(idx):
    x = _normalize_cam(idx)
    try:
        cap = cv2.VideoCapture(x, cv2.CAP_V4L2)
        if not cap.isOpened(): cap = cv2.VideoCapture(x, cv2.CAP_ANY)
        if not cap.isOpened(): return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        cap.set(cv2.CAP_PROP_FPS,6)
        ok,_ = cap.read()
        cap.release()
        return x if ok else None
    except Exception:
        return None

def run_pipeline():
    init_db(DB_PATH)

    i_ped, i_veh, i_tl = pick_cameras()
    cam_ped = CameraStream(i_ped, FRAME_W, FRAME_H, FPS_TARGET)
    cam_veh = CameraStream(i_veh, FRAME_W, FRAME_H, FPS_TARGET)
    cam_tl  = CameraStream(i_tl,  480,     270,     max(8, FPS_TARGET//2))
    print(f"[Pedestrian Cam] {i_ped}")
    print(f"[Vehicle Cam]    {i_veh}")
    print(f"[Traffic Light]  {i_tl}")

    # MQTT connect + loop thread
    mc = mqtt_client()
    threading.Thread(target=mc.loop_forever, daemon=True).start()

    frame_idx = 0
    last_log_ts = 0.0
    last_status_ts = 0.0
    last_frame_pub = 0.0

    try:
        while True:
            loop_start = time.time()
            frame_idx += 1

            rp, fp = cam_ped.read()
            rv, fv = cam_veh.read()
            rt, ft = cam_tl.read()

            ok_ped = bool(rp and fp is not None)
            ok_veh = bool(rv and fv is not None)
            ok_tl  = bool(rt and ft is not None)

            blank_640x360 = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            blank_480x270 = np.zeros((270, 480, 3), dtype=np.uint8)
            fp_s = fp if ok_ped else blank_640x360
            fv_s = fv if ok_veh else blank_640x360
            ft_s = ft if ok_tl  else blank_480x270

            # Publish frames to MQTT at PUBLISH_HZ
            now = time.time()
            if now - last_frame_pub >= (1.0 / max(1.0, PUBLISH_HZ)):
                mqtt_pub_jpeg(mc, TOPIC_FRAME_PED, fp_s)
                mqtt_pub_jpeg(mc, TOPIC_FRAME_VEH, fv_s)
                mqtt_pub_jpeg(mc, TOPIC_FRAME_TL,  ft_s)
                last_frame_pub = now

            # Draw local TL ROI box + (optional) heuristic label for operator context
            x,y,w,h = TRAFFIC_LIGHT_ROI
            cv2.rectangle(ft_s,(x,y),(x+w,y+h),COLOR_WHITE,2)
            tl_local = detect_traffic_light_color(ft_s)
            cv2.putText(ft_s,f"TL? {tl_local.upper()}",(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,COLOR_WHITE,2)

            # Show frames on local web UI (throttled)
            publish_frame_throttled("ped", fp_s)
            publish_frame_throttled("veh", fv_s)
            publish_frame_throttled("tl",  ft_s)

            # Consume latest decision (thread-safe)
            with _dec_lock:
                d = dict(latest_decision)

            # If stale, safe fallback
            stale = (now - float(d.get("ts",0.0))) > DECISION_STALE_S
            action   = "OFF" if stale else str(d.get("action","OFF"))
            tl_color = "unknown" if stale else str(d.get("tl_color","unknown"))
            ped_cnt  = 0 if stale else int(d.get("ped_count",0))
            veh_cnt  = 0 if stale else int(d.get("veh_count",0))
            nearest_m= float(d.get("nearest_m",0.0)) if not stale else 0.0
            avg_mps  = float(d.get("avg_mps",0.0)) if not stale else 0.0
            scenario = "baseline" if stale else str(d.get("scenario","baseline"))

            # LED
            show_led(action)

            # Overlay quick stats (vehicle view)
            cv2.putText(fv_s,f"Vehicles: {veh_cnt}",(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,COLOR_WHITE,2)
            cv2.line(fv_s,(0,PEDESTRIAN_LANE_Y),(FRAME_W,PEDESTRIAN_LANE_Y),COLOR_YELLOW,2)
            cv2.putText(fv_s,f"Nearest dist: {nearest_m:.1f} m",(8,44),cv2.FONT_HERSHEY_SIMPLEX,0.6,COLOR_WHITE,2)
            cv2.putText(fv_s,f"Avg speed: {avg_mps:.1f} m/s",(8,68),cv2.FONT_HERSHEY_SIMPLEX,0.6,COLOR_WHITE,2)

            # Status → dashboard (throttled)
            if now - last_status_ts >= STATUS_MIN_PERIOD:
                flags = {"night": time.localtime(now).tm_hour >= 21, "rush": time.localtime(now).tm_hour == 7}
                publish_status_from_loop(
                    now_ts=now if stale else float(d.get("ts",now)),
                    ped_count=ped_cnt,
                    veh_count=veh_cnt,
                    tl_color=tl_color,
                    nearest_m=nearest_m,
                    avg_mps=avg_mps,
                    flags=flags,
                    extra={"ambulance": False},
                )
                last_status_ts = now

            # DB logging (time-based)
            if now - last_log_ts >= LOG_EVERY_SEC:
                log_event(DB_PATH, now if stale else float(d.get("ts",now)), int(ped_cnt), int(veh_cnt),
                          tl_color, float(nearest_m), float(avg_mps), action)
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

# ---------- MAIN ----------
if __name__ == "__main__":
    threading.Thread(target=run_pipeline, daemon=True).start()
    start_http_server(host="0.0.0.0", port=5000)
