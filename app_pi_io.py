from flask_app import publish_frame, publish_status_from_loop, start_http_server

import os, cv2, time, sqlite3, threading, json
import numpy as np
from urllib.parse import urlparse
import paho.mqtt.client as mqtt

# Keep OpenCV light on Pi
cv2.setNumThreads(1)
try: cv2.ocl.setUseOpenCL(False)
except: pass

PRINT_DEBUG = True

# ---------- ENV / CONFIG ----------
SITE_ID = os.getenv("SC_SITE_ID", "adsmn-01")
MQTT_URL = os.getenv("SC_MQTT_URL", "mqtts://user:pass@host:8883")

FRAME_W = int(os.getenv("SC_FRAME_W", "640"))
FRAME_H = int(os.getenv("SC_FRAME_H", "360"))
FPS_TARGET = int(os.getenv("SC_FPS", "15"))
FRAME_TIME = 1.0 / max(1, FPS_TARGET)

PUBLISH_HZ = float(os.getenv("SC_PUBLISH_HZ", "6"))     # MQTT frame publish rate
STATUS_MIN_PERIOD = float(os.getenv("SC_STATUS_PERIOD", "0.25"))
DB_PATH = os.getenv("SC_DB", "smart_crosswalk.db")

TRAFFIC_LIGHT_ROI = tuple(map(int, os.getenv("SC_TL_ROI", "100,60,120,160").split(",")))

# MQTT parse
u = urlparse(MQTT_URL)
MQTT_HOST, MQTT_PORT = u.hostname or "localhost", u.port or (8883 if u.scheme=="mqtts" else 1883)
MQTT_TLS = (u.scheme == "mqtts")
MQTT_USER = u.username
MQTT_PASS = u.password

COLOR_WHITE=(255,255,255)

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
        print("[LED] MAX7219 initialized"); return show_led
    except Exception as e:
        print("[LED] Fallback console:", repr(e))
        return lambda msg: print("[LED]", msg)
show_led = _init_led()

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

# ---------- Camera helpers ----------
def _normalize_cam(value):
    if value is None: return None
    if isinstance(value,int): return value
    s = str(value).strip()
    if s.isdigit(): return int(s)
    if s.startswith("/dev/"): return s
    import re
    m = re.match(r"^/dev/video(\d+)$", s)
    return int(m.group(1)) if m else s

class CameraStream:
    def __init__(self, index, width, height, fps):
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.lock = threading.Lock(); self.frame=None; self.ret=False; self.stopped=False
        self._open_camera()
        threading.Thread(target=self._update, daemon=True).start()
    def _open_camera(self):
        idx = _normalize_cam(self.index)
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not self.cap.isOpened(): self.cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if not self.cap.isOpened(): raise RuntimeError(f"Could not open camera {idx}")
        try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        except: pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, min(self.fps,20))
        def set_fourcc(code):
            f = cv2.VideoWriter_fourcc(*code); self.cap.set(cv2.CAP_PROP_FOURCC, f)
            return int(self.cap.get(cv2.CAP_PROP_FOURCC)) == f
        if not set_fourcc('MJPG'): set_fourcc('YUYV') or set_fourcc('YUY2')
        self.cap.read()
    def _update(self):
        frame_interval = 1.0/max(1,self.fps); last_t=0.0
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if frame.shape[1]!=self.width or frame.shape[0]!=self.height:
                    frame = cv2.resize(frame,(self.width,self.height), interpolation=cv2.INTER_AREA)
                with self.lock: self.ret, self.frame = True, frame
            now=time.time(); sl=frame_interval-(now-last_t)
            if sl>0: time.sleep(sl); last_t=now
    def read(self):
        with self.lock: return self.ret, None if self.frame is None else self.frame.copy()
    def stop(self):
        self.stopped=True
        try: self.cap.release()
        except: pass

# ---------- Local TL helper (for UI label only) ----------
def detect_traffic_light_color(frame):
    x,y,w,h = TRAFFIC_LIGHT_ROI
    if y+h>frame.shape[0] or x+w>frame.shape[1]: return "unknown"
    roi=frame[y:y+h,x:x+w]
    hsv=cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    def m(lo,hi): return cv2.inRange(hsv, np.array(lo,np.uint8), np.array(hi,np.uint8))
    red1=((0,120,120),(10,255,255)); red2=((170,120,120),(180,255,255))
    yellow=((15,120,120),(35,255,255)); green=((40,70,70),(90,255,255))
    r=int(np.sum(cv2.bitwise_or(m(*red1),m(*red2))>0)); yv=int(np.sum(m(*yellow)>0)); g=int(np.sum(m(*green)>0))
    vals={"red":r,"yellow":yv,"green":g}; best=max(vals,key=vals.get)
    return best if vals[best]>=50 else "unknown"

# ---------- MQTT ----------
def mqtt_client():
    c = mqtt.Client(client_id=f"pi-{SITE_ID}", clean_session=True)
    if MQTT_USER: c.username_pw_set(MQTT_USER, MQTT_PASS or "")
    if MQTT_TLS: c.tls_set()  # system CA
    c.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    return c

latest_decision = {
    "ts":0.0,"ped_count":0,"veh_count":0,"tl_color":"unknown",
    "nearest_m":0.0,"avg_mps":0.0,"action":"OFF","scenario":"baseline"
}
_dec_lock = threading.Lock()

def _on_decision(client, userdata, msg):
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
        if PRINT_DEBUG: print("[MQTT decision err]", repr(e))

def mqtt_pub_jpeg(mc, topic, frame, quality=70):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if ok: mc.publish(topic, buf.tobytes(), qos=0, retain=False)

# ---------- Main pipeline ----------
def pick_cameras():
    ped_env, veh_env, tl_env = os.getenv("SC_CAM_PED"), os.getenv("SC_CAM_VEH"), os.getenv("SC_CAM_TL")
    if ped_env and veh_env and tl_env:
        return _normalize_cam(ped_env), _normalize_cam(veh_env), _normalize_cam(tl_env)
    found=[]
    for i in range(10):
        cap=cv2.VideoCapture(i, cv2.CAP_V4L2)
        if not cap.isOpened(): cap=cv2.VideoCapture(i, cv2.CAP_ANY)
        if not cap.isOpened(): continue
        cap.set(cv2.CAP_PROP_FPS,6); ok,_=cap.read(); cap.release()
        if ok: found.append(i)
        if len(found)>=3: break
    if len(found)<3: raise RuntimeError(f"Found only {found}")
    return found[0],found[1],found[2]

def run_pipeline():
    init_db(DB_PATH)

    i_ped,i_veh,i_tl = pick_cameras()
    cam_ped=CameraStream(i_ped, FRAME_W, FRAME_H, FPS_TARGET)
    cam_veh=CameraStream(i_veh, FRAME_W, FRAME_H, FPS_TARGET)
    cam_tl =CameraStream(i_tl, 480, 270, max(8,FPS_TARGET//2))
    print(f"[Cams] Ped={i_ped} Veh={i_veh} TL={i_tl}")

    mc = mqtt_client()
    mc.on_message = _on_decision
    mc.subscribe(f"crosswalk/{SITE_ID}/decision", qos=1)
    threading.Thread(target=mc.loop_forever, daemon=True).start()

    last_status=0.0; last_log=0.0
    last_pub = 0.0
    min_pub_interval = 1.0 / max(1.0, PUBLISH_HZ)

    try:
        while True:
            loop_t = time.time()

            rp,fp = cam_ped.read()
            rv,fv = cam_veh.read()
            rt,ft = cam_tl.read()

            if fp is None: fp=np.zeros((FRAME_H,FRAME_W,3),np.uint8)
            if fv is None: fv=np.zeros((FRAME_H,FRAME_W,3),np.uint8)
            if ft is None: ft=np.zeros((270,480,3),np.uint8)

            # Publish frames to MQTT (throttled)
            if loop_t - last_pub >= min_pub_interval:
                mqtt_pub_jpeg(mc, f"crosswalk/{SITE_ID}/frames/ped", fp)
                mqtt_pub_jpeg(mc, f"crosswalk/{SITE_ID}/frames/veh", fv)
                mqtt_pub_jpeg(mc, f"crosswalk/{SITE_ID}/frames/tl",  ft)
                last_pub = loop_t

            # UX overlay for TL ROI
            x,y,w,h = TRAFFIC_LIGHT_ROI
            cv2.rectangle(ft,(x,y),(x+w,y+h),COLOR_WHITE,2)
            loc_tl = detect_traffic_light_color(ft)
            cv2.putText(ft,f"TL? {loc_tl.upper()}",(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,COLOR_WHITE,2)

            # Show frames on local web UI
            publish_frame("ped", fp)
            publish_frame("veh", fv)
            publish_frame("tl",  ft)

            # LED + status from latest cloud decision
            with _dec_lock: d=dict(latest_decision)
            show_led(d.get("action","OFF"))

            now=time.time()
            if now - last_status >= STATUS_MIN_PERIOD:
                publish_status_from_loop(
                    now_ts=d.get("ts",now),
                    ped_count=d.get("ped_count",0),
                    veh_count=d.get("veh_count",0),
                    tl_color=d.get("tl_color","unknown"),
                    nearest_m=d.get("nearest_m",0.0),
                    avg_mps=d.get("avg_mps",0.0),
                    flags={"night": time.localtime(now).tm_hour >= 21,
                           "rush": time.localtime(now).tm_hour == 7},
                    extra={"ambulance": False},
                )
                last_status = now

            if now - last_log >= 30:
                log_event(DB_PATH, float(d.get("ts",now)), int(d.get("ped_count",0)),
                          int(d.get("veh_count",0)), str(d.get("tl_color","unknown")),
                          float(d.get("nearest_m",0.0)), float(d.get("avg_mps",0.0)),
                          str(d.get("action","OFF")))
                last_log = now

            elapsed = time.time() - loop_t
            if elapsed < FRAME_TIME: time.sleep(FRAME_TIME - elapsed)

    finally:
        for c in (cam_ped,cam_veh,cam_tl):
            try: c.stop()
            except: pass

if __name__=="__main__":
    threading.Thread(target=run_pipeline, daemon=True).start()
    start_http_server(host="0.0.0.0", port=5000)
