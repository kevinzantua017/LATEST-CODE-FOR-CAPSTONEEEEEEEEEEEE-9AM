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

# Import flask publishing functions only after the eventlet monkey
# patching is disabled via the environment variable.  We define
# placeholders here so that type checkers do not complain.  The
# actual objects will be replaced in ``_late_import_flask_app``.
def _noop_publish_frame(kind: str, frame_bgr: np.ndarray) -> None:
    pass

def _noop_publish_status_from_loop(**kwargs) -> None:
    pass

def _noop_start_http_server(*a, **k):
    print("[flask_app] not yet loaded")

publish_frame = _noop_publish_frame
publish_status_from_loop = _noop_publish_status_from_loop
start_http_server = _noop_start_http_server

# Make sure eventlet does not override DNS/SSL when running under
# uWSGI or Gunicorn.  See the original code for details.
os.environ.setdefault("EVENTLET_NO_GREENDNS", "YES")

# ---------- Environment ----------
SITE_ID = os.getenv("SC_SITE_ID", "adsmn-01")

# Camera configuration
FRAME_W = int(os.getenv("SC_FRAME_W", "640"))
FRAME_H = int(os.getenv("SC_FRAME_H", "360"))
FPS = int(os.getenv("SC_FPS", "8"))
FRAME_TIME = 1.0 / max(1, FPS)
SKIP_FRAMES = int(os.getenv("SC_SKIP", "1"))
PUBLISH_HZ = float(os.getenv("SC_PUBLISH_HZ", "6"))
JPEG_QUALITY = int(os.getenv("SC_JPEG_QUALITY", "60"))
LANE_Y = int(os.getenv("SC_LANE_Y", "250"))

# MQTT configuration
MQTT_HOST = os.getenv("MQTT_HOST")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_USER = os.getenv("MQTT_USERNAME")
MQTT_PASS = os.getenv("MQTT_PASSWORD")
MQTT_TLS = os.getenv("MQTT_TLS", "1") == "1"

TOPIC_PED = f"crosswalk/{SITE_ID}/frames/ped"
TOPIC_VEH = f"crosswalk/{SITE_ID}/frames/veh"
TOPIC_TL = f"crosswalk/{SITE_ID}/frames/tl"
TOPIC_DEC = f"crosswalk/{SITE_ID}/decision"
TOPIC_VPED = f"crosswalk/{SITE_ID}/viz/ped"
TOPIC_VVEH = f"crosswalk/{SITE_ID}/viz/veh"
TOPIC_VTL = f"crosswalk/{SITE_ID}/viz/tl"

# Database configuration
DB_PATH = os.getenv("SC_DB", "smart_crosswalk.db")
LOG_EVERY_SEC = int(os.getenv("SC_LOG_SEC", "30"))
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

# Disable OpenCL and multi‑threading to avoid CPU thrashing on the Pi
cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# ---------- LED matrix ----------
def _init_led():
    """
    Attempt to initialize the MAX7219 LED matrix.  If the hardware or
    dependencies are unavailable the function will fall back to
    printing messages to the console.  The returned callable takes a
    single string and displays it on the matrix.
    """
    try:
        from luma.core.interface.serial import spi, noop  # type: ignore
        from luma.led_matrix.device import max7219  # type: ignore
        from luma.core.render import canvas  # type: ignore
        from PIL import ImageFont  # type: ignore

        serial = spi(port=0, device=0, gpio=noop())
        device = max7219(
            serial,
            cascaded=int(os.getenv("SC_LED_CASCADE", "4")),
            block_orientation=int(os.getenv("SC_LED_ORIENTATION", "-90")),
            rotate=0,
        )
        font = ImageFont.load_default()

        def show_led(msg: str) -> None:
            with canvas(device) as draw:
                draw.text((1, -2), (msg or "")[:10], fill="white", font=font)

        print("[LED] MAX7219 initialized")
        return show_led
    except Exception as e:
        print("[LED] Fallback console:", repr(e))
        return lambda msg: print("[LED]", msg)

# Initialise LED display once
show_led = _init_led()

# ---------- Shared state ----------
viz_lock = threading.Lock()
viz_frames: dict[str, Tuple[Optional[np.ndarray], float]] = {
    "ped": (None, 0.0),
    "veh": (None, 0.0),
    "tl": (None, 0.0),
}
VIZ_MAX_AGE_S = 1.0

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

# Map scenarios to human‑friendly LED messages.  When a scenario is
# active this message will override the basic "GO"/"STOP"/"OFF" on
# the LED board.  Feel free to adjust the wording to suit your
# hardware display width.
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
    """
    Handle incoming MQTT messages.  Decision packets update the
    ``last_decision`` structure and push status updates to the
    dashboard.  Annotated frames received from the cloud worker are
    stored in ``viz_frames`` and will be used to overlay bounding
    boxes on the live camera streams.
    """
    global last_decision
    try:
        if msg.topic == TOPIC_DEC:
            d = json.loads(msg.payload.decode("utf-8", "ignore"))
            for k in last_decision.keys():
                if k in d:
                    last_decision[k] = d[k]
            # Publish status to the UI.  We always recompute the
            # night/rush flags on the Pi so they reflect local time.
            publish_status_from_loop(
                now_ts=float(last_decision.get("ts", time.time())),
                ped_count=int(last_decision.get("ped_count", 0)),
                veh_count=int(last_decision.get("veh_count", 0)),
                tl_color=str(last_decision.get("tl_color", "unknown")),
                nearest_m=float(last_decision.get("nearest_m", 0.0)),
                avg_mps=float(last_decision.get("avg_mps", 0.0)),
                action=last_decision.get("action", "OFF"),
                scenario=last_decision.get("scenario", "baseline"),
                flags={
                    "night": time.localtime().tm_hour >= 21,
                    "rush": time.localtime().tm_hour == 7,
                },
                extra={"ambulance": False},
            )
        else:
            # Annotated frames from the cloud worker (ped/veh/tl)
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
    """
    Initialise the MQTT client and connect to the broker.  TLS is
    optionally enabled based on the ``SC_MQTT_TLS`` environment
    variable.  Once connected the client subscribes to the decision
    and viz topics.
    """
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
    """
    Normalize camera indices.  Environment variables may specify
    integer indices (0, 1, 2), paths such as "/dev/videoN" or
    strings representing indices.  This helper converts digit
    strings to integers while leaving other strings untouched.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if s.isdigit():
        return int(s)
    m = re.match(r"^/dev/video(\d+)$", s)
    return int(m.group(1)) if m else s

class CameraStream:
    """
    Asynchronously capture frames from a USB camera.  A separate
    thread polls ``cv2.VideoCapture.read`` at the requested FPS and
    stores the most recent frame under a thread lock.  A watchdog
    thread periodically checks whether frames have stopped updating
    and reopens the camera if necessary.
    """

    def __init__(self, index: Union[int, str], width: int, height: int, fps: int) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
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
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        # Prefer V4L2 backend; fall back to any available driver
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index/path {idx}")
        # Reduce buffering on some drivers
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # Request MJPEG to save USB bandwidth
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
        # Set resolution and FPS (driver may clamp)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, min(self.fps, 20))
        # Discard initial frames that may be stale
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
        while not self.stopped:
            ok, frame = self.cap.read() if self.cap is not None else (False, None)
            now = time.time()
            if ok and frame is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                with self.lock:
                    self.ret = True
                    self.frame = frame
                    self.last_update_ts = now
            # pace reading
            sleep_left = max(self.read_interval, target_interval - (now - last_t))
            if sleep_left > 0:
                time.sleep(sleep_left)
            last_t = now

    def _watchdog(self) -> None:
        # Reopen camera if no new frames for N seconds
        while not self.stopped:
            time.sleep(0.5)
            if self.reopen_stale_sec <= 0:
                continue
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
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

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

def log_event(
    path: str,
    ts: float,
    ped_count: int,
    veh_count: int,
    tl_color: str,
    nearest_m: float,
    avg_mps: float,
    action: str,
    scenario: str,
) -> None:
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
def run_pipeline() -> None:
    """
    Main loop that captures frames from the cameras, publishes raw JPEG
    frames over MQTT, receives annotated frames and decisions from the
    cloud worker and logs events to the SQLite database.  It also
    updates the LED matrix based on the current decision and scenario.
    """
    init_db(DB_PATH)
    # Determine camera indices from environment variables.  Default
    # values choose consecutive indices starting at 0.
    i_ped = int(os.getenv("SC_CAM_PED", "0"))
    i_veh = int(os.getenv("SC_CAM_VEH", "1"))
    i_tl = int(os.getenv("SC_CAM_TL", "2"))
    cam_ped = CameraStream(i_ped, FRAME_W, FRAME_H, FPS)
    cam_veh = CameraStream(i_veh, FRAME_W, FRAME_H, FPS)
    cam_tl = CameraStream(i_tl, FRAME_W, FRAME_H, FPS)
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
        ok_tl = bool(rt and ft is not None)
        blank = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        fp_s = fp if ok_ped else blank
        fv_s = fv if ok_veh else blank
        ft_s = ft if ok_tl else blank
        # Prefer annotated frames from the cloud if they are recent
        now_chk = time.time()
        with viz_lock:
            vp, tp = viz_frames.get("ped", (None, 0.0))
            vv, tv = viz_frames.get("veh", (None, 0.0))
            vt, tt = viz_frames.get("tl", (None, 0.0))
        if vp is not None and (now_chk - tp) <= VIZ_MAX_AGE_S:
            fp_s = cv2.resize(vp, (FRAME_W, FRAME_H))
        if vv is not None and (now_chk - tv) <= VIZ_MAX_AGE_S:
            vv_resized = cv2.resize(vv, (FRAME_W, FRAME_H))
            # Draw the lane line on top of the annotated vehicle frame
            cv2.line(vv_resized, (0, LANE_Y), (FRAME_W, LANE_Y), (0, 255, 255), 2)
            fv_s = vv_resized
        if vt is not None and (now_chk - tt) <= VIZ_MAX_AGE_S:
            ft_s = cv2.resize(vt, (FRAME_W, FRAME_H))
        # Publish frames to the web dashboard via SocketIO
        now_ui = time.time()
        if now_ui - _last_pub.setdefault("ui", 0.0) >= (1.0 / max(1.0, PUBLISH_HZ)):
            publish_frame("ped", fp_s)
            publish_frame("veh", fv_s)
            publish_frame("tl", ft_s)
            _last_pub["ui"] = now_ui
        # Publish raw JPEG frames to the cloud worker (rate limited)
        now_mq = time.time()
        if now_mq - _last_pub.setdefault("mqtt", 0.0) >= (1.0 / max(1.0, PUBLISH_HZ)):
            jb = _jpeg(fp_s)
            jc = _jpeg(fv_s)
            jt = _jpeg(ft_s)
            if jb is not None:
                mc.publish(TOPIC_PED, jb, qos=0, retain=False)
            if jc is not None:
                mc.publish(TOPIC_VEH, jc, qos=0, retain=False)
            if jt is not None:
                mc.publish(TOPIC_TL, jt, qos=0, retain=False)
            _last_pub["mqtt"] = now_mq
        # Periodic status update and LED handling
        now = time.time()
        if now - last_status_ts >= STATUS_MIN_PERIOD:
            publish_status_from_loop(
                now_ts=now,
                ped_count=int(last_decision.get("ped_count", 0)),
                veh_count=int(last_decision.get("veh_count", 0)),
                tl_color=str(last_decision.get("tl_color", "unknown")),
                nearest_m=float(last_decision.get("nearest_m", 0.0)),
                avg_mps=float(last_decision.get("avg_mps", 0.0)),
                flags={
                    "night": time.localtime(now).tm_hour >= 21,
                    "rush": time.localtime(now).tm_hour == 7,
                },
                extra={"ambulance": False},
            )
            # Determine the message to display on the LED.  When a
            # special scenario is active use the descriptive message,
            # otherwise fall back to the simple action (GO/STOP/OFF).
            scen = str(last_decision.get("scenario", "baseline"))
            msg = SCENARIO_MESSAGES.get(scen)
            if not msg:
                # Use the basic directive and uppercase it for LED
                msg = str(last_decision.get("action", "OFF"))
            try:
                show_led(msg.upper())
            except Exception as e:
                print("[LED] update error:", repr(e))
            last_status_ts = now
        # Log events to SQLite every LOG_EVERY_SEC seconds
        if now - last_log_ts >= LOG_EVERY_SEC:
            log_event(
                DB_PATH,
                now,
                int(last_decision.get("ped_count", 0)),
                int(last_decision.get("veh_count", 0)),
                str(last_decision.get("tl_color", "unknown")),
                float(last_decision.get("nearest_m", 0.0)),
                float(last_decision.get("avg_mps", 0.0)),
                str(last_decision.get("action", "OFF")),
                str(last_decision.get("scenario", "baseline")),
            )
            last_log_ts = now
        # Frame pacing: sleep to honour target FPS
        elapsed = time.time() - t0
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)

_last_pub: dict[str, float] = {}

def _late_import_flask_app() -> None:
    """
    Import functions from ``flask_app`` only after the MQTT client has
    been configured.  This avoids eventlet applying its monkey
    patching to the standard library DNS and SSL modules before
    Paho initialises TLS.  If ``flask_app`` cannot be imported the
    placeholders defined at the top of the module will remain in
    place and the system will continue to publish only to MQTT.
    """
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
