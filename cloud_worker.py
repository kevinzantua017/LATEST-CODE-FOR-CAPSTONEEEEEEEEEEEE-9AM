# cloud_worker.py  (Cloud: subscribes to Pi frames via MQTT, runs YOLO, publishes decisions)
# Topics (must match the Pi side):
#   Sub frames:   crosswalk/{SITE_ID}/frames/{ped|veh|tl}   (binary JPEG)
#   Pub decision: crosswalk/{SITE_ID}/decision              (JSON)

import os, time, json, threading
import numpy as np, cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from urllib.parse import urlparse, unquote

# -------- ENV --------
SITE_ID = os.getenv("SC_SITE_ID", "adsmn-01")

# Example (URL-encode special chars in password!):
# mqtts://USERNAME:P%40ssw%3Ard@d399b4...hivemq.cloud:8883
BROKER_URL = os.getenv("SC_MQTT_URL", "mqtts://user:pass@host:8883")

MODEL   = os.getenv("SC_YOLO_MODEL", "yolov8n.pt")
CONF    = float(os.getenv("SC_YOLO_CONF", "0.20"))
IMG_SZ  = int(os.getenv("SC_YOLO_IMG", "416"))

# Crosswalk geometry (must match Pi’s UI overlays)
TL_ROI  = tuple(map(int, os.getenv("SC_TL_ROI", "100,60,120,160").split(",")))
PPM     = float(os.getenv("SC_VEH_PPM", "40.0"))
LANE_Y  = int(os.getenv("SC_LANE_Y", "250"))
CLOSE_M = float(os.getenv("SC_VEH_CLOSE_M", "6.0"))

# -------- MQTT parsing --------
u = urlparse(BROKER_URL)
HOST  = u.hostname or "localhost"
PORT  = u.port or (8883 if u.scheme == "mqtts" else 1883)
TLS   = (u.scheme == "mqtts")
USER  = unquote(u.username) if u.username else None
PASS  = unquote(u.password) if u.password else None

TOPIC_PED = f"crosswalk/{SITE_ID}/frames/ped"
TOPIC_VEH = f"crosswalk/{SITE_ID}/frames/veh"
TOPIC_TL  = f"crosswalk/{SITE_ID}/frames/tl"
TOPIC_DEC = f"crosswalk/{SITE_ID}/decision"

# -------- frame buffers --------
buf = {"ped": None, "veh": None, "tl": None}
buf_lock = threading.Lock()

def _decode_jpeg(b):
    arr = np.frombuffer(b, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def on_msg(client, userdata, msg):
    t = msg.topic
    key = "ped" if t.endswith("/ped") else "veh" if t.endswith("/veh") else "tl" if t.endswith("/tl") else None
    if not key:
        return
    im = _decode_jpeg(msg.payload)
    if im is not None:
        with buf_lock:
            buf[key] = im

def mqtt_client():
    c = mqtt.Client(client_id=f"gpu-{SITE_ID}", clean_session=True)
    if USER:
        c.username_pw_set(USER, PASS or "")
    if TLS:
        c.tls_set()  # system CA bundle
    c.on_message = on_msg
    c.connect(HOST, PORT, keepalive=30)
    for topic in (TOPIC_PED, TOPIC_VEH, TOPIC_TL):
        c.subscribe(topic, qos=0)
    threading.Thread(target=c.loop_forever, daemon=True).start()
    return c

# -------- helpers --------
def detect_traffic_light_color(frame):
    x,y,w,h = TL_ROI
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    def m(lo,hi):
        return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

    red1=((0,120,120),(10,255,255))
    red2=((170,120,120),(180,255,255))
    yellow=((15,120,120),(35,255,255))
    green=((40,70,70),(90,255,255))

    r  = int(np.sum(cv2.bitwise_or(m(*red1), m(*red2)) > 0))
    yv = int(np.sum(m(*yellow) > 0))
    g  = int(np.sum(m(*green)  > 0))

    vals = {"red": r, "yellow": yv, "green": g}
    best = max(vals, key=vals.get)
    return best if vals[best] >= 50 else "unknown"

# -------- main --------
def main():
    # load model once
    model = YOLO(MODEL)
    try:
        model.fuse()
    except Exception:
        pass

    mc = mqtt_client()

    COCO_PED = [0]                 # person
    COCO_VEH = [1, 2, 3, 5, 7]     # bicycle, car, motorcycle, bus, truck

    # loop
    loop_dt = 0.05  # ~20 Hz
    while True:
        time.sleep(loop_dt)
        with buf_lock:
            fp = buf["ped"]
            fv = buf["veh"]
            ft = buf["tl"]

        ped_count = 0
        veh_count = 0
        nearest_m = 0.0
        avg_mps   = 0.0  # simple baseline (no velocity tracker here for now)
        tl_color  = "unknown"

        if ft is not None:
            tl_color = detect_traffic_light_color(ft)

        if fp is not None:
            r = model.predict(fp, conf=CONF, imgsz=IMG_SZ, verbose=False, max_det=50, classes=COCO_PED)
            # count persons
            ped_count = sum(1 for b in r[0].boxes if int(b.cls[0]) == 0)

        if fv is not None:
            r = model.predict(fv, conf=CONF, imgsz=IMG_SZ, verbose=False, max_det=50, classes=COCO_VEH)
            boxes = []
            for b in r[0].boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                boxes.append((x1,y1,x2,y2))
            veh_count = len(boxes)

            # estimate closest vehicle distance to the crossing line
            for (_x1,_y1,_x2,y2) in boxes:
                dy_px = max(0, LANE_Y - y2)  # distance from box bottom to lane Y
                dist_m = abs(dy_px) / PPM
                nearest_m = dist_m if nearest_m == 0.0 else min(nearest_m, dist_m)

        # --- scenario logic (mirrors your baseline logic) ---
        action  = "OFF"
        scenario= "baseline"

        if tl_color == "red":
            action = "STOP"
        elif tl_color == "green":
            # stop if pedestrians + vehicles close; otherwise allow GO for pedestrians
            if ped_count > 0 and (veh_count > 0 and (nearest_m < CLOSE_M if nearest_m > 0 else True)):
                action = "STOP"
            elif ped_count > 0:
                action = "GO"
        elif tl_color == "yellow":
            action = "STOP" if ped_count > 0 else "OFF"
        else:  # unknown light
            if ped_count > 0 and veh_count > 0:
                action = "STOP"
            elif ped_count > 0:
                action = "GO"

        # publish decision JSON
        mc.publish(TOPIC_DEC, json.dumps({
            "ts": time.time(),
            "ped_count": int(ped_count),
            "veh_count": int(veh_count),
            "tl_color": str(tl_color),
            "nearest_m": float(nearest_m),
            "avg_mps": float(avg_mps),
            "action": str(action),
            "scenario": str(scenario),
        }), qos=1, retain=False)

if __name__ == "__main__":
    main()
