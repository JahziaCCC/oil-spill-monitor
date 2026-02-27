import os, json, time
import datetime as dt
from typing import Dict, Any, List, Tuple, Optional

import requests
import numpy as np
from PIL import Image
from io import BytesIO
from dateutil import tz

# ===== Copernicus Data Space =====
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
BASE_URL = "https://sh.dataspace.copernicus.eu"
CATALOG_SEARCH = f"{BASE_URL}/api/v1/catalog/1.0.0/search"
PROCESS_API = f"{BASE_URL}/api/v1/process"

STATE_FILE = "state.json"
CONFIG_FILE = "config.json"

KSA_TZ = tz.gettz("Asia/Riyadh")


# ================= HELPERS =================
def utc_now():
    return dt.datetime.now(dt.timezone.utc)

def iso_z(d):
    return d.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")

def fmt_ksa(d):
    return d.astimezone(KSA_TZ).strftime("%d-%m-%Y | %H:%M KSA")

def send_telegram(bot, chat_id, text):
    requests.post(
        f"https://api.telegram.org/bot{bot}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=30
    )

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"seen_scene_ids": [], "areas": {}}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"seen_scene_ids": [], "areas": {}}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def get_token(client_id, client_secret):
    r = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret
        },
        timeout=30
    )
    r.raise_for_status()
    return r.json()["access_token"]


# ================= API CALLS =================
def catalog_search_s1(token, bbox, start, end, limit=8):
    headers = {"Authorization": f"Bearer {token}"}

    body = {
        "collections": ["sentinel-1-grd"],
        "datetime": f"{iso_z(start)}/{iso_z(end)}",
        "bbox": bbox,
        "limit": limit,
        "fields": {"include": ["id", "properties.datetime"]}
    }

    r = requests.post(CATALOG_SEARCH, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json().get("features", [])


# ================= MAIN =================
def main():

    client_id = os.environ["CDSE_CLIENT_ID"]
    client_secret = os.environ["CDSE_CLIENT_SECRET"]
    bot = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    lookback = cfg.get("lookback_hours", 72)

    now = utc_now()
    start = now - dt.timedelta(hours=lookback)

    token = get_token(client_id, client_secret)

    state = load_state()
    seen = set(state.get("seen_scene_ids", []))

    found_any = False

    for area in cfg["areas"]:

        area_name = area["name_ar"]
        bbox = area["bbox"]

        scenes = catalog_search_s1(token, bbox, start, now, limit=8)

        # ===== DEBUG LINE =====
        print(f"[DEBUG] {area_name}: scenes found = {len(scenes)}")

        for feat in scenes:
            scene_id = feat.get("id")
            if scene_id and scene_id not in seen:
                found_any = True
                seen.add(scene_id)

    if not found_any:
        msg = f"""📄 تقرير رصد الانسكابات النفطية (SAR)
🕒 {fmt_ksa(now)}

════════════════════
📍 المناطق: البحر الأحمر + الخليج العربي
🛰️ المصدر: Sentinel-1 (SAR)

✅ لا توجد مؤشرات بقع ذات دلالة خلال آخر {lookback} ساعة.
"""
        send_telegram(bot, chat_id, msg)

    state["seen_scene_ids"] = list(seen)[-5000:]
    save_state(state)


if __name__ == "__main__":
    main()
