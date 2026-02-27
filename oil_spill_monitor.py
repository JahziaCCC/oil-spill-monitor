import os, json, time
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple

import requests
import numpy as np
from PIL import Image
from io import BytesIO
from dateutil import tz

# ===== Copernicus Data Space (Sentinel Hub) =====
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
BASE_URL = "https://sh.dataspace.copernicus.eu"
CATALOG_SEARCH = f"{BASE_URL}/api/v1/catalog/1.0.0/search"
PROCESS_API = f"{BASE_URL}/api/v1/process"

CONFIG_FILE = "config.json"
STATE_FILE = "state.json"

KSA_TZ = tz.gettz("Asia/Riyadh")


# ---------------- Helpers ----------------
def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def iso_z(d: dt.datetime) -> str:
    return d.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")

def fmt_ksa(d_utc: dt.datetime) -> str:
    return d_utc.astimezone(KSA_TZ).strftime("%d-%m-%Y | %H:%M KSA")

def send_telegram(bot: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{bot}/sendMessage"
    r = requests.post(url, json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True}, timeout=30)
    r.raise_for_status()

def risk_badge(score: int) -> str:
    if score >= 85: return "🔴 حرج"
    if score >= 70: return "🟠 مرتفع"
    if score >= 55: return "🟡 متوسط"
    return "🟢 منخفض"

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {"areas": {}}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"areas": {}}

def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def get_token(client_id: str, client_secret: str) -> str:
    r = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


# ---------------- Sentinel Hub: Catalog ----------------
def catalog_search_s1(token: str, bbox: List[float], start: dt.datetime, end: dt.datetime, limit: int = 20) -> List[Dict[str, Any]]:
    """
    NOTE: No 'sortby' here because CDSE may return 400 for it in some setups.
    """
    headers = {"Authorization": f"Bearer {token}"}
    body = {
        "collections": ["sentinel-1-grd"],
        "datetime": f"{iso_z(start)}/{iso_z(end)}",
        "bbox": bbox,
        "limit": limit,
        "fields": {"include": ["id", "properties.datetime"]},
    }
    r = requests.post(CATALOG_SEARCH, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json().get("features", [])


# ---------------- Sentinel Hub: Process API ----------------
def process_vv_db_thumbnail(token: str, bbox: List[float], time_from: dt.datetime, time_to: dt.datetime, w: int = 256, h: int = 256) -> np.ndarray:
    """
    Fetch VV backscatter (dB) thumbnail using Process API. Returns 2D float with NaNs for nodata.
    """
    headers = {"Authorization": f"Bearer {token}"}

    evalscript = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["VV", "dataMask"] }],
    output: { bands: 2, sampleType: "FLOAT32" }
  };
}
function toDB(x) { return 10.0 * Math.log(x) / Math.LN10; }
function evaluatePixel(s) {
  if (s.dataMask === 0) return [NaN, 0];
  return [toDB(s.VV), 1];
}
"""
    body = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {"from": iso_z(time_from), "to": iso_z(time_to)},
                    "acquisitionMode": "IW",
                    "polarization": "VV"
                },
                "processing": {
                    "speckleFilter": {"type": "LEE", "windowSizeX": 3, "windowSizeY": 3}
                }
            }]
        },
        "output": {
            "width": w,
            "height": h,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
        },
        "evalscript": evalscript
    }

    r = requests.post(PROCESS_API, headers=headers, json=body, timeout=120)
    r.raise_for_status()

    img = Image.open(BytesIO(r.content))
    arr = np.array(img)
    if arr.ndim < 3 or arr.shape[-1] < 2:
        raise RuntimeError("Unexpected TIFF returned from Process API")

    db = arr[..., 0].astype(np.float32)
    mask = arr[..., 1].astype(np.float32)
    db[mask < 0.5] = np.nan
    return db


# ---------------- Detection + Geolocation ----------------
def centroid_latlon(bbox: List[float], mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Convert centroid of True pixels into lat/lon using bbox mapping.
    bbox = [minLon, minLat, maxLon, maxLat]
    mask shape = (H, W)
    """
    ys, xs = np.where(mask)
    if xs.size < 20:
        return None

    H, W = mask.shape
    x_mean = float(xs.mean())
    y_mean = float(ys.mean())

    minLon, minLat, maxLon, maxLat = bbox
    lon = minLon + (x_mean / max(W - 1, 1)) * (maxLon - minLon)
    lat = maxLat - (y_mean / max(H - 1, 1)) * (maxLat - minLat)
    return (lat, lon)

def ops_card(area_name: str, ksa_time: str, scene_utc: str, lat: float, lon: float,
             dark_ratio: float, thr_db: float, score: int, note: str) -> str:
    return (
        "🚨 بطاقة عمليات بيئية – رصد انسكابات (SAR)\n"
        "════════════════════\n"
        "🛢️ الحدث: أفضل مرشح بقعة داكنة\n"
        f"📍 المنطقة: {area_name}\n"
        f"🌍 الإحداثيات: {lat:.4f}N , {lon:.4f}E\n"
        f"🕒 وقت التحديث: {ksa_time}\n\n"
        f"📊 مستوى الخطر: {risk_badge(score)} ({score}/100)\n\n"
        "🛰️ تحليل Sentinel-1 (VV)\n"
        f"• مؤشر البقعة الداكنة: {dark_ratio:.2%}\n"
        f"• العتبة (dB): أقل من {thr_db}\n"
        f"• وقت المشهد (UTC): {scene_utc}\n\n"
        f"🧾 ملاحظة: {note}\n"
        "════════════════════\n"
        "🎯 الإجراء:\n"
        "• متابعة التمريرة القادمة لنفس المنطقة.\n"
        "• إذا كانت قرب الساحل/محميات/منشآت: تصعيد.\n"
    )

def status_report(ksa_time: str, lookback_hours: int) -> str:
    return (
        "📄 تقرير رصد الانسكابات النفطية (SAR)\n"
        f"🕒 {ksa_time}\n"
        "════════════════════\n"
        "📍 المناطق: البحر الأحمر + الخليج العربي\n"
        "🛰️ المصدر: Sentinel-1 (SAR)\n\n"
        f"✅ لا توجد بقع تجاوزت عتبة التنبيه خلال آخر {lookback_hours} ساعة.\n"
    )


def main():
    # Secrets
    client_id = os.environ["CDSE_CLIENT_ID"]
    client_secret = os.environ["CDSE_CLIENT_SECRET"]
    bot = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    # Config
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    lookback = int(cfg.get("lookback_hours", 168))
    min_dark_ratio = float(cfg.get("min_dark_ratio", 0.001))
    thr_db = float(cfg.get("dark_db_threshold", -21.0))
    cooldown_h = int(cfg.get("cooldown_hours_per_area", 6))
    max_alerts = int(cfg.get("max_alerts_per_run", 2))

    now = utc_now()
    start = now - dt.timedelta(hours=lookback)
    ksa_time = fmt_ksa(now)

    token = get_token(client_id, client_secret)
    state = load_state()
    area_state = state.get("areas", {})

    best_candidates: List[Dict[str, Any]] = []
    sent_any = 0

    for area in cfg["areas"]:
        area_id = area["id"]
        area_name = area["name_ar"]
        bbox = area["bbox"]

        # cooldown check
        last_alert = area_state.get(area_id, {}).get("last_alert_utc")
        in_cooldown = False
        if last_alert:
            try:
                last_dt = dt.datetime.fromisoformat(last_alert.replace("Z", "+00:00"))
                in_cooldown = (now - last_dt) < dt.timedelta(hours=cooldown_h)
            except Exception:
                pass

        scenes = catalog_search_s1(token, bbox, start, now, limit=20)
        if not scenes:
            continue

        best = None

        # evaluate a few scenes (newest-ish)
        for feat in scenes[:8]:
            scene_time = (feat.get("properties", {}) or {}).get("datetime")
            if not scene_time:
                continue

            t = dt.datetime.fromisoformat(scene_time.replace("Z", "+00:00"))
            t_from = t - dt.timedelta(minutes=5)
            t_to = t + dt.timedelta(minutes=5)

            try:
                db = process_vv_db_thumbnail(token, bbox, t_from, t_to)
                valid = np.isfinite(db)
                if valid.sum() < 1000:
                    continue

                dark = (db < thr_db) & valid
                dark_ratio = float(dark.sum()) / float(valid.sum())

                c = centroid_latlon(bbox, dark)
                if c is None:
                    continue
                lat, lon = c

                score = int(min(95, max(10, (dark_ratio / max(min_dark_ratio, 1e-6)) * 60 + 20)))

                cand = {
                    "area_id": area_id,
                    "area_name": area_name,
                    "scene_utc": scene_time.replace("Z", ""),
                    "lat": lat,
                    "lon": lon,
                    "dark_ratio": dark_ratio,
                    "score": score,
                    "in_cooldown": in_cooldown
                }

                if (best is None) or (cand["dark_ratio"] > best["dark_ratio"]):
                    best = cand

            except Exception:
                continue

        if best:
            best_candidates.append(best)

    best_candidates.sort(key=lambda x: x["dark_ratio"], reverse=True)

    # Send "real alerts" (>= min_dark_ratio)
    for cand in best_candidates[:max_alerts]:
        if cand["in_cooldown"] and cand["score"] < 85:
            continue

        if cand["dark_ratio"] >= min_dark_ratio:
            msg = ops_card(
                cand["area_name"], ksa_time, cand["scene_utc"],
                cand["lat"], cand["lon"],
                cand["dark_ratio"], thr_db, cand["score"],
                "تم إرسال تنبيه لأن المؤشر تجاوز عتبة المرشح."
            )
            send_telegram(bot, chat_id, msg)
            sent_any += 1
            time.sleep(1.2)

            area_state.setdefault(cand["area_id"], {})
            area_state[cand["area_id"]]["last_alert_utc"] = iso_z(now)

    # If no real alerts, send 1 best candidate (Analyst)
    if sent_any == 0 and best_candidates:
        cand = best_candidates[0]
        msg = ops_card(
            cand["area_name"], ksa_time, cand["scene_utc"],
            cand["lat"], cand["lon"],
            cand["dark_ratio"], thr_db, cand["score"],
            "مرشح تحليلي (أعلى بقعة داكنة). قد تكون Look-alike وليست زيت."
        )
        send_telegram(bot, chat_id, msg)

    if not best_candidates:
        send_telegram(bot, chat_id, status_report(ksa_time, lookback))

    state["areas"] = area_state
    save_state(state)


if __name__ == "__main__":
    main()
