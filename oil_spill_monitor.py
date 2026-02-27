import os, json, time
import datetime as dt
from typing import Dict, Any, List, Tuple, Optional

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

STATE_FILE = "state.json"
CONFIG_FILE = "config.json"

KSA_TZ = tz.gettz("Asia/Riyadh")


# ----------------- Helpers -----------------
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

def trend_arrow(prev: Optional[float], cur: float) -> str:
    if prev is None:
        return "→ (لا توجد مقارنة)"
    if cur >= prev * 1.10: return "↑ (يزداد)"
    if cur <= prev * 0.90: return "↓ (ينخفض)"
    return "→ (ثابت)"

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {"seen_scene_ids": [], "areas": {}}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"seen_scene_ids": [], "areas": {}}

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


# ----------------- Sentinel Hub calls -----------------
def catalog_search_s1(token: str, bbox: List[float], start: dt.datetime, end: dt.datetime, limit: int = 8) -> List[Dict[str, Any]]:
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

def process_vv_db_thumbnail(token: str, bbox: List[float], time_from: dt.datetime, time_to: dt.datetime, w: int = 256, h: int = 256) -> np.ndarray:
    """
    Returns VV backscatter in dB as 2D float array with NaNs for nodata.
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
                    "polarization": "VV",
                },
                "processing": {
                    "speckleFilter": {"type": "LEE", "windowSizeX": 3, "windowSizeY": 3}
                }
            }],
        },
        "output": {
            "width": w,
            "height": h,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }

    r = requests.post(PROCESS_API, headers=headers, json=body, timeout=120)
    r.raise_for_status()

    img = Image.open(BytesIO(r.content))
    arr = np.array(img)
    if arr.ndim < 3 or arr.shape[-1] < 2:
        raise RuntimeError("Unexpected TIFF format returned from Process API.")

    db = arr[..., 0].astype(np.float32)
    mask = arr[..., 1].astype(np.float32)
    db[mask < 0.5] = np.nan
    return db


# ----------------- Detection + Geolocation -----------------
def dark_centroid_latlon(bbox: List[float], dark_mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute centroid of dark pixels and convert to lat/lon using bbox.
    bbox = [minLon, minLat, maxLon, maxLat]
    dark_mask shape: (H,W) booleans
    """
    ys, xs = np.where(dark_mask)
    if len(xs) < 10:
        return None

    H, W = dark_mask.shape
    x_mean = float(xs.mean())
    y_mean = float(ys.mean())

    minLon, minLat, maxLon, maxLat = bbox
    lon = minLon + (x_mean / max(W - 1, 1)) * (maxLon - minLon)

    # Note: image y increases downward; lat decreases downward
    lat = maxLat - (y_mean / max(H - 1, 1)) * (maxLat - minLat)

    return (lat, lon)

def ops_card_ar(area_name: str, ksa_time: str, scene_utc: str,
                lat: float, lon: float,
                dark_ratio: float, score: int, trend: str, thr_db: float) -> str:
    return (
        "🚨 بطاقة عمليات بيئية – رصد بحري\n"
        "════════════════════\n"
        "🛢️ الحدث: بقعة محتملة (SAR)\n"
        f"📍 المنطقة: {area_name}\n"
        f"🌍 الإحداثيات: {lat:.4f}N , {lon:.4f}E\n"
        f"🕒 آخر تحديث: {ksa_time}\n\n"
        f"📊 مستوى الخطر: {risk_badge(score)} ({score}/100)\n"
        f"📈 الاتجاه: {trend}\n\n"
        "🛰️ تحليل Sentinel-1 (SAR)\n"
        f"• مؤشر البقعة الداكنة: {dark_ratio:.2%}\n"
        f"• عتبة الاكتشاف (dB): أقل من {thr_db}\n"
        f"• وقت المشهد (UTC): {scene_utc}\n\n"
        "════════════════════\n"
        "🎯 الإجراء التشغيلي:\n"
        "• متابعة التمريرة القادمة لنفس المنطقة.\n"
        "• إذا كانت قرب الساحل/محميات/منشآت: تصعيد.\n"
        "• (اختياري لاحقاً) ربطها بـ AIS لتحديد سفن مشتبه بها.\n"
    )

def status_report_ar(ksa_time: str, lookback_hours: int) -> str:
    return (
        "📄 تقرير رصد الانسكابات النفطية (SAR)\n"
        f"🕒 {ksa_time}\n"
        "════════════════════\n"
        "📍 المناطق: البحر الأحمر + الخليج العربي\n"
        "🛰️ المصدر: Sentinel-1 (SAR)\n\n"
        f"✅ لا توجد مؤشرات بقع ذات دلالة خلال آخر {lookback_hours} ساعة.\n"
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

    lookback = int(cfg.get("lookback_hours", 72))
    min_dark_ratio = float(cfg.get("min_dark_ratio", 0.07))
    dark_thr = float(cfg.get("dark_db_threshold", -23.0))
    cooldown_h = int(cfg.get("cooldown_hours_per_area", 12))
    max_alerts = int(cfg.get("max_alerts_per_run", 4))

    now = utc_now()
    start = now - dt.timedelta(hours=lookback)
    ksa_time_str = fmt_ksa(now)

    token = get_token(client_id, client_secret)

    state = load_state()
    seen_ids = set(state.get("seen_scene_ids", []))
    area_state = state.get("areas", {})

    findings: List[Dict[str, Any]] = []

    for area in cfg["areas"]:
        area_id = area["id"]
        area_name = area["name_ar"]
        bbox = area["bbox"]

        # cooldown
        last_alert = area_state.get(area_id, {}).get("last_alert_utc")
        in_cooldown = False
        if last_alert:
            try:
                last_dt = dt.datetime.fromisoformat(last_alert.replace("Z", "+00:00"))
                in_cooldown = (now - last_dt) < dt.timedelta(hours=cooldown_h)
            except Exception:
                pass

        scenes = catalog_search_s1(token, bbox, start, now, limit=8)

        for feat in scenes:
            scene_id = feat.get("id")
            scene_time = (feat.get("properties", {}) or {}).get("datetime")
            if not scene_id or not scene_time:
                continue
            if scene_id in seen_ids:
                continue

            # request window around timestamp
            t = dt.datetime.fromisoformat(scene_time.replace("Z", "+00:00"))
            t_from = t - dt.timedelta(minutes=5)
            t_to = t + dt.timedelta(minutes=5)

            try:
                db = process_vv_db_thumbnail(token, bbox, t_from, t_to)
                valid = np.isfinite(db)
                if valid.sum() < 1000:
                    seen_ids.add(scene_id)
                    continue

                dark = (db < dark_thr) & valid
                dark_ratio = float(dark.sum()) / float(valid.sum())

                if dark_ratio < min_dark_ratio:
                    seen_ids.add(scene_id)
                    continue

                centroid = dark_centroid_latlon(bbox, dark)
                if centroid is None:
                    seen_ids.add(scene_id)
                    continue

                lat, lon = centroid

                # score (بسيط لكن عملي)
                score = int(min(95, max(10, (dark_ratio / max(min_dark_ratio, 1e-6)) * 60 + 20)))

                findings.append({
                    "area_id": area_id,
                    "area_name": area_name,
                    "bbox": bbox,
                    "scene_id": scene_id,
                    "scene_utc": scene_time.replace("Z", ""),
                    "dark_ratio": dark_ratio,
                    "score": score,
                    "lat": lat,
                    "lon": lon,
                    "in_cooldown": in_cooldown
                })

            except Exception:
                # لو فشل طلب مشهد معيّن، نتجاهله ولا نخليه يعلق التشغيل
                pass

            seen_ids.add(scene_id)

    # strongest first
    findings.sort(key=lambda x: x["dark_ratio"], reverse=True)

    sent = 0
    for fnd in findings:
        if sent >= max_alerts:
            break

        area_id = fnd["area_id"]

        # trend from last dark ratio
        prev_ratio = area_state.get(area_id, {}).get("last_dark_ratio")
        trend = trend_arrow(prev_ratio, float(fnd["dark_ratio"]))

        # cooldown: لا نرسل إلا لو عالي جداً
        if fnd["in_cooldown"] and fnd["score"] < 85:
            continue

        msg = ops_card_ar(
            fnd["area_name"],
            ksa_time_str,
            fnd["scene_utc"],
            fnd["lat"],
            fnd["lon"],
            fnd["dark_ratio"],
            int(fnd["score"]),
            trend,
            dark_thr
        )
        send_telegram(bot, chat_id, msg)
        sent += 1
        time.sleep(1.2)

        # update area state
        area_state.setdefault(area_id, {})
        area_state[area_id]["last_dark_ratio"] = float(fnd["dark_ratio"])
        area_state[area_id]["last_scene_utc"] = fnd["scene_utc"] + "Z"
        area_state[area_id]["last_alert_utc"] = iso_z(now)

    if sent == 0:
        send_telegram(bot, chat_id, status_report_ar(ksa_time_str, lookback))

    # persist state
    state["seen_scene_ids"] = list(seen_ids)[-5000:]
    state["areas"] = area_state
    save_state(state)


if __name__ == "__main__":
    main()
