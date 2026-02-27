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

def get_token(client_id: str, client_secret: str) -> str:
    r = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret},
        timeout=30
    )
    r.raise_for_status()
    return r.json()["access_token"]

def risk_badge(score: int) -> str:
    if score >= 85: return "🔴 حرج"
    if score >= 70: return "🟠 مرتفع"
    if score >= 55: return "🟡 متوسط"
    return "🟢 منخفض"


# ---------------- Catalog ----------------
def catalog_search_s1(token: str, bbox: List[float], start: dt.datetime, end: dt.datetime, limit: int = 20) -> List[Dict[str, Any]]:
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


# ---------------- Process API (PNG MASK) ----------------
def build_evalscript_mask(thr_db: float) -> str:
    # نُخرج قناتين:
    # band1 = mask (0/255) للبقعة الداكنة
    # band2 = dataMask (0/255) للبكسلات الصحيحة
    return f"""
//VERSION=3
function setup() {{
  return {{
    input: [{{ bands: ["VV", "dataMask"] }}],
    output: {{ bands: 2, sampleType: "UINT8" }}
  }};
}}
function toDB(x) {{ return 10.0 * Math.log(x) / Math.LN10; }}
function evaluatePixel(s) {{
  if (s.dataMask === 0) return [0, 0];
  var db = toDB(s.VV);
  var isDark = (db < {thr_db}) ? 255 : 0;
  return [isDark, 255];
}}
"""

def process_mask_png(token: str, bbox: List[float], time_from: dt.datetime, time_to: dt.datetime, thr_db: float, w: int = 256, h: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    يرجع:
      dark_mask: bool array (H,W)
      valid_mask: bool array (H,W)
    """
    headers = {"Authorization": f"Bearer {token}"}
    evalscript = build_evalscript_mask(thr_db)

    body = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {"from": iso_z(time_from), "to": iso_z(time_to)}
                }
            }]
        },
        "output": {
            "width": w,
            "height": h,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}]
        },
        "evalscript": evalscript
    }

    r = requests.post(PROCESS_API, headers=headers, json=body, timeout=120)

    if r.status_code != 200:
        # نرمي الخطأ مع تفاصيل نصية (أهم شيء عشان ما يصير تم تحليل=0 بدون سبب)
        snippet = (r.text or "")[:600]
        raise RuntimeError(f"Process API failed: HTTP {r.status_code}\n{snippet}")

    img = Image.open(BytesIO(r.content))

    # PNG بقناتين عادة يكون وضع "LA"
    arr = np.array(img)
    if arr.ndim == 2:
        # لو رجع قناة واحدة فقط (نادر) نعتبرها mask ونفترض valid=all
        dark = arr > 0
        valid = np.ones_like(dark, dtype=bool)
        return dark, valid

    if arr.ndim == 3 and arr.shape[2] >= 2:
        dark = arr[..., 0] > 0
        valid = arr[..., 1] > 0
        return dark, valid

    raise RuntimeError("Unexpected PNG shape from Process API")


# ---------------- Geolocation ----------------
def centroid_latlon(bbox: List[float], mask: np.ndarray) -> Optional[Tuple[float, float]]:
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
             dark_ratio: float, thr_db: float, score: int, mode_note: str,
             scenes_found: int, scenes_processed: int) -> str:
    return (
        "🚨 بطاقة عمليات بيئية – رصد انسكابات (SAR)\n"
        "════════════════════\n"
        f"📍 المنطقة: {area_name}\n"
        f"🌍 الإحداثيات: {lat:.4f}N , {lon:.4f}E\n"
        f"🕒 وقت التحديث: {ksa_time}\n\n"
        f"📊 مستوى الخطر: {risk_badge(score)} ({score}/100)\n\n"
        "🛰️ Sentinel-1 (SAR)\n"
        f"• مؤشر البقعة الداكنة: {dark_ratio:.2%}\n"
        f"• العتبة (dB): أقل من {thr_db}\n"
        f"• وقت المشهد (UTC): {scene_utc}\n\n"
        f"🔎 التغطية: مشاهد={scenes_found} | تم تحليل={scenes_processed}\n"
        f"🧾 الوضع: {mode_note}\n"
        "════════════════════\n"
        "🎯 الإجراء:\n"
        "• متابعة التمريرة القادمة.\n"
        "• إذا قرب الساحل/منشآت: تصعيد.\n"
    )

def diag_msg(ksa_time: str, lookback: int, lines: List[str]) -> str:
    return (
        "📄 تقرير تشخيص رصد الانسكابات (SAR)\n"
        f"🕒 {ksa_time}\n"
        "════════════════════\n"
        f"⏱️ نطاق البحث: آخر {lookback} ساعة\n\n"
        + "\n".join(lines)
    )


def main():
    client_id = os.environ["CDSE_CLIENT_ID"]
    client_secret = os.environ["CDSE_CLIENT_SECRET"]
    bot = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    lookback = int(cfg.get("lookback_hours", 168))
    thr_db = float(cfg.get("dark_db_threshold", -21.0))
    min_dark_ratio = float(cfg.get("min_dark_ratio", 0.001))
    max_alerts = int(cfg.get("max_alerts_per_run", 1))

    now = utc_now()
    start = now - dt.timedelta(hours=lookback)
    ksa_time = fmt_ksa(now)

    token = get_token(client_id, client_secret)

    best_candidates: List[Dict[str, Any]] = []
    diag_lines: List[str] = []

    for area in cfg["areas"]:
        area_name = area["name_ar"]
        bbox = area["bbox"]

        scenes = catalog_search_s1(token, bbox, start, now, limit=20)
        scenes_found = len(scenes)
        scenes_processed = 0

        if scenes_found == 0:
            diag_lines.append(f"• {area_name}: مشاهد=0")
            continue

        best = None
        process_errors = 0
        last_error_text = ""

        for feat in scenes[:6]:
            scene_time = (feat.get("properties", {}) or {}).get("datetime")
            if not scene_time:
                continue

            t = dt.datetime.fromisoformat(scene_time.replace("Z", "+00:00"))
            t_from = t - dt.timedelta(minutes=8)
            t_to = t + dt.timedelta(minutes=8)

            try:
                dark_mask, valid_mask = process_mask_png(token, bbox, t_from, t_to, thr_db, w=256, h=256)
                scenes_processed += 1

                valid_count = int(valid_mask.sum())
                if valid_count < 500:
                    continue

                dark_count = int((dark_mask & valid_mask).sum())
                dark_ratio = dark_count / float(valid_count)

                c = centroid_latlon(bbox, dark_mask & valid_mask)
                if c is None:
                    continue

                lat, lon = c
                score = int(min(95, max(10, (dark_ratio / max(min_dark_ratio, 1e-6)) * 60 + 20)))

                cand = {
                    "area_name": area_name,
                    "scene_utc": scene_time.replace("Z", ""),
                    "lat": lat,
                    "lon": lon,
                    "dark_ratio": dark_ratio,
                    "score": score,
                    "scenes_found": scenes_found,
                    "scenes_processed": scenes_processed,
                }

                if best is None or cand["dark_ratio"] > best["dark_ratio"]:
                    best = cand

            except Exception as e:
                process_errors += 1
                last_error_text = str(e)
                continue

        diag_lines.append(f"• {area_name}: مشاهد={scenes_found} | تم تحليل={scenes_processed} | أخطاء Process={process_errors}")

        # إذا Process فشل بالكامل في المنطقة، نرسل سبب آخر خطأ (مختصر)
        if scenes_processed == 0 and process_errors > 0:
            snippet = (last_error_text or "")[:700]
            diag_lines.append(f"  ↳ آخر خطأ: {snippet}")

        if best:
            best_candidates.append(best)

    # لو ما طلع أي مرشح (حتى Analyst)، نرسل تشخيص واضح
    if not best_candidates:
        send_telegram(bot, chat_id, diag_msg(ksa_time, lookback, diag_lines))
        return

    # رتّب الأقوى
    best_candidates.sort(key=lambda x: x["dark_ratio"], reverse=True)

    sent = 0
    for cand in best_candidates:
        if sent >= max_alerts:
            break

        mode_note = "🚨 Alert Mode (تجاوز العتبة)" if cand["dark_ratio"] >= min_dark_ratio else "📡 Analyst Mode (أفضل مرشح – قد يكون Look-alike)"
        msg = ops_card(
            cand["area_name"], ksa_time, cand["scene_utc"],
            cand["lat"], cand["lon"], cand["dark_ratio"], thr_db, cand["score"],
            mode_note, cand["scenes_found"], cand["scenes_processed"]
        )
        send_telegram(bot, chat_id, msg)
        sent += 1
        time.sleep(1.0)

    # أيضًا نرسل سطر تشخيص مختصر للتأكيد
    send_telegram(bot, chat_id, diag_msg(ksa_time, lookback, diag_lines))


if __name__ == "__main__":
    main()
