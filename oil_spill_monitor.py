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
    r = requests.post(
        url,
        json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
        timeout=30
    )
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

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


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
    # band1 = dark mask (0/255)
    # band2 = valid mask (0/255)
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

def process_mask_png(
    token: str,
    bbox: List[float],
    time_from: dt.datetime,
    time_to: dt.datetime,
    thr_db: float,
    w: int,
    h: int
) -> Tuple[np.ndarray, np.ndarray]:
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
                "dataFilter": {"timeRange": {"from": iso_z(time_from), "to": iso_z(time_to)}}
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
        snippet = (r.text or "")[:700]
        raise RuntimeError(f"Process API failed: HTTP {r.status_code}\n{snippet}")

    img = Image.open(BytesIO(r.content))
    arr = np.array(img)

    # Usually "LA" => HxWx2
    if arr.ndim == 2:
        dark = arr > 0
        valid = np.ones_like(dark, dtype=bool)
        return dark, valid

    if arr.ndim == 3 and arr.shape[2] >= 2:
        dark = arr[..., 0] > 0
        valid = arr[..., 1] > 0
        return dark, valid

    raise RuntimeError("Unexpected PNG shape from Process API")


# ---------------- Tiling (fix meters-per-pixel) ----------------
def split_bbox(bbox: List[float], nx: int = 3, ny: int = 3) -> List[List[float]]:
    """
    Split bbox into nx*ny smaller bboxes.
    bbox = [minLon, minLat, maxLon, maxLat]
    """
    minLon, minLat, maxLon, maxLat = bbox
    lons = np.linspace(minLon, maxLon, nx + 1)
    lats = np.linspace(minLat, maxLat, ny + 1)

    tiles = []
    for ix in range(nx):
        for iy in range(ny):
            tiles.append([
                float(lons[ix]),
                float(lats[iy]),
                float(lons[ix + 1]),
                float(lats[iy + 1]),
            ])
    return tiles


# ---------------- Shape / Smart Filter (A) ----------------
def smart_filter_metrics(mask: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Metrics from a boolean mask:
      - count: pixels count
      - aspect_ratio: elongation of bounding box
      - fill_ratio: how filled is the bounding box (compactness proxy)
      - confidence: 0..1 based on shape
    Returns None if too small.
    """
    ys, xs = np.where(mask)
    count = int(xs.size)
    if count < 60:
        return None  # too small / noise

    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)

    # avoid divide-by-zero
    short = max(1, min(w, h))
    long_ = max(w, h)
    aspect_ratio = long_ / short

    bbox_area = float(w * h)
    fill_ratio = count / bbox_area if bbox_area > 0 else 0.0

    # Heuristics:
    # - Very elongated => ship wake (reject)
    # - Very low fill (thin line) => wake-ish (reject)
    # - Moderate aspect + moderate fill => oil-like
    # Confidence combines both
    aspect_score = 1.0 - clamp((aspect_ratio - 1.5) / 6.0, 0.0, 1.0)  # best near 1.5..3
    fill_score = clamp((fill_ratio - 0.05) / 0.25, 0.0, 1.0)          # best >= ~0.15
    confidence = 0.55 * aspect_score + 0.45 * fill_score

    return {
        "count": float(count),
        "w": float(w),
        "h": float(h),
        "aspect_ratio": float(aspect_ratio),
        "fill_ratio": float(fill_ratio),
        "confidence": float(confidence),
    }

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


# ---------------- Messaging ----------------
def ops_card(
    area_name: str,
    ksa_time: str,
    scene_utc: str,
    lat: float,
    lon: float,
    dark_ratio: float,
    thr_db: float,
    score: int,
    mode_note: str,
    scenes_found: int,
    process_requests: int,
    shape_note: str,
    confidence_pct: int
) -> str:
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
        "🧠 Smart Filter (A)\n"
        f"• النتيجة: {shape_note}\n"
        f"• الثقة: {confidence_pct}%\n\n"
        f"🔎 التغطية: مشاهد={scenes_found} | طلبات Process={process_requests}\n"
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


# ---------------- Main ----------------
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

    # Smart Filter thresholds (tunable)
    # reject if too elongated or too thin
    MAX_ASPECT_REJECT = float(cfg.get("smart_max_aspect_reject", 7.0))      # wake lines often > 7
    MIN_FILL_REJECT = float(cfg.get("smart_min_fill_reject", 0.04))         # very thin line
    MIN_CONF_SEND = float(cfg.get("smart_min_confidence_send", 0.40))       # send only if >= 40%

    now = utc_now()
    start = now - dt.timedelta(hours=lookback)
    ksa_time = fmt_ksa(now)

    token = get_token(client_id, client_secret)

    # Tiling params (key for meters-per-pixel)
    NX, NY = int(cfg.get("tiles_nx", 3)), int(cfg.get("tiles_ny", 3))
    W, H = int(cfg.get("tile_width", 1024)), int(cfg.get("tile_height", 1024))

    best_candidates: List[Dict[str, Any]] = []
    diag_lines: List[str] = []

    for area in cfg["areas"]:
        area_name = area["name_ar"]
        bbox = area["bbox"]

        scenes = catalog_search_s1(token, bbox, start, now, limit=20)
        scenes_found = len(scenes)

        if scenes_found == 0:
            diag_lines.append(f"• {area_name}: مشاهد=0")
            continue

        tiles = split_bbox(bbox, NX, NY)

        best = None
        process_requests = 0
        process_errors = 0
        last_error = ""

        # reduce load: 3 scenes x 9 tiles = 27 calls per area (as you saw)
        for feat in scenes[:3]:
            scene_time = (feat.get("properties", {}) or {}).get("datetime")
            if not scene_time:
                continue

            t = dt.datetime.fromisoformat(scene_time.replace("Z", "+00:00"))
            t_from = t - dt.timedelta(minutes=10)
            t_to = t + dt.timedelta(minutes=10)

            for tbbox in tiles:
                try:
                    dark_mask, valid_mask = process_mask_png(token, tbbox, t_from, t_to, thr_db, w=W, h=H)
                    process_requests += 1

                    valid_count = int(valid_mask.sum())
                    if valid_count < 800:
                        continue

                    combo = dark_mask & valid_mask
                    dark_count = int(combo.sum())
                    dark_ratio = dark_count / float(valid_count)

                    # ---- Smart Filter A ----
                    metrics = smart_filter_metrics(combo)
                    if metrics is None:
                        continue

                    aspect = metrics["aspect_ratio"]
                    fill = metrics["fill_ratio"]
                    conf = metrics["confidence"]

                    # Reject wake-like shapes
                    if aspect >= MAX_ASPECT_REJECT:
                        continue
                    if fill <= MIN_FILL_REJECT:
                        continue
                    if conf < MIN_CONF_SEND:
                        continue

                    c = centroid_latlon(tbbox, combo)
                    if c is None:
                        continue
                    lat, lon = c

                    # score: combine dark_ratio + confidence
                    base = (dark_ratio / max(min_dark_ratio, 1e-6)) * 60 + 20
                    score = int(clamp(base * (0.65 + 0.35 * conf), 10, 95))
                    conf_pct = int(round(conf * 100))

                    # classify note
                    if conf >= 0.70:
                        shape_note = "Oil-like ✔️ (احتمال wake منخفض)"
                    elif conf >= 0.55:
                        shape_note = "مرشح جيد ✔️"
                    else:
                        shape_note = "مرشح متوسط (بحاجة متابعة)"

                    cand = {
                        "area_name": area_name,
                        "scene_utc": scene_time.replace("Z", ""),
                        "lat": lat,
                        "lon": lon,
                        "dark_ratio": dark_ratio,
                        "score": score,
                        "conf": conf,
                        "conf_pct": conf_pct,
                        "shape_note": shape_note,
                        "scenes_found": scenes_found,
                        "process_requests": process_requests,
                    }

                    # pick best by (score then dark_ratio)
                    if (best is None) or (cand["score"] > best["score"]) or (
                        cand["score"] == best["score"] and cand["dark_ratio"] > best["dark_ratio"]
                    ):
                        best = cand

                except Exception as e:
                    process_errors += 1
                    last_error = str(e)
                    continue

        diag_lines.append(
            f"• {area_name}: مشاهد={scenes_found} | طلبات Process={process_requests} | أخطاء={process_errors}"
        )
        if process_requests == 0 and process_errors > 0:
            diag_lines.append(f"  ↳ آخر خطأ: {(last_error or '')[:700]}")

        if best:
            best_candidates.append(best)

    # If nothing passes Smart Filter, still send diagnostics only (no false alarms)
    if not best_candidates:
        send_telegram(bot, chat_id, diag_msg(ksa_time, lookback, diag_lines))
        send_telegram(bot, chat_id, "🧠 Smart Filter (A): لم يمر أي مرشح موثوق اليوم. هذا طبيعي ويقلل الإنذارات الكاذبة.")
        return

    best_candidates.sort(key=lambda x: (x["score"], x["dark_ratio"]), reverse=True)

    sent = 0
    for cand in best_candidates:
        if sent >= max_alerts:
            break

        mode_note = "🚨 Alert Mode (تجاوز العتبة)" if cand["dark_ratio"] >= min_dark_ratio else "📡 Analyst Mode (مرشح بعد الفلترة)"
        msg = ops_card(
            cand["area_name"], ksa_time, cand["scene_utc"],
            cand["lat"], cand["lon"], cand["dark_ratio"], thr_db, cand["score"],
            mode_note, cand["scenes_found"], cand["process_requests"],
            cand["shape_note"], cand["conf_pct"]
        )
        send_telegram(bot, chat_id, msg)
        sent += 1
        time.sleep(1.0)

    # Send diagnostic summary as well
    send_telegram(bot, chat_id, diag_msg(ksa_time, lookback, diag_lines))


if __name__ == "__main__":
    main()
