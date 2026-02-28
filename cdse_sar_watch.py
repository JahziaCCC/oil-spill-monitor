# --- نفس imports عندك ---
import os
import json
import math
import datetime as dt
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from collections import defaultdict, deque

BOT = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
USERNAME = os.environ["CDSE_USERNAME"]
PASSWORD = os.environ["CDSE_PASSWORD"]

STATE_FILE = "cdse_sar_state.json"
KSA_TZ = dt.timezone(dt.timedelta(hours=3))

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STAC_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"
COLLECTION = "sentinel-1-grd"

LOOKBACK_HOURS = 72
LIMIT_PER_REGION = 100

REGIONS = [
    {"name_ar": "البحر الأحمر",  "bbox": [32.0, 12.0, 44.5, 30.5]},
    {"name_ar": "الخليج العربي", "bbox": [47.0, 23.0, 56.8, 30.8]},
]

TOP_N = 3
MIN_CANDIDATE_SCORE = 35
SHOW_ALWAYS_TOP_N = True

# ========= NEW FILTERS (التعديل المهم) =========
DARK_PERCENTILE_GLOBAL = 8          # كان 12 → خففنا الحساسية
MIN_PIXELS_BLOB = 600               # كان 300 → نرفعها
MAX_BLOB_AREA_KM2 = 25.0            # أي شيء أكبر غالباً ليس انسكاب
MAX_BBOX_FILL_RATIO = 0.35          # لو البقعة تعبّي الباتش كثير → wind shadow غالباً

def recency_points(hours):
    if hours <= 24: return 40
    if hours <= 48: return 30
    if hours <= 72: return 20
    return 10

def pass_points(n_groups):
    if n_groups >= 20: return 40
    if n_groups >= 10: return 30
    if n_groups >= 5:  return 20
    if n_groups >= 1:  return 10
    return 0

def balance_points(red_sea, gulf):
    total = red_sea + gulf
    if total == 0: return 0
    a = min(red_sea, gulf) / total
    if a >= 0.25: return 20
    if a > 0: return 10
    return 0

def score_label(score):
    if score >= 70: return "🟢 جيد"
    if score >= 40: return "🟠 متوسط"
    return "🔴 ضعيف"

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"seen_ids": [], "last_seen_dt_utc": None}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            s = json.load(f)
            s.setdefault("seen_ids", [])
            s.setdefault("last_seen_dt_utc", None)
            return s
    except Exception:
        return {"seen_ids": [], "last_seen_dt_utc": None}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def telegram_send(text: str):
    url = f"https://api.telegram.org/bot{BOT}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "disable_web_page_preview": False}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()

def fmt_dt(iso: str) -> str:
    try:
        t = dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(KSA_TZ)
        return t.strftime("%Y-%m-%d %H:%M KSA")
    except Exception:
        return iso

def safe_preview(text: str, n: int = 300) -> str:
    if text is None:
        return ""
    return text.replace("\n", " ").replace("\r", " ")[:n]

def iso_to_dt_utc(iso: str):
    try:
        t = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return t.astimezone(dt.timezone.utc)
    except Exception:
        return None

def hours_since(iso_utc: str, now_utc: dt.datetime) -> float:
    t = iso_to_dt_utc(iso_utc)
    if not t:
        return 9999.0
    d = now_utc - t
    return max(0.0, d.total_seconds() / 3600.0)

def get_access_token() -> str:
    payload = {"client_id": "cdse-public", "grant_type": "password", "username": USERNAME, "password": PASSWORD}
    r = requests.post(TOKEN_URL, data=payload, timeout=60)
    if r.status_code != 200:
        print("CDSE TOKEN STATUS:", r.status_code)
        print("CDSE TOKEN BODY:", safe_preview(r.text, 300))
        r.raise_for_status()
    data = r.json()
    if "access_token" not in data:
        raise RuntimeError("Token response missing access_token.")
    return data["access_token"]

def stac_search(token: str, bbox, start_utc: str, end_utc: str):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {
        "collections": [COLLECTION],
        "bbox": bbox,
        "datetime": f"{start_utc}/{end_utc}",
        "limit": LIMIT_PER_REGION,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "fields": {"include": ["id","properties.datetime","properties.platform","properties.sat:orbit_state",
                              "properties.sat:relative_orbit","properties.sar:instrument_mode",
                              "properties.sar:polarizations","assets","links"],
                   "exclude": ["geometry"]},
    }
    r = requests.post(STAC_SEARCH_URL, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    return r.json().get("features", [])

def get_latest_scene_datetime_utc(token: str):
    latest = None
    end_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for region in REGIONS:
        feats = stac_search(token, region["bbox"], "1970-01-01T00:00:00Z", end_utc)
        if not feats:
            continue
        dtu = (feats[0].get("properties", {}) or {}).get("datetime")
        if dtu and (latest is None or dtu > latest):
            latest = dtu
    return latest

def pick_preview_and_stac(item: dict):
    assets = item.get("assets", {}) or {}
    links = item.get("links", []) or []
    preview = None
    for k in ["thumbnail", "quicklook", "preview"]:
        if k in assets and isinstance(assets[k], dict) and assets[k].get("href"):
            preview = assets[k]["href"]
            break
    stac = None
    for l in links:
        if l.get("rel") == "self" and l.get("href"):
            stac = l["href"]
            break
    return preview, stac

def round_time_to_minute(iso: str) -> str:
    try:
        t = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        t = t.replace(second=0, microsecond=0, tzinfo=dt.timezone.utc)
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return iso

def make_group_key(item: dict) -> str:
    props = item.get("properties", {}) or {}
    region = item.get("_region_ar", "")
    platform = props.get("platform", "n/a")
    orbit = props.get("sat:orbit_state", "n/a")
    rel_orbit = props.get("sat:relative_orbit", "n/a")
    mode = props.get("sar:instrument_mode", "n/a")
    pol = str(props.get("sar:polarizations", "n/a"))
    t_round = round_time_to_minute(props.get("datetime", ""))
    return f"{region}|{platform}|{t_round}|{orbit}|{rel_orbit}|{mode}|{pol}"

def summarize_groups(groups: dict):
    rows = []
    for _, items in groups.items():
        items_sorted = sorted(items, key=lambda it: (it.get("properties", {}) or {}).get("datetime", ""), reverse=True)
        rep = items_sorted[0]
        props = rep.get("properties", {}) or {}
        preview, stac = pick_preview_and_stac(rep)
        rows.append({
            "region": rep.get("_region_ar", ""),
            "when": props.get("datetime", ""),
            "platform": props.get("platform", "n/a"),
            "orbit": props.get("sat:orbit_state", "n/a"),
            "rel_orbit": props.get("sat:relative_orbit", "n/a"),
            "mode": props.get("sar:instrument_mode", "n/a"),
            "pol": props.get("sar:polarizations", "n/a"),
            "count": len(items),
            "preview": preview,
            "stac": stac,
        })
    rows.sort(key=lambda r: r["when"], reverse=True)
    return rows

# ========= B2 Analysis =========
def download_preview_to_gray(preview_url: str, max_size=900):
    if not preview_url:
        return None
    r = requests.get(preview_url, timeout=90)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("L")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)))
    return np.array(img, dtype=np.uint8)

def connected_components(mask: np.ndarray, min_pixels=600):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)
    blobs = []
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            q = deque([(y, x)])
            visited[y, x] = 1
            area = 0
            minx = maxx = x
            miny = maxy = y
            while q:
                cy, cx = q.popleft()
                area += 1
                minx = min(minx, cx); maxx = max(maxx, cx)
                miny = min(miny, cy); maxy = max(maxy, cy)
                for dy, dx in dirs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        q.append((ny, nx))
            if area >= min_pixels:
                blobs.append({"area_px": area, "bbox": (minx, miny, maxx, maxy)})
    blobs.sort(key=lambda b: b["area_px"], reverse=True)
    return blobs

def estimate_km2(area_px: int, img_shape):
    h, w = img_shape
    frac = area_px / max(1, (h*w))
    # خفّضنا تحويل المساحة بشكل كبير عشان يصير واقعي في الإنذار الأولي
    # افترض "مساحة مرجعية" ~ 600 كم² بدلاً من 2500
    return frac * 600.0

def oil_likeness_score(gray: np.ndarray, bbox):
    minx, miny, maxx, maxy = bbox
    patch = gray[miny:maxy+1, minx:maxx+1].astype(np.float32)
    if patch.size == 0:
        return 0, None

    # threshold محلي أضيق (darkest 10% داخل patch)
    thr = np.percentile(patch, 10)
    local_mask = patch <= thr
    area = float(local_mask.sum())
    if area <= 0:
        return 0, None

    spot_mean = float(patch[local_mask].mean())
    bg_mean = float(patch[~local_mask].mean()) if (~local_mask).any() else float(patch.mean())
    contrast = max(0.0, bg_mean - spot_mean)

    width = (maxx - minx + 1)
    height = (maxy - miny + 1)
    ar = max(width, height) / max(1.0, min(width, height))

    # bbox fill ratio: لو البقعة تعبّي مساحة كبيرة من البوكس -> غالباً ظاهرة واسعة
    bbox_area = float(width * height)
    fill = area / max(1.0, bbox_area)

    # scoring أكثر تحفظاً
    c_score = min(35.0, (contrast / 50.0) * 35.0)     # سقف أقل
    if ar < 1.5:
        e_score = 6.0
    elif ar < 2.5:
        e_score = 16.0
    elif ar < 5.0:
        e_score = 26.0
    else:
        e_score = 32.0

    # penalty لو fill عالي
    if fill > 0.45:
        f_pen = 18.0
    elif fill > 0.30:
        f_pen = 10.0
    else:
        f_pen = 0.0

    score = int(max(0.0, min(100.0, c_score + e_score + 25.0 - f_pen)))

    details = {"contrast": round(contrast, 1), "elongation": round(ar, 2), "fill": round(fill, 2)}
    return score, details

def analyze_pass_preview(preview_url: str):
    gray = download_preview_to_gray(preview_url)
    if gray is None:
        return None

    thr = np.percentile(gray, DARK_PERCENTILE_GLOBAL)
    mask = gray <= thr

    blobs = connected_components(mask, min_pixels=MIN_PIXELS_BLOB)
    if not blobs:
        return None

    best = None
    for b in blobs[:6]:
        score, det = oil_likeness_score(gray, b["bbox"])
        if det is None:
            continue

        area_km2 = estimate_km2(b["area_px"], gray.shape)

        # ====== Filters to reduce false positives ======
        if area_km2 > MAX_BLOB_AREA_KM2:
            continue
        if det["fill"] > MAX_BBOX_FILL_RATIO:
            continue

        shape = "Oil-like" if (det["elongation"] >= 2.0 and det["contrast"] >= 10) else "غير منتظم/طبيعي محتمل"

        cand = {
            "score": score,
            "area_km2": round(area_km2, 2),
            "elongation": det["elongation"],
            "contrast": det["contrast"],
            "shape": shape,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand

    return best

def risk_label(score: int):
    if score >= 70: return "🔴 HIGH RISK"
    if score >= 50: return "🟠 MEDIUM RISK"
    return "🟡 LOW RISK"

def recommendation(score: int):
    if score >= 70: return "مراقبة فورية"
    if score >= 50: return "متابعة"
    return "مراقبة فقط"

def main():
    state = load_state()
    seen = set(state.get("seen_ids", []))

    now_utc = dt.datetime.now(dt.timezone.utc)
    start_utc_dt = now_utc - dt.timedelta(hours=LOOKBACK_HOURS)
    start_utc = start_utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    token = get_access_token()

    new_items = []
    for region in REGIONS:
        feats = stac_search(token, region["bbox"], start_utc, end_utc)
        for it in feats:
            _id = it.get("id")
            if not _id or _id in seen:
                continue
            it["_region_ar"] = region["name_ar"]
            new_items.append(it)

    if not new_items:
        latest_dt_utc = get_latest_scene_datetime_utc(token) or state.get("last_seen_dt_utc")
        latest_line = f"🛰️ آخر مرور/مشهد معروف: {fmt_dt(latest_dt_utc)}" if latest_dt_utc else ""
        telegram_send(
            "🚨🛢️ تقرير انسكابات SAR الذكي (Executive)\n"
            f"🕒 {dt.datetime.now(KSA_TZ).strftime('%H:%M KSA')}\n"
            "════════════════════\n"
            f"✅ لا توجد *مشاهد SAR جديدة* خلال آخر {LOOKBACK_HOURS} ساعة.\n"
            + (latest_line + "\n" if latest_line else "")
            + "ℹ️ الكشف الذكي يعتمد على تحليل Preview (إنذار أولي).\n"
        )
        return

    new_items.sort(key=lambda it: (it.get("properties", {}) or {}).get("datetime", ""), reverse=True)
    newest_dt_utc = (new_items[0].get("properties", {}) or {}).get("datetime")
    if newest_dt_utc:
        state["last_seen_dt_utc"] = newest_dt_utc

    groups = defaultdict(list)
    for it in new_items:
        groups[make_group_key(it)].append(it)
    grouped_rows = summarize_groups(groups)

    region_counts = defaultdict(int)
    for r in grouped_rows:
        region_counts[r["region"]] += r["count"]
    red_sea_count = region_counts.get("البحر الأحمر", 0)
    gulf_count = region_counts.get("الخليج العربي", 0)

    h = hours_since(newest_dt_utc, now_utc) if newest_dt_utc else 9999.0
    coverage = recency_points(h) + pass_points(len(grouped_rows)) + balance_points(red_sea_count, gulf_count)
    cov_label = score_label(coverage)

    candidates = []
    for r in grouped_rows:
        if not r.get("preview"):
            continue
        try:
            res = analyze_pass_preview(r["preview"])
            if not res:
                continue
            candidates.append({
                "region": r["region"],
                "when": r["when"],
                "preview": r["preview"],
                "stac": r["stac"],
                **res
            })
        except Exception as e:
            print("Preview analysis failed:", str(e)[:200])
            continue

    # المرشحات = اللي فوق الحد الأدنى
    cand_count = sum(1 for c in candidates if c["score"] >= MIN_CANDIDATE_SCORE)
    candidates.sort(key=lambda c: (c["score"], c["when"]), reverse=True)

    top = candidates[:TOP_N] if candidates else []
    if not top and candidates and SHOW_ALWAYS_TOP_N:
        top = candidates[:TOP_N]

    likely_spill = sum(1 for c in top if c["score"] >= 70)
    need_follow = sum(1 for c in top if 50 <= c["score"] < 70)
    natural = sum(1 for c in top if c["score"] < 50)

    lines = []
    lines.append("🚨🛢️ تقرير انسكابات SAR الذكي (Executive)")
    lines.append(f"🕒 {dt.datetime.now(KSA_TZ).strftime('%H:%M KSA')}")
    lines.append("════════════════════")
    lines.append(f"📊 مؤشر التغطية: {coverage}/100 — {cov_label}")
    lines.append(f"🧠 عدد المرشحات: {cand_count}")
    lines.append(f"🎯 المعروض: أعلى {TOP_N} فقط")
    lines.append("════════════════════")

    if not top:
        lines.append("✅ لا توجد مرشحات ذات معنى بعد تطبيق فلاتر الحجم/التمدد.")
        lines.append("ℹ️ هذا طبيعي — كثير من البقع الداكنة تكون ظواهر سطحية (رياح هادئة/أمواج داخلية).")
    else:
        for c in top:
            label = risk_label(c["score"])
            rec = recommendation(c["score"])
            lines.append(f"{label} — {c['region']}")
            lines.append(f"• الثقة: {c['score']}%")
            lines.append(f"• المساحة: {c['area_km2']} كم² (تقريبية)")
            lines.append(f"• الشكل: {c['shape']}")
            lines.append(f"• الاستطالة: {c['elongation']}")
            lines.append(f"• التباين: {c['contrast']}")
            lines.append(f"• التوصية: {rec}")
            lines.append("")
            lines.append(f"Preview: {c['preview']}")
            if c.get("stac"):
                lines.append(f"STAC: {c['stac']}")
            lines.append("════════════════════")

        lines.append("📌 الملخص التنفيذي:")
        lines.append(f"• انسكاب محتمل: {likely_spill}")
        lines.append(f"• يحتاج متابعة: {need_follow}")
        lines.append(f"• طبيعي غالباً: {natural}")

    telegram_send("\n".join(lines))

    for it in new_items:
        _id = it.get("id")
        if _id:
            seen.add(_id)
    state["seen_ids"] = list(seen)[-7000:]
    save_state(state)

if __name__ == "__main__":
    main()
