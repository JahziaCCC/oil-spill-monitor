import os
import json
import math
import datetime as dt
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from collections import defaultdict, deque

# ========= Secrets (GitHub Actions) =========
BOT = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
USERNAME = os.environ["CDSE_USERNAME"]     # لازم يكون الإيميل الكامل
PASSWORD = os.environ["CDSE_PASSWORD"]

# ========= Config =========
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

# Executive output
TOP_N = 3
MIN_CANDIDATE_SCORE = 30     # أقل درجة نحسبها "مرشح"
SHOW_ALWAYS_TOP_N = True     # يعرض أعلى 3 حتى لو الدرجات منخفضة

# Coverage score weights (كما اتفقنا)
# Recency 40 + PassCount 40 + Balance 20
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

# ========= Helpers =========
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

# ========= Auth =========
def get_access_token() -> str:
    payload = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": USERNAME,
        "password": PASSWORD,
    }
    r = requests.post(TOKEN_URL, data=payload, timeout=60)
    if r.status_code != 200:
        print("CDSE TOKEN STATUS:", r.status_code)
        print("CDSE TOKEN BODY (first 300 chars):", safe_preview(r.text, 300))
        r.raise_for_status()
    data = r.json()
    if "access_token" not in data:
        raise RuntimeError("Token response missing access_token.")
    return data["access_token"]

# ========= STAC Search =========
def stac_search(token: str, bbox, start_utc: str, end_utc: str):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {
        "collections": [COLLECTION],
        "bbox": bbox,
        "datetime": f"{start_utc}/{end_utc}",
        "limit": LIMIT_PER_REGION,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "fields": {
            "include": [
                "id",
                "properties.datetime",
                "properties.platform",
                "properties.sat:orbit_state",
                "properties.sat:relative_orbit",
                "properties.sar:instrument_mode",
                "properties.sar:polarizations",
                "assets",
                "links",
            ],
            "exclude": ["geometry"],
        },
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

# ========= Grouping (passes) =========
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
    t = props.get("datetime", "")
    t_round = round_time_to_minute(t)
    return f"{region}|{platform}|{t_round}|{orbit}|{rel_orbit}|{mode}|{pol}"

def summarize_groups(groups: dict):
    rows = []
    for _, items in groups.items():
        items_sorted = sorted(
            items,
            key=lambda it: (it.get("properties", {}) or {}).get("datetime", ""),
            reverse=True,
        )
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

# ========= B2: Smart Preview Analysis =========
def download_preview_to_gray(preview_url: str, max_size=900):
    """
    ينزّل Preview ويحوّله لGray numpy array (0..255).
    """
    if not preview_url:
        return None
    r = requests.get(preview_url, timeout=90)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("L")
    # تصغير لحفظ الوقت
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)))
    arr = np.array(img, dtype=np.uint8)
    return arr

def connected_components(mask: np.ndarray, min_pixels=250):
    """
    إيجاد مكونات متصلة بسيطة (8-neighbors) على مصفوفة bool
    يرجع قائمة blobs: dict(area, bbox, coords_sample)
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)
    blobs = []

    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            q = deque()
            q.append((y, x))
            visited[y, x] = 1

            area = 0
            minx = maxx = x
            miny = maxy = y

            # نحتفظ بعينة نقاط صغيرة فقط
            sample = []

            while q:
                cy, cx = q.popleft()
                area += 1
                if len(sample) < 50:
                    sample.append((cy, cx))

                minx = min(minx, cx); maxx = max(maxx, cx)
                miny = min(miny, cy); maxy = max(maxy, cy)

                for dy, dx in dirs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        q.append((ny, nx))

            if area >= min_pixels:
                blobs.append({
                    "area_px": area,
                    "bbox": (minx, miny, maxx, maxy),
                    "sample": sample,
                })

    # الأكبر أولاً
    blobs.sort(key=lambda b: b["area_px"], reverse=True)
    return blobs

def estimate_km2(area_px: int, img_shape):
    """
    مساحة تقريبية جداً من preview (ليست قياس علمي).
    نستخدم تحويل ثابت صغير لكي يعطي أرقام منطقية للتقرير.
    """
    h, w = img_shape
    # كل ما كبرت الصورة/الدقة تغيرت العلاقة؛ هذا "تقريب تشغيلي"
    norm = (h * w)
    frac = area_px / max(1, norm)
    # تحويل تقريبي: افترض مساحة مشهد تغطية فعالة ~ 2500 كم² (تقريب)
    return frac * 2500.0

def oil_likeness_score(gray: np.ndarray, blob):
    """
    يعطي Score 0..100 بناءً على:
    - تباين البقعة مع الخلفية
    - الاستطالة (aspect ratio)
    - نعومة الحواف (تقريب عبر compactness)
    """
    minx, miny, maxx, maxy = blob["bbox"]
    patch = gray[miny:maxy+1, minx:maxx+1]
    if patch.size == 0:
        return 0, {}

    # قناع البقعة داخل bbox: نعيد بناؤه من العينة؟ (تقريب)
    # بدل ذلك: نستخدم threshold محلي داخل bbox
    p = patch.astype(np.float32)
    thr = np.percentile(p, 15)  # dark threshold locally
    local_mask = p <= thr

    area = float(local_mask.sum())
    if area <= 0:
        return 0, {}

    # contrast: فرق متوسط الخلفية - متوسط البقعة
    spot_mean = float(p[local_mask].mean())
    bg_mean = float(p[~local_mask].mean()) if (~local_mask).any() else float(p.mean())
    contrast = max(0.0, bg_mean - spot_mean)  # كلما زاد أفضل

    # elongation: aspect ratio
    width = (maxx - minx + 1)
    height = (maxy - miny + 1)
    ar = max(width, height) / max(1.0, min(width, height))

    # compactness: perimeter^2 / area (تقريب بسيط)
    # نقرّب perimeter عبر count of boundary pixels
    m = local_mask.astype(np.uint8)
    # حدود تقريبية: بكسل حد إذا له جار صفر
    up = np.pad(m, ((1,0),(0,0)), mode="constant")[:-1,:]
    dn = np.pad(m, ((0,1),(0,0)), mode="constant")[1:,:]
    lf = np.pad(m, ((0,0),(1,0)), mode="constant")[:,:-1]
    rt = np.pad(m, ((0,0),(0,1)), mode="constant")[:,1:]
    boundary = (m == 1) & ((up==0) | (dn==0) | (lf==0) | (rt==0))
    perimeter = float(boundary.sum())
    compact = (perimeter * perimeter) / max(1.0, area)

    # scoring
    # contrast component (0..40)
    c_score = min(40.0, (contrast / 40.0) * 40.0)  # contrast~40 جيد جداً
    # elongation component (0..35): أفضل بين 2 و 6 تقريباً
    if ar < 1.3:
        e_score = 5.0
    elif ar < 2.0:
        e_score = 15.0
    elif ar < 4.0:
        e_score = 28.0
    else:
        e_score = 35.0
    # compactness penalty (0..25): قيم أقل = أنعم (أفضل)
    # إذا compact عالي جداً يعني شكل متعرج/خشِن
    if compact < 60:
        s_score = 25.0
    elif compact < 120:
        s_score = 15.0
    else:
        s_score = 8.0

    score = int(max(0.0, min(100.0, c_score + e_score + s_score)))

    details = {
        "contrast": round(contrast, 1),
        "elongation": round(ar, 2),
        "smoothness": round(s_score, 1),
        "compact": round(compact, 1),
    }
    return score, details

def analyze_pass_preview(preview_url: str):
    """
    يرجع أفضل blob كمرشح لهذا المرور:
    {score, area_km2, elongation, contrast, shape_label}
    """
    gray = download_preview_to_gray(preview_url)
    if gray is None:
        return None

    # threshold global: darkest 12%
    thr = np.percentile(gray, 12)
    mask = gray <= thr

    blobs = connected_components(mask, min_pixels=300)
    if not blobs:
        return None

    # جرّب أعلى 5 blobs فقط
    best = None
    for b in blobs[:5]:
        score, det = oil_likeness_score(gray, b)
        area_km2 = estimate_km2(b["area_px"], gray.shape)
        shape = "Oil-like" if det.get("elongation", 1) >= 2.0 and det.get("contrast", 0) >= 8 else "غير منتظم/طبيعي محتمل"
        cand = {
            "score": score,
            "area_km2": round(area_km2, 2),
            "elongation": det.get("elongation"),
            "contrast": det.get("contrast"),
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

# ========= Main =========
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
            "🛢️📡 تقرير انسكابات SAR الذكي (Executive)\n"
            f"🕒 {dt.datetime.now(KSA_TZ).strftime('%H:%M KSA')}\n"
            "════════════════════\n"
            f"✅ لا توجد *مشاهد SAR جديدة* خلال آخر {LOOKBACK_HOURS} ساعة.\n"
            + (latest_line + "\n" if latest_line else "")
            + "ℹ️ هذا رصد تغطية SAR (مصدر خام) — الكشف الذكي يعتمد على تحليل Preview.\n"
        )
        return

    # newest time
    new_items.sort(key=lambda it: (it.get("properties", {}) or {}).get("datetime", ""), reverse=True)
    newest_dt_utc = (new_items[0].get("properties", {}) or {}).get("datetime")
    if newest_dt_utc:
        state["last_seen_dt_utc"] = newest_dt_utc

    # group into passes
    groups = defaultdict(list)
    for it in new_items:
        groups[make_group_key(it)].append(it)

    grouped_rows = summarize_groups(groups)

    # region stats (counts by scenes)
    region_counts = defaultdict(int)
    region_latest = {}
    for r in grouped_rows:
        region_counts[r["region"]] += r["count"]
        region_latest[r["region"]] = max(region_latest.get(r["region"], ""), r["when"])

    red_sea_count = region_counts.get("البحر الأحمر", 0)
    gulf_count = region_counts.get("الخليج العربي", 0)

    # coverage score
    h = hours_since(newest_dt_utc, now_utc) if newest_dt_utc else 9999.0
    coverage = recency_points(h) + pass_points(len(grouped_rows)) + balance_points(red_sea_count, gulf_count)
    cov_label = score_label(coverage)

    # ===== B2 Analysis per pass (preview) =====
    candidates = []
    for r in grouped_rows:
        if not r.get("preview"):
            continue
        try:
            res = analyze_pass_preview(r["preview"])
            if res is None:
                continue
            candidates.append({
                "region": r["region"],
                "when": r["when"],
                "platform": r["platform"],
                "orbit": r["orbit"],
                "rel_orbit": r["rel_orbit"],
                "mode": r["mode"],
                "pol": r["pol"],
                "preview": r["preview"],
                "stac": r["stac"],
                **res
            })
        except Exception as e:
            # لا نوقف التشغيل — نتجاوز هذا المرور
            print("Preview analysis failed for one pass:", str(e)[:200])
            continue

    # count candidates (score>=MIN_CANDIDATE_SCORE)
    cand_count = sum(1 for c in candidates if c["score"] >= MIN_CANDIDATE_SCORE)

    # sort by score desc then recency
    candidates.sort(key=lambda c: (c["score"], c["when"]), reverse=True)

    # choose top N to show
    top = candidates[:TOP_N] if candidates else []

    # Executive summary counts
    likely_spill = sum(1 for c in top if c["score"] >= 70)
    need_follow = sum(1 for c in top if 50 <= c["score"] < 70)
    natural = sum(1 for c in top if c["score"] < 50)

    # ===== Build Executive message =====
    lines = []
    lines.append("🚨🛢️ تقرير انسكابات SAR الذكي (Executive)")
    lines.append(f"🕒 {dt.datetime.now(KSA_TZ).strftime('%H:%M KSA')}")
    lines.append("════════════════════")
    lines.append(f"📊 مؤشر التغطية: {coverage}/100 — {cov_label}")
    lines.append(f"🧠 عدد المرشحات: {cand_count}")
    lines.append(f"🎯 المعروض: أعلى {TOP_N} فقط")
    lines.append("════════════════════")

    if not top and candidates and SHOW_ALWAYS_TOP_N:
        # احتياط (نادر): لو cand_count=0 لكن فيه نتائج ضعيفة
        top = candidates[:TOP_N]

    if not top:
        lines.append("✅ لا توجد مرشحات واضحة حالياً بناءً على تحليل Preview.")
        lines.append("ℹ️ قد تظهر بقع داكنة طبيعية (رياح هادئة/أمواج داخلية) — يحتاج تأكيد عند توفر بيانات أكثر.")
    else:
        for c in top:
            label = risk_label(c["score"])
            rec = recommendation(c["score"])

            lines.append(f"{label} — {c['region']}")
            lines.append(f"• الثقة: {c['score']}%")
            lines.append(f"• المساحة: {c['area_km2']} كم² (تقريبية)")
            lines.append(f"• الشكل: {c['shape']}")
            if c.get("elongation") is not None:
                lines.append(f"• الاستطالة: {c['elongation']}")
            if c.get("contrast") is not None:
                lines.append(f"• التباين: {c['contrast']}")
            lines.append(f"• التوصية: {rec}")
            lines.append("")
            if c.get("preview"):
                lines.append(f"Preview: {c['preview']}")
            if c.get("stac"):
                lines.append(f"STAC: {c['stac']}")
            lines.append("════════════════════")

        lines.append("📌 الملخص التنفيذي:")
        lines.append(f"• انسكاب محتمل: {likely_spill}")
        lines.append(f"• يحتاج متابعة: {need_follow}")
        lines.append(f"• طبيعي غالباً: {natural}")

    telegram_send("\n".join(lines))

    # Save state
    for it in new_items:
        _id = it.get("id")
        if _id:
            seen.add(_id)
    state["seen_ids"] = list(seen)[-7000:]
    save_state(state)

if __name__ == "__main__":
    main()
