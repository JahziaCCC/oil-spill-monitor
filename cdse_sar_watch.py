import os
import json
import datetime as dt
import requests
from collections import defaultdict

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
COLLECTION = "sentinel-1-grd"   # Sentinel-1 GRD in CDSE STAC

LOOKBACK_HOURS = 72
LIMIT_PER_REGION = 100  # زدناها شوي عشان التجميع يعطي صورة أكمل

# البحر الأحمر + الخليج العربي (BBox)
REGIONS = [
    {"name_ar": "البحر الأحمر",  "bbox": [32.0, 12.0, 44.5, 30.5]},
    {"name_ar": "الخليج العربي", "bbox": [47.0, 23.0, 56.8, 30.8]},
]

# عرض
MAX_GROUPS_TO_SHOW = 5

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
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "disable_web_page_preview": False,
    }
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
    text = text.replace("\n", " ").replace("\r", " ")
    return text[:n]

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

# ========= CDSE Auth =========
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
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
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
        dt_utc = (feats[0].get("properties", {}) or {}).get("datetime")
        if dt_utc and (latest is None or dt_utc > latest):
            latest = dt_utc
    return latest

# ========= Grouping =========
def round_time_to_minute(iso: str) -> str:
    # Sentinel-1 tiles لنفس المرور تكون بفارق ثواني
    # نجمعها إذا كانت في نفس الدقيقة
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
    # groups: key -> list(items)
    rows = []
    for key, items in groups.items():
        # representative item = الأحدث داخل المجموعة
        items_sorted = sorted(
            items,
            key=lambda it: (it.get("properties", {}) or {}).get("datetime", ""),
            reverse=True,
        )
        rep = items_sorted[0]
        props = rep.get("properties", {}) or {}
        when = props.get("datetime", "")
        region = rep.get("_region_ar", "")
        platform = props.get("platform", "n/a")
        orbit = props.get("sat:orbit_state", "n/a")
        rel_orbit = props.get("sat:relative_orbit", "n/a")
        mode = props.get("sar:instrument_mode", "n/a")
        pol = props.get("sar:polarizations", "n/a")
        preview, stac = pick_preview_and_stac(rep)
        rows.append({
            "region": region,
            "when": when,
            "platform": platform,
            "orbit": orbit,
            "rel_orbit": rel_orbit,
            "mode": mode,
            "pol": pol,
            "count": len(items),
            "preview": preview,
            "stac": stac,
        })

    rows.sort(key=lambda r: r["when"], reverse=True)
    return rows

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

    # لا جديد
    if not new_items:
        latest_dt_utc = get_latest_scene_datetime_utc(token) or state.get("last_seen_dt_utc")
        latest_line = f"🛰️ آخر مرور/مشهد معروف: {fmt_dt(latest_dt_utc)}" if latest_dt_utc else ""

        telegram_send(
            "🛢️📡 رصد الانسكابات (SAR) — تغطية البحر الأحمر + الخليج العربي\n"
            f"🕒 {dt.datetime.now(KSA_TZ).strftime('%Y-%m-%d %H:%M KSA')}\n"
            "════════════════════\n"
            f"✅ لا توجد *مشاهد SAR جديدة* خلال آخر {LOOKBACK_HOURS} ساعة.\n"
            + (latest_line + "\n" if latest_line else "")
            + "ℹ️ هذا رصد تغطية SAR (مصدر خام) — كشف الانسكاب الفعلي يتم بتحليل الصورة.\n"
        )
        return

    # تحديث آخر مرّة (UTC) من الأحدث
    new_items.sort(key=lambda it: (it.get("properties", {}) or {}).get("datetime", ""), reverse=True)
    newest_dt_utc = (new_items[0].get("properties", {}) or {}).get("datetime")
    if newest_dt_utc:
        state["last_seen_dt_utc"] = newest_dt_utc

    # ====== Grouping ======
    groups = defaultdict(list)
    for it in new_items:
        groups[make_group_key(it)].append(it)

    grouped_rows = summarize_groups(groups)

    # ====== Region stats ======
    region_counts = defaultdict(int)
    region_latest = {}
    for r in grouped_rows:
        region_counts[r["region"]] += r["count"]
        if r["region"] not in region_latest:
            region_latest[r["region"]] = r["when"]
        else:
            if r["when"] > region_latest[r["region"]]:
                region_latest[r["region"]] = r["when"]

    # ====== Build message ======
    lines = []
    lines.append("🛢️📡 رصد الانسكابات (SAR) — البحر الأحمر + الخليج العربي")
    lines.append(f"🕒 {dt.datetime.now(KSA_TZ).strftime('%Y-%m-%d %H:%M KSA')}")
    lines.append("════════════════════")
    lines.append(f"✅ إجمالي المشاهد الجديدة: {len(new_items)} خلال آخر {LOOKBACK_HOURS} ساعة.")
    lines.append(f"🧩 بعد التجميع (تمريرات/مجموعات): {len(grouped_rows)}")
    if newest_dt_utc:
        lines.append(f"🛰️ أحدث مرور: {fmt_dt(newest_dt_utc)}")
    lines.append("════════════════════")

    # ملخص حسب المنطقة
    rs = []
    for reg in ["البحر الأحمر", "الخليج العربي"]:
        c = region_counts.get(reg, 0)
        latest = region_latest.get(reg)
        if latest:
            rs.append(f"• {reg}: {c} مشهد | أحدث: {fmt_dt(latest)}")
        else:
            rs.append(f"• {reg}: 0 مشهد")
    lines.append("📌 ملخص حسب المنطقة:")
    lines.extend(rs)
    lines.append("════════════════════")
    lines.append("📄 أبرز التمريرات (بعد التجميع):")

    # عرض Top groups
    for i, r in enumerate(grouped_rows[:MAX_GROUPS_TO_SHOW], start=1):
        lines.append(f"{i}️⃣ {r['region']}")
        lines.append(f"• الوقت: {fmt_dt(r['when'])}")
        lines.append(f"• Platform: {r['platform']} | Orbit: {r['orbit']} | RelOrbit: {r['rel_orbit']} | Mode: {r['mode']} | Pol: {r['pol']}")
        lines.append(f"• عدد المشاهد داخل المرور: {r['count']}")
        if r["preview"]:
            lines.append(f"• Preview: {r['preview']}")
        if r["stac"]:
            lines.append(f"• STAC: {r['stac']}")
        lines.append("")

    if len(grouped_rows) > MAX_GROUPS_TO_SHOW:
        lines.append(f"… ويوجد {len(grouped_rows)-MAX_GROUPS_TO_SHOW} تمريرات إضافية (بعد التجميع).")

    lines.append("ℹ️ ملاحظة: هذا رصد تغطية SAR (مصدر خام) — كشف الانسكاب الفعلي يتم بتحليل الصورة.")

    telegram_send("\n".join(lines))

    # تحديث الحالة
    for it in new_items:
        _id = it.get("id")
        if _id:
            seen.add(_id)

    state["seen_ids"] = list(seen)[-5000:]
    save_state(state)

if __name__ == "__main__":
    main()
