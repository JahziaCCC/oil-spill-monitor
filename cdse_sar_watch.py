import os
import json
import datetime as dt
import requests

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

LOOKBACK_HOURS = 72             # ✅ عدلناها إلى 72 ساعة
LIMIT_PER_REGION = 50

# البحر الأحمر + الخليج العربي (BBox)
REGIONS = [
    {"name_ar": "البحر الأحمر",  "bbox": [32.0, 12.0, 44.5, 30.5]},
    {"name_ar": "الخليج العربي", "bbox": [47.0, 23.0, 56.8, 30.8]},
]

# ========= Helpers =========
def load_state():
    if not os.path.exists(STATE_FILE):
        return {"seen_ids": [], "last_seen_dt_utc": None}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            s = json.load(f)
            if "seen_ids" not in s:
                s["seen_ids"] = []
            if "last_seen_dt_utc" not in s:
                s["last_seen_dt_utc"] = None
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
    # iso like 2026-03-01T01:27:00Z
    try:
        t = dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(KSA_TZ)
        return t.strftime("%Y-%m-%d %H:%M KSA")
    except Exception:
        return iso

def pick_links(item: dict):
    assets = item.get("assets", {}) or {}
    links = item.get("links", []) or []

    thumb = None
    for k in ["thumbnail", "quicklook", "preview"]:
        if k in assets and isinstance(assets[k], dict) and assets[k].get("href"):
            thumb = assets[k]["href"]
            break

    self_link = None
    for l in links:
        if l.get("rel") == "self" and l.get("href"):
            self_link = l["href"]
            break

    alt_link = None
    for l in links:
        if l.get("rel") in ("via", "alternate") and l.get("href"):
            alt_link = l["href"]
            break

    return self_link, alt_link, thumb

def safe_preview(text: str, n: int = 300) -> str:
    if text is None:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    return text[:n]

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
        # تشخيص آمن: نطبع فقط سبب الرفض بدون أي أسرار
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
    """
    يجيب أحدث مشهد (بدون فلترة زمنية) لكل منطقة — فقط لعرض "آخر مرور معروف"
    """
    latest = None
    for region in REGIONS:
        feats = stac_search(
            token=token,
            bbox=region["bbox"],
            start_utc="1970-01-01T00:00:00Z",
            end_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        if not feats:
            continue
        dt_utc = (feats[0].get("properties", {}) or {}).get("datetime")
        if not dt_utc:
            continue
        if latest is None or dt_utc > latest:
            latest = dt_utc
    return latest

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
        # جيب "آخر مرور معروف" (الأحدث) وعرضه
        latest_dt_utc = get_latest_scene_datetime_utc(token) or state.get("last_seen_dt_utc")
        latest_line = ""
        if latest_dt_utc:
            latest_line = f"🛰️ آخر مرور/مشهد معروف: {fmt_dt(latest_dt_utc)}"

        telegram_send(
            "🛢️📡 رصد الانسكابات (SAR)\n"
            f"🕒 {dt.datetime.now(KSA_TZ).strftime('%Y-%m-%d %H:%M KSA')}\n"
            "════════════════════\n"
            f"✅ لا توجد *مشاهد SAR جديدة* خلال آخر {LOOKBACK_HOURS} ساعة فوق البحر الأحمر والخليج العربي.\n"
            + (f"{latest_line}\n" if latest_line else "")
            + "ℹ️ هذا رصد تغطية SAR (مصدر خام) — كشف الانسكاب الفعلي يتم بتحليل الصورة.\n"
        )
        return

    # ترتيب الأحدث
    new_items.sort(key=lambda it: it.get("properties", {}).get("datetime", ""), reverse=True)

    # تحديث آخر مرّة (UTC)
    newest_dt_utc = (new_items[0].get("properties", {}) or {}).get("datetime")
    if newest_dt_utc:
        state["last_seen_dt_utc"] = newest_dt_utc

    lines = []
    lines.append("🛢️📡 رصد الانسكابات (SAR) — تغطية البحر الأحمر + الخليج العربي")
    lines.append(f"🕒 {dt.datetime.now(KSA_TZ).strftime('%Y-%m-%d %H:%M KSA')}")
    lines.append("════════════════════")
    lines.append(f"✅ تم رصد {len(new_items)} مشهد/مشاهد SAR جديدة خلال آخر {LOOKBACK_HOURS} ساعة.")
    if newest_dt_utc:
        lines.append(f"🛰️ أحدث مشهد: {fmt_dt(newest_dt_utc)}")
    lines.append("ℹ️ هذه *مشاهد خام* جاهزة للتحليل (احتمالات انسكاب تُؤكَّد بالتحليل).")
    lines.append("════════════════════")

    # أول 10 فقط لتفادي طول رسالة تيليجرام
    for i, it in enumerate(new_items[:10], start=1):
        props = it.get("properties", {}) or {}
        when = fmt_dt(props.get("datetime", ""))
        orbit = props.get("sat:orbit_state", "n/a")
        rel_orbit = props.get("sat:relative_orbit", "n/a")
        mode = props.get("sar:instrument_mode", "n/a")
        pol = props.get("sar:polarizations", "n/a")

        self_link, alt_link, thumb = pick_links(it)

        lines.append(f"{i}️⃣ {it.get('_region_ar','')}")
        lines.append(f"• الوقت: {when}")
        lines.append(f"• Orbit: {orbit} | RelOrbit: {rel_orbit} | Mode: {mode} | Pol: {pol}")
        if thumb:
            lines.append(f"• Preview: {thumb}")
        if self_link:
            lines.append(f"• STAC: {self_link}")
        elif alt_link:
            lines.append(f"• Link: {alt_link}")
        lines.append("")

    if len(new_items) > 10:
        lines.append(f"… ويوجد {len(new_items)-10} مشاهد إضافية (تظهر بالتشغيلات القادمة).")

    telegram_send("\n".join(lines))

    # تحديث الحالة
    for it in new_items:
        _id = it.get("id")
        if _id:
            seen.add(_id)

    state["seen_ids"] = list(seen)[-2000:]
    save_state(state)

if __name__ == "__main__":
    main()
