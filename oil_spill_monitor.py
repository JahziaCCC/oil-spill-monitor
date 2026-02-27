import os
import json
import datetime as dt
import requests

# ===== Copernicus Data Space / Sentinel Hub =====
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
BASE_URL = "https://sh.dataspace.copernicus.eu"
CATALOG_SEARCH = f"{BASE_URL}/api/v1/catalog/1.0.0/search"

CONFIG_FILE = "config.json"


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_z(d: dt.datetime) -> str:
    return d.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


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
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret
        },
        timeout=30
    )
    r.raise_for_status()
    return r.json()["access_token"]


def catalog_search_s1(token: str, bbox, start: dt.datetime, end: dt.datetime, limit: int = 20):
    headers = {"Authorization": f"Bearer {token}"}
    body = {
        "collections": ["sentinel-1-grd"],
        "datetime": f"{iso_z(start)}/{iso_z(end)}",
        "bbox": bbox,
        "limit": limit,
        "fields": {
            "include": ["id", "properties.datetime"]
        }
    }
    r = requests.post(CATALOG_SEARCH, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json().get("features", [])


def main():
    # Secrets
    client_id = os.environ["CDSE_CLIENT_ID"]
    client_secret = os.environ["CDSE_CLIENT_SECRET"]
    bot = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    # Load config
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    lookback_hours = int(cfg.get("lookback_hours", 168))

    now = utc_now()
    start = now - dt.timedelta(hours=lookback_hours)

    # Auth
    token = get_token(client_id, client_secret)

    # Debug header
    send_telegram(
        bot, chat_id,
        "🔎 DEBUG (Catalog)\n"
        f"⏱️ نطاق البحث: آخر {lookback_hours} ساعة\n"
        f"🕒 الآن (UTC): {iso_z(now)}"
    )

    # For each area: count scenes and show latest timestamp
    total_scenes = 0

    for area in cfg.get("areas", []):
        area_name = area.get("name_ar", "منطقة غير معروفة")
        bbox = area.get("bbox")

        scenes = catalog_search_s1(token, bbox, start, now, limit=20)
        count = len(scenes)
        total_scenes += count

        # get latest scene time
        latest = None
        if scenes:
            times = []
            for feat in scenes:
                t = (feat.get("properties", {}) or {}).get("datetime")
                if t:
                    times.append(t)
            if times:
                latest = sorted(times)[-1]

        msg = (
            "🔎 DEBUG (Catalog)\n"
            f"📍 {area_name}\n"
            f"📦 BBOX: {bbox}\n"
            f"🛰️ عدد المشاهد (Sentinel-1 GRD): {count}\n"
            f"🕒 أحدث مشهد (UTC): {latest if latest else '—'}"
        )
        send_telegram(bot, chat_id, msg)

    # Final status message
    if total_scenes == 0:
        send_telegram(
            bot, chat_id,
            "⚠️ ملاحظة تشخيصية:\n"
            "لم يتم العثور على أي مشاهد Sentinel-1 داخل النطاقات المحددة.\n"
            "هذا يعني غالباً أن BBOX يحتاج تعديل/تضييق أو أن الفترة قصيرة."
        )

    # Always send a normal report footer (so you know it finished)
    send_telegram(
        bot, chat_id,
        "📄 تقرير تشخيص رصد الانسكابات (SAR)\n"
        f"🕒 {now.astimezone(dt.timezone(dt.timedelta(hours=3))).strftime('%d-%m-%Y | %H:%M KSA')}\n"
        "════════════════════\n"
        "✅ تم تنفيذ فحص الكتالوج بنجاح.\n"
        "إذا ظهرت أرقام المشاهد، ننتقل بعدها للتحليل الحقيقي للبقع."
    )


if __name__ == "__main__":
    main()
