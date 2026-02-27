import os, json, datetime as dt
import requests
import numpy as np

# ===== ENV =====
BOT = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

CONFIG_FILE = "config.json"

# ===== Telegram =====
def send(msg):
    requests.post(
        f"https://api.telegram.org/bot{BOT}/sendMessage",
        json={"chat_id": CHAT_ID, "text": msg}
    )

# ===== Fake SAR candidate (DEBUG ANALYST MODE) =====
# الآن نرسل أقوى مرشح دائماً بدل لا توجد مؤشرات

def build_ops_card(area_name):
    now = dt.datetime.utcnow().strftime("%d-%m-%Y | %H:%M UTC")

    # قيم تجريبية تمثل أقوى بقعة
    lat = 22.41
    lon = 38.12
    dark_ratio = 0.032
    score = 62

    if score >= 80:
        level = "🔴 حرج"
    elif score >= 65:
        level = "🟠 مرتفع"
    elif score >= 50:
        level = "🟡 متوسط"
    else:
        level = "🟢 منخفض"

    return f"""🚨 بطاقة عمليات بيئية – SAR

════════════════════
🛢️ الحدث: أقوى بقعة داكنة (تحليل تجريبي)

📍 المنطقة: {area_name}
🌍 الإحداثيات: {lat}N , {lon}E

🕒 وقت التحليل: {now}

📊 مستوى الخطر: {level} ({score}/100)
📈 الاتجاه: → ثابت

🛰️ تحليل SAR:
• مؤشر البقعة الداكنة: {dark_ratio:.2%}
• الوضع: Analyst Mode (اختبار)

════════════════════
🎯 الإجراء:
• مراقبة مستمرة
• انتظار مرور قمر جديد
"""

def main():

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    send("🔎 Analyst Mode ON\nإرسال أقوى مرشح لكل منطقة.")

    for area in cfg["areas"]:
        send(build_ops_card(area["name_ar"]))

if __name__ == "__main__":
    main()
