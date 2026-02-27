import os, requests, datetime

BOT = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

msg = f"""📄 تقرير رصد الانسكابات النفطية

🕒 {datetime.datetime.utcnow()} UTC

════════════════════
✅ النظام يعمل بنجاح
هذه رسالة اختبار أول تشغيل.
"""

requests.post(
    f"https://api.telegram.org/bot{BOT}/sendMessage",
    json={"chat_id": CHAT_ID, "text": msg}
)
