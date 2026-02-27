import os, json, datetime as dt, requests

BOT = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

def send(msg):
    requests.post(
        f"https://api.telegram.org/bot{BOT}/sendMessage",
        json={"chat_id": CHAT_ID, "text": msg}
    )

def build_ops_card():
    now = dt.datetime.utcnow().strftime("%d-%m-%Y | %H:%M UTC")

    # حالياً مثال تجريبي (التحليل الحقيقي نضيفه بالخطوة القادمة)
    risk_score = 68
    trend = "↑ يزداد"

    if risk_score >= 80:
        level = "🔴 حرج"
    elif risk_score >= 65:
        level = "🟠 مرتفع"
    elif risk_score >= 50:
        level = "🟡 متوسط"
    else:
        level = "🟢 منخفض"

    return f"""🚨 بطاقة عمليات بيئية – رصد بحري

════════════════════
🛢️ الحدث: بقعة محتملة (SAR)
🕒 آخر تحديث: {now}

📊 مستوى الخطر: {level} ({risk_score}/100)
📈 الاتجاه: {trend}

🛰️ تحليل الأقمار الصناعية:
• تم رصد بقعة داكنة محتملة
• المصدر: Sentinel-1 SAR

════════════════════
🎯 الإجراء التشغيلي:
• متابعة التمريرة القادمة
• رفع المراقبة عند زيادة المؤشر
"""

if __name__ == "__main__":
    send(build_ops_card())
