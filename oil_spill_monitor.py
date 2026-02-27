import os, json, time
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple

import requests
import numpy as np
from PIL import Image
from io import BytesIO
from dateutil import tz

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
BASE_URL = "https://sh.dataspace.copernicus.eu"
CATALOG_SEARCH = f"{BASE_URL}/api/v1/catalog/1.0.0/search"
PROCESS_API = f"{BASE_URL}/api/v1/process"

CONFIG_FILE = "config.json"
KSA_TZ = tz.gettz("Asia/Riyadh")


# ---------------- Helpers ----------------
def utc_now():
    return dt.datetime.now(dt.timezone.utc)

def iso_z(d):
    return d.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")

def fmt_ksa(d_utc):
    return d_utc.astimezone(KSA_TZ).strftime("%d-%m-%Y | %H:%M KSA")

def send_telegram(bot, chat_id, text):
    url = f"https://api.telegram.org/bot{bot}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=30).raise_for_status()

def get_token(client_id, client_secret):
    r = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]

def risk_badge(score):
    if score >= 85: return "🔴 حرج"
    if score >= 70: return "🟠 مرتفع"
    if score >= 55: return "🟡 متوسط"
    return "🟢 منخفض"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---------------- Catalog ----------------
def catalog_search_s1(token, bbox, start, end, limit=20):
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


# ---------------- Process API ----------------
def build_evalscript_mask(thr_db):
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
  if (s.dataMask === 0) return [0,0];
  var db = toDB(s.VV);
  var isDark = (db < {thr_db}) ? 255 : 0;
  return [isDark,255];
}}
"""

def process_mask_png(token, bbox, time_from, time_to, thr_db, w, h):
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
        raise RuntimeError("Process API failed")

    img = Image.open(BytesIO(r.content))
    arr = np.array(img)

    dark = arr[...,0] > 0
    valid = arr[...,1] > 0
    return dark, valid


# ---------------- Tiling ----------------
def split_bbox(bbox, nx=3, ny=3):
    minLon, minLat, maxLon, maxLat = bbox
    lons = np.linspace(minLon, maxLon, nx+1)
    lats = np.linspace(minLat, maxLat, ny+1)

    tiles=[]
    for ix in range(nx):
        for iy in range(ny):
            tiles.append([float(lons[ix]), float(lats[iy]),
                          float(lons[ix+1]), float(lats[iy+1])])
    return tiles


# ---------------- Shape ----------------
def shape_metrics(mask):
    ys,xs=np.where(mask)
    if xs.size<50:
        return {"aspect":0,"fill":0,"conf":0.2}

    w=float(xs.max()-xs.min()+1)
    h=float(ys.max()-ys.min()+1)
    aspect=max(w,h)/max(1,min(w,h))
    fill=float(xs.size)/(w*h)

    conf=0.75
    if aspect>10: conf-=0.35
    elif aspect>6: conf-=0.2
    if fill<0.04: conf-=0.25
    elif fill<0.08: conf-=0.1

    conf=clamp(conf,0.2,0.9)
    return {"aspect":aspect,"fill":fill,"conf":conf}


def centroid_latlon(bbox, mask):
    ys,xs=np.where(mask)
    if xs.size<20:
        return None

    H,W=mask.shape
    x=float(xs.mean())
    y=float(ys.mean())

    minLon,minLat,maxLon,maxLat=bbox
    lon=minLon+(x/max(W-1,1))*(maxLon-minLon)
    lat=maxLat-(y/max(H-1,1))*(maxLat-minLat)
    return lat,lon


# ---------------- Message ----------------
def ops_card(area_name,ksa_time,scene_utc,lat,lon,
             dark_ratio,thr_db,score,mode_note,
             scenes_found,process_requests,
             conf_pct,aspect,fill):

    shape_hint="Oil-like ✔️" if conf_pct>=70 else ("مرشح متوسط" if conf_pct>=45 else "قد يكون Wake")

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
        "🧠 Smart Filter (A) — تقييم خفيف\n"
        f"• النتيجة: {shape_hint}\n"
        f"• الثقة: {conf_pct}%\n"
        f"• الاستطالة: {aspect:.1f} | الامتلاء: {fill:.2f}\n\n"
        "📊 مؤشرات التحليل:\n"
        f"• المشاهد المتاحة: {scenes_found}\n"
        f"• المشاهد المحللة: {process_requests}\n\n"
        f"🧾 الوضع: {mode_note}\n"
        "════════════════════\n"
        "🎯 الإجراء:\n"
        "• متابعة التمريرة القادمة.\n"
        "• إذا قرب الساحل/منشآت: تصعيد.\n"
    )


# ---------------- Main ----------------
def main():

    client_id=os.environ["CDSE_CLIENT_ID"]
    client_secret=os.environ["CDSE_CLIENT_SECRET"]
    bot=os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id=os.environ["TELEGRAM_CHAT_ID"]

    cfg=json.load(open(CONFIG_FILE,"r",encoding="utf-8"))

    lookback=cfg["lookback_hours"]
    thr_db=cfg["dark_db_threshold"]
    min_dark_ratio=cfg["min_dark_ratio"]

    now=utc_now()
    start=now-dt.timedelta(hours=lookback)
    ksa_time=fmt_ksa(now)

    token=get_token(client_id,client_secret)

    for area in cfg["areas"]:

        scenes=catalog_search_s1(token,area["bbox"],start,now,20)
        scenes_found=len(scenes)

        tiles=split_bbox(area["bbox"],3,3)

        best=None
        process_requests=0

        for feat in scenes[:3]:

            t=dt.datetime.fromisoformat(
                feat["properties"]["datetime"].replace("Z","+00:00")
            )

            for tb in tiles:

                dark,valid=process_mask_png(
                    token,tb,t-dt.timedelta(minutes=10),
                    t+dt.timedelta(minutes=10),
                    thr_db,1024,1024
                )

                process_requests+=1

                combo=dark & valid
                valid_count=int(valid.sum())
                if valid_count<800:
                    continue

                dark_ratio=int(combo.sum())/float(valid_count)

                c=centroid_latlon(tb,combo)
                if not c:
                    continue

                lat,lon=c
                m=shape_metrics(combo)
                conf_pct=int(round(m["conf"]*100))

                base=(dark_ratio/max(min_dark_ratio,1e-6))*60+20
                score=int(clamp(base*(0.7+0.3*m["conf"]),10,95))

                cand={
                    "lat":lat,"lon":lon,
                    "dark_ratio":dark_ratio,
                    "score":score,
                    "conf_pct":conf_pct,
                    "aspect":m["aspect"],
                    "fill":m["fill"],
                    "scene":feat["properties"]["datetime"],
                }

                if (best is None) or (cand["score"]>best["score"]):
                    best=cand

        if best:
            mode="🚨 Alert Mode (تجاوز العتبة)" if best["dark_ratio"]>=min_dark_ratio else "📡 Analyst Mode"

            msg=ops_card(
                area["name_ar"],ksa_time,best["scene"],
                best["lat"],best["lon"],
                best["dark_ratio"],thr_db,best["score"],
                mode,scenes_found,process_requests,
                best["conf_pct"],best["aspect"],best["fill"]
            )

            send_telegram(bot,chat_id,msg)


if __name__=="__main__":
    main()
