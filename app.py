import sys
import os
import glob
import json
import math
import time
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# ==========================================
# [0. 시크릿 로더 및 v8.1 공정 가중치 설정]
# ==========================================
def get_secret_key(key_name, default_val=""):
    val = os.environ.get(key_name, "")
    if val:
        return str(val).strip()
    try:
        import streamlit as st
        if key_name in st.secrets:
            return str(st.secrets[key_name]).strip()
    except Exception:
        pass
    return default_val

KST = timezone(timedelta(hours=9))
KMA_AUTH_KEY = get_secret_key("KMA_AUTH_KEY", "")
NASA_USER = get_secret_key("NASA_USER", "")
NASA_PASS = get_secret_key("NASA_PASS", "")

LEARNING_RATE = 0.20
W_PROB = 0.90
W_RAIN = 0.05
W_TEMP = 0.025
W_WIND = 0.025
MIN_EVAL_HOURS = 12
DAILY_SCORE_FLOOR = 2.0
DAILY_SCORE_CAP = 25.0
FINAL_WEIGHT_FLOOR = 2.0
FINAL_WEIGHT_CAP = 30.0
WEIGHT_RULE_VERSION = "V8_SNAPSHOT_FAIR_90_5"

SAVE_DIR = "./jeju_weather_data"
SNAPSHOT_DIR = os.path.join(SAVE_DIR, "forecast_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
WEIGHT_HISTORY_FILE = os.path.join(SAVE_DIR, "jeju_cumulative_weights_v8.json")
AUDIT_LOG_FILE = os.path.join(SAVE_DIR, "automation_audit_log_v8.csv")

SYSTEM_HEALTH = {
    "storage_mode": "✅ v8.1 스냅샷 정석 채점 & 공정 가중치(확률90%·강수5%) 가동 중",
    "storage_path": SAVE_DIR,
    "total_error_count": 0,
    "last_check_time": "-",
    "api_status": {
        "Open-Meteo (10대 통합 NWP)": {"status": "대기중", "code": "-", "latency_ms": 0, "errors": 0, "msg": "-"},
        "NASA GEOS-FP (순수 예보)":   {"status": "대기중", "code": "-", "latency_ms": 0, "errors": 0, "msg": "-"},
        "KMA 1.1 (AWS 매분자료)":     {"status": "대기중", "code": "-", "latency_ms": 0, "errors": 0, "msg": "-"},
        "KMA 6.3 (AWS 지점일람표)":   {"status": "대기중", "code": "-", "latency_ms": 0, "errors": 0, "msg": "-"},
        "KMA ASOS (종관기상관측)":    {"status": "대기중", "code": "-", "latency_ms": 0, "errors": 0, "msg": "-"}
    },
    "model_registry": {}
}

ASOS_HUBS = {
    184: {"name": "제주",   "lat": 33.5141, "lon": 126.5297},
    189: {"name": "서귀포", "lat": 33.2461, "lon": 126.5653},
    188: {"name": "성산",   "lat": 33.3868, "lon": 126.8802},
    185: {"name": "고산",   "lat": 33.2938, "lon": 126.1628}
}

JEJU_STATIONS = {
    "함덕":     {"stn": 780, "lat": 33.5432, "lon": 126.6695, "type": "AWS",  "ht": "14",   "desc": "조천읍 함덕해수욕장 일대"},
    "제주":     {"stn": 184, "lat": 33.5141, "lon": 126.5297, "type": "ASOS", "ht": "20",   "desc": "제주시 북부 도심"},
    "서귀포":   {"stn": 189, "lat": 33.2461, "lon": 126.5653, "type": "ASOS", "ht": "51",   "desc": "서귀포시 남부 도심"},
    "성산":     {"stn": 188, "lat": 33.3868, "lon": 126.8802, "type": "ASOS", "ht": "20",   "desc": "성산일출봉 및 동부 해안"},
    "고산":     {"stn": 185, "lat": 33.2938, "lon": 126.1628, "type": "ASOS", "ht": "71",   "desc": "한경면 고산리 및 서부 해안"},
    "애월":     {"stn": 893, "lat": 33.4628, "lon": 126.3292, "type": "AWS",  "ht": "22",   "desc": "애월읍 북서부 해안"},
    "구좌":     {"stn": 781, "lat": 33.5204, "lon": 126.8530, "type": "AWS",  "ht": "35",   "desc": "구좌읍 세화/월정 북동부"},
    "한림":     {"stn": 779, "lat": 33.4097, "lon": 126.2655, "type": "AWS",  "ht": "27",   "desc": "한림읍 협재/금능 일대"},
    "대정":     {"stn": 793, "lat": 33.2269, "lon": 126.2519, "type": "AWS",  "ht": "30",   "desc": "대정읍 모슬포 남서부"},
    "중문":     {"stn": 329, "lat": 33.2494, "lon": 126.4144, "type": "AWS",  "ht": "65",   "desc": "서귀포 중문관광단지"},
    "남원":     {"stn": 783, "lat": 33.2797, "lon": 126.7197, "type": "AWS",  "ht": "32",   "desc": "남원읍 남동부 해안"},
    "표선":     {"stn": 782, "lat": 33.3268, "lon": 126.8312, "type": "AWS",  "ht": "40",   "desc": "표선면 일대"},
    "성판악":   {"stn": 866, "lat": 33.3855, "lon": 126.6197, "type": "AWS",  "ht": "750",  "desc": "한라산 동부 중산간(750m)"},
    "윗세오름": {"stn": 871, "lat": 33.3622, "lon": 126.5186, "type": "AWS",  "ht": "1678", "desc": "한라산 고지대 산간(1700m)"},
    "우도":     {"stn": 885, "lat": 33.5064, "lon": 126.9531, "type": "AWS",  "ht": "18",   "desc": "우도면 도서 지역"},
    "추자도":   {"stn": 884, "lat": 33.9619, "lon": 126.2947, "type": "AWS",  "ht": "42",   "desc": "추자면 북방 도서"}
}

MODELS = {
    "ecmwf_ifs025":         {"name": "유럽(ECMWF IFS)",  "org": "ECMWF(유럽)",    "res": "25km"},
    "ecmwf_ifs":            {"name": "유럽(ECMWF 9km)",  "org": "ECMWF(유럽)",    "res": "9km"},
    "ecmwf_aifs025_single": {"name": "유럽(ECMWF AI)",   "org": "ECMWF(AI모델)",  "res": "25km"},
    "gfs_seamless":         {"name": "미국(NOAA GFS)",   "org": "NOAA(미국)",     "res": "11km"},
    "nasa_geos_fp":         {"name": "미국(NASA GEOS)",  "org": "NASA GMAO",      "res": "25km"},
    "icon_seamless":        {"name": "독일(DWD ICON)",   "org": "DWD(독일)",      "res": "11km"},
    "ukmo_seamless":        {"name": "영국(UKMO 10km)",  "org": "MetOffice(영국)","res": "10km"},
    "jma_seamless":         {"name": "일본(JMA 통합)",   "org": "JMA(일본)",      "res": "5~20km"},
    "cma_grapes_global":    {"name": "중국(CMA GRAPES)", "org": "CMA(중국)",      "res": "13km"},
    "meteofrance_seamless": {"name": "프랑스(ARPEGE)",   "org": "Météo-France",   "res": "25km"},
    "gem_seamless":         {"name": "캐나다(CMC GEM)",  "org": "CMC(캐나다)",    "res": "15km"}
}

def estimate_prob_from_rain(rain_series):
    conds = [rain_series.isna(), rain_series <= 0.0, rain_series <= 0.2, rain_series <= 1.0, rain_series <= 3.0]
    choices = [np.nan, 15.0, 40.0, 65.0, 82.0]
    return pd.Series(np.select(conds, choices, default=92.0), index=rain_series.index)

def calc_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

# ==========================================
# [1. 예보 스냅샷 아카이브 (저장 / 로드 / 14일 자동정리)]
# ==========================================
def _to_json_list(series):
    return [None if pd.isna(v) else round(float(v), 2) for v in series]

def prune_old_snapshots(keep_days=14):
    cutoff = (datetime.now(KST).date() - timedelta(days=keep_days)).strftime("%Y%m%d")
    for fp in glob.glob(os.path.join(SNAPSHOT_DIR, "snapshots_*.json")):
        ymd = os.path.basename(fp).replace("snapshots_", "").replace(".json", "")
        if ymd.isdigit() and ymd < cutoff:
            try:
                os.remove(fp)
            except Exception:
                pass

def save_forecast_snapshot(region_name, today_date, now_kst, df_mod, active_models):
    if df_mod.empty or not active_models:
        return False
    ymd = today_date.strftime("%Y%m%d")
    snap_path = os.path.join(SNAPSHOT_DIR, f"snapshots_{ymd}.json")

    snap_db = {}
    if os.path.exists(snap_path):
        try:
            with open(snap_path, "r", encoding="utf-8") as f:
                snap_db = json.load(f)
        except Exception:
            snap_db = {}

    if region_name in snap_db:
        return False

    start_ts = pd.Timestamp(today_date)
    end_ts = start_ts + pd.Timedelta(hours=71)
    df_win = df_mod[(df_mod.index >= start_ts) & (df_mod.index <= end_ts)]
    if df_win.empty:
        return False

    model_payload = {}
    for mc in active_models.keys():
        t_c, p_c, pr_c, w_c = f"temperature_2m_{mc}", f"precipitation_{mc}", f"precipitation_probability_{mc}", f"wind_speed_10m_{mc}"
        if all(c in df_win.columns for c in [t_c, p_c, pr_c, w_c]):
            model_payload[mc] = {
                "t": _to_json_list(df_win[t_c]),
                "p": _to_json_list(df_win[p_c]),
                "pr": _to_json_list(df_win[pr_c]),
                "w": _to_json_list(df_win[w_c])
            }

    if not model_payload:
        return False

    snap_db[region_name] = {
        "saved_at": now_kst.strftime("%Y-%m-%d %H:%M"),
        "times": [ts.strftime("%Y-%m-%d %H:%M") for ts in df_win.index],
        "models": model_payload
    }
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(snap_db, f, ensure_ascii=False, separators=(',', ':'))
    prune_old_snapshots(14)
    return True

def load_yesterday_snapshot(region_name, yesterday_date):
    y_start = pd.Timestamp(yesterday_date)
    y_end = y_start + pd.Timedelta(hours=23)
    candidates = [
        (yesterday_date - timedelta(days=1), "전일 발표 24h 리드타임"),
        (yesterday_date, "당일 새벽 발표")
    ]

    for c_date, lead_label in candidates:
        ymd = c_date.strftime("%Y%m%d")
        snap_path = os.path.join(SNAPSHOT_DIR, f"snapshots_{ymd}.json")
        if not os.path.exists(snap_path):
            continue
        try:
            with open(snap_path, "r", encoding="utf-8") as f:
                snap_db = json.load(f)
            if region_name not in snap_db:
                continue
            entry = snap_db[region_name]
            idx = pd.to_datetime(entry["times"])
            data_cols = {}
            for mc, md in entry["models"].items():
                data_cols[f"temperature_2m_{mc}"] = pd.to_numeric(md["t"], errors='coerce')
                data_cols[f"precipitation_{mc}"] = pd.to_numeric(md["p"], errors='coerce')
                data_cols[f"precipitation_probability_{mc}"] = pd.to_numeric(md["pr"], errors='coerce')
                data_cols[f"wind_speed_10m_{mc}"] = pd.to_numeric(md["w"], errors='coerce')
            df_snap = pd.DataFrame(data_cols, index=idx)
            df_y = df_snap[(df_snap.index >= y_start) & (df_snap.index <= y_end)]
            if len(df_y) >= MIN_EVAL_HOURS:
                saved_at = entry.get("saved_at", ymd)
                return df_y, f"📸 스냅샷 정석 채점 ({lead_label}, 저장시각 {saved_at})"
        except Exception:
            continue

    return pd.DataFrame(), "⏳ 스냅샷 미축적 (임시 Open-Meteo 단기치 채점 · 익일부터 정석 전환)"

def list_saved_snapshots():
    files = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "snapshots_*.json")), reverse=True)
    summary = []
    for fp in files[:7]:
        ymd = os.path.basename(fp).replace("snapshots_", "").replace(".json", "")
        try:
            with open(fp, "r", encoding="utf-8") as f:
                db = json.load(f)
            summary.append(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]} ({len(db)}개 지역)")
        except Exception:
            pass
    return summary

# ==========================================
# [2. Open-Meteo 3단계 자동 복구 + 15분 캐시 로더]
# ==========================================
_OM_CACHE = {}

def _parse_open_meteo_hourly(res_json, req_models, now_kst):
    df_raw = pd.DataFrame(res_json["hourly"])
    df_raw['time'] = pd.to_datetime(df_raw['time'])
    df_raw.set_index('time', inplace=True)

    df_out = pd.DataFrame(index=df_raw.index)
    loaded, max_hours = {}, {}

    for mc in req_models:
        t_c = f"temperature_2m_{mc}" if f"temperature_2m_{mc}" in df_raw.columns else ("temperature_2m" if len(req_models) == 1 and "temperature_2m" in df_raw.columns else None)
        if not t_c:
            continue
        p_c = f"precipitation_{mc}" if f"precipitation_{mc}" in df_raw.columns else ("precipitation" if "precipitation" in df_raw.columns else None)
        pr_c = f"precipitation_probability_{mc}" if f"precipitation_probability_{mc}" in df_raw.columns else ("precipitation_probability" if "precipitation_probability" in df_raw.columns else None)
        w_c = f"wind_speed_10m_{mc}" if f"wind_speed_10m_{mc}" in df_raw.columns else ("wind_speed_10m" if "wind_speed_10m" in df_raw.columns else None)

        out_t, out_p, out_pr, out_w = f"temperature_2m_{mc}", f"precipitation_{mc}", f"precipitation_probability_{mc}", f"wind_speed_10m_{mc}"
        s_t = pd.to_numeric(df_raw[t_c], errors='coerce').interpolate(method='linear', limit=2)
        s_p = pd.to_numeric(df_raw[p_c], errors='coerce') if p_c else pd.Series(np.nan, index=df_raw.index)
        s_w = pd.to_numeric(df_raw[w_c], errors='coerce').interpolate(method='linear', limit=2) if w_c else pd.Series(np.nan, index=df_raw.index)

        if pr_c and pd.to_numeric(df_raw[pr_c], errors='coerce').notna().sum() >= 10:
            s_pr = pd.to_numeric(df_raw[pr_c], errors='coerce')
            prob_info = "자체확률"
        else:
            s_pr = estimate_prob_from_rain(s_p)
            prob_info = "강수환산"

        valid_h = int(s_t[s_t.index >= now_kst].notna().sum())
        if valid_h >= 12:
            df_out[out_t] = s_t
            df_out[out_p] = s_p
            df_out[out_pr] = s_pr
            df_out[out_w] = s_w
            loaded[mc] = prob_info
            max_hours[mc] = valid_h

    return df_out, loaded, max_hours

def fetch_open_meteo_resilient(lat, lon, om_keys, past_days, now_kst):
    cache_key = (round(lat, 4), round(lon, 4), past_days)
    now_ts = time.time()
    if cache_key in _OM_CACHE and (now_ts - _OM_CACHE[cache_key]["ts"] < 900):
        c = _OM_CACHE[cache_key]
        return c["df"].copy(), dict(c["loaded"]), dict(c["max_hours"]), c["status"], c["msg"], c["ms"]

    url_om = "https://api.open-meteo.com/v1/forecast"
    t0 = time.time()
    last_code, last_reason = "ERR", "알 수 없는 오류"

    def _call_batch(model_list):
        nonlocal last_code, last_reason
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": ["temperature_2m", "precipitation_probability", "precipitation", "wind_speed_10m"],
            "wind_speed_unit": "ms", "timezone": "Asia/Seoul",
            "past_days": past_days, "forecast_days": 7
        }
        if model_list:
            params["models"] = model_list
        for attempt in range(2):
            try:
                resp = requests.get(url_om, params=params, timeout=15)
                last_code = str(resp.status_code)
                data = resp.json()
                if resp.status_code == 200 and "hourly" in data:
                    return data
                last_reason = str(data.get("reason", f"HTTP {resp.status_code}"))[:80]
                if resp.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    break
            except Exception as e:
                last_reason = str(e)[:80]
                time.sleep(1.0)
        return None

    df_all = pd.DataFrame()
    loaded_all, mh_all = {}, {}

    data_full = _call_batch(om_keys)
    if data_full:
        df_all, loaded_all, mh_all = _parse_open_meteo_hourly(data_full, om_keys, now_kst)
    else:
        parts = []
        for ch in [om_keys[:5], om_keys[5:]]:
            time.sleep(0.5)
            d_ch = _call_batch(ch)
            if d_ch:
                df_p, ld_p, mh_p = _parse_open_meteo_hourly(d_ch, ch, now_kst)
                if not df_p.empty:
                    parts.append(df_p)
                    loaded_all.update(ld_p)
                    mh_all.update(mh_p)
        if parts:
            df_all = pd.concat(parts, axis=1)

    if df_all.empty:
        d_bm = _call_batch(None)
        if d_bm:
            df_all, loaded_all, mh_all = _parse_open_meteo_hourly(d_bm, ["ecmwf_ifs025"], now_kst)

    ms = int((time.time() - t0) * 1000)
    if not df_all.empty:
        st_txt = "✅ 정상" if len(loaded_all) >= 8 else "⚠️ 부분복구"
        msg_txt = f"{len(loaded_all)}/{len(om_keys)}개 글로벌 모델 로드"
        _OM_CACHE[cache_key] = {"ts": now_ts, "df": df_all, "loaded": loaded_all, "max_hours": mh_all, "status": st_txt, "msg": msg_txt, "ms": ms}
        return df_all, loaded_all, mh_all, st_txt, msg_txt, ms

    return pd.DataFrame(), {}, {}, "❌ 통신실패", f"HTTP {last_code}: {last_reason}", ms

# ==========================================
# [3. NASA GEOS-FP 순수 예보 수집기 (격자 캐시 + 10분 회로차단기)]
# ==========================================
_NASA_CACHE = {}
_NASA_DOWN_UNTIL = 0.0

def _parse_opendap_ascii_1d(ascii_text, var_name):
    vals, capture = [], False
    for line in ascii_text.split('\n'):
        s = line.strip()
        if not s: continue
        if s.startswith(f"{var_name},") or s.startswith(f"{var_name}["):
            capture = True
            continue
        if capture:
            if s.startswith(("time,", "lat,", "lon,")): break
            parts = [p.strip() for p in s.split(',')]
            if len(parts) >= 2 and parts[0].startswith('['):
                v = pd.to_numeric(parts[1], errors='coerce')
                vals.append(float(v) if pd.notnull(v) else np.nan)
    return vals

def fetch_nasa_geos_forecast_only(lat, lon):
    global _NASA_DOWN_UNTIL
    lat_idx, lon_idx = int(round((lat + 90.0) / 0.25)), int(round((lon + 180.0) / 0.25))
    ck = (lat_idx, lon_idx)
    now_ts = time.time()

    if ck in _NASA_CACHE and (now_ts - _NASA_CACHE[ck]["ts"] < 900):
        c = _NASA_CACHE[ck]
        return c["df"].copy(), c["sec"], c["note"], c["has_rain"]

    if now_ts < _NASA_DOWN_UNTIL:
        return pd.DataFrame(), 0.0, "OpenDAP 일시차단(10분 대기)", False

    t0 = time.time()
    session = requests.Session()
    if NASA_USER and NASA_PASS:
        session.auth = (NASA_USER, NASA_PASS)

    now_utc = datetime.now(timezone.utc)
    for run_dt, cyc in [(now_utc, "00"), (now_utc - timedelta(days=1), "12")]:
        ymd = run_dt.strftime("%Y%m%d")
        base = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/fcast"
        slv_u = f"{base}/tavg1_2d_slv_Nx/tavg1_2d_slv_Nx.{ymd}_{cyc}.ascii?t2m[0:1:119][{lat_idx}][{lon_idx}],u10m[0:1:119][{lat_idx}][{lon_idx}],v10m[0:1:119][{lat_idx}][{lon_idx}]"
        flx_u = f"{base}/tavg1_2d_flx_Nx/tavg1_2d_flx_Nx.{ymd}_{cyc}.ascii?prectot[0:1:119][{lat_idx}][{lon_idx}]"
        try:
            r_s = session.get(slv_u, timeout=4)
            if r_s.status_code == 200 and "t2m" in r_s.text:
                t_v = _parse_opendap_ascii_1d(r_s.text, "t2m")
                u_v = _parse_opendap_ascii_1d(r_s.text, "u10m")
                v_v = _parse_opendap_ascii_1d(r_s.text, "v10m")

                p_v, has_rain = [], False
                try:
                    r_f = session.get(flx_u, timeout=4)
                    if r_f.status_code == 200 and "prectot" in r_f.text:
                        p_v = _parse_opendap_ascii_1d(r_f.text, "prectot")
                        has_rain = sum(pd.notnull(x) for x in p_v) >= 24
                except Exception:
                    has_rain = False

                if sum(pd.notnull(x) for x in t_v) >= 24:
                    st_kst = (datetime.strptime(f"{ymd}{cyc}", "%Y%m%d%H") + timedelta(hours=9)).replace(minute=0, second=0)
                    rows = []
                    for i, tk in enumerate(t_v):
                        if pd.notnull(tk) and 180.0 < tk < 340.0:
                            uv = u_v[i] if i < len(u_v) and pd.notnull(u_v[i]) and abs(u_v[i]) < 200 else np.nan
                            vv = v_v[i] if i < len(v_v) and pd.notnull(v_v[i]) and abs(v_v[i]) < 200 else np.nan
                            ws_val = float(math.sqrt(uv**2 + vv**2)) if (pd.notnull(uv) and pd.notnull(vv)) else np.nan
                            pr_mm = float(p_v[i] * 3600.0) if (has_rain and i < len(p_v) and pd.notnull(p_v[i]) and 0.0 <= p_v[i] < 10.0) else np.nan
                            rows.append({
                                "time": st_kst + timedelta(hours=i),
                                "temperature_2m_nasa_geos_fp": float(tk - 273.15),
                                "precipitation_nasa_geos_fp": pr_mm,
                                "wind_speed_10m_nasa_geos_fp": ws_val
                            })
                    if rows:
                        df_fc = pd.DataFrame(rows).set_index("time")
                        df_fc["precipitation_probability_nasa_geos_fp"] = estimate_prob_from_rain(df_fc["precipitation_nasa_geos_fp"])
                        sec = time.time() - t0
                        note = f"GEOS-FP({ymd}_{cyc}Z, 강수{'정상' if has_rain else '미수신-NaN유지'})"
                        SYSTEM_HEALTH["api_status"]["NASA GEOS-FP (순수 예보)"] = {
                            "status": "✅ 정상" if has_rain else "⚠️ 강수누락",
                            "code": "200 OK", "latency_ms": int(sec * 1000),
                            "errors": 0 if has_rain else 1, "msg": note
                        }
                        _NASA_CACHE[ck] = {"ts": now_ts, "df": df_fc, "sec": sec, "note": note, "has_rain": has_rain}
                        return df_fc, sec, note, has_rain
        except Exception:
            continue

    sec = time.time() - t0
    _NASA_DOWN_UNTIL = time.time() + 600.0
    SYSTEM_HEALTH["api_status"]["NASA GEOS-FP (순수 예보)"] = {
        "status": "⚠️ 미수신", "code": "TIMEOUT", "latency_ms": int(sec * 1000),
        "errors": 1, "msg": "OpenDAP 응답 지연 (10분간 호출 생략)"
    }
    return pd.DataFrame(), sec, "OpenDAP 미응답(제외)", False

# ==========================================
# [4. 기상청 6.3 지점일람표 및 관측소(1.1 AWS / ASOS) 파서]
# ==========================================
_DISCOVERED = False
def auto_discover_jeju_aws_stations(auth_key):
    global _DISCOVERED
    if _DISCOVERED or not auth_key: return
    _DISCOVERED = True
    url = "https://apihub.kma.go.kr/api/typ02/openApi/AwsYearlyInfoService/getAwsStnLstTbl"
    t0 = time.time()
    for yr in ["2024", "2023", "2022"]:
        try:
            res = requests.get(url, params={"pageNo": 1, "numOfRows": 700, "dataType": "XML", "year": yr, "month": "09", "authKey": auth_key}, timeout=5)
            if res.status_code == 200 and "<stn_id>" in res.text:
                root = ET.fromstring(res.text)
                cnt = 0
                for it in root.iter():
                    sid, sko, lat_e, lon_e, ht_e = it.find("stn_id"), it.find("stn_ko"), it.find("lat"), it.find("lon"), it.find("ht")
                    if None not in (sid, sko, lat_e, lon_e):
                        lv, lnv = float(lat_e.text), float(lon_e.text)
                        nk, sn = sko.text.strip(), int(sid.text.strip())
                        hv = ht_e.text.strip() if ht_e is not None and ht_e.text else "0"
                        if 33.10 <= lv <= 34.05 and 126.00 <= lnv <= 127.05 and nk not in JEJU_STATIONS and sn not in [184, 185, 188, 189]:
                            JEJU_STATIONS[nk] = {"stn": sn, "lat": lv, "lon": lnv, "type": "AWS", "ht": hv, "desc": f"제주 AWS (해발 {hv}m)"}
                            cnt += 1
                if cnt > 0:
                    SYSTEM_HEALTH["api_status"]["KMA 6.3 (AWS 지점일람표)"] = {"status": "✅ 정상", "code": "200 OK", "latency_ms": int((time.time()-t0)*1000), "errors": 0, "msg": f"제주 AWS {cnt}개소 확장"}
                    return
        except Exception:
            pass

_OBS_CACHE = {}
def fetch_kma_obs_v8(stn_info, date_str, auth_key):
    if not auth_key:
        return pd.DataFrame(), "기상청 API 키 미설정 (Secrets에 KMA_AUTH_KEY 등록 필요)"
    stn_id = stn_info["stn"]
    ck = (stn_id, date_str)
    if ck in _OBS_CACHE: return _OBS_CACHE[ck]
    tm1, tm2 = f"{date_str}0000", f"{date_str}2300"

    if stn_info["type"] == "ASOS":
        t0 = time.time()
        url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php?tm1={tm1}&tm2={tm2}&stn={stn_id}&help=0&authKey={auth_key}"
        try:
            res = requests.get(url, timeout=8)
            ms = int((time.time() - t0) * 1000)
            recs = []
            if res.status_code == 200:
                for ln in res.text.split('\n'):
                    s = ln.strip()
                    if not s or s.startswith('#'): continue
                    p = s.replace(',', ' ').split()
                    if len(p) > 15 and p[0].isdigit() and len(p[0]) == 12:
                        ot = pd.to_datetime(p[0], format='%Y%m%d%H%M', errors='coerce')
                        ws, ta, rn = pd.to_numeric(p[3], errors='coerce'), pd.to_numeric(p[11], errors='coerce'), pd.to_numeric(p[15], errors='coerce')
                        if pd.notnull(ot) and pd.notnull(ta) and ta > -50:
                            recs.append({'time': ot, 'obs_temp': float(ta), 'obs_rain': float(rn) if (pd.notnull(rn) and rn >= 0) else 0.0, 'obs_wind': float(ws) if (pd.notnull(ws) and ws >= 0) else 0.0})
            dfr = pd.DataFrame(recs).drop_duplicates('time').set_index('time') if recs else pd.DataFrame()
            SYSTEM_HEALTH["api_status"]["KMA ASOS (종관기상관측)"] = {"status": "✅ 정상" if not dfr.empty else "⚠️ 빈데이터", "code": str(res.status_code), "latency_ms": ms, "errors": 0, "msg": f"지점 {stn_id} {len(dfr)}시간"}
            _OBS_CACHE[ck] = (dfr, f"지점 {stn_id} (ASOS 직접 관측)")
            return _OBS_CACHE[ck]
        except Exception as e:
            return pd.DataFrame(), f"ASOS 지연({str(e)[:20]})"

    t0 = time.time()
    url_m = f"https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min?tm1={tm1}&tm2={tm2}&stn={stn_id}&disp=1&help=0&authKey={auth_key}"
    try:
        res = requests.get(url_m, timeout=8)
        ms = int((time.time() - t0) * 1000)
        if res.status_code == 200 and len(res.text.strip()) > 40 and "auth" not in res.text.lower():
            recs = []
            for ln in res.text.split('\n'):
                s = ln.strip()
                if not s or s.startswith('#'): continue
                p = [x.strip() for x in s.replace(',', ' ').split() if x.strip()]
                if len(p) >= 12 and p[0].isdigit() and len(p[0]) == 12 and p[0].endswith('00'):
                    ot = pd.to_datetime(p[0], format='%Y%m%d%H%M', errors='coerce')
                    ws10, ta, rn60 = pd.to_numeric(p[7], errors='coerce'), pd.to_numeric(p[8], errors='coerce'), pd.to_numeric(p[11], errors='coerce')
                    if pd.notnull(ot) and pd.notnull(ta) and -40 < ta < 50:
                        recs.append({
                            'time': ot, 'obs_temp': float(ta),
                            'obs_rain': float(rn60) if (pd.notnull(rn60) and rn60 >= 0) else 0.0,
                            'obs_wind': float(ws10) if (pd.notnull(ws10) and ws10 >= 0) else 0.0
                        })
            if recs:
                dfr = pd.DataFrame(recs).drop_duplicates('time').set_index('time')
                SYSTEM_HEALTH["api_status"]["KMA 1.1 (AWS 매분자료)"] = {"status": "✅ 정상", "code": "200 OK", "latency_ms": ms, "errors": 0, "msg": f"지점 {stn_id} {len(dfr)}시간 직결"}
                _OBS_CACHE[ck] = (dfr, f"지점 {stn_id} (1.1 AWS 매분자료 직접 관측)")
                return _OBS_CACHE[ck]
    except Exception:
        pass

    dists = sorted([(aid, h["name"], max(0.5, calc_distance_km(stn_info["lat"], stn_info["lon"], h["lat"], h["lon"]))) for aid, h in ASOS_HUBS.items()], key=lambda x: x[2])
    n1_id, n1_name, d1 = dists[0]
    n2_id, n2_name, d2 = dists[1]
    w1 = (1.0 / d1) / ((1.0 / d1) + (1.0 / d2))
    df1, _ = fetch_kma_obs_v8({"stn": n1_id, "type": "ASOS"}, date_str, auth_key)
    df2, _ = fetch_kma_obs_v8({"stn": n2_id, "type": "ASOS"}, date_str, auth_key)
    if not df1.empty and not df2.empty:
        dfb = ((df1 * w1) + (df2 * (1.0 - w1))).dropna()
        _OBS_CACHE[ck] = (dfb, f"거리보간({n1_name} {w1*100:.0f}%+{n2_name} {(1-w1)*100:.0f}%)")
        return _OBS_CACHE[ck]
    return pd.DataFrame(), "관측 데이터 대기"

# ==========================================
# [5. v8.1 공정 4중 채점 (스무딩 + 상한선 + 결측 모델 동결)]
# ==========================================
def evaluate_4way_fair(df_eval, candidate_models, df_obs):
    metrics = {}
    scored_keys = []

    for m_code in candidate_models:
        t_c, p_c, pr_c, w_c = f"temperature_2m_{m_code}", f"precipitation_{m_code}", f"precipitation_probability_{m_code}", f"wind_speed_10m_{m_code}"
        if not df_obs.empty and not df_eval.empty and all(c in df_eval.columns for c in [t_c, p_c, pr_c, w_c]):
            df_pair = pd.concat([df_eval[[t_c, p_c, pr_c, w_c]], df_obs[['obs_temp', 'obs_rain', 'obs_wind']]], axis=1).dropna()
            if len(df_pair) >= MIN_EVAL_HOURS:
                t_bias = float(np.mean(df_pair[t_c] - df_pair['obs_temp']))
                t_mae = float(np.mean(np.abs(df_pair[t_c] - df_pair['obs_temp'])))
                r_mae = float(np.mean(np.abs(df_pair[p_c] - df_pair['obs_rain'])))
                act_bin = (df_pair['obs_rain'] >= 0.1).astype(float)
                pred_p = (df_pair[pr_c] / 100.0).clip(0.0, 1.0)
                p_brier = float(np.mean((pred_p - act_bin) ** 2))
                w_bias = float(np.mean(df_pair[w_c] - df_pair['obs_wind']))
                w_mae = float(np.mean(np.abs(df_pair[w_c] - df_pair['obs_wind'])))
                metrics[m_code] = {
                    'scored': True, 'eval_hours': len(df_pair),
                    'temp_mae': round(t_mae, 2), 'temp_bias': round(t_bias, 2),
                    'rain_mae': round(r_mae, 2), 'prob_brier': round(p_brier, 3),
                    'wind_mae': round(w_mae, 2), 'wind_bias': round(w_bias, 2)
                }
                scored_keys.append(m_code)
                continue

        metrics[m_code] = {
            'scored': False, 'eval_hours': 0,
            'temp_mae': 0.0, 'temp_bias': 0.0, 'rain_mae': 0.0,
            'prob_brier': 0.0, 'wind_mae': 0.0, 'wind_bias': 0.0,
            'score_temp': 0.0, 'score_prob': 0.0, 'score_rain': 0.0, 'score_wind': 0.0
        }

    if not scored_keys:
        return {}, metrics

    t_inv = {k: 1.0 / (metrics[k]['temp_mae'] + 0.50) for k in scored_keys}
    p_inv = {k: 1.0 / (metrics[k]['prob_brier'] + 0.12) for k in scored_keys}
    r_inv = {k: 1.0 / (metrics[k]['rain_mae'] + 0.30) for k in scored_keys}
    w_inv = {k: 1.0 / (metrics[k]['wind_mae'] + 0.50) for k in scored_keys}
    st, sp, sr, sw = sum(t_inv.values()), sum(p_inv.values()), sum(r_inv.values()), sum(w_inv.values())

    raw_daily = {}
    for k in scored_keys:
        score_t = (t_inv[k] / st) * 100.0
        score_p = (p_inv[k] / sp) * 100.0
        score_r = (r_inv[k] / sr) * 100.0
        score_w = (w_inv[k] / sw) * 100.0
        metrics[k]['score_temp'] = round(score_t, 1)
        metrics[k]['score_prob'] = round(score_p, 1)
        metrics[k]['score_rain'] = round(score_r, 1)
        metrics[k]['score_wind'] = round(score_w, 1)
        raw_daily[k] = (W_TEMP * score_t) + (W_PROB * score_p) + (W_RAIN * score_r) + (W_WIND * score_w)

    capped = {k: float(np.clip(v, DAILY_SCORE_FLOOR, DAILY_SCORE_CAP)) for k, v in raw_daily.items()}
    c_tot = max(1e-9, sum(capped.values()))
    daily_w = {k: round((v / c_tot) * 100.0, 2) for k, v in capped.items()}
    return daily_w, metrics

def update_weights_fair(prev_weights, daily_weights, all_model_keys):
    eq_w = round(100.0 / max(1, len(all_model_keys)), 2)
    aligned_prev = {m: prev_weights.get(m, eq_w) for m in all_model_keys}
    p_tot = max(1e-9, sum(aligned_prev.values()))
    aligned_prev = {m: (v / p_tot) * 100.0 for m, v in aligned_prev.items()}

    if not daily_weights:
        return {m: round(v, 2) for m, v in aligned_prev.items()}, {m: round(v, 2) for m, v in aligned_prev.items()}

    scored_pool = sum(aligned_prev[m] for m in daily_weights.keys() if m in aligned_prev)
    d_tot = max(1e-9, sum(daily_weights.values()))

    new_w = {}
    for m in all_model_keys:
        if m in daily_weights:
            target_m = (daily_weights[m] / d_tot) * scored_pool
            new_w[m] = (1.0 - LEARNING_RATE) * aligned_prev[m] + LEARNING_RATE * target_m
        else:
            new_w[m] = aligned_prev[m]

    bounded = {m: float(np.clip(v, FINAL_WEIGHT_FLOOR, FINAL_WEIGHT_CAP)) for m, v in new_w.items()}
    b_tot = max(1e-9, sum(bounded.values()))
    final_w = {m: round((v / b_tot) * 100.0, 2) for m, v in bounded.items()}
    return {m: round(v, 2) for m, v in aligned_prev.items()}, final_w

def write_audit_log(region, eval_date, status, eval_mode, obs_source, scored_cnt, err_cnt, top_model, top_weight):
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([{
        "실행시각(KST)": now_str, "지역": region, "채점기준일": eval_date,
        "갱신상태": status, "채점방식": eval_mode, "관측소소스": obs_source,
        "채점모델수": scored_cnt, "오류건수": err_cnt, "1위모델": top_model, "1위가중치(%)": round(top_weight, 1)
    }])
    row.to_csv(AUDIT_LOG_FILE, mode='a' if os.path.exists(AUDIT_LOG_FILE) else 'w', header=not os.path.exists(AUDIT_LOG_FILE), index=False, encoding='utf-8-sig')

# ==========================================
# [6. v8.1 11대 모델 통합 파이프라인]
# ==========================================
def run_station_pipeline(matched_key, progress_cb=None, log_cb=None):
    auto_discover_jeju_aws_stations(KMA_AUTH_KEY)
    stn_info = JEJU_STATIONS.get(matched_key, JEJU_STATIONS["함덕"])
    now_kst = datetime.now(KST).replace(tzinfo=None)
    today_date = now_kst.date()
    today_str = today_date.strftime("%Y-%m-%d")
    yesterday_date = today_date - timedelta(days=1)
    yesterday_str = yesterday_date.strftime("%Y%m%d")

    SYSTEM_HEALTH["last_check_time"] = now_kst.strftime("%Y-%m-%d %H:%M:%S")
    run_error_count = 0

    history = {}
    if os.path.exists(WEIGHT_HISTORY_FILE):
        try:
            with open(WEIGHT_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = {}

    active_models, model_max_hours, prob_source_map = {}, {}, {}
    om_keys = [k for k in MODELS.keys() if k != "nasa_geos_fp"]

    if progress_cb:
        progress_cb(15, "🌍 [1/4] Open-Meteo 10대 글로벌 모델 로드 중...")

    df_mod, loaded_om, mh_om, om_status, om_msg, om_ms = fetch_open_meteo_resilient(
        stn_info["lat"], stn_info["lon"], om_keys, past_days=2, now_kst=now_kst
    )
    if df_mod.empty:
        run_error_count += 1

    for idx, (mc, prob_info) in enumerate(loaded_om.items(), 1):
        mm = MODELS[mc]
        active_models[mc] = mm["name"]
        model_max_hours[mc] = mh_om[mc]
        prob_source_map[mc] = prob_info
        SYSTEM_HEALTH["model_registry"][mc] = {"status": "✅ 사용중", "max_hours": mh_om[mc], "latency_s": round(om_ms/10000.0, 2), "note": f"정상 ({prob_info})"}
        if log_cb:
            log_cb(f"✔ [{idx:02d}/11] {mm['name']} ({mm['res']}) 로드 완료 ({mh_om[mc]}h, {prob_info})")

    SYSTEM_HEALTH["api_status"]["Open-Meteo (10대 통합 NWP)"] = {
        "status": om_status, "code": "200 OK" if not df_mod.empty else "ERR",
        "latency_ms": om_ms, "errors": 0 if not df_mod.empty else 1, "msg": om_msg
    }

    if progress_cb:
        progress_cb(55, "🛰️ [2/4] 미국(NASA GEOS-FP) 순수 예보 호출 중...")
    df_nasa, nasa_sec, nasa_note, nasa_has_rain = fetch_nasa_geos_forecast_only(stn_info["lat"], stn_info["lon"])

    if not df_nasa.empty and not df_mod.empty:
        df_na = df_nasa.reindex(df_mod.index)
        nasa_fh = int(df_na[df_na.index >= now_kst]["temperature_2m_nasa_geos_fp"].notna().sum())
        if nasa_fh >= 12:
            df_mod = pd.concat([df_mod, df_na], axis=1)
            active_models["nasa_geos_fp"] = MODELS["nasa_geos_fp"]["name"]
            model_max_hours["nasa_geos_fp"] = nasa_fh
            prob_source_map["nasa_geos_fp"] = "강수환산" if nasa_has_rain else "강수미수신(NaN)"
            SYSTEM_HEALTH["model_registry"]["nasa_geos_fp"] = {
                "status": "✅ 사용중" if nasa_has_rain else "⚠️ 강수누락",
                "max_hours": nasa_fh, "latency_s": round(nasa_sec, 2), "note": nasa_note
            }
            if log_cb:
                log_cb(f"✔ [11/11] 미국(NASA GEOS) 로드 ({nasa_fh}h, {nasa_note})")
    else:
        SYSTEM_HEALTH["model_registry"]["nasa_geos_fp"] = {
            "status": "⚠️ 미수신(제외)", "max_hours": 0, "latency_s": round(nasa_sec, 2), "note": nasa_note
        }
        if log_cb:
            log_cb(f"⚠️ [11/11] 미국(NASA GEOS) {nasa_note}")

    snap_saved_now = save_forecast_snapshot(matched_key, today_date, now_kst, df_mod, active_models)

    if progress_cb:
        progress_cb(75, f"📡 [3/4] 기상청 '{matched_key}' 어제 실측 & 스냅샷 대조 중...")
    df_obs_y, obs_source_msg = fetch_kma_obs_v8(stn_info, yesterday_str, KMA_AUTH_KEY)

    if df_mod.empty or not active_models:
        if progress_cb:
            progress_cb(100, "⚠️ 기상 모델 응답 지연")
        return {
            "region_name": matched_key, "station_info": stn_info, "now_kst": now_kst,
            "yesterday_str": yesterday_str, "obs_source_msg": obs_source_msg, "eval_mode_msg": "모델 응답 지연",
            "obs_summary": {}, "weight_status_msg": f"⚠️ Open-Meteo 응답 지연 ({om_msg})",
            "active_models": {}, "model_max_hours": {}, "prob_source_map": {},
            "metrics": {}, "cum_weights": {}, "prev_weights": {}, "daily_weights": {},
            "df_future_all": pd.DataFrame(), "run_error_count": run_error_count + 1, "snap_saved_now": False
        }

    df_snap_y, eval_mode_msg = load_yesterday_snapshot(matched_key, yesterday_date)
    if not df_snap_y.empty:
        df_eval = df_snap_y
        eval_candidates = list(MODELS.keys())
    else:
        y_start, y_end = pd.Timestamp(yesterday_date), pd.Timestamp(yesterday_date) + pd.Timedelta(hours=23)
        df_eval = df_mod[(df_mod.index >= y_start) & (df_mod.index <= y_end)]
        eval_candidates = [k for k in active_models.keys() if k != "nasa_geos_fp"]

    daily_weights, metrics = evaluate_4way_fair(df_eval, eval_candidates, df_obs_y)
    for k in MODELS.keys():
        if k not in metrics:
            metrics[k] = {'scored': False, 'eval_hours': 0, 'temp_mae': 0.0, 'temp_bias': 0.0, 'rain_mae': 0.0, 'prob_brier': 0.0, 'wind_mae': 0.0, 'wind_bias': 0.0, 'score_temp': 0.0, 'score_prob': 0.0, 'score_rain': 0.0, 'score_wind': 0.0}

    obs_summary = {}
    if not df_obs_y.empty:
        obs_summary = {
            "obs_hours": len(df_obs_y),
            "temp_avg": round(float(df_obs_y['obs_temp'].mean()), 1),
            "temp_min": round(float(df_obs_y['obs_temp'].min()), 1),
            "temp_max": round(float(df_obs_y['obs_temp'].max()), 1),
            "rain_sum": round(float(df_obs_y['obs_rain'].sum()), 1),
            "wind_avg": round(float(df_obs_y['obs_wind'].mean()), 1),
            "wind_max": round(float(df_obs_y['obs_wind'].max()), 1)
        }

    all_keys = list(MODELS.keys())
    if matched_key not in history or history[matched_key].get("rule_version") != WEIGHT_RULE_VERSION:
        eq_init = {m: round(100.0 / len(all_keys), 2) for m in all_keys}
        aligned_prev, new_cum = update_weights_fair(eq_init, daily_weights, all_keys)
        history[matched_key] = {
            "created_date": today_str, "last_updated": today_str, "update_count": 1,
            "rule_version": WEIGHT_RULE_VERSION, "eval_mode": eval_mode_msg,
            "prev_weights": aligned_prev, "daily_weights": daily_weights,
            "cumulative_weights": new_cum, "latest_metrics": metrics,
            "obs_source": obs_source_msg, "obs_summary": obs_summary, "last_error_count": run_error_count
        }
        top_m, top_w = max(new_cum.items(), key=lambda x: x[1])
        write_audit_log(matched_key, yesterday_str, "v8신규초기화", eval_mode_msg[:18], obs_source_msg, len(daily_weights), run_error_count, MODELS[top_m]["name"], top_w)
        weight_status_msg = f"🆕 v8.1 공정 가중치 장부 생성 완료 ({len(daily_weights)}개 모델 실측 채점)"
    else:
        reg_data = history[matched_key]
        if reg_data.get("last_updated") != today_str:
            aligned_prev, new_cum = update_weights_fair(reg_data.get("cumulative_weights", {}), daily_weights, all_keys)
            reg_data["update_count"] = reg_data.get("update_count", 1) + 1
            reg_data["prev_weights"] = aligned_prev
            reg_data["daily_weights"] = daily_weights
            reg_data["cumulative_weights"] = new_cum
            reg_data["last_updated"] = today_str
            reg_data["eval_mode"] = eval_mode_msg
            top_m, top_w = max(new_cum.items(), key=lambda x: x[1])
            write_audit_log(matched_key, yesterday_str, f"갱신({reg_data['update_count']}일차)", eval_mode_msg[:18], obs_source_msg, len(daily_weights), run_error_count, MODELS[top_m]["name"], top_w)
            weight_status_msg = f"🔄 일일 가중치 갱신 완료 (누적 {reg_data['update_count']}일 차, {len(daily_weights)}개 채점)"
        else:
            weight_status_msg = f"🔒 오늘({today_str}) 가중치 갱신 완료 상태 (누적 {reg_data['update_count']}일 차 유지)"

        reg_data["latest_metrics"] = metrics
        reg_data["obs_source"] = obs_source_msg
        reg_data["obs_summary"] = obs_summary
        reg_data["last_error_count"] = run_error_count

    cum_weights = history[matched_key]["cumulative_weights"]
    prev_weights = history[matched_key]["prev_weights"]

    if progress_cb:
        progress_cb(92, "✨ [4/4] 영점 보정 및 7일 앙상블 예보 합성 중...")

    ens_t, ens_p, ens_r, ens_w, conf_list, act_cnt = [], [], [], [], [], []
    for _, row in df_mod.iterrows():
        t_sum, t_wsum = 0.0, 0.0
        p_sum, p_wsum = 0.0, 0.0
        r_sum, r_wsum = 0.0, 0.0
        w_sum, w_wsum = 0.0, 0.0
        cnt = 0
        hour_temps, hour_probs = [], []

        for m_code in active_models.keys():
            w = cum_weights.get(m_code, 9.0) / 100.0
            raw_t = row.get(f"temperature_2m_{m_code}", np.nan)
            if pd.notnull(raw_t):
                mt = metrics.get(m_code, {})
                t_bias = np.clip(mt.get('temp_bias', 0.0) * 0.7, -3.0, 3.0) if mt.get('scored') else 0.0
                w_bias = np.clip(mt.get('wind_bias', 0.0) * 0.5, -3.0, 3.0) if mt.get('scored') else 0.0
                corr_t = raw_t - t_bias
                t_sum += corr_t * w; t_wsum += w
                hour_temps.append(corr_t)
                cnt += 1

                raw_w = row.get(f"wind_speed_10m_{m_code}", np.nan)
                if pd.notnull(raw_w):
                    w_sum += max(0.0, raw_w - w_bias) * w; w_wsum += w

                raw_p = row.get(f"precipitation_probability_{m_code}", np.nan)
                if pd.notnull(raw_p):
                    p_sum += raw_p * w; p_wsum += w
                    hour_probs.append(raw_p)

                raw_r = row.get(f"precipitation_{m_code}", np.nan)
                if pd.notnull(raw_r):
                    r_sum += raw_r * w; r_wsum += w

        if t_wsum > 0:
            ens_t.append(t_sum / t_wsum)
            ens_p.append(p_sum / p_wsum if p_wsum > 0 else 0.0)
            ens_r.append(r_sum / r_wsum if r_wsum > 0 else 0.0)
            ens_w.append(w_sum / w_wsum if w_wsum > 0 else 0.0)
            t_std = float(np.std(hour_temps)) if len(hour_temps) > 1 else 0.5
            p_std = float(np.std(hour_probs)) if len(hour_probs) > 1 else 10.0
            conf_list.append(np.clip(98.0 - (t_std * 11.0) - (p_std * 0.35), 40.0, 98.0))
        else:
            ens_t.append(np.nan); ens_p.append(np.nan); ens_r.append(np.nan); ens_w.append(np.nan); conf_list.append(np.nan)
        act_cnt.append(cnt)

    df_mod['Ensemble_Temp'], df_mod['Ensemble_Prob'], df_mod['Ensemble_Rain'] = ens_t, ens_p, ens_r
    df_mod['Ensemble_Wind'], df_mod['Confidence'], df_mod['Active_Models'] = ens_w, conf_list, act_cnt
    df_future_all = df_mod[df_mod.index >= now_kst].dropna(subset=['Ensemble_Temp'])
    history[matched_key]["avg_confidence_48h"] = round(float(df_future_all.head(48)['Confidence'].mean()), 1) if not df_future_all.empty else 0.0

    with open(WEIGHT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    if progress_cb:
        progress_cb(100, "✅ 완료!")

    return {
        "region_name": matched_key, "station_info": stn_info, "now_kst": now_kst,
        "yesterday_str": yesterday_str, "obs_source_msg": obs_source_msg, "eval_mode_msg": eval_mode_msg,
        "obs_summary": obs_summary, "weight_status_msg": weight_status_msg,
        "active_models": active_models, "model_max_hours": model_max_hours, "prob_source_map": prob_source_map,
        "metrics": metrics, "cum_weights": cum_weights, "prev_weights": prev_weights,
        "daily_weights": daily_weights, "df_future_all": df_future_all,
        "run_error_count": run_error_count, "snap_saved_now": snap_saved_now
    }

def format_status_report():
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    sec_kma = f"🔒연동됨 (길이 {len(KMA_AUTH_KEY)}자)" if KMA_AUTH_KEY else "⚠️미설정(Secrets 등록필요)"
    api_lines = [f"• {k}: {v['status']} ({v['latency_ms']}ms) - {v['msg']}" for k, v in SYSTEM_HEALTH["api_status"].items()]
    snaps = list_saved_snapshots()
    snap_txt = ", ".join(snaps) if snaps else "아직 저장된 스냅샷 없음"

    reg_lines = []
    if os.path.exists(WEIGHT_HISTORY_FILE):
        with open(WEIGHT_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        for reg, info in list(history.items())[:16]:
            w_map = info.get("cumulative_weights", {})
            top_m, top_w = max(w_map.items(), key=lambda x: x[1]) if w_map else ("-", 0)
            reg_lines.append(f"• {reg}({info.get('update_count',1)}일차): 1위 {MODELS.get(top_m,{}).get('name',top_m)}({top_w:.1f}%) | {info.get('obs_source','-')[:22]}")

    return (
        f"🖥️ [제주 기상 앙상블 v8.1 상태 & 스냅샷 대시보드]\n"
        f"🕒 조회시각: {now_str} | 기상청키: {sec_kma}\n"
        f"⚖️ 가중치: 강수확률 90% + 강수량 5% + 기온 2.5% + 풍속 2.5% (상한 30%)\n"
        f"📸 보관된 예보 스냅샷: {snap_txt}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📡 [1. API 통신 건전성]\n" + "\n".join(api_lines) + "\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📍 [2. 지역별 1위 모델 & 관측소스]\n" + ("\n".join(reg_lines) if reg_lines else "• 지역명을 먼저 조회해 주세요.")
    )

# ==========================================
# [7. GitHub Actions 무인 배치 모드 (--auto-update)]
# ==========================================
if "--auto-update" in sys.argv:
    print(f"🚀 [무인 스케줄러 v8.1] 시작 | KMA_AUTH_KEY 감지: {bool(KMA_AUTH_KEY)} (길이: {len(KMA_AUTH_KEY)}자)")
    auto_discover_jeju_aws_stations(KMA_AUTH_KEY)
    for idx, reg in enumerate(list(JEJU_STATIONS.keys())[:16], 1):
        try:
            r_obj = run_station_pipeline(reg)
            print(f" -> [{idx:02d}/16] '{reg}' | 채점모델:{len(r_obj['daily_weights'])}개 | {r_obj['obs_source_msg']} | {r_obj['weight_status_msg']}")
            time.sleep(0.8)
        except Exception as e:
            print(f" ⚠️ [{idx:02d}/16] '{reg}' 예외 발생: {e}")
    print("✅ [무인 스케줄러 v8.1 완료]")
    sys.exit(0)

# ==========================================
# [8. 스마트폰 모바일 웹 앱 화면 (Streamlit UI)]
# ==========================================
import streamlit as st

st.set_page_config(page_title="제주 맞춤형 기상 앙상블 v8.1", page_icon="🌴", layout="wide")
auto_discover_jeju_aws_stations(KMA_AUTH_KEY)

st.markdown("### 🌴 제주도 지역별 맞춤형 기상 앙상블 v8.1 (스냅샷 정석 채점 · 공정 가중치)")
st.caption("가중치: 강수확률 90% + 강수량 5% + 기온 2.5% + 풍속 2.5% | 스냅샷 익일 대조 · 결측 동결 · 누적 상한 30%")

if not KMA_AUTH_KEY:
    st.warning("⚠️ Streamlit Secrets에 `KMA_AUTH_KEY`가 등록되지 않아 기상청 실측 채점이 비활성화되었습니다. 우측 하단 Manage app → Settings → Secrets에 등록해 주세요.")

tab_forecast, tab_status = st.tabs([
    "🌤️ 지역별 맞춤 예보 (터치 조회)",
    "🖥️ 스냅샷 · 다중 인자 상태 대시보드"
])

with tab_forecast:
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected_region = st.selectbox(
            "📍 날씨를 확인할 제주 지역(관측소)을 선택하세요",
            options=list(JEJU_STATIONS.keys()), index=0,
            format_func=lambda x: f"{x} (지점 {JEJU_STATIONS[x]['stn']} · {JEJU_STATIONS[x]['desc']})"
        )
    with col_btn:
        st.write(""); st.write("")
        run_clicked = st.button("🔍 실시간 예보 & 채점 실행", type="primary", use_container_width=True)

    if run_clicked or "last_result" not in st.session_state or st.session_state.get("last_region") != selected_region:
        if run_clicked:
            _OM_CACHE.clear()
            _NASA_CACHE.clear()
        with st.status(f"🔄 [{selected_region} 관측소] 11대 모델 로드 · 스냅샷 검증 · 공정 채점 중...", expanded=True) as status_box:
            p_bar = st.progress(0, text="초기화 중...")
            log_box = st.empty()
            logs = []
            result = run_station_pipeline(
                selected_region,
                progress_cb=lambda pct, txt: p_bar.progress(min(100, pct), text=txt),
                log_cb=lambda msg: (logs.append(msg), log_box.code("\n".join(logs[-6:])))
            )
            st.session_state["last_result"] = result
            st.session_state["last_region"] = selected_region
            status_box.update(label=f"✅ [{selected_region}] v8.1 분석 완료!", state="complete", expanded=False)

    res = st.session_state["last_result"]
    df_fut = res["df_future_all"]
    df_24h = df_fut.head(24)

    if df_24h.empty or not res["cum_weights"]:
        st.error(f"❌ **[{selected_region}] 기상 모델 데이터를 불러오지 못했습니다.** ({res['weight_status_msg']})")
        st.info("💡 10초 뒤 우측의 **[🔍 실시간 예보 & 채점 실행]** 버튼을 다시 눌러주세요.")
        st.code(format_status_report())
    else:
        sorted_w = sorted(res["cum_weights"].items(), key=lambda x: x[1], reverse=True)
        top_m_code, top_m_w = sorted_w[0]
        top_m_name = MODELS.get(top_m_code, {}).get("name", top_m_code)

        st.success(f"✅ **{res['eval_mode_msg']}** | 관측소: {res['obs_source_msg']} | {res['weight_status_msg']}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("☔ 24시간 강수확률 / 강수량", f"최대 {df_24h['Ensemble_Prob'].max():.0f} %", f"누적 {df_24h['Ensemble_Rain'].sum():.1f} mm")
        m2.metric("🌡️ 향후 24시간 기온", f"{df_24h['Ensemble_Temp'].min():.1f} ~ {df_24h['Ensemble_Temp'].max():.1f} ℃", f"현재 {df_24h['Ensemble_Temp'].iloc[0]:.1f} ℃")
        m3.metric("💨 24시간 최대 풍속", f"{df_24h['Ensemble_Wind'].max():.1f} m/s", f"평균 신뢰도 {df_24h['Confidence'].mean():.0f}%")
        m4.metric("🏆 강수 특화 1위 모델", top_m_name, f"가중치 {top_m_w:.1f}%")

        st.markdown(f"#### 📊 1. [{selected_region}] 11대 기상모델 공정 채점 성적표 (강수확률 90% · 강수량 5% · 기온 2.5% · 풍속 2.5%)")
        score_rows = []
        for rk, (mc, fw) in enumerate(sorted_w, 1):
            me = res["metrics"].get(mc, {})
            pw = res["prev_weights"].get(mc, 9.1)
            diff = fw - pw
            is_scored = me.get("scored", False)
            m_hrs = res["model_max_hours"].get(mc, 0)
            p_src = res["prob_source_map"].get(mc, "미수신")
            score_rows.append({
                "순위": f"{rk}위",
                "기상 모델(예보범위)": f"{MODELS[mc]['name']} ({m_hrs}h)" if m_hrs > 0 else f"{MODELS[mc]['name']} (미수신)",
                "채점 상태": f"✅ 채점({me.get('eval_hours',0)}h·{p_src})" if is_scored else "⏸️ 제외(가중치 동결)",
                "확률오차(BS, 90%)": f"{me['prob_brier']:.3f} ({me.get('score_prob',0):.1f}점)" if is_scored else "-",
                "강수오차(5%)": f"±{me['rain_mae']:.2f}mm ({me.get('score_rain',0):.1f}점)" if is_scored else "-",
                "기온/풍속오차(각2.5%)": f"±{me['temp_mae']:.1f}℃(보정{-me['temp_bias']*0.7:+.1f}℃) / ±{me['wind_mae']:.1f}m/s" if is_scored else "-",
                "이전(%)": f"{pw:.1f}%",
                "당일(%)": f"{res['daily_weights'][mc]:.1f}%" if mc in res["daily_weights"] else "동결",
                "최종 가중치(%)": round(fw, 1),
                "변동": f"▲+{diff:.1f}%" if diff > 0 else (f"▼{diff:.1f}%" if diff < 0 else "0.0%")
            })
        st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

        st.markdown(f"#### 🕒 2. [{selected_region}] 향후 48시간 시간별 상세 예측표")
        rows_48h = []
        for idx, row in df_fut.head(48).iterrows():
            c_val = row['Confidence']
            c_tag = "🟢매우높음" if c_val >= 85 else ("🟡보통" if c_val >= 70 else "🟠모델혼조")
            rows_48h.append({
                "예측 시각(KST)": idx.strftime("%m/%d(%a) %H시"),
                "강수확률(%)": f"{row['Ensemble_Prob']:.0f} %",
                "예상 강수량(mm)": f"🌧️ {row['Ensemble_Rain']:.1f} mm" if row['Ensemble_Rain'] >= 0.1 else "- (0.0 mm)",
                "맞춤 기온(℃)": f"{row['Ensemble_Temp']:.1f} ℃",
                "예상 풍속(m/s)": f"💨 {row['Ensemble_Wind']:.1f}" if row['Ensemble_Wind'] >= 8.0 else f"{row['Ensemble_Wind']:.1f}",
                "예보 신뢰도(참여모델)": f"{c_val:.0f}% ({c_tag}, {int(row['Active_Models'])}/{len(res['active_models'])}개)"
            })
        st.dataframe(pd.DataFrame(rows_48h), use_container_width=True, hide_index=True, height=400)

        df_after_48h = df_fut.iloc[48:168].copy()
        if not df_after_48h.empty:
            st.markdown(f"#### 📅 3. [{selected_region}] 48시간 이후 ~ 7일 중기 예측표 (12시간 단위)")
            df_after_48h['group_key'] = df_after_48h.index.strftime("%m월 %d일 ") + np.where(df_after_48h.index.hour < 12, "오전(00~12시)", "오후(12~24시)")
            rows_7d = []
            for g_key, grp in df_after_48h.groupby('group_key', sort=False):
                r_sum = grp['Ensemble_Rain'].sum()
                rows_7d.append({
                    "날짜 및 시간대": g_key,
                    "최대 강수확률": f"{grp['Ensemble_Prob'].max():.0f} %",
                    "12시간 누적강수": f"☔ {r_sum:.1f} mm" if r_sum >= 0.5 else f"- ({r_sum:.1f} mm)",
                    "기온(최저~최고 / 평균)": f"{grp['Ensemble_Temp'].min():.1f}~{grp['Ensemble_Temp'].max():.1f}℃ (평균 {grp['Ensemble_Temp'].mean():.1f}℃)",
                    "최대 풍속": f"{grp['Ensemble_Wind'].max():.1f} m/s",
                    "평균 신뢰도(모델수)": f"{grp['Confidence'].mean():.0f}% ({int(round(grp['Active_Models'].mean()))}/{len(res['active_models'])}개 합산)"
                })
            st.dataframe(pd.DataFrame(rows_7d), use_container_width=True, hide_index=True)

with tab_status:
    st.markdown("#### 🖥️ 스냅샷 아카이브 · 다중 인자 상태 & 보안 대시보드")
    st.code(format_status_report())
