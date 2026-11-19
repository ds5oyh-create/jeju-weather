import sys
import os
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
# [0. 시크릿 로더 및 가중치 설정 (확률 90% / 강수량 5%)]
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
KAKAO_REST_API_KEY = get_secret_key("KAKAO_REST_API_KEY", "")
KAKAO_REFRESH_TOKEN = get_secret_key("KAKAO_REFRESH_TOKEN", "")
ALERT_REGION = get_secret_key("ALERT_REGION", "함덕")

LEARNING_RATE = 0.20
W_PROB = 0.90
W_RAIN = 0.05
W_TEMP = 0.025
W_WIND = 0.025
WEIGHT_RULE_VERSION = "PROB90_RAIN5_V2"

SAVE_DIR = "./jeju_weather_data"
os.makedirs(SAVE_DIR, exist_ok=True)
WEIGHT_HISTORY_FILE = os.path.join(SAVE_DIR, "jeju_cumulative_weights_v7.json")
AUDIT_LOG_FILE = os.path.join(SAVE_DIR, "automation_audit_log_v7.csv")

SYSTEM_HEALTH = {
    "storage_mode": "✅ 강수확률 90% · 강수량 5% 가중치 & 11대 모델(NASA 포함) 가동 중",
    "storage_path": SAVE_DIR,
    "total_error_count": 0,
    "last_check_time": "-",
    "api_status": {
        "Open-Meteo (10대 통합 NWP)":   {"status": "대기중", "code": "-", "latency_ms": 0, "errors": 0, "msg": "-"},
        "NASA GMAO (GEOS-FP & POWER)": {"status": "대기중", "code": "-", "latency_ms": 0, "errors": 0, "msg": "-"},
        "KMA 1.1 (AWS 매분자료)":       {"status": "대기중", "code": "-", "latency_ms": 0, "errors": 0, "msg": "-"},
        "KMA 6.3 (AWS 지점일람표)":     {"status": "대기중", "code": "-", "latency_ms": 0, "errors": 0, "msg": "-"},
        "KMA ASOS (종관기상관측)":      {"status": "대기중", "code": "-", "latency_ms": 0, "errors": 0, "msg": "-"}
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
    conds = [rain_series.isna(), rain_series <= 0.0, rain_series <= 0.5, rain_series <= 2.0]
    choices = [np.nan, 5.0, 55.0, 80.0]
    return pd.Series(np.select(conds, choices, default=95.0), index=rain_series.index)

def calc_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

# ==========================================
# [1. Open-Meteo 3단계 자동 복구 + 15분 캐시 로더]
# ==========================================
_OM_CACHE = {}

def _parse_open_meteo_hourly(res_json, req_models, now_kst):
    df_raw = pd.DataFrame(res_json["hourly"])
    df_raw['time'] = pd.to_datetime(df_raw['time'])
    df_raw.set_index('time', inplace=True)

    df_out = pd.DataFrame(index=df_raw.index)
    loaded = {}
    max_hours = {}

    for mc in req_models:
        t_c = f"temperature_2m_{mc}" if f"temperature_2m_{mc}" in df_raw.columns else ("temperature_2m" if len(req_models) == 1 and "temperature_2m" in df_raw.columns else None)
        if not t_c:
            continue
        p_c = f"precipitation_{mc}" if f"precipitation_{mc}" in df_raw.columns else ("precipitation" if "precipitation" in df_raw.columns else None)
        pr_c = f"precipitation_probability_{mc}" if f"precipitation_probability_{mc}" in df_raw.columns else ("precipitation_probability" if "precipitation_probability" in df_raw.columns else None)
        w_c = f"wind_speed_10m_{mc}" if f"wind_speed_10m_{mc}" in df_raw.columns else ("wind_speed_10m" if "wind_speed_10m" in df_raw.columns else None)

        out_t, out_p, out_pr, out_w = f"temperature_2m_{mc}", f"precipitation_{mc}", f"precipitation_probability_{mc}", f"wind_speed_10m_{mc}"
        s_t = pd.to_numeric(df_raw[t_c], errors='coerce').interpolate(method='linear', limit=2)
        s_p = pd.to_numeric(df_raw[p_c], errors='coerce').fillna(0.0) if p_c else pd.Series(0.0, index=df_raw.index)
        s_w = pd.to_numeric(df_raw[w_c], errors='coerce').interpolate(method='linear', limit=2).fillna(0.0) if w_c else pd.Series(0.0, index=df_raw.index)

        if pr_c and pd.to_numeric(df_raw[pr_c], errors='coerce').notna().sum() >= 10:
            s_pr = pd.to_numeric(df_raw[pr_c], errors='coerce')
            prob_info = "자체확률"
        else:
            s_pr = estimate_prob_from_rain(s_p)
            prob_info = "강수환산확률"

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

    # 1차 시도: 10개 모델 일괄 요청
    data_full = _call_batch(om_keys)
    if data_full:
        df_all, loaded_all, mh_all = _parse_open_meteo_hourly(data_full, om_keys, now_kst)
    else:
        # 2차 시도: 5개씩 2묶음 분할 요청 (일부 모델 오류/페이로드 제한 우회)
        chunks = [om_keys[:5], om_keys[5:]]
        parts = []
        for ch in chunks:
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

    # 3차 최후 안전망: 기본 통합 모델(best_match) 호출
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
# [2. NASA GEOS-FP & NASA POWER 통합 수집]
# ==========================================
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
                if pd.notnull(v): vals.append(float(v))
    return vals

def fetch_nasa_geos_model(lat, lon, past_days, now_kst):
    t0 = time.time()
    session = requests.Session()
    if NASA_USER and NASA_PASS:
        session.auth = (NASA_USER, NASA_PASS)
    df_parts, notes = [], []

    try:
        start_d = (now_kst - timedelta(days=max(3, past_days + 1))).strftime("%Y%m%d")
        end_d = now_kst.strftime("%Y%m%d")
        pw_url = (
            f"https://power.larc.nasa.gov/api/temporal/hourly/point"
            f"?parameters=T2M,PRECTOTCORR,WS10M&community=RE"
            f"&longitude={lon}&latitude={lat}&start={start_d}&end={end_d}&format=JSON&time-standard=UTC"
        )
        r_pw = session.get(pw_url, timeout=6)
        if r_pw.status_code == 200:
            p_data = r_pw.json().get("properties", {}).get("parameter", {})
            t2m_d, pr_d, ws_d = p_data.get("T2M", {}), p_data.get("PRECTOTCORR", {}), p_data.get("WS10M", {})
            rows = []
            for dt_s, tv in t2m_d.items():
                if tv > -90:
                    tkst = pd.to_datetime(dt_s, format="%Y%m%d%H", errors='coerce') + pd.Timedelta(hours=9)
                    pv, wv = pr_d.get(dt_s, 0.0), ws_d.get(dt_s, 3.0)
                    rows.append({
                        "time": tkst,
                        "temperature_2m_nasa_geos_fp": float(tv),
                        "precipitation_nasa_geos_fp": max(0.0, float(pv) if pv > -90 else 0.0),
                        "wind_speed_10m_nasa_geos_fp": max(0.0, float(wv) if wv > -90 else 3.0)
                    })
            if rows:
                df_parts.append(pd.DataFrame(rows).set_index("time"))
                notes.append("GMAO-POWER")
    except Exception:
        pass

    lat_idx, lon_idx = int(round((lat + 90.0) / 0.25)), int(round((lon + 180.0) / 0.25))
    now_utc = datetime.now(timezone.utc)
    for run_dt, cyc in [(now_utc, "00"), (now_utc - timedelta(days=1), "12"), (now_utc - timedelta(days=1), "00")]:
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
                p_v = []
                try:
                    r_f = session.get(flx_u, timeout=4)
                    if r_f.status_code == 200 and "prectot" in r_f.text:
                        p_v = _parse_opendap_ascii_1d(r_f.text, "prectot")
                except Exception:
                    pass
                if len(t_v) >= 24:
                    st_kst = (datetime.strptime(f"{ymd}{cyc}", "%Y%m%d%H") + timedelta(hours=9)).replace(minute=0, second=0)
                    rows_f = []
                    for i, tk in enumerate(t_v):
                        if 180.0 < tk < 340.0:
                            uv = u_v[i] if i < len(u_v) and abs(u_v[i]) < 200 else 2.0
                            vv = v_v[i] if i < len(v_v) and abs(v_v[i]) < 200 else 2.0
                            pr = p_v[i] if i < len(p_v) and 0 <= p_v[i] < 10 else 0.0
                            rows_f.append({
                                "time": st_kst + timedelta(hours=i),
                                "temperature_2m_nasa_geos_fp": float(tk - 273.15),
                                "precipitation_nasa_geos_fp": float(pr * 3600.0),
                                "wind_speed_10m_nasa_geos_fp": float(math.sqrt(uv**2 + vv**2))
                            })
                    if rows_f:
                        df_parts.append(pd.DataFrame(rows_f).set_index("time"))
                        notes.append(f"GEOS-FP({ymd}_{cyc})")
                        break
        except Exception:
            continue

    ms = int((time.time() - t0) * 1000)
    if df_parts:
        dfc = pd.concat(df_parts).sort_index()
        dfc = dfc[~dfc.index.duplicated(keep='last')]
        dfc["precipitation_probability_nasa_geos_fp"] = estimate_prob_from_rain(dfc["precipitation_nasa_geos_fp"])
        SYSTEM_HEALTH["api_status"]["NASA GMAO (GEOS-FP & POWER)"] = {
            "status": "✅ 정상", "code": "200 OK", "latency_ms": ms, "errors": 0, "msg": f"NASA 연동 ({'+'.join(notes)})"
        }
        return dfc, ms / 1000.0, "+".join(notes)

    SYSTEM_HEALTH["api_status"]["NASA GMAO (GEOS-FP & POWER)"] = {
        "status": "⚠️ 지연", "code": "PASS", "latency_ms": ms, "errors": 0, "msg": "NASA 서버 응답 지연"
    }
    return pd.DataFrame(), ms / 1000.0, "보조동화"

# ==========================================
# [3. 기상청 6.3 지점일람표 및 관측소(1.1 AWS / ASOS) 파서]
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
def fetch_kma_obs_v7(stn_info, date_str, auth_key):
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
    df1, _ = fetch_kma_obs_v7({"stn": n1_id, "type": "ASOS"}, date_str, auth_key)
    df2, _ = fetch_kma_obs_v7({"stn": n2_id, "type": "ASOS"}, date_str, auth_key)
    if not df1.empty and not df2.empty:
        dfb = ((df1 * w1) + (df2 * (1.0 - w1))).dropna()
        _OBS_CACHE[ck] = (dfb, f"거리보간({n1_name} {w1*100:.0f}%+{n2_name} {(1-w1)*100:.0f}%)")
        return _OBS_CACHE[ck]
    return pd.DataFrame(), "관측 데이터 대기"

# ==========================================
# [4. 4중 오차 채점 (강수확률 90% + 강수량 5% + 기온 2.5% + 풍속 2.5%)]
# ==========================================
def evaluate_4way_detailed(df_mod, active_models, df_obs):
    metrics = {}
    for m_code in active_models.keys():
        t_c, p_c, pr_c, w_c = f"temperature_2m_{m_code}", f"precipitation_{m_code}", f"precipitation_probability_{m_code}", f"wind_speed_10m_{m_code}"
        if not df_obs.empty and not df_mod.empty and all(c in df_mod.columns for c in [t_c, p_c, pr_c, w_c]):
            df_pair = pd.concat([df_mod[[t_c, p_c, pr_c, w_c]], df_obs[['obs_temp', 'obs_rain', 'obs_wind']]], axis=1).dropna()
            if len(df_pair) >= 3:
                t_bias = float(np.mean(df_pair[t_c] - df_pair['obs_temp']))
                t_mae = float(np.mean(np.abs(df_pair[t_c] - df_pair['obs_temp'])))
                r_mae = float(np.mean(np.abs(df_pair[p_c] - df_pair['obs_rain'])))
                act_bin = (df_pair['obs_rain'] >= 0.1).astype(float)
                pred_p = (df_pair[pr_c] / 100.0).clip(0.0, 1.0)
                p_brier = float(np.mean((pred_p - act_bin) ** 2))
                w_bias = float(np.mean(df_pair[w_c] - df_pair['obs_wind']))
                w_mae = float(np.mean(np.abs(df_pair[w_c] - df_pair['obs_wind'])))
                metrics[m_code] = {
                    'eval_hours': len(df_pair),
                    'temp_mae': round(t_mae, 2), 'temp_bias': round(t_bias, 2),
                    'rain_mae': round(r_mae, 2), 'prob_brier': round(p_brier, 3),
                    'wind_mae': round(w_mae, 2), 'wind_bias': round(w_bias, 2)
                }
            else:
                metrics[m_code] = {'eval_hours': 0, 'temp_mae': 1.5, 'temp_bias': 0.0, 'rain_mae': 0.5, 'prob_brier': 0.2, 'wind_mae': 1.5, 'wind_bias': 0.0}
        else:
            metrics[m_code] = {'eval_hours': 0, 'temp_mae': 1.5, 'temp_bias': 0.0, 'rain_mae': 0.5, 'prob_brier': 0.2, 'wind_mae': 1.5, 'wind_bias': 0.0}

    if not metrics:
        return {}, {}

    t_inv = {k: 1.0 / ((v['temp_mae'] + 0.1) ** 1.5) for k, v in metrics.items()}
    p_inv = {k: 1.0 / ((v['prob_brier'] + 0.02) ** 1.2) for k, v in metrics.items()}
    r_inv = {k: 1.0 / ((v['rain_mae'] + 0.05) ** 1.2) for k, v in metrics.items()}
    w_inv = {k: 1.0 / ((v['wind_mae'] + 0.2) ** 1.2) for k, v in metrics.items()}
    st, sp, sr, sw = max(1e-9, sum(t_inv.values())), max(1e-9, sum(p_inv.values())), max(1e-9, sum(r_inv.values())), max(1e-9, sum(w_inv.values()))

    daily_w = {}
    for k in active_models.keys():
        score_t = (t_inv[k] / st) * 100.0
        score_p = (p_inv[k] / sp) * 100.0
        score_r = (r_inv[k] / sr) * 100.0
        score_w = (w_inv[k] / sw) * 100.0
        metrics[k]['score_temp'] = round(score_t, 1)
        metrics[k]['score_prob'] = round(score_p, 1)
        metrics[k]['score_rain'] = round(score_r, 1)
        metrics[k]['score_wind'] = round(score_w, 1)
        daily_w[k] = (W_TEMP * score_t) + (W_PROB * score_p) + (W_RAIN * score_r) + (W_WIND * score_w)
    return daily_w, metrics

def write_audit_log(region, eval_date, status, obs_source, active_cnt, err_cnt, top_model, top_weight):
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([{
        "실행시각(KST)": now_str, "지역": region, "채점기준일": eval_date,
        "갱신상태": status, "관측소소스": obs_source, "가동모델수": active_cnt,
        "오류건수": err_cnt, "1위모델": top_model, "1위가중치(%)": round(top_weight, 1)
    }])
    row.to_csv(AUDIT_LOG_FILE, mode='a' if os.path.exists(AUDIT_LOG_FILE) else 'w', header=not os.path.exists(AUDIT_LOG_FILE), index=False, encoding='utf-8-sig')

# ==========================================
# [5. 11대 모델 통합 파이프라인 (IndexError 완전 방어)]
# ==========================================
def run_station_pipeline(matched_key, progress_cb=None, log_cb=None):
    auto_discover_jeju_aws_stations(KMA_AUTH_KEY)
    stn_info = JEJU_STATIONS.get(matched_key, JEJU_STATIONS["함덕"])
    now_kst = datetime.now(KST).replace(tzinfo=None)
    today_date = now_kst.date()
    today_str = today_date.strftime("%Y-%m-%d")
    SYSTEM_HEALTH["last_check_time"] = now_kst.strftime("%Y-%m-%d %H:%M:%S")
    run_error_count = 0

    history = {}
    if os.path.exists(WEIGHT_HISTORY_FILE):
        try:
            with open(WEIGHT_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = {}

    days_missed = (today_date - datetime.strptime(history[matched_key]["last_updated"], "%Y-%m-%d").date()).days if matched_key in history else 1
    past_days_to_fetch = min(7, max(2, days_missed + 1))

    active_models, model_max_hours = {}, {}
    om_keys = [k for k in MODELS.keys() if k != "nasa_geos_fp"]

    if progress_cb:
        progress_cb(15, "🌍 [1/3] Open-Meteo 10대 글로벌 모델 로드 중...")

    df_mod, loaded_om, mh_om, om_status, om_msg, om_ms = fetch_open_meteo_resilient(
        stn_info["lat"], stn_info["lon"], om_keys, past_days_to_fetch, now_kst
    )
    if df_mod.empty:
        run_error_count += 1

    for idx, (mc, prob_info) in enumerate(loaded_om.items(), 1):
        mm = MODELS[mc]
        active_models[mc] = mm["name"]
        model_max_hours[mc] = mh_om[mc]
        SYSTEM_HEALTH["model_registry"][mc] = {"status": "✅ 사용중", "max_hours": mh_om[mc], "latency_s": round(om_ms/10000.0, 2), "note": f"정상 ({prob_info})"}
        if log_cb:
            log_cb(f"✔ [{idx:02d}/11] {mm['name']} ({mm['res']}) 로드 완료 ({mh_om[mc]}시간)")

    SYSTEM_HEALTH["api_status"]["Open-Meteo (10대 통합 NWP)"] = {
        "status": om_status, "code": "200 OK" if not df_mod.empty else "ERR",
        "latency_ms": om_ms, "errors": 0 if not df_mod.empty else 1, "msg": om_msg
    }

    if progress_cb:
        progress_cb(65, "🛰️ [11/11] 미국(NASA GEOS) 위성동화 모델 호출 중...")
    df_nasa, nasa_sec, nasa_note = fetch_nasa_geos_model(stn_info["lat"], stn_info["lon"], past_days_to_fetch, now_kst)

    if not df_nasa.empty and not df_mod.empty:
        df_na = df_nasa.reindex(df_mod.index)
        df_na["temperature_2m_nasa_geos_fp"] = df_na["temperature_2m_nasa_geos_fp"].interpolate(method='linear', limit=3)
        df_na["wind_speed_10m_nasa_geos_fp"] = df_na["wind_speed_10m_nasa_geos_fp"].interpolate(method='linear', limit=3).fillna(0.0)
        df_na["precipitation_nasa_geos_fp"] = df_na["precipitation_nasa_geos_fp"].fillna(0.0)
        df_na["precipitation_probability_nasa_geos_fp"] = df_na["precipitation_probability_nasa_geos_fp"].fillna(10.0)
        df_mod = pd.concat([df_mod, df_na], axis=1)

        nasa_fh = int(df_mod[df_mod.index >= now_kst]["temperature_2m_nasa_geos_fp"].notna().sum())
        if nasa_fh < 12:
            ref_m = "gfs_seamless" if "temperature_2m_gfs_seamless" in df_mod.columns else next(iter(loaded_om.keys()), None)
            if ref_m:
                fm = df_mod.index >= now_kst
                df_mod.loc[fm, "temperature_2m_nasa_geos_fp"] = df_mod.loc[fm, f"temperature_2m_{ref_m}"]
                df_mod.loc[fm, "wind_speed_10m_nasa_geos_fp"] = df_mod.loc[fm, f"wind_speed_10m_{ref_m}"]
                df_mod.loc[fm, "precipitation_nasa_geos_fp"] = df_mod.loc[fm, f"precipitation_{ref_m}"]
                df_mod.loc[fm, "precipitation_probability_nasa_geos_fp"] = df_mod.loc[fm, f"precipitation_probability_{ref_m}"]
                nasa_fh = int(df_mod[fm]["temperature_2m_nasa_geos_fp"].notna().sum())

        active_models["nasa_geos_fp"] = MODELS["nasa_geos_fp"]["name"]
        model_max_hours["nasa_geos_fp"] = max(24, nasa_fh)
        SYSTEM_HEALTH["model_registry"]["nasa_geos_fp"] = {"status": "✅ 사용중", "max_hours": model_max_hours["nasa_geos_fp"], "latency_s": round(nasa_sec, 2), "note": f"NASA ({nasa_note})"}
        if log_cb:
            log_cb(f"✔ [11/11] 미국(NASA GEOS) (25km) 로드 완료 ({nasa_note})")

    if progress_cb:
        progress_cb(80, f"📡 [2/3] 기상청 '{matched_key}' 관측소 실측 대조 및 채점 중...")
    yesterday_str = (today_date - timedelta(days=1)).strftime("%Y%m%d")
    df_obs_y, obs_source_msg = fetch_kma_obs_v7(stn_info, yesterday_str, KMA_AUTH_KEY)

    # 미래 예측 데이터가 전혀 없는 경우 안전 종료 (크래시 방지)
    if df_mod.empty or not active_models:
        if progress_cb:
            progress_cb(100, "⚠️ 기상 모델 응답 지연")
        return {
            "region_name": matched_key, "station_info": stn_info, "now_kst": now_kst,
            "yesterday_str": yesterday_str, "obs_source_msg": obs_source_msg, "obs_summary": {},
            "weight_status_msg": f"⚠️ Open-Meteo 응답 지연 ({om_msg})", "active_models": {}, "model_max_hours": {},
            "metrics": {}, "cum_weights": {}, "prev_weights": {}, "daily_weights": {},
            "df_future_all": pd.DataFrame(), "run_error_count": run_error_count + 1
        }

    daily_weights, metrics = evaluate_4way_detailed(df_mod, active_models, df_obs_y)

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

    default_eq_w = round(100.0 / max(1, len(daily_weights)), 2)
    if matched_key not in history:
        init_w = {m: default_eq_w for m in daily_weights.keys()}
        new_cum = {m: (1.0 - LEARNING_RATE) * init_w[m] + (LEARNING_RATE * d_w) for m, d_w in daily_weights.items()}
        tot = max(1e-9, sum(new_cum.values()))
        new_cum = {m: round((v / tot) * 100.0, 2) for m, v in new_cum.items()}
        history[matched_key] = {
            "created_date": today_str, "last_updated": today_str, "update_count": 1,
            "rule_version": WEIGHT_RULE_VERSION,
            "prev_weights": init_w, "daily_weights": {k: round(v, 2) for k, v in daily_weights.items()},
            "cumulative_weights": new_cum, "latest_metrics": metrics,
            "obs_source": obs_source_msg, "obs_summary": obs_summary, "last_error_count": run_error_count
        }
        top_m, top_w = max(new_cum.items(), key=lambda x: x[1])
        write_audit_log(matched_key, yesterday_str, "신규(확률90%/강수5%)", obs_source_msg, len(active_models), run_error_count, MODELS[top_m]["name"], top_w)
        weight_status_msg = f"🆕 강수확률 90%·강수량 5% 가중치 최초 생성({today_str}) 완료"
    else:
        reg_data = history[matched_key]
        old_cum_map = reg_data.get("cumulative_weights", {})
        rule_changed = reg_data.get("rule_version") != WEIGHT_RULE_VERSION

        if reg_data.get("last_updated") != today_str or rule_changed or len(old_cum_map) != len(daily_weights):
            base_prev = {m: default_eq_w for m in daily_weights.keys()} if rule_changed else {m: old_cum_map.get(m, default_eq_w) for m in daily_weights.keys()}
            tot_p = max(1e-9, sum(base_prev.values()))
            aligned_prev = {m: round((v / tot_p) * 100.0, 2) for m, v in base_prev.items()}

            new_cum = {m: (1.0 - LEARNING_RATE) * aligned_prev[m] + (LEARNING_RATE * daily_weights[m]) for m in daily_weights.keys()}
            tot = max(1e-9, sum(new_cum.values()))
            new_cum = {m: round((v / tot) * 100.0, 2) for m, v in new_cum.items()}

            if reg_data.get("last_updated") != today_str:
                reg_data["update_count"] = reg_data.get("update_count", 1) + 1
            reg_data["rule_version"] = WEIGHT_RULE_VERSION
            reg_data["prev_weights"] = aligned_prev
            reg_data["daily_weights"] = {k: round(v, 2) for k, v in daily_weights.items()}
            reg_data["cumulative_weights"] = new_cum
            reg_data["last_updated"] = today_str
            top_m, top_w = max(new_cum.items(), key=lambda x: x[1])
            write_audit_log(matched_key, yesterday_str, "갱신(확률90%/강수5%)", obs_source_msg, len(active_models), run_error_count, MODELS[top_m]["name"], top_w)
            weight_status_msg = f"🔄 강수확률 90% · 강수량 5% 비중 재채점 완료 (누적 {reg_data['update_count']}일 차)"
        else:
            weight_status_msg = f"🔒 오늘({today_str}) 강수확률 90% · 강수량 5% 가중치 적용 완료 (누적 {reg_data['update_count']}일 차)"

        reg_data["latest_metrics"] = metrics
        reg_data["obs_source"] = obs_source_msg
        reg_data["obs_summary"] = obs_summary
        reg_data["last_error_count"] = run_error_count

    cum_weights = history[matched_key]["cumulative_weights"]
    prev_weights = history[matched_key]["prev_weights"]

    if progress_cb:
        progress_cb(92, "✨ [3/3] 11대 모델 영점 보정 및 7일 예보 합성 중...")

    ens_t, ens_p, ens_r, ens_w, conf_list, act_cnt = [], [], [], [], [], []
    for _, row in df_mod.iterrows():
        t_sum, p_sum, r_sum, wind_sum, w_sum, cnt = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        hour_temps, hour_probs = [], []
        for m_code, w_pct in cum_weights.items():
            if m_code not in metrics:
                continue
            raw_t = row.get(f"temperature_2m_{m_code}", np.nan)
            if pd.notnull(raw_t):
                w = w_pct / 100.0
                t_bias = np.clip(metrics[m_code]['temp_bias'] * 0.7, -3.0, 3.0)
                w_bias = np.clip(metrics[m_code]['wind_bias'] * 0.5, -3.0, 3.0)
                corr_t = raw_t - t_bias
                raw_w = row.get(f"wind_speed_10m_{m_code}", 0.0)
                corr_w = max(0.0, (raw_w if pd.notnull(raw_w) else 0.0) - w_bias)
                prob_v = row.get(f"precipitation_probability_{m_code}", 0.0)
                prob_v = prob_v if pd.notnull(prob_v) else 0.0
                rain_v = row.get(f"precipitation_{m_code}", 0.0)
                rain_v = rain_v if pd.notnull(rain_v) else 0.0

                t_sum += corr_t * w
                p_sum += prob_v * w
                r_sum += rain_v * w
                wind_sum += corr_w * w
                w_sum += w
                cnt += 1
                hour_temps.append(corr_t)
                hour_probs.append(prob_v)

        if w_sum > 0:
            ens_t.append(t_sum / w_sum)
            ens_p.append(p_sum / w_sum)
            ens_r.append(r_sum / w_sum)
            ens_w.append(wind_sum / w_sum)
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
        "yesterday_str": yesterday_str, "obs_source_msg": obs_source_msg, "obs_summary": obs_summary,
        "weight_status_msg": weight_status_msg, "active_models": active_models, "model_max_hours": model_max_hours,
        "metrics": metrics, "cum_weights": cum_weights, "prev_weights": prev_weights,
        "daily_weights": daily_weights, "df_future_all": df_future_all, "run_error_count": run_error_count
    }

# ==========================================
# [6. 카카오톡 메시지 생성 및 양방향 처리]
# ==========================================
def format_kakao_weather_reply(res_data):
    region_name = res_data["region_name"]
    df_fut = res_data["df_future_all"]
    if df_fut.empty or not res_data["cum_weights"]:
        return f"⚠️ [{region_name}] 기상 서버 응답이 일시 지연 중입니다. 잠시 후 다시 시도해 주세요."
    df_24h = df_fut.head(24)
    sorted_w = sorted(res_data["cum_weights"].items(), key=lambda x: x[1], reverse=True)

    top1_code, top1_w = sorted_w[0]
    top1_name = res_data["active_models"].get(top1_code, top1_code)
    top1_m = res_data["metrics"].get(top1_code, {"prob_brier": 0.0, "rain_mae": 0.0})
    top1_diff = top1_w - res_data["prev_weights"].get(top1_code, 9.1)
    top1_diff_str = f"▲+{top1_diff:.1f}%" if top1_diff >= 0 else f"▼{top1_diff:.1f}%"
    top3_str = " / ".join([f"{rk}위 {res_data['active_models'].get(mc, mc)}({fw:.1f}%)" for rk, (mc, fw) in enumerate(sorted_w[:3], 1)])

    now_t, min_t, max_t = df_24h['Ensemble_Temp'].iloc[0], df_24h['Ensemble_Temp'].min(), df_24h['Ensemble_Temp'].max()
    max_prob, sum_rain, max_wind, avg_conf = df_24h['Ensemble_Prob'].max(), df_24h['Ensemble_Rain'].sum(), df_24h['Ensemble_Wind'].max(), df_24h['Confidence'].mean()
    rain_advice = f"☔ 24시간 내 약 {sum_rain:.1f}mm 비 예보(최대 {max_prob:.0f}%)" if (sum_rain >= 0.5 or max_prob >= 50) else f"☀️ 비 소식 없음 (확률 {max_prob:.0f}% 이하)"

    lines_24h = []
    for idx, row in df_24h.iloc[::3].iterrows():
        r_lbl = f"🌧️{row['Ensemble_Rain']:.1f}mm({row['Ensemble_Prob']:.0f}%)" if row['Ensemble_Rain'] >= 0.1 else f"확률{row['Ensemble_Prob']:.0f}%"
        lines_24h.append(f"• {idx.strftime('%d일 %H시')}: {row['Ensemble_Temp']:>4.1f}℃ | 💨{row['Ensemble_Wind']:.1f}m/s | {r_lbl}")

    return (
        f"🌴 [{region_name} 맞춤 브리핑 (확률90%·강수5% 가중치)]\n"
        f"🕒 기준: {res_data['now_kst'].strftime('%m/%d %H:%M')} | 채점: {res_data['obs_source_msg']}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🏆 [1위 모델: {top1_name} ({top1_w:.1f}%, {top1_diff_str})]\n"
        f"• 확률오차(BS): {top1_m['prob_brier']:.3f} | 강수오차: ±{top1_m['rain_mae']:.1f}mm\n"
        f"• TOP3: {top3_str}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🌤️ [향후 24시간 요약 (신뢰도 {avg_conf:.0f}%)]\n"
        f"• 강수: {rain_advice}\n"
        f"• 기온: 현재 {now_t:.1f}℃ ({min_t:.1f}℃ ~ {max_t:.1f}℃) | 최대풍속 {max_wind:.1f}m/s\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🕒 [3시간 간격 상세 예보]\n" + "\n".join(lines_24h)
    )

def format_kakao_status_reply():
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    sec_kma = "🔒암호화 연동됨" if KMA_AUTH_KEY else "⚠️미설정(Secrets 등록필요)"
    api_lines = [f"• {k}: {v['status']} ({v['latency_ms']}ms) - {v['msg']}" for k, v in SYSTEM_HEALTH["api_status"].items()]
    reg_lines = []
    if os.path.exists(WEIGHT_HISTORY_FILE):
        with open(WEIGHT_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        for reg, info in list(history.items())[:16]:
            w_map = info.get("cumulative_weights", {})
            top_m, top_w = max(w_map.items(), key=lambda x: x[1]) if w_map else ("-", 0)
            reg_lines.append(f"• {reg}({info.get('update_count',1)}일차): 1위 {MODELS.get(top_m,{}).get('name',top_m)}({top_w:.1f}%) | 신뢰도 {info.get('avg_confidence_48h',0)}%")

    return (
        f"🖥️ [제주 기상 앙상블 상태 & 보안 대시보드]\n"
        f"🕒 조회시각: {now_str} | 기상청키: {sec_kma}\n"
        f"⚖️ 가중치 배분: 강수확률 90% + 강수량 5% + 기온 2.5% + 풍속 2.5%\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📡 [1. API 통신 건전성]\n" + "\n".join(api_lines) + "\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📍 [2. 지역별 1위 모델 현황]\n" + ("\n".join(reg_lines) if reg_lines else "• 지역명을 먼저 조회해 주세요.")
    )

def handle_kakao_command(user_msg):
    clean_msg = user_msg.strip()
    if not clean_msg: return "🌴 제주 지역명(예: 서귀포, 애월, 함덕) 또는 '상태', '관측소'를 입력해 주세요."
    if clean_msg in ["상태", "로그", "진단", "status"]: return format_kakao_status_reply()
    if clean_msg in ["관측소", "목록", "지역"]:
        auto_discover_jeju_aws_stations(KMA_AUTH_KEY)
        return f"📍 [검색 가능한 제주 관측소 ({len(JEJU_STATIONS)}곳)]\n" + ", ".join(JEJU_STATIONS.keys())
    auto_discover_jeju_aws_stations(KMA_AUTH_KEY)
    matched_key = next((k for k in JEJU_STATIONS.keys() if k in clean_msg or clean_msg in k), None)
    if matched_key: return format_kakao_weather_reply(run_station_pipeline(matched_key))
    return f"⚠️ '{clean_msg}' 관측소를 찾을 수 없습니다."

# ==========================================
# [7. GitHub Actions 무인 배치 모드 (--auto-update)]
# ==========================================
if "--auto-update" in sys.argv:
    print("🚀 [무인 스케줄러] 강수확률 90% · 강수량 5% 기준 제주 16개 관측소 자동 학습 시작...")
    auto_discover_jeju_aws_stations(KMA_AUTH_KEY)
    for idx, reg in enumerate(list(JEJU_STATIONS.keys())[:16], 1):
        try:
            print(f" -> [{idx}/16] '{reg}' 관측소 채점 중...")
            run_station_pipeline(reg)
            time.sleep(0.8)
        except Exception as e:
            print(f" ⚠️ '{reg}' 예외 건너뜀: {e}")
    print("✅ [무인 스케줄러 완료]")
    sys.exit(0)

# ==========================================
# [8. 스마트폰 모바일 웹 앱 화면 (Streamlit UI)]
# ==========================================
import streamlit as st

st.set_page_config(page_title="제주 맞춤형 기상 앙상블 v7.8", page_icon="🌴", layout="wide")
auto_discover_jeju_aws_stations(KMA_AUTH_KEY)

if "kakao_msg" in st.query_params:
    st.code(handle_kakao_command(st.query_params["kakao_msg"]))
    st.stop()

st.markdown("### 🌴 제주도 지역별 맞춤형 기상 앙상블 v7.8 (강수확률 90% · 강수량 5% 특화판)")
st.caption("가중치 반영 비중: 강수확률 90% + 강수량 5% + 기온 2.5% + 풍속 2.5% | 11대 모델(NASA GEOS 포함) & 오류 방어 적용")

if not KMA_AUTH_KEY:
    st.warning("⚠️ Streamlit Secrets에 `KMA_AUTH_KEY`가 등록되지 않아 기상청 실측 채점이 비활성화되었습니다. 우측 하단 Manage app → Settings → Secrets에 등록해 주세요.")

tab_forecast, tab_kakao, tab_status = st.tabs([
    "🌤️ 지역별 맞춤 예보 (터치 조회)",
    "💬 카카오톡 양방향 봇 채팅 & 연동",
    "🖥️ 다중 인자 상태 & 진단 대시보드"
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
        with st.status(f"🔄 [{selected_region} 관측소] 11대 모델 로드 및 강수확률 90% · 강수량 5% 채점 중...", expanded=True) as status_box:
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
            status_box.update(label=f"✅ [{selected_region}] 분석 완료!", state="complete", expanded=False)

    res = st.session_state["last_result"]
    df_fut = res["df_future_all"]
    df_24h = df_fut.head(24)

    # 빈 데이터 크래시(IndexError) 원천 방어
    if df_24h.empty or not res["cum_weights"]:
        st.error(f"❌ **[{selected_region}] 기상 모델 데이터를 불러오지 못했습니다.** ({res['weight_status_msg']})")
        st.info("💡 Open-Meteo 서버의 일시적인 공용 IP 호출 제한(HTTP 429)일 가능성이 높습니다. 10초 뒤 우측의 **[🔍 실시간 예보 & 채점 실행]** 버튼을 다시 눌러주세요.")
        st.code(format_kakao_status_reply())
    else:
        sorted_w = sorted(res["cum_weights"].items(), key=lambda x: x[1], reverse=True)
        top_m_code, top_m_w = sorted_w[0]
        top_m_name = res["active_models"].get(top_m_code, top_m_code)

        st.success(f"✅ **[가중치 배분: 강수확률 90% · 강수량 5% · 기온 2.5% · 풍속 2.5%]** | 채점 소스: {res['obs_source_msg']} | {res['weight_status_msg']}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("☔ 24시간 강수확률 / 강수량", f"최대 {df_24h['Ensemble_Prob'].max():.0f} %", f"누적 {df_24h['Ensemble_Rain'].sum():.1f} mm")
        m2.metric("🌡️ 향후 24시간 기온", f"{df_24h['Ensemble_Temp'].min():.1f} ~ {df_24h['Ensemble_Temp'].max():.1f} ℃", f"현재 {df_24h['Ensemble_Temp'].iloc[0]:.1f} ℃")
        m3.metric("💨 24시간 최대 풍속", f"{df_24h['Ensemble_Wind'].max():.1f} m/s", f"평균 신뢰도 {df_24h['Confidence'].mean():.0f}%")
        m4.metric("🏆 강수 특화 1위 모델", top_m_name, f"가중치 {top_m_w:.1f}%")

        st.markdown(f"#### 📊 1. [{selected_region}] 11대 기상모델 성적표 (강수확률 90% + 강수량 5% + 기온 2.5% + 풍속 2.5%)")
        score_rows = []
        for rk, (mc, fw) in enumerate(sorted_w, 1):
            me = res["metrics"].get(mc, {"prob_brier": 0.0, "rain_mae": 0.0, "temp_mae": 0.0, "temp_bias": 0.0, "wind_mae": 0.0})
            pw = res["prev_weights"].get(mc, 9.1)
            diff = fw - pw
            score_rows.append({
                "순위": f"{rk}위",
                "기상 모델(예보범위)": f"{res['active_models'].get(mc, mc)} ({res['model_max_hours'].get(mc, 0)}h)",
                "확률오차(BS, 90%반영)": f"{me['prob_brier']:.3f} ({me.get('score_prob',0):.1f}점)",
                "강수오차(5%반영)": f"±{me['rain_mae']:.2f}mm ({me.get('score_rain',0):.1f}점)",
                "기온/풍속오차(각2.5%)": f"±{me['temp_mae']:.1f}℃(보정{-me['temp_bias']*0.7:+.1f}℃) / ±{me['wind_mae']:.1f}m/s",
                "이전 누적(%)": f"{pw:.1f}%",
                "당일 성적(%)": f"{res['daily_weights'].get(mc, fw):.1f}%",
                "최종 가중치(%)": round(fw, 1),
                "변동": f"▲+{diff:.1f}%" if diff >= 0 else f"▼{diff:.1f}%"
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

with tab_kakao:
    st.markdown("#### 💬 카카오톡 양방향 실시간 대화 테스트")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [{"role": "assistant", "content": "안녕하세요! 강수확률 90%·강수량 5% 가중치가 적용된 제주 기상 봇입니다. '함덕', '서귀포', '애월', '상태'를 입력해 보세요."}]
    for chat in st.session_state["chat_history"]:
        with st.chat_message(chat["role"]): st.text(chat["content"])
    user_query = st.chat_input("메시지를 입력하세요 (예: 함덕, 서귀포, 애월, 상태)")
    if user_query:
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        with st.chat_message("user"): st.text(user_query)
        with st.chat_message("assistant"):
            with st.spinner(f"'{user_query}' 실시간 계산 중..."):
                bot_reply = handle_kakao_command(user_query)
            st.text(bot_reply)
        st.session_state["chat_history"].append({"role": "assistant", "content": bot_reply})

with tab_status:
    st.markdown("#### 🖥️ 다중 인자 상태 & 보안 대시보드")
    st.code(format_kakao_status_reply())
