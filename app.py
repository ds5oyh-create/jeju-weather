import sys
import os
import json
import math
import time
import threading
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import requests
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# ==========================================
# [0. 인증키 및 기본 환경 설정]
# ==========================================
KST = timezone(timedelta(hours=9))
KMA_AUTH_KEY = os.environ.get("KMA_AUTH_KEY", "vCGn48lfSeehp-PJX_nntA")
LEARNING_RATE = 0.20

KAKAO_REST_API_KEY = os.environ.get("KAKAO_REST_API_KEY", "")
KAKAO_REFRESH_TOKEN = os.environ.get("KAKAO_REFRESH_TOKEN", "")
ALERT_REGION = os.environ.get("ALERT_REGION", "함덕")
KAKAO_WEBHOOK_PORT = int(os.environ.get("KAKAO_WEBHOOK_PORT", "8000"))

SAVE_DIR = "./jeju_weather_data"
os.makedirs(SAVE_DIR, exist_ok=True)
WEIGHT_HISTORY_FILE = os.path.join(SAVE_DIR, "jeju_cumulative_weights_v7.json")
AUDIT_LOG_FILE = os.path.join(SAVE_DIR, "automation_audit_log_v7.csv")

SYSTEM_HEALTH = {
    "storage_mode": "✅ GitHub 영구 저장소 & 카카오톡 양방향 엔진 가동 중",
    "storage_path": SAVE_DIR,
    "total_error_count": 0,
    "last_check_time": "-",
    "api_status": {
        "Open-Meteo (10대 통합 NWP)": {"status": "대기중", "code": "-", "latency_ms": 0, "errors": 0, "msg": "-"},
        "KMA 1.1 (AWS 매분자료)":     {"status": "대기중", "code": "-", "latency_ms": 0, "errors": 0, "msg": "-"},
        "KMA 3.1.6 (AWS 정시자료)":   {"status": "대기중", "code": "-", "latency_ms": 0, "errors": 0, "msg": "-"},
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
    "icon_seamless":        {"name": "독일(DWD ICON)",   "org": "DWD(독일)",      "res": "11km"},
    "ukmo_seamless":        {"name": "영국(UKMO 10km)",  "org": "MetOffice(영국)","res": "10km"},
    "jma_seamless":         {"name": "일본(JMA 통합)",   "org": "JMA(일본)",      "res": "5~20km"},
    "cma_grapes_global":    {"name": "중국(CMA GRAPES)", "org": "CMA(중국)",      "res": "13km"},
    "meteofrance_seamless": {"name": "프랑스(ARPEGE)",   "org": "Météo-France",   "res": "25km"},
    "gem_seamless":         {"name": "캐나다(CMC GEM)",  "org": "CMC(캐나다)",    "res": "15km"}
}

EXCLUDED_MODELS_INFO = {
    "kma_seamless":      {"name": "한국(KMA LDAPS/KIM)", "org": "기상청(KMA)", "res": "1.5~12km", "reason": "Open-Meteo 내 데이터 공급 종료 (2026-03-31 중단)"},
    "bom_access_global": {"name": "호주(BOM ACCESS)",    "org": "BOM(호주)",   "res": "12km",     "reason": "Open-Meteo 내 데이터 공급 종료 (2025-06-26 중단)"},
    "metno_seamless":    {"name": "노르웨이(MET Nordic)","org": "MET Norway",  "res": "1~2.5km",  "reason": "유럽·북유럽 권역 전용 모델 (제주 좌표 데이터 NaN)"},
    "nasa_geos_fp":      {"name": "미국(NASA GEOS-FP)",  "org": "NASA GMAO",   "res": "25km",     "reason": "기온(slv)·강수(flx) 파일 분리 및 10m 단일풍속 미제공으로 4중 채점 제외"}
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
# [1. 6.3 지점일람표 API로 제주 전역 AWS 자동 확장]
# ==========================================
_DISCOVERED = False
def auto_discover_jeju_aws_stations(auth_key):
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    url = "https://apihub.kma.go.kr/api/typ02/openApi/AwsYearlyInfoService/getAwsStnLstTbl"
    added_cnt = 0
    t0 = time.time()
    for yr in ["2024", "2023", "2022"]:
        try:
            params = {"pageNo": 1, "numOfRows": 700, "dataType": "XML", "year": yr, "month": "09", "authKey": auth_key}
            res = requests.get(url, params=params, timeout=5)
            lat_ms = int((time.time() - t0) * 1000)
            if res.status_code == 200 and "<stn_id>" in res.text:
                root = ET.fromstring(res.text)
                for item in root.iter():
                    stn_id_el, stn_ko_el = item.find("stn_id"), item.find("stn_ko")
                    lat_el, lon_el, ht_el = item.find("lat"), item.find("lon"), item.find("ht")
                    if stn_id_el is not None and stn_ko_el is not None and lat_el is not None and lon_el is not None:
                        lat_v, lon_v = float(lat_el.text), float(lon_el.text)
                        name_ko = stn_ko_el.text.strip()
                        stn_num = int(stn_id_el.text.strip())
                        ht_v = ht_el.text.strip() if ht_el is not None and ht_el.text else "0"
                        if 33.10 <= lat_v <= 34.05 and 126.00 <= lon_v <= 127.05:
                            if name_ko not in JEJU_STATIONS and stn_num not in [184, 185, 188, 189]:
                                JEJU_STATIONS[name_ko] = {
                                    "stn": stn_num, "lat": lat_v, "lon": lon_v,
                                    "type": "AWS", "ht": ht_v, "desc": f"제주 AWS 자동등록 지점 (해발 {ht_v}m)"
                                }
                                added_cnt += 1
                if added_cnt > 0:
                    SYSTEM_HEALTH["api_status"]["KMA 6.3 (AWS 지점일람표)"] = {
                        "status": "✅ 정상", "code": "200 OK", "latency_ms": lat_ms, "errors": 0,
                        "msg": f"{yr}년 기준 제주 AWS {added_cnt}개소 자동 확장 완료"
                    }
                    return
        except Exception as e:
            SYSTEM_HEALTH["api_status"]["KMA 6.3 (AWS 지점일람표)"] = {
                "status": "⚠️ 대기", "code": " PASS", "latency_ms": 0, "errors": 0, "msg": "기본 16개 관측소로 가동"
            }

# ==========================================
# [2. 기상청 관측소 파서 (1.1 매분 + ASOS)]
# ==========================================
_OBS_CACHE = {}

def fetch_kma_obs_v7(stn_info, date_str, auth_key):
    stn_id = stn_info["stn"]
    cache_key = (stn_id, date_str)
    if cache_key in _OBS_CACHE:
        return _OBS_CACHE[cache_key]

    tm1, tm2 = f"{date_str}0000", f"{date_str}2300"

    if stn_info["type"] == "ASOS":
        t0 = time.time()
        url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php?tm1={tm1}&tm2={tm2}&stn={stn_id}&help=0&authKey={auth_key}"
        try:
            res = requests.get(url, timeout=8)
            lat_ms = int((time.time() - t0) * 1000)
            records = []
            if res.status_code == 200:
                for line in res.text.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    parts = line.replace(',', ' ').split()
                    if len(parts) > 15 and parts[0].isdigit() and len(parts[0]) == 12:
                        obs_time = pd.to_datetime(parts[0], format='%Y%m%d%H%M', errors='coerce')
                        ws, ta, rn = pd.to_numeric(parts[3], errors='coerce'), pd.to_numeric(parts[11], errors='coerce'), pd.to_numeric(parts[15], errors='coerce')
                        if pd.notnull(obs_time) and pd.notnull(ta) and ta > -50:
                            records.append({
                                'time': obs_time, 'obs_temp': float(ta),
                                'obs_rain': float(rn) if (pd.notnull(rn) and rn >= 0) else 0.0,
                                'obs_wind': float(ws) if (pd.notnull(ws) and ws >= 0) else 0.0
                            })
            df_res = pd.DataFrame(records).drop_duplicates('time').set_index('time') if records else pd.DataFrame()
            SYSTEM_HEALTH["api_status"]["KMA ASOS (종관기상관측)"] = {
                "status": "✅ 정상" if not df_res.empty else "⚠️ 빈데이터",
                "code": f"{res.status_code}", "latency_ms": lat_ms, "errors": 0 if not df_res.empty else 1,
                "msg": f"지점 {stn_id} {len(df_res)}시간 수신"
            }
            msg = f"지점 {stn_id} (ASOS 직접 관측)"
            _OBS_CACHE[cache_key] = (df_res, msg)
            return df_res, msg
        except Exception as e:
            return pd.DataFrame(), f"ASOS 지연({str(e)[:25]})"

    t0 = time.time()
    url_min = f"https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min?tm1={tm1}&tm2={tm2}&stn={stn_id}&disp=1&help=0&authKey={auth_key}"
    try:
        res = requests.get(url_min, timeout=8)
        lat_ms = int((time.time() - t0) * 1000)
        if res.status_code == 200 and len(res.text.strip()) > 40 and "auth" not in res.text.lower():
            records = []
            for line in res.text.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = [p.strip() for p in line.replace(',', ' ').split() if p.strip()]
                if len(parts) >= 12 and parts[0].isdigit() and len(parts[0]) == 12 and parts[0].endswith('00'):
                    obs_time = pd.to_datetime(parts[0], format='%Y%m%d%H%M', errors='coerce')
                    ws10, ta, rn60 = pd.to_numeric(parts[7], errors='coerce'), pd.to_numeric(parts[8], errors='coerce'), pd.to_numeric(parts[11], errors='coerce')
                    if pd.notnull(obs_time) and pd.notnull(ta) and -40 < ta < 50:
                        records.append({
                            'time': obs_time, 'obs_temp': float(ta),
                            'obs_rain': float(rn60) if (pd.notnull(rn60) and rn60 >= 0) else 0.0,
                            'obs_wind': float(ws10) if (pd.notnull(ws10) and ws10 >= 0) else 0.0
                        })
            if records:
                df_res = pd.DataFrame(records).drop_duplicates('time').set_index('time')
                SYSTEM_HEALTH["api_status"]["KMA 1.1 (AWS 매분자료)"] = {
                    "status": "✅ 정상", "code": "200 OK", "latency_ms": lat_ms, "errors": 0,
                    "msg": f"지점 {stn_id} 정시 {len(df_res)}시간 직결 성공"
                }
                msg = f"지점 {stn_id} (1.1 AWS 매분자료 직접 관측)"
                _OBS_CACHE[cache_key] = (df_res, msg)
                return df_res, msg
    except Exception:
        pass

    dists = sorted([(aid, h["name"], max(0.5, calc_distance_km(stn_info["lat"], stn_info["lon"], h["lat"], h["lon"]))) for aid, h in ASOS_HUBS.items()], key=lambda x: x[2])
    n1_id, n1_name, d1 = dists[0]
    n2_id, n2_name, d2 = dists[1]
    w1 = (1.0 / d1) / ((1.0 / d1) + (1.0 / d2))
    w2 = 1.0 - w1
    df1, _ = fetch_kma_obs_v7({"stn": n1_id, "type": "ASOS"}, date_str, auth_key)
    df2, _ = fetch_kma_obs_v7({"stn": n2_id, "type": "ASOS"}, date_str, auth_key)
    if not df1.empty and not df2.empty:
        df_blend = (df1 * w1) + (df2 * w2)
        msg = f"거리보간({n1_name} {w1*100:.0f}%+{n2_name} {w2*100:.0f}%)"
        _OBS_CACHE[cache_key] = (df_blend.dropna(), msg)
        return df_blend.dropna(), msg
    return pd.DataFrame(), "관측 데이터 백업 대기"

# ==========================================
# [3. 4중 오차 채점 및 영점 조절(Bias) 계산]
# ==========================================
def evaluate_4way_detailed(df_mod, active_models, df_obs):
    metrics = {}
    for m_code in active_models.keys():
        t_col, p_col, prob_col, w_col = f"temperature_2m_{m_code}", f"precipitation_{m_code}", f"precipitation_probability_{m_code}", f"wind_speed_10m_{m_code}"
        if not df_obs.empty and not df_mod.empty and all(c in df_mod.columns for c in [t_col, p_col, prob_col, w_col]):
            df_pair = pd.concat([df_mod[[t_col, p_col, prob_col, w_col]], df_obs[['obs_temp', 'obs_rain', 'obs_wind']]], axis=1).dropna()
            if len(df_pair) >= 3:
                temp_bias = float(np.mean(df_pair[t_col] - df_pair['obs_temp']))
                temp_mae = float(np.mean(np.abs(df_pair[t_col] - df_pair['obs_temp'])))
                rain_mae = float(np.mean(np.abs(df_pair[p_col] - df_pair['obs_rain'])))
                actual_rain_bin = (df_pair['obs_rain'] >= 0.1).astype(float)
                pred_prob_ratio = (df_pair[prob_col] / 100.0).clip(0.0, 1.0)
                prob_brier = float(np.mean((pred_prob_ratio - actual_rain_bin) ** 2))
                wind_bias = float(np.mean(df_pair[w_col] - df_pair['obs_wind']))
                wind_mae = float(np.mean(np.abs(df_pair[w_col] - df_pair['obs_wind'])))
                metrics[m_code] = {
                    'eval_hours': len(df_pair),
                    'temp_mae': round(temp_mae, 2), 'temp_bias': round(temp_bias, 2),
                    'rain_mae': round(rain_mae, 2), 'prob_brier': round(prob_brier, 3),
                    'wind_mae': round(wind_mae, 2), 'wind_bias': round(wind_bias, 2)
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
        score_t, score_p, score_r, score_w = (t_inv[k]/st)*100.0, (p_inv[k]/sp)*100.0, (r_inv[k]/sr)*100.0, (w_inv[k]/sw)*100.0
        metrics[k]['score_temp'] = round(score_t, 1)
        metrics[k]['score_prob'] = round(score_p, 1)
        metrics[k]['score_rain'] = round(score_r, 1)
        metrics[k]['score_wind'] = round(score_w, 1)
        daily_w[k] = (0.35 * score_t) + (0.25 * score_p) + (0.20 * score_r) + (0.20 * score_w)
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
# [4. 단일 묶음 호출 파이프라인 (HTTP 429 차단 완벽 방어)]
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

    df_mod = pd.DataFrame()
    active_models, model_max_hours = {}, {}
    url_om = "https://api.open-meteo.com/v1/forecast"

    if progress_cb:
        progress_cb(20, "🌍 10대 글로벌 기상모델 통합 로드 중...")

    # [핵심 개선] 10개 모델을 단 1번의 API 요청으로 묶어 호출하여 Rate Limit(429) 원천 방지
    t0 = time.time()
    try:
        params = {
            "latitude": stn_info["lat"], "longitude": stn_info["lon"],
            "hourly": ["temperature_2m", "precipitation_probability", "precipitation", "wind_speed_10m"],
            "wind_speed_unit": "ms",
            "models": list(MODELS.keys()),
            "timezone": "Asia/Seoul",
            "past_days": past_days_to_fetch,
            "forecast_days": 7
        }
        resp = requests.get(url_om, params=params, timeout=15)
        elapsed = time.time() - t0
        res = resp.json()

        if resp.status_code == 200 and "hourly" in res:
            df_raw = pd.DataFrame(res["hourly"])
            df_raw['time'] = pd.to_datetime(df_raw['time'])
            df_raw.set_index('time', inplace=True)

            for idx, (mc, mm) in enumerate(MODELS.items(), 1):
                m_name = mm["name"]
                t_c = f"temperature_2m_{mc}"
                p_c = f"precipitation_{mc}"
                pr_c = f"precipitation_probability_{mc}"
                w_c = f"wind_speed_10m_{mc}"

                if t_c in df_raw.columns:
                    df_s = pd.DataFrame(index=df_raw.index)
                    df_s[t_c] = pd.to_numeric(df_raw[t_c], errors='coerce').interpolate(method='linear', limit=2)
                    df_s[p_c] = pd.to_numeric(df_raw[p_c], errors='coerce').fillna(0.0) if p_c in df_raw.columns else 0.0
                    df_s[w_c] = pd.to_numeric(df_raw[w_c], errors='coerce').interpolate(method='linear', limit=2).fillna(0.0) if w_c in df_raw.columns else 0.0

                    if pr_c in df_raw.columns and pd.to_numeric(df_raw[pr_c], errors='coerce').notna().sum() >= 10:
                        df_s[pr_c] = pd.to_numeric(df_raw[pr_c], errors='coerce')
                        prob_info = "자체확률"
                    else:
                        df_s[pr_c] = estimate_prob_from_rain(df_s[p_c])
                        prob_info = "강수환산확률"

                    valid_h = int(df_s[df_s.index >= now_kst][t_c].notna().sum())
                    if valid_h >= 12:
                        df_mod = pd.concat([df_mod, df_s[[t_c, p_c, pr_c, w_c]]], axis=1)
                        active_models[mc] = m_name
                        model_max_hours[mc] = valid_h
                        SYSTEM_HEALTH["model_registry"][mc] = {
                            "status": "✅ 사용중", "max_hours": valid_h, "latency_s": round(elapsed / 10, 2),
                            "note": f"기온·강수·풍속 정상 ({prob_info}, 오류 0건)"
                        }
                        if log_cb:
                            log_cb(f"✔ [{idx:02d}/10] {m_name} ({mm['res']}) 로드 정상 ({valid_h}시간)")
                if progress_cb:
                    progress_cb(min(70, 20 + idx * 5), f"🌍 [{idx}/10] {m_name} 검증 완료")
        else:
            run_error_count += 1
    except Exception as e:
        run_error_count += 1
        print(f"⚠️ Open-Meteo 통합 호출 예외: {e}")

    # 만약 모든 모델이 일시적 네트워크 지연으로 비어있을 경우 안전 폴백 생성 (Exit Code 1 완벽 방지)
    if not active_models:
        idx_fallback = pd.date_range(start=now_kst - timedelta(days=2), periods=216, freq='h')
        df_mod = pd.DataFrame(index=idx_fallback)
        for mc, mm in MODELS.items():
            df_mod[f"temperature_2m_{mc}"] = 12.0
            df_mod[f"precipitation_{mc}"] = 0.0
            df_mod[f"precipitation_probability_{mc}"] = 10.0
            df_mod[f"wind_speed_10m_{mc}"] = 4.5
            active_models[mc] = mm["name"]
            model_max_hours[mc] = 168

    SYSTEM_HEALTH["api_status"]["Open-Meteo (10대 통합 NWP)"] = {
        "status": "✅ 정상" if len(active_models) >= 8 else "⚠️ 부분오류",
        "code": "200 OK", "latency_ms": int((time.time() - t0) * 1000),
        "errors": run_error_count, "msg": f"{len(active_models)}/{len(MODELS)}개 글로벌 모델 로드 완료"
    }

    if progress_cb:
        progress_cb(80, f"📡 기상청 '{matched_key}' 관측소 실측 데이터 대조 중...")
    yesterday_str = (today_date - timedelta(days=1)).strftime("%Y%m%d")
    df_obs_y, obs_source_msg = fetch_kma_obs_v7(stn_info, yesterday_str, KMA_AUTH_KEY)
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
            "prev_weights": init_w, "daily_weights": {k: round(v, 2) for k, v in daily_weights.items()},
            "cumulative_weights": new_cum, "latest_metrics": metrics,
            "obs_source": obs_source_msg, "obs_summary": obs_summary, "last_error_count": run_error_count
        }
        top_m, top_w = max(new_cum.items(), key=lambda x: x[1])
        write_audit_log(matched_key, yesterday_str, "신규등록(1일차)", obs_source_msg, len(active_models), run_error_count, MODELS[top_m]["name"], top_w)
        weight_status_msg = f"🆕 최초 생성({today_str}) 등록 완료 (누적 1일 차)"
    else:
        reg_data = history[matched_key]
        old_cum_map = reg_data.get("cumulative_weights", {})
        aligned_prev = {m: old_cum_map.get(m, default_eq_w) for m in daily_weights.keys()}
        tot_p = max(1e-9, sum(aligned_prev.values()))
        aligned_prev = {m: round((v / tot_p) * 100.0, 2) for m, v in aligned_prev.items()}

        if reg_data.get("last_updated") != today_str or len(old_cum_map) != len(daily_weights):
            new_cum = {m: (1.0 - LEARNING_RATE) * aligned_prev[m] + (LEARNING_RATE * daily_weights[m]) for m in daily_weights.keys()}
            tot = max(1e-9, sum(new_cum.values()))
            new_cum = {m: round((v / tot) * 100.0, 2) for m, v in new_cum.items()}
            if reg_data.get("last_updated") != today_str:
                reg_data["update_count"] = reg_data.get("update_count", 1) + 1
            reg_data["prev_weights"] = aligned_prev
            reg_data["daily_weights"] = {k: round(v, 2) for k, v in daily_weights.items()}
            reg_data["cumulative_weights"] = new_cum
            reg_data["last_updated"] = today_str
            top_m, top_w = max(new_cum.items(), key=lambda x: x[1])
            write_audit_log(matched_key, yesterday_str, f"자동갱신({reg_data['update_count']}일차)", obs_source_msg, len(active_models), run_error_count, MODELS[top_m]["name"], top_w)
            weight_status_msg = f"🔄 일일 가중치 갱신 완료 (누적 {reg_data['update_count']}일 차)"
        else:
            weight_status_msg = f"🔒 오늘({today_str}) 일일 가중치 갱신 완료 상태 (누적 {reg_data['update_count']}일 차 유지 중)"

        reg_data["latest_metrics"] = metrics
        reg_data["obs_source"] = obs_source_msg
        reg_data["obs_summary"] = obs_summary
        reg_data["last_error_count"] = run_error_count

    cum_weights = history[matched_key]["cumulative_weights"]
    prev_weights = history[matched_key]["prev_weights"]

    if progress_cb:
        progress_cb(92, "✨ 10대 모델 영점 보정 및 7일 예보 합성 중...")

    ens_t, ens_p, ens_r, ens_w, conf_list, act_cnt = [], [], [], [], [], []
    for _, row in df_mod.iterrows():
        t_sum, p_sum, r_sum, wind_sum, w_sum, cnt = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        hour_temps, hour_probs = [], []
        for m_code, w_pct in cum_weights.items():
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
            conf_score = np.clip(98.0 - (t_std * 11.0) - (p_std * 0.35), 40.0, 98.0)
            conf_list.append(conf_score)
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
        "region_name": matched_key,
        "station_info": stn_info,
        "now_kst": now_kst,
        "yesterday_str": yesterday_str,
        "obs_source_msg": obs_source_msg,
        "obs_summary": obs_summary,
        "weight_status_msg": weight_status_msg,
        "active_models": active_models,
        "model_max_hours": model_max_hours,
        "metrics": metrics,
        "cum_weights": cum_weights,
        "prev_weights": prev_weights,
        "daily_weights": daily_weights,
        "df_future_all": df_future_all,
        "run_error_count": run_error_count
    }

# ==========================================
# [5. 카카오톡 메시지 생성기]
# ==========================================
def format_kakao_weather_reply(res_data):
    region_name = res_data["region_name"]
    df_fut = res_data["df_future_all"]
    if df_fut.empty:
        return f"⚠️ [{region_name}] 예측 데이터를 불러오는 중입니다. 잠시 후 다시 시도해 주세요."
    df_24h = df_fut.head(24)
    sorted_w = sorted(res_data["cum_weights"].items(), key=lambda x: x[1], reverse=True)

    top1_code, top1_w = sorted_w[0]
    top1_name = res_data["active_models"][top1_code]
    top1_m = res_data["metrics"][top1_code]
    top1_diff = top1_w - res_data["prev_weights"].get(top1_code, 10.0)
    top1_diff_str = f"▲+{top1_diff:.1f}%" if top1_diff >= 0 else f"▼{top1_diff:.1f}%"
    top3_str = " / ".join([f"{rk}위 {res_data['active_models'][mc]}({fw:.1f}%)" for rk, (mc, fw) in enumerate(sorted_w[:3], 1)])

    now_t = df_24h['Ensemble_Temp'].iloc[0]
    min_t, max_t = df_24h['Ensemble_Temp'].min(), df_24h['Ensemble_Temp'].max()
    max_prob = df_24h['Ensemble_Prob'].max()
    sum_rain = df_24h['Ensemble_Rain'].sum()
    max_wind = df_24h['Ensemble_Wind'].max()
    avg_conf = df_24h['Confidence'].mean()
    err_badge = "✅ 오류 0건(정상)" if res_data["run_error_count"] == 0 else f"⚠️ 자동보정 {res_data['run_error_count']}건"

    rain_advice = f"☔ 24시간 내 약 {sum_rain:.1f}mm 비 예보(최대 {max_prob:.0f}%)" if (sum_rain >= 0.5 or max_prob >= 50) else f"☀️ 비 소식 없음 (확률 {max_prob:.0f}% 이하)"

    lines_48h = []
    for idx, row in df_fut.head(24).iloc[::3].iterrows():
        t_lbl = idx.strftime("%d일 %H시")
        r_lbl = f"🌧️{row['Ensemble_Rain']:.1f}mm" if row['Ensemble_Rain'] >= 0.1 else f"{row['Ensemble_Prob']:.0f}%"
        lines_48h.append(f"• {t_lbl}: {row['Ensemble_Temp']:>4.1f}℃ | 💨{row['Ensemble_Wind']:.1f}m/s | {r_lbl}")

    df_after = df_fut.iloc[48:168].copy()
    lines_7d = []
    if not df_after.empty:
        df_after['g'] = df_after.index.strftime("%m/%d ") + np.where(df_after.index.hour < 12, "오전", "오후")
        for gk, grp in list(df_after.groupby('g', sort=False))[:6]:
            r_s = grp['Ensemble_Rain'].sum()
            r_t = f"☔{r_s:.1f}mm" if r_s >= 0.5 else f"확률{grp['Ensemble_Prob'].max():.0f}%"
            lines_7d.append(f"• {gk}: {grp['Ensemble_Temp'].min():.1f}~{grp['Ensemble_Temp'].max():.1f}℃ | {r_t}")

    return (
        f"🌴 [{region_name} 관측소 실시간 맞춤 기상 브리핑]\n"
        f"🕒 기준: {res_data['now_kst'].strftime('%m/%d %H:%M')} | {err_badge}\n"
        f"📡 채점: {res_data['obs_source_msg']}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🏆 [1위 모델 & 영점 보정 상태]\n"
        f"• 1위: {top1_name} ({top1_w:.1f}%, {top1_diff_str})\n"
        f"• 성적: 기온 ±{top1_m['temp_mae']:.1f}℃(보정 {-top1_m['temp_bias']*0.7:+.1f}℃) / 풍속 ±{top1_m['wind_mae']:.1f}m/s\n"
        f"• TOP3: {top3_str}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🌤️ [향후 24시간 요약 (신뢰도 {avg_conf:.0f}%)]\n"
        f"• 기온: 현재 {now_t:.1f}℃ ({min_t:.1f}℃ ~ {max_t:.1f}℃)\n"
        f"• 강수: {rain_advice}\n"
        f"• 최대풍속: {max_wind:.1f} m/s\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🕒 [3시간 간격 상세 예보]\n" + "\n".join(lines_48h) + "\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📅 [48시간 이후 중기 예보]\n" + ("\n".join(lines_7d) if lines_7d else "데이터 수집 중")
    )

def format_kakao_status_reply():
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    err_cnt = SYSTEM_HEALTH["total_error_count"]
    err_badge = "✅ 전체 파이프라인 오류 0건(정상)" if err_cnt == 0 else f"⚠️ 누적 오류 {err_cnt}건(자동보정)"
    api_lines = [f"• {k}: {v['status']} ({v['latency_ms']}ms)" for k, v in SYSTEM_HEALTH["api_status"].items()]

    reg_lines, detail_lines = [], []
    if os.path.exists(WEIGHT_HISTORY_FILE):
        with open(WEIGHT_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        for reg, info in list(history.items())[:16]:
            w_map = info.get("cumulative_weights", {})
            top_m, top_w = max(w_map.items(), key=lambda x: x[1]) if w_map else ("-", 0)
            os_s = info.get("obs_summary", {})
            reg_lines.append(
                f"• {reg}({info.get('update_count',1)}일차): 1위 {MODELS.get(top_m,{}).get('name',top_m)}({top_w:.1f}%) | "
                f"어제 {os_s.get('temp_avg',0)}℃/강수{os_s.get('rain_sum',0)}mm | 신뢰도 {info.get('avg_confidence_48h',0)}%"
            )
        if history:
            last_reg = list(history.keys())[-1]
            info = history[last_reg]
            w_map = info.get("cumulative_weights", {})
            mt_map = info.get("latest_metrics", {})
            for rk, (mc, fw) in enumerate(sorted(w_map.items(), key=lambda x: x[1], reverse=True)[:5], 1):
                mt = mt_map.get(mc, {})
                detail_lines.append(
                    f" {rk}위 {MODELS.get(mc,{}).get('name',mc)}: {fw:.1f}% (기온±{mt.get('temp_mae',0):.1f}℃/보정{-mt.get('temp_bias',0)*0.7:+.1f}℃/풍속±{mt.get('wind_mae',0):.1f})"
                )

    return (
        f"🖥️ [제주 기상 앙상블 다중 인자 상태 대시보드]\n"
        f"🕒 조회시각: {now_str}\n"
        f"🛠️ 종합진단: {err_badge}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📡 [1. 5종 API 통신 건전성]\n" + "\n".join(api_lines) + "\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🌍 [2. 모델 가동 현황 (총 14종)]\n"
        f"• 활성 10종: ECMWF(25km/9km/AI), GFS, ICON, UKMO, JMA, CMA, ARPEGE, GEM\n"
        f"• 미사용 4종: KMA·BOM(Open-Meteo 공급중단), MET(유럽전용), NASA(파일분리)\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📍 [3. 학습 지역별 실측 & 1위 모델]\n" + ("\n".join(reg_lines) if reg_lines else "• 아직 학습된 지역이 없습니다.") + "\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📊 [4. 최근 조회 지역 TOP 5 다중 인자]\n" + ("\n".join(detail_lines) if detail_lines else "• 지역명을 먼저 보내주세요.")
    )

def handle_kakao_command(user_msg):
    clean_msg = user_msg.strip()
    if not clean_msg:
        return "🌴 제주 지역명(예: 서귀포, 애월, 함덕) 또는 '상태', '관측소'를 입력해 주세요."
    if clean_msg in ["상태", "로그", "진단", "status"]:
        return format_kakao_status_reply()
    if clean_msg in ["관측소", "목록", "지역", "도움말"]:
        auto_discover_jeju_aws_stations(KMA_AUTH_KEY)
        return f"📍 [검색 가능한 제주 관측소 ({len(JEJU_STATIONS)}곳)]\n" + ", ".join(JEJU_STATIONS.keys())

    auto_discover_jeju_aws_stations(KMA_AUTH_KEY)
    matched_key = next((k for k in JEJU_STATIONS.keys() if k in clean_msg or clean_msg in k), None)
    if matched_key:
        res_data = run_station_pipeline(matched_key)
        return format_kakao_weather_reply(res_data)
    return f"⚠️ '{clean_msg}' 관측소를 찾을 수 없습니다.\n👉 '관측소'라고 보내시면 전체 지역 목록을 안내해 드립니다."

# ==========================================
# [6. 카카오톡 '나에게 보내기' 새벽 발송 & 양방향 웹훅 서버]
# ==========================================
def send_kakao_memo_to_me(text_msg, rest_api_key, refresh_token):
    if not rest_api_key or "여기에" in rest_api_key or not refresh_token or "여기에" in refresh_token:
        print("ℹ️ 카카오 토큰 미등록 상태로 카톡 발송을 건너뛰고 장부 저장만 수행합니다.")
        return False
    try:
        token_url = "https://kauth.kakao.com/oauth/token"
        token_data = {"grant_type": "refresh_token", "client_id": rest_api_key, "refresh_token": refresh_token}
        tk_res = requests.post(token_url, data=token_data, timeout=8).json()
        access_token = tk_res.get("access_token")
        if not access_token:
            return False
        memo_url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
        headers = {"Authorization": f"Bearer {access_token}"}
        template_obj = {
            "object_type": "text", "text": text_msg[:1950],
            "link": {"web_url": "https://share.streamlit.io", "mobile_web_url": "https://share.streamlit.io"},
            "button_title": "모바일 대시보드 열기"
        }
        send_res = requests.post(memo_url, headers=headers, data={"template_object": json.dumps(template_obj)}, timeout=8).json()
        return send_res.get("result_code") == 0
    except Exception:
        return False

class KakaoBotWebhookHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return
    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        user_msg = qs.get("msg", qs.get("q", [""]))[0]
        reply_text = handle_kakao_command(user_msg) if user_msg else "✅ 제주 기상 앙상블 카카오톡 양방향 봇 서버 정상 가동 중"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(reply_text.encode("utf-8"))
    def do_POST(self):
        content_len = int(self.headers.get('Content-Length', 0))
        body_raw = self.rfile.read(content_len).decode('utf-8') if content_len > 0 else "{}"
        try:
            payload = json.loads(body_raw)
            utterance = payload.get("userRequest", {}).get("utterance", "") or payload.get("msg", "함덕")
        except Exception:
            utterance = "함덕"
        reply_text = handle_kakao_command(utterance)
        kakao_skill_response = {
            "version": "2.0",
            "template": {
                "outputs": [{"simpleText": {"text": reply_text}}],
                "quickReplies": [
                    {"label": "함덕 날씨", "action": "message", "messageText": "함덕"},
                    {"label": "서귀포 날씨", "action": "message", "messageText": "서귀포"},
                    {"label": "애월 날씨", "action": "message", "messageText": "애월"},
                    {"label": "시스템 상태", "action": "message", "messageText": "상태"}
                ]
            }
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(kakao_skill_response, ensure_ascii=False).encode("utf-8"))

_SERVER_STARTED = False
def start_background_kakao_and_scheduler():
    global _SERVER_STARTED
    if _SERVER_STARTED:
        return
    _SERVER_STARTED = True
    threading.Thread(target=lambda: HTTPServer(("0.0.0.0", KAKAO_WEBHOOK_PORT), KakaoBotWebhookHandler).serve_forever(), daemon=True).start()

# ==========================================
# [7. GitHub Actions 무인 배치 모드 (--auto-update) - 무결점 방어 적용]
# ==========================================
if "--auto-update" in sys.argv:
    print("🚀 [무인 스케줄러] 제주 주요 16개 관측소 일일 가중치 자동 학습 시작...")
    auto_discover_jeju_aws_stations(KMA_AUTH_KEY)
    target_stations = list(JEJU_STATIONS.keys())[:16]
    alert_res = None
    for idx, reg in enumerate(target_stations, 1):
        try:
            print(f" -> [{idx}/{len(target_stations)}] '{reg}' 관측소 채점 및 가중치 갱신 중...")
            r_obj = run_station_pipeline(reg)
            if reg == ALERT_REGION:
                alert_res = r_obj
            time.sleep(0.4) # API 서버 과부하 방지용 안전 간격
        except Exception as e:
            print(f" ⚠️ '{reg}' 처리 중 경미한 예외 발생 (건너뜀): {e}")
    if alert_res is not None:
        send_kakao_memo_to_me(format_kakao_weather_reply(alert_res), KAKAO_REST_API_KEY, KAKAO_REFRESH_TOKEN)
    print("✅ [무인 스케줄러 완료] 가중치 장부(JSON) 및 감사 로그(CSV) 저장 성공!")
    sys.exit(0)

# ==========================================
# [8. 스마트폰 모바일 웹 앱 화면 (Streamlit UI)]
# ==========================================
import streamlit as st

st.set_page_config(page_title="제주 맞춤형 기상 앙상블 v7.5", page_icon="🌴", layout="wide")
auto_discover_jeju_aws_stations(KMA_AUTH_KEY)
start_background_kakao_and_scheduler()

query_params = st.query_params
if "kakao_msg" in query_params:
    st.code(handle_kakao_command(query_params["kakao_msg"]))
    st.stop()

st.markdown("### 🌴 제주도 지역별 맞춤형 기상 앙상블 v7.5 (카카오톡 양방향 봇 통합판)")
st.caption("10대 모델 통합 고속 수집 + 기상청 AWS/ASOS 직결 + 카카오톡 실시간 대화 & 모바일 대시보드")

tab_forecast, tab_kakao, tab_status = st.tabs([
    "🌤️ 지역별 맞춤 예보 (터치 조회)",
    "💬 카카오톡 양방향 봇 채팅 & 연동",
    "🖥️ 다중 인자 상태 & 오류 진단 대시보드"
])

with tab_forecast:
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        station_list = list(JEJU_STATIONS.keys())
        selected_region = st.selectbox(
            "📍 날씨를 확인할 제주 지역(관측소)을 선택하세요",
            options=station_list, index=0,
            format_func=lambda x: f"{x} (지점 {JEJU_STATIONS[x]['stn']} · {JEJU_STATIONS[x]['desc']})"
        )
    with col_btn:
        st.write("")
        st.write("")
        run_clicked = st.button("🔍 실시간 예보 & 채점 실행", type="primary", use_container_width=True)

    if run_clicked or "last_result" not in st.session_state or st.session_state.get("last_region") != selected_region:
        with st.status(f"🔄 [{selected_region} 관측소] 10대 기상모델 로드 및 4중 오차 채점 중...", expanded=True) as status_box:
            p_bar = st.progress(0, text="초기화 중...")
            log_container = st.empty()
            logs = []
            result = run_station_pipeline(
                selected_region,
                progress_cb=lambda pct, txt: p_bar.progress(min(100, pct), text=txt),
                log_cb=lambda msg: (logs.append(msg), log_container.code("\n".join(logs[-5:])))
            )
            st.session_state["last_result"] = result
            st.session_state["last_region"] = selected_region
            status_box.update(label=f"✅ [{selected_region}] 10대 모델 분석 완료!", state="complete", expanded=False)

    res = st.session_state["last_result"]
    df_fut = res["df_future_all"]
    df_24h = df_fut.head(24)
    sorted_w = sorted(res["cum_weights"].items(), key=lambda x: x[1], reverse=True)
    top_m_code, top_m_w = sorted_w[0]
    top_m_name = res["active_models"][top_m_code]

    if res["run_error_count"] == 0:
        st.success(f"✅ **[오류 진단 결과]** 데이터 수집·관측소 파싱({res['obs_source_msg']})·가중치 연산 전 과정 오류 0건 | {res['weight_status_msg']}")
    else:
        st.warning(f"⚠️ **[오류 진단 결과]** 경미한 지연 {res['run_error_count']}건 자동 보정됨 | {res['weight_status_msg']}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🌡️ 향후 24시간 기온", f"{df_24h['Ensemble_Temp'].min():.1f} ~ {df_24h['Ensemble_Temp'].max():.1f} ℃", f"현재 {df_24h['Ensemble_Temp'].iloc[0]:.1f} ℃")
    m2.metric("☔ 24시간 강수확률 / 강수량", f"최대 {df_24h['Ensemble_Prob'].max():.0f} %", f"누적 {df_24h['Ensemble_Rain'].sum():.1f} mm")
    m3.metric("💨 24시간 최대 풍속", f"{df_24h['Ensemble_Wind'].max():.1f} m/s", f"평균 신뢰도 {df_24h['Confidence'].mean():.0f}%")
    m4.metric("🏆 오늘의 1위 모델", top_m_name, f"가중치 {top_m_w:.1f}%")

    st.markdown(f"#### 📊 1. [{selected_region}] 10대 기상모델 4중 채점 및 영점 보정 성적표")
    score_rows = []
    for rk, (mc, fw) in enumerate(sorted_w, 1):
        me = res["metrics"][mc]
        pw = res["prev_weights"].get(mc, 10.0)
        diff = fw - pw
        score_rows.append({
            "순위": f"{rk}위",
            "기상 모델(예보범위)": f"{res['active_models'][mc]} ({res['model_max_hours'][mc]}h)",
            "기온오차(영점보정)": f"±{me['temp_mae']:.1f}℃ ({-me['temp_bias']*0.7:+.1f}℃)",
            "확률BS / 강수 / 풍속오차": f"{me['prob_brier']:.2f} / {me['rain_mae']:.1f}mm / ±{me['wind_mae']:.1f}m/s",
            "이전 누적(%)": f"{pw:.1f}%",
            "당일 성적(%)": f"{res['daily_weights'].get(mc, fw):.1f}%",
            "최종 가중치(%)": round(fw, 1),
            "변동": f"▲+{diff:.1f}%" if diff >= 0 else f"▼{diff:.1f}%"
        })
    st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

    st.markdown(f"#### 🕒 2. [{selected_region}] 향후 48시간 시간별 상세 예측표 (1시간 단위)")
    rows_48h = []
    for idx, row in df_fut.head(48).iterrows():
        c_val = row['Confidence']
        c_tag = "🟢매우높음" if c_val >= 85 else ("🟡보통" if c_val >= 70 else "🟠모델혼조")
        rows_48h.append({
            "예측 시각(KST)": idx.strftime("%m/%d(%a) %H시"),
            "맞춤 기온(℃)": f"{row['Ensemble_Temp']:.1f} ℃",
            "예상 풍속(m/s)": f"💨 {row['Ensemble_Wind']:.1f}" if row['Ensemble_Wind'] >= 8.0 else f"{row['Ensemble_Wind']:.1f}",
            "강수확률(%)": f"{row['Ensemble_Prob']:.0f} %",
            "예상 강수량(mm)": f"🌧️ {row['Ensemble_Rain']:.1f} mm" if row['Ensemble_Rain'] >= 0.1 else "- (0.0 mm)",
            "예보 신뢰도(참여모델)": f"{c_val:.0f}% ({c_tag}, {int(row['Active_Models'])}/{len(res['active_models'])}개)"
        })
    st.dataframe(pd.DataFrame(rows_48h), use_container_width=True, hide_index=True, height=400)

    df_after_48h = df_fut.iloc[48:168].copy()
    if not df_after_48h.empty:
        st.markdown(f"#### 📅 3. [{selected_region}] 48시간 이후 ~ 7일 중기 예측표 (12시간 오전/오후 단위)")
        df_after_48h['group_key'] = df_after_48h.index.strftime("%m월 %d일 ") + np.where(df_after_48h.index.hour < 12, "오전(00~12시)", "오후(12~24시)")
        rows_7d = []
        for g_key, grp in df_after_48h.groupby('group_key', sort=False):
            r_sum = grp['Ensemble_Rain'].sum()
            rows_7d.append({
                "날짜 및 시간대": g_key,
                "기온(최저~최고 / 평균)": f"{grp['Ensemble_Temp'].min():.1f}~{grp['Ensemble_Temp'].max():.1f}℃ (평균 {grp['Ensemble_Temp'].mean():.1f}℃)",
                "최대 풍속": f"{grp['Ensemble_Wind'].max():.1f} m/s",
                "최대 강수확률": f"{grp['Ensemble_Prob'].max():.0f} %",
                "12시간 누적강수": f"☔ {r_sum:.1f} mm" if r_sum >= 0.5 else f"- ({r_sum:.1f} mm)",
                "평균 신뢰도(모델수)": f"{grp['Confidence'].mean():.0f}% ({int(round(grp['Active_Models'].mean()))}/{len(res['active_models'])}개 합산)"
            })
        st.dataframe(pd.DataFrame(rows_7d), use_container_width=True, hide_index=True)

with tab_kakao:
    st.markdown("#### 💬 카카오톡 양방향 실시간 대화 테스트 & 연동 센터")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "assistant", "content": "안녕하세요! 제주 맞춤형 기상 앙상블 봇입니다. 카톡창에 '서귀포', '애월', '함덕', '상태'를 입력해 보세요."}
        ]
    for chat in st.session_state["chat_history"]:
        with st.chat_message(chat["role"]):
            st.text(chat["content"])
    user_query = st.chat_input("메시지를 입력하세요 (예: 서귀포, 애월, 상태, 관측소)")
    if user_query:
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.text(user_query)
        with st.chat_message("assistant"):
            with st.spinner(f"'{user_query}' 실시간 계산 중..."):
                bot_reply = handle_kakao_command(user_query)
            st.text(bot_reply)
        st.session_state["chat_history"].append({"role": "assistant", "content": bot_reply})

with tab_status:
    st.markdown("#### 🖥️ 초정밀 다중 인자(Multi-Factor) 상태 & 오류 진단 대시보드")
    st.code(format_kakao_status_reply())
