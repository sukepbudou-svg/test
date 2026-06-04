"""
直前情報スクレイピングモジュール
boatrace.jp から展示タイム・展示STを取得する
"""

import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

BEFOREINFO_URL = "https://www.boatrace.jp/owpc/pc/race/beforeinfo"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


def fetch_beforeinfo(date: datetime, venue_code: str, race_no: int, timeout: int = 10) -> dict:
    """
    指定レースの直前情報（展示タイム・展示ST）を取得する

    Returns:
        {
            1: {"exhibition_time": 6.82, "exhibition_st": 0.12},
            2: {"exhibition_time": 6.91, "exhibition_st": 0.15},
            ...
        }
        取得失敗時は空dict。
    """
    hd = date.strftime("%Y%m%d")
    params = {"rno": race_no, "jcd": venue_code.zfill(2), "hd": hd}

    try:
        resp = requests.get(BEFOREINFO_URL, params=params, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[WARN] 直前情報取得失敗 {venue_code}-R{race_no}: {e}")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_beforeinfo(soup, venue_code, race_no)


def _parse_weather_conditions(soup: BeautifulSoup) -> dict:
    """直前情報ページから天候・風速・波高・風向をパースする"""
    conditions = {"weather": None, "wind_speed": 0, "wave_height": 0, "wind_direction": None}

    # weather系クラスの要素を優先、なければページ全体のテキスト
    weather_el = soup.find(class_=re.compile(r'weather', re.I))
    text = weather_el.get_text() if weather_el else soup.get_text()

    m = re.search(r'風速[：:\s]*(\d+)', text)
    if m:
        conditions["wind_speed"] = int(m.group(1))

    m = re.search(r'波高[：:\s]*(\d+)', text)
    if m:
        conditions["wave_height"] = int(m.group(1))

    for kw, code in [("雨", "rain"), ("曇", "cloudy"), ("晴", "sunny")]:
        if kw in text:
            conditions["weather"] = code
            break

    # 風向パース（テキスト優先 → CSSクラス回転角フォールバック）
    if "向かい" in text:
        conditions["wind_direction"] = "head"
    elif "追い" in text:
        conditions["wind_direction"] = "tail"
    elif "横" in text and "風" in text:
        conditions["wind_direction"] = "side"
    else:
        # boatrace.jp は is-windN クラスの回転角で風向を表示する場合がある
        wind_el = soup.find(class_=re.compile(r'is-wind', re.I))
        if wind_el:
            style = wind_el.get("style", "")
            m2 = re.search(r'rotate\(([\d.]+)deg\)', style)
            if m2:
                deg = float(m2.group(1)) % 360
                # 回転0°=北、追い風(南北コース)は0°or180°、向かい風はその逆
                # 簡易判定: 315〜45° or 135〜225° → headwind/tailwind として扱う
                if 315 <= deg or deg < 45:
                    conditions["wind_direction"] = "head"
                elif 135 <= deg < 225:
                    conditions["wind_direction"] = "tail"
                else:
                    conditions["wind_direction"] = "side"

    return conditions


def _parse_beforeinfo(soup: BeautifulSoup, venue_code: str, race_no: int) -> dict:
    """BeautifulSoupオブジェクトから展示タイム・STおよび天候をパースする"""
    result = {}
    absent_boats = []

    # テーブルを全て取得して展示タイムが含まれる行を探す
    # boatrace.jp の直前情報ページ構造:
    #   艇番 | 選手名 | ... | 展示タイム | 展示ST
    tables = soup.find_all("table")

    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue

            # 艇番（1〜6）を含む行を検索
            boat_no = _extract_boat_no(cells)
            if boat_no is None:
                continue

            # 欠場チェック（行テキストに「欠場」が含まれる場合）
            row_text = row.get_text()
            if "欠場" in row_text:
                absent_boats.append(boat_no)
                continue

            exh_time = _extract_exhibition_time(cells)
            exh_st = _extract_exhibition_st(cells)

            if exh_time is not None:
                result[boat_no] = {
                    "exhibition_time": exh_time,
                    "exhibition_st": exh_st,
                }

    if not result:
        # フォールバック: テキスト全体から数値パターンを探す
        result = _parse_fallback(soup)

    # 天候・風速・波高を取得して "weather" キーに追加
    weather = _parse_weather_conditions(soup)
    result["weather"] = weather
    result["absent_boats"] = absent_boats

    boat_count = sum(1 for k in result if isinstance(k, int))
    if boat_count > 0:
        w = weather
        dir_label = {"head": "向かい風", "tail": "追い風", "side": "横風"}.get(w.get("wind_direction"), "")
        weather_str = (f" 天候:{w['weather']} 風速:{w['wind_speed']}m{dir_label} 波高:{w['wave_height']}cm"
                       if w.get("wind_speed") else "")
        absent_str = f" 欠場:{absent_boats}" if absent_boats else ""
        print(f"  [OK] 直前情報 場{venue_code} R{race_no}: {boat_count}艇分{weather_str}{absent_str}")
    else:
        print(f"  [--] 直前情報 場{venue_code} R{race_no}: 取得できず（レース前または構造変更）")

    return result


def _extract_boat_no(cells: list) -> int | None:
    """セルリストから艇番（1〜6）を抽出する"""
    for cell in cells[:3]:  # 先頭3セルに艇番があるはず
        text = cell.get_text(strip=True)
        if text in ("1", "2", "3", "4", "5", "6"):
            return int(text)
    return None


def _extract_exhibition_time(cells: list) -> float | None:
    """セルリストから展示タイム（6.XX形式）を抽出する"""
    for cell in cells:
        text = cell.get_text(strip=True)
        m = re.match(r"^(6\.\d{2}|7\.\d{2})$", text)
        if m:
            return float(m.group(1))
    return None


def _extract_exhibition_st(cells: list) -> float | None:
    """セルリストから展示ST（0.XX または F）を抽出する"""
    for cell in cells:
        text = cell.get_text(strip=True)
        # 通常ST: 0.01〜0.49、フライング: F
        if text == "F":
            return -0.01
        m = re.match(r"^(0\.\d{2}|\-0\.\d{2})$", text)
        if m:
            return float(m.group(1))
    return None


def _parse_fallback(soup: BeautifulSoup) -> dict:
    """構造解析失敗時のフォールバック（テキスト全体からパターン抽出）"""
    result = {}
    text = soup.get_text()
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 展示タイム行のパターン: "1 ... 6.82 ... 0.12"
    for i, line in enumerate(lines):
        if line in ("1", "2", "3", "4", "5", "6"):
            boat_no = int(line)
            # 前後10行以内で展示タイムを探す
            search = lines[i:i+10]
            exh_time = None
            exh_st = None
            for s in search:
                if re.match(r"^[67]\.\d{2}$", s) and exh_time is None:
                    exh_time = float(s)
                if re.match(r"^0\.\d{2}$", s) and exh_st is None:
                    exh_st = float(s)
            if exh_time is not None and boat_no not in result:
                result[boat_no] = {"exhibition_time": exh_time, "exhibition_st": exh_st}

    return result


def fetch_beforeinfo_for_races(
    df_today,
    date: datetime = None,
    interval: float = 1.5,
) -> dict:
    """
    本日の全レースの直前情報を取得する

    Returns:
        {(venue_code, race_no): {1: {"exhibition_time": 6.82, ...}, 2: {...}, ...}}
    """
    if date is None:
        date = datetime.now()

    races = df_today[["venue_code", "race_no"]].drop_duplicates()
    all_info = {}

    print(f"=== 直前情報取得中: {len(races)}レース ===")
    for _, row in races.iterrows():
        venue_code = str(row["venue_code"]).zfill(2)
        race_no = int(row["race_no"])
        key = (venue_code, race_no)

        info = fetch_beforeinfo(date, venue_code, race_no)
        if info:
            all_info[key] = info

        time.sleep(interval)

    success = len(all_info)
    print(f"=== 直前情報取得完了: {success}/{len(races)}レース成功 ===")
    return all_info
