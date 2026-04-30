"""
JRA公式サイトスクレイパー
出馬表・レース結果・オッズを取得する
"""

import re
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import pandas as pd

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# JRA場コード → 場名
VENUE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}

# 馬場状態コード
TRACK_CONDITIONS = {"1": "良", "2": "稍重", "3": "重", "4": "不良"}

# レースグレード
GRADE_MAP = {"G1": 5, "G2": 4, "G3": 3, "OP": 2, "L": 2, "3勝": 1, "2勝": 1, "1勝": 1, "未勝利": 0, "新馬": 0}


def fetch_race_card(date: datetime, venue_code: str, race_no: int,
                    race_id: str = None, timeout: int = 15) -> dict:
    """
    JRA公式から1レース分の出馬表を取得する

    Returns:
        {
            "venue": "東京", "race_no": 11, "grade": "G1",
            "distance": 2400, "surface": "芝", "condition": "良",
            "weather": "晴", "horses": [
                {"horse_no": 1, "horse_name": "...", "jockey": "...",
                 "weight": 57.0, "horse_weight": 498, "weight_diff": +2,
                 "odds_win": 3.5, "popularity": 1,
                 "past_results": [...]}
            ]
        }
    """
    rid = race_id or _make_race_id(date, venue_code, race_no)
    url = (
        f"https://race.netkeiba.com/race/shutuba.html"
        f"?race_id={rid}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = "EUC-JP"
    except requests.RequestException as e:
        print(f"[WARN] 出馬表取得失敗 {venue_code}-R{race_no}: {e}")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_race_card(soup, date, venue_code, race_no)


def fetch_race_result(date: datetime, venue_code: str, race_no: int,
                      race_id: str = None, timeout: int = 15) -> dict:
    """
    JRA公式から1レース分の結果を取得する

    Returns:
        {
            "available": True,
            "winner": 5, "second": 3,
            "quinella": "3-5", "quinella_payout": 1240,
            "win_payout": 350,
        }
    """
    rid = race_id or _make_race_id(date, venue_code, race_no)
    url = (
        f"https://race.netkeiba.com/race/result.html"
        f"?race_id={rid}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = "EUC-JP"
    except requests.RequestException as e:
        print(f"[WARN] 結果取得失敗 {venue_code}-R{race_no}: {e}")
        return {"available": False}

    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_race_result(soup)


def fetch_today_schedule(date: datetime = None, timeout: int = 15) -> list[dict]:
    """
    本日の開催レース一覧を取得する

    Returns:
        [{"venue_code": "05", "venue": "東京", "race_no": 1, "scheduled_time": "10:00"}, ...]
    """
    if date is None:
        date = datetime.now()

    url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date.strftime('%Y%m%d')}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = "EUC-JP"
    except requests.RequestException as e:
        print(f"[WARN] 開催スケジュール取得失敗: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_schedule(soup, date)


def fetch_odds_quinella(date: datetime, venue_code: str, race_no: int,
                        race_id: str = None, timeout: int = 15) -> dict:
    """
    馬連オッズを取得する

    Returns:
        {"1-2": 15.3, "1-3": 8.2, ...}
    """
    rid = race_id or _make_race_id(date, venue_code, race_no)
    url = (
        f"https://odds.netkeiba.com/odds/odds_get_form.cgi"
        f"?race_id={rid}&type=b5"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[WARN] 馬連オッズ取得失敗 {venue_code}-R{race_no}: {e}")
        return {}

    return _parse_quinella_odds(resp.text)


def fetch_jockey_stats(jockey_id: str, timeout: int = 15) -> dict:
    """
    騎手の成績統計を取得する（今年の勝率・連対率等）

    Returns:
        {"win_rate": 0.18, "top2_rate": 0.35, "top3_rate": 0.48}
    """
    url = f"https://db.netkeiba.com/jockey/{jockey_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = "EUC-JP"
    except requests.RequestException as e:
        print(f"[WARN] 騎手成績取得失敗 {jockey_id}: {e}")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_jockey_stats(soup)


def fetch_horse_past_results(horse_id: str, n: int = 5, timeout: int = 15) -> list[dict]:
    """
    馬の直近n走の成績を取得する

    Returns:
        [{"date": "2026-03-01", "venue": "阪神", "rank": 1,
          "distance": 2000, "surface": "芝", "condition": "良",
          "time": "1:58.2", "jockey": "川田将雅", "weight": 57.0}, ...]
    """
    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = "EUC-JP"
    except requests.RequestException as e:
        print(f"[WARN] 馬成績取得失敗 {horse_id}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_horse_results(soup, n)


# ─── 内部パース関数 ───

def _make_race_id(date: datetime, venue_code: str, race_no: int) -> str:
    """レースIDを生成 例: 202605010111 (年+場コード+開催回+日+レース番号)"""
    # netkeiba形式: YYYYJJKKNN (年4桁+場2桁+開催回2桁+日2桁+レース2桁)
    # 開催回・日は番組表から取る必要があるが、スケジュール取得で補完
    year = date.strftime("%Y")
    return f"{year}{venue_code.zfill(2)}0101{str(race_no).zfill(2)}"


def _parse_race_card(soup: BeautifulSoup, date: datetime, venue_code: str, race_no: int) -> dict:
    """出馬表HTMLをパースしてdictを返す"""
    result = {
        "date": date.strftime("%Y-%m-%d"),
        "venue_code": venue_code,
        "venue": VENUE_CODES.get(venue_code, "不明"),
        "race_no": race_no,
        "grade": "一般",
        "distance": 0,
        "surface": "芝",
        "condition": "良",
        "weather": "晴",
        "horses": [],
    }

    # レース情報
    race_info = soup.find(class_=re.compile(r'RaceData|race_info', re.I))
    if race_info:
        text = race_info.get_text()
        # 距離・馬場
        m = re.search(r'(芝|ダート)\s*(\d{3,4})', text)
        if m:
            result["surface"] = m.group(1)
            result["distance"] = int(m.group(2))
        # グレード
        for g in ["G1", "G2", "G3", "L", "OP"]:
            if g in text:
                result["grade"] = g
                break
        # 馬場状態
        for cond in ["不良", "重", "稍重", "良"]:
            if cond in text:
                result["condition"] = cond
                break
        # 天候
        for w in ["雨", "曇", "晴"]:
            if w in text:
                result["weather"] = w
                break

    # 出走馬リスト
    horse_rows = soup.select("tr.HorseList, tr.Shutuba_HorseList")
    for row in horse_rows:
        cells = row.find_all("td")
        if len(cells) < 8:
            continue
        try:
            horse_no_text = cells[0].get_text(strip=True)
            if not horse_no_text.isdigit():
                continue
            horse_no = int(horse_no_text)

            horse_name_el = row.find(class_=re.compile(r'HorseName|horse_name', re.I))
            horse_name = horse_name_el.get_text(strip=True) if horse_name_el else ""

            jockey_el = row.find(class_=re.compile(r'Jockey|jockey', re.I))
            jockey = jockey_el.get_text(strip=True) if jockey_el else ""

            # 斤量
            weight_text = ""
            for cell in cells:
                t = cell.get_text(strip=True)
                if re.match(r'^\d{2}\.\d$', t):
                    weight_text = t
                    break
            weight = float(weight_text) if weight_text else 55.0

            result["horses"].append({
                "horse_no": horse_no,
                "horse_name": horse_name,
                "jockey": jockey,
                "weight": weight,
            })
        except (ValueError, IndexError):
            continue

    return result


def _parse_race_result(soup: BeautifulSoup) -> dict:
    """レース結果HTMLをパースしてdictを返す"""
    result = {"available": False}

    # 着順テーブルを探す
    result_table = soup.find(class_=re.compile(r'RaceResult|race_result', re.I))
    if not result_table:
        result_table = soup.find("table", class_=re.compile(r'result', re.I))

    if not result_table:
        return result

    rows = result_table.find_all("tr")
    ranks = {}
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        rank_text = cells[0].get_text(strip=True)
        if rank_text in ("1", "2", "3"):
            horse_no_text = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            if horse_no_text.isdigit():
                ranks[int(rank_text)] = int(horse_no_text)

    if 1 not in ranks or 2 not in ranks:
        return result

    w = ranks[1]
    s = ranks[2]
    combo = f"{min(w,s)}-{max(w,s)}"

    # 払戻情報
    win_payout = 0
    quinella_payout = 0
    payout_table = soup.find(class_=re.compile(r'Payout|payout', re.I))
    if payout_table:
        text = payout_table.get_text()
        m = re.search(r'単勝.*?(\d[\d,]+)円', text)
        if m:
            win_payout = int(m.group(1).replace(",", ""))
        m = re.search(r'馬連.*?(\d[\d,]+)円', text)
        if m:
            quinella_payout = int(m.group(1).replace(",", ""))

    result.update({
        "available": True,
        "winner": w,
        "second": s,
        "quinella": combo,
        "win_payout": win_payout,
        "quinella_payout": quinella_payout,
        "ranks": ranks,
    })
    return result


def _parse_schedule(soup: BeautifulSoup, date: datetime) -> list[dict]:
    """開催スケジュールHTMLをパースしてリストを返す"""
    schedule = []
    race_links = soup.select("a[href*='race_id']")
    seen = set()

    for link in race_links:
        href = link.get("href", "")
        m = re.search(r'race_id=(\d{12})', href)
        if not m:
            continue
        race_id = m.group(1)
        if race_id in seen:
            continue
        seen.add(race_id)

        venue_code = race_id[4:6]
        race_no = int(race_id[10:12])
        time_el = link.find_previous(class_=re.compile(r'time|Time', re.I))
        scheduled_time = time_el.get_text(strip=True) if time_el else None

        schedule.append({
            "race_id": race_id,
            "venue_code": venue_code,
            "venue": VENUE_CODES.get(venue_code, "不明"),
            "race_no": race_no,
            "scheduled_time": scheduled_time,
            "date": date.strftime("%Y-%m-%d"),
        })

    return sorted(schedule, key=lambda x: (x["venue_code"], x["race_no"]))


def _parse_quinella_odds(text: str) -> dict:
    """馬連オッズのレスポンスをパースしてdictを返す"""
    odds = {}
    # タブ区切り or カンマ区切りで馬番1,馬番2,オッズ の形式が多い
    for line in text.splitlines():
        parts = re.split(r'[\t,]', line.strip())
        if len(parts) >= 3:
            try:
                h1, h2, o = int(parts[0]), int(parts[1]), float(parts[2])
                key = f"{min(h1,h2)}-{max(h1,h2)}"
                odds[key] = o
            except (ValueError, IndexError):
                continue
    return odds


def _parse_jockey_stats(soup: BeautifulSoup) -> dict:
    """騎手成績HTMLをパースして勝率等を返す"""
    stats = {"win_rate": 0.15, "top2_rate": 0.30, "top3_rate": 0.45}
    table = soup.find("table", class_=re.compile(r'nk_tb_common', re.I))
    if not table:
        return stats
    rows = table.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        # 今年の成績行（1着・2着・3着・着外・勝率・連対率等）
        if len(cells) >= 8:
            try:
                win_rate_text = cells[5].get_text(strip=True).replace("%", "")
                top2_rate_text = cells[6].get_text(strip=True).replace("%", "")
                top3_rate_text = cells[7].get_text(strip=True).replace("%", "")
                stats["win_rate"] = float(win_rate_text) / 100
                stats["top2_rate"] = float(top2_rate_text) / 100
                stats["top3_rate"] = float(top3_rate_text) / 100
                break
            except (ValueError, IndexError):
                continue
    return stats


def _parse_horse_results(soup: BeautifulSoup, n: int) -> list[dict]:
    """馬の過去成績HTMLをパースして直近n走を返す"""
    results = []
    table = soup.find("table", class_=re.compile(r'db_h_race_results|race_table', re.I))
    if not table:
        return results

    rows = table.find_all("tr")[1:]  # ヘッダー除外
    for row in rows[:n]:
        cells = row.find_all("td")
        if len(cells) < 12:
            continue
        try:
            rank_text = cells[11].get_text(strip=True) if len(cells) > 11 else ""
            rank = int(rank_text) if rank_text.isdigit() else 99

            dist_text = cells[6].get_text(strip=True) if len(cells) > 6 else ""
            surface = "芝" if dist_text.startswith("芝") else "ダート"
            dist_m = re.search(r'\d{3,4}', dist_text)
            distance = int(dist_m.group()) if dist_m else 0

            condition = cells[7].get_text(strip=True) if len(cells) > 7 else "良"

            results.append({
                "rank": rank,
                "surface": surface,
                "distance": distance,
                "condition": condition,
            })
        except (ValueError, IndexError):
            continue

    return results
