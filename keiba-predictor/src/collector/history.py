"""
JRA過去レース成績一括取得モジュール
学習用データを収集する
"""

import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pandas as pd

from src.collector.scraper import HEADERS, VENUE_CODES, GRADE_MAP, CONDITION_NUM, SURFACE_NUM

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def fetch_history(months: int = 3, interval: float = 1.5) -> pd.DataFrame:
    """
    過去n月分のJRAレース成績を取得して学習用DataFrameを返す

    Returns:
        DataFrame（馬連組み合わせごとに1行、result列=的中1/外れ0）
    """
    today = datetime.now()
    all_records = []

    # 対象日付を列挙（土日のみ）
    target_dates = _get_race_dates(today, months)
    print(f"=== 過去データ取得開始: {len(target_dates)}日分 ===")

    for i, date in enumerate(target_dates):
        print(f"[{i+1}/{len(target_dates)}] {date.strftime('%Y-%m-%d')} 取得中...")
        day_records = _fetch_day_results(date, interval)
        all_records.extend(day_records)
        time.sleep(interval)

    if not all_records:
        print("[WARN] 取得できたデータがありません")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    print(f"=== 取得完了: {len(df)}組み合わせ / 的中数: {df['result'].sum()} ===")
    return df


def _get_race_dates(today: datetime, months: int) -> list[datetime]:
    """過去n月の土日日付リストを返す"""
    dates = []
    current = today - timedelta(days=1)
    cutoff = today - timedelta(days=months * 30)
    while current >= cutoff:
        if current.weekday() in (5, 6):  # 土曜=5, 日曜=6
            dates.append(current)
        current -= timedelta(days=1)
    return list(reversed(dates))


def _fetch_day_results(date: datetime, interval: float = 1.5) -> list[dict]:
    """1日分の全レース結果を取得して馬連組み合わせレコードのリストを返す"""
    date_str = date.strftime("%Y%m%d")
    records = []

    # netkeiba のレース一覧ページから当日の全race_idを取得
    url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = "EUC-JP"
    except requests.RequestException as e:
        print(f"  [WARN] {date_str} スケジュール取得失敗: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    race_ids = _extract_race_ids(soup)
    if not race_ids:
        print(f"  [INFO] {date_str} 開催なし")
        return []

    print(f"  {len(race_ids)}レース取得開始")
    for race_id in race_ids:
        race_records = _fetch_race_quinella_records(race_id, date)
        records.extend(race_records)
        time.sleep(interval)

    return records


def _extract_race_ids(soup: BeautifulSoup) -> list[str]:
    """レース一覧ページからrace_idリストを抽出する"""
    race_ids = []
    seen = set()
    for link in soup.select("a[href*='race_id']"):
        href = link.get("href", "")
        m = re.search(r'race_id=(\d{12})', href)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            race_ids.append(m.group(1))
    return race_ids


def _fetch_race_quinella_records(race_id: str, date: datetime) -> list[dict]:
    """
    1レース分の結果を取得し、馬連全組み合わせ×的中フラグのレコードリストを返す
    """
    from itertools import combinations

    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = "EUC-JP"
    except requests.RequestException:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # レース情報
    race_info = _parse_race_info(soup)
    if not race_info:
        return []

    # 着順取得
    ranks = _parse_result_ranks(soup)
    if 1 not in ranks or 2 not in ranks:
        return []

    winner = ranks[1]
    second = ranks[2]
    winning_combo = f"{min(winner, second)}-{max(winner, second)}"

    # 馬連払戻
    quinella_payout = _parse_quinella_payout(soup)

    # 出走馬番リスト
    horse_nos = _parse_horse_nos(soup)
    if len(horse_nos) < 2:
        return []

    # 騎手・馬の成績（簡易版: 出走表から取得）
    jockey_stats = _parse_jockey_stats_from_result(soup, horse_nos)
    horse_stats = _parse_horse_stats_from_result(soup, horse_nos)

    # レース条件
    grade_num = GRADE_MAP.get(race_info.get("grade", "一般"), 1)
    condition_num = CONDITION_NUM.get(race_info.get("condition", "良"), 4)
    surface_num = SURFACE_NUM.get(race_info.get("surface", "芝"), 1)
    distance = race_info.get("distance", 2000)
    weather_num = {"晴": 3, "曇": 2, "雨": 1}.get(race_info.get("weather", "晴"), 3)

    records = []
    for h1, h2 in combinations(sorted(horse_nos), 2):
        combo = f"{h1}-{h2}"
        result = 1 if combo == winning_combo else 0

        s1 = jockey_stats.get(h1, {})
        s2 = jockey_stats.get(h2, {})
        hs1 = horse_stats.get(h1, {})
        hs2 = horse_stats.get(h2, {})

        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "race_id": race_id,
            "combination": combo,
            "horse1": h1,
            "horse2": h2,
            "result": result,
            "quinella_payout": quinella_payout if result == 1 else 0,
            # レース条件
            "grade_num": grade_num,
            "condition_num": condition_num,
            "weather_num": weather_num,
            "surface_num": surface_num,
            "distance": distance,
            # 馬1の特徴量
            "h1_weight": hs1.get("weight", 55.0),
            "h1_horse_weight": hs1.get("horse_weight", 480),
            "h1_weight_diff": hs1.get("weight_diff", 0),
            "h1_jockey_win_rate": s1.get("win_rate", 0.15),
            "h1_jockey_top2_rate": s1.get("top2_rate", 0.30),
            "h1_jockey_top3_rate": s1.get("top3_rate", 0.45),
            "h1_past_avg_rank": hs1.get("past_avg_rank", 8.0),
            "h1_past_win_rate": hs1.get("past_win_rate", 0.10),
            "h1_past_top3_rate": hs1.get("past_top3_rate", 0.30),
            "h1_same_cond_rate": hs1.get("same_cond_rate", 0.30),
            "h1_recent_form": hs1.get("recent_form", 1),
            # 馬2の特徴量
            "h2_weight": hs2.get("weight", 55.0),
            "h2_horse_weight": hs2.get("horse_weight", 480),
            "h2_weight_diff": hs2.get("weight_diff", 0),
            "h2_jockey_win_rate": s2.get("win_rate", 0.15),
            "h2_jockey_top2_rate": s2.get("top2_rate", 0.30),
            "h2_jockey_top3_rate": s2.get("top3_rate", 0.45),
            "h2_past_avg_rank": hs2.get("past_avg_rank", 8.0),
            "h2_past_win_rate": hs2.get("past_win_rate", 0.10),
            "h2_past_top3_rate": hs2.get("past_top3_rate", 0.30),
            "h2_same_cond_rate": hs2.get("same_cond_rate", 0.30),
            "h2_recent_form": hs2.get("recent_form", 1),
            # 差分・和
            "diff_jockey_win_rate": s1.get("win_rate", 0.15) - s2.get("win_rate", 0.15),
            "sum_jockey_win_rate": s1.get("win_rate", 0.15) + s2.get("win_rate", 0.15),
            "diff_past_avg_rank": hs1.get("past_avg_rank", 8.0) - hs2.get("past_avg_rank", 8.0),
            "sum_past_avg_rank": hs1.get("past_avg_rank", 8.0) + hs2.get("past_avg_rank", 8.0),
            "diff_past_win_rate": hs1.get("past_win_rate", 0.10) - hs2.get("past_win_rate", 0.10),
            "sum_past_win_rate": hs1.get("past_win_rate", 0.10) + hs2.get("past_win_rate", 0.10),
            "diff_past_top3_rate": hs1.get("past_top3_rate", 0.30) - hs2.get("past_top3_rate", 0.30),
            "sum_past_top3_rate": hs1.get("past_top3_rate", 0.30) + hs2.get("past_top3_rate", 0.30),
            "diff_same_cond_rate": hs1.get("same_cond_rate", 0.30) - hs2.get("same_cond_rate", 0.30),
            "sum_same_cond_rate": hs1.get("same_cond_rate", 0.30) + hs2.get("same_cond_rate", 0.30),
            "diff_recent_form": hs1.get("recent_form", 1) - hs2.get("recent_form", 1),
            "sum_recent_form": hs1.get("recent_form", 1) + hs2.get("recent_form", 1),
            "diff_weight": hs1.get("weight", 55.0) - hs2.get("weight", 55.0),
            "sum_weight": hs1.get("weight", 55.0) + hs2.get("weight", 55.0),
            "diff_horse_weight": hs1.get("horse_weight", 480) - hs2.get("horse_weight", 480),
            "sum_horse_weight": hs1.get("horse_weight", 480) + hs2.get("horse_weight", 480),
            "diff_weight_diff": hs1.get("weight_diff", 0) - hs2.get("weight_diff", 0),
            "sum_weight_diff": hs1.get("weight_diff", 0) + hs2.get("weight_diff", 0),
        })

    return records


def _parse_race_info(soup: BeautifulSoup) -> dict:
    info = {}
    el = soup.find(class_=re.compile(r'RaceData01|race_data', re.I))
    if not el:
        return info
    text = el.get_text()
    m = re.search(r'(芝|ダート)\s*(\d{3,4})', text)
    if m:
        info["surface"] = m.group(1)
        info["distance"] = int(m.group(2))
    for g in ["G1", "G2", "G3", "L", "OP"]:
        if g in text:
            info["grade"] = g
            break
    else:
        info["grade"] = "一般"
    for c in ["不良", "重", "稍重", "良"]:
        if c in text:
            info["condition"] = c
            break
    for w in ["雨", "曇", "晴"]:
        if w in text:
            info["weather"] = w
            break
    return info


def _parse_result_ranks(soup: BeautifulSoup) -> dict:
    ranks = {}
    table = soup.find(class_=re.compile(r'RaceTable|race_table_01', re.I))
    if not table:
        return ranks
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        rank_text = cells[0].get_text(strip=True)
        horse_text = cells[2].get_text(strip=True) if len(cells) > 2 else ""
        if rank_text.isdigit() and horse_text.isdigit():
            ranks[int(rank_text)] = int(horse_text)
    return ranks


def _parse_quinella_payout(soup: BeautifulSoup) -> int:
    text = soup.get_text()
    m = re.search(r'馬連[^\d]*(\d[\d,]+)円', text)
    if m:
        return int(m.group(1).replace(",", ""))
    return 0


def _parse_horse_nos(soup: BeautifulSoup) -> list[int]:
    horse_nos = []
    table = soup.find(class_=re.compile(r'RaceTable|race_table_01', re.I))
    if not table:
        return horse_nos
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) > 2:
            t = cells[2].get_text(strip=True)
            if t.isdigit():
                horse_nos.append(int(t))
    return horse_nos


def _parse_jockey_stats_from_result(soup: BeautifulSoup, horse_nos: list) -> dict:
    """結果ページから騎手の簡易成績を取得（デフォルト値で補完）"""
    # 詳細な騎手成績取得はAPIコスト削減のためデフォルト値を使用
    # 実際のスクレイピング精度向上時に拡張する
    return {bn: {"win_rate": 0.15, "top2_rate": 0.30, "top3_rate": 0.45}
            for bn in horse_nos}


def _parse_horse_stats_from_result(soup: BeautifulSoup, horse_nos: list) -> dict:
    """結果ページから馬の基本情報を取得"""
    stats = {}
    table = soup.find(class_=re.compile(r'RaceTable|race_table_01', re.I))
    if not table:
        return {bn: {} for bn in horse_nos}

    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) < 6:
            continue
        horse_no_text = cells[2].get_text(strip=True) if len(cells) > 2 else ""
        if not horse_no_text.isdigit():
            continue
        horse_no = int(horse_no_text)

        # 斤量
        weight = 55.0
        for cell in cells:
            t = cell.get_text(strip=True)
            if re.match(r'^\d{2}\.\d$', t):
                try:
                    weight = float(t)
                except ValueError:
                    pass
                break

        # 馬体重・増減
        horse_weight, weight_diff = 480, 0
        for cell in cells:
            t = cell.get_text(strip=True)
            m = re.match(r'(\d{3})\(([+-]?\d+)\)', t)
            if m:
                horse_weight = int(m.group(1))
                weight_diff = int(m.group(2))
                break

        stats[horse_no] = {
            "weight": weight,
            "horse_weight": horse_weight,
            "weight_diff": weight_diff,
            "past_avg_rank": 6.0,
            "past_win_rate": 0.10,
            "past_top3_rate": 0.30,
            "same_cond_rate": 0.30,
            "recent_form": 1,
        }

    return stats
