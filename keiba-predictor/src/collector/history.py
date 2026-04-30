"""
JRA過去レース成績一括取得モジュール
JRA公式データファイル（seiseki.jra.go.jp）から学習用データを収集する
"""

import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from itertools import combinations

import requests
from bs4 import BeautifulSoup
import pandas as pd

from src.features.builder import CONDITION_NUM, SURFACE_NUM

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
GRADE_NUM = {"G1": 5, "G2": 4, "G3": 3, "L": 2, "OP": 2, "一般": 1}


def fetch_history(months: int = 3, interval: float = 1.0) -> pd.DataFrame:
    """
    過去n月分のJRAレース成績を取得して学習用DataFrameを返す
    """
    today = datetime.now()
    target_dates = _get_race_dates(today, months)
    print(f"=== 過去データ取得開始: {len(target_dates)}日分（土日） ===")

    all_records = []
    for i, date in enumerate(target_dates):
        print(f"[{i+1}/{len(target_dates)}] {date.strftime('%Y-%m-%d')} 取得中...")
        records = _fetch_day(date, interval)
        if records:
            all_records.extend(records)
            print(f"  → {len(records)}組み合わせ取得")
        time.sleep(interval)

    if not all_records:
        print("[WARN] データが取得できませんでした")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    print(f"=== 取得完了: {len(df)}組み合わせ / 的中率:{df['result'].mean()*100:.2f}% ===")
    return df


def _get_race_dates(today: datetime, months: int) -> list[datetime]:
    """過去n月の土日リストを返す"""
    dates = []
    current = today - timedelta(days=1)
    cutoff = today - timedelta(days=months * 30)
    while current >= cutoff:
        if current.weekday() in (5, 6):
            dates.append(current)
        current -= timedelta(days=1)
    return list(reversed(dates))


def _fetch_day(date: datetime, interval: float) -> list[dict]:
    """1日分の全レース結果を取得する"""
    date_str = date.strftime("%Y%m%d")
    race_links = _fetch_netkeiba_race_list(date_str)

    if not race_links:
        print(f"  [INFO] {date_str} 開催なし（またはデータ未取得）")
        return []

    records = []
    for link_info in race_links:
        r = _fetch_race_result(link_info, date)
        if r:
            records.extend(r)
        time.sleep(interval)

    return records


def _fetch_netkeiba_race_list(date_str: str) -> list[dict]:
    """netkeibaからrace_idリストを取得する"""
    urls = [
        # db.netkeiba.com 日付別レース一覧（最も安定）
        f"https://db.netkeiba.com/race/list/{date_str}/",
        # race.netkeiba.com トップページのリスト
        f"https://race.netkeiba.com/top/race_list_2.html?kaisai_date={date_str}",
        # Yahoo競馬
        f"https://keiba.yahoo.co.jp/race/list/{date_str}/",
    ]

    for url in urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"  [DEBUG] HTTP{resp.status_code}: {url[:60]}")
                continue
            for enc in ["EUC-JP", "utf-8"]:
                resp.encoding = enc
                text = resp.text
                ids_found = set()
                for m in re.finditer(r'race_id=(\d{10,12})', text):
                    ids_found.add(m.group(1))
                for m in re.finditer(r'/race/(\d{10,12})/', text):
                    rid = m.group(1)
                    # 日付一致チェック（race_idの先頭4桁=年）
                    if rid.startswith(date_str[:4]):
                        ids_found.add(rid)
                if ids_found:
                    # 有効なJRA race_idのみ残す（12桁: venue=01-10, race_no=01-12）
                    valid = []
                    for rid in sorted(ids_found):
                        if len(rid) == 12:
                            venue = int(rid[4:6])
                            race_no = int(rid[10:12])
                            if 1 <= venue <= 10 and 1 <= race_no <= 12:
                                valid.append({"race_id": rid})
                    if valid:
                        print(f"  [OK] {len(valid)}レース発見 ({url[:60]})")
                        return valid
        except requests.RequestException as e:
            print(f"  [WARN] {url[:60]}: {e}")
            continue

    return []


def _fetch_race_result(link_info: dict, date: datetime) -> list[dict]:
    """1レース分の結果を取得して馬連組み合わせレコードを返す"""
    race_id = link_info.get("race_id")
    if not race_id:
        return []

    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"    [SKIP] {race_id}: {e}")
        return []

    records = _parse_result_to_records(soup, race_id, date)
    if records:
        winner_combo = next((r["combination"] for r in records if r["result"] == 1), "-")
        print(f"    {race_id}: {len(records)}通り 馬連={winner_combo}")
    else:
        print(f"    {race_id}: パース失敗")
    return records


def _parse_result_to_records(soup: BeautifulSoup, race_id: str, date: datetime) -> list[dict]:
    """レース結果ページをパースして馬連組み合わせレコードを生成する"""

    # レース情報
    race_info_el = soup.find(class_="race_otherdata") or soup.find("div", class_=re.compile(r'data_intro'))
    grade = "一般"
    surface = "芝"
    distance = 2000
    condition = "良"
    weather = "晴"

    if race_info_el:
        text = race_info_el.get_text()
        m = re.search(r'(芝|ダート)\s*(\d{3,4})', text)
        if m:
            surface = m.group(1)
            distance = int(m.group(2))
        for g in ["G1", "G2", "G3", "オープン", "OP"]:
            if g in text:
                grade = g.replace("オープン", "OP")
                break
        for c in ["不良", "重", "稍重", "良"]:
            if c in text:
                condition = c
                break
        for w in ["雨", "曇", "晴"]:
            if w in text:
                weather = w
                break

    # 着順テーブル
    table = soup.find("table", class_=re.compile(r'race_table_01|Shutuba_HorseList'))
    if not table:
        return []

    horse_data = {}
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) < 6:
            continue
        try:
            rank_text = cells[0].get_text(strip=True)
            rank = int(rank_text) if rank_text.isdigit() else 99
            horse_no_text = cells[2].get_text(strip=True) if len(cells) > 2 else ""
            if not horse_no_text.isdigit():
                continue
            horse_no = int(horse_no_text)

            weight = 55.0
            horse_weight = 480
            weight_diff = 0
            for cell in cells:
                t = cell.get_text(strip=True)
                if re.match(r'^\d{2}\.\d$', t):
                    weight = float(t)
                mw = re.match(r'(\d{3})\(([+-]?\d+)\)', t)
                if mw:
                    horse_weight = int(mw.group(1))
                    weight_diff = int(mw.group(2))

            horse_data[horse_no] = {
                "rank": rank,
                "weight": weight,
                "horse_weight": horse_weight,
                "weight_diff": weight_diff,
            }
        except (ValueError, IndexError):
            continue

    if len(horse_data) < 2:
        return []

    # 1着・2着を確認
    ranked = {v["rank"]: k for k, v in horse_data.items() if v["rank"] <= 2}
    if 1 not in ranked or 2 not in ranked:
        return []
    winner = ranked[1]
    second = ranked[2]
    winning_combo = f"{min(winner, second)}-{max(winner, second)}"

    # 払戻取得
    quinella_payout = 0
    text = soup.get_text()
    m = re.search(r'馬連[^\d]*(\d[\d,]+)', text)
    if m:
        quinella_payout = int(m.group(1).replace(",", ""))

    # 特徴量
    grade_num = GRADE_NUM.get(grade, 1)
    condition_num = CONDITION_NUM.get(condition, 4)
    surface_num = SURFACE_NUM.get(surface, 1)
    weather_num = {"晴": 3, "曇": 2, "雨": 1}.get(weather, 3)

    records = []
    horse_nos = sorted(horse_data.keys())
    for h1, h2 in combinations(horse_nos, 2):
        combo = f"{h1}-{h2}"
        result = 1 if combo == winning_combo else 0
        d1 = horse_data[h1]
        d2 = horse_data[h2]

        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "race_id": race_id,
            "combination": combo,
            "horse1": h1,
            "horse2": h2,
            "result": result,
            "quinella_payout": quinella_payout if result == 1 else 0,
            "grade_num": grade_num,
            "condition_num": condition_num,
            "weather_num": weather_num,
            "surface_num": surface_num,
            "distance": distance,
            "h1_weight": d1["weight"],
            "h1_horse_weight": d1["horse_weight"],
            "h1_weight_diff": d1["weight_diff"],
            "h1_jockey_win_rate": 0.15,
            "h1_jockey_top2_rate": 0.30,
            "h1_jockey_top3_rate": 0.45,
            "h1_past_avg_rank": 6.0,
            "h1_past_win_rate": 0.10,
            "h1_past_top3_rate": 0.30,
            "h1_same_cond_rate": 0.30,
            "h1_recent_form": 1,
            "h2_weight": d2["weight"],
            "h2_horse_weight": d2["horse_weight"],
            "h2_weight_diff": d2["weight_diff"],
            "h2_jockey_win_rate": 0.15,
            "h2_jockey_top2_rate": 0.30,
            "h2_jockey_top3_rate": 0.45,
            "h2_past_avg_rank": 6.0,
            "h2_past_win_rate": 0.10,
            "h2_past_top3_rate": 0.30,
            "h2_same_cond_rate": 0.30,
            "h2_recent_form": 1,
            "diff_jockey_win_rate": 0.0,
            "sum_jockey_win_rate": 0.30,
            "diff_jockey_top2_rate": 0.0,
            "sum_jockey_top2_rate": 0.60,
            "diff_jockey_top3_rate": 0.0,
            "sum_jockey_top3_rate": 0.90,
            "diff_past_avg_rank": 0.0,
            "sum_past_avg_rank": 12.0,
            "diff_past_win_rate": 0.0,
            "sum_past_win_rate": 0.20,
            "diff_past_top3_rate": 0.0,
            "sum_past_top3_rate": 0.60,
            "diff_same_cond_rate": 0.0,
            "sum_same_cond_rate": 0.60,
            "diff_recent_form": 0,
            "sum_recent_form": 2,
            "diff_weight": d1["weight"] - d2["weight"],
            "sum_weight": d1["weight"] + d2["weight"],
            "diff_horse_weight": d1["horse_weight"] - d2["horse_weight"],
            "sum_horse_weight": d1["horse_weight"] + d2["horse_weight"],
            "diff_weight_diff": d1["weight_diff"] - d2["weight_diff"],
            "sum_weight_diff": d1["weight_diff"] + d2["weight_diff"],
        })

    return records
