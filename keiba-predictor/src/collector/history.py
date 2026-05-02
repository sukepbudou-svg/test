"""
JRA過去レース成績一括取得モジュール
netkeibaから学習用データを収集する
"""

import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

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

GRADE_NUM = {"G1": 5, "G2": 4, "G3": 3, "L": 2, "OP": 2, "一般": 1}


def fetch_history(months: int = 3, interval: float = 1.0) -> pd.DataFrame:
    """過去n月分のJRAレース成績を取得して学習用DataFrameを返す"""
    today = datetime.now()
    target_dates = _get_race_dates(today, months)
    print(f"=== 過去データ取得開始: {len(target_dates)}日分（土日） ===")

    all_combo_records = []
    all_raw_records = []
    for i, date in enumerate(target_dates):
        print(f"[{i+1}/{len(target_dates)}] {date.strftime('%Y-%m-%d')} 取得中...")
        combos, raw = _fetch_day(date, interval)
        if combos:
            all_combo_records.extend(combos)
            all_raw_records.extend(raw)
            print(f"  → {len(combos)}組み合わせ取得")
        time.sleep(interval)

    if not all_combo_records:
        print("[WARN] データが取得できませんでした")
        return pd.DataFrame()

    print("  騎手・馬の過去成績を計算中...")
    df = _build_enriched_df(all_combo_records, all_raw_records)
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


def _fetch_day(date: datetime, interval: float) -> tuple[list, list]:
    """1日分の全レース結果を取得する。(combo_records, raw_horse_records)を返す"""
    date_str = date.strftime("%Y%m%d")
    race_links = _fetch_netkeiba_race_list(date_str)
    if not race_links:
        print(f"  [INFO] {date_str} 開催なし（またはデータ未取得）")
        return [], []

    all_combos, all_raw = [], []
    for link_info in race_links:
        combos, raw = _fetch_race_result(link_info, date)
        all_combos.extend(combos)
        all_raw.extend(raw)
        time.sleep(interval)
    return all_combos, all_raw


def _fetch_netkeiba_race_list(date_str: str) -> list[dict]:
    """netkeibaからrace_idリストを取得する"""
    urls = [
        f"https://db.netkeiba.com/race/list/{date_str}/",
        f"https://race.netkeiba.com/top/race_list_2.html?kaisai_date={date_str}",
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
                    if rid.startswith(date_str[:4]):
                        ids_found.add(rid)
                if ids_found:
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


def _fetch_race_result(link_info: dict, date: datetime) -> tuple[list, list]:
    """1レース分の結果を取得する。(combo_records, raw_horse_records)を返す"""
    race_id = link_info.get("race_id")
    if not race_id:
        return [], []

    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"    [SKIP] {race_id}: {e}")
        return [], []

    combos, raw = _parse_result_to_records(soup, race_id, date)
    if combos:
        winner = next((r["combination"] for r in combos if r["result"] == 1), "-")
        print(f"    {race_id}: {len(combos)}通り 馬連={winner}")
    else:
        print(f"    {race_id}: パース失敗")
    return combos, raw


def _parse_result_to_records(soup: BeautifulSoup, race_id: str, date: datetime) -> tuple[list, list]:
    """レース結果ページをパースして (combo_records, raw_horse_records) を返す"""

    # レース情報
    race_info_el = (soup.find(class_="race_otherdata")
                    or soup.find("div", class_=re.compile(r'data_intro')))
    grade, surface, distance, condition, weather = "一般", "芝", 2000, "良", "晴"
    if race_info_el:
        t = race_info_el.get_text()
        m = re.search(r'(芝|ダート)\s*(\d{3,4})', t)
        if m:
            surface = m.group(1)
            distance = int(m.group(2))
        for g in ["G1", "G2", "G3", "オープン", "OP"]:
            if g in t:
                grade = g.replace("オープン", "OP")
                break
        for c in ["不良", "重", "稍重", "良"]:
            if c in t:
                condition = c
                break
        for w in ["雨", "曇", "晴"]:
            if w in t:
                weather = w
                break

    grade_num = GRADE_NUM.get(grade, 1)
    condition_num = CONDITION_NUM.get(condition, 4)
    surface_num = SURFACE_NUM.get(surface, 1)
    weather_num = {"晴": 3, "曇": 2, "雨": 1}.get(weather, 3)

    # 着順テーブル
    table = soup.find("table", class_=re.compile(r'race_table_01|Shutuba_HorseList'))
    if not table:
        return [], []

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

            # 馬名・馬ID（リンクから取得）
            horse_name, horse_id = "", ""
            h_link = row.find("a", href=re.compile(r'/horse/'))
            if h_link:
                horse_name = h_link.get_text(strip=True)
                m = re.search(r'/horse/(\d+)', h_link.get("href", ""))
                horse_id = m.group(1) if m else ""

            # 騎手・騎手ID（リンクから取得）
            jockey, jockey_id = "", ""
            j_link = row.find("a", href=re.compile(r'/jockey/'))
            if j_link:
                jockey = j_link.get_text(strip=True)
                m = re.search(r'/jockey/(\w+)', j_link.get("href", ""))
                jockey_id = m.group(1) if m else ""

            weight, horse_weight, weight_diff, win_odds = 55.0, 480, 0, 0.0
            for cell in cells:
                t = cell.get_text(strip=True).replace(",", "")
                mw = re.match(r'(\d{3})\(([+-]?\d+)\)', t)
                if mw:
                    horse_weight = int(mw.group(1))
                    weight_diff = int(mw.group(2))
                    continue
                mv = re.match(r'^(\d+\.\d)$', t)
                if mv:
                    val = float(mv.group(1))
                    if 50.0 <= val <= 60.0:
                        weight = val
                    elif val >= 1.0 and win_odds == 0.0:
                        win_odds = val

            horse_data[horse_no] = {
                "rank": rank,
                "horse_name": horse_name,
                "horse_id": horse_id,
                "jockey": jockey,
                "jockey_id": jockey_id,
                "weight": weight,
                "horse_weight": horse_weight,
                "weight_diff": weight_diff,
                "win_odds": win_odds,
            }
        except (ValueError, IndexError):
            continue

    if len(horse_data) < 2:
        return [], []

    # 人気順位: win_odds昇順でランク付け（0.0=不明は最下位）
    sorted_by_odds = sorted(
        horse_data.items(),
        key=lambda x: x[1]["win_odds"] if x[1]["win_odds"] > 0 else float("inf")
    )
    for pop_rank, (hn, _) in enumerate(sorted_by_odds, 1):
        horse_data[hn]["popularity"] = pop_rank

    ranked = {v["rank"]: k for k, v in horse_data.items() if v["rank"] <= 2}
    if 1 not in ranked or 2 not in ranked:
        return [], []
    winner, second = ranked[1], ranked[2]
    winning_combo = f"{min(winner, second)}-{max(winner, second)}"

    # 払戻
    quinella_payout = 0
    text = soup.get_text()
    m = re.search(r'馬連[^\d]*(\d[\d,]+)', text)
    if m:
        quinella_payout = int(m.group(1).replace(",", ""))

    date_str = date.strftime("%Y-%m-%d")

    # 馬別生データ（ローリング集計用）
    raw_records = [
        {
            "date": date_str,
            "race_id": race_id,
            "horse_no": hn,
            "horse_name": hd["horse_name"],
            "jockey": hd["jockey"],
            "rank": hd["rank"],
            "surface_num": surface_num,
            "distance": distance,
        }
        for hn, hd in horse_data.items()
    ]

    # 馬連コンボレコード
    combo_records = []
    for h1, h2 in combinations(sorted(horse_data.keys()), 2):
        combo = f"{h1}-{h2}"
        d1, d2 = horse_data[h1], horse_data[h2]
        combo_records.append({
            "date": date_str,
            "race_id": race_id,
            "combination": combo,
            "horse1": h1,
            "horse2": h2,
            "horse_name1": d1["horse_name"],
            "horse_name2": d2["horse_name"],
            "jockey1": d1["jockey"],
            "jockey2": d2["jockey"],
            "result": 1 if combo == winning_combo else 0,
            "quinella_payout": quinella_payout if combo == winning_combo else 0,
            "grade_num": grade_num,
            "condition_num": condition_num,
            "weather_num": weather_num,
            "surface_num": surface_num,
            "distance": distance,
            "h1_win_odds": d1["win_odds"],
            "h1_popularity": d1["popularity"],
            "h1_weight": d1["weight"],
            "h1_horse_weight": d1["horse_weight"],
            "h1_weight_diff": d1["weight_diff"],
            "h1_jockey_win_rate": 0.15,
            "h1_jockey_top2_rate": 0.30,
            "h1_jockey_top3_rate": 0.45,
            "h1_past_avg_rank": 8.0,
            "h1_past_win_rate": 0.10,
            "h1_past_top3_rate": 0.30,
            "h1_same_cond_rate": 0.30,
            "h1_recent_form": 1,
            "h2_win_odds": d2["win_odds"],
            "h2_popularity": d2["popularity"],
            "h2_weight": d2["weight"],
            "h2_horse_weight": d2["horse_weight"],
            "h2_weight_diff": d2["weight_diff"],
            "h2_jockey_win_rate": 0.15,
            "h2_jockey_top2_rate": 0.30,
            "h2_jockey_top3_rate": 0.45,
            "h2_past_avg_rank": 8.0,
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
            "sum_past_avg_rank": 16.0,
            "diff_past_win_rate": 0.0,
            "sum_past_win_rate": 0.20,
            "diff_past_top3_rate": 0.0,
            "sum_past_top3_rate": 0.60,
            "diff_same_cond_rate": 0.0,
            "sum_same_cond_rate": 0.60,
            "diff_recent_form": 0,
            "sum_recent_form": 2,
            "diff_win_odds": d1["win_odds"] - d2["win_odds"],
            "sum_win_odds": d1["win_odds"] + d2["win_odds"],
            "diff_popularity": d1["popularity"] - d2["popularity"],
            "sum_popularity": d1["popularity"] + d2["popularity"],
            "diff_weight": d1["weight"] - d2["weight"],
            "sum_weight": d1["weight"] + d2["weight"],
            "diff_horse_weight": d1["horse_weight"] - d2["horse_weight"],
            "sum_horse_weight": d1["horse_weight"] + d2["horse_weight"],
            "diff_weight_diff": d1["weight_diff"] - d2["weight_diff"],
            "sum_weight_diff": d1["weight_diff"] + d2["weight_diff"],
        })

    return combo_records, raw_records


def _build_enriched_df(combo_records: list, raw_records: list) -> pd.DataFrame:
    """コンボレコードに騎手・馬のローリング過去成績を付与する（データリークなし）"""
    df = pd.DataFrame(combo_records)
    if not raw_records:
        return df

    df_raw = (pd.DataFrame(raw_records)
              .sort_values(["date", "race_id", "horse_no"])
              .reset_index(drop=True))

    # ── 騎手スタッツをローリング計算（5戦以上で実績値、それ未満はデフォルト）──
    jockey_stats: dict[tuple, dict] = {}
    jockey_acc: dict[str, dict] = defaultdict(lambda: {"w": 0, "t2": 0, "t3": 0, "n": 0})

    for _, r in df_raw.iterrows():
        j, d = r["jockey"], r["date"]
        if not j:
            continue
        acc = jockey_acc[j]
        n = acc["n"]
        jockey_stats[(d, j)] = {
            "win_rate":  acc["w"]  / n if n >= 5 else 0.15,
            "top2_rate": acc["t2"] / n if n >= 5 else 0.30,
            "top3_rate": acc["t3"] / n if n >= 5 else 0.45,
        }
        acc["w"]  += 1 if r["rank"] == 1 else 0
        acc["t2"] += 1 if r["rank"] <= 2 else 0
        acc["t3"] += 1 if r["rank"] <= 3 else 0
        acc["n"]  += 1

    # ── 馬スタッツをローリング計算 ──
    horse_stats: dict[tuple, dict] = {}
    horse_acc: dict[str, list] = defaultdict(list)

    for _, r in df_raw.iterrows():
        h, d = r["horse_name"], r["date"]
        if not h:
            continue
        past = horse_acc[h]
        if past:
            past_ranks = [p for p in past if p < 99]
            avg_rank   = sum(past_ranks) / len(past_ranks) if past_ranks else 8.0
            win_rate   = sum(1 for p in past if p == 1) / len(past)
            top3_rate  = sum(1 for p in past if p <= 3) / len(past)
            recent     = past[-3:]
            recent_form = sum(1 for p in recent if p <= 3)
            # 同条件（馬場×距離±200m）での3着以内率
            same_cond = [
                p for p, s, dist in zip(
                    horse_acc[h + "_ranks"],
                    horse_acc[h + "_surface"],
                    horse_acc[h + "_dist"],
                )
                if s == r["surface_num"] and abs(dist - r["distance"]) <= 200
            ] if (h + "_ranks") in horse_acc else []
            same_cond_rate = (sum(1 for p in same_cond if p <= 3) / len(same_cond)
                              if same_cond else 0.30)
        else:
            avg_rank, win_rate, top3_rate, recent_form, same_cond_rate = 8.0, 0.10, 0.30, 1, 0.30

        horse_stats[(d, h)] = {
            "avg_rank": avg_rank,
            "win_rate": win_rate,
            "top3_rate": top3_rate,
            "recent_form": recent_form,
            "same_cond_rate": same_cond_rate,
        }
        past.append(r["rank"])
        horse_acc[h + "_ranks"].append(r["rank"])
        horse_acc[h + "_surface"].append(r["surface_num"])
        horse_acc[h + "_dist"].append(r["distance"])

    # ── DataFrameに反映 ──
    _DJ = {"win_rate": 0.15, "top2_rate": 0.30, "top3_rate": 0.45}
    _DH = {"avg_rank": 8.0, "win_rate": 0.10, "top3_rate": 0.30,
           "recent_form": 1, "same_cond_rate": 0.30}

    dates = df["date"].tolist()
    j1 = df.get("jockey1", pd.Series([""] * len(df))).tolist()
    j2 = df.get("jockey2", pd.Series([""] * len(df))).tolist()
    h1 = df.get("horse_name1", pd.Series([""] * len(df))).tolist()
    h2 = df.get("horse_name2", pd.Series([""] * len(df))).tolist()

    for stat, defval in [("win_rate", 0.15), ("top2_rate", 0.30), ("top3_rate", 0.45)]:
        v1 = [jockey_stats.get((d, j), _DJ)[stat] for d, j in zip(dates, j1)]
        v2 = [jockey_stats.get((d, j), _DJ)[stat] for d, j in zip(dates, j2)]
        df[f"h1_jockey_{stat}"] = v1
        df[f"h2_jockey_{stat}"] = v2
        df[f"diff_jockey_{stat}"] = [a - b for a, b in zip(v1, v2)]
        df[f"sum_jockey_{stat}"]  = [a + b for a, b in zip(v1, v2)]

    for stat, defval in [("avg_rank", 8.0), ("win_rate", 0.10), ("top3_rate", 0.30),
                         ("recent_form", 1), ("same_cond_rate", 0.30)]:
        col = "same_cond_rate" if stat == "same_cond_rate" else f"past_{stat}" if stat != "recent_form" else "recent_form"
        v1 = [horse_stats.get((d, h), _DH)[stat] for d, h in zip(dates, h1)]
        v2 = [horse_stats.get((d, h), _DH)[stat] for d, h in zip(dates, h2)]
        df[f"h1_{col}"] = v1
        df[f"h2_{col}"] = v2
        df[f"diff_{col}"] = [a - b for a, b in zip(v1, v2)]
        df[f"sum_{col}"]  = [a + b for a, b in zip(v1, v2)]

    return df
