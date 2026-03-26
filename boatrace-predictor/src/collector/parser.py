"""
競艇公式テキストデータパーサー
番組表（bYYMMDD.txt）と競走成績（kYYMMDD.txt）を解析してDataFrameに変換する
"""

import re
from pathlib import Path

import pandas as pd


# 場コード → 場名のマッピング
VENUE_CODES = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島",
    "05": "多摩川", "06": "浜名湖", "07": "蒲郡", "08": "常滑",
    "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島",
    "17": "宮島", "18": "徳山", "19": "下関", "20": "若松",
    "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}


def parse_program(txt_path: Path) -> pd.DataFrame:
    """
    番組表テキストを解析して選手情報DataFrameを返す

    Returns:
        DataFrame columns:
          date, venue_code, venue_name, race_no, boat_no,
          racer_no, racer_name, age, branch, weight, grade,
          national_win_rate, national_2rate, local_win_rate, local_2rate,
          motor_no, motor_2rate, boat_no_equip, boat_2rate
    """
    try:
        with open(txt_path, encoding="cp932", errors="replace") as f:
            content = f.read()
    except Exception as e:
        print(f"[ERROR] ファイル読み込み失敗: {txt_path} - {e}")
        return pd.DataFrame()

    # 日付をファイル名から取得
    yymmdd = txt_path.stem[1:]  # b240101 → 240101
    date_str = f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"

    records = []
    lines = content.splitlines()

    venue_code = None
    race_no = None

    for i, line in enumerate(lines):
        # 場コード取得
        if re.match(r"^\d{2}BBGN", line):
            venue_code = line[:2]

        # レース番号取得
        race_match = re.match(r"[\s　]*(\d{1,2})Ｒ", line)
        if race_match:
            race_no = int(race_match.group(1))

        # 選手データ行のパース（艇番1〜6で始まる行）
        boat_match = re.match(
            r"^([1-6])\s+(\d{4})\s*(.+?)\s+(\d{2})\s+(\S+)\s+(\d{2,3})"
            r"\s+([AB][12])\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
            r"\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)",
            line
        )
        if boat_match and venue_code and race_no:
            g = boat_match.groups()
            records.append({
                "date": date_str,
                "venue_code": venue_code,
                "venue_name": VENUE_CODES.get(venue_code, "不明"),
                "race_no": race_no,
                "boat_no": int(g[0]),
                "racer_no": int(g[1]),
                "racer_name": g[2].strip(),
                "age": int(g[3]),
                "branch": g[4].strip(),
                "weight": int(g[5]),
                "grade": g[6],
                "national_win_rate": float(g[7]),
                "national_2rate": float(g[8]),
                "local_win_rate": float(g[9]),
                "local_2rate": float(g[10]),
                "motor_no": int(g[11]),
                "motor_2rate": float(g[12]),
                "boat_no_equip": int(g[13]),
                "boat_2rate": float(g[14]),
            })

    df = pd.DataFrame(records)
    print(f"[OK] 番組表パース完了: {txt_path.name} - {len(df)}行")
    return df


def parse_result(txt_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    競走成績テキストを解析して着順・払戻DataFrameを返す

    Returns:
        (着順DataFrame, 払戻DataFrame)

        着順 columns:
          date, venue_code, venue_name, race_no, rank, boat_no,
          racer_no, racer_name, motor_no, boat_no_equip,
          exhibition_time, course, start_timing, race_time

        払戻 columns:
          date, venue_code, venue_name, race_no, bet_type,
          combination, payout, popularity
    """
    try:
        with open(txt_path, encoding="cp932", errors="replace") as f:
            content = f.read()
    except Exception as e:
        print(f"[ERROR] ファイル読み込み失敗: {txt_path} - {e}")
        return pd.DataFrame(), pd.DataFrame()

    yymmdd = txt_path.stem[1:]  # k240101 → 240101
    date_str = f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"

    rank_records = []
    payout_records = []

    lines = content.splitlines()
    venue_code = None
    race_no = None

    i = 0
    while i < len(lines):
        line = lines[i]

        # 場コード取得
        if re.match(r"^\d{2}KBGN", line):
            venue_code = line[:2]

        # レース番号取得
        race_match = re.match(r"[\s　]*(\d{1,2})Ｒ", line)
        if race_match:
            race_no = int(race_match.group(1))

        # 着順行パース（1〜6着）
        rank_match = re.match(
            r"^([1-6])\s+([1-6])\s+(\d{4})\s*(.+?)\s+(\d+)\s+(\d+)"
            r"\s+([\d.]+)\s+([1-6])\s+([\d.F+L]+)\s+([\d:']+)",
            line
        )
        if rank_match and venue_code and race_no:
            g = rank_match.groups()
            rank_records.append({
                "date": date_str,
                "venue_code": venue_code,
                "venue_name": VENUE_CODES.get(venue_code, "不明"),
                "race_no": race_no,
                "rank": int(g[0]),
                "boat_no": int(g[1]),
                "racer_no": int(g[2]),
                "racer_name": g[3].strip(),
                "motor_no": int(g[4]),
                "boat_no_equip": int(g[5]),
                "exhibition_time": float(g[6]),
                "course": int(g[7]),
                "start_timing": g[8],
                "race_time": g[9],
            })

        # 払戻行パース
        # 3連単: 例「３連単  1-2-3  12340円  人気 5」
        payout_match = re.match(
            r"^(３連単|３連複|２連単|２連複|単勝|複勝|拡連複)\s+"
            r"([\d\-]+)\s+([\d,]+)円?\s*(?:人気\s+(\d+))?",
            line
        )
        if payout_match and venue_code and race_no:
            g = payout_match.groups()
            payout_records.append({
                "date": date_str,
                "venue_code": venue_code,
                "venue_name": VENUE_CODES.get(venue_code, "不明"),
                "race_no": race_no,
                "bet_type": g[0],
                "combination": g[1],
                "payout": int(g[2].replace(",", "")),
                "popularity": int(g[3]) if g[3] else None,
            })

        i += 1

    df_rank = pd.DataFrame(rank_records)
    df_payout = pd.DataFrame(payout_records)
    print(f"[OK] 競走成績パース完了: {txt_path.name} - 着順{len(df_rank)}行 払戻{len(df_payout)}行")
    return df_rank, df_payout


def parse_all_programs(txt_dir: Path) -> pd.DataFrame:
    """ディレクトリ内の番組表テキストをすべてパースして結合"""
    dfs = []
    for txt_path in sorted(txt_dir.glob("b*.txt")):
        df = parse_program(txt_path)
        if not df.empty:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def parse_all_results(txt_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ディレクトリ内の競走成績テキストをすべてパースして結合"""
    rank_dfs, payout_dfs = [], []
    for txt_path in sorted(txt_dir.glob("k*.txt")):
        df_rank, df_payout = parse_result(txt_path)
        if not df_rank.empty:
            rank_dfs.append(df_rank)
        if not df_payout.empty:
            payout_dfs.append(df_payout)

    df_rank = pd.concat(rank_dfs, ignore_index=True) if rank_dfs else pd.DataFrame()
    df_payout = pd.concat(payout_dfs, ignore_index=True) if payout_dfs else pd.DataFrame()
    return df_rank, df_payout
