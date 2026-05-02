"""
特徴量エンジニアリング
出馬表・成績データから予測モデル用の特徴量を生成する
"""

import numpy as np
import pandas as pd

# グレード数値化
GRADE_NUM = {"G1": 5, "G2": 4, "G3": 3, "L": 2, "OP": 2, "3勝": 1, "2勝": 1, "1勝": 1, "未勝利": 0, "新馬": 0, "一般": 1}

# 馬場状態数値化（良いほど高い）
CONDITION_NUM = {"良": 4, "稍重": 3, "重": 2, "不良": 1}

# 天候数値化
WEATHER_NUM = {"晴": 3, "曇": 2, "雨": 1}

# 馬場種別
SURFACE_NUM = {"芝": 1, "ダート": 0}


def build_race_features(race_card: dict, jockey_stats: dict = None,
                        horse_histories: dict = None) -> pd.DataFrame:
    """
    1レース分の出馬表から特徴量DataFrameを生成する（1行=1頭）

    Args:
        race_card: fetch_race_card()の出力
        jockey_stats: {jockey_name: {"win_rate": ..., "top2_rate": ..., "top3_rate": ...}}
        horse_histories: {horse_name: [過去成績リスト]}

    Returns:
        DataFrame（1行=1頭、horse_noでソート）
    """
    if not race_card or not race_card.get("horses"):
        return pd.DataFrame()

    grade_num = GRADE_NUM.get(race_card.get("grade", "一般"), 1)
    condition_num = CONDITION_NUM.get(race_card.get("condition", "良"), 4)
    weather_num = WEATHER_NUM.get(race_card.get("weather", "晴"), 3)
    surface_num = SURFACE_NUM.get(race_card.get("surface", "芝"), 1)
    distance = int(race_card.get("distance", 2000))

    rows = []
    for horse in race_card["horses"]:
        horse_no = horse["horse_no"]
        jockey = horse.get("jockey", "")
        horse_name = horse.get("horse_name", "")

        # 騎手成績
        j_stats = (jockey_stats or {}).get(jockey, {})
        jockey_win_rate = j_stats.get("win_rate", 0.15)
        jockey_top2_rate = j_stats.get("top2_rate", 0.30)
        jockey_top3_rate = j_stats.get("top3_rate", 0.45)

        # 馬の過去成績（直近5走）
        history = (horse_histories or {}).get(horse_name, [])
        past_avg_rank = _calc_avg_rank(history)
        past_win_rate = _calc_win_rate(history)
        past_top3_rate = _calc_top3_rate(history)
        # 同距離・同馬場での成績
        same_cond_rate = _calc_same_condition_rate(
            history, surface_num, distance
        )
        # 連続好走（直近3走で3着以内の数）
        recent_form = _calc_recent_form(history, n=3)

        rows.append({
            "date": race_card.get("date", ""),
            "venue_code": race_card.get("venue_code", ""),
            "venue": race_card.get("venue", ""),
            "race_no": race_card.get("race_no", 0),
            "horse_no": horse_no,
            "horse_name": horse_name,
            "jockey": jockey,
            # レース条件（全馬共通）
            "grade_num": grade_num,
            "condition_num": condition_num,
            "weather_num": weather_num,
            "surface_num": surface_num,
            "distance": distance,
            # 馬の特徴量
            "weight": horse.get("weight", 55.0),
            "horse_weight": horse.get("horse_weight", 480),
            "weight_diff": horse.get("weight_diff", 0),
            # 騎手成績
            "jockey_win_rate": jockey_win_rate,
            "jockey_top2_rate": jockey_top2_rate,
            "jockey_top3_rate": jockey_top3_rate,
            # 馬の過去成績
            "past_avg_rank": past_avg_rank,
            "past_win_rate": past_win_rate,
            "past_top3_rate": past_top3_rate,
            "same_cond_rate": same_cond_rate,
            "recent_form": recent_form,
        })

    return pd.DataFrame(rows).sort_values("horse_no").reset_index(drop=True)


def build_quinella_features(df_race: pd.DataFrame) -> pd.DataFrame:
    """
    馬ごとの特徴量から馬連の全組み合わせ特徴量を生成する

    Returns:
        DataFrame（1行=馬連1通り）
        列: horse1, horse2, 各馬の特徴量の差・和・積など
    """
    from itertools import combinations

    if df_race.empty or len(df_race) < 2:
        return pd.DataFrame()

    feature_cols = [
        "grade_num", "condition_num", "weather_num", "surface_num", "distance",
        "weight", "horse_weight", "weight_diff",
        "jockey_win_rate", "jockey_top2_rate", "jockey_top3_rate",
        "past_avg_rank", "past_win_rate", "past_top3_rate",
        "same_cond_rate", "recent_form",
    ]

    rows = []
    horse_nos = sorted(set(df_race["horse_no"].tolist()))

    for h1, h2 in combinations(horse_nos, 2):
        r1 = df_race[df_race["horse_no"] == h1].iloc[0]
        r2 = df_race[df_race["horse_no"] == h2].iloc[0]

        row = {
            "date": r1["date"],
            "venue_code": r1["venue_code"],
            "race_no": r1["race_no"],
            "horse1": h1,
            "horse2": h2,
            "combination": f"{h1}-{h2}",
        }

        # レース条件（共通値）
        for col in ["grade_num", "condition_num", "weather_num", "surface_num", "distance"]:
            row[col] = r1[col]

        # 各馬の個別特徴量
        indiv_cols = [
            "weight", "horse_weight", "weight_diff",
            "jockey_win_rate", "jockey_top2_rate", "jockey_top3_rate",
            "past_avg_rank", "past_win_rate", "past_top3_rate",
            "same_cond_rate", "recent_form",
        ]
        for col in indiv_cols:
            row[f"h1_{col}"] = r1[col]
            row[f"h2_{col}"] = r2[col]
            row[f"diff_{col}"] = r1[col] - r2[col]
            row[f"sum_{col}"] = r1[col] + r2[col]

        rows.append(row)

    return pd.DataFrame(rows)


def get_feature_columns() -> list[str]:
    """モデルに使用する特徴量カラム名リストを返す"""
    cols = [
        "grade_num", "condition_num", "weather_num", "surface_num", "distance",
    ]
    indiv_cols = [
        "weight", "horse_weight", "weight_diff",
        "jockey_win_rate", "jockey_top2_rate", "jockey_top3_rate",
        "past_avg_rank", "past_win_rate", "past_top3_rate",
        "same_cond_rate", "recent_form",
    ]
    for col in indiv_cols:
        cols += [f"h1_{col}", f"h2_{col}", f"diff_{col}", f"sum_{col}"]
    return cols


# ─── 過去成績集計ヘルパー ───

def _calc_avg_rank(history: list) -> float:
    ranks = [h["rank"] for h in history if h.get("rank", 99) < 99]
    return float(np.mean(ranks)) if ranks else 8.0


def _calc_win_rate(history: list) -> float:
    if not history:
        return 0.10
    wins = sum(1 for h in history if h.get("rank") == 1)
    return wins / len(history)


def _calc_top3_rate(history: list) -> float:
    if not history:
        return 0.30
    top3 = sum(1 for h in history if h.get("rank", 99) <= 3)
    return top3 / len(history)


def _calc_same_condition_rate(history: list, surface_num: int, distance: int) -> float:
    """同馬場・同距離帯（±200m）での3着以内率"""
    same = [h for h in history
            if SURFACE_NUM.get(h.get("surface", "芝"), 1) == surface_num
            and abs(h.get("distance", 0) - distance) <= 200]
    if not same:
        return 0.30
    top3 = sum(1 for h in same if h.get("rank", 99) <= 3)
    return top3 / len(same)


def _calc_recent_form(history: list, n: int = 3) -> float:
    """直近n走での3着以内数（0〜n）"""
    recent = history[:n]
    return sum(1 for h in recent if h.get("rank", 99) <= 3)
