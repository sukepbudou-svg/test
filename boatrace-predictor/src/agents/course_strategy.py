"""
コース戦略エージェント
競艇のコース有利・展示タイム・展示スタートタイミングをもとに
各艇の勝利確率を計算する
"""

import numpy as np
import pandas as pd

# ─── 全国平均コース別勝率（フォールバック用）───
COURSE_WIN_RATES = {
    1: 0.550, 2: 0.170, 3: 0.110,
    4: 0.080, 5: 0.055, 6: 0.035,
}

# ─── 競艇場別コース勝率プロファイル ───
# 実際の統計に基づいた各会場のコース別1着率
# 値が大きいほどそのコースが有利な会場
VENUE_COURSE_PROFILES = {
    # ── 荒れにくい会場（インコース強い） ──
    "大村":   {1: 0.610, 2: 0.155, 3: 0.100, 4: 0.070, 5: 0.045, 6: 0.020},
    "住之江": {1: 0.575, 2: 0.165, 3: 0.105, 4: 0.075, 5: 0.050, 6: 0.030},
    "尼崎":   {1: 0.560, 2: 0.168, 3: 0.108, 4: 0.077, 5: 0.052, 6: 0.035},
    "若松":   {1: 0.555, 2: 0.168, 3: 0.108, 4: 0.078, 5: 0.053, 6: 0.038},
    "芦屋":   {1: 0.550, 2: 0.170, 3: 0.110, 4: 0.080, 5: 0.055, 6: 0.035},
    "多摩川": {1: 0.548, 2: 0.170, 3: 0.110, 4: 0.082, 5: 0.055, 6: 0.035},
    "常滑":   {1: 0.545, 2: 0.170, 3: 0.112, 4: 0.082, 5: 0.056, 6: 0.035},
    "唐津":   {1: 0.542, 2: 0.172, 3: 0.112, 4: 0.082, 5: 0.057, 6: 0.035},
    "下関":   {1: 0.540, 2: 0.172, 3: 0.113, 4: 0.082, 5: 0.057, 6: 0.036},
    "福岡":   {1: 0.538, 2: 0.172, 3: 0.113, 4: 0.083, 5: 0.058, 6: 0.036},

    # ── 普通の会場 ──
    "蒲郡":   {1: 0.535, 2: 0.173, 3: 0.113, 4: 0.083, 5: 0.058, 6: 0.038},
    "丸亀":   {1: 0.533, 2: 0.173, 3: 0.114, 4: 0.083, 5: 0.059, 6: 0.038},
    "児島":   {1: 0.530, 2: 0.174, 3: 0.114, 4: 0.084, 5: 0.060, 6: 0.038},
    "宮島":   {1: 0.525, 2: 0.174, 3: 0.115, 4: 0.085, 5: 0.061, 6: 0.040},
    "津":     {1: 0.522, 2: 0.175, 3: 0.115, 4: 0.085, 5: 0.062, 6: 0.041},
    "徳山":   {1: 0.520, 2: 0.175, 3: 0.116, 4: 0.086, 5: 0.062, 6: 0.041},
    "桐生":   {1: 0.518, 2: 0.175, 3: 0.116, 4: 0.086, 5: 0.063, 6: 0.042},
    "鳴門":   {1: 0.515, 2: 0.176, 3: 0.116, 4: 0.087, 5: 0.063, 6: 0.043},
    "平和島": {1: 0.510, 2: 0.176, 3: 0.117, 4: 0.088, 5: 0.064, 6: 0.045},

    # ── 荒れやすい会場（アウト有利・風の影響大） ──
    "びわこ": {1: 0.505, 2: 0.177, 3: 0.118, 4: 0.088, 5: 0.065, 6: 0.047},
    "三国":   {1: 0.500, 2: 0.178, 3: 0.119, 4: 0.089, 5: 0.066, 6: 0.048},
    "浜名湖": {1: 0.490, 2: 0.180, 3: 0.120, 4: 0.090, 5: 0.068, 6: 0.052},
    "戸田":   {1: 0.460, 2: 0.182, 3: 0.130, 4: 0.100, 5: 0.075, 6: 0.053},
    "江戸川": {1: 0.400, 2: 0.185, 3: 0.145, 4: 0.120, 5: 0.090, 6: 0.060},
}

# ─── レース番号による荒れやすさ補正 ───
# 後半レースほど荒れやすい傾向がある
# インコース勝率への補正係数
RACE_NO_INNER_FACTOR = {
    1:  1.03,   # 序盤は本命決着多い
    2:  1.02,
    3:  1.01,
    4:  1.00,
    5:  1.00,
    6:  0.99,
    7:  0.98,
    8:  0.97,
    9:  0.96,
    10: 0.95,   # 後半は荒れやすい
    11: 0.94,
    12: 0.93,   # 最終レースが最も荒れやすい
}

# 展示タイム基準値（全国平均）
EXHTIME_BASELINE = 6.70
EXHTIME_WEIGHT   = 0.15

# 展示スタート基準値
EXHST_BASELINE = 0.15
EXHST_WEIGHT   = 0.10


def predict_win_probs(race_row: pd.Series) -> np.ndarray:
    """
    コース戦略ベースで各艇（1〜6号艇）の勝率を計算する

    Args:
        race_row: 1レース分の特徴量Series
            必要列: venue_name
            任意列: boat{n}_exhibition_time, boat{n}_exhibition_st

    Returns:
        shape=(6,) の勝率配列（インデックス0=1号艇, ..., 5=6号艇）
        合計は1.0
    """
    venue_name = str(race_row.get("venue_name", ""))
    race_no = int(race_row.get("race_no", 6))

    # 会場別コース勝率プロファイルを取得（なければ全国平均）
    venue_profile = VENUE_COURSE_PROFILES.get(venue_name, COURSE_WIN_RATES)

    # レース番号補正係数
    race_inner_factor = RACE_NO_INNER_FACTOR.get(race_no, 1.0)

    # ── レース内の展示ST平均を計算（相対比較用）──
    st_vals_in_race = []
    for b in range(1, 7):
        v = race_row.get(f"boat{b}_exhibition_st", None)
        if v is not None:
            try:
                fv = float(v)
                if fv >= 0:
                    st_vals_in_race.append(fv)
            except (ValueError, TypeError):
                pass
    race_avg_st = float(np.mean(st_vals_in_race)) if st_vals_in_race else EXHST_BASELINE

    # ── レース内の展示タイム平均を計算（相対比較用）──
    exhtime_vals_in_race = []
    for b in range(1, 7):
        v = race_row.get(f"boat{b}_exhibition_time", None)
        if v and float(v) > 0:
            exhtime_vals_in_race.append(float(v))
    race_avg_exhtime = float(np.mean(exhtime_vals_in_race)) if exhtime_vals_in_race else EXHTIME_BASELINE

    probs = np.zeros(6)

    for boat in range(1, 7):
        course = boat  # デフォルト: 艇番=コース番号

        # 会場別プロファイルからコース勝率を取得
        base_rate = venue_profile.get(course, COURSE_WIN_RATES.get(course, 0.05))

        # レース番号補正（インコースは後半レースほど弱まる、外コースは逆）
        if course == 1:
            base_rate *= race_inner_factor
        elif course >= 4:
            # 外コースは後半レースで有利になる
            base_rate *= (2.0 - race_inner_factor)

        # ── 展示タイム補正（レース内相対比較）──
        exh_time = race_row.get(f"boat{boat}_exhibition_time", None)
        if exh_time and float(exh_time) > 0:
            # レース内平均より速いほどプラス補正
            diff = race_avg_exhtime - float(exh_time)
            base_rate *= (1.0 + diff * EXHTIME_WEIGHT)

        # ── 展示スタートタイミング補正（レース内相対比較）──
        exh_st = race_row.get(f"boat{boat}_exhibition_st", None)
        if exh_st is not None:
            try:
                st_val = float(exh_st)
                if st_val >= 0:  # F(フライング)除外
                    # レース内平均より早いSTほどプラス補正
                    diff = race_avg_st - st_val
                    base_rate *= (1.0 + diff * EXHST_WEIGHT)
            except (ValueError, TypeError):
                pass

        probs[boat - 1] = max(base_rate, 0.001)

    # 合計を1.0に正規化
    probs /= probs.sum()
    return probs


def trifecta_probs_from_win(win_probs: np.ndarray) -> dict[str, float]:
    """
    勝率からHarville公式で3連単全120通りの確率を計算する

    Args:
        win_probs: shape=(6,) の1着勝率

    Returns:
        {"1-2-3": 0.123, ...} の確率dict
    """
    from itertools import permutations
    result = {}
    for b1, b2, b3 in permutations(range(1, 7), 3):
        p1 = win_probs[b1 - 1]
        rem1 = np.array([win_probs[i] for i in range(6) if i != b1 - 1])
        p2 = win_probs[b2 - 1] / rem1.sum() if rem1.sum() > 0 else 0
        rem2 = np.array([win_probs[i] for i in range(6) if i not in (b1 - 1, b2 - 1)])
        p3 = win_probs[b3 - 1] / rem2.sum() if rem2.sum() > 0 else 0
        result[f"{b1}-{b2}-{b3}"] = p1 * p2 * p3
    return result
