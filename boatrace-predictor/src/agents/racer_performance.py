"""
選手成績エージェント
全国勝率・当地勝率・モーター2連率・選手グレード・展示STをもとに
各艇の勝利確率を計算する
"""

import numpy as np
import pandas as pd

# ─── グレード係数（A1が最強）───
GRADE_FACTOR = {4: 1.20, 3: 1.05, 2: 0.90, 1: 0.75, 0: 0.85}

# ─── 各指標の基準値（全国平均）───
BASELINE_NATIONAL_WIN  = 0.30  # 全国勝率の平均
BASELINE_LOCAL_WIN     = 0.28  # 当地勝率の平均
BASELINE_MOTOR_2RATE   = 0.35  # モーター2連率の平均
BASELINE_NATIONAL_2    = 0.46  # 全国2連率の平均
# ─── 枠番別展示ST基準値 ───
# 外枠ほど積極的なスタートを狙う傾向がある
BASELINE_ST_BY_LANE = {1: 0.155, 2: 0.152, 3: 0.150, 4: 0.148, 5: 0.145, 6: 0.142}

# ─── 展示タイムランク別補正係数 ───
EXHTIME_RANK_FACTOR = {1: 1.12, 2: 1.06, 3: 1.02, 4: 0.98, 5: 0.94, 6: 0.88}

# ─── 各指標の重み ───
W_NATIONAL_WIN  = 0.28
W_LOCAL_WIN     = 0.22
W_MOTOR_2RATE   = 0.18
W_NATIONAL_2    = 0.12
W_ST            = 0.10  # 展示ST重み
W_EXHTIME       = 0.10  # 展示タイムランク重み


def predict_win_probs(race_row: pd.Series) -> np.ndarray:
    """
    選手成績ベースで各艇（1〜6号艇）の勝率を計算する

    Args:
        race_row: 1レース分の特徴量Series
            使用列: boat{n}_national_win_rate, boat{n}_local_win_rate,
                    boat{n}_motor_2rate, boat{n}_national_2rate,
                    boat{n}_grade_num, boat{n}_exhibition_st

    Returns:
        shape=(6,) の勝率配列（インデックス0=1号艇, ..., 5=6号艇）
        合計は1.0
    """
    scores = np.zeros(6)

    # ── 展示タイムのレース内ランクを事前計算（直近調子の指標）──
    exh_times = []
    for b in range(1, 7):
        v = race_row.get(f"boat{b}_exhibition_time")
        try:
            fv = float(v)
            if fv > 0:
                exh_times.append((b, fv))
        except (TypeError, ValueError):
            pass
    exh_times_sorted = sorted(exh_times, key=lambda x: x[1])  # 速い順
    exh_rank = {b: rank + 1 for rank, (b, _) in enumerate(exh_times_sorted)}

    for boat in range(1, 7):
        nw  = float(race_row.get(f"boat{boat}_national_win_rate", BASELINE_NATIONAL_WIN) or BASELINE_NATIONAL_WIN)
        lw  = float(race_row.get(f"boat{boat}_local_win_rate",    BASELINE_LOCAL_WIN)    or BASELINE_LOCAL_WIN)
        m2  = float(race_row.get(f"boat{boat}_motor_2rate",       BASELINE_MOTOR_2RATE)  or BASELINE_MOTOR_2RATE)
        n2  = float(race_row.get(f"boat{boat}_national_2rate",    BASELINE_NATIONAL_2)   or BASELINE_NATIONAL_2)
        gn  = int(race_row.get(f"boat{boat}_grade_num", 2) or 2)

        # 各指標を基準値からの比率でスコア化
        score = (
            W_NATIONAL_WIN * (nw / BASELINE_NATIONAL_WIN) +
            W_LOCAL_WIN    * (lw / BASELINE_LOCAL_WIN)    +
            W_MOTOR_2RATE  * (m2 / BASELINE_MOTOR_2RATE)  +
            W_NATIONAL_2   * (n2 / BASELINE_NATIONAL_2)
        )

        # グレード補正
        score *= GRADE_FACTOR.get(gn, 1.0)

        # 展示ST補正（枠番別基準値と比較）
        exh_st = race_row.get(f"boat{boat}_exhibition_st")
        if exh_st is not None and not (isinstance(exh_st, float) and np.isnan(exh_st)):
            exh_st = float(exh_st)
            if exh_st > 0:  # フライング(負値)は除外
                lane_baseline = BASELINE_ST_BY_LANE.get(boat, 0.150)
                st_multiplier = 1.0 + (lane_baseline - exh_st) * W_ST * 50
                st_multiplier = np.clip(st_multiplier, 0.6, 1.4)
                score *= st_multiplier

        # 展示タイムランク補正（直近調子: レース内で速いほど今日の調子が良い）
        if boat in exh_rank and len(exh_times) >= 3:
            rank_factor = EXHTIME_RANK_FACTOR.get(exh_rank[boat], 1.0)
            score *= (1.0 + (rank_factor - 1.0) * W_EXHTIME * 10)

        scores[boat - 1] = max(score, 0.001)

    # 合計を1.0に正規化
    scores /= scores.sum()
    return scores
