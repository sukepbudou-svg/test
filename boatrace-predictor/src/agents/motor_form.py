"""
モーター状態エージェント
同一レース内での相対比較によりモーター・ボートの良し悪しを判定する

競艇ではモーターの出力差が勝敗に直結するため、
絶対値ではなくレース内での相対的な強さを評価する
"""

import numpy as np
import pandas as pd

# ─── フォールバック基準値 ───
BASELINE_MOTOR_2RATE = 0.35
BASELINE_BOAT_2RATE  = 0.35

# ─── 各指標の重み ───
W_MOTOR = 0.50  # モーター2連率（シーズン実績）
W_BOAT  = 0.20  # ボート2連率（補助）
W_EXH   = 0.30  # 展示タイム（当日のモーター状態・最重要）

# スコアの平滑化（極端な差を抑えるクリップ範囲）
SCORE_MIN = 0.40
SCORE_MAX = 2.50


def predict_win_probs(race_row: pd.Series) -> np.ndarray:
    """
    モーター・ボート状態ベースで各艇（1〜6号艇）の相対的な強さを計算する

    同一レース内の平均と比較することで、レースごとの装備格差を反映する

    Args:
        race_row: 1レース分の特徴量Series
            使用列: boat{n}_motor_2rate, boat{n}_boat_2rate

    Returns:
        shape=(6,) の勝率配列（インデックス0=1号艇, ..., 5=6号艇）
        合計は1.0
    """
    motor_rates = np.array([
        float(race_row.get(f"boat{b}_motor_2rate", BASELINE_MOTOR_2RATE) or BASELINE_MOTOR_2RATE)
        for b in range(1, 7)
    ])
    boat_rates = np.array([
        float(race_row.get(f"boat{b}_boat_2rate", BASELINE_BOAT_2RATE) or BASELINE_BOAT_2RATE)
        for b in range(1, 7)
    ])

    # レース内平均（0除算防止）
    motor_avg = motor_rates.mean() if motor_rates.mean() > 0 else BASELINE_MOTOR_2RATE
    boat_avg  = boat_rates.mean()  if boat_rates.mean()  > 0 else BASELINE_BOAT_2RATE

    # レース内相対スコア（平均=1.0）
    motor_rel = motor_rates / motor_avg
    boat_rel  = boat_rates  / boat_avg

    # 展示タイム（当日モーター状態の最直接指標: 小さいほど速い → 逆数で高スコア）
    exh_times = np.array([
        float(race_row.get(f"boat{b}_exhibition_time", 0) or 0)
        for b in range(1, 7)
    ])
    valid_exh = exh_times > 0
    if valid_exh.sum() >= 2:
        exh_avg = exh_times[valid_exh].mean()
        exh_rel = np.where(valid_exh, exh_avg / exh_times, 1.0)
        exh_rel = np.clip(exh_rel, 0.85, 1.15)
    else:
        exh_rel = np.ones(6)

    # 合成スコア（極端な差は抑制）
    scores = W_MOTOR * motor_rel + W_BOAT * boat_rel + W_EXH * exh_rel
    scores = np.clip(scores, SCORE_MIN, SCORE_MAX)
    scores = np.maximum(scores, 0.001)

    # コース基本確率を乗算（モーター差はコース有利を前提に作用する）
    # 1コースで良いモーターなら特に有利、6コースで良くても限界がある
    COURSE_BASE = np.array([0.550, 0.170, 0.110, 0.080, 0.055, 0.035])
    scores = scores * COURSE_BASE
    scores /= scores.sum()

    return scores
