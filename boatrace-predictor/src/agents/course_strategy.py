"""
コース戦略エージェント
競艇のコース有利・展示タイム・展示スタートタイミングをもとに
各艇の勝利確率を計算する
"""

import numpy as np
import pandas as pd

# ─── 全国平均コース別勝率 ───
# 1コースが圧倒的有利（特にインコース）
COURSE_WIN_RATES = {
    1: 0.550,
    2: 0.170,
    3: 0.110,
    4: 0.080,
    5: 0.055,
    6: 0.035,
}

# ─── 競艇場別コース特性補正 ───
# 外コースが強い会場（流れ・水面の影響）は1コース有利を弱める
# 値は1コース勝率の補正係数（1.0=全国平均、0.9=やや弱め）
VENUE_COURSE_FACTOR = {
    "江戸川":  0.72,  # 流れが強く外コース有利
    "戸田":    0.80,  # 幅が狭くアウト有利
    "平和島":  0.85,
    "多摩川":  0.90,
    "浜名湖":  0.95,
    "三国":    0.90,
    "びわこ":  0.88,
    "尼崎":    0.95,
    "鳴門":    0.90,
    "宮島":    0.88,
    "住之江":  1.05,  # インコースが特に強い
    "大村":    1.08,  # インコースが最も強い競艇場の一つ
}

# 展示タイム基準値（全国平均）
EXHTIME_BASELINE = 6.70  # 秒
EXHTIME_WEIGHT   = 0.15   # 展示タイムの影響度

# 展示スタート基準値
EXHST_BASELINE = 0.15    # 秒（平均的なスタートタイミング）
EXHST_WEIGHT   = 0.10    # 展示STの影響度


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
    venue_factor = VENUE_COURSE_FACTOR.get(venue_name, 1.0)

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
        base_rate = COURSE_WIN_RATES.get(course, 0.05)

        # 会場補正（1コースは venue_factor、外コースは逆補正）
        if course == 1:
            base_rate *= venue_factor
        else:
            # 1コースが弱い会場では外コースの確率が上がる
            base_rate *= (2.0 - venue_factor)

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
