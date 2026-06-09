"""
レース予想エンジン
学習済みモデルを使って3連単の買い目と期待回収率を予測する
"""

import json
from itertools import permutations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.builder import get_feature_columns, add_course_advantage
from src.agents.course_strategy import predict_win_probs as course_win_probs
from src.agents.racer_performance import predict_win_probs as racer_win_probs, _RACER_STYLE
from src.agents.motor_form import predict_win_probs as motor_win_probs

# エージェントの重み（合計1.0）
WEIGHT_ML     = 0.40  # AIモデルエージェント
WEIGHT_COURSE = 0.25  # コース戦略エージェント（競艇では枠番が最重要）
WEIGHT_RACER  = 0.20  # 選手成績エージェント
WEIGHT_MOTOR  = 0.15  # モーター状態エージェント

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"
PAYOUT_LOOKUP_PATH = MODEL_DIR / "payout_by_rank.json"

# 荒れ条件: 対象レースとして選出するための最低スコア
ARARE_MIN_SCORE = 7

# 荒れやすい会場の加点（江戸川のみ2点、他は1点）
ARARE_VENUES = {
    "03": 2,  # 江戸川（河川・最難関）
    "02": 1,  # 戸田
    "04": 1,  # 平和島
    "14": 1,  # 鳴門
    "10": 1,  # 三国
    "19": 1,  # 下関
}


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f  # NaN → None
    except (TypeError, ValueError):
        return None


def _pick_condition_based_ana(
    race_row: pd.Series,
    arare_reasons: list[str],
    absent_boats: list | None,
    by_prob: "pd.DataFrame | None" = None,
) -> tuple[str | None, str | None]:
    """
    展示タイム・ST・今節成績・前付け・グレード・積極性から脅威アウト艇を特定し、
    新本命穴・新超穴の組み合わせを返す

    Returns:
        (新本命穴コンボ, 新超穴コンボ) or (None, None)
    """
    available = [b for b in range(1, 7) if not (absent_boats and b in absent_boats)]
    outer_boats = [b for b in [3, 4, 5, 6] if b in available]
    if len(outer_boats) < 1:
        return None, None

    # MLモデル確率フィルタ: 外枠艇のうちML1着確率が上位2艇のみを候補とする
    if by_prob is not None and not by_prob.empty:
        outer_win_probs = {}
        for bn in outer_boats:
            mask = by_prob["boat1"] == bn
            outer_win_probs[bn] = float(by_prob[mask]["prob"].sum()) if mask.any() else 0.0
        outer_boats = sorted(outer_boats, key=lambda b: outer_win_probs.get(b, 0), reverse=True)[:2]
        if not outer_boats:
            return None, None

    # 展示タイム・STを収集
    et_vals: dict[int, float] = {}
    for bn in available:
        et = _safe_float(race_row.get(f"boat{bn}_exhibition_time"))
        if et and et > 0:
            et_vals[bn] = et

    # 展示タイムを速い順にランク付け（全艇中の順位）
    et_sorted = sorted(et_vals.values())

    # 統合脅威スコアリング（外枠絞込み済み艇のみ）
    # _calc_threat_score: 展示ST・今節ST・グレード・racer積極性 を合算
    outer_scores: dict[int, float] = {}
    for bn in outer_boats:
        s = _calc_threat_score(bn, race_row)

        # 展示タイムランク加点（全艇中: 1位+3, 2位+2, 3位+1）
        if bn in et_vals:
            rank = et_sorted.index(et_vals[bn])
            s += max(0.0, 3.0 - rank)

        # 前付け加点（実際のコースが艇番より内側 = 積極的侵入）
        ac_raw = race_row.get(f"boat{bn}_actual_course")
        try:
            ac = int(ac_raw) if ac_raw is not None else bn
        except (TypeError, ValueError):
            ac = bn
        if ac < bn:
            s += 2.0

        # 今節成績加点（平均着順: 1着平均=+1.5, 3着平均=0, 6着平均=-1.5）
        meet_avg_rank = _safe_float(race_row.get(f"boat{bn}_meet_avg_rank"))
        meet_races = int(race_row.get(f"boat{bn}_meet_races", 0) or 0)
        if meet_avg_rank is not None and meet_races >= 1:
            s += (3.0 - meet_avg_rank) * 0.5

        outer_scores[bn] = s

    if not outer_scores:
        return None, None

    ranked = sorted(outer_scores, key=lambda b: outer_scores[b], reverse=True)
    threat1 = ranked[0]
    threat2 = ranked[1] if len(ranked) >= 2 else None

    # 1号艇の弱さを判定（ST遅い・B級・モーター不良）
    boat1_weak = any("1号ST" in r or "1号B" in r or "1号M" in r for r in arare_reasons)

    def _best_third(used: set) -> int | None:
        cands = sorted(
            [(b, et_vals.get(b, 9.99)) for b in available if b not in used],
            key=lambda x: x[1],
        )
        return cands[0][0] if cands else None

    # ── 新本命穴: 脅威艇1着 × 1号艇2着 × 残り展示上位3着 ──
    if 1 in available and not boat1_weak:
        second_h = 1
    elif threat2:
        second_h = threat2
    else:
        others = [b for b in available if b != threat1]
        second_h = others[0] if others else None

    if second_h is None:
        return None, None
    third_h = _best_third({threat1, second_h})
    if third_h is None:
        return None, None
    shinhonmei = f"{threat1}-{second_h}-{third_h}"

    # ── 新超穴: 脅威艇1着 × 1号艇除外の候補2着 × 残り3着 ──
    # threat1・1号艇を除いた全候補を展示タイム順に列挙し、新本命穴と被らない組み合わせを探す
    cands_c = sorted(
        [(b, et_vals.get(b, 9.99)) for b in available if b != threat1 and b != 1],
        key=lambda x: x[1],
    )
    # threat2を優先（second_hと異なる場合は先頭へ）
    if threat2 and threat2 != second_h:
        cands_c = [(threat2, et_vals.get(threat2, 9.99))] + [
            (b, e) for b, e in cands_c if b != threat2
        ]

    shinchoana = None
    for second_c, _ in cands_c:
        third_c = _best_third({threat1, second_c})
        if third_c is None:
            continue
        candidate = f"{threat1}-{second_c}-{third_c}"
        if candidate != shinhonmei:
            shinchoana = candidate
            break

    return shinhonmei, shinchoana



# 各会場のイン逃げ率（全国統計ベース）
_VENUE_INNER_ESCAPE = {
    "01": 0.580, "02": 0.520, "03": 0.420, "04": 0.510, "05": 0.560,
    "06": 0.590, "07": 0.590, "08": 0.570, "09": 0.570, "10": 0.530,
    "11": 0.590, "12": 0.600, "13": 0.590, "14": 0.540, "15": 0.580,
    "16": 0.580, "17": 0.600, "18": 0.590, "19": 0.540, "20": 0.590,
    "21": 0.580, "22": 0.560, "23": 0.590, "24": 0.570,
}
# レース番号補正（早いレースほど逃げやすい）
_RACE_NO_NIGERATE_FACTOR = {
    1: 1.06, 2: 1.04, 3: 1.02, 4: 1.01, 5: 1.00, 6: 0.99,
    7: 0.98, 8: 0.97, 9: 0.96, 10: 0.95, 11: 0.94, 12: 0.93,
}
# 全国平均 コース1番逃げ率（選手個人データのベースライン）
_INNER_WIN_BASELINE = 0.55
# 全国平均 コース2番2着内率（差し圧力のベースライン）
_COURSE2_PRESSURE_BASELINE = 0.45


def _calc_nigerate(race_row: pd.Series) -> str:
    """
    1号艇のイン逃げ推定率を計算して "逃げ推定XX%" 文字列で返す。

    会場ベース × レース番号補正 × 選手逃げ率補正 × 展示ST補正 × 2号艇差し圧力補正
    """
    venue_code = str(race_row.get("venue_code", "")).zfill(2)
    race_no = int(race_row.get("race_no", 6) or 6)

    base = _VENUE_INNER_ESCAPE.get(venue_code, 0.570)
    rn_factor = _RACE_NO_NIGERATE_FACTOR.get(race_no, 1.00)

    # 一般戦は1号艇が弱くなりやすい → 微減
    mg = _safe_float(race_row.get("meet_grade_num"))
    series_factor = 0.97 if mg is not None and mg <= 1 else 1.00

    # 選手1号艇の逃げ率実績補正
    racer1_no = int(race_row.get("boat1_racer_no", 0) or 0)
    racer1_inner = _INNER_WIN_BASELINE
    if racer1_no in _RACER_STYLE:
        sd = _RACER_STYLE[racer1_no]
        if isinstance(sd, dict) and "inner_win_rate" in sd:
            racer1_inner = sd["inner_win_rate"]
    inner_factor = np.clip(racer1_inner / _INNER_WIN_BASELINE, 0.65, 1.35)

    # 展示ST補正（速いほど逃げやすい）
    st1 = _safe_float(race_row.get("boat1_exhibition_st"))
    if st1 and st1 > 0:
        st_factor = np.clip(1.0 + (0.155 - st1) * 2.5, 0.70, 1.30)
    else:
        st_factor = 1.0

    # 2号艇の差し圧力補正（高いほど1号艇逃げ率が下がる）
    racer2_no = int(race_row.get("boat2_racer_no", 0) or 0)
    c2_pressure = _COURSE2_PRESSURE_BASELINE
    if racer2_no in _RACER_STYLE:
        sd = _RACER_STYLE[racer2_no]
        if isinstance(sd, dict) and "course2_pressure" in sd:
            c2_pressure = sd["course2_pressure"]
    pressure_factor = np.clip(1.0 - 0.4 * (c2_pressure - _COURSE2_PRESSURE_BASELINE), 0.80, 1.10)

    nigerate = base * rn_factor * series_factor * inner_factor * st_factor * pressure_factor
    nigerate = max(0.05, min(0.95, nigerate))
    return f"逃げ推定{int(round(nigerate * 100))}%"


def _calc_threat_score(bn: int, race_row: pd.Series) -> float:
    """
    荒れ時の脅威艇スコアを計算する。
    展示ST速度・グレード・アウトコース積極性を合算して返す。
    フォーメーション3着枠の「確率外脅威艇」選出に使用。
    """
    score = 0.0

    # 展示ST（低い＝速い → 脅威度高）: 0.10→+0.50, 0.15→0, 0.20→-0.50
    exh_st = race_row.get(f"boat{bn}_exhibition_st")
    if exh_st is not None:
        try:
            st_val = float(exh_st)
            if st_val > 0:
                score += (0.150 - st_val) * 10.0
        except (TypeError, ValueError):
            pass

    # 今節ST実績（展示STを補強）
    meet_avg_st = race_row.get(f"boat{bn}_meet_avg_st")
    if meet_avg_st is not None:
        try:
            score += (0.150 - float(meet_avg_st)) * 5.0
        except (TypeError, ValueError):
            pass

    # グレード（A1=4→+0.6, A2=3→+0.3, B1/B2=0）
    gn_raw = race_row.get(f"boat{bn}_grade_num", 2)
    try:
        gn = int(float(gn_raw)) if gn_raw is not None else 2
        if gn >= 3:
            score += (gn - 2) * 0.3
    except (TypeError, ValueError):
        pass

    # アウトコース積極性（3〜6号艇のみ: 積極的な選手ほど荒れの主役になりやすい）
    if bn >= 3:
        racer_no = int(race_row.get(f"boat{bn}_racer_no", 0) or 0)
        if racer_no in _RACER_STYLE:
            style = _RACER_STYLE[racer_no]
            aggression = style.get("aggression_score", 1.0) if isinstance(style, dict) else float(style)
            score += max(0.0, (aggression - 1.0) * 0.5)

    return score


def _calc_boat1_risk(race_row: pd.Series) -> str:
    """1号艇のリスク指標（モーター・展示タイム・ST・今節ST・実力）を文字列で返す"""
    flags = []
    m1 = _safe_float(race_row.get("boat1_motor_2rate"))
    if m1 is not None and m1 < 0.33:
        flags.append("M弱")
    et_vals = [v for bn in range(1, 7)
               if (v := _safe_float(race_row.get(f"boat{bn}_exhibition_time"))) and v > 0]
    et1 = _safe_float(race_row.get("boat1_exhibition_time"))
    if et1 and et1 > 0 and et_vals and et1 > sum(et_vals) / len(et_vals):
        flags.append("展遅")
    st1 = _safe_float(race_row.get("boat1_exhibition_st"))
    if st1 is not None and st1 >= 0.18:
        flags.append("ST遅")
    meet_st1 = _safe_float(race_row.get("boat1_meet_avg_st"))
    if meet_st1 is not None and meet_st1 >= 0.17:
        flags.append("節ST遅")
    wr1 = _safe_float(race_row.get("boat1_win_rate"))
    if wr1 is not None and wr1 < 0.05:
        flags.append("実力低")
    return " / ".join(flags) if flags else "-"


def _calc_arare_score(race_row: pd.Series, weather: dict = None) -> tuple[int, list[str]]:
    """
    荒れ条件スコアを計算する。
    2点条件: 1号艇の展示ST遅い / モーター不良 / B級 / 外艇展示タイム上位 / 前付け
    1点条件: 1号艇の全国2連率低い / 今節不調 / 展示最遅 / 外艇A1選手 / 外艇ST速い(複数可) /
             風速強 / 追い風 / 波高 / 荒れ会場 / 一般戦
    Returns: (score, [条件説明リスト])
    """
    score = 0
    reasons = []

    # ── 1号艇の弱点（各2点） ──
    st1 = _safe_float(race_row.get("boat1_exhibition_st"))
    if st1 is not None and st1 >= 0.18:
        score += 2
        reasons.append(f"1号ST{st1:.2f}")

    m1 = _safe_float(race_row.get("boat1_motor_2rate"))
    if m1 is not None and m1 < 0.35:
        score += 2
        reasons.append(f"1号M{m1:.0%}")

    g1 = _safe_float(race_row.get("boat1_grade_num"))
    if g1 is not None and g1 <= 2:
        score += 2
        reasons.append("1号B級")

    # ── 外艇(4-6号)の最速タイムが全艇1位（2点） ──
    # 「TOP3内」だと6艇中95%の確率で成立してしまうため、より厳格に判定
    et_vals = {}
    for bn in range(1, 7):
        v = _safe_float(race_row.get(f"boat{bn}_exhibition_time"))
        if v and v > 0:
            et_vals[bn] = v
    if len(et_vals) >= 4:
        fastest_boat = min(et_vals, key=lambda b: et_vals[b])
        if fastest_boat in {4, 5, 6}:
            score += 2
            reasons.append(f"{fastest_boat}号艇展示最速({et_vals[fastest_boat]:.2f}s)")

        # 1号艇の展示タイムが全艇中最遅（外艇最速とは独立したシグナル）
        if 1 in et_vals:
            slowest_boat = max(et_vals, key=lambda b: et_vals[b])
            if slowest_boat == 1:
                score += 1
                reasons.append(f"1号展示最遅({et_vals[1]:.2f}s)")

    # ── 補助条件（各1点） ──
    n2_1 = _safe_float(race_row.get("boat1_national_2rate"))
    if n2_1 is not None and n2_1 < 0.40:
        score += 1
        reasons.append(f"1号2率{n2_1:.0%}")

    # 1号艇の今節成績が不振（2R以上出走かつ平均着順4.0以上）
    meet_avg_rank1 = _safe_float(race_row.get("boat1_meet_avg_rank"))
    meet_races1 = int(race_row.get("boat1_meet_races", 0) or 0)
    if meet_avg_rank1 is not None and meet_races1 >= 2 and meet_avg_rank1 >= 4.0:
        score += 1
        reasons.append(f"1号今節{meet_avg_rank1:.1f}着平均")

    for bn in [4, 5, 6]:
        g = _safe_float(race_row.get(f"boat{bn}_grade_num"))
        if g is not None and g >= 4:
            score += 1
            reasons.append(f"{bn}号A1")
            break

    # 外艇ST速い（複数艇カウント・上限2点）
    fast_st_count = 0
    for bn in [4, 5, 6]:
        st = _safe_float(race_row.get(f"boat{bn}_exhibition_st"))
        if st is not None and st <= 0.12:
            score += 1
            reasons.append(f"{bn}号ST速({st:.2f})")
            fast_st_count += 1
            if fast_st_count >= 2:
                break

    wind = _safe_float((weather or {}).get("wind_speed"))
    wind_dir = (weather or {}).get("wind_direction", "")
    if wind and wind >= 5:
        score += 1
        dir_suffix = {"tail": "追", "head": "向", "side": "横"}.get(wind_dir, "")
        reasons.append(f"風速{int(wind)}m{dir_suffix}")
        # 追い風は外艇のまくりが決まりやすく荒れが増大する
        if wind_dir == "tail":
            score += 1

    wave = _safe_float((weather or {}).get("wave_height"))
    if wave and wave >= 15:
        score += 1
        reasons.append(f"波高{int(wave)}cm")

    vc = str(race_row.get("venue_code", "")).zfill(2)
    venue_pts = ARARE_VENUES.get(vc, 0)
    if venue_pts:
        score += venue_pts
        reasons.append("江戸川(最難関)" if venue_pts == 2 else "荒れ会場")

    mg = _safe_float(race_row.get("meet_grade_num"))
    if mg is not None and mg <= 1:
        score += 1
        reasons.append("一般戦")

    # 前付け（高番号艇がインコース進入）検出 ─ 複数艇が同時に前付けする場合も全艇カウント
    for bn in [4, 5, 6]:
        ac_raw = race_row.get(f"boat{bn}_actual_course")
        try:
            ac = int(ac_raw) if ac_raw is not None else bn
        except (TypeError, ValueError):
            ac = bn
        if ac <= 2 and ac != bn:   # 4-6号艇が1-2コースに前付け
            score += 2
            reasons.append(f"{bn}号艇前付け(→{ac}コース)")
        elif ac <= 3 and ac != bn: # 4-6号艇が3コースに進入
            score += 1
            reasons.append(f"{bn}号艇進入変更(→{ac}コース)")

    return score, reasons


def build_payout_lookup(df_payout: pd.DataFrame, model: lgb.Booster, df_features: pd.DataFrame) -> dict:
    """
    過去データから人気順位ごとの平均払戻金を計算してJSONに保存する

    人気順位 = モデルが予測した確率で120通りをソートした時の順位
    """
    df_features = add_course_advantage(df_features)
    feature_cols = get_feature_columns() + [f"boat{bn}_course_advantage" for bn in range(1, 7)]
    # 旧モデルとの互換性: 新規追加順に除外して列数を合わせる
    _new_cols = ["meet_grade_num", "is_final_day_num", "series_day"]
    for col in _new_cols:
        if model.num_feature() >= len(feature_cols):
            break
        feature_cols = [c for c in feature_cols if c != col]
    trifecta_payout = df_payout[df_payout["bet_type"] == "３連単"].copy()

    rank_payouts = {i: [] for i in range(1, 121)}

    for _, race_row in df_features.iterrows():
        # この日・場・レースの払戻を取得
        match = trifecta_payout[
            (trifecta_payout["date"] == race_row.get("date", "")) &
            (trifecta_payout["venue_code"] == race_row.get("venue_code", "")) &
            (trifecta_payout["race_no"] == race_row.get("race_no", 0))
        ]
        if match.empty:
            continue

        actual_payout = match.iloc[0]["payout"]
        actual_combination = match.iloc[0]["combination"]

        # モデルで各組み合わせの確率を計算
        X = race_row[feature_cols].fillna(0).values.reshape(1, -1)
        win_probs = model.predict(X)[0]

        combos = list(permutations(range(1, 7), 3))
        probs = []
        for b1, b2, b3 in combos:
            p1 = win_probs[b1 - 1]
            rem1 = np.array([win_probs[i] for i in range(6) if i != b1 - 1])
            p2 = win_probs[b2 - 1] / rem1.sum() if rem1.sum() > 0 else 0
            rem2 = np.array([win_probs[i] for i in range(6) if i not in (b1 - 1, b2 - 1)])
            p3 = win_probs[b3 - 1] / rem2.sum() if rem2.sum() > 0 else 0
            probs.append((f"{b1}-{b2}-{b3}", p1 * p2 * p3))

        # 確率の高い順にソート（=人気順）
        probs.sort(key=lambda x: x[1], reverse=True)
        combo_to_rank = {c: i + 1 for i, (c, _) in enumerate(probs)}

        if actual_combination in combo_to_rank:
            rank = combo_to_rank[actual_combination]
            rank_payouts[rank].append(actual_payout)

    # 各ランクの平均払戻を計算
    lookup = {}
    for rank, payouts in rank_payouts.items():
        if payouts:
            lookup[str(rank)] = int(np.mean(payouts))
        else:
            # データなしの場合は前後の値から補間
            lookup[str(rank)] = int(300 * (rank ** 0.8))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(PAYOUT_LOOKUP_PATH, "w") as f:
        json.dump(lookup, f)

    print(f"[OK] 払戻ルックアップ保存: {PAYOUT_LOOKUP_PATH}")
    print(f"     人気1位平均: {lookup.get('1', 'N/A')}円 / 人気10位平均: {lookup.get('10', 'N/A')}円 / 人気50位平均: {lookup.get('50', 'N/A')}円")
    return lookup


def load_payout_lookup() -> dict:
    """払戻ルックアップを読み込む"""
    if PAYOUT_LOOKUP_PATH.exists():
        with open(PAYOUT_LOOKUP_PATH) as f:
            return json.load(f)
    # フォールバック: 経験則による推定値
    return {str(i): int(300 * (i ** 0.8)) for i in range(1, 121)}


def _apply_weather_adjustment(win_probs: np.ndarray, weather: dict) -> np.ndarray:
    """
    天候・風速・風向・波高に基づいて艇別勝率を調整する。

    向かい風の内艇優位は7mでピーク、8m以降は減衰し9m以上でゼロ。
    代わりに8m以上では混戦フラット化補正を適用（強風時の不確実性増大）。
    追い風・横風は線形スケール（上限7単位）。
    """
    wind     = float(weather.get("wind_speed", 0) or 0)
    wave     = float(weather.get("wave_height", 0) or 0)
    cond     = weather.get("weather") or ""
    wind_dir = weather.get("wind_direction") or ""

    adj = np.zeros(6)

    if wind >= 5:
        if wind_dir == "head":
            # 向かい風: 内艇有利は7mでピーク（山型）、9m以上はゼロ
            # intensity = max(0, min(wind-4, 10-wind))
            #   5m→1, 6m→2, 7m→3(peak), 8m→2, 9m→1, 10m→0
            inner_intensity = max(0.0, min(wind - 4, 10.0 - wind))
            adj += inner_intensity * np.array([+0.009, +0.006, +0.002, -0.001, -0.006, -0.010])

        elif wind_dir == "tail":
            # 追い風: 外艇まくり有利・線形スケール
            intensity = min(wind - 4, 7)
            adj += intensity * np.array([-0.007, -0.004, +0.001, +0.005, +0.006, +0.004])

        elif wind_dir == "side":
            # 横風: 混戦化・外艇微有利
            intensity = min(wind - 4, 7)
            adj += intensity * np.array([-0.002, +0.001, +0.002, +0.003, +0.002, -0.001])

        else:
            # 不明: 保守的な内有利（弱め）
            intensity = min(wind - 4, 7)
            adj += intensity * np.array([+0.004, +0.003, +0.001, 0.0, -0.003, -0.005])

    elif wind >= 3:
        adj += np.array([+0.008, +0.005, +0.002, 0.0, -0.005, -0.010])

    # 波高補正（独立）
    if wave >= 20:
        adj += np.array([+0.015, +0.010, +0.005, 0.0, -0.010, -0.020])
    elif wave >= 10:
        adj += np.array([+0.008, +0.005, +0.002, 0.0, -0.005, -0.010])

    # 雨天補正
    if cond == "rain":
        adj += np.array([+0.005, +0.003, +0.001, 0.0, -0.003, -0.006])

    probs = np.clip(win_probs + adj, 0.001, None)
    probs = probs / probs.sum()

    # 強向かい風（8m以上）: 混戦フラット化補正
    # 1マークの荒れで着順が読みにくくなる → 確率を均等分布に近づける
    # 8m=7%, 9m=14%, 10m=21%（上限22%）
    if wind >= 8 and wind_dir == "head":
        chaos = min((wind - 7) * 0.07, 0.22)
        uniform = np.ones(6) / 6.0
        probs = (1.0 - chaos) * probs + chaos * uniform

    return probs / probs.sum()


def _zero_absent(probs: np.ndarray, absent_boats: list) -> np.ndarray:
    """欠場艇の勝率を0にして残り艇で正規化する"""
    if not absent_boats:
        return probs
    p = probs.copy()
    for bn in absent_boats:
        if 1 <= bn <= 6:
            p[bn - 1] = 0.0
    total = p.sum()
    return p / total if total > 0 else p


def predict_race(
    model: lgb.Booster,
    race_features: pd.Series,
    payout_lookup: dict = None,
    live_odds: dict = None,
    weather: dict = None,
    absent_boats: list = None,
) -> pd.DataFrame:
    """
    1レース分の3連単予想を生成する

    Args:
        model: 学習済みLightGBMモデル
        race_features: 1レース分の特徴量
        payout_lookup: 人気順位→平均払戻のルックアップ（live_odds未指定時に使用）
        live_odds: リアルタイムオッズ {"1-2-3": 3.5, ...}（指定時はこちらを優先）

    Returns:
        DataFrame: 3連単の全組み合わせと期待回収率
    """
    if payout_lookup is None:
        payout_lookup = load_payout_lookup()

    feature_cols = get_feature_columns() + [f"boat{bn}_course_advantage" for bn in range(1, 7)]
    # 旧モデルとの互換性: 新規追加順に除外して列数を合わせる
    _new_cols = ["meet_grade_num", "is_final_day_num", "series_day"]
    for col in _new_cols:
        if model.num_feature() >= len(feature_cols):
            break
        feature_cols = [c for c in feature_cols if c != col]
    X = race_features.reindex(feature_cols, fill_value=0).fillna(0).values.reshape(1, -1)
    ml_probs = model.predict(X)[0]

    # 温度スケーリングで確率の過信を補正（T=2.5: 高いほど保守的）
    TEMPERATURE = 2.5
    logits = np.log(np.clip(ml_probs, 1e-10, 1.0))
    scaled = np.exp(logits / TEMPERATURE)
    ml_probs = scaled / scaled.sum()

    # コース戦略エージェントの勝率
    cs_probs = course_win_probs(race_features)

    # 選手成績エージェントの勝率
    rp_probs = racer_win_probs(race_features)

    # モーター状態エージェントの勝率
    mt_probs = motor_win_probs(race_features)

    # 欠場艇を全エージェントから除外して正規化
    if absent_boats:
        ml_probs = _zero_absent(ml_probs, absent_boats)
        cs_probs = _zero_absent(cs_probs, absent_boats)
        rp_probs = _zero_absent(rp_probs, absent_boats)
        mt_probs = _zero_absent(mt_probs, absent_boats)

    # 4エージェントの勝率を重み付け合成
    win_probs = (WEIGHT_ML * ml_probs + WEIGHT_COURSE * cs_probs
                 + WEIGHT_RACER * rp_probs + WEIGHT_MOTOR * mt_probs)

    # 天候・風速・波高による調整
    if weather:
        win_probs = _apply_weather_adjustment(win_probs, weather)

    # 全120通りの確率を計算
    combinations = list(permutations(range(1, 7), 3))
    combo_probs = []

    for b1, b2, b3 in combinations:
        p1 = win_probs[b1 - 1]
        rem1 = np.array([win_probs[i] for i in range(6) if i != b1 - 1])
        p2 = win_probs[b2 - 1] / rem1.sum() if rem1.sum() > 0 else 0
        rem2 = np.array([win_probs[i] for i in range(6) if i not in (b1 - 1, b2 - 1)])
        p3 = win_probs[b3 - 1] / rem2.sum() if rem2.sum() > 0 else 0
        prob = p1 * p2 * p3
        combo_probs.append((f"{b1}-{b2}-{b3}", b1, b2, b3, prob))

    # 確率の高い順に並べて人気順位を付与
    combo_probs.sort(key=lambda x: x[4], reverse=True)

    using_live = bool(live_odds)

    results = []
    for rank, (combination, b1, b2, b3, prob) in enumerate(combo_probs, start=1):
        if using_live and combination in live_odds:
            actual_odds = live_odds[combination]
            odds_value = round(actual_odds, 1)
            expected_roi = prob * actual_odds
            odds_source = "live"
        else:
            hist_payout = payout_lookup.get(str(rank), int(300 * (rank ** 0.8)))
            odds_value = round(hist_payout / 100, 1)
            expected_roi = prob * odds_value
            odds_source = "history"

        results.append({
            "combination": combination,
            "boat1": b1, "boat2": b2, "boat3": b3,
            "prob": round(prob, 6),
            "popularity_rank": rank,
            "odds_value": odds_value,
            "expected_roi": round(expected_roi, 4),
            "odds_source": odds_source,
        })

    df = pd.DataFrame(results).sort_values("expected_roi", ascending=False).reset_index(drop=True)

    # 欠場艇を含む組み合わせをプールから完全除去（確率0でも残るため明示的に削除）
    if absent_boats:
        absent_set = set(absent_boats)
        df = df[
            ~df["boat1"].isin(absent_set) &
            ~df["boat2"].isin(absent_set) &
            ~df["boat3"].isin(absent_set)
        ].reset_index(drop=True)

    df["rank"] = df.index + 1
    return df


def _is_race_worth_betting(by_prob: pd.DataFrame, race_row: pd.Series = None) -> tuple[bool, str]:
    """
    このレースで勝負すべきかを判定する

    判定基準:
    1. 展示STデータが取れているか（当日情報の有無）
    2. エージェント合議度: 上位5組に合議2以上が1組以上あるか
    3. シグナル強度: 1位と2位の確率差が十分あるか

    Returns:
        (should_bet: bool, skip_reason: str)
    """
    if by_prob.empty or len(by_prob) < 6:
        return False, "データ不足"

    # ── 展示STデータチェック ──
    if race_row is not None:
        st_count = 0
        for bn in range(1, 7):
            v = race_row.get(f"boat{bn}_exhibition_st")
            try:
                if v is not None and float(v) > 0:
                    st_count += 1
            except (TypeError, ValueError):
                pass
        if st_count < 3:
            return False, f"展示STデータ不足（{st_count}/6艇）"

    # ── 1着合議チェック: 予想1着艇が3/4エージェント以上で一致しているか ──
    b1_votes = int(by_prob.iloc[0].get("b1_votes", 0))
    if b1_votes < 3:
        return False, f"1着合議不足（{b1_votes}/4エージェント）"

    # ── 展示タイムチェック: 予想1着艇が全艇中TOP3以内か ──
    if race_row is not None and not by_prob.empty:
        top_b1 = int(str(by_prob.iloc[0]["combination"]).split("-")[0])
        et_vals = {}
        for bn in range(1, 7):
            v = race_row.get(f"boat{bn}_exhibition_time")
            try:
                t = float(v)
                if t > 0:
                    et_vals[bn] = t
            except (TypeError, ValueError):
                pass
        if len(et_vals) >= 4:
            sorted_times = sorted(et_vals.values())
            top4_threshold = sorted_times[3]
            b1_time = et_vals.get(top_b1)
            if b1_time is not None and b1_time > top4_threshold:
                return False, f"展示タイム不利（{b1_time}s / TOP4={top4_threshold}s）"

    # ── シグナル強度 ──
    top_prob = float(by_prob.iloc[0]["prob"])
    second_prob = float(by_prob.iloc[1]["prob"])
    signal = top_prob - second_prob

    if signal < 0.004:
        return False, "混戦（シグナル弱）"

    return True, f"合議{int(by_prob.head(5)['agreement'].max())}/4 シグナル{signal:.4f}"


def get_recommendations(
    model: lgb.Booster,
    df_today: pd.DataFrame,
    top_n: int = 5,
    min_roi: float = 1.15,
    payout_lookup: dict = None,
    all_live_odds: dict = None,
    all_weather: dict = None,
    all_absent: dict = None,
) -> pd.DataFrame:
    """
    本日の全レースから推奨買い目を選出する

    Args:
        all_live_odds: {(venue_code, race_no): {"1-2-3": 3.5, ...}} のdict（省略可）
    """
    if payout_lookup is None:
        payout_lookup = load_payout_lookup()

    df_today = add_course_advantage(df_today)
    all_recommendations = []

    for _, race_row in df_today.iterrows():
        # このレースのリアルタイムオッズ・天候を取得（あれば）
        venue_code = str(race_row.get("venue_code", "")).zfill(2)
        race_no = int(race_row.get("race_no", 0))
        live_odds = all_live_odds.get((venue_code, race_no)) if all_live_odds else None
        weather = all_weather.get((venue_code, race_no)) if all_weather else None
        absent_boats = all_absent.get((venue_code, race_no)) if all_absent else None

        # 1号艇逃げ推定率（J列表示用）
        nigerate_str = _calc_nigerate(race_row)

        # 荒れ条件チェック: スコア不足でも予想は生成するが見送りになる
        arare_score, arare_reasons = _calc_arare_score(race_row, weather)
        # 神熱条件: 荒れPT7以上 かつ 追い風5m以上
        _wind_spd = _safe_float((weather or {}).get("wind_speed"))
        _wind_dir = (weather or {}).get("wind_direction", "")
        _tail_ok  = (_wind_spd is not None and _wind_spd >= 5 and _wind_dir == "tail")
        arare_ok  = arare_score >= ARARE_MIN_SCORE and _tail_ok
        venue_name_log = race_row.get("venue_name", "")
        if arare_ok:
            print(f"  [荒れ対象] {venue_name_log} {race_no}R スコア{arare_score} 条件:[{', '.join(arare_reasons)}]")
        else:
            print(f"  [荒れ対象外] {venue_name_log} {race_no}R スコア{arare_score}/{ARARE_MIN_SCORE}"
                  + (f" 条件:[{', '.join(arare_reasons)}]" if arare_reasons else ""))

        predictions = predict_race(model, race_row, payout_lookup, live_odds, weather, absent_boats)
        by_prob = predictions.sort_values("prob", ascending=False).reset_index(drop=True)

        def pick_by_edge(pool, tier_name, n=2, min_odds=0, max_odds=None,
                         hot_threshold=2.0, fire_threshold=3.0, exclude_boats=None,
                         min_prob=None, exclude_first_boats=None, sort_by="edge"):
            """
            指定オッズ範囲の組み合わせからエッジ上位n点を返す。
            exclude_boats: 除外する号艇番号のリスト（いずれかの着順に含まれる組み合わせを除外）
            exclude_first_boats: 1着から除外する号艇番号のリスト
            min_prob: 最低確率フィルター（これ未満の組み合わせを除外）
            sort_by: "edge"（エッジ順）または"prob"（確率順）
            """
            filtered = pool.copy()
            filtered["odds_value"] = pd.to_numeric(filtered["odds_value"], errors="coerce")
            filtered = filtered.dropna(subset=["odds_value"])
            if min_odds > 0:
                filtered = filtered[filtered["odds_value"] >= min_odds]
            if max_odds is not None:
                filtered = filtered[filtered["odds_value"] <= max_odds]
            if exclude_boats:
                for col in ("boat1", "boat2", "boat3"):
                    if col in filtered.columns:
                        filtered = filtered[~filtered[col].isin(exclude_boats)]
            if exclude_first_boats and "boat1" in filtered.columns:
                filtered = filtered[~filtered["boat1"].isin(exclude_first_boats)]
            if min_prob is not None:
                filtered = filtered[filtered["prob"] >= min_prob]
            if filtered.empty:
                return pd.DataFrame(), 0, False, None
            filtered["edge_score"] = filtered["prob"] * filtered["odds_value"]
            if sort_by == "prob":
                filtered = filtered.sort_values("prob", ascending=False).reset_index(drop=True)
            elif sort_by == "blend":
                # 確率とedgeの幾何平均（両方バランスよく高い組み合わせを優先）
                filtered["blend_score"] = (filtered["prob"] * filtered["edge_score"]) ** 0.5
                filtered = filtered.sort_values("blend_score", ascending=False).reset_index(drop=True)
            else:
                filtered = filtered.sort_values("edge_score", ascending=False).reset_index(drop=True)
            top_edge = float(filtered.iloc[0]["edge_score"])
            if top_edge >= fire_threshold:
                star_level = 3   # ★★★★ 灼熱
            elif top_edge >= hot_threshold:
                star_level = 2   # ★★★☆ 激熱
            elif top_edge >= 1.3:
                star_level = 1   # ★★☆☆ 見送り
            else:
                star_level = 0   # ★☆☆☆ 見送り
            is_confident = star_level >= 2
            topN = filtered.head(n).copy()
            topN["tier"] = tier_name
            second_b2 = int(filtered.iloc[n]["boat2"]) if len(filtered) > n else None
            return topN.reset_index(drop=True), star_level, is_confident, second_b2

        # 地熊目: 倍率制限なし・全120通りからprob上位2点
        lucky_pool, lucky_star, _, _ = pick_by_edge(
            by_prob, "地熊目", n=20, min_odds=0,
            hot_threshold=99, fire_threshold=4.0, sort_by="prob")
        lucky_rows = []
        used_combos: set = set()
        for _, row in lucky_pool.iterrows():
            if row["combination"] not in used_combos:
                used_combos.add(row["combination"])
                lucky_rows.append(row)
            if len(lucky_rows) >= 2:
                break
        lucky_recs = pd.DataFrame(lucky_rows).reset_index(drop=True) if lucky_rows else pd.DataFrame()

        if lucky_recs.empty:
            continue  # 欠場艇が多い等でプールにデータなし

        def _make_row(rec, star_lv, second):
            if arare_ok:
                confidence, bet_label = "★★★★", "神熱"
            elif arare_score >= ARARE_MIN_SCORE:
                confidence, bet_label = "★★★☆", "灼熱"
            elif arare_score == 1:
                confidence, bet_label = "★☆☆☆", "熊熱"
            else:
                confidence, bet_label = "★☆☆☆", "見送り"
            src = rec.get("odds_source", "history")
            edge_val = round(float(rec["prob"]) * float(rec["odds_value"]), 2)
            combo = "見送り" if arare_score >= 4 else rec["combination"]
            return {
                "date": race_row.get("date", ""),
                "venue_name": race_row.get("venue_name", ""),
                "race_no": race_row.get("race_no", ""),
                "combination": combo,
                "prob": f"{rec['prob']*100:.2f}%",
                "odds": f"{rec['odds_value']}倍" if src == "live" else f"{rec['odds_value']}倍(履歴)",
                "expected_roi": f"{rec['expected_roi']*100:.0f}%",
                "confidence": confidence,
                "odds_source": nigerate_str,
                "tier": rec.get("tier", ""),
                "bet_label": bet_label,
                "edge": str(edge_val),
                "arare_score": arare_score,
                "arare_reasons": " / ".join(arare_reasons),
                "boat1_risk": _calc_boat1_risk(race_row),
            }

        for _, rec in lucky_recs.iterrows():
            all_recommendations.append(_make_row(rec, lucky_star, None))

        # 熊フォメ（全PT対象・全艇・1着2艇×2着3艇×3着3-4艇）
        kuma_wp = {}
        for bn in range(1, 7):
            if absent_boats and bn in absent_boats:
                continue
            mask = by_prob["boat1"] == bn
            kuma_wp[bn] = float(by_prob[mask]["prob"].sum()) if mask.any() else 0.0
        kuma_ranked = sorted(kuma_wp, key=lambda b: kuma_wp[b], reverse=True)
        if len(kuma_ranked) >= 3:
            kuma_first = kuma_ranked[:2]
            kuma_pool2 = [b for b in kuma_ranked if b not in set(kuma_first)]
            kuma_threat_a = (max(kuma_pool2, key=lambda b: _calc_threat_score(b, race_row))
                             if kuma_pool2 else None)
            kuma_second = sorted(set(kuma_first) | {kuma_threat_a}) if kuma_threat_a else sorted(kuma_first)
            if arare_score >= 7:
                kuma_pool3 = [b for b in kuma_ranked if b not in set(kuma_second)]
                kuma_threat_b = (max(kuma_pool3, key=lambda b: _calc_threat_score(b, race_row))
                                 if kuma_pool3 else None)
                kuma_third = sorted(set(kuma_second) | {kuma_threat_b}) if kuma_threat_b else kuma_second
            else:
                kuma_third = kuma_second
            kuma_str = (
                f"{''.join(str(b) for b in sorted(kuma_first))}-"
                f"{''.join(str(b) for b in kuma_second)}-"
                f"{''.join(str(b) for b in kuma_third)}"
            )
            all_recommendations.append({
                "date":          race_row.get("date", ""),
                "venue_name":    race_row.get("venue_name", ""),
                "race_no":       race_row.get("race_no", ""),
                "combination":   "見送り" if arare_score >= 4 else kuma_str,
                "prob":          "-",
                "odds":          "-",
                "expected_roi":  "-",
                "confidence":    "熊フォメ",
                "odds_source":   nigerate_str,
                "tier":          "熊フォメ",
                "bet_label":     "神熱" if arare_ok else ("灼熱" if arare_score >= ARARE_MIN_SCORE else "熊熱" if arare_score == 1 else "見送り"),
                "edge":          "-",
                "arare_score":   arare_score,
                "arare_reasons": " / ".join(arare_reasons),
                "boat1_risk":    _calc_boat1_risk(race_row),
            })


        # 穴フォメ（全PT対象・全艇対象・カスケード方式）
        # 1着: ML1位 + ML3位（全艇・1号艇含む）
        # 穴フォメ: 1着=ML3+4位、2着=ML3+4+1位、3着=ML3+4+1+5位（34-134-1345形）
        if True:
            available_for_form = [b for b in range(1, 7) if not (absent_boats and b in absent_boats)]
            if len(available_for_form) >= 3:
                form_win_probs = {}
                for bn in available_for_form:
                    mask = by_prob["boat1"] == bn
                    form_win_probs[bn] = float(by_prob[mask]["prob"].sum()) if mask.any() else 0.0
                ranked_all = sorted(form_win_probs, key=lambda b: form_win_probs[b], reverse=True)

                # 1着: ML4位 + ML5位
                ml4 = ranked_all[3] if len(ranked_all) >= 4 else ranked_all[-1]
                ml5 = ranked_all[4] if len(ranked_all) >= 5 else ranked_all[-1]
                form_first_set = {ml4, ml5}
                form_first = sorted(form_first_set)

                # 2着: ML4位 + ML5位 + ML1位
                ml1 = ranked_all[0]
                form_second_set = form_first_set | {ml1}
                form_second = sorted(form_second_set)

                # 3着: ML4位 + ML5位 + ML1位 + ML3位
                ml3 = ranked_all[2] if len(ranked_all) >= 3 else ranked_all[-1]
                form_third_set = form_second_set | {ml3}
                form_third = sorted(form_third_set)

                formation_str = (
                    f"{''.join(str(b) for b in form_first)}-"
                    f"{''.join(str(b) for b in form_second)}-"
                    f"{''.join(str(b) for b in form_third)}"
                )
                bet_label = "神熱" if arare_ok else ("灼熱" if arare_score >= ARARE_MIN_SCORE else "熊熱" if arare_score == 1 else "見送り")
                all_recommendations.append({
                    "date":          race_row.get("date", ""),
                    "venue_name":    race_row.get("venue_name", ""),
                    "race_no":       race_row.get("race_no", ""),
                    "combination":   formation_str,
                    "prob":          "-",
                    "odds":          "-",
                    "expected_roi":  "-",
                    "confidence":    "穴フォメ",
                    "odds_source":   nigerate_str,
                    "tier":          "穴フォメ",
                    "bet_label":     bet_label,
                    "edge":          "-",
                    "arare_score":   arare_score,
                    "arare_reasons": " / ".join(arare_reasons),
                    "boat1_risk":    _calc_boat1_risk(race_row),
                })

                # ── 穴フォメ改 ──
                # 追い風あり: 穴スコア上位2艇を1着（穴艇軸）
                # 追い風なし + 荒れPT7以上: 1号艇 + 脅威スコア最高艇を1着（1号艇軸）
                et_vals_kai = {}
                for bn2 in available_for_form:
                    et2 = _safe_float(race_row.get(f"boat{bn2}_exhibition_time"))
                    if et2 and et2 > 0:
                        et_vals_kai[bn2] = et2
                et_sorted_kai = sorted(et_vals_kai.values())

                kai_scores: dict[int, float] = {}
                for bn2 in available_for_form:
                    try:
                        ml_rank_idx = ranked_all.index(bn2)
                    except ValueError:
                        continue
                    if ml_rank_idx < 2:
                        continue
                    ml_prob2 = form_win_probs.get(bn2, 0.0)
                    et_bonus = 0.0
                    if bn2 in et_vals_kai:
                        et_rank = et_sorted_kai.index(et_vals_kai[bn2])
                        et_bonus = max(0.0, 3.0 - et_rank)
                    threat2 = _calc_threat_score(bn2, race_row)
                    kai_scores[bn2] = (threat2 + et_bonus) / (ml_prob2 + 0.01)

                kai_bet_label = "神熱" if arare_ok else ("灼熱" if arare_score >= ARARE_MIN_SCORE else "見送り")

                if _tail_ok:
                    # 追い風あり: 穴スコア上位2艇を1着
                    if len(kai_scores) >= 2:
                        kai_first_boats = sorted(kai_scores, key=lambda b: kai_scores[b], reverse=True)[:2]
                    elif len(kai_scores) == 1:
                        extra = ranked_all[2] if len(ranked_all) > 2 else None
                        kai_first_boats = list(kai_scores.keys()) + ([extra] if extra else [])
                    else:
                        kai_first_boats = ranked_all[2:4]

                    if len(kai_first_boats) >= 2:
                        kai_first_set = set(kai_first_boats[:2])
                        kai_first = sorted(kai_first_set)
                        kai_second_set = kai_first_set | {ranked_all[0]}
                        kai_second = sorted(kai_second_set)
                        et_ranked_kai = sorted(
                            [(b, et_vals_kai.get(b, 9.99)) for b in available_for_form],
                            key=lambda x: x[1]
                        )
                        kai_third_set = {b for b, _ in et_ranked_kai[:4]}
                        kai_third = sorted(kai_third_set)
                        kai_str = (
                            f"{''.join(str(b) for b in kai_first)}-"
                            f"{''.join(str(b) for b in kai_second)}-"
                            f"{''.join(str(b) for b in kai_third)}"
                        )
                        all_recommendations.append({
                            "date":          race_row.get("date", ""),
                            "venue_name":    race_row.get("venue_name", ""),
                            "race_no":       race_row.get("race_no", ""),
                            "combination":   kai_str,
                            "prob":          "-",
                            "odds":          "-",
                            "expected_roi":  "-",
                            "confidence":    "穴フォメ改",
                            "odds_source":   nigerate_str,
                            "tier":          "穴フォメ改",
                            "bet_label":     kai_bet_label,
                            "edge":          "-",
                            "arare_score":   arare_score,
                            "arare_reasons": " / ".join(arare_reasons),
                            "boat1_risk":    _calc_boat1_risk(race_row),
                        })
                else:
                    # 追い風なし: 1号艇 + 脅威スコア最高艇を1着
                    boat1_kai = 1 if not (absent_boats and 1 in absent_boats) else ranked_all[0]
                    threat_top = (
                        max(kai_scores, key=lambda b: kai_scores[b])
                        if kai_scores
                        else (ranked_all[2] if len(ranked_all) >= 3 else ranked_all[-1])
                    )
                    kai_first_set = {boat1_kai, threat_top}
                    kai_first = sorted(kai_first_set)

                    ml2_kai = ranked_all[1] if len(ranked_all) >= 2 else ranked_all[-1]
                    kai_second_set = kai_first_set | {ml2_kai}
                    kai_second = sorted(kai_second_set)

                    ml3_kai = ranked_all[2] if len(ranked_all) >= 3 else None
                    kai_third_set = kai_second_set | ({ml3_kai} if ml3_kai is not None else set())
                    kai_third = sorted(kai_third_set)

                    kai_str = (
                        f"{''.join(str(b) for b in kai_first)}-"
                        f"{''.join(str(b) for b in kai_second)}-"
                        f"{''.join(str(b) for b in kai_third)}"
                    )
                    all_recommendations.append({
                        "date":          race_row.get("date", ""),
                        "venue_name":    race_row.get("venue_name", ""),
                        "race_no":       race_row.get("race_no", ""),
                        "combination":   kai_str,
                        "prob":          "-",
                        "odds":          "-",
                        "expected_roi":  "-",
                        "confidence":    "穴フォメ改",
                        "odds_source":   nigerate_str,
                        "tier":          "穴フォメ改",
                        "bet_label":     kai_bet_label,
                        "edge":          "-",
                        "arare_score":   arare_score,
                        "arare_reasons": " / ".join(arare_reasons),
                        "boat1_risk":    _calc_boat1_risk(race_row),
                    })

    return pd.DataFrame(all_recommendations)


def calculate_roi_history(df_result: pd.DataFrame, df_payout: pd.DataFrame, recommendations: pd.DataFrame) -> dict:
    """過去の推奨買い目の実際の回収率を計算する（バックテスト用）"""
    total_bet = 0
    total_return = 0

    for _, rec in recommendations.iterrows():
        if rec["combination"] == "見送り":
            continue

        total_bet += 100

        match = df_payout[
            (df_payout["date"] == rec["date"]) &
            (df_payout["venue_name"] == rec["venue_name"]) &
            (df_payout["race_no"] == rec["race_no"]) &
            (df_payout["bet_type"] == "３連単") &
            (df_payout["combination"] == rec["combination"])
        ]

        if not match.empty:
            total_return += match.iloc[0]["payout"]

    roi = total_return / total_bet if total_bet > 0 else 0
    return {
        "total_bet": total_bet,
        "total_return": total_return,
        "roi": round(roi, 4),
        "roi_pct": f"{roi*100:.1f}%",
    }
