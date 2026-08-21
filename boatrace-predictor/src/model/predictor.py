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

# 荒れ条件: 参戦ベースラインスコア（PT5〜9 AND 1号艇ライブ5〜9倍で参戦）
ARARE_MIN_SCORE = 5

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


# ANSI カラーコード（コンソール出力用）
_C_RED    = "\033[1;31m"   # プチュン
_C_YELLOW = "\033[1;33m"   # 黒船熱
_C_CYAN   = "\033[1;36m"   # 中穴
_C_GREEN  = "\033[1;32m"   # 参戦確定
_C_RESET  = "\033[0m"


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



# 各会場のイン逃げ率（1コース逃げ率・実績ベース統計値）
_VENUE_INNER_ESCAPE = {
    "01": 0.580,  # 桐生
    "02": 0.450,  # 戸田
    "03": 0.370,  # 江戸川（特殊コース・全国最低）
    "04": 0.460,  # 平和島
    "05": 0.560,  # 多摩川
    "06": 0.570,  # 浜名湖
    "07": 0.640,  # 蒲郡
    "08": 0.550,  # 常滑
    "09": 0.540,  # 津
    "10": 0.500,  # 三国
    "11": 0.500,  # びわこ
    "12": 0.610,  # 住之江
    "13": 0.560,  # 尼崎
    "14": 0.480,  # 鳴門
    "15": 0.550,  # 丸亀
    "16": 0.520,  # 児島
    "17": 0.600,  # 宮島
    "18": 0.500,  # 徳山
    "19": 0.480,  # 下関
    "20": 0.530,  # 若松
    "21": 0.540,  # 芦屋
    "22": 0.530,  # 福岡
    "23": 0.520,  # 唐津
    "24": 0.670,  # 大村（全国最高）
}
# レース番号補正（早いレースほど逃げやすい傾向）
_RACE_NO_NIGERATE_FACTOR = {
    1: 1.06, 2: 1.04, 3: 1.02, 4: 1.01, 5: 1.00, 6: 0.99,
    7: 0.98, 8: 0.97, 9: 0.96, 10: 0.95, 11: 0.94, 12: 0.93,
}


def _calc_nigerate(race_row: pd.Series) -> str:
    """
    1号艇のイン逃げ推定率を計算して "逃げ推定XX%" 文字列で返す。
    会場実績ベース率 × レース番号係数 + リアルタイム補正（ST・グレード）
    """
    venue_code = str(race_row.get("venue_code", "")).zfill(2)
    race_no = int(race_row.get("race_no", 6) or 6)

    base = _VENUE_INNER_ESCAPE.get(venue_code, 0.540)
    rn_factor = _RACE_NO_NIGERATE_FACTOR.get(race_no, 1.00)
    nigerate = base * rn_factor

    # 1号艇の展示ST補正（最重要リアルタイム情報、±8%）
    st1 = _safe_float(race_row.get("boat1_exhibition_st"))
    if st1 and st1 > 0:
        nigerate += np.clip((0.155 - st1) * 2.0, -0.08, 0.08)

    # 2号艇のグレード補正（差し圧力）
    g2 = _safe_float(race_row.get("boat2_grade_num")) or 2.0
    if g2 >= 4:    # A1 → 差し来る可能性高い
        nigerate -= 0.04
    elif g2 >= 3:  # A2
        nigerate -= 0.02

    # 2号艇の展示ST補正（速いほど差しに来る）
    st2 = _safe_float(race_row.get("boat2_exhibition_st"))
    if st2 and st2 > 0:
        nigerate += np.clip((st2 - 0.155) * 1.0, -0.03, 0.03)

    # 一般戦は微減
    mg = _safe_float(race_row.get("meet_grade_num"))
    if mg is not None and mg <= 1:
        nigerate -= 0.02

    nigerate = max(0.05, min(0.95, nigerate))
    return f"逃げ{int(round(nigerate * 100))}%"


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

    # グレード（A1=4→+0.8, A2=3→+0.4, B1/B2=0）
    gn_raw = race_row.get(f"boat{bn}_grade_num", 2)
    try:
        gn = int(float(gn_raw)) if gn_raw is not None else 2
        if gn >= 3:
            score += (gn - 2) * 0.4
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


def _calc_inner_score(bn: int, race_row: pd.Series) -> float:
    """
    内側艇（2・3号）の2着適性スコア。
    展示ST・展示タイム・モーター・グレード・今節成績を合算。
    """
    score = 0.0

    exh_st = _safe_float(race_row.get(f"boat{bn}_exhibition_st"))
    if exh_st and exh_st > 0:
        score += (0.155 - exh_st) * 10.0

    et_vals = [_safe_float(race_row.get(f"boat{b}_exhibition_time")) for b in range(1, 7)]
    et_vals = [v for v in et_vals if v and v > 0]
    et_bn = _safe_float(race_row.get(f"boat{bn}_exhibition_time"))
    if et_bn and et_bn > 0 and et_vals:
        avg_et = sum(et_vals) / len(et_vals)
        score += (avg_et - et_bn) * 5.0

    m = _safe_float(race_row.get(f"boat{bn}_motor_2rate"))
    if m is not None:
        score += m * 2.0

    gn_raw = race_row.get(f"boat{bn}_grade_num", 2)
    try:
        gn = int(float(gn_raw)) if gn_raw is not None else 2
        if gn >= 4:
            score += 0.8
        elif gn >= 3:
            score += 0.4
    except (TypeError, ValueError):
        pass

    meet_avg_rank = _safe_float(race_row.get(f"boat{bn}_meet_avg_rank"))
    meet_races = int(race_row.get(f"boat{bn}_meet_races", 0) or 0)
    if meet_avg_rank is not None and meet_races >= 1:
        score += (3.5 - meet_avg_rank) * 0.3

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


def _calc_boat1_weakness(race_row: pd.Series, weather: dict = None) -> tuple[int, list[str]]:
    """
    1号艇が弱くなる条件を8項目でカウントする。
    3個以上で「弱イン」ラベルを付与する判断に使う。
    Returns: (count, [条件フラグリスト])
    """
    count = 0
    flags = []

    # 展示ST遅い（0.18以上）
    st1 = _safe_float(race_row.get("boat1_exhibition_st"))
    if st1 is not None and st1 >= 0.18:
        count += 1
        flags.append(f"展示ST{st1:.2f}")

    # 今節平均ST遅い（0.17以上）
    meet_st1 = _safe_float(race_row.get("boat1_meet_avg_st"))
    if meet_st1 is not None and meet_st1 >= 0.17:
        count += 1
        flags.append(f"節ST{meet_st1:.2f}")

    # モーター2連率低い（0.35未満）
    m1 = _safe_float(race_row.get("boat1_motor_2rate"))
    if m1 is not None and m1 < 0.35:
        count += 1
        flags.append(f"M{m1:.0%}")

    # B級選手（grade_num≤2）
    g1 = _safe_float(race_row.get("boat1_grade_num"))
    if g1 is not None and g1 <= 2:
        count += 1
        flags.append("B級")

    # 全国2連率低い（0.40未満）
    n2_1 = _safe_float(race_row.get("boat1_national_2rate"))
    if n2_1 is not None and n2_1 < 0.40:
        count += 1
        flags.append(f"2率{n2_1:.0%}")

    # 今節平均着順不振（4.0以上、2R以上出走）
    meet_avg_rank1 = _safe_float(race_row.get("boat1_meet_avg_rank"))
    meet_races1 = int(race_row.get("boat1_meet_races", 0) or 0)
    if meet_avg_rank1 is not None and meet_races1 >= 2 and meet_avg_rank1 >= 4.0:
        count += 1
        flags.append(f"今節{meet_avg_rank1:.1f}着")

    # 展示タイム全艇中最遅
    et_vals = {}
    for bn in range(1, 7):
        v = _safe_float(race_row.get(f"boat{bn}_exhibition_time"))
        if v and v > 0:
            et_vals[bn] = v
    if len(et_vals) >= 4 and 1 in et_vals:
        if max(et_vals, key=lambda b: et_vals[b]) == 1:
            count += 1
            flags.append(f"展示最遅({et_vals[1]:.2f}s)")

    # 風速5m以上（強風はインに不利）
    wind = _safe_float((weather or {}).get("wind_speed"))
    if wind and wind >= 5:
        count += 1
        wind_dir = (weather or {}).get("wind_direction", "")
        dir_suffix = {"tail": "追", "head": "向", "side": "横"}.get(wind_dir, "")
        flags.append(f"風速{int(wind)}m{dir_suffix}")

    return count, flags


def _calc_arare_score(race_row: pd.Series, weather: dict = None, by_prob: "pd.DataFrame | None" = None) -> tuple[int, list[str]]:
    """
    荒れ条件スコアを計算する（新版）
    ① 1号艇の弱さ（最大9点）: ST遅+3 / モーター低+3 / B級+2 / 展示最遅+1
    ①' 2号艇の弱さ（最大4点）: ST遅(≥0.17)+2 / モーター低(<0.35)+2
    ② 外艇の脅威（最大10点）: 前付け+3 / 外A1+2 / 外ST速+2 / 外タイム最速+1 / 複数外艇ST速(2艇以上)+2
    ③ 環境・条件（最大4点）: 強風+1 / 波高+1 / 荒れ会場+1〜2 / 一般戦+1
    ④ 市場集中度（最大3点・ライブオッズ時のみ）: 1号艇1着最安≤4倍+3 / ≤7倍+2 / ≤12倍+1
    """
    score = 0
    reasons = []

    # ── ① 1号艇の弱さ（最大9点） ──
    st1 = _safe_float(race_row.get("boat1_exhibition_st"))
    if st1 is not None and st1 >= 0.18:
        score += 3
        reasons.append(f"1号ST遅({st1:.2f})")

    m1 = _safe_float(race_row.get("boat1_motor_2rate"))
    if m1 is not None and m1 < 0.30:
        score += 3
        reasons.append(f"1号M低({m1:.0%})")

    g1 = _safe_float(race_row.get("boat1_grade_num"))
    if g1 is not None and g1 <= 2:
        score += 2
        reasons.append("1号B級")

    et_vals = {}
    for bn in range(1, 7):
        v = _safe_float(race_row.get(f"boat{bn}_exhibition_time"))
        if v and v > 0:
            et_vals[bn] = v
    if len(et_vals) >= 4 and 1 in et_vals:
        if max(et_vals, key=lambda b: et_vals[b]) == 1:
            score += 1
            reasons.append(f"1号展示最遅({et_vals[1]:.2f}s)")

    # ── ①' 2号艇の弱さ（最大4点） ──
    st2 = _safe_float(race_row.get("boat2_exhibition_st"))
    if st2 is not None and st2 >= 0.17:
        score += 2
        reasons.append(f"2号ST遅({st2:.2f})")

    m2 = _safe_float(race_row.get("boat2_motor_2rate"))
    if m2 is not None and m2 < 0.35:
        score += 2
        reasons.append(f"2号M低({m2:.0%})")

    # ── ② 外艇の脅威（最大10点） ──
    # 前付け（4-6号艇が1-2コース進入、1艇でOK）
    for bn in [4, 5, 6]:
        ac_raw = race_row.get(f"boat{bn}_actual_course")
        try:
            ac = int(ac_raw) if ac_raw is not None else bn
        except (TypeError, ValueError):
            ac = bn
        if ac <= 2 and ac != bn:
            score += 3
            reasons.append(f"{bn}号前付け(→{ac}C)")
            break

    # 外艇A1選手（1号艇がA1でない場合のみ）
    g1_is_a1 = g1 is not None and g1 >= 4
    if not g1_is_a1:
        for bn in [4, 5, 6]:
            g = _safe_float(race_row.get(f"boat{bn}_grade_num"))
            if g is not None and g >= 4:
                score += 2
                reasons.append(f"{bn}号A1選手")
                break

    # 外艇展示ST ≤ 0.12（最速1艇のみ・3〜6号対象）
    best_outer_st = 9.99
    best_outer_st_bn = None
    for bn in [3, 4, 5, 6]:
        st = _safe_float(race_row.get(f"boat{bn}_exhibition_st"))
        if st and st > 0 and st < best_outer_st:
            best_outer_st = st
            best_outer_st_bn = bn
    if best_outer_st_bn is not None and best_outer_st <= 0.12:
        score += 2
        reasons.append(f"{best_outer_st_bn}号ST速({best_outer_st:.2f})")

    # 複数外艇ST速い（3〜6号で≤0.13が2艇以上）
    fast_outer_count = sum(
        1 for bn in [3, 4, 5, 6]
        if (_ost := _safe_float(race_row.get(f"boat{bn}_exhibition_st"))) and _ost > 0 and _ost <= 0.13
    )
    if fast_outer_count >= 2:
        score += 2
        reasons.append(f"外艇ST速({fast_outer_count}艇≤0.13)")

    # 外艇展示タイム最速（3〜6号）
    if len(et_vals) >= 4:
        fastest_boat = min(et_vals, key=lambda b: et_vals[b])
        if fastest_boat in {3, 4, 5, 6}:
            score += 1
            reasons.append(f"{fastest_boat}号展示最速({et_vals[fastest_boat]:.2f}s)")

    # ── ③ 環境・条件（最大4点） ──
    wind = _safe_float((weather or {}).get("wind_speed"))
    if wind and wind >= 7:
        score += 1
        reasons.append(f"強風{int(wind)}m")

    wave = _safe_float((weather or {}).get("wave_height"))
    if wave and wave >= 15:
        score += 1
        reasons.append(f"波高{int(wave)}cm")

    vc = str(race_row.get("venue_code", "")).zfill(2)
    venue_pts = ARARE_VENUES.get(vc, 0)
    if venue_pts:
        score += venue_pts
        reasons.append("江戸川" if venue_pts == 2 else "荒れ会場")

    mg = _safe_float(race_row.get("meet_grade_num"))
    if mg is not None and mg <= 1:
        score += 1
        reasons.append("一般戦")

    # ── ④ 市場集中度（ライブオッズがある場合のみ・最大3点） ──
    # 1号艇が1着の3連単オッズが低い = 公衆が1号艇に集中 = 他の組み合わせが膨らむ
    if by_prob is not None and not by_prob.empty and "odds_source" in by_prob.columns:
        b1_live = by_prob[(by_prob["odds_source"] == "live") & (by_prob["boat1"] == 1)]
        if not b1_live.empty:
            min_b1_odds = float(b1_live["odds_value"].min())
            if min_b1_odds <= 4.0:
                score += 3
                reasons.append(f"超人気集中(1号最安{min_b1_odds:.1f}倍)")
            elif min_b1_odds <= 7.0:
                score += 2
                reasons.append(f"人気集中(1号最安{min_b1_odds:.1f}倍)")
            elif min_b1_odds <= 12.0:
                score += 1
                reasons.append(f"1号人気(1号最安{min_b1_odds:.1f}倍)")

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

        # ML予測を先に計算（荒れPTにML外艇シェアを使うため）
        predictions = predict_race(model, race_row, payout_lookup, live_odds, weather, absent_boats)
        by_prob = predictions.sort_values("prob", ascending=False).reset_index(drop=True)

        # 荒れ条件スコアリング
        arare_score, arare_reasons = _calc_arare_score(race_row, weather, by_prob)
        venue_name_log = race_row.get("venue_name", "")
        print(f"  [スコア] {venue_name_log} {race_no}R PT={arare_score}"
              + (f" [{', '.join(arare_reasons)}]" if arare_reasons else ""))

        # 1号艇弱体チェック（荒れPTと独立した弱イン判定）
        boat1_weak_count, boat1_weak_flags = _calc_boat1_weakness(race_row, weather)
        weak_in = boat1_weak_count >= 3
        if weak_in:
            print(f"  [弱イン検出] {venue_name_log} {race_no}R {boat1_weak_count}条件:[{', '.join(boat1_weak_flags)}]")

        # ── 共通準備: ET・ST・MLランク ───────────────────────────
        available_kuma = [b for b in range(1, 7) if not (absent_boats and b in absent_boats)]
        et_vals_k: dict = {}
        for _b in available_kuma:
            _v = _safe_float(race_row.get(f"boat{_b}_exhibition_time"))
            if _v and _v > 0:
                et_vals_k[_b] = _v
        et_ranks_k: dict = {}
        if len(et_vals_k) >= 2:
            _et_s = sorted(et_vals_k.items(), key=lambda x: x[1])
            for _ri, (_bk, _) in enumerate(_et_s):
                et_ranks_k[_bk] = 1.0 - _ri / max(len(_et_s) - 1, 1)

        def _kuma_st_sc(bn):
            st = _safe_float(race_row.get(f"boat{bn}_exhibition_st"))
            return max(0.0, min(1.0, (0.20 - st) / 0.10)) if (st and st > 0) else 0.5

        def _kuma_ml_1st(bn):
            return float(by_prob[by_prob["boat1"] == bn]["prob"].sum())

        def _kuma_place_score(bn, first_set):
            """2着・3着候補スコア: ML条件付き確率×45% + ET×25% + ST×20% + 展開ボーナス×10%"""
            prob = float(by_prob[
                (by_prob["boat1"].isin(list(first_set))) &
                ((by_prob["boat2"] == bn) | (by_prob["boat3"] == bn))
            ]["prob"].sum())
            et   = et_ranks_k.get(bn, 0.5)
            st   = _kuma_st_sc(bn)
            # 1号艇: まくり展開でも逃げ残りパターンが多いため中程度のボーナス
            if bn == 1:
                b1_motor = _safe_float(race_row.get("boat1_motor_2rate")) or 0.0
                b1_grade = _safe_float(race_row.get("boat1_grade_num")) or 2.0
                bonus = 0.03 + min(0.02, b1_motor * 0.04) + (0.02 if b1_grade >= 3 else 0.0)
            elif bn in (2, 3):
                bonus = 0.05
            else:
                bonus = 0.03
            return prob * 0.45 + et * 0.25 + st * 0.20 + bonus * 0.10

        def _kuma_min_odds(formation_str):
            """フォーメーション内最低オッズと出所を返す"""
            parts = formation_str.split("-")
            if len(parts) != 3:
                return None, None
            min_val, min_src = None, None
            for _f in parts[0]:
                for _s in parts[1]:
                    if _s == _f:
                        continue
                    for _t in parts[2]:
                        if _t == _f or _t == _s:
                            continue
                        _om = by_prob[
                            (by_prob["boat1"] == int(_f)) &
                            (by_prob["boat2"] == int(_s)) &
                            (by_prob["boat3"] == int(_t))
                        ]
                        if not _om.empty and "odds_value" in _om.columns:
                            _ov = float(_om.iloc[0].get("odds_value", 0) or 0)
                            if _ov > 0 and (min_val is None or _ov < min_val):
                                min_val = _ov
                                min_src = str(_om.iloc[0].get("odds_source", ""))
            return min_val, min_src

        def _fmt_kuma_odds(min_val, min_src):
            if min_val and min_src == "live":
                return f"{min_val:.1f}倍"
            elif min_val:
                return f"{min_val:.1f}倍(履歴)"
            return "-"

        outer_kuma = [b for b in available_kuma if b != 1]

        _et_count = len(et_vals_k)
        _st_count = sum(
            1 for _b in available_kuma
            if (_safe_float(race_row.get(f"boat{_b}_exhibition_st")) or 0) > 0
        )
        _data_tag = ("ET○" if _et_count >= 4 else "ET×") + ("ST○" if _st_count >= 4 else "ST×")

        def _get_aggression(bn):
            """_RACER_STYLEから選手の積極性スコアを取得（データなし時は1.0=ニュートラル）"""
            racer_no = int(race_row.get(f"boat{bn}_racer_no", 0) or 0)
            if racer_no in _RACER_STYLE:
                style = _RACER_STYLE[racer_no]
                return style.get("aggression_score", 1.0) if isinstance(style, dict) else float(style)
            return 1.0

        def _infer_tactic(bn):
            """展開推定: 選手データがあれば利用、なければポジションで判断（balanced廃止）"""
            racer_no = int(race_row.get(f"boat{bn}_racer_no", 0) or 0)
            if racer_no in _RACER_STYLE:
                agg = _get_aggression(bn)
                if agg < 0.7:
                    return "sashi"
                elif agg > 1.3:
                    return "makuri"
            return "sashi" if bn <= 3 else "makuri"

        def _first_cand_score(bn):
            """1着2艇目スコア: コース別重み(内=ML重視/外=ET重視) + コース位置ボーナス + 1号艇との相対ボーナス"""
            ml    = _kuma_ml_1st(bn)
            et    = et_ranks_k.get(bn, 0.5)
            motor = _safe_float(race_row.get(f"boat{bn}_motor_2rate")) or 0.40
            grade = _safe_float(race_row.get(f"boat{bn}_grade_num")) or 2.0
            g_bonus  = 0.06 if grade >= 4 else (0.03 if grade >= 3 else 0.0)
            pos_bonus = {2: 0.08, 3: 0.04, 4: 0.03, 5: 0.01, 6: 0.0}.get(bn, 0.0)
            if bn <= 3:  # 差し展開: ML重め
                base = ml * 0.45 + et * 0.30 + motor * 0.15 + g_bonus * 0.10
            else:        # まくり展開: ET重め
                base = ml * 0.30 + et * 0.45 + motor * 0.15 + g_bonus * 0.10
            b1_et    = et_ranks_k.get(1, 0.5)
            b1_motor = _safe_float(race_row.get("boat1_motor_2rate")) or 0.40
            et_adv    = et - b1_et
            motor_adv = motor - b1_motor
            rel = max(-0.05, min(0.15, et_adv * 0.7 + motor_adv * 0.3))
            return base + pos_bonus + rel * 0.15

        def _second_cand_score(bn, adj_outside):
            """2着+1艇(C)スコア: ベーススコア + 直外展開ボーナス（展開重視型）"""
            ml    = _kuma_ml_1st(bn)
            et    = et_ranks_k.get(bn, 0.5)
            motor = _safe_float(race_row.get(f"boat{bn}_motor_2rate")) or 0.40
            base  = ml * 0.40 + et * 0.40 + motor * 0.20
            return base + (0.19 if bn == adj_outside else 0.0)

        def _third_cand_score(bn):
            """3着D/E選出スコア"""
            ml    = _kuma_ml_1st(bn)
            et    = et_ranks_k.get(bn, 0.5)
            motor = _safe_float(race_row.get(f"boat{bn}_motor_2rate")) or 0.40
            return ml * 0.40 + et * 0.40 + motor * 0.20

        # ── 大熊選出（hero1着固定 × MLモデル確率上位2点）──
        _valid = by_prob[
            (by_prob["odds_value"] > 0) &
            (by_prob["prob"] > 0)
        ].copy()

        # ── 大穴シグナル判定（PT閾値廃止・条件の組み合わせで参戦） ──
        _sig_text = " ".join(arare_reasons)
        _sig_count = sum([
            "M低" in _sig_text,
            "展示最遅" in _sig_text,
            "A1選手" in _sig_text,
            ("荒れ会場" in _sig_text or "江戸川" in _sig_text),
        ])
        _super_conc = "超人気集中" in _sig_text

        if _sig_count >= 2 and not _super_conc:
            _okuma_label = "神熱"
            _label_color = _C_YELLOW
        else:
            _okuma_label = "見送り"
            _label_color = ""
        print(f"  {_label_color}[{_okuma_label}]{_C_RESET} 荒れPT={arare_score} シグナル={_sig_count}/4"
              + (" ⚠超人気集中除外" if _super_conc else ""))

        def _add_rec(row, tier, label_override=None):
            f, s, t  = int(row["boat1"]), int(row["boat2"]), int(row["boat3"])
            combo    = f"{f}-{s}-{t}"
            ev_val   = float(row.get("ev", float(row["prob"]) * float(row["odds_value"])))
            odds_val = float(row["odds_value"])
            src      = str(row.get("odds_source", ""))
            odds_str = f"{odds_val:.0f}倍" if src == "live" else f"{odds_val:.0f}倍(履歴)"
            bl       = label_override if label_override is not None else _okuma_label
            _clr = _label_color if bl != "見送り" else ""
            print(f"  [{tier}] {venue_name_log} {race_no}R {combo} {odds_str} {_clr}PT:{arare_score}={bl}{_C_RESET}")
            all_recommendations.append({
                "date":          race_row.get("date", ""),
                "venue_name":    race_row.get("venue_name", ""),
                "race_no":       race_row.get("race_no", ""),
                "race_grade":    race_row.get("meet_grade", ""),
                "combination":   combo,
                "prob":          f"{float(row['prob']):.4f}",
                "odds":          odds_str,
                "expected_roi":  f"{ev_val:.2f}",
                "confidence":    tier,
                "odds_source":   src,
                "nigerate_str":  nigerate_str,
                "tier":          tier,
                "bet_label":     bl,
                "edge":          f"{ev_val:.2f}",
                "arare_score":        arare_score,
                "arare_reasons":      " / ".join(arare_reasons),
                "boat1_risk":         _calc_boat1_risk(race_row),
                "okuma_signal_count": _sig_count,
            })

        # ── 全艇波乱スコア（大穴シグナル方式・hero固定なし） ──
        _all_et_sorted = sorted(et_vals_k.keys(), key=lambda b: et_vals_k[b]) if et_vals_k else []
        _all_n_et = len(_all_et_sorted)

        def _hairan_t2_score(b1, b2, b3):
            """全艇波乱スコア: 1着(50%) + 2着(35%) + 3着(15%) × 攻撃性/ST/ET"""
            sc = 0.0
            for wt, bn in [(0.50, b1), (0.35, b2), (0.15, b3)]:
                bn_maku = min(1.0, _get_aggression(bn) / 2.0) * 0.55
                bn_st_raw = _safe_float(race_row.get(f"boat{bn}_exhibition_st"))
                bn_st = min(1.0, max(0.0, (0.20 - bn_st_raw) / 0.10)) * 0.30 if bn_st_raw and bn_st_raw > 0 else 0.0
                bn_et_rank = _all_et_sorted.index(bn) if bn in _all_et_sorted and _all_n_et > 1 else _all_n_et - 1
                bn_et = (1.0 - bn_et_rank / max(_all_n_et - 1, 1)) * 0.15
                sc += (bn_maku + bn_st + bn_et) * wt
            return sc

        def _pick_by_ev(cand_df, label_override, n_each=2):
            """1号艇頭EV上位n点 + 2〜5号艇頭EV上位n点を選出してDBに登録"""
            cand = cand_df.copy()
            cand["ev_score"] = cand["prob"].astype(float) * cand["odds_value"].astype(float)
            g1 = cand[cand["boat1"].astype(int) == 1].sort_values("ev_score", ascending=False)
            g2 = cand[(cand["boat1"].astype(int) >= 2) & (cand["boat1"].astype(int) <= 5)].sort_values("ev_score", ascending=False)
            g1_picks = list(g1.head(n_each).iterrows())
            g2_picks = list(g2.head(n_each).iterrows())
            print(f"  [1号艇頭EV上位] {[(int(r['boat1']),int(r['boat2']),int(r['boat3']),round(r['ev_score'],1)) for _,r in g1_picks]}")
            print(f"  [他艇頭EV上位] {[(int(r['boat1']),int(r['boat2']),int(r['boat3']),round(r['ev_score'],1)) for _,r in g2_picks]}")
            for _, row in g1_picks + g2_picks:
                r = row.copy()
                r["ev"] = float(r["ev_score"])
                _add_rec(r, "神熱", label_override=label_override)

        if _okuma_label == "神熱" and not _valid.empty:
            print(f"  {_C_GREEN}[神熱参戦]{_C_RESET} PT={arare_score} シグナル={_sig_count}/4 → 80倍超・1号艇頭EV上位2点＋他艇頭EV上位2点")
            _t2_cand = _valid[_valid["odds_value"] > 80.0].copy()
            if not _t2_cand.empty:
                _pick_by_ev(_t2_cand, _okuma_label)
            else:
                print(f"  [80倍超なし] {venue_name_log} {race_no}R → 見送り")
        else:
            _skip_rsn = (f"シグナル={_sig_count}/4 超人気集中除外" if _super_conc
                         else f"シグナル={_sig_count}/4（2未満）PT={arare_score}")
            print(f"  [見送り] {venue_name_log} {race_no}R {_skip_rsn}")
            # データ収集: 80倍超の組み合わせを記録（見送り扱い）
            if not _valid.empty:
                _dc_cand = _valid[_valid["odds_value"] > 80.0].copy()
                if not _dc_cand.empty:
                    _pick_by_ev(_dc_cand, "見送り")

    result_df = pd.DataFrame(all_recommendations)
    if not result_df.empty:
        # 同一レース・同一買い目の重複除去（df_featuresが複数行の場合の保険）
        result_df = result_df.drop_duplicates(
            subset=["date", "venue_name", "race_no", "combination"], keep="first"
        ).reset_index(drop=True)
    return result_df


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
