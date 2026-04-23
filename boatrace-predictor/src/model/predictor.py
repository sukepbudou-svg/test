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
from src.agents.racer_performance import predict_win_probs as racer_win_probs
from src.agents.motor_form import predict_win_probs as motor_win_probs

# エージェントの重み（合計1.0）
WEIGHT_ML     = 0.45  # AIモデルエージェント
WEIGHT_COURSE = 0.20  # コース戦略エージェント
WEIGHT_RACER  = 0.20  # 選手成績エージェント
WEIGHT_MOTOR  = 0.15  # モーター状態エージェント

# 3連単の控除率（約75%が払い戻し）
TRIFECTA_RETURN_RATE = 0.75

# 推奨する最低期待回収率
MIN_EXPECTED_ROI = 1.10  # 110%以上のみ推奨

# 推奨する最低的中確率（これ未満は大穴すぎて除外）
MIN_PROB = 0.02  # 2%以上のみ推奨

# 推奨する最低的中確率（これ未満は除外）
# ※オッズは市場の値をそのまま使用し、倍率によるフィルタリングは行わない

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"
PAYOUT_LOOKUP_PATH = MODEL_DIR / "payout_by_rank.json"


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
    天候・風速・波高に基づいて艇別勝率を微調整する。
    一般則: 風強・波高 → 内コース（1,2番）有利、外コース（5,6番）不利
    """
    wind = weather.get("wind_speed", 0) or 0
    wave = weather.get("wave_height", 0) or 0
    cond = weather.get("weather") or ""

    # 各艇への調整量（合計ゼロで正規化不要）
    adj = np.zeros(6)

    if wind >= 5:
        adj += np.array([+0.025, +0.015, +0.005, 0.0, -0.015, -0.030])
    elif wind >= 3:
        adj += np.array([+0.010, +0.007, +0.003, 0.0, -0.007, -0.013])

    if wave >= 20:
        adj += np.array([+0.015, +0.010, +0.005, 0.0, -0.010, -0.020])
    elif wave >= 10:
        adj += np.array([+0.008, +0.005, +0.002, 0.0, -0.005, -0.010])

    if cond == "rain":
        adj += np.array([+0.005, +0.003, +0.001, 0.0, -0.003, -0.006])

    probs = np.clip(win_probs + adj, 0.001, None)
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

    # 各エージェントの単独トップ5を取得（合議チェック用）
    def _top5(probs: np.ndarray) -> set:
        """勝率配列からHarville公式でトップ5の3連単組み合わせを返す"""
        top = {}
        for b1, b2, b3 in permutations(range(1, 7), 3):
            p1 = probs[b1-1]
            r1 = np.array([probs[i] for i in range(6) if i != b1-1])
            p2 = probs[b2-1] / r1.sum() if r1.sum() > 0 else 0
            r2 = np.array([probs[i] for i in range(6) if i not in (b1-1, b2-1)])
            p3 = probs[b3-1] / r2.sum() if r2.sum() > 0 else 0
            top[f"{b1}-{b2}-{b3}"] = p1 * p2 * p3
        return set(sorted(top, key=top.get, reverse=True)[:5])

    ml_top5 = _top5(ml_probs)
    cs_top5 = _top5(cs_probs)
    rp_top5 = _top5(rp_probs)
    mt_top5 = _top5(mt_probs)

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
            # 市場オッズをそのまま使用（フィルタなし）
            actual_odds = live_odds[combination]
            odds_value = round(actual_odds, 1)
            expected_roi = prob * actual_odds
            odds_source = "live"
        else:
            # 履歴ルックアップ使用（払戻円 ÷ 100 = 倍率）
            hist_payout = payout_lookup.get(str(rank), int(300 * (rank ** 0.8)))
            odds_value = round(hist_payout / 100, 1)
            expected_roi = prob * odds_value
            odds_source = "history"

        # エージェント合議数（0〜4）
        agreement = sum([
            combination in ml_top5,
            combination in cs_top5,
            combination in rp_top5,
            combination in mt_top5,
        ])

        results.append({
            "combination": combination,
            "boat1": b1, "boat2": b2, "boat3": b3,
            "prob": round(prob, 6),
            "popularity_rank": rank,
            "odds_value": odds_value,
            "expected_roi": round(expected_roi, 4),
            "odds_source": odds_source,
            "agreement": agreement,
        })

    df = pd.DataFrame(results).sort_values("expected_roi", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def get_recommendations(
    model: lgb.Booster,
    df_today: pd.DataFrame,
    top_n: int = 5,
    min_roi: float = MIN_EXPECTED_ROI,
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

        predictions = predict_race(model, race_row, payout_lookup, live_odds, weather, absent_boats)
        by_prob = predictions.sort_values("prob", ascending=False).reset_index(drop=True)

        def pick_tier(pool, exclude_combos, min_odds, max_odds, n, tier_name, fallback_min=None, fallback_max=None):
            """指定オッズ範囲からエージェント合議＋ROIで上位n点を選出"""
            candidates = pool[
                (pool["odds_value"] >= min_odds) & (pool["odds_value"] <= max_odds)
            ].copy()
            candidates = candidates[~candidates["combination"].isin(exclude_combos)]
            candidates = candidates.sort_values(["agreement", "expected_roi"], ascending=[False, False])
            result = candidates.head(n).copy()

            # 足りない場合はフォールバック範囲で補完
            if len(result) < n and fallback_min is not None:
                fb = pool[
                    (pool["odds_value"] >= fallback_min) & (pool["odds_value"] <= fallback_max)
                ].copy()
                fb = fb[~fb["combination"].isin(exclude_combos)]
                fb = fb[~fb["combination"].isin(result["combination"])]
                fb = fb.sort_values(["agreement", "expected_roi"], ascending=[False, False])
                extra = fb.head(n - len(result)).copy()
                result = pd.concat([result, extra]).reset_index(drop=True)

            result["tier"] = tier_name
            return result

        used = set()

        # ── 小穴3点: 26〜80倍 ──
        t0b = pick_tier(by_prob, used, 26.0, 80.0, 3, "小穴")
        used.update(t0b["combination"].tolist())

        # ── 大穴100～3点: 100〜150倍 ──
        t1 = pick_tier(by_prob, used, 100.0, 150.0, 3, "大穴100～", 81.0, 170.0)
        used.update(t1["combination"].tolist())

        # ── 大大アナ2点: 201〜300倍 ──
        t2 = pick_tier(by_prob, used, 201.0, 300.0, 2, "大大アナ", 180.0, 320.0)
        used.update(t2["combination"].tolist())

        # ── 大大大アナ2点: 301〜500倍 ──
        t3 = pick_tier(by_prob, used, 301.0, 500.0, 2, "大大大アナ", 260.0, 550.0)
        used.update(t3["combination"].tolist())

        # ── 特大アナ4点: 501倍以上 ──
        t4 = pick_tier(by_prob, used, 501.0, 99999.0, 4, "特大アナ", 400.0, 99999.0)
        used.update(t4["combination"].tolist())

        recommended = pd.concat([t0b, t1, t2, t3, t4]).reset_index(drop=True)

        if recommended.empty:
            all_recommendations.append({
                "date": race_row.get("date", ""),
                "venue_name": race_row.get("venue_name", ""),
                "race_no": race_row.get("race_no", ""),
                "combination": "見送り",
                "prob": "0%",
                "odds": "-",
                "expected_roi": "0%",
                "confidence": "見送り",
                "odds_source": "-",
                "tier": "-",
            })
        else:
            for _, rec in recommended.iterrows():
                agreement = int(rec.get("agreement", 0))
                # 4エージェント中: ★★★=3〜4合意, ★★☆=2合意, ★☆☆=1以下
                confidence = "★★★★" if agreement >= 4 else "★★★☆" if agreement >= 3 else "★★☆☆" if agreement >= 2 else "★☆☆☆"
                src = rec.get("odds_source", "history")
                odds_display = f"{rec['odds_value']}倍" if src == "live" else f"{rec['odds_value']}倍(履歴)"
                all_recommendations.append({
                    "date": race_row.get("date", ""),
                    "venue_name": race_row.get("venue_name", ""),
                    "race_no": race_row.get("race_no", ""),
                    "combination": rec["combination"],
                    "prob": f"{rec['prob']*100:.2f}%",
                    "odds": odds_display,
                    "expected_roi": f"{rec['expected_roi']*100:.0f}%",
                    "confidence": confidence,
                    "odds_source": "リアルタイム" if src == "live" else "履歴平均",
                    "tier": rec.get("tier", "本命"),  # 狙い列の値
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
