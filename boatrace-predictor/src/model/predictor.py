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


def predict_race(
    model: lgb.Booster,
    race_features: pd.Series,
    payout_lookup: dict = None,
    live_odds: dict = None,
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

    # 4エージェントの勝率を重み付け合成
    win_probs = (WEIGHT_ML * ml_probs + WEIGHT_COURSE * cs_probs
                 + WEIGHT_RACER * rp_probs + WEIGHT_MOTOR * mt_probs)

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
        # このレースのリアルタイムオッズを取得（あれば）
        live_odds = None
        if all_live_odds:
            venue_code = str(race_row.get("venue_code", "")).zfill(2)
            race_no = int(race_row.get("race_no", 0))
            live_odds = all_live_odds.get((venue_code, race_no))

        predictions = predict_race(model, race_row, payout_lookup, live_odds)
        by_prob = predictions.sort_values("prob", ascending=False).reset_index(drop=True)

        # ── 狙い撃ち3点: オッズ50〜100倍の範囲で4エージェント合議＋ROI重視 ──
        tier1_pool = by_prob[
            (by_prob["odds_value"] >= 50.0) & (by_prob["odds_value"] < 100.0)
        ].copy()
        # 合議数が高い順→ROI順でソート
        tier1_pool = tier1_pool.sort_values(
            ["agreement", "expected_roi"], ascending=[False, False]
        )
        tier1 = tier1_pool.head(3).copy()
        tier1["tier"] = "狙い撃ち"

        # 足りない場合は範囲を広げて補完
        if len(tier1) < 3:
            extra = by_prob[
                (by_prob["odds_value"] >= 40.0) & (by_prob["odds_value"] < 120.0)
            ].copy()
            extra = extra[~extra["combination"].isin(tier1["combination"])]
            extra = extra.sort_values(["agreement", "expected_roi"], ascending=[False, False])
            need = 3 - len(tier1)
            extra_pick = extra.head(need).copy()
            extra_pick["tier"] = "狙い撃ち"
            tier1 = pd.concat([tier1, extra_pick]).reset_index(drop=True)

        # ── 中穴2点: オッズ100〜150倍の範囲でROI重視 ──
        tier2_pool = by_prob[
            (by_prob["odds_value"] >= 100.0) & (by_prob["odds_value"] < 150.0)
        ].copy()
        tier2_pool = tier2_pool[~tier2_pool["combination"].isin(tier1["combination"])]
        tier2 = tier2_pool.sort_values("expected_roi", ascending=False).head(2).copy()
        tier2["tier"] = "中穴"

        # 足りない場合は範囲を広げて補完
        if len(tier2) < 2:
            extra2 = by_prob[
                (by_prob["odds_value"] >= 80.0) & (by_prob["odds_value"] < 200.0)
            ].copy()
            extra2 = extra2[~extra2["combination"].isin(tier1["combination"])]
            extra2 = extra2[~extra2["combination"].isin(tier2["combination"])]
            extra2 = extra2.sort_values("expected_roi", ascending=False)
            need2 = 2 - len(tier2)
            extra2_pick = extra2.head(need2).copy()
            extra2_pick["tier"] = "中穴"
            tier2 = pd.concat([tier2, extra2_pick]).reset_index(drop=True)

        recommended = pd.concat([tier1, tier2]).reset_index(drop=True)

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
                confidence = "★★★" if agreement >= 3 else "★★☆" if agreement >= 2 else "★☆☆"
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
