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

# 3連単の控除率（約75%が払い戻し）
TRIFECTA_RETURN_RATE = 0.75

# 推奨する最低期待回収率
MIN_EXPECTED_ROI = 1.10  # 110%以上のみ推奨

# 推奨する最低的中確率（これ未満は大穴すぎて除外）
MIN_PROB = 0.02  # 2%以上のみ推奨

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
    win_probs = model.predict(X)[0]

    # 温度スケーリングで確率の過信を補正（T=2.5: 高いほど保守的）
    TEMPERATURE = 2.5
    logits = np.log(np.clip(win_probs, 1e-10, 1.0))
    scaled = np.exp(logits / TEMPERATURE)
    win_probs = scaled / scaled.sum()

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
            # リアルタイムオッズ使用: オッズは倍率（例: 3.5 → 350円払戻）
            actual_odds = live_odds[combination]
            avg_payout = int(actual_odds * 100)
            # 期待回収率 = 的中確率 × オッズ倍率
            expected_roi = prob * actual_odds
        else:
            # 履歴ルックアップ使用
            avg_payout = payout_lookup.get(str(rank), int(300 * (rank ** 0.8)))
            expected_roi = prob * (avg_payout / 100)

        results.append({
            "combination": combination,
            "boat1": b1, "boat2": b2, "boat3": b3,
            "prob": round(prob, 6),
            "popularity_rank": rank,
            "avg_payout": avg_payout,
            "expected_roi": round(expected_roi, 4),
            "odds_source": "live" if (using_live and combination in live_odds) else "history",
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
        # 確率2%以上 かつ 期待回収率の条件を両方満たすものだけ推奨
        recommended = predictions[
            (predictions["prob"] >= MIN_PROB) &
            (predictions["expected_roi"] >= min_roi)
        ].head(top_n)

        if recommended.empty:
            all_recommendations.append({
                "date": race_row.get("date", ""),
                "venue_name": race_row.get("venue_name", ""),
                "race_no": race_row.get("race_no", ""),
                "combination": "見送り",
                "prob": "0%",
                "avg_payout": "-",
                "expected_roi": "0%",
                "confidence": "見送り",
                "odds_source": "-",
            })
        else:
            for _, rec in recommended.iterrows():
                roi = rec["expected_roi"]
                confidence = "★★★" if roi >= 1.3 else "★★☆" if roi >= 1.2 else "★☆☆"
                src = rec.get("odds_source", "history")
                all_recommendations.append({
                    "date": race_row.get("date", ""),
                    "venue_name": race_row.get("venue_name", ""),
                    "race_no": race_row.get("race_no", ""),
                    "combination": rec["combination"],
                    "prob": f"{rec['prob']*100:.2f}%",
                    "avg_payout": f"{rec['avg_payout']:,}円",
                    "expected_roi": f"{rec['expected_roi']*100:.0f}%",
                    "confidence": confidence,
                    "odds_source": "リアルタイム" if src == "live" else "履歴平均",
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
