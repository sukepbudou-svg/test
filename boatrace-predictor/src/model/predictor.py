"""
レース予想エンジン
学習済みモデルを使って3連単の買い目と期待回収率を予測する
"""

from itertools import permutations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.builder import get_feature_columns, add_course_advantage

# 3連単の平均的な控除率（約75%が払い戻し）
TRIFECTA_RETURN_RATE = 0.75

# 推奨する最低期待回収率
MIN_EXPECTED_ROI = 1.10  # 110%以上のみ推奨


def predict_race(model: lgb.Booster, race_features: pd.Series) -> pd.DataFrame:
    """
    1レース分の3連単予想を生成する

    Args:
        model: 学習済みLightGBMモデル
        race_features: 1レース分の特徴量（build_features出力の1行）

    Returns:
        DataFrame: 3連単の全組み合わせと期待回収率
          columns: combination, prob, expected_roi, rank
    """
    feature_cols = get_feature_columns() + [f"boat{bn}_course_advantage" for bn in range(1, 7)]

    X = race_features[feature_cols].fillna(0).values.reshape(1, -1)
    win_probs = model.predict(X)[0]  # 各艇の1着確率（6次元）

    # 3連単の全120通りの確率と期待回収率を計算
    combinations = list(permutations(range(1, 7), 3))
    results = []

    for combo in combinations:
        b1, b2, b3 = combo
        # 簡易確率計算（条件付き確率の近似）
        p1 = win_probs[b1 - 1]
        # 2着: 残り5艇中での相対確率
        remaining_probs = np.array([win_probs[i] for i in range(6) if i != b1 - 1])
        p2_given_p1 = win_probs[b2 - 1] / remaining_probs.sum() if remaining_probs.sum() > 0 else 0
        # 3着: 残り4艇中での相対確率
        remaining_probs2 = np.array([
            win_probs[i] for i in range(6) if i not in (b1 - 1, b2 - 1)
        ])
        p3_given_p12 = win_probs[b3 - 1] / remaining_probs2.sum() if remaining_probs2.sum() > 0 else 0

        prob = p1 * p2_given_p1 * p3_given_p12

        # 期待回収率 = モデル予測確率 / 市場確率（均等配分） × 控除率
        # 市場が均等と仮定した場合の確率 = 1/120（3連単全120通り）
        market_prob = 1 / 120
        expected_roi = (prob / market_prob) * TRIFECTA_RETURN_RATE if prob > 0 else 0

        results.append({
            "combination": f"{b1}-{b2}-{b3}",
            "boat1": b1, "boat2": b2, "boat3": b3,
            "prob": round(prob, 6),
            "expected_roi": round(expected_roi, 4),
        })

    df = pd.DataFrame(results).sort_values("prob", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def get_recommendations(
    model: lgb.Booster,
    df_today: pd.DataFrame,
    top_n: int = 5,
    min_roi: float = MIN_EXPECTED_ROI
) -> pd.DataFrame:
    """
    本日の全レースから推奨買い目を選出する

    Args:
        model: 学習済みモデル
        df_today: 本日の番組表特徴量DataFrame
        top_n: 各レースで推奨する買い目数
        min_roi: 最低期待回収率（この値以上のみ推奨）

    Returns:
        推奨買い目DataFrame
    """
    df_today = add_course_advantage(df_today)
    all_recommendations = []

    for _, race_row in df_today.iterrows():
        predictions = predict_race(model, race_row)

        # 期待回収率が閾値以上の買い目を抽出
        recommended = predictions[predictions["expected_roi"] >= min_roi].head(top_n)

        if recommended.empty:
            # 閾値を満たすものがない場合は「見送り」
            all_recommendations.append({
                "date": race_row.get("date", ""),
                "venue_name": race_row.get("venue_name", ""),
                "race_no": race_row.get("race_no", ""),
                "combination": "見送り",
                "prob": 0,
                "expected_roi": 0,
                "confidence": "見送り",
            })
        else:
            for _, rec in recommended.iterrows():
                roi = rec["expected_roi"]
                confidence = "★★★" if roi >= 1.3 else "★★☆" if roi >= 1.2 else "★☆☆"
                all_recommendations.append({
                    "date": race_row.get("date", ""),
                    "venue_name": race_row.get("venue_name", ""),
                    "race_no": race_row.get("race_no", ""),
                    "combination": rec["combination"],
                    "prob": f"{rec['prob']*100:.2f}%",
                    "expected_roi": f"{rec['expected_roi']*100:.0f}%",
                    "confidence": confidence,
                })

    return pd.DataFrame(all_recommendations)


def calculate_roi_history(df_result: pd.DataFrame, df_payout: pd.DataFrame, recommendations: pd.DataFrame) -> dict:
    """
    過去の推奨買い目の実際の回収率を計算する（バックテスト用）

    Returns:
        {"total_bet": 総購入額, "total_return": 総払戻額, "roi": 実際の回収率}
    """
    total_bet = 0
    total_return = 0

    for _, rec in recommendations.iterrows():
        if rec["combination"] == "見送り":
            continue

        total_bet += 100  # 1点100円として計算

        # 実際の払戻を検索
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
