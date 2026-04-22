"""
競馬予想エンジン
学習済みモデルを使って馬連の買い目と期待回収率を予測する
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from itertools import combinations

from src.features.builder import get_feature_columns, build_quinella_features

# 馬連の控除率（約77.5%が払い戻し）
QUINELLA_RETURN_RATE = 0.775

# 倍率帯ごとの設定
TIERS = [
    {"name": "中アナ",   "min": 10.0,  "max": 30.0,  "n": 2},
    {"name": "大アナ",   "min": 31.0,  "max": 80.0,  "n": 2},
    {"name": "穴",       "min": 81.0,  "max": 9999.0, "n": 1},
]


def predict_race(
    model: lgb.Booster,
    df_race: pd.DataFrame,
    live_odds: dict = None,
) -> pd.DataFrame:
    """
    1レース分の馬連予想を生成する

    Args:
        model: 学習済みLightGBMモデル
        df_race: build_race_features()出力（1行=1頭）
        live_odds: {"1-2": 15.3, ...} リアルタイム馬連オッズ

    Returns:
        DataFrame: 馬連全組み合わせと期待回収率
    """
    df_quinella = build_quinella_features(df_race)
    if df_quinella.empty:
        return pd.DataFrame()

    feature_cols = get_feature_columns()
    X = df_quinella[feature_cols].fillna(0).values
    probs = model.predict(X)

    results = []
    for i, row in df_quinella.iterrows():
        combo = row["combination"]
        prob = float(probs[i])

        if live_odds and combo in live_odds:
            odds_val = live_odds[combo]
            expected_roi = prob * odds_val
            odds_source = "live"
        else:
            # オッズなし時は確率から推定倍率を計算
            est_odds = QUINELLA_RETURN_RATE / max(prob, 0.001)
            odds_val = round(est_odds, 1)
            expected_roi = prob * odds_val
            odds_source = "estimated"

        results.append({
            "combination": combo,
            "horse1": int(row["horse1"]),
            "horse2": int(row["horse2"]),
            "prob": round(prob, 6),
            "odds_value": odds_val,
            "expected_roi": round(expected_roi, 4),
            "odds_source": odds_source,
        })

    df_result = pd.DataFrame(results).sort_values("prob", ascending=False).reset_index(drop=True)
    return df_result


def get_recommendations(
    model: lgb.Booster,
    df_race: pd.DataFrame,
    live_odds: dict = None,
) -> pd.DataFrame:
    """
    1レース分の推奨買い目を選出する（倍率帯別）

    Returns:
        DataFrame: 推奨買い目リスト
    """
    predictions = predict_race(model, df_race, live_odds)
    if predictions.empty:
        return pd.DataFrame()

    used = set()
    recommended = []

    for tier in TIERS:
        candidates = predictions[
            (predictions["odds_value"] >= tier["min"]) &
            (predictions["odds_value"] <= tier["max"]) &
            (~predictions["combination"].isin(used))
        ].sort_values("prob", ascending=False)

        picked = candidates.head(tier["n"])
        for _, rec in picked.iterrows():
            used.add(rec["combination"])
            src = rec["odds_source"]
            odds_disp = f"{rec['odds_value']}倍" if src == "live" else f"{rec['odds_value']}倍(推定)"
            recommended.append({
                "date": df_race["date"].iloc[0] if not df_race.empty else "",
                "venue": df_race["venue"].iloc[0] if not df_race.empty else "",
                "race_no": df_race["race_no"].iloc[0] if not df_race.empty else 0,
                "tier": tier["name"],
                "combination": rec["combination"],
                "prob": f"{rec['prob']*100:.2f}%",
                "odds": odds_disp,
                "expected_roi": f"{rec['expected_roi']*100:.0f}%",
                "odds_source": "リアルタイム" if src == "live" else "推定",
            })

    if not recommended:
        return pd.DataFrame([{
            "date": df_race["date"].iloc[0] if not df_race.empty else "",
            "venue": df_race["venue"].iloc[0] if not df_race.empty else "",
            "race_no": df_race["race_no"].iloc[0] if not df_race.empty else 0,
            "tier": "-", "combination": "見送り",
            "prob": "0%", "odds": "-", "expected_roi": "0%", "odds_source": "-",
        }])

    return pd.DataFrame(recommended)
