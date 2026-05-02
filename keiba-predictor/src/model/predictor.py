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

# 本命: 確率上位N点
HONMEI_N = 2
# 本命改_馬連: エッジ上位N点（期待値 > EDGE_THRESHOLD）
HONMEI_KAI_UMAREN_N = 2
EDGE_THRESHOLD = 1.0
# 本命改_3連複: 上位N頭ボックス
TOP_HORSES_N = 4
# 本命改_ワイド: エッジ上位N点
WIDE_N = 2


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
    live_wide_odds: dict = None,
) -> pd.DataFrame:
    """
    1レース分の推奨買い目を選出する

    買い方:
      本命        : 確率上位2点（馬連）
      本命改_馬連 : エッジ上位2点（period × odds > 1.0）
      本命改_3連複: 上位4頭ボックス4点
      本命改_ワイド: エッジ上位2点（ワイド）

    Returns:
        DataFrame: 推奨買い目リスト
    """
    predictions = predict_race(model, df_race, live_odds)
    if predictions.empty:
        return pd.DataFrame()

    meta = {
        "date": df_race["date"].iloc[0] if not df_race.empty else "",
        "venue": df_race["venue"].iloc[0] if not df_race.empty else "",
        "race_no": df_race["race_no"].iloc[0] if not df_race.empty else 0,
    }
    recommended = []

    def _row(tier, combo, prob, odds_disp, roi_disp, src_disp):
        return {**meta, "tier": tier, "combination": combo,
                "prob": prob, "odds": odds_disp,
                "expected_roi": roi_disp, "odds_source": src_disp}

    def _fmt(rec):
        src = rec["odds_source"]
        odds_disp = f"{rec['odds_value']}倍" if src == "live" else f"{rec['odds_value']}倍(推定)"
        src_disp = "リアルタイム" if src == "live" else "推定"
        prob_disp = f"{rec['prob']*100:.2f}%"
        roi_disp = f"{rec['expected_roi']*100:.0f}%"
        return odds_disp, src_disp, prob_disp, roi_disp

    # 1. 本命: 確率上位2点（馬連）
    honmei_picks = predictions.head(HONMEI_N)
    for _, rec in honmei_picks.iterrows():
        odds_disp, src_disp, prob_disp, roi_disp = _fmt(rec)
        recommended.append(_row("本命", rec["combination"], prob_disp, odds_disp, roi_disp, src_disp))

    # 2. 本命改_馬連: ライブオッズあり→エッジ上位2点 / なし→確率3〜4位
    has_live = (predictions["odds_source"] == "live").any()
    if has_live:
        edge_candidates = (
            predictions[predictions["expected_roi"] >= EDGE_THRESHOLD]
            .sort_values("expected_roi", ascending=False)
        )
    else:
        # 推定オッズ時はエッジが全て0.775で均一のため確率順で本命の次点を選ぶ
        edge_candidates = predictions.iloc[HONMEI_N:].copy()
    for _, rec in edge_candidates.head(HONMEI_KAI_UMAREN_N).iterrows():
        odds_disp, src_disp, prob_disp, roi_disp = _fmt(rec)
        recommended.append(_row("本命改_馬連", rec["combination"], prob_disp, odds_disp, roi_disp, src_disp))

    # 3. 本命改_3連複: 上位4頭ボックス（C(4,3)=4点）
    top_horses = _get_top_horses(predictions, n=TOP_HORSES_N)
    for trio in combinations(sorted(top_horses), 3):
        combo_str = f"{trio[0]}-{trio[1]}-{trio[2]}"
        recommended.append(_row("本命改_3連複", combo_str, "-", "-", "-", "-"))

    # 4. 本命改_ワイド: エッジ上位2点
    if live_wide_odds:
        wide_recs = _calc_wide_recs(predictions, live_wide_odds, n=WIDE_N)
        for combo, w_odds, w_edge in wide_recs:
            prob_val = predictions.loc[
                predictions["combination"] == combo, "prob"
            ].values
            prob_disp = f"{prob_val[0]*100:.2f}%" if len(prob_val) else "-"
            recommended.append(_row(
                "本命改_ワイド", combo, prob_disp,
                f"{w_odds}倍", f"{w_edge*100:.0f}%", "リアルタイム"
            ))

    if not recommended:
        return pd.DataFrame([{**meta,
            "tier": "-", "combination": "見送り",
            "prob": "0%", "odds": "-", "expected_roi": "0%", "odds_source": "-",
        }])

    return pd.DataFrame(recommended)


def _get_top_horses(predictions: pd.DataFrame, n: int = 4) -> list:
    """確率上位の組み合わせから上位n頭を抽出する"""
    seen: set = set()
    horses: list = []
    for _, row in predictions.iterrows():
        for h in [int(row["horse1"]), int(row["horse2"])]:
            if h not in seen:
                seen.add(h)
                horses.append(h)
        if len(horses) >= n:
            break
    return horses[:n]


def _calc_wide_recs(predictions: pd.DataFrame, live_wide_odds: dict, n: int = 2) -> list:
    """ワイドオッズでエッジを計算し上位n点を返す [(combo, odds, edge), ...]"""
    results = []
    for _, row in predictions.iterrows():
        combo = row["combination"]
        if combo in live_wide_odds:
            w_odds = live_wide_odds[combo]
            edge = row["prob"] * w_odds
            results.append((combo, w_odds, edge))
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:n]


def apply_training_filter(recs: pd.DataFrame, training_times: dict) -> pd.DataFrame:
    """
    追い切りタイムで推奨買い目を評価する

    フィールド内3Fタイム順位で評価:
      A = 上位1/3（速い）
      B = 中位1/3
      C = 下位1/3（遅い）
      ? = データなし

    両馬がCの場合は見送りに変更する
    """
    recs = recs.copy()
    if not training_times:
        recs["training_eval"] = "-"
        return recs

    # 3Fタイムのある馬だけでランク付け（小さいほど速い＝良い）
    times = {hn: d["time_3f"] for hn, d in training_times.items()
             if d.get("time_3f") is not None}
    if not times:
        recs["training_eval"] = "-"
        return recs

    sorted_horses = sorted(times, key=lambda h: times[h])
    n = len(sorted_horses)
    rank_map = {hn: i for i, hn in enumerate(sorted_horses)}

    def _grade(horse_no: int) -> str:
        if horse_no not in rank_map:
            return "?"
        r = rank_map[horse_no]
        if r < n / 3:
            return "A"
        elif r < 2 * n / 3:
            return "B"
        return "C"

    evals = []
    for _, rec in recs.iterrows():
        combo = rec.get("combination", "")
        parts = combo.split("-")
        if len(parts) >= 2:
            try:
                grades = [_grade(int(p)) for p in parts]
                evals.append("/".join(grades))
            except ValueError:
                evals.append("-")
        else:
            evals.append("-")

    recs["training_eval"] = evals

    # 全馬がC評価 → 見送りに変更
    def _is_both_c(ev: str) -> bool:
        parts = ev.split("/")
        return len(parts) >= 2 and all(p == "C" for p in parts)

    mask = recs["training_eval"].apply(_is_both_c)
    if mask.any():
        recs.loc[mask, "combination"] = "見送り(追い切り不調)"
        recs.loc[mask, "tier"] = "-"

    return recs
