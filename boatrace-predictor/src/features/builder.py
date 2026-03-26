"""
特徴量エンジニアリング
番組表・競走成績データから予測モデル用の特徴量を生成する
"""

import pandas as pd
import numpy as np


def build_features(df_program: pd.DataFrame, df_rank: pd.DataFrame, df_payout: pd.DataFrame) -> pd.DataFrame:
    """
    番組表・成績データを結合して特徴量DataFrameを生成する

    Args:
        df_program: 番組表DataFrame（parse_program出力）
        df_rank: 着順DataFrame（parse_result出力）
        df_payout: 払戻DataFrame（parse_result出力）

    Returns:
        特徴量DataFrame（モデル学習・予測用）
    """
    if df_program.empty or df_rank.empty:
        return pd.DataFrame()

    # 3連単の払戻を結合用に整形
    trifecta = df_payout[df_payout["bet_type"] == "３連単"][
        ["date", "venue_code", "race_no", "combination", "payout", "popularity"]
    ].copy()

    # 着順から1〜3着の艇番を取得
    top3 = (
        df_rank[df_rank["rank"] <= 3]
        .sort_values(["date", "venue_code", "race_no", "rank"])
        .groupby(["date", "venue_code", "race_no"])["boat_no"]
        .apply(list)
        .reset_index()
        .rename(columns={"boat_no": "top3_boats"})
    )
    top3 = top3[top3["top3_boats"].apply(len) == 3].copy()
    top3["result_combination"] = top3["top3_boats"].apply(
        lambda x: f"{x[0]}-{x[1]}-{x[2]}"
    )
    top3["winner_boat"] = top3["top3_boats"].apply(lambda x: x[0])

    # レースごとに6艇分の特徴量をピボット
    program_pivot = _pivot_program(df_program)

    # 結合
    df = program_pivot.merge(top3, on=["date", "venue_code", "race_no"], how="inner")
    df = df.merge(
        trifecta[["date", "venue_code", "race_no", "payout", "popularity"]],
        on=["date", "venue_code", "race_no"],
        how="left"
    )

    # 回収率計算用の期待値フラグ（学習用ラベル）
    # 1: 的中（3連単）, 0: 外れ
    df["trifecta_payout"] = df["payout"].fillna(0).astype(int)

    return df


def _pivot_program(df_program: pd.DataFrame) -> pd.DataFrame:
    """
    番組表を「1レース1行」形式に変換する
    各艇番の特徴量をカラムとして展開
    """
    feature_cols = [
        "national_win_rate", "national_2rate",
        "local_win_rate", "local_2rate",
        "motor_2rate", "boat_2rate",
        "age", "weight",
    ]

    grade_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
    df = df_program.copy()
    df["grade_num"] = df["grade"].map(grade_map).fillna(0)

    all_feature_cols = feature_cols + ["grade_num"]
    rows = []

    for (date, venue_code, race_no), group in df.groupby(["date", "venue_code", "race_no"]):
        row = {"date": date, "venue_code": venue_code, "race_no": race_no,
               "venue_name": group["venue_name"].iloc[0]}
        for _, racer in group.iterrows():
            bn = int(racer["boat_no"])
            for col in all_feature_cols:
                row[f"boat{bn}_{col}"] = racer[col]
        rows.append(row)

    return pd.DataFrame(rows)


def get_feature_columns() -> list[str]:
    """モデルに使用する特徴量カラム名リストを返す"""
    base_cols = [
        "national_win_rate", "national_2rate",
        "local_win_rate", "local_2rate",
        "motor_2rate", "boat_2rate",
        "age", "weight", "grade_num",
    ]
    cols = []
    for bn in range(1, 7):
        for col in base_cols:
            cols.append(f"boat{bn}_{col}")
    return cols


def add_course_advantage(df: pd.DataFrame) -> pd.DataFrame:
    """
    コース有利不利の特徴量を追加
    1コースの1着率は高い傾向があるため補正値を追加
    """
    course_win_rate = {1: 0.55, 2: 0.11, 3: 0.10, 4: 0.09, 5: 0.08, 6: 0.07}
    for bn in range(1, 7):
        df[f"boat{bn}_course_advantage"] = course_win_rate.get(bn, 0.0)
    return df
