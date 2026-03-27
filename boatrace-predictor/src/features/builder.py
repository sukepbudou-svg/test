"""
特徴量エンジニアリング
番組表・競走成績データから予測モデル用の特徴量を生成する
"""

import pandas as pd
import numpy as np


def build_features(
    df_program: pd.DataFrame,
    df_rank: pd.DataFrame,
    df_payout: pd.DataFrame,
    df_beforeinfo: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    番組表・成績データを結合して特徴量DataFrameを生成する

    Args:
        df_program: 番組表DataFrame（parse_program出力）
        df_rank: 着順DataFrame（parse_result出力）
        df_payout: 払戻DataFrame（parse_result出力）

    Returns:
        特徴量DataFrame（モデル学習・予測用）
    """
    if df_program.empty:
        return pd.DataFrame()

    # レースごとに6艇分の特徴量をピボット（予測モードでは成績なしで動作）
    predict_only = df_rank.empty

    # 展示タイムDataFrameを準備（過去成績から or 直前情報スクレイピングから）
    if df_beforeinfo is not None and not df_beforeinfo.empty:
        # 直前情報スクレイピング結果（予測時）
        exh_df = df_beforeinfo
    elif not predict_only and "exhibition_time" in df_rank.columns:
        # 過去競走成績から（学習時）
        exh_df = df_rank[["date", "venue_code", "race_no", "boat_no", "exhibition_time", "start_timing"]].copy()
        exh_df = exh_df.rename(columns={"start_timing": "exhibition_st_raw"})
        exh_df["exhibition_st"] = exh_df["exhibition_st_raw"].apply(_parse_st)
    else:
        exh_df = None

    # 3連単の払戻を結合用に整形
    if not predict_only and not df_payout.empty and "bet_type" in df_payout.columns:
        trifecta = df_payout[df_payout["bet_type"] == "３連単"][
            ["date", "venue_code", "race_no", "combination", "payout", "popularity"]
        ].copy()
    else:
        trifecta = pd.DataFrame(columns=["date", "venue_code", "race_no", "combination", "payout", "popularity"])

    # 着順から1〜3着の艇番を取得
    if not predict_only:
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
    else:
        top3 = pd.DataFrame(columns=["date", "venue_code", "race_no", "result_combination", "winner_boat"])

    # レースごとに6艇分の特徴量をピボット
    program_pivot = _pivot_program(df_program)

    # 展示タイムをピボットして結合
    if exh_df is not None and not exh_df.empty:
        exh_pivot = _pivot_exhibition(exh_df)
        program_pivot = program_pivot.merge(
            exh_pivot, on=["date", "venue_code", "race_no"], how="left"
        )
    else:
        # 展示タイムなし → NaN で埋める
        for bn in range(1, 7):
            program_pivot[f"boat{bn}_exhibition_time"] = np.nan
            program_pivot[f"boat{bn}_exhibition_st"] = np.nan

    # 結合
    if predict_only:
        # 予測モード: 番組表のみ（成績なし）
        df = program_pivot
        df["winner_boat"] = None
        df["trifecta_payout"] = 0
    else:
        df = program_pivot.merge(top3, on=["date", "venue_code", "race_no"], how="inner")
        if not trifecta.empty:
            df = df.merge(
                trifecta[["date", "venue_code", "race_no", "payout", "popularity"]],
                on=["date", "venue_code", "race_no"],
                how="left"
            )
        df["trifecta_payout"] = df.get("payout", pd.Series(0, index=df.index)).fillna(0).astype(int)

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


def _parse_st(st_raw) -> float | None:
    """スタートタイミング文字列を数値に変換"""
    if st_raw is None or (isinstance(st_raw, float) and np.isnan(st_raw)):
        return None
    s = str(st_raw).strip()
    if s in ("F", "L", "K", ""):
        return None  # フライング・出遅れ等はNaN
    try:
        return float(s)
    except ValueError:
        return None


def _pivot_exhibition(exh_df: pd.DataFrame) -> pd.DataFrame:
    """展示タイム・STデータを1レース1行形式に変換"""
    rows = []
    for (date, venue_code, race_no), group in exh_df.groupby(["date", "venue_code", "race_no"]):
        row = {"date": date, "venue_code": venue_code, "race_no": race_no}
        for _, r in group.iterrows():
            bn = int(r["boat_no"])
            row[f"boat{bn}_exhibition_time"] = r.get("exhibition_time")
            row[f"boat{bn}_exhibition_st"] = r.get("exhibition_st")
        rows.append(row)
    return pd.DataFrame(rows)


def get_feature_columns() -> list[str]:
    """モデルに使用する特徴量カラム名リストを返す"""
    base_cols = [
        "national_win_rate", "national_2rate",
        "local_win_rate", "local_2rate",
        "motor_2rate", "boat_2rate",
        "age", "weight", "grade_num",
        "exhibition_time", "exhibition_st",
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
