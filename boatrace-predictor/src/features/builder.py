"""
特徴量エンジニアリング
番組表・競走成績データから予測モデル用の特徴量を生成する
"""

import pandas as pd
import numpy as np
from pathlib import Path


def _calculate_series_day(df_program: pd.DataFrame, raw_dir: Path = None) -> pd.DataFrame:
    """
    各(venue_code, date)がシリーズの何日目かを計算する

    複数日分のデータがある場合: 連続開催日から自動計算
    単日データ(予測時)の場合: raw_dirのBファイル履歴から推定
    """
    if df_program.empty:
        return pd.DataFrame(columns=["venue_code", "date", "series_day", "is_final_day_num"])

    venue_dates = (
        df_program[["venue_code", "date"]]
        .drop_duplicates()
        .copy()
    )
    venue_dates["date_dt"] = pd.to_datetime(venue_dates["date"])
    venue_dates = venue_dates.sort_values(["venue_code", "date_dt"])

    unique_dates = venue_dates["date"].nunique()
    results = []

    if unique_dates >= 3:
        # 複数日分のデータがある場合: 連続開催日から計算
        for vc, grp in venue_dates.groupby("venue_code"):
            dates = sorted(grp["date_dt"].tolist())
            series_day_map = {}
            series_start_idx = 0
            for i, d in enumerate(dates):
                if i == 0 or (d - dates[i - 1]).days > 1:
                    series_start_idx = i
                series_day_map[d] = i - series_start_idx + 1

            final_day_set = set()
            for i, d in enumerate(dates):
                if i == len(dates) - 1 or (dates[i + 1] - d).days > 1:
                    final_day_set.add(d)

            for d in dates:
                results.append({
                    "venue_code": vc,
                    "date": d.strftime("%Y-%m-%d"),
                    "series_day": series_day_map[d],
                    "is_final_day_num": int(d in final_day_set),
                })
    else:
        # 単日データ: Bファイル履歴から推定
        for vc, grp in venue_dates.groupby("venue_code"):
            date_str = grp["date"].iloc[0]
            sd, is_final = _estimate_series_day_from_files(vc, date_str, raw_dir)
            results.append({
                "venue_code": vc,
                "date": date_str,
                "series_day": sd,
                "is_final_day_num": int(is_final),
            })

    return pd.DataFrame(results)


def _estimate_series_day_from_files(venue_code: str, date_str: str, raw_dir: Path = None) -> tuple:
    """raw_dirのBファイル履歴から開催シリーズ日数を推定する"""
    from datetime import datetime, timedelta

    if raw_dir is None:
        raw_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "B"

    try:
        today = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return 1, False

    series_day = 1
    for days_back in range(1, 8):
        prev = today - timedelta(days=days_back)
        b_file = raw_dir / f"b{prev.strftime('%y%m%d')}.txt"
        if not b_file.exists():
            break
        try:
            with open(b_file, encoding="cp932", errors="replace") as f:
                if f"{venue_code}BBGN" in f.read():
                    series_day += 1
                else:
                    break
        except Exception:
            break

    # 翌日のBファイルに同会場がなければ最終日
    next_date = today + timedelta(days=1)
    next_file = raw_dir / f"b{next_date.strftime('%y%m%d')}.txt"
    is_final = False
    if next_file.exists():
        try:
            with open(next_file, encoding="cp932", errors="replace") as f:
                is_final = f"{venue_code}BBGN" not in f.read()
        except Exception:
            pass
    elif series_day >= 3:
        is_final = True  # 翌日ファイルなし+3日以上経過は最終日とみなす

    return series_day, is_final


# 着順ポイント（ボートレース公式）
_RANK_POINTS = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2, 6: 1}
# グレード昇降級ボーダー（勝率スケール 0-10）
_GRADE_BORDERS = [3.30, 4.30, 5.50]  # B2/B1境界, B1/A2境界, A2/A1境界


def _get_evaluation_period(date_str: str) -> tuple[str, str]:
    """評価期間の開始・終了日を返す（上期:5-10月 / 下期:11-4月）"""
    from datetime import datetime as _dt
    d = _dt.strptime(date_str, "%Y-%m-%d")
    if 5 <= d.month <= 10:
        return f"{d.year}-05-01", f"{d.year}-10-31"
    elif d.month <= 4:
        return f"{d.year - 1}-11-01", f"{d.year}-04-30"
    else:
        return f"{d.year}-11-01", f"{d.year + 1}-04-30"


def compute_border_lookup(df_rank: pd.DataFrame, race_date: str) -> dict:
    """
    評価期間内の期別勝率を計算し、昇降級ボーダー接近度からモチベーション係数を返す

    Returns:
        {racer_no: motivation_factor} の辞書
        1.0=中立, >1.0=ボーダー接近で意欲↑ (最大1.05)
    """
    if df_rank.empty or "racer_no" not in df_rank.columns or "rank" not in df_rank.columns:
        return {}

    period_start, period_end = _get_evaluation_period(race_date)
    period_df = df_rank[
        (df_rank["date"] >= period_start) & (df_rank["date"] <= period_end)
    ]
    if period_df.empty:
        return {}

    lookup = {}
    for racer_no, grp in period_df.groupby("racer_no"):
        valid = grp[grp["rank"].between(1, 6)]
        if len(valid) < 3:
            continue
        period_rate = valid["rank"].map(_RANK_POINTS).sum() / len(valid)
        min_dist = min(abs(period_rate - b) for b in _GRADE_BORDERS)
        # ボーダー0.5pt以内は意欲↑（近いほど最大1.05）
        motivation = 1.0 + max(0.0, (0.5 - min_dist) * 0.10) if min_dist <= 0.5 else 1.0
        lookup[int(racer_no)] = motivation
    return lookup


def compute_recent_form_lookup(df_rank: pd.DataFrame, lookback: int = 10) -> dict:
    """
    選手ごとの直近調子スコアを計算する

    Returns:
        {racer_no: form_score} の辞書
        form_score: 0-1 (rank1=1.0, rank6≈0.167, 平均≈0.583)
    """
    if df_rank.empty or "racer_no" not in df_rank.columns or "rank" not in df_rank.columns:
        return {}

    lookup = {}
    df_sorted = df_rank.sort_values("date")
    for racer_no, grp in df_sorted.groupby("racer_no"):
        recent = grp.tail(lookback)
        valid = recent[recent["rank"].between(1, 6)]
        if valid.empty:
            continue
        scores = (7 - valid["rank"]) / 6  # rank1→1.0, rank6→0.167
        lookup[int(racer_no)] = float(scores.mean())
    return lookup


def build_features(
    df_program: pd.DataFrame,
    df_rank: pd.DataFrame,
    df_payout: pd.DataFrame,
    df_beforeinfo: pd.DataFrame = None,
    recent_form_lookup: dict = None,
    border_lookup: dict = None,
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
    program_pivot = _pivot_program(df_program, recent_form_lookup=recent_form_lookup,
                                   border_lookup=border_lookup)

    # シリーズ日（初日/最終日）を計算して結合
    series_df = _calculate_series_day(df_program)
    if not series_df.empty:
        program_pivot = program_pivot.merge(
            series_df[["venue_code", "date", "series_day", "is_final_day_num"]],
            on=["venue_code", "date"], how="left"
        )
        program_pivot["series_day"] = program_pivot["series_day"].fillna(1).astype(int)
        program_pivot["is_final_day_num"] = program_pivot["is_final_day_num"].fillna(0).astype(int)
    else:
        program_pivot["series_day"] = 1
        program_pivot["is_final_day_num"] = 0

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


def _pivot_program(df_program: pd.DataFrame, recent_form_lookup: dict = None,
                   border_lookup: dict = None) -> pd.DataFrame:
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
    meet_grade_map = {"SG": 5, "G1": 4, "G2": 3, "G3": 2, "一般": 1}
    df = df_program.copy()
    df["grade_num"] = df["grade"].map(grade_map).fillna(0)

    all_feature_cols = feature_cols + ["grade_num"]
    rows = []

    for (date, venue_code, race_no), group in df.groupby(["date", "venue_code", "race_no"]):
        mg = group["meet_grade"].iloc[0] if "meet_grade" in group.columns else "一般"
        row = {
            "date": date, "venue_code": venue_code, "race_no": race_no,
            "venue_name": group["venue_name"].iloc[0],
            "meet_grade": mg,
            "meet_grade_num": meet_grade_map.get(str(mg), 1),
        }
        for _, racer in group.iterrows():
            bn = int(racer["boat_no"])
            for col in all_feature_cols:
                row[f"boat{bn}_{col}"] = racer[col]
            # 全国3連率は番組データに含まれないため2連率から近似計算
            n2 = float(racer.get("national_2rate", 0.46) or 0.46)
            row[f"boat{bn}_national_3rate"] = round(n2 * 1.30, 2)
            # 直近調子スコア（過去N走の着順から計算）
            racer_no = int(racer.get("racer_no", 0) or 0)
            row[f"boat{bn}_racer_no"] = racer_no
            row[f"boat{bn}_recent_form_score"] = (
                recent_form_lookup.get(racer_no, 0.583)
                if recent_form_lookup else 0.583
            )
            # 持ちpt・昇降級ボーダー接近度によるモチベーション係数
            row[f"boat{bn}_motivation_factor"] = (
                border_lookup.get(racer_no, 1.0)
                if border_lookup else 1.0
            )
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
        "grade_num",
        "exhibition_time", "exhibition_st",
    ]
    cols = []
    for bn in range(1, 7):
        for col in base_cols:
            cols.append(f"boat{bn}_{col}")
    # レース全体の特徴量（シリーズ日程・開催グレード）
    cols += ["series_day", "is_final_day_num", "meet_grade_num"]
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
