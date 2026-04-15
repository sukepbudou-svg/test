"""
自動予想スケジューラー
発走10分前に各レースの予想を生成してスプレッドシートに書き込む
発走後に結果を取得して成績を記録する
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path


# 発走何分前に予想を実行するか
PREDICT_BEFORE_MIN = 10
# 発走後何分後に結果を取得するか
RESULT_AFTER_MIN = 8
# ループの確認間隔（秒）
LOOP_INTERVAL_SEC = 30


def _run_all_at_once(df_program, model, payout_lookup, today, spreadsheet_id, credentials_path):
    """
    発走時刻不明時のフォールバック: 全レースをまとめて予想してスプレッドシートに書き込む
    結果取得は翌日以降の手動バックテストで対応
    """
    import pandas as pd
    from src.collector.beforeinfo import fetch_beforeinfo_for_races
    from src.collector.odds import fetch_odds_for_races
    from src.features.builder import build_features
    from src.model.predictor import get_recommendations
    from src.output.sheets import write_predictions

    print("=== 全レース一括予想モード ===")

    # 展示タイム取得
    df_features_tmp = build_features(df_program, pd.DataFrame(), pd.DataFrame())
    beforeinfo_raw = fetch_beforeinfo_for_races(df_features_tmp, today)
    records = []
    for (vc, rn), boats in beforeinfo_raw.items():
        for bn, info in boats.items():
            records.append({
                "date": today.strftime("%Y-%m-%d"), "venue_code": vc,
                "race_no": rn, "boat_no": bn,
                "exhibition_time": info.get("exhibition_time"),
                "exhibition_st": info.get("exhibition_st"),
            })
    df_beforeinfo = pd.DataFrame(records) if records else pd.DataFrame()

    df_features = build_features(df_program, pd.DataFrame(), pd.DataFrame(),
                                 df_beforeinfo if not df_beforeinfo.empty else None)

    # リアルタイムオッズ取得
    all_live_odds = fetch_odds_for_races(df_features, today)

    # 予想生成
    recs = get_recommendations(model, df_features, payout_lookup=payout_lookup,
                               all_live_odds=all_live_odds)

    print("\n【本日の推奨買い目】")
    print(recs[recs["combination"] != "見送り"].to_string(index=False))

    # スプレッドシートに書き込み
    write_predictions(spreadsheet_id, recs, credentials_path=credentials_path)


def build_race_schedule(df_program) -> list[dict]:
    """
    番組表から全レースのスケジュールを生成する

    Returns:
        [{"venue_code", "venue_name", "race_no", "scheduled_time_str", "scheduled_dt"}, ...]
        scheduled_time が取得できないレースは除外
    """
    today = datetime.now().date()
    schedule = []

    if "scheduled_time" not in df_program.columns:
        return schedule

    seen = set()
    for _, row in df_program.iterrows():
        t_str = row.get("scheduled_time")
        venue_code = str(row.get("venue_code", "")).zfill(2)
        race_no = int(row.get("race_no", 0))
        key = (venue_code, race_no)

        if key in seen or not t_str:
            continue
        seen.add(key)

        try:
            hh, mm = map(int, t_str.split(":"))
            scheduled_dt = datetime(today.year, today.month, today.day, hh, mm)
        except (ValueError, AttributeError):
            continue

        schedule.append({
            "venue_code": venue_code,
            "venue_name": row.get("venue_name", ""),
            "race_no": race_no,
            "scheduled_time_str": t_str,
            "scheduled_dt": scheduled_dt,
            "predicted": False,
            "result_fetched": False,
        })

    # 発走時刻順にソート
    schedule.sort(key=lambda x: x["scheduled_dt"])
    return schedule


def run_auto(spreadsheet_id: str, credentials_path: str = None) -> None:
    """
    自動予想ループのメイン関数
    Ctrl+C で停止する
    """
    from src.collector.downloader import download_file, extract_lzh
    from src.collector.parser import parse_program
    from src.collector.beforeinfo import fetch_beforeinfo_for_races
    from src.collector.odds import fetch_odds_for_races
    from src.collector.result_scraper import fetch_race_result
    from src.features.builder import build_features
    from src.model.trainer import load_model
    from src.model.predictor import get_recommendations, load_payout_lookup
    from src.output.sheets import (
        append_prediction_row, update_result_row, update_summary_sheet,
        apply_colors_to_results_sheet,
    )

    today = datetime.now()
    print(f"=== 自動予想モード開始: {today.strftime('%Y-%m-%d')} ===")
    print(f"  発走{PREDICT_BEFORE_MIN}分前に予想、発走{RESULT_AFTER_MIN}分後に結果取得")
    print("  停止するには Ctrl+C を押してください\n")

    # 番組表ダウンロード
    b_path = download_file("B", today)
    if not b_path:
        print("[ERROR] 番組表のダウンロードに失敗しました")
        return
    txt_path = extract_lzh(b_path)
    if not txt_path:
        print("[ERROR] 番組表の解凍に失敗しました")
        return
    df_program = parse_program(txt_path)
    if df_program.empty:
        print("[ERROR] 番組表のパースに失敗しました")
        return

    # モデル読み込み
    model = load_model()
    if not model:
        print("[ERROR] モデルが見つかりません。先に --mode train を実行してください")
        return
    payout_lookup = load_payout_lookup()

    # レーススケジュール作成
    schedule = build_race_schedule(df_program)
    if not schedule:
        print("[WARN] 発走時刻が番組表から取得できませんでした")
        print("       → 全レースを今すぐ予想します（時刻管理なし）")
        _run_all_at_once(
            df_program, model, payout_lookup, today,
            spreadsheet_id, credentials_path,
        )
        return

    # 起動時点で既に発走済みのレースはスキップ
    now_start = datetime.now()
    for race in schedule:
        if race["scheduled_dt"] <= now_start:
            race["predicted"] = True
            race["result_fetched"] = True

    upcoming = [r for r in schedule if not r["predicted"]]
    total = len(schedule)
    print(f"本日のレース数: {total}レース（残り{len(upcoming)}レース）")
    for r in upcoming:
        predict_at = r["scheduled_dt"] - timedelta(minutes=PREDICT_BEFORE_MIN)
        print(f"  {r['venue_name']} {r['race_no']}R - 発走:{r['scheduled_time_str']} / 予想:{predict_at.strftime('%H:%M')}")
    if not upcoming:
        print("  本日の残りレースはありません")
    print()

    daily_race_count = 0  # 本日の予想レースカウンター

    try:
        while True:
            now = datetime.now()
            all_done = all(r["predicted"] and r["result_fetched"] for r in schedule)
            if all_done:
                print("\n=== 本日の全レース処理完了 ===")
                apply_colors_to_results_sheet(spreadsheet_id, credentials_path)
                update_summary_sheet(spreadsheet_id, credentials_path)
                break

            for race in schedule:
                scheduled_dt = race["scheduled_dt"]

                # ── 予想タイミング: 発走10分前 ──
                predict_at = scheduled_dt - timedelta(minutes=PREDICT_BEFORE_MIN)
                if not race["predicted"] and now >= predict_at:
                    daily_race_count += 1
                    pred_rows = _predict_one_race(
                        race, df_program, model, payout_lookup,
                        today, spreadsheet_id, credentials_path,
                        fetch_beforeinfo_for_races, fetch_odds_for_races,
                        build_features, get_recommendations, append_prediction_row,
                        daily_race_count=daily_race_count,
                    )
                    race["pred_rows"] = pred_rows or []
                    race["daily_race_count"] = daily_race_count
                    race["predicted"] = True

                # ── 結果取得タイミング: 発走8分後 ──
                result_at = scheduled_dt + timedelta(minutes=RESULT_AFTER_MIN)
                if race["predicted"] and not race["result_fetched"] and now >= result_at:
                    success = _fetch_and_record_result(
                        race, today, spreadsheet_id, credentials_path,
                        fetch_race_result, update_result_row,
                        pred_rows_override=race.get("pred_rows", []),
                        race_count=race.get("daily_race_count"),
                    )
                    if success:
                        race["result_fetched"] = True
                        update_summary_sheet(spreadsheet_id, credentials_path)

            # 次の予想・結果取得までの待機時間を表示
            next_action = _next_action_time(schedule, now)
            if next_action:
                wait_sec = max(0, (next_action - now).total_seconds())
                print(f"  [{now.strftime('%H:%M:%S')}] 次のアクションまで {int(wait_sec//60)}分{int(wait_sec%60)}秒待機...",
                      end="\r")

            time.sleep(LOOP_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n自動予想を停止しました")
        apply_colors_to_results_sheet(spreadsheet_id, credentials_path)
        update_summary_sheet(spreadsheet_id, credentials_path)


def _predict_one_race(
    race: dict, df_program, model, payout_lookup,
    today, spreadsheet_id, credentials_path,
    fetch_beforeinfo_for_races, fetch_odds_for_races,
    build_features, get_recommendations, append_prediction_row,
    daily_race_count: int = None,
) -> list:
    """1レース分の予想を実行してスプレッドシートに書き込む。予想行リストを返す（メモリキャッシュ用）"""
    import pandas as pd

    venue_code = race["venue_code"]
    race_no = race["race_no"]
    venue_name = race["venue_name"]
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 予想開始: {venue_name} {race_no}R")

    # このレースのみの番組表を絞り込む
    df_prog_race = df_program[
        (df_program["venue_code"] == venue_code) &
        (df_program["race_no"] == race_no)
    ]

    # 展示タイム取得
    from src.collector.beforeinfo import fetch_beforeinfo
    beforeinfo_raw = fetch_beforeinfo(today, venue_code, race_no)
    df_beforeinfo = pd.DataFrame([
        {"date": today.strftime("%Y-%m-%d"), "venue_code": venue_code,
         "race_no": race_no, "boat_no": bn,
         "exhibition_time": info.get("exhibition_time"),
         "exhibition_st": info.get("exhibition_st")}
        for bn, info in beforeinfo_raw.items()
    ]) if beforeinfo_raw else pd.DataFrame()

    # 特徴量生成
    df_features = build_features(df_prog_race, pd.DataFrame(), pd.DataFrame(),
                                 df_beforeinfo if not df_beforeinfo.empty else None)
    if df_features.empty:
        print(f"  [WARN] 特徴量生成失敗: {venue_name} {race_no}R")
        return

    # リアルタイムオッズ取得
    from src.collector.odds import fetch_odds
    live_odds = fetch_odds(today, venue_code, race_no)
    all_live_odds = {(venue_code, race_no): live_odds} if live_odds else {}

    # 予想生成
    recs = get_recommendations(model, df_features, payout_lookup=payout_lookup,
                               all_live_odds=all_live_odds)

    # スプレッドシートに書き込む＆メモリキャッシュ用にリストを作成
    pred_rows = []
    date_str = today.strftime("%Y-%m-%d")
    for _, rec in recs.iterrows():
        if rec.get("combination") in ("見送り", "", "-"):
            continue
        row_dict = rec.to_dict()
        append_prediction_row(spreadsheet_id, row_dict, credentials_path=credentials_path,
                              race_count=daily_race_count)
        print(f"  → {rec['combination']} 確率:{rec['prob']} 期待回収率:{rec['expected_roi']}")
        # 成績2シートの列名に合わせてキャッシュ用dictを作成
        pred_rows.append({
            "日付": date_str,
            "競艇場": venue_name,
            "レース": str(race_no),
            "買い目（3連単）": rec.get("combination", ""),
            "的中確率": rec.get("prob", "-"),
            "期待回収率": rec.get("expected_roi", "-"),
        })

    if recs[recs["combination"] != "見送り"].empty:
        print(f"  → 見送り（期待回収率が基準未満）")

    return pred_rows


def _fetch_and_record_result(
    race: dict, today, spreadsheet_id, credentials_path,
    fetch_race_result, update_result_row,
    pred_rows_override: list = None,
    race_count: int = None,
) -> bool:
    """レース結果を取得して成績2シートに記録する。成功したらTrueを返す。"""
    venue_code = race["venue_code"]
    race_no = race["race_no"]
    venue_name = race["venue_name"]
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 結果取得: {venue_name} {race_no}R")

    result = fetch_race_result(today, venue_code, race_no)
    if not result.get("available"):
        print(f"  [WARN] 結果未確定 → 2分後に再試行します")
        return False

    update_result_row(
        spreadsheet_id,
        date=today.strftime("%Y-%m-%d"),
        venue_name=venue_name,
        race_no=race_no,
        actual_combination=result["combination"],
        actual_payout=result["payout"],
        credentials_path=credentials_path,
        pred_rows_override=pred_rows_override,
        race_count=race_count,
    )
    return True


def _next_action_time(schedule: list, now: datetime):
    """次の予想または結果取得の時刻を返す"""
    times = []
    for race in schedule:
        if not race["predicted"]:
            times.append(race["scheduled_dt"] - timedelta(minutes=PREDICT_BEFORE_MIN))
        elif not race["result_fetched"]:
            times.append(race["scheduled_dt"] + timedelta(minutes=RESULT_AFTER_MIN))
    future = [t for t in times if t > now]
    return min(future) if future else None
