"""
自動予想スケジューラー
発走10分前に各レースの予想を生成してスプレッドシートに書き込む
発走後に結果を取得して成績を記録する
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# 発走何分前に予想を実行するか
PREDICT_BEFORE_MIN = 10
# 発走後何分後に結果を取得するか
RESULT_AFTER_MIN = 12
# 結果取得の最大リトライ回数（超えたら諦めてスキップ）
MAX_RESULT_RETRIES = 10  # 12分後から2分おき×10回 = 最大32分後まで試行
# ループの確認間隔（秒）
LOOP_INTERVAL_SEC = 30


def _run_all_at_once(df_program, model, payout_lookup, today, spreadsheet_id, credentials_path,
                     recent_form_lookup=None, border_lookup=None):
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
    df_features_tmp = build_features(df_program, pd.DataFrame(), pd.DataFrame(),
                                     recent_form_lookup=recent_form_lookup,
                                     border_lookup=border_lookup)
    beforeinfo_raw = fetch_beforeinfo_for_races(df_features_tmp, today)
    records = []
    for (vc, rn), boats in beforeinfo_raw.items():
        for bn, info in boats.items():
            records.append({
                "date": today.strftime("%Y-%m-%d"), "venue_code": vc,
                "race_no": rn, "boat_no": bn,
                "exhibition_time": info.get("exhibition_time"),
                "exhibition_st": info.get("exhibition_st"),
                "actual_course": info.get("actual_course", bn),
                "meet_ranks": info.get("meet_ranks", []),
                "meet_sts": info.get("meet_sts", []),
            })
    df_beforeinfo = pd.DataFrame(records) if records else pd.DataFrame()

    df_features = build_features(df_program, pd.DataFrame(), pd.DataFrame(),
                                 df_beforeinfo if not df_beforeinfo.empty else None,
                                 recent_form_lookup=recent_form_lookup,
                                 border_lookup=border_lookup)

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
            "last_result_attempt": None,
            "result_fetch_attempts": 0,
        })

    # 発走時刻順にソート
    schedule.sort(key=lambda x: x["scheduled_dt"])
    return schedule


def run_auto(spreadsheet_id: str, credentials_path: str = None) -> None:
    """
    自動予想ループのメイン関数
    Ctrl+C で停止する
    """
    from src.collector.downloader import download_file, extract_lzh, download_range
    from src.collector.parser import parse_program
    from src.collector.beforeinfo import fetch_beforeinfo_for_races
    from src.collector.odds import fetch_odds_for_races
    from src.collector.result_scraper import fetch_race_result
    from src.features.builder import build_features
    from src.model.trainer import load_model
    from src.model.predictor import get_recommendations, load_payout_lookup
    from web.database import save_prediction, sync_race_predictions, update_result as db_update_result

    # Sheets書き込みは無効化（DBとブラウザで管理するため不要）
    _sheets_ok = False
    if _sheets_ok:
        from src.output.sheets import (
            append_prediction_row, update_result_row, update_summary_sheet,
            apply_colors_to_results_sheet,
        )
    else:
        append_prediction_row = update_result_row = update_summary_sheet = apply_colors_to_results_sheet = None

    today = datetime.now()
    print(f"=== 自動予想モード開始: {today.strftime('%Y-%m-%d')} ===")
    print(f"  発走{PREDICT_BEFORE_MIN}分前に予想、発走{RESULT_AFTER_MIN}分後に結果取得")
    print("  停止するには Ctrl+C を押してください\n")

    # 直近K（競走成績）ファイルをダウンロードして選手の直近調子を計算
    from src.collector.downloader import extract_all as _extract_all_k
    from src.collector.parser import parse_all_results
    from src.features.builder import compute_recent_form_lookup, compute_border_lookup
    k_start = today - timedelta(days=60)  # ボーダー判定には評価期間全体が必要
    k_end = today - timedelta(days=1)
    print("  直近成績データ取得中（既取得分はスキップ）...")
    download_range("K", k_start, k_end, interval=1.0)
    _extract_all_k("K")
    k_raw_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "K"
    df_rank_hist, _ = parse_all_results(k_raw_dir) if k_raw_dir.exists() else (pd.DataFrame(), pd.DataFrame())
    if not df_rank_hist.empty:
        recent_form_lookup = compute_recent_form_lookup(df_rank_hist)
        border_lookup = compute_border_lookup(df_rank_hist, today.strftime("%Y-%m-%d"))
    else:
        recent_form_lookup = {}
        border_lookup = {}
    print(f"  直近調子: {len(recent_form_lookup)}選手 / ボーダー判定: {len(border_lookup)}選手\n")

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
            recent_form_lookup=recent_form_lookup,
            border_lookup=border_lookup,
        )
        return

    # 起動時点で既に発走済みのレースはスキップ
    now_start = datetime.now()
    for race in schedule:
        if race["scheduled_dt"] <= now_start:
            race["predicted"] = True
            race["result_fetched"] = True

    # ── 再起動補完: 予想済みで結果未取得のレースを一括チェック ──
    _catchup_missing_results(
        today, df_program, db_update_result, fetch_race_result
    )

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
            try:
                now = datetime.now()
                all_done = all(r["predicted"] and r["result_fetched"] for r in schedule)
                if all_done:
                    print("\n=== 本日の全レース処理完了 ===")
                    if _sheets_ok:
                        apply_colors_to_results_sheet(spreadsheet_id, credentials_path)
                        update_summary_sheet(spreadsheet_id, credentials_path)
                    break

                for race in schedule:
                    scheduled_dt = race["scheduled_dt"]

                    # ── 予想タイミング: 発走10分前 ──
                    predict_at = scheduled_dt - timedelta(minutes=PREDICT_BEFORE_MIN)
                    if not race["predicted"] and now >= predict_at:
                        daily_race_count += 1
                        # 全会場・本日の通し番号: 記録済みレース数ではなく本日の全番組表
                        # (schedule)から算出するため、途中再起動・遅延起動でも
                        # 正しい「当日何レース目か」を維持できる
                        day_race_no = sum(1 for r in schedule if r["scheduled_dt"] <= scheduled_dt)
                        pred_rows = _predict_one_race(
                            race, df_program, model, payout_lookup,
                            today, spreadsheet_id, credentials_path,
                            fetch_beforeinfo_for_races, fetch_odds_for_races,
                            build_features, get_recommendations,
                            append_prediction_row if _sheets_ok else None,
                            daily_race_count=daily_race_count,
                            recent_form_lookup=recent_form_lookup,
                            border_lookup=border_lookup,
                            db_save_fn=save_prediction,
                            db_sync_fn=sync_race_predictions,
                            day_race_no=day_race_no,
                        )
                        race["pred_rows"] = pred_rows or []
                        race["daily_race_count"] = daily_race_count
                        race["predicted"] = True

                    # ── 結果取得タイミング: 発走12分後（2分間隔でリトライ、最大10回）──
                    result_at = scheduled_dt + timedelta(minutes=RESULT_AFTER_MIN)
                    last_attempt = race.get("last_result_attempt")
                    retry_ok = (last_attempt is None or
                                now >= last_attempt + timedelta(minutes=2))
                    attempts = race.get("result_fetch_attempts", 0)
                    if race["predicted"] and not race["result_fetched"] and now >= result_at and retry_ok:
                        if attempts >= MAX_RESULT_RETRIES:
                            print(f"\n  [SKIP] {race['venue_name']} {race['race_no']}R: "
                                  f"結果取得{MAX_RESULT_RETRIES}回失敗のためスキップ")
                            race["result_fetched"] = True
                        else:
                            race["last_result_attempt"] = now
                            race["result_fetch_attempts"] = attempts + 1
                            success = _fetch_and_record_result(
                                race, today, spreadsheet_id, credentials_path,
                                fetch_race_result,
                                update_result_row if _sheets_ok else None,
                                pred_rows_override=race.get("pred_rows", []),
                                race_count=race.get("daily_race_count"),
                                db_update_fn=db_update_result,
                            )
                            if success:
                                race["result_fetched"] = True
                                if _sheets_ok:
                                    update_summary_sheet(spreadsheet_id, credentials_path)

                # 次の予想・結果取得までの待機時間を表示
                next_action = _next_action_time(schedule, now)
                if next_action:
                    wait_sec = max(0, (next_action - now).total_seconds())
                    print(f"  [{now.strftime('%H:%M:%S')}] 次のアクションまで {int(wait_sec//60)}分{int(wait_sec%60)}秒待機...",
                          end="\r")

                time.sleep(LOOP_INTERVAL_SEC)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\n  [WARN] ループエラー: {e} → 30秒後に再試行")
                time.sleep(30)

    except KeyboardInterrupt:
        print("\n\n自動予想を停止しました")
        if _sheets_ok:
            apply_colors_to_results_sheet(spreadsheet_id, credentials_path)
            update_summary_sheet(spreadsheet_id, credentials_path)


def _catchup_missing_results(today, df_program, db_update_fn, fetch_race_result_fn):
    """
    起動時補完: 今日の予想済みレースで結果未取得のものを一括チェックして記録する。
    発走時刻を過ぎているレースだけを対象にする。
    """
    from web.database import get_conn

    date_str = today.strftime("%Y-%m-%d")
    now = datetime.now()

    # venue_name → venue_code マッピング
    vc_map = {}
    for _, row in df_program.iterrows():
        vn = str(row.get("venue_name", ""))
        vc = str(row.get("venue_code", "")).zfill(2)
        if vn and vc:
            vc_map[vn] = vc

    # 結果未取得の予想レースを取得
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT DISTINCT venue_name, race_no,
                   MIN(race_time) as race_time,
                   MIN(created_at) as created_at
            FROM predictions
            WHERE date=? AND result_recorded_at IS NULL
            GROUP BY venue_name, race_no
        """, (date_str,)).fetchall()

    if not rows:
        return

    targets = []
    for row in rows:
        venue_name = row["venue_name"]
        race_no    = row["race_no"]
        race_time_str = row["race_time"] or ""
        created_at    = row["created_at"] or ""

        # 発走済み判定: race_timeがあればそれ基準、なければcreated_at+20分
        should_fetch = False
        if race_time_str:
            try:
                hh, mm = map(int, race_time_str.split(":"))
                scheduled_dt = datetime(now.year, now.month, now.day, hh, mm)
                if now >= scheduled_dt + timedelta(minutes=RESULT_AFTER_MIN):
                    should_fetch = True
            except Exception:
                should_fetch = True
        elif created_at:
            try:
                ct = datetime.strptime(created_at[:19], "%Y-%m-%d %H:%M:%S")
                if now >= ct + timedelta(minutes=20):
                    should_fetch = True
            except Exception:
                should_fetch = True

        if should_fetch:
            targets.append((venue_name, race_no))

    if not targets:
        return

    print(f"\n[補完] 結果未取得レース {len(targets)}件 → 取得開始")
    for venue_name, race_no in targets:
        venue_code = vc_map.get(venue_name, "")
        if not venue_code:
            print(f"  [SKIP] {venue_name} {race_no}R: 会場コード不明")
            continue

        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {venue_name} {race_no}R 結果取得中...")
        try:
            result = fetch_race_result_fn(today, venue_code, race_no)
            if result.get("available"):
                combo  = result["combination"]
                payout = result["payout"]
                print(f"  → {combo} ¥{payout:,}")
                if db_update_fn:
                    db_update_fn(date_str, venue_name, race_no, combo, payout)
            else:
                print(f"  → 結果未確定（スキップ）")
        except Exception as e:
            print(f"  [WARN] {venue_name} {race_no}R 結果取得失敗: {e}")
        time.sleep(1.0)  # 連続アクセス防止

    print("[補完] 完了\n")


def _predict_one_race(
    race: dict, df_program, model, payout_lookup,
    today, spreadsheet_id, credentials_path,
    fetch_beforeinfo_for_races, fetch_odds_for_races,
    build_features, get_recommendations, append_prediction_row,
    daily_race_count: int = None,
    recent_form_lookup: dict = None,
    border_lookup: dict = None,
    db_save_fn=None,
    db_sync_fn=None,
    day_race_no: int = None,
) -> list:
    """1レース分の予想を実行してDB(+オプションでSheets)に書き込む。予想行リストを返す"""
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

    # 展示タイム・天候・欠場艇取得
    from src.collector.beforeinfo import fetch_beforeinfo
    beforeinfo_raw = fetch_beforeinfo(today, venue_code, race_no)
    race_weather = beforeinfo_raw.pop("weather", None) if isinstance(beforeinfo_raw, dict) else None
    absent_boats = beforeinfo_raw.pop("absent_boats", []) if isinstance(beforeinfo_raw, dict) else []
    df_beforeinfo = pd.DataFrame([
        {"date": today.strftime("%Y-%m-%d"), "venue_code": venue_code,
         "race_no": race_no, "boat_no": bn,
         "exhibition_time": info.get("exhibition_time"),
         "exhibition_st": info.get("exhibition_st"),
         "actual_course": info.get("actual_course", bn),
         "meet_ranks": info.get("meet_ranks", []),
         "meet_sts": info.get("meet_sts", [])}
        for bn, info in beforeinfo_raw.items()
        if isinstance(bn, int)
    ]) if beforeinfo_raw else pd.DataFrame()
    if race_weather and race_weather.get("wind_speed"):
        w = race_weather
        print(f"  天候: {w.get('weather','不明')} 風速:{w.get('wind_speed')}m 波高:{w.get('wave_height')}cm")
    if absent_boats:
        print(f"  欠場艇: {absent_boats} → 残り{6 - len(absent_boats)}艇で予想")

    # 特徴量生成
    df_features = build_features(df_prog_race, pd.DataFrame(), pd.DataFrame(),
                                 df_beforeinfo if not df_beforeinfo.empty else None,
                                 recent_form_lookup=recent_form_lookup,
                                 border_lookup=border_lookup)
    if df_features.empty:
        print(f"  [WARN] 特徴量生成失敗: {venue_name} {race_no}R")
        return

    # リアルタイムオッズ取得（3連単）
    from src.collector.odds import fetch_odds
    live_odds = fetch_odds(today, venue_code, race_no)
    all_live_odds = {(venue_code, race_no): live_odds} if live_odds else {}
    all_weather = {(venue_code, race_no): race_weather} if race_weather else {}
    all_absent = {(venue_code, race_no): absent_boats} if absent_boats else {}

    # 予想生成
    recs = get_recommendations(model, df_features, payout_lookup=payout_lookup,
                               all_live_odds=all_live_odds, all_weather=all_weather,
                               all_absent=all_absent)

    # 再計算で選出組み合わせが変わった場合、未確定の古い買い目を削除してから保存する
    if db_sync_fn and not recs.empty:
        try:
            db_sync_fn(today.strftime("%Y-%m-%d"), venue_name, race_no, recs["combination"].tolist())
        except Exception as _e:
            print(f"  [WARN] DB同期失敗: {_e}")

    # DB + Sheetsに書き込む＆メモリキャッシュ用にリストを作成
    pred_rows = []
    date_str = today.strftime("%Y-%m-%d")
    race_time_str = race.get("scheduled_time_str", "")
    has_bet = False
    for _, rec in recs.iterrows():
        row_dict = rec.to_dict()
        row_dict["race_time"] = race_time_str
        row_dict["day_race_no"] = day_race_no
        # DBに保存（常時）
        if db_save_fn:
            try:
                db_save_fn(row_dict)
            except Exception as _e:
                print(f"  [WARN] DB保存失敗: {_e}")
        # Sheetsに書き込み（認証情報ある場合のみ）
        if append_prediction_row:
            append_prediction_row(spreadsheet_id, row_dict, credentials_path=credentials_path,
                                  race_count=daily_race_count)
        label = rec.get("bet_label", "")
        tier  = rec.get("tier", "")
        arare = rec.get("arare_reasons", "")
        if label not in ("見送り", ""):
            has_bet = True
            print(f"  → [{label}/{tier}] {rec['combination']} オッズ:{rec['odds']} PT:{rec.get('arare_score','')} 荒れ:{arare}")
        else:
            print(f"  → [見送り/{tier}] {rec['combination']} オッズ:{rec['odds']} PT:{rec.get('arare_score','')}")
        # 全行（見送り含む）を成績記録対象に追加
        pred_rows.append({
            "日付": date_str,
            "競艇場": venue_name,
            "レース": str(race_no),
            "狙い": tier if tier else "-",
            "買い目（3連単）": rec.get("combination", ""),
            "的中確率": rec.get("prob", "-"),
            "期待回収率": rec.get("expected_roi", "-"),
            "信頼度": rec.get("confidence", "-"),
            "イン逃げ率": rec.get("nigerate_str", "-"),
            "勝負推奨": label,
            "荒れPT": rec.get("arare_score", ""),
        })
    if not has_bet:
        print(f"  → 見送り（参考予想のみ）")

    return pred_rows


def _fetch_and_record_result(
    race: dict, today, spreadsheet_id, credentials_path,
    fetch_race_result, update_result_row,
    pred_rows_override: list = None,
    race_count: int = None,
    db_update_fn=None,
) -> bool:
    """レース結果を取得してDB(+オプションでSheets)に記録する。成功したらTrueを返す。"""
    venue_code = race["venue_code"]
    race_no = race["race_no"]
    venue_name = race["venue_name"]
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 結果取得: {venue_name} {race_no}R")

    result = fetch_race_result(today, venue_code, race_no)
    if not result.get("available"):
        print(f"  [WARN] 結果未確定 → {LOOP_INTERVAL_SEC}秒後に再試行します")
        return False

    date_str = today.strftime("%Y-%m-%d")
    combo = result["combination"]
    payout = result["payout"]
    print(f"  結果: {combo} 払戻:¥{payout:,}")

    # DB更新（常時）
    if db_update_fn:
        try:
            db_update_fn(date_str, venue_name, race_no, combo, payout)
        except Exception as _e:
            print(f"  [WARN] DB結果更新失敗: {_e}")

    # Sheets更新（認証情報ある場合のみ）
    if update_result_row:
        update_result_row(
            spreadsheet_id,
            date=date_str,
            venue_name=venue_name,
            race_no=race_no,
            actual_combination=combo,
            actual_payout=payout,
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


if __name__ == "__main__":
    _spreadsheet_id = os.environ.get("SPREADSHEET_ID", "")
    _credentials_path = os.environ.get("GOOGLE_CREDENTIALS_PATH", "")

    if not _spreadsheet_id:
        print("[ERROR] 環境変数 SPREADSHEET_ID が未設定です")
        print("  例: set SPREADSHEET_ID=your_spreadsheet_id_here")
        raise SystemExit(1)
    if not _credentials_path:
        print("[ERROR] 環境変数 GOOGLE_CREDENTIALS_PATH が未設定です")
        print("  例: set GOOGLE_CREDENTIALS_PATH=C:\\path\\to\\credentials.json")
        raise SystemExit(1)

    run_auto(_spreadsheet_id, _credentials_path)
