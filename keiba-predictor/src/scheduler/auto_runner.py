"""
競馬自動予想スケジューラー
発走6分前に予想を生成してスプレッドシートに書き込む
発走後12分で結果を取得して成績を記録する
"""

import time
from datetime import datetime, timedelta

PREDICT_BEFORE_MIN = 6
RESULT_AFTER_MIN = 12
MAX_RESULT_RETRIES = 10
LOOP_INTERVAL_SEC = 30


def run_auto(spreadsheet_id: str, credentials_path: str = None) -> None:
    """自動予想ループのメイン関数（Ctrl+C で停止）"""
    from src.collector.scraper import fetch_today_schedule, fetch_race_card, fetch_odds_quinella, fetch_race_result
    from src.features.builder import build_race_features
    from src.model.trainer import load_model
    from src.model.predictor import get_recommendations
    from src.output.sheets import append_prediction_row, update_result_row, update_summary_sheet

    today = datetime.now()
    print(f"=== 競馬自動予想モード開始: {today.strftime('%Y-%m-%d')} ===")
    print(f"  発走{PREDICT_BEFORE_MIN}分前に予想、発走{RESULT_AFTER_MIN}分後に結果取得")
    print("  停止するには Ctrl+C を押してください\n")

    model = load_model()
    if not model:
        print("[ERROR] モデルが見つかりません。先に --mode train を実行してください")
        return

    # 本日のスケジュール取得
    raw_schedule = fetch_today_schedule(today)
    if not raw_schedule:
        print("[ERROR] 本日の開催情報が取得できませんでした")
        return

    # スケジュールにメタ情報を追加
    schedule = []
    today_date = today.date()
    for r in raw_schedule:
        t_str = r.get("scheduled_time")
        if not t_str:
            continue
        try:
            hh, mm = map(int, t_str.split(":"))
            scheduled_dt = datetime(today_date.year, today_date.month, today_date.day, hh, mm)
        except (ValueError, AttributeError):
            continue
        schedule.append({
            **r,
            "scheduled_dt": scheduled_dt,
            "predicted": False,
            "result_fetched": False,
            "last_result_attempt": None,
            "result_fetch_attempts": 0,
            "pred_rows": [],
        })

    schedule.sort(key=lambda x: x["scheduled_dt"])

    # 起動時点で発走済みはスキップ
    now_start = datetime.now()
    for race in schedule:
        if race["scheduled_dt"] <= now_start:
            race["predicted"] = True
            race["result_fetched"] = True

    upcoming = [r for r in schedule if not r["predicted"]]
    print(f"本日のレース数: {len(schedule)}レース（残り{len(upcoming)}レース）")
    for r in upcoming:
        predict_at = r["scheduled_dt"] - timedelta(minutes=PREDICT_BEFORE_MIN)
        print(f"  {r['venue']} {r['race_no']}R - 発走:{r.get('scheduled_time','')} / 予想:{predict_at.strftime('%H:%M')}")
    print()

    try:
        while True:
            now = datetime.now()
            all_done = all(r["predicted"] and r["result_fetched"] for r in schedule)
            if all_done:
                print("\n=== 本日の全レース処理完了 ===")
                update_summary_sheet(spreadsheet_id, credentials_path)
                break

            for race in schedule:
                scheduled_dt = race["scheduled_dt"]

                # 予想タイミング
                predict_at = scheduled_dt - timedelta(minutes=PREDICT_BEFORE_MIN)
                if not race["predicted"] and now >= predict_at:
                    pred_rows = _predict_one_race(
                        race, today, model,
                        get_recommendations, fetch_race_card,
                        fetch_odds_quinella, build_race_features,
                        append_prediction_row, spreadsheet_id, credentials_path,
                    )
                    race["pred_rows"] = pred_rows or []
                    race["predicted"] = True

                # 結果取得タイミング（2分おきに最大10回）
                result_at = scheduled_dt + timedelta(minutes=RESULT_AFTER_MIN)
                last_attempt = race.get("last_result_attempt")
                retry_ok = (last_attempt is None or
                            now >= last_attempt + timedelta(minutes=2))
                attempts = race.get("result_fetch_attempts", 0)

                if race["predicted"] and not race["result_fetched"] and now >= result_at and retry_ok:
                    if attempts >= MAX_RESULT_RETRIES:
                        print(f"\n  [SKIP] {race['venue']} {race['race_no']}R: "
                              f"結果取得{MAX_RESULT_RETRIES}回失敗のためスキップ")
                        race["result_fetched"] = True
                    else:
                        race["last_result_attempt"] = now
                        race["result_fetch_attempts"] = attempts + 1
                        success = _fetch_result(
                            race, today, spreadsheet_id, credentials_path,
                            fetch_race_result, update_result_row,
                        )
                        if success:
                            race["result_fetched"] = True
                            update_summary_sheet(spreadsheet_id, credentials_path)

            # 次アクションまでの待機表示
            next_action = _next_action_time(schedule, now)
            if next_action:
                wait_sec = max(0, (next_action - now).total_seconds())
                print(f"  [{now.strftime('%H:%M:%S')}] 次のアクションまで "
                      f"{int(wait_sec//60)}分{int(wait_sec%60)}秒待機...", end="\r")

            time.sleep(LOOP_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n自動予想を停止しました")
        update_summary_sheet(spreadsheet_id, credentials_path)


def _predict_one_race(
    race, today, model,
    get_recommendations, fetch_race_card, fetch_odds_quinella,
    build_race_features, append_prediction_row,
    spreadsheet_id, credentials_path,
) -> list:
    """1レース分の予想を実行してスプレッドシートに書き込む"""
    venue_code = race["venue_code"]
    race_no = race["race_no"]
    venue = race["venue"]
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 予想開始: {venue} {race_no}R")

    # スケジュールから取得した正しいrace_idを使う
    race_id = race.get("race_id")

    # 出馬表取得
    card = fetch_race_card(today, venue_code, race_no, race_id=race_id)
    if not card or not card.get("horses"):
        print(f"  [WARN] 出馬表取得失敗: {venue} {race_no}R")
        return []

    # 特徴量生成
    df_race = build_race_features(card)
    if df_race.empty:
        print(f"  [WARN] 特徴量生成失敗: {venue} {race_no}R")
        return []

    # リアルタイム馬連オッズ取得
    live_odds = fetch_odds_quinella(today, venue_code, race_no, race_id=race_id)

    # 予想生成
    recs = get_recommendations(model, df_race, live_odds)

    pred_rows = []
    date_str = today.strftime("%Y-%m-%d")
    for _, rec in recs.iterrows():
        if rec.get("combination") in ("見送り", "-", ""):
            continue
        row_dict = {**rec.to_dict(), "date": date_str, "venue": venue}
        append_prediction_row(spreadsheet_id, row_dict, credentials_path)
        print(f"  → {rec['combination']} 確率:{rec['prob']} 期待回収率:{rec['expected_roi']}")
        pred_rows.append({
            "馬連買い目": rec.get("combination", ""),
            "的中確率": rec.get("prob", "-"),
            "期待回収率": rec.get("expected_roi", "-"),
        })

    if not pred_rows:
        print(f"  → 見送り")

    return pred_rows


def _fetch_result(race, today, spreadsheet_id, credentials_path,
                  fetch_race_result, update_result_row) -> bool:
    """レース結果を取得して成績シートに記録する"""
    venue_code = race["venue_code"]
    race_no = race["race_no"]
    venue = race["venue"]
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 結果取得: {venue} {race_no}R")

    race_id = race.get("race_id")
    result = fetch_race_result(today, venue_code, race_no, race_id=race_id)
    if not result.get("available"):
        print(f"  [WARN] 結果未確定 → 2分後に再試行します")
        return False

    update_result_row(
        spreadsheet_id,
        date=today.strftime("%Y-%m-%d"),
        venue=venue,
        race_no=race_no,
        winner=result["winner"],
        second=result["second"],
        quinella_payout=result.get("quinella_payout", 0),
        credentials_path=credentials_path,
        pred_rows=race.get("pred_rows"),
    )
    print(f"  [OK] 結果: {result['quinella']} 払戻:{result.get('quinella_payout',0)}円")
    return True


def _next_action_time(schedule: list, now: datetime):
    times = []
    for race in schedule:
        if not race["predicted"]:
            times.append(race["scheduled_dt"] - timedelta(minutes=PREDICT_BEFORE_MIN))
        elif not race["result_fetched"]:
            times.append(race["scheduled_dt"] + timedelta(minutes=RESULT_AFTER_MIN))
    future = [t for t in times if t > now]
    return min(future) if future else None
