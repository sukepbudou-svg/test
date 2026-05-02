"""
競馬予想ツール メインエントリーポイント
"""

import argparse
import os
import sys
from pathlib import Path

# keiba-predictorフォルダをsys.pathに追加（どこから実行しても動くように）
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# credentials.jsonの検索順: 環境変数 → keiba-predictorフォルダ → boatrace-predictorフォルダ
def _find_credentials() -> str:
    env = os.environ.get("CREDENTIALS_PATH")
    if env and Path(env).exists():
        return env
    local = Path(__file__).parent / "credentials.json"
    if local.exists():
        return str(local)
    sibling = Path(__file__).parent.parent / "boatrace-predictor" / "credentials.json"
    if sibling.exists():
        return str(sibling)
    return str(local)  # 見つからない場合はデフォルトパスを返す

CREDENTIALS_PATH = _find_credentials()


def cmd_train():
    """過去データを学習してモデルを作成する"""
    import pandas as pd
    from src.collector.history import fetch_history
    from src.features.builder import build_race_features, build_quinella_features
    from src.model.trainer import train_model

    months = int(os.environ.get("TRAIN_MONTHS", 3))
    print(f"=== 学習モード: 過去{months}ヶ月分のデータで学習 ===")

    df_history = fetch_history(months=months)
    if df_history.empty:
        print("[ERROR] 学習データが取得できませんでした")
        return

    train_model(df_history)
    print("=== 学習完了 ===")


def cmd_auto():
    """自動予想モードを開始する"""
    from src.scheduler.auto_runner import run_auto

    spreadsheet_id = os.environ.get("KEIBA_SPREADSHEET_ID", "")
    if not spreadsheet_id or spreadsheet_id == "your_spreadsheet_id_here":
        print("[ERROR] KEIBA_SPREADSHEET_ID 環境変数を設定してください")
        print("  例: set KEIBA_SPREADSHEET_ID=your_spreadsheet_id")
        return

    credentials_path = os.environ.get("CREDENTIALS_PATH", CREDENTIALS_PATH)
    run_auto(spreadsheet_id, credentials_path)


def cmd_test_one(race_id_arg: str = None):
    """次の1レースだけ予想してスプレッドシートに書き込む（動作確認用）"""
    import time
    from datetime import datetime
    import requests
    from src.collector.scraper import (
        fetch_today_schedule, fetch_race_card,
        fetch_odds_quinella, fetch_odds_wide,
        fetch_jockey_stats, fetch_horse_past_results, fetch_training_times,
        VENUE_CODES, HEADERS,
    )
    from src.features.builder import build_race_features
    from src.model.trainer import load_model
    from src.model.predictor import get_recommendations, apply_training_filter
    from src.output.sheets import append_prediction_row

    spreadsheet_id = os.environ.get("KEIBA_SPREADSHEET_ID", "")
    if not spreadsheet_id:
        print("[ERROR] KEIBA_SPREADSHEET_ID 環境変数を設定してください")
        return
    credentials_path = os.environ.get("CREDENTIALS_PATH", CREDENTIALS_PATH)

    today = datetime.now()
    print(f"=== 1レーステスト予想: {today.strftime('%Y-%m-%d %H:%M')} ===\n")

    model = load_model()
    if not model:
        print("[ERROR] モデルが見つかりません。先に --mode train を実行してください")
        return

    # race_id直接指定の場合はスケジュール取得をスキップ
    if race_id_arg:
        venue_code = race_id_arg[4:6]
        race_no = int(race_id_arg[10:12])
        venue = VENUE_CODES.get(venue_code, "不明")
        race_id = race_id_arg
        print(f"[指定] race_id={race_id} → {venue} {race_no}R\n")
    else:
        # スケジュール取得
        date_str = today.strftime('%Y%m%d')
        schedule_url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}"
        print(f"スケジュール取得中: {schedule_url}")
        schedule = fetch_today_schedule(today)

        if not schedule:
            print("[ERROR] 本日の開催情報が取得できませんでした")
            print("  考えられる原因:")
            print("  1. 早朝のためスケジュールがまだ公開されていない（9時以降に再試行）")
            print("  2. 本日は開催なし")
            print("  3. ネットワークエラー")
            print()
            print("  race_idを直接指定して実行することもできます:")
            print("  python main.py --mode test_one --race-id 202605020511")
            return

        print(f"スケジュール取得: {len(schedule)}レース")
        for r in schedule[:5]:
            print(f"  {r.get('venue')} {r.get('race_no')}R  {r.get('scheduled_time','--:--')}  race_id={r.get('race_id')}")
        if len(schedule) > 5:
            print(f"  ... 他{len(schedule)-5}レース")
        print()

        # 発走時刻が未来のレースを探す（時刻なしも含めて最初の1つを選ぶ）
        target = None
        now = today
        for r in schedule:
            t_str = r.get("scheduled_time")
            if t_str:
                try:
                    hh, mm = map(int, t_str.split(":"))
                    dt = datetime(today.year, today.month, today.day, hh, mm)
                    if dt <= now:
                        continue  # 発走済みはスキップ
                except (ValueError, AttributeError):
                    pass
            target = r
            break

        if not target:
            print("[INFO] 本日の未発走レースが見つかりませんでした（全レース発走済みの可能性）")
            return

        venue_code = target["venue_code"]
        race_no = target["race_no"]
        venue = target["venue"]
        race_id = target.get("race_id")
        print(f"対象: {venue} {race_no}R (発走 {target.get('scheduled_time','--:--')} / race_id={race_id})\n")

    # 出馬表
    card = fetch_race_card(today, venue_code, race_no, race_id=race_id)
    if not card or not card.get("horses"):
        print(f"[ERROR] 出馬表が取得できませんでした")
        return
    print(f"出走頭数: {len(card['horses'])}頭")

    # 騎手・馬の成績
    jockey_stats, horse_histories = {}, {}
    for horse in card["horses"]:
        jockey = horse.get("jockey", "")
        jockey_id = horse.get("jockey_id", "")
        horse_name = horse.get("horse_name", "")
        horse_id = horse.get("horse_id", "")
        if jockey_id and jockey and jockey not in jockey_stats:
            stats = fetch_jockey_stats(jockey_id)
            if stats:
                jockey_stats[jockey] = stats
            time.sleep(0.5)
        if horse_id and horse_name and horse_name not in horse_histories:
            history = fetch_horse_past_results(horse_id, n=5)
            if history:
                horse_histories[horse_name] = history
            time.sleep(0.5)
    print(f"騎手:{len(jockey_stats)}名 / 馬:{len(horse_histories)}頭 の成績取得完了")

    # 特徴量・オッズ
    df_race = build_race_features(card,
                                  jockey_stats=jockey_stats or None,
                                  horse_histories=horse_histories or None)
    if df_race.empty:
        print("[ERROR] 特徴量生成に失敗しました")
        return

    live_odds = fetch_odds_quinella(today, venue_code, race_no, race_id=race_id)
    live_wide_odds = fetch_odds_wide(today, venue_code, race_no, race_id=race_id)
    print(f"馬連オッズ:{len(live_odds)}通り / ワイドオッズ:{len(live_wide_odds)}通り 取得")

    # 予想
    recs = get_recommendations(model, df_race, live_odds, live_wide_odds=live_wide_odds)

    # 追い切り評価
    training_times = fetch_training_times(race_id) if race_id else {}
    if training_times:
        print(f"追い切りデータ:{len(training_times)}頭")
    recs = apply_training_filter(recs, training_times)

    print("\n--- 予想結果 ---")
    print(recs[["tier", "combination", "prob", "odds", "expected_roi", "training_eval"]].to_string(index=False))
    print()

    # スプレッドシートへ書き込み
    date_str = today.strftime("%Y-%m-%d")
    written = 0
    for _, rec in recs.iterrows():
        combo = rec.get("combination", "")
        if combo in ("見送り", "-", "") or combo.startswith("見送り"):
            continue
        row_dict = {**rec.to_dict(), "date": date_str, "venue": venue}
        append_prediction_row(spreadsheet_id, row_dict, credentials_path)
        written += 1

    print(f"[OK] スプレッドシートに {written} 行書き込み完了")


def cmd_predict():
    """本日のレースを一括予想して表示する（スプレッドシートへの書き込みなし）"""
    from datetime import datetime
    from src.collector.scraper import fetch_today_schedule, fetch_race_card, fetch_odds_quinella
    from src.features.builder import build_race_features
    from src.model.trainer import load_model
    from src.model.predictor import get_recommendations

    today = datetime.now()
    model = load_model()
    if not model:
        return

    schedule = fetch_today_schedule(today)
    if not schedule:
        print("[WARN] 本日の開催情報が取得できませんでした")
        return

    print(f"本日の開催: {len(schedule)}レース\n")
    for race in schedule:
        venue_code = race["venue_code"]
        race_no = race["race_no"]
        venue = race["venue"]

        card = fetch_race_card(today, venue_code, race_no)
        if not card or not card.get("horses"):
            continue

        df_race = build_race_features(card)
        live_odds = fetch_odds_quinella(today, venue_code, race_no)
        recs = get_recommendations(model, df_race, live_odds)

        print(f"【{venue} {race_no}R】")
        print(recs.to_string(index=False))
        print()


def cmd_debug():
    """スケジュール取得の診断を行う"""
    from datetime import datetime
    from src.collector.scraper import fetch_today_schedule
    today = datetime.now()
    print(f"=== スケジュール診断: {today.strftime('%Y-%m-%d %H:%M')} ===\n")
    schedule = fetch_today_schedule(today)
    if schedule:
        print(f"\n取得成功: {len(schedule)}レース")
        for r in schedule:
            print(f"  {r['venue']} {r['race_no']}R  {r.get('scheduled_time','--:--')}  {r['race_id']}")
    else:
        print("\n[結果] スケジュール取得できませんでした")
        print("  ※出馬表の探索が完了しても見つからない場合:")
        print("  1. まだ朝早い時間帯（レース情報は通常8時頃から公開）")
        print("  2. 本日は開催なし")
        print("  → 直接race_idを指定する場合: python main.py --mode test_one --race-id 202605XXXXXXXX")


def cmd_debug_card(race_id_arg: str = None):
    """出馬表HTMLの最初の馬行を出力して horse_id 抽出の問題を診断する"""
    import requests
    from bs4 import BeautifulSoup
    from datetime import datetime
    from src.collector.scraper import fetch_today_schedule, HEADERS

    today = datetime.now()
    if race_id_arg:
        race_id = race_id_arg
    else:
        schedule = fetch_today_schedule(today)
        if not schedule:
            print("[ERROR] スケジュール取得失敗")
            return
        race_id = schedule[0]["race_id"]

    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    print(f"URL: {url}\n")
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.encoding = "EUC-JP"
    soup = BeautifulSoup(resp.text, "html.parser")

    # HorseList行を探す
    horse_rows = soup.select("tr.HorseList, tr.Shutuba_HorseList")
    print(f"HorseList行数: {len(horse_rows)}")
    if not horse_rows:
        print("\n--- 全trタグのclass一覧 ---")
        for tr in soup.find_all("tr")[:10]:
            print(f"  class={tr.get('class')}")
        return

    row = horse_rows[0]
    print(f"\n--- 最初の馬行のHTML（最初の1000文字）---")
    print(str(row)[:1000])

    print(f"\n--- 行内のaタグ一覧 ---")
    for a in row.find_all("a", href=True):
        print(f"  href={a.get('href')!r:60}  text={a.get_text(strip=True)!r}")

    print(f"\n--- 行内のtdタグ（最初の5個）---")
    for i, td in enumerate(row.find_all("td")[:5]):
        print(f"  cells[{i}]: class={td.get('class')}  text={td.get_text(strip=True)!r}")

    print(f"\n--- horse/jockey パターン検索 ---")
    import re
    row_html = str(row)
    for pattern, label in [
        (r'/horse/(\d{4,})', 'href /horse/'),
        (r'horse[_\-]?id["\s:=\']+(\d{4,})', 'JS horse_id'),
        (r'"horse":.*?"(\d{10})"', 'JSON horse'),
        (r'/jockey/(\w+)', 'jockey href'),
    ]:
        found = re.findall(pattern, row_html, re.IGNORECASE)
        print(f"  {label}: {found[:5]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="競馬馬連予想ツール")
    parser.add_argument(
        "--mode", required=True,
        choices=["train", "auto", "predict", "test_one", "debug", "debug_card"],
        help="実行モード: train=学習 / auto=自動予想 / predict=本日予想表示 / test_one=1レーステスト / debug=診断 / debug_card=出馬表HTML診断",
    )
    parser.add_argument(
        "--race-id", default=None,
        help="test_one用: 予想するレースのrace_id（例: 202605020511）",
    )
    args = parser.parse_args()

    if args.mode == "train":
        cmd_train()
    elif args.mode == "auto":
        cmd_auto()
    elif args.mode == "predict":
        cmd_predict()
    elif args.mode == "test_one":
        cmd_test_one(race_id_arg=args.race_id)
    elif args.mode == "debug":
        cmd_debug()
    elif args.mode == "debug_card":
        cmd_debug_card(race_id_arg=args.race_id)
