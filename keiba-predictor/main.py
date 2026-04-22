"""
競馬予想ツール メインエントリーポイント
"""

import argparse
import os
from pathlib import Path

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="競馬馬連予想ツール")
    parser.add_argument(
        "--mode", required=True,
        choices=["train", "auto", "predict"],
        help="実行モード: train=学習 / auto=自動予想 / predict=本日予想表示",
    )
    args = parser.parse_args()

    if args.mode == "train":
        cmd_train()
    elif args.mode == "auto":
        cmd_auto()
    elif args.mode == "predict":
        cmd_predict()
