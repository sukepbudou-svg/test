"""
競艇3連単予想ツール - メインスクリプト
使い方:
  python main.py --mode download   # 今日のデータをダウンロード
  python main.py --mode train      # モデルを学習（初回・再学習時）
  python main.py --mode predict    # 本日の予想を生成してスプレッドシートに出力
  python main.py --mode backtest   # 過去データで回収率を検証
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"


def cmd_download(days_back: int = 0):
    """データダウンロード"""
    from src.collector.downloader import download_file, extract_lzh

    today = datetime.now()
    target = today - timedelta(days=days_back)

    print(f"=== データダウンロード: {target.strftime('%Y-%m-%d')} ===")

    # 今日の番組表
    b_path = download_file("B", target)
    if b_path:
        extract_lzh(b_path)

    # 昨日以前の競走成績
    if days_back > 0:
        k_path = download_file("K", target)
        if k_path:
            extract_lzh(k_path)
    else:
        # 昨日の競走成績
        yesterday = today - timedelta(days=1)
        k_path = download_file("K", yesterday)
        if k_path:
            extract_lzh(k_path)


def cmd_download_history(months: int = 3):
    """過去データの一括ダウンロード（学習用）"""
    from src.collector.downloader import download_range, extract_all

    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=30 * months)

    print(f"=== 過去データ一括ダウンロード: {start_date.strftime('%Y-%m-%d')} 〜 {end_date.strftime('%Y-%m-%d')} ===")
    print("※ サーバー負荷軽減のため3秒間隔でアクセスします。時間がかかります。")

    download_range("B", start_date, end_date, interval=3.0)
    download_range("K", start_date, end_date, interval=3.0)

    print("=== 解凍中 ===")
    extract_all("B")
    extract_all("K")


def cmd_train():
    """モデル学習"""
    import pandas as pd
    from src.collector.parser import parse_all_programs, parse_all_results
    from src.features.builder import build_features
    from src.model.trainer import train_model, load_model
    from src.model.predictor import build_payout_lookup

    print("=== データ読み込み中 ===")
    df_program = parse_all_programs(RAW_DIR / "B")
    df_rank, df_payout = parse_all_results(RAW_DIR / "K")

    if df_program.empty or df_rank.empty:
        print("[ERROR] データがありません。先に python main.py --mode download_history を実行してください")
        return

    print(f"番組表: {len(df_program)}行, 競走成績: {len(df_rank)}行")

    print("=== 特徴量生成中 ===")
    df_features = build_features(df_program, df_rank, df_payout)
    print(f"特徴量: {len(df_features)}レース分")

    print("=== モデル学習中 ===")
    train_model(df_features)

    print("=== 払戻ルックアップ生成中 ===")
    model = load_model()
    if model:
        build_payout_lookup(df_payout, model, df_features)
    else:
        print("[WARN] モデル読み込みに失敗したため払戻ルックアップをスキップしました")

    print("=== 学習完了 ===")


def _convert_beforeinfo(raw: dict) -> "pd.DataFrame":
    """
    fetch_beforeinfo_for_races の出力を DataFrame に変換する
    raw: {(venue_code, race_no): {1: {"exhibition_time": 6.82, "exhibition_st": 0.12}, ...}}
    """
    import pandas as pd
    from datetime import datetime
    records = []
    today = datetime.now().strftime("%Y-%m-%d")
    for (venue_code, race_no), boats in raw.items():
        for boat_no, info in boats.items():
            records.append({
                "date": today,
                "venue_code": venue_code,
                "race_no": race_no,
                "boat_no": boat_no,
                "exhibition_time": info.get("exhibition_time"),
                "exhibition_st": info.get("exhibition_st"),
            })
    return pd.DataFrame(records) if records else pd.DataFrame()


def _filter_upcoming_races(df_program: "pd.DataFrame", df_features: "pd.DataFrame", now: datetime) -> "pd.DataFrame":
    """
    発走時刻が現在時刻より後のレースのみを残す。
    発走時刻が取得できないレースは除外しない（全件残す）。
    """
    import pandas as pd

    now_str = now.strftime("%H:%M")

    if "scheduled_time" not in df_program.columns:
        return df_features

    # venue_code + race_no → scheduled_time のマッピング
    time_map = (
        df_program[["venue_code", "race_no", "scheduled_time"]]
        .dropna(subset=["scheduled_time"])
        .drop_duplicates(subset=["venue_code", "race_no"])
        .set_index(["venue_code", "race_no"])["scheduled_time"]
        .to_dict()
    )

    if not time_map:
        print("[INFO] 発走時刻データなし: 全レースを対象とします")
        return df_features

    def is_upcoming(row):
        key = (str(row["venue_code"]).zfill(2), int(row["race_no"]))
        t = time_map.get(key)
        if t is None:
            return True  # 時刻不明なレースは残す
        return t > now_str

    mask = df_features.apply(is_upcoming, axis=1)
    filtered = df_features[mask]

    skipped = len(df_features) - len(filtered)
    print(f"[INFO] 現在時刻: {now_str} / 発走済み除外: {skipped}レース / 残り: {len(filtered)}レース")
    return filtered


def cmd_predict(venue: str = None, race_no: int = None):
    """本日の予想生成"""
    from src.collector.downloader import download_file, extract_lzh
    from src.collector.parser import parse_program
    from src.collector.odds import fetch_odds_for_races
    from src.collector.beforeinfo import fetch_beforeinfo_for_races
    from src.features.builder import build_features, add_course_advantage, get_feature_columns
    from src.model.trainer import load_model
    from src.model.predictor import get_recommendations
    from src.output.sheets import write_predictions

    today = datetime.now()

    # 今日の番組表をダウンロード
    b_path = download_file("B", today)
    if not b_path:
        print("[ERROR] 番組表のダウンロードに失敗しました")
        return

    txt_path = extract_lzh(b_path)
    if not txt_path:
        print("[ERROR] 番組表の解凍に失敗しました")
        return

    # パース
    df_program = parse_program(txt_path)
    if df_program.empty:
        print("[ERROR] 番組表のパースに失敗しました")
        return

    # 直前情報（展示タイム）取得 - レース直前のみ利用可能
    import pandas as pd
    print("=== 直前情報（展示タイム）取得中 ===")
    # まず番組表のみで仮の特徴量を生成してレース一覧を取得
    df_features_tmp = build_features(df_program, pd.DataFrame(), pd.DataFrame())
    df_beforeinfo_raw = fetch_beforeinfo_for_races(df_features_tmp, today)

    # 直前情報をDataFrame形式に変換
    df_beforeinfo = _convert_beforeinfo(df_beforeinfo_raw)

    # 特徴量生成（展示タイム込み）
    df_features = build_features(df_program, pd.DataFrame(), pd.DataFrame(), df_beforeinfo)

    # 未発走レースのみに絞り込む（競艇場・レース番号の指定がない場合）
    if venue is None and race_no is None:
        df_features = _filter_upcoming_races(df_program, df_features, today)

    # 競艇場・レース番号で絞り込む
    if venue is not None or race_no is not None:
        # 場名または場コードで検索
        from src.collector.parser import VENUE_CODES
        target_codes = []
        if venue is not None:
            # 場名（例: "福岡"）または場コード（例: "22"）で検索
            for code, name in VENUE_CODES.items():
                if venue in (code, name):
                    target_codes.append(code)
            if not target_codes:
                print(f"[ERROR] 競艇場が見つかりません: '{venue}'")
                print(f"  使用可能: {', '.join(VENUE_CODES.values())}")
                return
            mask = df_features["venue_code"].isin(target_codes)
            df_features = df_features[mask]

        if race_no is not None:
            df_features = df_features[df_features["race_no"] == race_no]

        if df_features.empty:
            print(f"[ERROR] 指定のレースが見つかりません（競艇場: {venue or '全場'} / レース: {race_no or '全レース'}R）")
            return
        print(f"[INFO] 絞り込み: {venue or '全場'} {race_no or '全'}R → {len(df_features)}行")

    # モデル読み込み
    model = load_model()
    if not model:
        print("[ERROR] モデルが見つかりません。先に python main.py --mode train を実行してください")
        return

    # リアルタイムオッズ取得
    print("=== リアルタイムオッズ取得中 ===")
    all_live_odds = fetch_odds_for_races(df_features, today)

    # 予想生成
    print("=== 予想生成中 ===")
    recommendations = get_recommendations(model, df_features, all_live_odds=all_live_odds)

    # 結果表示
    print("\n【本日の推奨買い目】")
    print(recommendations[recommendations["combination"] != "見送り"].to_string(index=False))

    # スプレッドシートに出力
    spreadsheet_id = os.environ.get("SPREADSHEET_ID", "")
    credentials_path = os.environ.get("GOOGLE_CREDENTIALS_PATH", "")
    if spreadsheet_id and spreadsheet_id != "your_spreadsheet_id_here" and Path(credentials_path).exists():
        write_predictions(spreadsheet_id, recommendations)
    else:
        print("\n[INFO] スプレッドシート出力をスキップしました（未設定）")
        print("      Google連携を設定する場合は .env ファイルを編集してください")


def cmd_auto():
    """自動予想モード（発走10分前に自動予想・結果記録）"""
    from src.scheduler.auto_runner import run_auto

    spreadsheet_id = os.environ.get("SPREADSHEET_ID", "")
    credentials_path = os.environ.get("GOOGLE_CREDENTIALS_PATH", "")

    if not spreadsheet_id or spreadsheet_id == "your_spreadsheet_id_here":
        print("[ERROR] SPREADSHEET_ID が設定されていません（.env ファイルを確認してください）")
        return
    if not Path(credentials_path).exists():
        print("[ERROR] Google認証ファイルが見つかりません（GOOGLE_CREDENTIALS_PATH を確認してください）")
        return

    run_auto(spreadsheet_id, credentials_path)


def cmd_results(date_str: str = None):
    """今日の予想と実際の結果を照合して成績シートに書き込む"""
    import time
    from src.collector.parser import VENUE_CODES
    from src.collector.result_scraper import fetch_race_result
    from src.output.sheets import update_result_row, update_summary_sheet, apply_colors_to_results_sheet

    spreadsheet_id = os.environ.get("SPREADSHEET_ID", "")
    credentials_path = os.environ.get("GOOGLE_CREDENTIALS_PATH", "")
    if not spreadsheet_id or spreadsheet_id == "your_spreadsheet_id_here":
        print("[ERROR] SPREADSHEET_ID が設定されていません")
        return
    if not Path(credentials_path).exists():
        print("[ERROR] Google認証ファイルが見つかりません")
        return

    import gspread
    from google.oauth2.service_account import Credentials
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(spreadsheet_id)

    today = datetime.now()
    target_date = date_str or today.strftime("%Y-%m-%d")

    # 予想シートを読み込む
    try:
        sheet = spreadsheet.worksheet(target_date)
        rows = sheet.get_all_records()
    except Exception:
        print(f"[ERROR] 予想シート「{target_date}」が見つかりません")
        return

    # venue_name → venue_code の逆引きマップ
    name_to_code = {v: k for k, v in VENUE_CODES.items()}

    # 有効な予想行（見送り・ヘッダー以外）を抽出してレース単位でまとめる
    races = {}
    for row in rows:
        combo = str(row.get("買い目（3連単）", ""))
        venue_name = str(row.get("競艇場", ""))
        race_no_raw = row.get("レース", "")
        if combo in ("", "見送り", "-", "買い目（3連単）"):
            continue
        try:
            race_no = int(race_no_raw)
        except (ValueError, TypeError):
            continue
        key = (venue_name, race_no)
        races[key] = venue_name  # 重複排除用（結果取得は1回）

    if not races:
        print(f"[INFO] {target_date} の予想データが見つかりません")
        return

    print(f"=== 結果照合開始: {target_date} / {len(races)}レース ===")
    hit = 0
    total = 0

    for (venue_name, race_no), _ in sorted(races.items(), key=lambda x: x[0][1]):
        venue_code = name_to_code.get(venue_name)
        if not venue_code:
            print(f"  [SKIP] 場コード不明: {venue_name}")
            continue

        result = fetch_race_result(today, venue_code, race_no)
        if not result.get("available"):
            print(f"  [--] {venue_name} {race_no}R: 結果未確定（レース未終了の可能性）")
            continue

        combination = result["combination"]
        payout = result["payout"]
        print(f"  [OK] {venue_name} {race_no}R: 結果={combination} 払戻={payout:,}円")

        update_result_row(
            spreadsheet_id,
            date=target_date,
            venue_name=venue_name,
            race_no=race_no,
            actual_combination=combination,
            actual_payout=payout,
            credentials_path=credentials_path,
        )

        # 的中チェック（このレースの予想と照合）
        race_preds = [r for r in rows
                      if str(r.get("競艇場", "")) == venue_name
                      and str(r.get("レース", "")) == str(race_no)
                      and str(r.get("買い目（3連単）", "")) not in ("", "見送り", "-")]
        for pred in race_preds:
            total += 1
            if pred.get("買い目（3連単）") == combination:
                hit += 1

        time.sleep(1.0)

    print(f"\n=== 照合完了 ===")
    if total > 0:
        print(f"  予想数: {total}点 / 的中: {hit}点 / 的中率: {hit/total*100:.1f}%")
    apply_colors_to_results_sheet(spreadsheet_id, credentials_path)
    update_summary_sheet(spreadsheet_id, credentials_path)
    print(f"  → スプレッドシートの「成績」「サマリー」シートを更新しました")


def cmd_backtest():
    """バックテスト（過去データで回収率検証）"""
    import pandas as pd
    from src.collector.parser import parse_all_programs, parse_all_results
    from src.features.builder import build_features
    from src.model.trainer import load_model
    from src.model.predictor import get_recommendations, calculate_roi_history

    print("=== バックテスト開始 ===")
    df_program = parse_all_programs(RAW_DIR / "B")
    df_rank, df_payout = parse_all_results(RAW_DIR / "K")
    df_features = build_features(df_program, df_rank, df_payout)

    model = load_model()
    if not model:
        print("[ERROR] モデルが見つかりません")
        return

    # 直近30日分でバックテスト
    recent_dates = sorted(df_features["date"].unique())[-30:]
    df_recent = df_features[df_features["date"].isin(recent_dates)]

    recommendations = get_recommendations(model, df_recent)
    roi_history = calculate_roi_history(df_rank, df_payout, recommendations)

    print(f"\n=== バックテスト結果（直近{len(recent_dates)}日間） ===")
    print(f"総購入額  : ¥{roi_history['total_bet']:,}")
    print(f"総払戻額  : ¥{roi_history['total_return']:,}")
    print(f"実際の回収率: {roi_history['roi_pct']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="競艇3連単予想ツール")
    parser.add_argument(
        "--mode",
        choices=["download", "download_history", "train", "predict", "backtest", "auto", "results"],
        required=True,
        help="実行モード"
    )
    parser.add_argument("--months", type=int, default=3, help="過去データ取得月数（download_historyモード用）")
    parser.add_argument("--days-back", type=int, default=0, help="何日前のデータを取得するか（downloadモード用）")
    parser.add_argument("--venue", type=str, default=None, help="競艇場名または場コード（例: 福岡 または 22）")
    parser.add_argument("--race", type=int, default=None, help="レース番号（例: 6）")
    args = parser.parse_args()

    if args.mode == "download":
        cmd_download(args.days_back)
    elif args.mode == "download_history":
        cmd_download_history(args.months)
    elif args.mode == "train":
        cmd_train()
    elif args.mode == "predict":
        cmd_predict(venue=args.venue, race_no=args.race)
    elif args.mode == "backtest":
        cmd_backtest()
    elif args.mode == "auto":
        cmd_auto()
    elif args.mode == "results":
        cmd_results()
