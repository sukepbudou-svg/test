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


def cmd_predict():
    """本日の予想生成"""
    from src.collector.downloader import download_file, extract_lzh
    from src.collector.parser import parse_program
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

    # 特徴量生成（番組表のみ・予測モード）
    import pandas as pd
    df_features = build_features(df_program, pd.DataFrame(), pd.DataFrame())

    # モデル読み込み
    model = load_model()
    if not model:
        print("[ERROR] モデルが見つかりません。先に python main.py --mode train を実行してください")
        return

    # 予想生成
    print("=== 予想生成中 ===")
    recommendations = get_recommendations(model, df_features)

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
        choices=["download", "download_history", "train", "predict", "backtest"],
        required=True,
        help="実行モード"
    )
    parser.add_argument("--months", type=int, default=3, help="過去データ取得月数（download_historyモード用）")
    parser.add_argument("--days-back", type=int, default=0, help="何日前のデータを取得するか（downloadモード用）")
    args = parser.parse_args()

    if args.mode == "download":
        cmd_download(args.days_back)
    elif args.mode == "download_history":
        cmd_download_history(args.months)
    elif args.mode == "train":
        cmd_train()
    elif args.mode == "predict":
        cmd_predict()
    elif args.mode == "backtest":
        cmd_backtest()
