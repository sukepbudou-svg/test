"""
Google スプレッドシート出力モジュール
予想結果をスプレッドシートに書き込む
"""

import os
from datetime import datetime
from pathlib import Path

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def get_client(credentials_path: str = None) -> gspread.Client:
    """
    Google Sheets APIクライアントを取得する

    Args:
        credentials_path: サービスアカウントJSONファイルのパス
                         未指定の場合は環境変数 GOOGLE_CREDENTIALS_PATH を参照

    Returns:
        gspread.Client
    """
    cred_path = credentials_path or os.environ.get("GOOGLE_CREDENTIALS_PATH")
    if not cred_path or not Path(cred_path).exists():
        raise FileNotFoundError(
            "Google認証ファイルが見つかりません。\n"
            "環境変数 GOOGLE_CREDENTIALS_PATH にサービスアカウントJSONのパスを設定してください。"
        )

    creds = Credentials.from_service_account_file(cred_path, scopes=SCOPES)
    return gspread.authorize(creds)


def write_predictions(
    spreadsheet_id: str,
    recommendations: pd.DataFrame,
    sheet_name: str = None,
    credentials_path: str = None,
) -> None:
    """
    予想結果をスプレッドシートに書き込む

    Args:
        spreadsheet_id: スプレッドシートのID（URLから取得）
        recommendations: get_recommendations()の出力DataFrame
        sheet_name: シート名（未指定の場合は今日の日付）
        credentials_path: 認証ファイルパス
    """
    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    today = datetime.now().strftime("%Y-%m-%d")
    sheet_name = sheet_name or today

    # シートを取得または作成
    try:
        sheet = spreadsheet.worksheet(sheet_name)
        sheet.clear()
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=sheet_name, rows=200, cols=10)

    # ヘッダー行
    headers = ["日付", "競艇場", "レース", "買い目（3連単）", "的中確率", "期待回収率", "信頼度"]
    sheet.update("A1", [headers])

    # データ行
    if not recommendations.empty:
        rows = recommendations[
            ["date", "venue_name", "race_no", "combination", "prob", "expected_roi", "confidence"]
        ].values.tolist()
        sheet.update(f"A2", rows)

    # ヘッダーの書式設定（太字・背景色）
    sheet.format("A1:G1", {
        "textFormat": {"bold": True},
        "backgroundColor": {"red": 0.2, "green": 0.4, "blue": 0.8},
        "horizontalAlignment": "CENTER",
    })

    # 見送り行をグレーに
    for i, row in enumerate(recommendations.itertuples(), start=2):
        if row.combination == "見送り":
            sheet.format(f"A{i}:G{i}", {
                "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
            })
        elif "★★★" in str(row.confidence):
            sheet.format(f"A{i}:G{i}", {
                "backgroundColor": {"red": 1.0, "green": 0.95, "blue": 0.8}
            })

    print(f"[OK] スプレッドシートに書き込み完了: シート「{sheet_name}」 {len(recommendations)}行")


def write_backtest_summary(
    spreadsheet_id: str,
    roi_history: dict,
    credentials_path: str = None,
) -> None:
    """
    バックテスト結果（回収率サマリー）をスプレッドシートに書き込む

    Args:
        spreadsheet_id: スプレッドシートのID
        roi_history: calculate_roi_history()の出力dict
        credentials_path: 認証ファイルパス
    """
    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    try:
        sheet = spreadsheet.worksheet("サマリー")
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title="サマリー", rows=50, cols=5)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = [
        ["集計日時", now],
        ["総購入額", f"¥{roi_history['total_bet']:,}"],
        ["総払戻額", f"¥{roi_history['total_return']:,}"],
        ["実際の回収率", roi_history["roi_pct"]],
    ]
    sheet.update("A1", rows)
    print(f"[OK] サマリー書き込み完了: 回収率 {roi_history['roi_pct']}")
