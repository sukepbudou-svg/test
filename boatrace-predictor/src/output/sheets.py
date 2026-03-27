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
    headers = ["日付", "競艇場", "レース", "買い目（3連単）", "的中確率", "平均払戻", "期待回収率", "信頼度", "オッズ元"]
    sheet.update("A1", [headers])

    # データ行
    if not recommendations.empty:
        cols = ["date", "venue_name", "race_no", "combination", "prob", "avg_payout", "expected_roi", "confidence", "odds_source"]
        # 列がない場合（見送り行など）は"-"で埋める
        for col in cols:
            if col not in recommendations.columns:
                recommendations[col] = "-"
        rows = recommendations[cols].values.tolist()
        sheet.update("A2", rows)

    print(f"[OK] スプレッドシートに書き込み完了: シート「{sheet_name}」 {len(recommendations)}行")


def append_prediction_row(
    spreadsheet_id: str,
    row: dict,
    sheet_name: str = "予想",
    credentials_path: str = None,
) -> None:
    """
    1レース分の予想行を「予想」シートに追記する（自動モード用）

    Args:
        row: {date, venue_name, race_no, combination, prob, avg_payout,
               expected_roi, confidence, odds_source}
    """
    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    try:
        sheet = spreadsheet.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=12)
        headers = ["日付", "競艇場", "レース", "買い目（3連単）", "的中確率",
                   "平均払戻", "期待回収率", "信頼度", "オッズ元"]
        sheet.update("A1", [headers])

    cols = ["date", "venue_name", "race_no", "combination", "prob",
            "avg_payout", "expected_roi", "confidence", "odds_source"]
    values = [row.get(c, "-") for c in cols]
    sheet.append_row(values, value_input_option="USER_ENTERED")


def update_result_row(
    spreadsheet_id: str,
    date: str,
    venue_name: str,
    race_no: int,
    actual_combination: str,
    actual_payout: int,
    credentials_path: str = None,
) -> None:
    """
    「成績」シートに1レース分の結果を記録・的中判定を更新する

    成績シートの列構成:
    日付 | 競艇場 | レース | 予想買い目 | 的中確率 | 期待回収率 |
    実際の結果 | 実際の払戻 | 的中 | 収支
    """
    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    RESULT_SHEET = "成績"
    try:
        r_sheet = spreadsheet.worksheet(RESULT_SHEET)
    except gspread.WorksheetNotFound:
        r_sheet = spreadsheet.add_worksheet(title=RESULT_SHEET, rows=2000, cols=12)
        headers = ["日付", "競艇場", "レース", "予想買い目", "的中確率", "期待回収率",
                   "実際の結果", "実際の払戻", "的中", "収支（円）"]
        r_sheet.update("A1", [headers])

    # 「予想」シートから該当レースの予想行を取得
    pred_sheet_name = "予想"
    try:
        p_sheet = spreadsheet.worksheet(pred_sheet_name)
        pred_rows = p_sheet.get_all_records()
    except Exception:
        pred_rows = []

    # 該当レースの予想行を抽出
    race_preds = [
        r for r in pred_rows
        if str(r.get("日付", "")) == date
        and str(r.get("競艇場", "")) == venue_name
        and str(r.get("レース", "")) == str(race_no)
        and r.get("買い目（3連単）", "") not in ("", "見送り", "-")
    ]

    if not race_preds:
        # 予想なしの場合でも結果だけ記録
        r_sheet.append_row(
            [date, venue_name, race_no, "（予想なし）", "-", "-",
             actual_combination, actual_payout, "-", 0],
            value_input_option="USER_ENTERED"
        )
        return

    for pred in race_preds:
        combination = pred.get("買い目（3連単）", "")
        hit = "○" if combination == actual_combination else "×"
        payout = actual_payout if hit == "○" else 0
        profit = payout - 100  # 100円賭け基準

        r_sheet.append_row(
            [date, venue_name, race_no, combination,
             pred.get("的中確率", "-"), pred.get("期待回収率", "-"),
             actual_combination, actual_payout, hit, profit],
            value_input_option="USER_ENTERED"
        )

    print(f"[OK] 成績記録: {venue_name} {race_no}R 結果={actual_combination} 払戻={actual_payout}円")


def update_summary_sheet(
    spreadsheet_id: str,
    credentials_path: str = None,
) -> None:
    """
    「成績」シートを集計して「サマリー」シートを更新する
    的中率・回収率を自動計算する
    """
    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    try:
        r_sheet = spreadsheet.worksheet("成績")
        records = r_sheet.get_all_records()
    except Exception:
        print("[WARN] 成績シートが見つかりません")
        return

    if not records:
        return

    total_bets = 0
    total_hits = 0
    total_return = 0

    for rec in records:
        combination = rec.get("予想買い目", "")
        if combination in ("", "（予想なし）", "見送り", "-"):
            continue
        total_bets += 100  # 100円賭け基準
        if rec.get("的中", "") == "○":
            total_hits += 1
            try:
                total_return += int(str(rec.get("実際の払戻", 0)).replace(",", ""))
            except (ValueError, TypeError):
                pass

    hit_rate = f"{total_hits / (total_bets // 100) * 100:.1f}%" if total_bets > 0 else "0%"
    roi = f"{total_return / total_bets * 100:.1f}%" if total_bets > 0 else "0%"

    try:
        s_sheet = spreadsheet.worksheet("サマリー")
    except gspread.WorksheetNotFound:
        s_sheet = spreadsheet.add_worksheet(title="サマリー", rows=20, cols=3)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = [
        ["集計日時", now],
        ["総購入額（円）", total_bets],
        ["総払戻額（円）", total_return],
        ["的中数", total_hits],
        ["的中率", hit_rate],
        ["回収率", roi],
        ["収支（円）", total_return - total_bets],
    ]
    s_sheet.update("A1", rows)
    print(f"[OK] サマリー更新: 的中率={hit_rate} 回収率={roi} 収支=¥{total_return - total_bets:,}")


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
