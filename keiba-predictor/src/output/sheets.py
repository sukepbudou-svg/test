"""
Googleスプレッドシート出力モジュール（競馬版）
"""

import time
from datetime import datetime
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

DEFAULT_CREDENTIALS = Path(__file__).parent.parent.parent / "credentials.json"

# 会場別背景色
_VENUE_COLORS = {
    "東京": {"red": 0.90, "green": 0.95, "blue": 1.00},
    "中山": {"red": 1.00, "green": 0.95, "blue": 0.90},
    "阪神": {"red": 0.95, "green": 1.00, "blue": 0.90},
    "京都": {"red": 1.00, "green": 0.90, "blue": 0.95},
    "中京": {"red": 0.95, "green": 0.95, "blue": 1.00},
    "新潟": {"red": 0.90, "green": 1.00, "blue": 0.95},
    "福島": {"red": 1.00, "green": 0.95, "blue": 0.95},
    "小倉": {"red": 0.95, "green": 0.90, "blue": 1.00},
    "札幌": {"red": 0.90, "green": 0.95, "blue": 0.95},
    "函館": {"red": 0.95, "green": 1.00, "blue": 1.00},
}
_DEFAULT_BG = {"red": 1.0, "green": 1.0, "blue": 1.0}

RESULT_SHEET = "成績"
SUMMARY_SHEET = "サマリー"


def _get_client(credentials_path=None):
    path = Path(credentials_path) if credentials_path else DEFAULT_CREDENTIALS
    creds = Credentials.from_service_account_file(str(path), scopes=SCOPES)
    return gspread.authorize(creds)


def _retry_api(func, retries=3):
    for i in range(retries):
        try:
            return func()
        except gspread.exceptions.APIError as e:
            if "429" in str(e) and i < retries - 1:
                time.sleep(2 ** (i + 2))
            else:
                raise


def append_prediction_row(
    spreadsheet_id: str,
    row: dict,
    credentials_path: str = None,
) -> None:
    """予想行を日付シートに追記する"""
    sheet_name = datetime.now().strftime("%Y-%m-%d")
    client = _get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    try:
        sheet = spreadsheet.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=sheet_name, rows=500, cols=10)
        headers = ["日付", "競馬場", "レース", "狙い", "馬連買い目", "的中確率", "オッズ", "期待回収率", "オッズ元"]
        _retry_api(lambda: sheet.update("A1", [headers]))

    cols = ["date", "venue", "race_no", "tier", "combination", "prob", "odds", "expected_roi", "odds_source"]
    values = [row.get(c, "-") for c in cols]
    _retry_api(lambda: sheet.append_row(values, value_input_option="RAW"))

    # 会場色付け
    last_row = len(_retry_api(sheet.get_all_values))
    bg = _VENUE_COLORS.get(str(row.get("venue", "")), _DEFAULT_BG)
    sid = sheet.id
    spreadsheet.batch_update({"requests": [{"repeatCell": {
        "range": {"sheetId": sid,
                  "startRowIndex": last_row - 1, "endRowIndex": last_row,
                  "startColumnIndex": 0, "endColumnIndex": 9},
        "cell": {"userEnteredFormat": {"backgroundColor": bg}},
        "fields": "userEnteredFormat.backgroundColor",
    }}]})


def update_result_row(
    spreadsheet_id: str,
    date: str,
    venue: str,
    race_no: int,
    winner: int,
    second: int,
    quinella_payout: int,
    credentials_path: str = None,
    pred_rows: list = None,
) -> None:
    """成績シートに結果を記録する"""
    client = _get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)
    actual_combo = f"{min(winner,second)}-{max(winner,second)}"

    headers = ["日付", "競馬場", "レース", "予想買い目", "実際の結果", "払戻", "的中", "収支（円）"]
    try:
        r_sheet = spreadsheet.worksheet(RESULT_SHEET)
        if not r_sheet.get_all_values():
            _retry_api(lambda: r_sheet.update("A1", [headers]))
    except gspread.WorksheetNotFound:
        r_sheet = spreadsheet.add_worksheet(title=RESULT_SHEET, rows=2000, cols=8)
        _retry_api(lambda: r_sheet.update("A1", [headers]))

    # 予想行が渡されていない場合は日付シートから読む
    if pred_rows is None:
        try:
            p_sheet = spreadsheet.worksheet(date)
            all_rows = p_sheet.get_all_records()
            pred_rows = [r for r in all_rows
                         if str(r.get("競馬場", "")) == venue
                         and str(r.get("レース", "")) == str(race_no)]
        except Exception:
            pred_rows = []

    if not pred_rows:
        pred_rows = [{"馬連買い目": "-", "的中確率": "-", "期待回収率": "-"}]

    for pr in pred_rows:
        combo = pr.get("馬連買い目", pr.get("combination", "-"))
        hit = "◎" if combo == actual_combo else "×"
        payout = quinella_payout if hit == "◎" else 0
        profit = payout - 100

        _retry_api(lambda: r_sheet.append_row(
            [date, venue, race_no, combo, actual_combo, quinella_payout, hit, profit],
            value_input_option="RAW"
        ))


def update_summary_sheet(spreadsheet_id: str, credentials_path: str = None) -> None:
    """成績シートを集計してサマリーシートを更新する"""
    client = _get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    try:
        r_sheet = spreadsheet.worksheet(RESULT_SHEET)
        rows = r_sheet.get_all_records()
    except Exception:
        return

    if not rows:
        return

    total = len(rows)
    hits = sum(1 for r in rows if r.get("的中") == "◎")
    total_bet = total * 100
    total_return = sum(int(r.get("払戻", 0) or 0) for r in rows if r.get("的中") == "◎")
    profit = total_return - total_bet
    hit_rate = hits / total if total > 0 else 0
    roi = total_return / total_bet if total_bet > 0 else 0

    summary_rows = [
        ["【競馬予想サマリー】"],
        ["予想点数", total],
        ["的中数", hits],
        ["的中率", f"{hit_rate*100:.1f}%"],
        ["投資額", f"¥{total_bet:,}"],
        ["回収額", f"¥{total_return:,}"],
        ["収支", f"¥{profit:,}"],
        ["回収率", f"{roi*100:.1f}%"],
    ]

    try:
        s_sheet = spreadsheet.worksheet(SUMMARY_SHEET)
        s_sheet.clear()
    except gspread.WorksheetNotFound:
        s_sheet = spreadsheet.add_worksheet(title=SUMMARY_SHEET, rows=50, cols=3)

    _retry_api(lambda: s_sheet.update("A1", summary_rows))
    print(f"[OK] サマリー更新: {hits}/{total}的中 回収率:{roi*100:.1f}% 収支:¥{profit:,}")
