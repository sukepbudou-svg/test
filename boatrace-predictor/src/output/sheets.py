"""
Google スプレッドシート出力モジュール
予想結果をスプレッドシートに書き込む
"""

import os
import time
from datetime import datetime
from pathlib import Path

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# 競艇場ごとの背景色（地域別に色分け）
_VENUE_BG_COLORS = {
    "桐生":   {"red": 0.82, "green": 0.91, "blue": 0.98},
    "戸田":   {"red": 0.79, "green": 0.89, "blue": 0.98},
    "江戸川": {"red": 0.76, "green": 0.88, "blue": 0.97},
    "平和島": {"red": 0.74, "green": 0.86, "blue": 0.97},
    "多摩川": {"red": 0.72, "green": 0.85, "blue": 0.96},
    "浜名湖": {"red": 0.82, "green": 0.96, "blue": 0.82},
    "蒲郡":   {"red": 0.80, "green": 0.94, "blue": 0.80},
    "常滑":   {"red": 0.78, "green": 0.93, "blue": 0.78},
    "津":     {"red": 0.76, "green": 0.92, "blue": 0.76},
    "三国":   {"red": 0.74, "green": 0.91, "blue": 0.74},
    "びわこ": {"red": 0.91, "green": 0.82, "blue": 0.98},
    "住之江": {"red": 0.89, "green": 0.80, "blue": 0.97},
    "尼崎":   {"red": 0.87, "green": 0.78, "blue": 0.96},
    "鳴門":   {"red": 1.00, "green": 0.97, "blue": 0.77},
    "丸亀":   {"red": 1.00, "green": 0.95, "blue": 0.74},
    "児島":   {"red": 1.00, "green": 0.93, "blue": 0.71},
    "宮島":   {"red": 1.00, "green": 0.91, "blue": 0.68},
    "徳山":   {"red": 1.00, "green": 0.88, "blue": 0.80},
    "下関":   {"red": 1.00, "green": 0.86, "blue": 0.78},
    "若松":   {"red": 1.00, "green": 0.84, "blue": 0.76},
    "芦屋":   {"red": 1.00, "green": 0.82, "blue": 0.74},
    "福岡":   {"red": 1.00, "green": 0.80, "blue": 0.72},
    "唐津":   {"red": 1.00, "green": 0.78, "blue": 0.70},
    "大村":   {"red": 1.00, "green": 0.76, "blue": 0.68},
}
_DEFAULT_BG = {"red": 0.95, "green": 0.95, "blue": 0.95}
_HIGH_PAYOUT_THRESHOLD = 10000


def _retry_get_records(sheet, max_attempts: int = 3) -> list:
    """get_all_records() を最大 max_attempts 回リトライする"""
    for attempt in range(max_attempts):
        try:
            return sheet.get_all_records()
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)  # 1s, 2s
            else:
                raise e
    return []


def _format_header(spreadsheet, sheet, num_cols: int = 9) -> None:
    sid = sheet.id
    try:
        spreadsheet.batch_update({"requests": [
            {"repeatCell": {
                "range": {"sheetId": sid, "startRowIndex": 0, "endRowIndex": 1,
                          "startColumnIndex": 0, "endColumnIndex": num_cols},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": {"red": 0.82, "green": 0.86, "blue": 0.92},
                    "textFormat": {"bold": True},
                    "horizontalAlignment": "CENTER",
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat.bold,horizontalAlignment)",
            }},
            {"updateSheetProperties": {
                "properties": {"sheetId": sid, "gridProperties": {"frozenRowCount": 1}},
                "fields": "gridProperties.frozenRowCount",
            }},
        ]})
    except Exception:
        pass


def _apply_venue_color(sheet, row_no: int, venue_name: str, num_cols: int = 9) -> None:
    color = _VENUE_BG_COLORS.get(venue_name, _DEFAULT_BG)
    last_col = chr(ord("A") + num_cols - 1)
    try:
        sheet.format(f"A{row_no}:{last_col}{row_no}", {"backgroundColor": color})
    except Exception:
        pass


def get_client(credentials_path: str = None) -> gspread.Client:
    cred_path = credentials_path or os.environ.get("GOOGLE_CREDENTIALS_PATH")
    if not cred_path or not Path(cred_path).exists():
        raise FileNotFoundError(
            "Google認証ファイルが見つかりません。\n"
            "環境変数 GOOGLE_CREDENTIALS_PATH にサービスアカウントJSONのパスを設定してください。"
        )
    creds = Credentials.from_service_account_file(cred_path, scopes=SCOPES)
    return gspread.authorize(creds)


def _ensure_spreadsheet_title(spreadsheet) -> None:
    try:
        if not spreadsheet.title.startswith("BOAT LAB"):
            spreadsheet.update_title("BOAT LAB")
            print("[OK] スプレッドシートのタイトルを「BOAT LAB」に設定しました")
    except Exception:
        pass


def write_predictions(
    spreadsheet_id: str,
    recommendations: pd.DataFrame,
    sheet_name: str = None,
    credentials_path: str = None,
) -> None:
    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)
    _ensure_spreadsheet_title(spreadsheet)

    today = datetime.now().strftime("%Y-%m-%d")
    sheet_name = sheet_name or today

    try:
        sheet = spreadsheet.worksheet(sheet_name)
        sheet.clear()
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=sheet_name, rows=200, cols=10)

    headers = ["日付", "競艇場", "レース", "狙い", "買い目（3連単）", "的中確率", "オッズ", "期待回収率", "信頼度", "オッズ元"]
    sheet.update("A1", [headers])
    _format_header(spreadsheet, sheet, num_cols=10)

    if not recommendations.empty:
        cols = ["date", "venue_name", "race_no", "tier", "combination", "prob", "odds", "expected_roi", "confidence", "odds_source"]
        for col in cols:
            if col not in recommendations.columns:
                recommendations[col] = "-"
        rows = recommendations[cols].values.tolist()
        sheet.update("A2", rows)

    print(f"[OK] スプレッドシートに書き込み完了: シート「{sheet_name}」 {len(recommendations)}行")


def append_prediction_row(
    spreadsheet_id: str,
    row: dict,
    sheet_name: str = None,
    credentials_path: str = None,
) -> None:
    if sheet_name is None:
        sheet_name = datetime.now().strftime("%Y-%m-%d")

    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)
    _ensure_spreadsheet_title(spreadsheet)

    try:
        sheet = spreadsheet.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=12)
        headers = ["日付", "競艇場", "レース", "狙い", "買い目（3連単）", "的中確率",
                   "オッズ", "期待回収率", "信頼度", "オッズ元"]
        sheet.update("A1", [headers])
        _format_header(spreadsheet, sheet, num_cols=10)

    cols = ["date", "venue_name", "race_no", "tier", "combination", "prob",
            "odds", "expected_roi", "confidence", "odds_source"]
    values = [row.get(c, "-") for c in cols]
    sheet.append_row(values, value_input_option="RAW")

    last_row = len(sheet.get_all_values())
    _apply_venue_color(sheet, last_row, str(row.get("venue_name", "")))


def _color_result_row(spreadsheet, sheet, row_no: int, venue_name: str, hit: str) -> None:
    try:
        sid = sheet.id
        bg = _VENUE_BG_COLORS.get(venue_name, _DEFAULT_BG)
        requests = [
            {"repeatCell": {
                "range": {"sheetId": sid, "startRowIndex": row_no - 1, "endRowIndex": row_no,
                          "startColumnIndex": 0, "endColumnIndex": 10},
                "cell": {"userEnteredFormat": {"backgroundColor": bg}},
                "fields": "userEnteredFormat.backgroundColor",
            }}
        ]
        if hit in ("○", "×"):
            hit_bg = (
                {"red": 0.7, "green": 0.95, "blue": 0.7} if hit == "○"
                else {"red": 0.98, "green": 0.85, "blue": 0.85}
            )
            requests.append({"repeatCell": {
                "range": {"sheetId": sid, "startRowIndex": row_no - 1, "endRowIndex": row_no,
                          "startColumnIndex": 8, "endColumnIndex": 9},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": hit_bg,
                    "textFormat": {"bold": True},
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat.bold)",
            }})
        spreadsheet.batch_update({"requests": requests})
    except Exception:
        pass


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
        _format_header(spreadsheet, r_sheet, num_cols=10)

    # 予想シートから該当レースの予想行を取得（503エラーに対しリトライ）
    pred_rows = []
    for sheet_title in [date, "予想"]:
        for attempt in range(3):
            try:
                p_sheet = spreadsheet.worksheet(sheet_title)
                pred_rows = _retry_get_records(p_sheet)
                break
            except gspread.WorksheetNotFound:
                break  # 次のシート名へ
            except Exception:
                if attempt < 2:
                    time.sleep(2 ** attempt)
        if pred_rows:
            break

    race_preds = [
        r for r in pred_rows
        if str(r.get("日付", "")) == date
        and str(r.get("競艇場", "")) == venue_name
        and str(r.get("レース", "")) == str(race_no)
        and r.get("買い目（3連単）", "") not in ("", "見送り", "-")
    ]

    if not race_preds:
        race_preds = [
            r for r in pred_rows
            if str(r.get("日付", "")) == date
            and str(r.get("競艇場", "")) == venue_name
            and str(r.get("レース", "")) == str(race_no)
        ]
        race_preds = [
            r for r in race_preds
            if any(
                str(v) not in ("", "見送り", "-", "（予想なし）")
                and "-" in str(v)
                for v in r.values()
            )
        ]
        for r in race_preds:
            if "買い目（3連単）" not in r or r["買い目（3連単）"] in ("", "-"):
                for k, v in r.items():
                    if str(v).count("-") == 2 and all(c.isdigit() or c == "-" for c in str(v)):
                        r["買い目（3連単）"] = str(v)
                        break

    if not race_preds:
        r_sheet.append_row(
            [date, venue_name, race_no, "（予想なし）", "-", "-",
             actual_combination, actual_payout, "-", 0],
            value_input_option="RAW"
        )
        _color_result_row(spreadsheet, r_sheet, len(r_sheet.get_all_values()), venue_name, "-")
        return

    for pred in race_preds:
        combination = pred.get("買い目（3連単）", "")
        hit = "○" if combination == actual_combination else "×"
        payout = actual_payout if hit == "○" else 0
        profit = payout - 100

        r_sheet.append_row(
            [date, venue_name, race_no, combination,
             pred.get("的中確率", "-"), pred.get("期待回収率", "-"),
             actual_combination, actual_payout, hit, profit],
            value_input_option="RAW"
        )
        _color_result_row(spreadsheet, r_sheet, len(r_sheet.get_all_values()), venue_name, hit)

    print(f"[OK] 成績記録: {venue_name} {race_no}R 結果={actual_combination} 払戻={actual_payout}円")


def apply_colors_to_results_sheet(
    spreadsheet_id: str,
    credentials_path: str = None,
) -> None:
    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    try:
        r_sheet = spreadsheet.worksheet("成績")
    except gspread.WorksheetNotFound:
        return

    all_rows = r_sheet.get_all_values()
    if len(all_rows) <= 1:
        return

    fmt_requests = []
    sheet_id = r_sheet.id

    for i, row in enumerate(all_rows[1:], start=2):
        venue_name = row[1] if len(row) > 1 else ""
        hit = row[8] if len(row) > 8 else ""
        bg = _VENUE_BG_COLORS.get(venue_name, _DEFAULT_BG)

        fmt_requests.append({
            "repeatCell": {
                "range": {"sheetId": sheet_id, "startRowIndex": i - 1, "endRowIndex": i,
                          "startColumnIndex": 0, "endColumnIndex": 10},
                "cell": {"userEnteredFormat": {"backgroundColor": bg}},
                "fields": "userEnteredFormat.backgroundColor",
            }
        })

        if hit in ("○", "×"):
            hit_bg = (
                {"red": 0.7, "green": 0.95, "blue": 0.7} if hit == "○"
                else {"red": 0.98, "green": 0.85, "blue": 0.85}
            )
            fmt_requests.append({
                "repeatCell": {
                    "range": {"sheetId": sheet_id, "startRowIndex": i - 1, "endRowIndex": i,
                              "startColumnIndex": 8, "endColumnIndex": 9},
                    "cell": {"userEnteredFormat": {
                        "backgroundColor": hit_bg,
                        "textFormat": {"bold": True},
                    }},
                    "fields": "userEnteredFormat(backgroundColor,textFormat.bold)",
                }
            })

    if fmt_requests:
        spreadsheet.batch_update({"requests": fmt_requests})
        print(f"[OK] 成績シート色付け完了: {len(all_rows)-1}行")


def update_summary_sheet(
    spreadsheet_id: str,
    credentials_path: str = None,
) -> None:
    """
    「成績」シートを集計して「サマリー」シートを更新する
    """
    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    # 成績シートをリトライ付きで読み込む
    records = []
    for attempt in range(3):
        try:
            r_sheet = spreadsheet.worksheet("成績")
            records = _retry_get_records(r_sheet)
            break
        except gspread.WorksheetNotFound:
            print("[WARN] 成績シートがまだ作成されていません")
            return
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print("[WARN] 成績シートの読み込みに失敗しました（APIエラー）")
                return

    if not records:
        return

    total_bets = 0
    total_return = 0
    total_pred_races: set = set()
    total_hit_races: set = set()
    daily: dict = {}
    venues: dict = {}
    race_payouts: dict = {}

    for rec in records:
        combination = rec.get("予想買い目", "")
        if combination in ("", "（予想なし）", "見送り", "-"):
            continue
        d = str(rec.get("日付", ""))
        venue = str(rec.get("競艇場", ""))
        race_no = str(rec.get("レース", ""))
        race_key = (d, venue, race_no)

        if d not in daily:
            daily[d] = {"bets": 0, "ret": 0, "pred_races": set(), "hit_races": set()}
        if venue not in venues:
            venues[venue] = {"bets": 0, "ret": 0, "pred_races": set(), "hit_races": set()}

        total_bets += 100
        daily[d]["bets"] += 100
        venues[venue]["bets"] += 100
        total_pred_races.add(race_key)
        daily[d]["pred_races"].add(race_key)
        venues[venue]["pred_races"].add(race_key)

        try:
            ap = int(str(rec.get("実際の払戻", 0)).replace(",", ""))
            if ap > 0:
                race_payouts[race_key] = ap
        except (ValueError, TypeError):
            pass

        if rec.get("的中", "") == "○":
            total_hit_races.add(race_key)
            daily[d]["hit_races"].add(race_key)
            venues[venue]["hit_races"].add(race_key)
            try:
                v = int(str(rec.get("実際の払戻", 0)).replace(",", ""))
                total_return += v
                daily[d]["ret"] += v
                venues[venue]["ret"] += v
            except (ValueError, TypeError):
                pass

    venue_high_payout: dict = {v: set() for v in venues}
    for race_key, payout in race_payouts.items():
        if payout >= _HIGH_PAYOUT_THRESHOLD:
            v = race_key[1]
            if v in venue_high_payout:
                venue_high_payout[v].add(race_key)

    total_points = total_bets // 100
    pred_race_count = len(total_pred_races)
    hit_race_count = len(total_hit_races)
    hit_rate = f"{hit_race_count / pred_race_count * 100:.1f}%" if pred_race_count > 0 else "0.0%"
    roi = f"{total_return / total_bets * 100:.1f}%" if total_bets > 0 else "0.0%"
    profit = total_return - total_bets

    try:
        s_sheet = spreadsheet.worksheet("サマリー")
    except gspread.WorksheetNotFound:
        s_sheet = spreadsheet.add_worksheet(title="サマリー", rows=200, cols=9)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = [
        ["【予想成績サマリー】", "", "", "", "", "", "", "", ""],
        ["集計日時", now, "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["■ 全期間合計", "", "", "", "", "", "", "", ""],
        ["予想点数", "予想レース数", "的中数", "的中率", "総払戻", "回収率", "収支", "", ""],
        [total_points, pred_race_count, hit_race_count, hit_rate,
         f"¥{total_return:,}", roi, f"¥{profit:,}", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["■ 日付別内訳", "", "", "", "", "", "", "", ""],
        ["日付", "予想点数", "予想レース数", "的中数", "的中率", "払戻合計", "収支", "", ""],
    ]

    for d in sorted(daily.keys()):
        dd = daily[d]
        n = dd["bets"] // 100
        pred_n = len(dd["pred_races"])
        hit_n = len(dd["hit_races"])
        r = dd["ret"]
        dr = f"{hit_n / pred_n * 100:.1f}%" if pred_n > 0 else "0.0%"
        dp = r - dd["bets"]
        rows.append([d, n, pred_n, hit_n, dr, f"¥{r:,}", f"¥{dp:,}", "", ""])

    rows.append(["", "", "", "", "", "", "", "", ""])
    rows.append(["■ 会場別成績", "", "", "", "", "", "", "", ""])
    venue_header_row = len(rows) + 1
    rows.append(["会場", "予想点数", "予想レース数", "的中数", "的中率",
                 "高配当数(100倍以上)", "高配当出現率", "払戻合計", "回収率"])

    sorted_venues = sorted(
        venues.keys(),
        key=lambda v: len(venue_high_payout.get(v, set())) / max(len(venues[v]["pred_races"]), 1),
        reverse=True,
    )

    for v in sorted_venues:
        vd = venues[v]
        vn = vd["bets"] // 100
        vpred_n = len(vd["pred_races"])
        vhit_n = len(vd["hit_races"])
        vr = vd["ret"]
        vhp = len(venue_high_payout.get(v, set()))
        vhit_rate = f"{vhit_n / vpred_n * 100:.1f}%" if vpred_n > 0 else "0.0%"
        vhp_rate = f"{vhp / vpred_n * 100:.1f}%" if vpred_n > 0 else "0.0%"
        vroi = f"{vr / vd['bets'] * 100:.1f}%" if vd["bets"] > 0 else "0.0%"
        rows.append([v, vn, vpred_n, vhit_n, vhit_rate, vhp, vhp_rate, f"¥{vr:,}", vroi])

    s_sheet.clear()
    s_sheet.update("A1", rows)

    try:
        s_sheet.format("A5:G5", {
            "backgroundColor": {"red": 0.2, "green": 0.2, "blue": 0.2},
            "textFormat": {"foregroundColor": {"red": 1, "green": 1, "blue": 1}, "bold": True},
        })
        s_sheet.format("A9:G9", {
            "backgroundColor": {"red": 0.85, "green": 0.85, "blue": 0.85},
            "textFormat": {"bold": True},
        })
        profit_color = (
            {"red": 0.8, "green": 0.95, "blue": 0.8} if profit >= 0
            else {"red": 0.95, "green": 0.8, "blue": 0.8}
        )
        s_sheet.format("G6", {"backgroundColor": profit_color, "textFormat": {"bold": True}})
        s_sheet.format(f"A{venue_header_row}:I{venue_header_row}", {
            "backgroundColor": {"red": 0.85, "green": 0.85, "blue": 0.85},
            "textFormat": {"bold": True},
        })
    except Exception:
        pass

    print(f"[OK] サマリー更新: 的中率={hit_rate} 回収率={roi} 収支=¥{profit:,}")


def write_backtest_summary(
    spreadsheet_id: str,
    roi_history: dict,
    credentials_path: str = None,
) -> None:
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
