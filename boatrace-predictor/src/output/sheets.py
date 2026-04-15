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

# 高配当しきい値（100円換算で100倍以上 = 10,000円以上）
_HIGH_PAYOUT_THRESHOLD = 10000


def _retry_get_records(sheet, max_attempts: int = 3) -> list:
    """API エラー時にリトライして records を取得する"""
    for attempt in range(max_attempts):
        try:
            return sheet.get_all_records()
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    return []

# 競艇場ごとの背景色（地域別に色分け）
# RGB値は0〜1の範囲
_VENUE_BG_COLORS = {
    # 関東 - 水色系
    "桐生":   {"red": 0.82, "green": 0.91, "blue": 0.98},
    "戸田":   {"red": 0.79, "green": 0.89, "blue": 0.98},
    "江戸川": {"red": 0.76, "green": 0.88, "blue": 0.97},
    "平和島": {"red": 0.74, "green": 0.86, "blue": 0.97},
    "多摩川": {"red": 0.72, "green": 0.85, "blue": 0.96},
    # 中部 - 緑系
    "浜名湖": {"red": 0.82, "green": 0.96, "blue": 0.82},
    "蒲郡":   {"red": 0.80, "green": 0.94, "blue": 0.80},
    "常滑":   {"red": 0.78, "green": 0.93, "blue": 0.78},
    "津":     {"red": 0.76, "green": 0.92, "blue": 0.76},
    "三国":   {"red": 0.74, "green": 0.91, "blue": 0.74},
    # 近畿 - 紫系
    "びわこ": {"red": 0.91, "green": 0.82, "blue": 0.98},
    "住之江": {"red": 0.89, "green": 0.80, "blue": 0.97},
    "尼崎":   {"red": 0.87, "green": 0.78, "blue": 0.96},
    # 中国・四国 - 黄色系
    "鳴門":   {"red": 1.00, "green": 0.97, "blue": 0.77},
    "丸亀":   {"red": 1.00, "green": 0.95, "blue": 0.74},
    "児島":   {"red": 1.00, "green": 0.93, "blue": 0.71},
    "宮島":   {"red": 1.00, "green": 0.91, "blue": 0.68},
    # 九州 - オレンジ・ピンク系
    "徳山":   {"red": 1.00, "green": 0.88, "blue": 0.80},
    "下関":   {"red": 1.00, "green": 0.86, "blue": 0.78},
    "若松":   {"red": 1.00, "green": 0.84, "blue": 0.76},
    "芦屋":   {"red": 1.00, "green": 0.82, "blue": 0.74},
    "福岡":   {"red": 1.00, "green": 0.80, "blue": 0.72},
    "唐津":   {"red": 1.00, "green": 0.78, "blue": 0.70},
    "大村":   {"red": 1.00, "green": 0.76, "blue": 0.68},
}
_DEFAULT_BG = {"red": 0.95, "green": 0.95, "blue": 0.95}  # 不明場所はグレー


def _format_header(spreadsheet, sheet, num_cols: int = 9) -> None:
    """1行目を太字・薄いグレー背景にして行を固定する"""
    sid = sheet.id
    try:
        spreadsheet.batch_update({"requests": [
            # 太字 + 背景色
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
            # 1行目を固定
            {"updateSheetProperties": {
                "properties": {"sheetId": sid, "gridProperties": {"frozenRowCount": 1}},
                "fields": "gridProperties.frozenRowCount",
            }},
        ]})
    except Exception:
        pass


def _apply_venue_color(sheet, row_no: int, venue_name: str, num_cols: int = 9) -> None:
    """指定行に競艇場カラーの背景色を適用する"""
    color = _VENUE_BG_COLORS.get(venue_name, _DEFAULT_BG)
    last_col = chr(ord("A") + num_cols - 1)
    try:
        sheet.format(
            f"A{row_no}:{last_col}{row_no}",
            {"backgroundColor": color},
        )
    except Exception:
        pass  # 色付けに失敗しても予想データは書き込み済みのためスルー


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


def _ensure_spreadsheet_title(spreadsheet) -> None:
    """スプレッドシートのタイトルが 'BOAT LAB' でなければ変更する"""
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
    _ensure_spreadsheet_title(spreadsheet)

    today = datetime.now().strftime("%Y-%m-%d")
    sheet_name = sheet_name or today

    # シートを取得または作成
    try:
        sheet = spreadsheet.worksheet(sheet_name)
        sheet.clear()
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=sheet_name, rows=200, cols=10)

    # ヘッダー行
    headers = ["日付", "競艇場", "レース", "狙い", "買い目（3連単）", "的中確率", "オッズ", "期待回収率", "信頼度", "オッズ元"]
    sheet.update("A1", [headers])
    _format_header(spreadsheet, sheet, num_cols=10)

    # データ行
    if not recommendations.empty:
        cols = ["date", "venue_name", "race_no", "tier", "combination", "prob", "odds", "expected_roi", "confidence", "odds_source"]
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
    sheet_name: str = None,
    credentials_path: str = None,
    race_count: int = None,
) -> None:
    """
    1レース分の予想行を日付シートに追記する（自動モード用）

    Args:
        row: {date, venue_name, race_no, combination, prob, avg_payout,
               expected_roi, confidence, odds_source}
        race_count: 本日何レース目か（K列に記入）
    """
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
                   "オッズ", "期待回収率", "信頼度", "オッズ元", "本日レース数"]
        sheet.update("A1", [headers])
        _format_header(spreadsheet, sheet, num_cols=11)

    cols = ["date", "venue_name", "race_no", "tier", "combination", "prob",
            "odds", "expected_roi", "confidence", "odds_source"]
    values = [row.get(c, "-") for c in cols]
    values.append(race_count if race_count is not None else "-")
    sheet.append_row(values, value_input_option="RAW")

    # 競艇場ごとに背景色を適用
    last_row = len(sheet.get_all_values())
    _apply_venue_color(sheet, last_row, str(row.get("venue_name", "")), num_cols=11)


def _color_result_row(spreadsheet, sheet, row_no: int, venue_name: str, hit: str) -> None:
    """成績シートの1行に会場色＋的中色をリアルタイムで適用する"""
    try:
        sid = sheet.id
        bg = _VENUE_BG_COLORS.get(venue_name, _DEFAULT_BG)
        requests = [
            # 行全体に会場カラー
            {"repeatCell": {
                "range": {"sheetId": sid, "startRowIndex": row_no - 1, "endRowIndex": row_no,
                          "startColumnIndex": 0, "endColumnIndex": 10},
                "cell": {"userEnteredFormat": {"backgroundColor": bg}},
                "fields": "userEnteredFormat.backgroundColor",
            }}
        ]
        # 的中セル（I列=index8）に色付け
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
        pass  # 色付け失敗は無視（データは書き込み済み）


def update_result_row(
    spreadsheet_id: str,
    date: str,
    venue_name: str,
    race_no: int,
    actual_combination: str,
    actual_payout: int,
    credentials_path: str = None,
    pred_rows_override: list = None,
) -> None:
    """
    「成績2」シートに1レース分の結果を記録・的中判定を更新する

    成績シートの列構成:
    日付 | 競艇場 | レース | 予想買い目 | 的中確率 | 期待回収率 |
    実際の結果 | 実際の払戻 | 的中 | 収支
    """
    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    RESULT_SHEET = "成績2"
    RESULT_HEADERS = ["日付", "競艇場", "レース", "予想買い目", "的中確率", "期待回収率",
                      "実際の結果", "実際の払戻", "的中", "収支（円）"]
    try:
        r_sheet = spreadsheet.worksheet(RESULT_SHEET)
        # クリア後など空の場合はヘッダーを再作成
        if not r_sheet.get_all_values():
            r_sheet.update("A1", [RESULT_HEADERS])
            _format_header(spreadsheet, r_sheet, num_cols=10)
    except gspread.WorksheetNotFound:
        r_sheet = spreadsheet.add_worksheet(title=RESULT_SHEET, rows=2000, cols=12)
        r_sheet.update("A1", [RESULT_HEADERS])
        _format_header(spreadsheet, r_sheet, num_cols=10)

    # メモリキャッシュ（auto_runner から渡された場合）を優先使用
    # → Google Sheets API 503 エラーを回避するため
    if pred_rows_override is not None:
        pred_rows = pred_rows_override
    else:
        pred_rows = []
        for sheet_title in [date, "予想"]:
            for attempt in range(3):
                try:
                    p_sheet = spreadsheet.worksheet(sheet_title)
                    pred_rows = _retry_get_records(p_sheet)
                    break
                except gspread.WorksheetNotFound:
                    break
                except Exception:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
            if pred_rows:
                break
        if not pred_rows:
            print(f"  [WARN] 予想データの読み込みに失敗しました（APIエラー）")

    # 該当レースの予想行を抽出
    # 「買い目（3連単）」列の名前が異なる場合も考慮
    race_preds = [
        r for r in pred_rows
        if str(r.get("日付", "")) == date
        and str(r.get("競艇場", "")) == venue_name
        and str(r.get("レース", "")) == str(race_no)
        and r.get("買い目（3連単）", "") not in ("", "見送り", "-")
    ]

    # 列名がずれている場合のフォールバック（旧フォーマット対応）
    if not race_preds:
        race_preds = [
            r for r in pred_rows
            if str(r.get("日付", "")) == date
            and str(r.get("競艇場", "")) == venue_name
            and str(r.get("レース", "")) == str(race_no)
        ]
        # 「見送り」「-」「空」以外のものを買い目として扱う
        race_preds = [
            r for r in race_preds
            if any(
                str(v) not in ("", "見送り", "-", "（予想なし）")
                and "-" in str(v)  # "1-2-3" 形式の買い目を探す
                for v in r.values()
            )
        ]
        # 買い目列を特定して正規化
        for r in race_preds:
            if "買い目（3連単）" not in r or r["買い目（3連単）"] in ("", "-"):
                for k, v in r.items():
                    if str(v).count("-") == 2 and all(c.isdigit() or c == "-" for c in str(v)):
                        r["買い目（3連単）"] = str(v)
                        break

    if not race_preds:
        # 予想なしの場合でも結果だけ記録
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
        profit = payout - 100  # 100円賭け基準

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
    """成績シートの全行に競艇場カラーと的中色を一括適用する"""
    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    try:
        r_sheet = spreadsheet.worksheet("成績2")
    except gspread.WorksheetNotFound:
        return

    all_rows = r_sheet.get_all_values()
    if len(all_rows) <= 1:
        return

    fmt_requests = []
    sheet_id = r_sheet.id

    for i, row in enumerate(all_rows[1:], start=2):  # 2行目からデータ行
        venue_name = row[1] if len(row) > 1 else ""
        hit = row[8] if len(row) > 8 else ""

        bg = _VENUE_BG_COLORS.get(venue_name, _DEFAULT_BG)

        # 行全体に競艇場カラー
        fmt_requests.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": i - 1,
                    "endRowIndex": i,
                    "startColumnIndex": 0,
                    "endColumnIndex": 10,
                },
                "cell": {"userEnteredFormat": {"backgroundColor": bg}},
                "fields": "userEnteredFormat.backgroundColor",
            }
        })

        # 的中セル（I列=index8）: ○は緑、×は薄赤
        if hit in ("○", "×"):
            hit_bg = (
                {"red": 0.7, "green": 0.95, "blue": 0.7} if hit == "○"
                else {"red": 0.98, "green": 0.85, "blue": 0.85}
            )
            fmt_requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": i - 1,
                        "endRowIndex": i,
                        "startColumnIndex": 8,
                        "endColumnIndex": 9,
                    },
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
    「成績2」シートを集計して「サマリー2」シートを更新する
    的中率（レースベース）・回収率・会場別高配当出現率を自動計算する
    """
    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    # 成績2シート読み込み（リトライ付き）
    try:
        r_sheet = spreadsheet.worksheet("成績2")
    except gspread.WorksheetNotFound:
        print("[WARN] 成績2シートが見つかりません")
        return
    try:
        records = _retry_get_records(r_sheet)
    except Exception:
        print("[WARN] 成績2シートの読み込みに失敗しました（APIエラー）")
        return

    if not records:
        return

    total_bets = 0
    total_hits = 0
    total_return = 0
    # 予想レース数カウント（重複排除）
    predicted_race_keys: set = set()
    hit_race_keys: set = set()
    # 日付別集計
    daily: dict = {}
    # 会場別集計（高配当出現率用）
    venue_stats: dict = {}

    for rec in records:
        combination = rec.get("予想買い目", "")
        if combination in ("", "（予想なし）", "見送り", "-"):
            continue
        d = str(rec.get("日付", ""))
        v = str(rec.get("競艇場", ""))
        rn = str(rec.get("レース", ""))
        race_key = (d, v, rn)

        predicted_race_keys.add(race_key)

        if d not in daily:
            daily[d] = {"bets": 0, "ret": 0, "race_keys": set(), "hit_race_keys": set()}
        if v not in venue_stats:
            venue_stats[v] = {"race_keys": set(), "high_payout_keys": set()}

        total_bets += 100
        daily[d]["bets"] += 100
        daily[d]["race_keys"].add(race_key)
        venue_stats[v]["race_keys"].add(race_key)

        # 実際の払戻で高配当チェック（的中有無問わず）
        try:
            actual_payout = int(str(rec.get("実際の払戻", 0)).replace(",", ""))
            if actual_payout >= _HIGH_PAYOUT_THRESHOLD:
                venue_stats[v]["high_payout_keys"].add(race_key)
        except (ValueError, TypeError):
            pass

        if rec.get("的中", "") == "○":
            total_hits += 1
            hit_race_keys.add(race_key)
            daily[d]["hit_race_keys"].add(race_key)
            try:
                payout_val = int(str(rec.get("実際の払戻", 0)).replace(",", ""))
                total_return += payout_val
                daily[d]["ret"] += payout_val
            except (ValueError, TypeError):
                pass

    total_pred_points = total_bets // 100
    pred_races = len(predicted_race_keys)
    hit_races = len(hit_race_keys)
    race_hit_rate = f"{hit_races / pred_races * 100:.1f}%" if pred_races > 0 else "0.0%"
    roi = f"{total_return / total_bets * 100:.1f}%" if total_bets > 0 else "0.0%"
    profit = total_return - total_bets

    try:
        s_sheet = spreadsheet.worksheet("サマリー2")
    except gspread.WorksheetNotFound:
        s_sheet = spreadsheet.add_worksheet(title="サマリー2", rows=200, cols=7)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = [
        ["【予想成績サマリー】", "", "", "", "", "", ""],
        ["集計日時", now, "", "", "", "", ""],
        ["", "", "", "", "", "", ""],
        ["■ 全期間合計", "", "", "", "", "", ""],
        ["予想点数", "予想レース数", "的中数（レース）", "的中率（レース）", "総払戻", "回収率", "収支"],
        [total_pred_points, pred_races, hit_races, race_hit_rate,
         f"¥{total_return:,}", roi, f"¥{profit:,}"],
        ["", "", "", "", "", "", ""],
        ["■ 日付別内訳", "", "", "", "", "", ""],
        ["日付", "予想点数", "予想レース数", "的中数", "的中率", "払戻合計", "収支"],
    ]

    for d in sorted(daily.keys()):
        dd = daily[d]
        n = dd["bets"] // 100
        pr = len(dd["race_keys"])
        h = len(dd["hit_race_keys"])
        r = dd["ret"]
        dr = f"{h / pr * 100:.1f}%" if pr > 0 else "0.0%"
        dp = r - dd["bets"]
        rows.append([d, n, pr, h, dr, f"¥{r:,}", f"¥{dp:,}"])

    # 会場別高配当出現率セクション
    rows.append(["", "", "", "", "", "", ""])
    rows.append(["■ 会場別高配当出現率（払戻10,000円以上）", "", "", "", "", "", ""])
    rows.append(["会場", "予想レース数", "高配当レース数", "高配当出現率", "", "", ""])

    venue_list = []
    for vn, vs in venue_stats.items():
        vr = len(vs["race_keys"])
        vh = len(vs["high_payout_keys"])
        vrate = vh / vr if vr > 0 else 0.0
        venue_list.append((vn, vr, vh, vrate))
    venue_list.sort(key=lambda x: x[3], reverse=True)

    for vn, vr, vh, vrate in venue_list:
        rows.append([vn, vr, vh, f"{vrate * 100:.1f}%", "", "", ""])

    s_sheet.clear()
    s_sheet.update("A1", rows)

    # フォーマット
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
    except Exception:
        pass

    print(f"[OK] サマリー2更新: 予想{pred_races}レース 的中率={race_hit_rate} 回収率={roi} 収支=¥{profit:,}")


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
        sheet = spreadsheet.worksheet("サマリー2")
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title="サマリー2", rows=50, cols=5)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = [
        ["集計日時", now],
        ["総購入額", f"¥{roi_history['total_bet']:,}"],
        ["総払戻額", f"¥{roi_history['total_return']:,}"],
        ["実際の回収率", roi_history["roi_pct"]],
    ]
    sheet.update("A1", rows)
    print(f"[OK] サマリー書き込み完了: 回収率 {roi_history['roi_pct']}")
