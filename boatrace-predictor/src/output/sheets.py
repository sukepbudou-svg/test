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
    headers = ["日付", "競艇場", "レース", "狙い", "買い目（3連単）", "的中確率", "オッズ", "期待回収率", "信頼度", "オッズ元", "勝負推奨"]
    sheet.update("A1", [headers])
    _format_header(spreadsheet, sheet, num_cols=11)

    # データ行
    if not recommendations.empty:
        cols = ["date", "venue_name", "race_no", "tier", "combination", "prob", "odds", "expected_roi", "confidence", "odds_source", "bet_label"]
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
                   "オッズ", "期待回収率", "信頼度", "オッズ元", "本日レース数", "勝負推奨"]
        sheet.update("A1", [headers])
        _format_header(spreadsheet, sheet, num_cols=12)

    cols = ["date", "venue_name", "race_no", "tier", "combination", "prob",
            "odds", "expected_roi", "confidence", "odds_source"]
    values = [row.get(c, "-") for c in cols]
    values.append(race_count if race_count is not None else "-")
    values.append(row.get("bet_label", ""))
    sheet.append_row(values, value_input_option="RAW")

    # 会場色 + 信頼度色 + 激熱色を1回のbatch_updateで適用
    last_row = len(sheet.get_all_values())
    try:
        sid = sheet.id
        bg = _VENUE_BG_COLORS.get(str(row.get("venue_name", "")), _DEFAULT_BG)
        reqs = [{"repeatCell": {
            "range": {"sheetId": sid, "startRowIndex": last_row - 1, "endRowIndex": last_row,
                      "startColumnIndex": 0, "endColumnIndex": 12},
            "cell": {"userEnteredFormat": {"backgroundColor": bg}},
            "fields": "userEnteredFormat.backgroundColor",
        }}]
        confidence = str(row.get("confidence", ""))
        if confidence == "★★★★":
            reqs.append({"repeatCell": {
                "range": {"sheetId": sid, "startRowIndex": last_row - 1, "endRowIndex": last_row,
                          "startColumnIndex": 8, "endColumnIndex": 9},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": {"red": 0.6, "green": 0.9, "blue": 0.6},
                    "textFormat": {"bold": True},
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat.bold)",
            }})
        elif confidence == "★★★☆":
            reqs.append({"repeatCell": {
                "range": {"sheetId": sid, "startRowIndex": last_row - 1, "endRowIndex": last_row,
                          "startColumnIndex": 8, "endColumnIndex": 9},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": {"red": 0.8, "green": 0.95, "blue": 0.8},
                }},
                "fields": "userEnteredFormat.backgroundColor",
            }})
        # 激熱ラベルのときはL列を薄いオレンジで色付け・太字
        if row.get("bet_label") == "激熱":
            reqs.append({"repeatCell": {
                "range": {"sheetId": sid, "startRowIndex": last_row - 1, "endRowIndex": last_row,
                          "startColumnIndex": 11, "endColumnIndex": 12},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": {"red": 1.0, "green": 0.85, "blue": 0.6},
                    "textFormat": {"bold": True},
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat.bold)",
            }})
        spreadsheet.batch_update({"requests": reqs})
    except Exception:
        pass


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
    race_count: int = None,
) -> None:
    """
    「成績3」シートに1レース分の結果を記録・的中判定を更新する

    成績シートの列構成:
    日付 | 競艇場 | レース | 予想買い目 | 的中確率 | 期待回収率 |
    実際の結果 | 実際の払戻 | 的中 | 収支
    """
    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    RESULT_SHEET = "成績3"
    RESULT_HEADERS = ["日付", "競艇場", "レース", "予想買い目", "的中確率", "期待回収率",
                      "実際の結果", "実際の払戻", "的中", "収支（円）", "本日レース数"]
    try:
        r_sheet = spreadsheet.worksheet(RESULT_SHEET)
        # クリア後など空の場合はヘッダーを再作成
        if not r_sheet.get_all_values():
            r_sheet.update("A1", [RESULT_HEADERS])
            _format_header(spreadsheet, r_sheet, num_cols=11)
    except gspread.WorksheetNotFound:
        r_sheet = spreadsheet.add_worksheet(title=RESULT_SHEET, rows=2000, cols=12)
        r_sheet.update("A1", [RESULT_HEADERS])
        _format_header(spreadsheet, r_sheet, num_cols=11)

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

    rc = race_count if race_count is not None else "-"

    if not race_preds:
        # 予想なしの場合でも結果だけ記録
        r_sheet.append_row(
            [date, venue_name, race_no, "（予想なし）", "-", "-",
             actual_combination, actual_payout, "-", 0, rc],
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
             actual_combination, actual_payout, hit, profit, rc],
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
        r_sheet = spreadsheet.worksheet("成績3")
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
    「成績3」シートを集計して「サマリー3」シートを更新する
    的中率（レースベース）・回収率・会場別高配当出現率を自動計算する
    """
    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    # 成績3シート読み込み（リトライ付き）
    try:
        r_sheet = spreadsheet.worksheet("成績3")
    except gspread.WorksheetNotFound:
        print("[WARN] 成績3シートが見つかりません")
        return
    try:
        records = _retry_get_records(r_sheet)
    except Exception:
        print("[WARN] 成績3シートの読み込みに失敗しました（APIエラー）")
        return

    if not records:
        return

    total_bets = 0
    total_hits = 0
    total_return = 0
    predicted_race_keys: set = set()
    hit_race_keys: set = set()
    daily: dict = {}
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
            venue_stats[v] = {
                "race_keys": set(),
                "high_payout_keys": set(),
                "hit_race_keys": set(),
                "bets": 0,
                "ret": 0,
                "race_payouts": {},   # race_key → actual_payout
            }

        total_bets += 100
        daily[d]["bets"] += 100
        daily[d]["race_keys"].add(race_key)
        venue_stats[v]["race_keys"].add(race_key)
        venue_stats[v]["bets"] += 100

        # 実際の払戻を記録（倍率帯・高配当集計に使用）
        try:
            actual_payout = int(str(rec.get("実際の払戻", 0)).replace(",", ""))
            venue_stats[v]["race_payouts"][race_key] = actual_payout
            if actual_payout >= _HIGH_PAYOUT_THRESHOLD:
                venue_stats[v]["high_payout_keys"].add(race_key)
        except (ValueError, TypeError):
            pass

        if rec.get("的中", "") == "○":
            total_hits += 1
            hit_race_keys.add(race_key)
            daily[d]["hit_race_keys"].add(race_key)
            venue_stats[v]["hit_race_keys"].add(race_key)
            try:
                payout_val = int(str(rec.get("実際の払戻", 0)).replace(",", ""))
                total_return += payout_val
                daily[d]["ret"] += payout_val
                venue_stats[v]["ret"] += payout_val
            except (ValueError, TypeError):
                pass

    total_pred_points = total_bets // 100
    pred_races = len(predicted_race_keys)
    hit_races = len(hit_race_keys)
    race_hit_rate = f"{hit_races / pred_races * 100:.1f}%" if pred_races > 0 else "0.0%"
    roi = f"{total_return / total_bets * 100:.1f}%" if total_bets > 0 else "0.0%"
    profit = total_return - total_bets

    try:
        s_sheet = spreadsheet.worksheet("サマリー3")
    except gspread.WorksheetNotFound:
        s_sheet = spreadsheet.add_worksheet(title="サマリー3", rows=400, cols=7)

    # 会場リスト（高配当出現率降順）
    venue_list = []
    for vn, vs in venue_stats.items():
        vr = len(vs["race_keys"])
        vh = len(vs["high_payout_keys"])
        vrate = vh / vr if vr > 0 else 0.0
        venue_list.append((vn, vs, vrate))
    venue_list.sort(key=lambda x: x[2], reverse=True)

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

    # ── 会場別高配当出現率 ──
    rows.append(["", "", "", "", "", "", ""])
    rows.append(["", "", "", "", "", "", ""])
    rows.append(["", "", "", "", "", "", ""])
    rows.append(["■ 会場別高配当出現率（払戻10,000円以上）", "", "", "", "", "", ""])
    rows.append(["会場", "予想レース数", "高配当レース数", "高配当出現率", "", "", ""])
    high_payout_venue_rows = []
    for vn, vs, _ in venue_list:
        vr = len(vs["race_keys"])
        vh = len(vs["high_payout_keys"])
        vrate = vh / vr if vr > 0 else 0.0
        high_payout_venue_rows.append((len(rows), vn))
        rows.append([vn, vr, vh, f"{vrate * 100:.1f}%", "", "", ""])

    # ── 会場別収支・勝率 ──
    rows.append(["", "", "", "", "", "", ""])
    rows.append(["", "", "", "", "", "", ""])
    rows.append(["■ 会場別収支・勝率", "", "", "", "", "", ""])
    rows.append(["会場", "予想R数", "的中数", "的中率", "収支", "回収率", ""])
    profit_venue_rows = []
    for vn, vs, _ in venue_list:
        vr = len(vs["race_keys"])
        v_hit = len(vs["hit_race_keys"])
        v_bet = vs["bets"]
        v_ret = vs["ret"]
        v_hit_rate = f"{v_hit / vr * 100:.1f}%" if vr > 0 else "0.0%"
        v_profit = v_ret - v_bet
        v_roi = f"{v_ret / v_bet * 100:.1f}%" if v_bet > 0 else "0.0%"
        profit_venue_rows.append((len(rows), vn))
        rows.append([vn, vr, v_hit, v_hit_rate, f"¥{v_profit:,}", v_roi, ""])

    # ── 倍率帯別出現率 ──
    rows.append(["", "", "", "", "", "", ""])
    rows.append(["", "", "", "", "", "", ""])
    rows.append(["■ 倍率帯別出現率", "", "", "", "", "", ""])
    rows.append(["会場", "〜25倍(回)", "〜25倍(%)", "26〜100倍(回)", "26〜100倍(%)",
                 "101倍〜(回)", "101倍〜(%)"])
    odds_range_venue_rows = []
    for vn, vs, _ in venue_list:
        payouts = list(vs["race_payouts"].values())
        total_p = len(payouts)
        r1 = sum(1 for p in payouts if p <= 2500)
        r2 = sum(1 for p in payouts if 2501 <= p <= 10000)
        r3 = sum(1 for p in payouts if p >= 10001)
        r1p = f"{r1 / total_p * 100:.1f}%" if total_p > 0 else "-"
        r2p = f"{r2 / total_p * 100:.1f}%" if total_p > 0 else "-"
        r3p = f"{r3 / total_p * 100:.1f}%" if total_p > 0 else "-"
        odds_range_venue_rows.append((len(rows), vn))
        rows.append([vn, r1, r1p, r2, r2p, r3, r3p])

    s_sheet.clear()
    s_sheet.update("A1", rows)

    # フォーマット
    try:
        sid = s_sheet.id
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

        # 全会場セクションに会場カラーを適用（7列分）
        color_requests = []
        for row_idx, vn in (high_payout_venue_rows + profit_venue_rows + odds_range_venue_rows):
            bg = _VENUE_BG_COLORS.get(vn, _DEFAULT_BG)
            color_requests.append({"repeatCell": {
                "range": {"sheetId": sid,
                          "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                          "startColumnIndex": 0, "endColumnIndex": 7},
                "cell": {"userEnteredFormat": {"backgroundColor": bg}},
                "fields": "userEnteredFormat.backgroundColor",
            }})
        if color_requests:
            spreadsheet.batch_update({"requests": color_requests})
    except Exception:
        pass

    print(f"[OK] サマリー3更新: 予想{pred_races}レース 的中率={race_hit_rate} 回収率={roi} 収支=¥{profit:,}")


def analyze_nirentan(
    spreadsheet_id: str,
    date: str,
    credentials_path: str = None,
) -> None:
    """
    指定日のド本命・小穴予想について2連単的中分析を行い「2連単分析」タブに書き出す
    """
    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    # 予想シート（日付タブ）読み込み
    try:
        pred_sheet = spreadsheet.worksheet(date)
        pred_records = _retry_get_records(pred_sheet)
    except gspread.WorksheetNotFound:
        print(f"[ERROR] {date}の予想シートが見つかりません")
        return
    except Exception:
        print(f"[ERROR] 予想シートの読み込みに失敗しました")
        return

    # 成績3から実際の結果を読み込み
    try:
        result_sheet = spreadsheet.worksheet("成績3")
        result_records = _retry_get_records(result_sheet)
    except gspread.WorksheetNotFound:
        print(f"[ERROR] 成績3シートが見つかりません")
        return
    except Exception:
        print(f"[ERROR] 成績3シートの読み込みに失敗しました")
        return

    # ド本命・小穴のみ抽出
    target_preds = [
        r for r in pred_records
        if r.get("狙い", "") in ("ド本命", "小穴")
        and str(r.get("日付", "")) == date
    ]

    # 実際の結果を (会場, レース) → 実際の結果 のdictに
    result_dict = {}
    for r in result_records:
        if str(r.get("日付", "")) != date:
            continue
        key = (str(r.get("競艇場", "")), str(r.get("レース", "")))
        actual = str(r.get("実際の結果", ""))
        if actual and actual not in ("-", ""):
            result_dict[key] = actual

    # 2連単配当をレースごとに取得（会場+レースNo → payout）
    from src.collector.result_scraper import fetch_nirentan_payout
    from src.collector.parser import VENUE_CODES
    venue_name_to_code = {v: k for k, v in VENUE_CODES.items()}
    date_dt = datetime.strptime(date, "%Y-%m-%d")
    nirentan_payouts = {}
    seen_races = set()
    for r in result_records:
        if str(r.get("日付", "")) != date:
            continue
        venue = str(r.get("競艇場", ""))
        race_no = str(r.get("レース", ""))
        key = (venue, race_no)
        if key in seen_races:
            continue
        seen_races.add(key)
        venue_code = venue_name_to_code.get(venue)
        if not venue_code:
            continue
        ni_result = fetch_nirentan_payout(date_dt, venue_code, int(race_no))
        if ni_result.get("available"):
            nirentan_payouts[key] = ni_result["payout"]
        time.sleep(0.5)  # サーバー負荷軽減

    # 分析テーブル作成
    rows = []
    hit_count = 0
    for pred in target_preds:
        venue = str(pred.get("競艇場", ""))
        race_no = str(pred.get("レース", ""))
        tier = str(pred.get("狙い", ""))
        combo = str(pred.get("買い目（3連単）", ""))

        # 3連単 → 2連単（1着-2着）
        parts = combo.split("-")
        pred_ni = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else "-"

        actual_combo = result_dict.get((venue, race_no), "")
        actual_parts = actual_combo.split("-")
        actual_ni = f"{actual_parts[0]}-{actual_parts[1]}" if len(actual_parts) >= 2 else "-"

        hit = "○" if pred_ni != "-" and pred_ni == actual_ni else "×"
        if hit == "○":
            hit_count += 1

        payout = nirentan_payouts.get((venue, race_no), "-")
        payout_str = f"¥{payout:,}" if isinstance(payout, int) else "-"
        rows.append([date, venue, race_no, tier, combo, pred_ni, actual_ni, hit, payout_str])

    # ── 会場別集計 ──
    venue_summary: dict = {}
    for row in rows:
        v = row[1]
        hit = row[7]
        payout_str = row[8]
        if v not in venue_summary:
            venue_summary[v] = {"bets": 0, "hits": 0, "total_payout": 0}
        venue_summary[v]["bets"] += 1
        if hit == "○":
            venue_summary[v]["hits"] += 1
            try:
                payout_val = int(payout_str.replace("¥", "").replace(",", ""))
                venue_summary[v]["total_payout"] += payout_val
            except (ValueError, AttributeError):
                pass

    grand_bets = 0
    grand_hits = 0
    grand_payout = 0
    summary_data_rows = []
    for vn in sorted(venue_summary.keys()):
        vs = venue_summary[vn]
        n = vs["bets"]
        h = vs["hits"]
        tp = vs["total_payout"]
        hit_rate = f"{h / n * 100:.1f}%" if n > 0 else "0.0%"
        profit = tp - n * 100
        grand_bets += n
        grand_hits += h
        grand_payout += tp
        summary_data_rows.append([vn, n, h, hit_rate, f"¥{tp:,}", f"¥{profit:,}", "", "", ""])

    grand_profit = grand_payout - grand_bets * 100
    grand_hit_rate = f"{grand_hits / grand_bets * 100:.1f}%" if grand_bets > 0 else "0.0%"

    summary_rows = [
        ["", "", "", "", "", "", "", "", ""],
        ["■ 会場別集計", "", "", "", "", "", "", "", ""],
        ["会場", "予想点数", "的中数", "的中率", "総配当", "収支", "", "", ""],
    ] + summary_data_rows + [
        ["【合計】", grand_bets, grand_hits, grand_hit_rate,
         f"¥{grand_payout:,}", f"¥{grand_profit:,}", "", "", ""],
    ]

    # 「2連単分析」タブに書き出し
    try:
        out_sheet = spreadsheet.worksheet("2連単分析")
        out_sheet.clear()
    except gspread.WorksheetNotFound:
        out_sheet = spreadsheet.add_worksheet(title="2連単分析", rows=500, cols=10)

    headers = ["日付", "会場", "レース", "狙い", "予想3連単", "予想2連単",
               "実際の2連単", "2連単的中", "2連単配当"]
    all_data = [headers] + rows + summary_rows
    out_sheet.update("A1", all_data)
    _format_header(spreadsheet, out_sheet, num_cols=9)

    # 色付け: 的中行は緑、会場別集計ヘッダーはグレー、合計行は濃いグレー、会場行に会場色
    try:
        sid = out_sheet.id
        color_reqs = []

        # 明細行の的中を緑に
        for i, row in enumerate(rows, start=2):
            if row[7] == "○":
                color_reqs.append({"repeatCell": {
                    "range": {"sheetId": sid,
                              "startRowIndex": i - 1, "endRowIndex": i,
                              "startColumnIndex": 0, "endColumnIndex": 9},
                    "cell": {"userEnteredFormat": {
                        "backgroundColor": {"red": 0.7, "green": 0.95, "blue": 0.7}
                    }},
                    "fields": "userEnteredFormat.backgroundColor",
                }})

        # 集計セクションの位置（details + 空行 + "■ 会場別集計" + 列ヘッダー）
        summary_start = len(rows) + 1  # 1-indexed: header row = 1, rows end at len(rows)+1
        section_header_idx = summary_start + 1   # "■ 会場別集計" row (0-indexed)
        col_header_idx = summary_start + 2        # 列ヘッダー行 (0-indexed)

        # "■ 会場別集計" 行をダークグレーに
        color_reqs.append({"repeatCell": {
            "range": {"sheetId": sid,
                      "startRowIndex": section_header_idx, "endRowIndex": section_header_idx + 1,
                      "startColumnIndex": 0, "endColumnIndex": 9},
            "cell": {"userEnteredFormat": {
                "backgroundColor": {"red": 0.3, "green": 0.3, "blue": 0.3},
                "textFormat": {"foregroundColor": {"red": 1, "green": 1, "blue": 1}, "bold": True},
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat)",
        }})
        # 列ヘッダー行をグレーに
        color_reqs.append({"repeatCell": {
            "range": {"sheetId": sid,
                      "startRowIndex": col_header_idx, "endRowIndex": col_header_idx + 1,
                      "startColumnIndex": 0, "endColumnIndex": 9},
            "cell": {"userEnteredFormat": {
                "backgroundColor": {"red": 0.82, "green": 0.86, "blue": 0.92},
                "textFormat": {"bold": True},
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat.bold)",
        }})

        # 会場別行に会場カラー
        for j, (vn, vs) in enumerate([(vn, venue_summary[vn]) for vn in sorted(venue_summary.keys())]):
            row_idx = col_header_idx + 1 + j
            bg = _VENUE_BG_COLORS.get(vn, _DEFAULT_BG)
            color_reqs.append({"repeatCell": {
                "range": {"sheetId": sid,
                          "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                          "startColumnIndex": 0, "endColumnIndex": 9},
                "cell": {"userEnteredFormat": {"backgroundColor": bg}},
                "fields": "userEnteredFormat.backgroundColor",
            }})

        # 合計行をゴールド系に
        total_row_idx = col_header_idx + 1 + len(venue_summary)
        profit_color = (
            {"red": 0.8, "green": 0.95, "blue": 0.8} if grand_profit >= 0
            else {"red": 0.95, "green": 0.8, "blue": 0.8}
        )
        color_reqs.append({"repeatCell": {
            "range": {"sheetId": sid,
                      "startRowIndex": total_row_idx, "endRowIndex": total_row_idx + 1,
                      "startColumnIndex": 0, "endColumnIndex": 9},
            "cell": {"userEnteredFormat": {
                "backgroundColor": profit_color,
                "textFormat": {"bold": True},
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat.bold)",
        }})

        if color_reqs:
            spreadsheet.batch_update({"requests": color_reqs})
    except Exception:
        pass

    total = len(rows)
    print(f"[OK] 2連単分析完了: {total}点中{hit_count}点的中 "
          f"({hit_count/total*100:.1f}%)" if total > 0 else "[OK] 対象データなし")


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
        sheet = spreadsheet.worksheet("サマリー3")
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title="サマリー3", rows=50, cols=5)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = [
        ["集計日時", now],
        ["総購入額", f"¥{roi_history['total_bet']:,}"],
        ["総払戻額", f"¥{roi_history['total_return']:,}"],
        ["実際の回収率", roi_history["roi_pct"]],
    ]
    sheet.update("A1", rows)
    print(f"[OK] サマリー書き込み完了: 回収率 {roi_history['roi_pct']}")
