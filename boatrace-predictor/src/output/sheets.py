"""
Google スプレッドシート出力モジュール
予想結果をスプレッドシートに書き込む
"""

import os
import re
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

RESULT_SHEET = "成績18"
SUMMARY_SHEET = "サマリー18"

# 24場 荒れやすさランキング（万舟率・コース特性ベース、1位=最も荒れやすい）
_VENUE_RANKING = [
    "1  ▲江戸川",
    "2  ▲平和島",
    "3  ▲戸田",
    "4  ▲三国",
    "5  　びわこ",
    "6  ▲鳴門",
    "7  ▲下関",
    "8  　多摩川",
    "9  　若松",
    "10 　福岡",
    "11 　唐津",
    "12 ◎桐生",
    "13 　尼崎",
    "14 　芦屋",
    "15 　津",
    "16 　常滑",
    "17 　徳山",
    "18 ◎浜名湖",
    "19 　児島",
    "20 　丸亀",
    "21 ◎蒲郡",
    "22 ◎宮島",
    "23 ◎住之江",
    "24 ◎大村",
]

# 会場マーク: ▲=荒れやすい  ◎=イン強・荒れにくい
_VENUE_MARK = {
    "江戸川": "▲", "戸田": "▲", "平和島": "▲",
    "三国": "▲", "鳴門": "▲", "下関": "▲",
    "大村": "◎", "宮島": "◎", "住之江": "◎",
    "蒲郡": "◎", "浜名湖": "◎", "桐生": "◎",
}

def _marked_venue(venue_name: str) -> str:
    mark = _VENUE_MARK.get(venue_name, "")
    return f"{venue_name}{mark}" if mark else venue_name


def _expand_formation(formation_str: str) -> set:
    """
    "2-13-1356" → {"2-1-3", "2-1-5", "2-1-6", "2-3-1", "2-3-5", "2-3-6"}
    """
    parts = formation_str.split("-")
    if len(parts) != 3:
        return set()
    combos = set()
    for f in parts[0]:
        for s in parts[1]:
            if s == f:
                continue
            for t in parts[2]:
                if t == f or t == s:
                    continue
                combos.add(f"{f}-{s}-{t}")
    return combos


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


def _write_venue_ranking(spreadsheet, sheet) -> None:
    """日付別シートのM1:M25に24場荒れランキングを書き込む"""
    try:
        header = [["【荒れランキング】"]]
        rows = [[v] for v in _VENUE_RANKING]
        sheet.update("M1", header + rows, value_input_option="RAW")
        sid = sheet.id
        reqs = []
        # ヘッダー行: 薄グレー+太字
        reqs.append({"repeatCell": {
            "range": {"sheetId": sid, "startRowIndex": 0, "endRowIndex": 1,
                      "startColumnIndex": 12, "endColumnIndex": 13},
            "cell": {"userEnteredFormat": {
                "backgroundColor": {"red": 0.85, "green": 0.85, "blue": 0.85},
                "textFormat": {"bold": True, "fontSize": 9},
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat)",
        }})
        # 上位7場（荒れ色: 薄赤）
        reqs.append({"repeatCell": {
            "range": {"sheetId": sid, "startRowIndex": 1, "endRowIndex": 8,
                      "startColumnIndex": 12, "endColumnIndex": 13},
            "cell": {"userEnteredFormat": {
                "backgroundColor": {"red": 1.0, "green": 0.88, "blue": 0.88},
                "textFormat": {"fontSize": 9},
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat)",
        }})
        # 中位（8-20位: 白）
        reqs.append({"repeatCell": {
            "range": {"sheetId": sid, "startRowIndex": 8, "endRowIndex": 21,
                      "startColumnIndex": 12, "endColumnIndex": 13},
            "cell": {"userEnteredFormat": {
                "backgroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
                "textFormat": {"fontSize": 9},
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat)",
        }})
        # 下位5場（イン強: 薄緑）
        reqs.append({"repeatCell": {
            "range": {"sheetId": sid, "startRowIndex": 21, "endRowIndex": 25,
                      "startColumnIndex": 12, "endColumnIndex": 13},
            "cell": {"userEnteredFormat": {
                "backgroundColor": {"red": 0.85, "green": 0.95, "blue": 0.85},
                "textFormat": {"fontSize": 9},
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat)",
        }})
        spreadsheet.batch_update({"requests": reqs})
    except Exception:
        pass


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
    headers = ["日付", "競艇場", "レース", "狙い", "買い目（3連単）", "オッズ", "1号逃げ率", "勝負推奨", "1号艇状態"]
    sheet.update("A1", [headers])
    _format_header(spreadsheet, sheet, num_cols=9)

    # データ行
    if not recommendations.empty:
        cols = ["date", "venue_name", "race_no", "tier", "combination", "odds", "odds_source", "bet_label", "boat1_risk"]
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
        sheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=13)
        headers = ["日付", "競艇場", "レース", "狙い", "買い目（3連単）",
                   "オッズ", "イン逃げ率", "本日レース数", "勝負推奨", "荒れPT", "荒れ条件", "1号艇状態"]
        sheet.update("A1", [headers])
        _format_header(spreadsheet, sheet, num_cols=12)
        _write_venue_ranking(spreadsheet, sheet)

    cols = ["date", "venue_name", "race_no", "confidence", "combination",
            "odds", "odds_source"]
    values = [row.get(c, "-") for c in cols]
    values[1] = _marked_venue(str(values[1]))
    # confidence がない場合は tier にフォールバック
    if not values[3] or values[3] == "-":
        values[3] = row.get("tier", "-")
    values.append(race_count if race_count is not None else "-")
    values.append(row.get("bet_label", ""))
    values.append(row.get("arare_score", ""))
    values.append(row.get("arare_reasons", ""))
    values.append(row.get("boat1_risk", "-"))
    for attempt in range(4):
        try:
            sheet.append_row(values, value_input_option="RAW")
            break
        except gspread.exceptions.APIError as e:
            if attempt == 3:
                raise
            wait = 2 ** attempt
            print(f"  [WARN] Sheets API エラー ({e}) → {wait}秒後にリトライ ({attempt+1}/4)")
            time.sleep(wait)

    # 行全体の色 + 勝負推奨色 + エッジ色を1回のbatch_updateで適用
    for attempt in range(4):
        try:
            last_row = len(sheet.get_all_values())
            break
        except gspread.exceptions.APIError as e:
            if attempt == 3:
                last_row = None
            else:
                wait = 2 ** attempt
                print(f"  [WARN] Sheets API エラー ({e}) → {wait}秒後にリトライ ({attempt+1}/4)")
                time.sleep(wait)
    if last_row is None:
        return  # 色付けは諦めるがデータは書き込み済み
    try:
        sid = sheet.id
        is_skip   = row.get("bet_label", "") in ("見送り", "")

        if is_skip:
            row_bg = {"red": 0.91, "green": 0.91, "blue": 0.91}
        else:
            row_bg = _VENUE_BG_COLORS.get(str(row.get("venue_name", "")), _DEFAULT_BG)

        reqs = [{"repeatCell": {
            "range": {"sheetId": sid, "startRowIndex": last_row - 1, "endRowIndex": last_row,
                      "startColumnIndex": 0, "endColumnIndex": 12},
            "cell": {"userEnteredFormat": {"backgroundColor": row_bg}},
            "fields": "userEnteredFormat.backgroundColor",
        }}]

        if not is_skip:
            bet_label = row.get("bet_label", "")
            if bet_label == "暴れ熊(強)":
                _ab_bg = {"red": 0.72, "green": 0.07, "blue": 0.07}  # 深紅
            elif bet_label == "暴れ熊(中)":
                _ab_bg = {"red": 0.85, "green": 0.33, "blue": 0.10}  # オレンジ赤
            elif bet_label == "暴れ熊(弱)":
                _ab_bg = {"red": 0.90, "green": 0.55, "blue": 0.30}  # 薄オレンジ
            else:
                _ab_bg = None
            if _ab_bg:
                reqs.append({"repeatCell": {
                    "range": {"sheetId": sid, "startRowIndex": last_row - 1, "endRowIndex": last_row,
                              "startColumnIndex": 8, "endColumnIndex": 9},
                    "cell": {"userEnteredFormat": {
                        "backgroundColor": _ab_bg,
                        "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
                    }},
                    "fields": "userEnteredFormat(backgroundColor,textFormat)",
                }})


        # イン逃げ率（G列=index6）: 65%以上で薄赤、50%未満で薄青
        nigerate_src = row.get("odds_source", "")
        m_nig = re.search(r'(\d+)%', str(nigerate_src))
        if m_nig:
            nig_val = int(m_nig.group(1))
            if nig_val > 57:
                nig_bg = {"red": 1.0, "green": 0.80, "blue": 0.80}
            elif nig_val < 50:
                nig_bg = {"red": 0.80, "green": 0.90, "blue": 1.0}
            else:
                nig_bg = None
            if nig_bg:
                reqs.append({"repeatCell": {
                    "range": {"sheetId": sid, "startRowIndex": last_row - 1, "endRowIndex": last_row,
                              "startColumnIndex": 6, "endColumnIndex": 7},
                    "cell": {"userEnteredFormat": {"backgroundColor": nig_bg}},
                    "fields": "userEnteredFormat.backgroundColor",
                }})

        # 1号艇状態（L列=index11）: フラグ数で色分け
        boat1_risk = row.get("boat1_risk", "-")
        if boat1_risk and boat1_risk != "-":
            flag_count = len(boat1_risk.split(" / "))
            risk_bg = ({"red": 1.0, "green": 0.80, "blue": 0.80} if flag_count >= 2
                       else {"red": 1.0, "green": 0.98, "blue": 0.80})
            reqs.append({"repeatCell": {
                "range": {"sheetId": sid, "startRowIndex": last_row - 1, "endRowIndex": last_row,
                          "startColumnIndex": 11, "endColumnIndex": 12},
                "cell": {"userEnteredFormat": {"backgroundColor": risk_bg}},
                "fields": "userEnteredFormat.backgroundColor",
            }})

        spreadsheet.batch_update({"requests": reqs})
    except Exception:
        pass


def _color_result_row(spreadsheet, sheet, row_no: int, venue_name: str, hit: str,
                      num_cols: int = 13, hit_col_idx: int = 8) -> None:
    """成績シートの1行に会場色＋的中色をリアルタイムで適用する"""
    try:
        sid = sheet.id
        bg = _VENUE_BG_COLORS.get(venue_name, _DEFAULT_BG)
        requests = [
            {"repeatCell": {
                "range": {"sheetId": sid, "startRowIndex": row_no - 1, "endRowIndex": row_no,
                          "startColumnIndex": 0, "endColumnIndex": num_cols},
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
                          "startColumnIndex": hit_col_idx, "endColumnIndex": hit_col_idx + 1},
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
    pred_rows_override: list = None,
    race_count: int = None,
) -> None:
    """
    RESULT_SHEETに1レース分の結果を記録・的中判定を更新する

    成績シートの列構成:
    日付 | 競艇場 | レース | 狙い | 予想買い目 | 的中確率 | 期待回収率 |
    実際の結果 | 実際の払戻 | 的中 | 収支
    """
    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    RESULT_HEADERS = ["日付", "競艇場", "レース", "狙い", "予想買い目",
                      "実際の結果", "イン逃げ率", "実際の払戻", "的中", "収支（円）", "本日レース数", "勝負推奨", "荒れPT"]
    try:
        r_sheet = spreadsheet.worksheet(RESULT_SHEET)
        if not r_sheet.get_all_values():
            r_sheet.update("A1", [RESULT_HEADERS])
            _format_header(spreadsheet, r_sheet, num_cols=13)
    except gspread.WorksheetNotFound:
        r_sheet = spreadsheet.add_worksheet(title=RESULT_SHEET, rows=2000, cols=13)
        r_sheet.update("A1", [RESULT_HEADERS])
        _format_header(spreadsheet, r_sheet, num_cols=13)

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

    # 該当レースの予想行を抽出（見送り含む全行）
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

    marked_venue = _marked_venue(venue_name)

    if not race_preds:
        r_sheet.append_row(
            [date, marked_venue, race_no, "-", "（予想なし）",
             actual_combination, "-", actual_payout, "-", 0, rc, "", ""],
            value_input_option="RAW"
        )
        _color_result_row(spreadsheet, r_sheet, len(r_sheet.get_all_values()), venue_name, "-",
                          num_cols=13, hit_col_idx=8)
        return

    for pred in race_preds:
        combination = pred.get("買い目（3連単）", "")
        _tier_raw = str(pred.get("狙い", "-"))
        # 狙い列は "小熊(外軸:3号/差し)[ET○ST○]" 形式の場合があるためベース名を抽出
        tier = next((t for t in ("小熊", "大熊", "神熊") if _tier_raw.startswith(t)), _tier_raw)
        bet_label = pred.get("勝負推奨", "")
        arare_pt = pred.get("荒れPT", "")
        nigerate_val = pred.get("イン逃げ率", pred.get("オッズ", "-"))  # 日付シートからイン逃げ率取得
        hit    = "○" if combination == actual_combination else "×"
        payout = actual_payout if hit == "○" else 0
        profit = payout - 100

        r_sheet.append_row(
            [date, marked_venue, race_no, tier, combination,
             actual_combination, nigerate_val, actual_payout, hit, profit, rc, bet_label, arare_pt],
            value_input_option="RAW"
        )
        _color_result_row(spreadsheet, r_sheet, len(r_sheet.get_all_values()), venue_name, hit,
                          num_cols=13, hit_col_idx=8)

    print(f"[OK] 成績記録: {venue_name} {race_no}R 結果={actual_combination} 払戻={actual_payout}円")


def apply_colors_to_results_sheet(
    spreadsheet_id: str,
    credentials_path: str = None,
) -> None:
    """成績シートの全行に競艇場カラーと的中色を一括適用する"""
    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    try:
        r_sheet = spreadsheet.worksheet(RESULT_SHEET)
    except gspread.WorksheetNotFound:
        return

    all_rows = r_sheet.get_all_values()
    if len(all_rows) <= 1:
        return

    fmt_requests = []
    sheet_id = r_sheet.id

    for i, row in enumerate(all_rows[1:], start=2):
        venue_name = row[1] if len(row) > 1 else ""
        hit = row[8] if len(row) > 8 else ""  # 成績18: 的中はindex8

        bg = _VENUE_BG_COLORS.get(venue_name, _DEFAULT_BG)

        fmt_requests.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": i - 1,
                    "endRowIndex": i,
                    "startColumnIndex": 0,
                    "endColumnIndex": 13,
                },
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


def _compute_tier_stats(records: list, tier_check) -> dict:
    """指定tierのレコードから集計統計を計算する"""
    total_bets = 0
    total_return = 0
    predicted_race_keys: set = set()
    hit_race_keys: set = set()
    daily: dict = {}

    for rec in records:
        combination = rec.get("予想買い目", "")
        if combination in ("", "（予想なし）", "見送り", "-"):
            continue
        if not tier_check(rec):
            continue

        d = str(rec.get("日付", ""))
        v = str(rec.get("競艇場", ""))
        rn = str(rec.get("レース", ""))
        race_key = (d, v, rn)
        predicted_race_keys.add(race_key)

        if d not in daily:
            daily[d] = {"bets": 0, "ret": 0, "race_keys": set(), "hit_race_keys": set()}
        tier_name = str(rec.get("狙い", ""))
        bet_amount = 100
        total_bets += bet_amount
        daily[d]["bets"] += bet_amount
        daily[d]["race_keys"].add(race_key)

        if rec.get("的中", "") == "○":
            hit_race_keys.add(race_key)
            daily[d]["hit_race_keys"].add(race_key)
            try:
                payout = int(str(rec.get("実際の払戻", 0)).replace(",", ""))
                total_return += payout
                daily[d]["ret"] += payout
            except (ValueError, TypeError):
                pass

    return {
        "total_bets": total_bets,
        "total_return": total_return,
        "pred_races": len(predicted_race_keys),
        "hit_races": len(hit_race_keys),
        "daily": daily,
    }


def _build_summary_cf_rules(sid: int, full_range: dict, profit_range: dict) -> list:
    """サマリーシート用の条件付き書式ルール（内容ベースでレイアウト崩れなし）"""
    return [
        # 本命セクションヘッダー（■ + 本命 を含む行 → 緑）
        {
            "ranges": [full_range],
            "booleanRule": {
                "condition": {
                    "type": "CUSTOM_FORMULA",
                    "values": [{"userEnteredValue": '=AND(LEFT($A1,1)="■",ISNUMBER(FIND("本命",$A1)))'}],
                },
                "format": {
                    "backgroundColor": {"red": 0.15, "green": 0.55, "blue": 0.25},
                    "textFormat": {
                        "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
                        "bold": True,
                    },
                },
            },
        },
        # 小穴セクションヘッダー（■ + 小穴 を含む行 → 青）
        {
            "ranges": [full_range],
            "booleanRule": {
                "condition": {
                    "type": "CUSTOM_FORMULA",
                    "values": [{"userEnteredValue": '=AND(LEFT($A1,1)="■",ISNUMBER(FIND("小穴",$A1)))'}],
                },
                "format": {
                    "backgroundColor": {"red": 0.18, "green": 0.36, "blue": 0.72},
                    "textFormat": {
                        "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
                        "bold": True,
                    },
                },
            },
        },
        # 大穴セクションヘッダー（■ + 大穴 を含む行 → 赤）
        {
            "ranges": [full_range],
            "booleanRule": {
                "condition": {
                    "type": "CUSTOM_FORMULA",
                    "values": [{"userEnteredValue": '=AND(LEFT($A1,1)="■",ISNUMBER(FIND("大穴",$A1)))'}],
                },
                "format": {
                    "backgroundColor": {"red": 0.72, "green": 0.18, "blue": 0.18},
                    "textFormat": {
                        "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
                        "bold": True,
                    },
                },
            },
        },
        # 列ヘッダー行（A列が"予想点数"または"日付" → グレー）
        {
            "ranges": [full_range],
            "booleanRule": {
                "condition": {
                    "type": "CUSTOM_FORMULA",
                    "values": [{"userEnteredValue": '=OR($A1="予想点数",$A1="日付")'}],
                },
                "format": {
                    "backgroundColor": {"red": 0.85, "green": 0.85, "blue": 0.85},
                    "textFormat": {"bold": True},
                },
            },
        },
        # 収支がプラス → 緑（G列）
        {
            "ranges": [profit_range],
            "booleanRule": {
                "condition": {"type": "NUMBER_GREATER", "values": [{"userEnteredValue": "0"}]},
                "format": {
                    "backgroundColor": {"red": 0.8, "green": 0.95, "blue": 0.8},
                    "textFormat": {"bold": True},
                },
            },
        },
        # 収支がマイナス → 赤（G列）
        {
            "ranges": [profit_range],
            "booleanRule": {
                "condition": {"type": "NUMBER_LESS", "values": [{"userEnteredValue": "0"}]},
                "format": {
                    "backgroundColor": {"red": 0.95, "green": 0.8, "blue": 0.8},
                    "textFormat": {"bold": True},
                },
            },
        },
    ]


def update_summary_sheet(
    spreadsheet_id: str,
    credentials_path: str = None,
) -> None:
    """
    RESULT_SHEETを集計してSUMMARY_SHEETを更新する
    神熱（全ティア合計）/ 見送り（全ティア合計）の2セクション
    """
    client = get_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    try:
        r_sheet = spreadsheet.worksheet(RESULT_SHEET)
    except gspread.WorksheetNotFound:
        print(f"[WARN] {RESULT_SHEET}シートが見つかりません")
        return
    try:
        records = _retry_get_records(r_sheet)
    except Exception:
        print(f"[WARN] {RESULT_SHEET}シートの読み込みに失敗しました（APIエラー）")
        return

    if not records:
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    NUM_COLS = 12

    def _r(*args):
        lst = list(args)
        return lst + [""] * (NUM_COLS - len(lst))

    def _pt_tier_stats(tier_name, pt_filter=None, label_filter=None):
        """tier_name: 狙い列. label_filter: None=全, '弱'/'中'/'強'=暴れ熊ラベル絞り"""
        race_keys: set = set()
        hit_race_keys: set = set()
        total_bets = 0
        total_ret = 0
        daily: dict = {}
        for rec in records:
            tier_val = str(rec.get("狙い", ""))
            if tier_val != tier_name:
                continue
            combo = str(rec.get("予想買い目", ""))
            if combo in ("", "（予想なし）", "見送り", "-"):
                continue
            _bl = str(rec.get("勝負推奨", ""))
            if _bl == "見送り":
                continue
            if label_filter is not None:
                if _bl != f"暴れ熊({label_filter})":
                    continue
            d  = str(rec.get("日付", ""))
            v  = str(rec.get("競艇場", ""))
            rn = str(rec.get("レース", ""))
            race_key = (d, v, rn)
            race_keys.add(race_key)
            bet = 100
            total_bets += bet
            if d not in daily:
                daily[d] = {"bets": 0, "ret": 0, "race_keys": set(), "hit_race_keys": set()}
            daily[d]["bets"] += bet
            daily[d]["race_keys"].add(race_key)
            if rec.get("的中", "") == "○":
                hit_race_keys.add(race_key)
                daily[d]["hit_race_keys"].add(race_key)
                try:
                    pay = int(str(rec.get("実際の払戻", 0)).replace(",", ""))
                    total_ret += pay
                    daily[d]["ret"] += pay
                except (ValueError, TypeError):
                    pass
        return {"race_keys": race_keys, "hit_race_keys": hit_race_keys,
                "bets": total_bets, "ret": total_ret, "daily": daily}

    def _fmt(s):
        rc   = len(s["race_keys"])
        hc   = len(s["hit_race_keys"])
        bets = s["bets"]
        ret  = s["ret"]
        hitr = f"{hc/rc*100:.1f}%" if rc > 0 else "0.0%"
        roi  = f"{ret/bets*100:.1f}%" if bets > 0 else "0.0%"
        pft  = ret - bets
        ivl  = f"{rc/hc:.1f}回に1回" if hc > 0 else "未的中"
        return rc, hc, hitr, ivl, ret, roi, pft

    def _write_pt_section(title, s):
        rc, hc, hitr, ivl, ret, roi, pft = _fmt(s)
        rows.append(_r(f"▶ {title}  全期間合計"))
        rows.append(_r("予想R数", "的中数", "的中率", "間隔", "総払戻", "回収率", "収支"))
        rows.append(_r(rc, hc, hitr, ivl, f"¥{ret:,}", roi, pft))
        rows.append(_r())
        rows.append(_r(f"▶ {title}  日付別内訳"))
        rows.append(_r("日付", "予想R数", "的中数", "的中率", "払戻合計", "収支"))
        for d in sorted(s["daily"].keys()):
            dd  = s["daily"][d]
            drc = len(dd["race_keys"])
            dhc = len(dd["hit_race_keys"])
            dr  = dd["ret"]
            dh  = f"{dhc/drc*100:.1f}%" if drc > 0 else "0.0%"
            dp  = dr - dd["bets"]
            rows.append(_r(d, drc, dhc, dh, f"¥{dr:,}", dp))
        rows.append(_r())

    rows: list = []
    rows.append(_r("【予想成績サマリー】"))
    rows.append(_r("集計日時", now))
    rows.append(_r())

    rows.append(_r("■ ティア別グループ比較"))
    rows.append(_r("グループ", "予想R数", "的中数", "的中率", "間隔", "総払戻", "回収率", "収支"))
    for grp, tn, lf in [
        ("小熊 弱", "小熊", "弱"), ("小熊 中", "小熊", "中"), ("小熊 強", "小熊", "強"),
        ("大熊 弱", "大熊", "弱"), ("大熊 中", "大熊", "中"), ("大熊 強", "大熊", "強"),
        ("神熊 弱", "神熊", "弱"), ("神熊 中", "神熊", "中"), ("神熊 強", "神熊", "強"),
    ]:
        s = _pt_tier_stats(tn, label_filter=lf)
        rc, hc, hitr, ivl, ret, roi, pft = _fmt(s)
        rows.append(_r(grp, rc, hc, hitr, ivl, f"¥{ret:,}", roi, pft))
    rows.append(_r())
    rows.append(_r())

    # ② 小熊セクション
    rows.append(_r("■ 小熊セクション（ラベル別）"))
    for lbl in ["弱", "中", "強"]:
        _write_pt_section(f"小熊 {lbl}", _pt_tier_stats("小熊", label_filter=lbl))
    rows.append(_r())

    # ③ 大熊セクション
    rows.append(_r("■ 大熊セクション（ラベル別）"))
    for lbl in ["弱", "中", "強"]:
        _write_pt_section(f"大熊 {lbl}", _pt_tier_stats("大熊", label_filter=lbl))
    rows.append(_r())

    # ④ 神熊セクション
    rows.append(_r("■ 神熊セクション（ラベル別）"))
    for lbl in ["弱", "中", "強"]:
        _write_pt_section(f"神熊 {lbl}", _pt_tier_stats("神熊", label_filter=lbl))
    rows.append(_r())

    # ④ 暴れ熊ラベル別集計（小熊+大熊合算）
    rows.append(_r("■ 暴れ熊集計（小熊＋大熊）"))
    rows.append(_r("ティア", "予想R数", "的中数", "的中率", "間隔", "総払戻", "回収率", "収支"))
    for _tn in ("小熊", "大熊"):
        def _abare_tier_stats(tn=_tn):
            _rk: set = set()
            _hrk: set = set()
            _tb = 0
            _tr = 0
            _dl: dict = {}
            for rec in records:
                if str(rec.get("狙い", "")) != tn:
                    continue
                _bl = str(rec.get("勝負推奨", ""))
                if not _bl.startswith("暴れ熊"):
                    continue
                combo = str(rec.get("予想買い目", ""))
                if combo in ("", "（予想なし）", "見送り", "-"):
                    continue
                d = str(rec.get("日付", ""))
                v = str(rec.get("競艇場", ""))
                rn = str(rec.get("レース", ""))
                rk = (d, v, rn)
                _rk.add(rk)
                if d not in _dl:
                    _dl[d] = {"bets": 0, "ret": 0, "race_keys": set(), "hit_race_keys": set()}
                _tb += 100
                _dl[d]["bets"] += bet
                _dl[d]["race_keys"].add(rk)
                if rec.get("的中", "") == "○":
                    _hrk.add(rk)
                    _dl[d]["hit_race_keys"].add(rk)
                    try:
                        pay = int(str(rec.get("実際の払戻", 0)).replace(",", ""))
                        _tr += pay
                        _dl[d]["ret"] += pay
                    except (ValueError, TypeError):
                        pass
            return {"bets": _tb, "ret": _tr, "race_keys": _rk, "hit_race_keys": _hrk, "daily": _dl}
        s = _abare_tier_stats()
        rc, hc, hitr, ivl, ret, roi, pft = _fmt(s)
        rows.append(_r(f"暴れ熊/{_tn}", rc, hc, hitr, ivl, f"¥{ret:,}", roi, pft))
    rows.append(_r())


    # シートへ書き込み
    try:
        s_sheet = spreadsheet.worksheet(SUMMARY_SHEET)
    except gspread.WorksheetNotFound:
        s_sheet = spreadsheet.add_worksheet(title=SUMMARY_SHEET, rows=1500, cols=NUM_COLS)

    s_sheet.clear()
    # 書式（%フォーマット等）もリセットして収支欄の-375000%バグを防ぐ
    try:
        spreadsheet.batch_update({"requests": [{
            "updateCells": {
                "range": {"sheetId": s_sheet.id},
                "fields": "userEnteredFormat"
            }
        }]})
    except Exception:
        pass
    s_sheet.update("A1", rows, value_input_option="USER_ENTERED")

    sid = s_sheet.id
    max_rows = 1000
    full_range   = {"sheetId": sid, "startRowIndex": 0, "endRowIndex": max_rows,
                    "startColumnIndex": 0, "endColumnIndex": NUM_COLS}
    profit_range = {"sheetId": sid, "startRowIndex": 0, "endRowIndex": max_rows,
                    "startColumnIndex": 6, "endColumnIndex": 7}

    try:
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet.id}"
        response = spreadsheet.client.request(
            "GET", url, params={"fields": "sheets(properties.sheetId,conditionalFormats)"}
        )
        sheet_data = response.json()
        target = next(
            (s for s in sheet_data.get("sheets", []) if s["properties"]["sheetId"] == sid), {}
        )
        num_existing = len(target.get("conditionalFormats", []))
        reqs = []
        for i in range(num_existing - 1, -1, -1):
            reqs.append({"deleteConditionalFormatRule": {"sheetId": sid, "index": i}})
        for i, rule in enumerate(_build_summary_cf_rules(sid, full_range, profit_range)):
            reqs.append({"addConditionalFormatRule": {"rule": rule, "index": i}})
        if reqs:
            spreadsheet.batch_update({"requests": reqs})
    except Exception as e:
        print(f"[WARN] 条件付き書式設定エラー: {e}")

    ko_s   = _pt_tier_stats("小熊")
    ok_s   = _pt_tier_stats("大熊")
    ko_roi = f"{ko_s['ret']/ko_s['bets']*100:.1f}%" if ko_s['bets'] > 0 else "0.0%"
    ok_roi = f"{ok_s['ret']/ok_s['bets']*100:.1f}%" if ok_s['bets'] > 0 else "0.0%"
    print(
        f"[OK] {SUMMARY_SHEET}更新: "
        f"小熊 {len(ko_s['race_keys'])}R ROI={ko_roi} / "
        f"大熊 {len(ok_s['race_keys'])}R ROI={ok_roi}"
    )


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

    try:
        result_sheet = spreadsheet.worksheet(RESULT_SHEET)
        result_records = _retry_get_records(result_sheet)
    except gspread.WorksheetNotFound:
        print(f"[ERROR] {RESULT_SHEET}シートが見つかりません")
        return
    except Exception:
        print(f"[ERROR] {RESULT_SHEET}シートの読み込みに失敗しました")
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
        sheet = spreadsheet.worksheet(SUMMARY_SHEET)
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=SUMMARY_SHEET, rows=50, cols=5)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = [
        ["集計日時", now],
        ["総購入額", f"¥{roi_history['total_bet']:,}"],
        ["総払戻額", f"¥{roi_history['total_return']:,}"],
        ["実際の回収率", roi_history["roi_pct"]],
    ]
    sheet.update("A1", rows)
    print(f"[OK] サマリー書き込み完了: 回収率 {roi_history['roi_pct']}")
