"""
レース結果スクレイピングモジュール
boatrace.jp から当日のレース結果（着順・払戻）を取得する
"""

import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

RESULT_URL = "https://www.boatrace.jp/owpc/pc/race/raceresult"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

_FW2HW = str.maketrans("０１２３４５６７８９", "0123456789")


def fetch_race_result(date: datetime, venue_code: str, race_no: int, timeout: int = 10) -> dict:
    """
    指定レースの結果を取得する

    Returns:
        {
            "combination": "2-1-3",     # 3連単の着順
            "payout": 3540,             # 3連単払戻額（円）
            "available": True,          # 結果取得できたか
        }
        取得失敗・未確定の場合は available=False
    """
    hd = date.strftime("%Y%m%d")
    params = {"rno": race_no, "jcd": venue_code.zfill(2), "hd": hd}

    for attempt in range(3):
        try:
            resp = requests.get(RESULT_URL, params=params, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(2)
            else:
                return {"available": False, "error": str(e)}

    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_result(soup)


def _parse_result(soup: BeautifulSoup) -> dict:
    """HTMLから着順・払戻をパースする"""
    result = {"available": False, "combination": "", "payout": 0}

    # ── アプローチ1: テーブルセル構造から「3連単」行を探す ──
    # boatrace.jpの払戻テーブルは各行に [舟券種類][組み合わせ][払戻額] の構造
    for row in soup.find_all("tr"):
        cells = [td.get_text(strip=True).translate(_FW2HW) for td in row.find_all("td")]
        if not cells:
            continue

        # いずれかのセルに "3連単" が含まれる行を探す
        row_joined = "".join(cells)
        if "3連単" not in row_joined:
            continue

        # 組み合わせ (X-Y-Z) を探す
        combo = None
        payout = 0
        for cell in cells:
            if re.match(r"^[1-6]-[1-6]-[1-6]$", cell):
                combo = cell
            # 払戻額は最低100円（3桁以上の数字）、カンマ区切りも対応
            elif re.match(r"^[\d,]{3,}$", cell) and not combo is None:
                try:
                    v = int(cell.replace(",", ""))
                    if v >= 100:  # 最低払戻は100円
                        payout = v
                except ValueError:
                    pass

        # 組み合わせとセル分離した払戻が取れた場合
        if combo:
            result["combination"] = combo
            result["available"] = True
            if payout:
                result["payout"] = payout
            # 払戻が同一行に見つからない場合は次のセルや行から取得を試みる
            if not payout:
                for cell in cells:
                    m = re.match(r"^[\d,]{3,}$", cell)
                    if m:
                        try:
                            v = int(cell.replace(",", ""))
                            if v >= 100:
                                result["payout"] = v
                                break
                        except ValueError:
                            pass
            return result

    # ── アプローチ2: ページ全文テキストから3連単行を探す ──
    # セルが結合されている場合のフォールバック
    text = soup.get_text(" ", strip=True).translate(_FW2HW)
    # 3連単の後ろに X-Y-Z パターン、さらに後ろに3桁以上の数字
    m = re.search(r"3連単\D{0,20}([1-6]-[1-6]-[1-6])\D{0,30}?([\d]{3,}[\d,]*)", text)
    if m:
        result["combination"] = m.group(1)
        result["available"] = True
        try:
            result["payout"] = int(m.group(2).replace(",", ""))
        except ValueError:
            pass
        return result

    # ── アプローチ3: 着順テーブルから1〜3着艇番を収集 ──
    top3 = []
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if not cells:
                continue
            rank_text = cells[0].get_text(strip=True).translate(_FW2HW)
            if rank_text in ("1", "2", "3"):
                for cell in cells[1:5]:
                    boat_text = cell.get_text(strip=True).translate(_FW2HW)
                    if re.match(r"^[1-6]$", boat_text):
                        top3.append(int(boat_text))
                        break

    if len(top3) >= 3:
        result["combination"] = f"{top3[0]}-{top3[1]}-{top3[2]}"
        result["available"] = True
        pay_m = re.search(r"3連単\D{0,30}([\d,]{4,})", text)
        if pay_m:
            try:
                result["payout"] = int(pay_m.group(1).replace(",", ""))
            except ValueError:
                pass

    return result
