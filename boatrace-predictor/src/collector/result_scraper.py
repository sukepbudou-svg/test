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
    """HTMLから着順・払戻をパースする（複数の方法を試みる）"""
    result = {"available": False, "combination": "", "payout": 0}

    # ── アプローチ1: 払戻テーブルから3連単の組み合わせと払戻を直接取得 ──
    # boatrace.jpの払戻テーブルは "3連単" 行に "X-Y-Z" と払戻額が含まれる
    text = soup.get_text(" ", strip=True).translate(_FW2HW)
    m = re.search(r"3連単\s*([1-6]-[1-6]-[1-6])\s*([\d,]+)", text)
    if m:
        result["combination"] = m.group(1)
        try:
            result["payout"] = int(m.group(2).replace(",", ""))
        except ValueError:
            result["payout"] = 0
        result["available"] = True
        return result

    # ── アプローチ2: テーブルセルを直接スキャン ──
    # 払戻テーブルの行から "3連単" セルを探す
    for row in soup.find_all("tr"):
        cells = [td.get_text(" ", strip=True).translate(_FW2HW) for td in row.find_all("td")]
        if not cells:
            continue
        row_text = " ".join(cells)
        if "3連単" in row_text:
            # 同じ行またはセル内に X-Y-Z パターンを探す
            combo_m = re.search(r"([1-6]-[1-6]-[1-6])", row_text)
            pay_m = re.search(r"([\d,]{3,})", row_text)
            if combo_m:
                result["combination"] = combo_m.group(1)
                result["available"] = True
            if pay_m:
                try:
                    result["payout"] = int(pay_m.group(1).replace(",", ""))
                except ValueError:
                    pass
            if result["available"]:
                return result

    # ── アプローチ3: 着順テーブルから1〜3着艇番を収集 ──
    # 全角・半角両対応で順位セルを探す
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
        # 払戻額をテキストから取得
        pay_m = re.search(r"3連単[^\d]*([\d,]+)", text)
        if pay_m:
            try:
                result["payout"] = int(pay_m.group(1).replace(",", ""))
            except ValueError:
                pass

    return result
