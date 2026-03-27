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

    try:
        resp = requests.get(RESULT_URL, params=params, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"available": False, "error": str(e)}

    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_result(soup)


def _parse_result(soup: BeautifulSoup) -> dict:
    """HTMLから着順・払戻をパースする"""
    result = {"available": False}

    # 1〜3着の艇番を取得
    # 着順テーブルのパターン: 順位セルに "1", "2", "3" が入っている
    top3 = []
    tables = soup.find_all("table")

    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue
            # 順位が1〜3の行を探す
            rank_text = cells[0].get_text(strip=True) if cells else ""
            if rank_text in ("1", "2", "3"):
                # 艇番は2列目以降にある
                for cell in cells[1:4]:
                    boat_text = cell.get_text(strip=True)
                    if re.match(r"^[1-6]$", boat_text):
                        top3.append(int(boat_text))
                        break

    if len(top3) >= 3:
        result["combination"] = f"{top3[0]}-{top3[1]}-{top3[2]}"
        result["available"] = True

    # 3連単払戻を取得
    # 払戻テーブルに "3連単" または "３連単" が含まれる行
    text = soup.get_text()
    payout_match = re.search(
        r"[3３]連単[^\d]*([\d,]+)", text
    )
    if payout_match:
        try:
            result["payout"] = int(payout_match.group(1).replace(",", ""))
        except ValueError:
            result["payout"] = 0
    else:
        result["payout"] = 0

    return result
