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
    """HTMLから3連単の着順・払戻をパースする"""
    result = {"available": False, "combination": "", "payout": 0}

    # 「3連単」テキストを持つ <td> を探し、同じ行から組み合わせと払戻を取得
    # HTML構造: <td>3連単</td> <td>[numberSet1スパン]</td> <td><span class="is-payout1">¥6,990</span></td>
    for td in soup.find_all("td"):
        if td.get_text(strip=True).translate(_FW2HW) != "3連単":
            continue

        row = td.parent

        # 組み合わせ: numberSet1_number クラスのスパンから艇番を取得
        numbers = [
            sp.get_text(strip=True).translate(_FW2HW)
            for sp in row.find_all("span")
            if sp.get("class") and any("numberSet1_number" in c for c in sp.get("class", []))
        ]
        if len(numbers) >= 3:
            result["combination"] = f"{numbers[0]}-{numbers[1]}-{numbers[2]}"
            result["available"] = True

        # 払戻額: is-payout1 クラスのスパンから取得（¥6,990 → 6990）
        payout_span = row.find("span", class_="is-payout1")
        if payout_span:
            pay_text = re.sub(r"[^\d]", "", payout_span.get_text(strip=True))
            try:
                result["payout"] = int(pay_text)
            except ValueError:
                pass

        if result["available"]:
            return result

    return result
