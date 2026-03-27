"""
リアルタイムオッズ取得モジュール
boatrace.jp PC版から3連単オッズをスクレイピングする
"""

import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# 3連単の全120組み合わせを boatrace.jp の表示順で生成
def _generate_trifecta_order() -> list[str]:
    combos = []
    for b1 in range(1, 7):
        for b2 in range(1, 7):
            if b2 == b1:
                continue
            for b3 in range(1, 7):
                if b3 == b1 or b3 == b2:
                    continue
                combos.append(f"{b1}-{b2}-{b3}")
    return combos


TRIFECTA_ORDER = _generate_trifecta_order()

ODDS_URL = "https://www.boatrace.jp/owpc/pc/race/odds3t"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


def fetch_odds(date: datetime, venue_code: str, race_no: int, timeout: int = 10) -> dict[str, float]:
    """
    指定レースの3連単オッズを取得する

    Args:
        date: レース日
        venue_code: 場コード（例: "01"〜"24"）
        race_no: レース番号（1〜12）
        timeout: HTTPタイムアウト秒数

    Returns:
        {"1-2-3": 3.5, "1-2-4": 12.0, ...} のdict。
        取得失敗時は空dict。
    """
    hd = date.strftime("%Y%m%d")
    params = {"rno": race_no, "jcd": venue_code.zfill(2), "hd": hd}

    for attempt in range(3):  # 最大3回リトライ
        try:
            resp = requests.get(ODDS_URL, params=params, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(3)
            else:
                print(f"[WARN] オッズ取得失敗 {venue_code}-R{race_no}: {e}")
                return {}

    soup = BeautifulSoup(resp.text, "html.parser")

    # .table1 クラスの2番目テーブルからオッズを取得
    tables = soup.select(".table1")
    if len(tables) < 2:
        # フォールバック: すべての oddsPoint を順に収集
        points = soup.select("td.oddsPoint")
    else:
        points = tables[1].select("td.oddsPoint")

    if len(points) != 120:
        # ページ構造が変わっている場合のフォールバック
        points = soup.select("td.oddsPoint")

    if len(points) != 120:
        print(f"[WARN] オッズ数が不正 {venue_code}-R{race_no}: {len(points)}件（期待値120）")
        if points:
            sample = [td.get_text(strip=True) for td in points[:5]]
            print(f"       先頭5件サンプル: {sample}")
        return {}

    odds_dict: dict[str, float] = {}
    for combo, td in zip(TRIFECTA_ORDER, points):
        text = td.get_text(strip=True).replace(",", "")
        try:
            odds_dict[combo] = float(text)
        except ValueError:
            # "欠場" 等の場合はスキップ
            pass

    return odds_dict


def fetch_odds_for_races(
    df_today,
    date: datetime = None,
    interval: float = 1.5,
) -> dict[tuple, dict[str, float]]:
    """
    本日の全レースのオッズを取得する

    Args:
        df_today: 番組表DataFrame（venue_code, race_no 列が必要）
        date: 日付（未指定の場合は今日）
        interval: リクエスト間隔（秒）

    Returns:
        {(venue_code, race_no): {"1-2-3": 3.5, ...}} のdict
    """
    if date is None:
        date = datetime.now()

    races = df_today[["venue_code", "race_no"]].drop_duplicates()
    all_odds: dict[tuple, dict[str, float]] = {}

    print(f"=== リアルタイムオッズ取得中: {len(races)}レース ===")
    for _, row in races.iterrows():
        venue_code = str(row["venue_code"]).zfill(2)
        race_no = int(row["race_no"])
        key = (venue_code, race_no)

        odds = fetch_odds(date, venue_code, race_no)
        if odds:
            all_odds[key] = odds
            print(f"  [OK] 場{venue_code} R{race_no}: {len(odds)}通りのオッズ取得")
        else:
            print(f"  [--] 場{venue_code} R{race_no}: オッズ未取得（历史データで代替）")

        time.sleep(interval)

    success = sum(1 for v in all_odds.values() if v)
    print(f"=== オッズ取得完了: {success}/{len(races)}レース成功 ===")
    return all_odds
