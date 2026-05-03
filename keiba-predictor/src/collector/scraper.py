"""
JRA公式サイトスクレイパー
出馬表・レース結果・オッズを取得する
"""

import re
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import pandas as pd

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# JRA場コード → 場名
VENUE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}

# 馬場状態コード
TRACK_CONDITIONS = {"1": "良", "2": "稍重", "3": "重", "4": "不良"}

# レースグレード
GRADE_MAP = {"G1": 5, "G2": 4, "G3": 3, "OP": 2, "L": 2, "3勝": 1, "2勝": 1, "1勝": 1, "未勝利": 0, "新馬": 0}


def fetch_race_card(date: datetime, venue_code: str, race_no: int,
                    race_id: str = None, timeout: int = 15) -> dict:
    """
    JRA公式から1レース分の出馬表を取得する

    Returns:
        {
            "venue": "東京", "race_no": 11, "grade": "G1",
            "distance": 2400, "surface": "芝", "condition": "良",
            "weather": "晴", "horses": [
                {"horse_no": 1, "horse_name": "...", "jockey": "...",
                 "weight": 57.0, "horse_weight": 498, "weight_diff": +2,
                 "odds_win": 3.5, "popularity": 1,
                 "past_results": [...]}
            ]
        }
    """
    rid = race_id or _make_race_id(date, venue_code, race_no)
    url = (
        f"https://race.netkeiba.com/race/shutuba.html"
        f"?race_id={rid}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = "EUC-JP"
    except requests.RequestException as e:
        print(f"[WARN] 出馬表取得失敗 {venue_code}-R{race_no}: {e}")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_race_card(soup, date, venue_code, race_no)


def fetch_race_result(date: datetime, venue_code: str, race_no: int,
                      race_id: str = None, timeout: int = 15) -> dict:
    """
    JRA公式から1レース分の結果を取得する

    Returns:
        {
            "available": True,
            "winner": 5, "second": 3,
            "quinella": "3-5", "quinella_payout": 1240,
            "win_payout": 350,
        }
    """
    rid = race_id or _make_race_id(date, venue_code, race_no)
    url = (
        f"https://race.netkeiba.com/race/result.html"
        f"?race_id={rid}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = "EUC-JP"
    except requests.RequestException as e:
        print(f"[WARN] 結果取得失敗 {venue_code}-R{race_no}: {e}")
        return {"available": False}

    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_race_result(soup)


def fetch_today_schedule(date: datetime = None, timeout: int = 15) -> list[dict]:
    """
    本日の開催レース一覧を取得する

    Returns:
        [{"venue_code": "05", "venue": "東京", "race_no": 1, "scheduled_time": "10:00",
          "race_id": "202605050511"}, ...]
    """
    if date is None:
        date = datetime.now()

    date_str = date.strftime("%Y%m%d")

    # Yahoo競馬の静的HTMLから取得を試みる
    yahoo_url = f"https://keiba.yahoo.co.jp/race/list/{date_str}/"
    try:
        resp = requests.get(yahoo_url, headers=HEADERS, timeout=timeout)
        if resp.status_code == 200:
            resp.encoding = "utf-8"
            text = resp.text
            ids_found = set()
            for m in re.finditer(r'race_id=(\d{12})', text):
                ids_found.add(m.group(1))
            for m in re.finditer(r'/race/(?:denma|result|card)/(\d{12})', text):
                ids_found.add(m.group(1))
            if ids_found:
                valid = [rid for rid in sorted(ids_found)
                         if 1 <= int(rid[4:6]) <= 10 and 1 <= int(rid[10:12]) <= 12]
                if valid:
                    print(f"  [OK] Yahoo競馬から{len(valid)}レース発見")
                    return _build_schedule(valid, text, date)
    except requests.RequestException:
        pass

    # キャッシュ確認（同じ日は再探索しない）
    import json
    from pathlib import Path
    cache_path = Path(__file__).parent.parent.parent / "data" / f"schedule_{date_str}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"  [OK] キャッシュから{len(cached)}レース読み込み")
            return cached
        except Exception:
            pass

    # プローブ方式: 出馬表URLに直接アクセスして今日の日付を確認
    print("  [INFO] スケジュールページからIDが見つからないため出馬表を直接探索します...")
    result = _probe_today_races(date, timeout)
    if result:
        cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def fetch_odds_quinella(date: datetime, venue_code: str, race_no: int,
                        race_id: str = None, timeout: int = 15) -> dict:
    """
    馬連オッズを取得する

    Returns:
        {"1-2": 15.3, "1-3": 8.2, ...}
    """
    rid = race_id or _make_race_id(date, venue_code, race_no)

    # 方法1: 内部APIエンドポイント（JSONレスポンス）
    result = _fetch_odds_api(rid, odds_type="b4", timeout=timeout)
    if len(result) > 10:
        return result

    # 方法2: HTMLオッズページをパース
    url = f"https://race.netkeiba.com/odds/index.html?type=b4&race_id={rid}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = "EUC-JP"
        result = _parse_odds_html(resp.text, n_horses=18)
        if result:
            return result
    except requests.RequestException as e:
        print(f"[WARN] 馬連オッズ取得失敗 {venue_code}-R{race_no}: {e}")
    return {}


def fetch_odds_wide(date: datetime, venue_code: str, race_no: int,
                    race_id: str = None, timeout: int = 15) -> dict:
    """
    ワイドオッズを取得する（最小払戻ベース）

    Returns:
        {"1-2": 3.5, "1-3": 2.8, ...}
    """
    rid = race_id or _make_race_id(date, venue_code, race_no)

    # 方法1: 内部APIエンドポイント（JSONレスポンス）
    result = _fetch_odds_api(rid, odds_type="b5", timeout=timeout)
    if len(result) > 10:
        return result

    # 方法2: HTMLオッズページをパース
    url = f"https://race.netkeiba.com/odds/index.html?type=b5&race_id={rid}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = "EUC-JP"
        result = _parse_odds_html(resp.text, n_horses=18)
        if result:
            return result
    except requests.RequestException as e:
        print(f"[WARN] ワイドオッズ取得失敗 {venue_code}-R{race_no}: {e}")
    return {}


def fetch_jockey_stats(jockey_id: str, timeout: int = 15) -> dict:
    """
    騎手の成績統計を取得する（今年の勝率・連対率等）

    Returns:
        {"win_rate": 0.18, "top2_rate": 0.35, "top3_rate": 0.48}
    """
    url = f"https://db.netkeiba.com/jockey/{jockey_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = "EUC-JP"
    except requests.RequestException as e:
        print(f"[WARN] 騎手成績取得失敗 {jockey_id}: {e}")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_jockey_stats(soup)


def fetch_horse_past_results(horse_id: str, n: int = 5, timeout: int = 15) -> list[dict]:
    """
    馬の直近n走の成績を取得する

    Returns:
        [{"date": "2026-03-01", "venue": "阪神", "rank": 1,
          "distance": 2000, "surface": "芝", "condition": "良",
          "time": "1:58.2", "jockey": "川田将雅", "weight": 57.0}, ...]
    """
    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = "EUC-JP"
    except requests.RequestException as e:
        print(f"[WARN] 馬成績取得失敗 {horse_id}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    result = _parse_horse_results(soup, n)
    if not result:
        tables = soup.find_all("table")
        has_only_prof = all(
            "db_prof_table" in (t.get("class") or []) for t in tables
        )
        if not has_only_prof:
            table_classes = [t.get("class") for t in tables[:3]]
            print(f"  [WARN] 馬成績パース失敗 {horse_id}: テーブル数={len(tables)} classes={table_classes}")
    return result


def fetch_training_times(race_id: str, timeout: int = 15) -> dict:
    """
    追い切りタイム（最終追い切り）を取得する

    Returns:
        {horse_no: {"time_3f": 38.5, "time_1f": 12.1, "track": "ウッド"}}
    """
    url = f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = "EUC-JP"
    except requests.RequestException as e:
        print(f"[WARN] 追い切り取得失敗 {race_id}: {e}")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_training_times(soup)


# ─── 内部パース関数 ───

def _dump_horse_debug(soup: BeautifulSoup, horse_rows: list) -> None:
    """出馬表の最初の馬行HTMLをファイルに保存する（horse_id診断用・1回だけ書く）"""
    from pathlib import Path
    debug_path = Path(__file__).parent.parent.parent / "data" / "horse_debug.txt"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(debug_path), "w", encoding="utf-8", errors="replace") as f:
        f.write(f"horse_rows count: {len(horse_rows)}\n\n")
        if horse_rows:
            row = horse_rows[0]
            f.write("=== FIRST ROW HTML (first 2000 chars) ===\n")
            f.write(str(row)[:2000])
            f.write("\n\n=== ALL ANCHOR TAGS IN FIRST ROW ===\n")
            for a in row.find_all("a"):
                f.write(f"  href={str(a.get('href',''))!r:70}  text={a.get_text(strip=True)!r}\n")
            f.write("\n=== FIRST 6 TD CELLS ===\n")
            for i, td in enumerate(row.find_all("td")[:6]):
                f.write(f"  cells[{i}] class={td.get('class')}  text={td.get_text(strip=True)!r}\n")
        else:
            f.write("=== NO HorseList ROWS FOUND ===\n")
            f.write("=== ALL TR CLASSES IN PAGE ===\n")
            for tr in soup.find_all("tr")[:20]:
                f.write(f"  class={tr.get('class')}  text_start={tr.get_text(strip=True)[:40]!r}\n")
            f.write("\n=== PAGE HTML (first 3000 chars) ===\n")
            f.write(str(soup)[:3000])


def _make_race_id(date: datetime, venue_code: str, race_no: int) -> str:
    """レースIDを生成 例: 202605010111 (年+場コード+開催回+日+レース番号)"""
    # netkeiba形式: YYYYJJKKNN (年4桁+場2桁+開催回2桁+日2桁+レース2桁)
    # 開催回・日は番組表から取る必要があるが、スケジュール取得で補完
    year = date.strftime("%Y")
    return f"{year}{venue_code.zfill(2)}0101{str(race_no).zfill(2)}"


def _parse_race_card(soup: BeautifulSoup, date: datetime, venue_code: str, race_no: int) -> dict:
    """出馬表HTMLをパースしてdictを返す"""
    result = {
        "date": date.strftime("%Y-%m-%d"),
        "venue_code": venue_code,
        "venue": VENUE_CODES.get(venue_code, "不明"),
        "race_no": race_no,
        "grade": "一般",
        "distance": 0,
        "surface": "芝",
        "condition": "良",
        "weather": "晴",
        "horses": [],
    }

    # レース情報
    race_info = soup.find(class_=re.compile(r'RaceData|race_info', re.I))
    if race_info:
        text = race_info.get_text()
        # 距離・馬場
        m = re.search(r'(芝|ダート)\s*(\d{3,4})', text)
        if m:
            result["surface"] = m.group(1)
            result["distance"] = int(m.group(2))
        # グレード
        for g in ["G1", "G2", "G3", "L", "OP"]:
            if g in text:
                result["grade"] = g
                break
        # 馬場状態
        for cond in ["不良", "重", "稍重", "良"]:
            if cond in text:
                result["condition"] = cond
                break
        # 天候
        for w in ["雨", "曇", "晴"]:
            if w in text:
                result["weather"] = w
                break

    # 出走馬リスト
    horse_rows = soup.select("tr.HorseList, tr.Shutuba_HorseList")

    # デバッグ: 最初の馬行HTMLをファイルに保存（horse_idが取れない問題の診断用）
    _dump_horse_debug(soup, horse_rows)

    for row in horse_rows:
        cells = row.find_all("td")
        if len(cells) < 8:
            continue
        try:
            # 馬番: Umaban classのtd → なければ cells[1]（cells[0]は枠番）
            horse_no = None
            umaban_td = row.find("td", class_=re.compile(r'Umaban|umaban', re.I))
            if umaban_td:
                t = umaban_td.get_text(strip=True)
                if t.isdigit() and 1 <= int(t) <= 18:
                    horse_no = int(t)
            if horse_no is None:
                for cell in cells[1:3]:  # cells[1]が馬番（cells[0]は枠番）
                    t = cell.get_text(strip=True)
                    if t.isdigit() and 1 <= int(t) <= 18:
                        horse_no = int(t)
                        break
            if horse_no is None:
                continue

            # 馬名・馬ID（行内のすべてのaタグから積極的に探す）
            horse_name, horse_id = "", ""
            for a in row.find_all("a", href=True):
                href = a.get("href", "")
                m = re.search(r'/horse/(\d{6,})', href)
                if m:
                    horse_id = m.group(1)
                    horse_name = a.get("title") or a.get_text(strip=True)
                    break

            # 騎手・騎手ID
            jockey, jockey_id = "", ""
            jockey_el = row.find(class_=re.compile(r'Jockey|jockey', re.I))
            if jockey_el:
                jockey = jockey_el.get_text(strip=True)
            j_link = row.find("a", href=re.compile(r'/jockey/'))
            if j_link:
                if not jockey:
                    jockey = j_link.get_text(strip=True)
                m = re.search(r'/jockey/(?:result/recent/)?(\d+)', j_link.get("href", ""))
                jockey_id = m.group(1) if m else ""

            # 斤量・馬体重・増減・単勝オッズ
            weight, horse_weight, weight_diff, win_odds = 55.0, 480, 0, 0.0
            for cell in cells:
                t = cell.get_text(strip=True).replace(",", "")
                mw = re.match(r'(\d{3})\(([+-]?\d+)\)', t)
                if mw:
                    horse_weight = int(mw.group(1))
                    weight_diff = int(mw.group(2))
                    continue
                mv = re.match(r'^(\d+\.\d)$', t)
                if mv:
                    val = float(mv.group(1))
                    if 50.0 <= val <= 60.0:
                        weight = val          # 斤量
                    elif val >= 1.0 and win_odds == 0.0:
                        win_odds = val        # 単勝オッズ（最初の非斤量小数）

            result["horses"].append({
                "horse_no": horse_no,
                "horse_name": horse_name,
                "horse_id": horse_id,
                "jockey": jockey,
                "jockey_id": jockey_id,
                "weight": weight,
                "horse_weight": horse_weight,
                "weight_diff": weight_diff,
                "win_odds": win_odds,
            })
        except (ValueError, IndexError):
            continue

    return result


def _parse_race_result(soup: BeautifulSoup) -> dict:
    """レース結果HTMLをパースしてdictを返す"""
    result = {"available": False}

    # 着順テーブルを探す
    result_table = soup.find(class_=re.compile(r'RaceResult|race_result', re.I))
    if not result_table:
        result_table = soup.find("table", class_=re.compile(r'result', re.I))

    if not result_table:
        return result

    rows = result_table.find_all("tr")
    ranks = {}
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        rank_text = cells[0].get_text(strip=True)
        if rank_text in ("1", "2", "3"):
            horse_no_text = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            if horse_no_text.isdigit():
                ranks[int(rank_text)] = int(horse_no_text)

    if 1 not in ranks or 2 not in ranks:
        return result

    w = ranks[1]
    s = ranks[2]
    combo = f"{min(w,s)}-{max(w,s)}"

    # 払戻情報
    win_payout = 0
    quinella_payout = 0
    payout_table = soup.find(class_=re.compile(r'Payout|payout', re.I))
    if payout_table:
        text = payout_table.get_text()
        m = re.search(r'単勝.*?(\d[\d,]+)円', text)
        if m:
            win_payout = int(m.group(1).replace(",", ""))
        m = re.search(r'馬連.*?(\d[\d,]+)円', text)
        if m:
            quinella_payout = int(m.group(1).replace(",", ""))

    result.update({
        "available": True,
        "winner": w,
        "second": s,
        "quinella": combo,
        "win_payout": win_payout,
        "quinella_payout": quinella_payout,
        "ranks": ranks,
    })
    return result


def _probe_today_races(date: datetime, timeout: int = 10) -> list[dict]:
    """出馬表URLを直接探索して本日開催のrace_idを特定する"""
    year = date.strftime("%Y")
    # 今日の日付が出馬表ページに含まれる形式（Windows互換）
    m_val, d_val = date.month, date.day
    date_patterns = [
        f"{year}年{m_val}月{d_val}日",          # 2026年5月2日
        f"{year}年{m_val:02d}月{d_val:02d}日",  # 2026年05月02日
        date.strftime("%Y/%m/%d"),               # 2026/05/02
    ]

    # 5月上旬はkai=3〜5が多い。順に試す
    kai_order = [3, 4, 2, 5, 1, 6]
    nichi_order = list(range(1, 9))
    venues = ["05", "06", "07", "08", "09", "04", "03", "01", "02", "10"]

    found: list[dict] = []
    found_venues: set = set()

    for venue_code in venues:
        if venue_code in found_venues:
            continue
        for kai in kai_order:
            hit_nichi = None
            for nichi in nichi_order:
                probe_id = f"{year}{venue_code}{kai:02d}{nichi:02d}01"
                url = f"https://race.netkeiba.com/race/shutuba.html?race_id={probe_id}"
                try:
                    resp = requests.get(url, headers=HEADERS, timeout=timeout)
                    if resp.status_code != 200:
                        continue
                    resp.encoding = "EUC-JP"
                    text = resp.text
                    if any(p in text for p in date_patterns):
                        hit_nichi = nichi
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.3)

            if hit_nichi is not None:
                print(f"  [OK] {VENUE_CODES.get(venue_code,'不明')} kai={kai} nichi={hit_nichi} を発見")
                # 1Rの出馬表ページから発走時刻を取得
                r1_time = _extract_race_time(text)
                for race_no in range(1, 13):
                    # 取得できた1Rの時刻を基に以降のレースを推定（1レース約35分間隔）
                    if r1_time:
                        h, m = r1_time
                        total_min = h * 60 + m + (race_no - 1) * 35
                        stime = f"{total_min // 60:02d}:{total_min % 60:02d}"
                    else:
                        stime = None
                    found.append({
                        "race_id": f"{year}{venue_code}{kai:02d}{hit_nichi:02d}{race_no:02d}",
                        "venue_code": venue_code,
                        "venue": VENUE_CODES.get(venue_code, "不明"),
                        "race_no": race_no,
                        "scheduled_time": stime,
                        "date": date.strftime("%Y-%m-%d"),
                    })
                found_venues.add(venue_code)
                break  # 次の会場へ

    return sorted(found, key=lambda x: (x["venue_code"], x["race_no"]))


def _extract_race_time(html_text: str):
    """出馬表HTMLから発走予定時刻を (hour, minute) で返す。見つからない場合は None"""
    for pattern in [r'(\d{1,2}):(\d{2})発走', r'発走\D{0,5}(\d{1,2}):(\d{2})']:
        m = re.search(pattern, html_text)
        if m:
            h, mn = int(m.group(1)), int(m.group(2))
            if 8 <= h <= 18:
                return h, mn
    return None


def _build_schedule(race_ids: list, page_text: str, date: datetime) -> list[dict]:
    """race_idリストとページテキストからスケジュールdictリストを生成する"""
    schedule = []
    for rid in race_ids:
        venue_code = rid[4:6]
        race_no = int(rid[10:12])
        # 発走時刻をページテキストから取得（race_idの近くにある HH:MM 形式）
        scheduled_time = None
        idx = page_text.find(rid)
        if idx >= 0:
            snippet = page_text[max(0, idx - 200):idx + 200]
            m = re.search(r'(\d{1,2}):(\d{2})', snippet)
            if m:
                hh, mm = int(m.group(1)), int(m.group(2))
                if 8 <= hh <= 18:
                    scheduled_time = f"{hh:02d}:{mm:02d}"
        schedule.append({
            "race_id": rid,
            "venue_code": venue_code,
            "venue": VENUE_CODES.get(venue_code, "不明"),
            "race_no": race_no,
            "scheduled_time": scheduled_time,
            "date": date.strftime("%Y-%m-%d"),
        })
    return sorted(schedule, key=lambda x: (x["venue_code"], x["race_no"]))


def _parse_odds_html(html: str, n_horses: int = 18) -> dict:
    """
    race.netkeiba.comのオッズHTMLをパース。
    JavaScript変数・HTMLテーブル・テキストパターンの順で試みる。
    """
    odds = {}

    # 方法1: JavaScriptの変数/JSON埋め込みから抽出
    # 例: {"1":{"2":"15.3","3":"8.2",...},...}  や  [["1","2","15.3"],...]
    for pattern in [
        r'"(\d{1,2})"\s*:\s*\{([^}]+)\}',          # {"1": {"2": "15.3", ...}}
        r'\[(\d{1,2})\s*,\s*(\d{1,2})\s*,\s*"?([\d.]+)"?\]',  # [1, 2, "15.3"]
    ]:
        for m in re.finditer(pattern, html):
            try:
                if len(m.groups()) == 3:  # [h1, h2, odds] 形式
                    h1, h2, o = int(m.group(1)), int(m.group(2)), float(m.group(3))
                    if 1 <= h1 <= n_horses and 1 <= h2 <= n_horses and h1 != h2 and o > 1.0:
                        key = f"{min(h1,h2)}-{max(h1,h2)}"
                        odds[key] = o
                else:  # {"1": {"2": "15.3"}} 形式
                    h1 = int(m.group(1))
                    inner = m.group(2)
                    for m2 in re.finditer(r'"(\d{1,2})"\s*:\s*"?([\d.]+)"?', inner):
                        h2, o = int(m2.group(1)), float(m2.group(2))
                        if 1 <= h2 <= n_horses and h1 != h2 and o > 1.0:
                            key = f"{min(h1,h2)}-{max(h1,h2)}"
                            odds[key] = o
            except (ValueError, IndexError):
                continue
        if len(odds) > 10:
            return odds

    # 方法2: HTMLテーブルから抽出
    soup = BeautifulSoup(html, "html.parser")
    tables = (soup.find_all("table", id=re.compile(r'odds_', re.I)) or
              soup.find_all("table", class_=re.compile(r'Odds|odds', re.I)) or
              soup.find_all("table"))
    for table in tables:
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            nums, float_vals = [], []
            for cell in cells:
                t = cell.get_text(strip=True).replace(",", "")
                if re.match(r'^\d{1,2}$', t) and 1 <= int(t) <= n_horses:
                    nums.append(int(t))
                elif re.match(r'^\d+\.\d$', t) and float(t) > 1.0:
                    float_vals.append(float(t))
            if len(nums) >= 2 and float_vals:
                h1, h2 = min(nums[0], nums[1]), max(nums[0], nums[1])
                if h1 != h2:
                    odds[f"{h1}-{h2}"] = float_vals[0]
    if len(odds) > 10:
        return odds

    # 方法3: テキストから「X-Y 15.3」パターンを抽出
    for m in re.finditer(r'(\d{1,2})-(\d{1,2})[^\d]{1,8}?([\d]+\.[\d])', html):
        h1, h2, o = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if 1 <= h1 <= n_horses and 1 <= h2 <= n_horses and h1 != h2 and o > 1.0:
            key = f"{min(h1,h2)}-{max(h1,h2)}"
            if key not in odds:
                odds[key] = o

    return odds


def _fetch_odds_api(race_id: str, odds_type: str = "b4", timeout: int = 15) -> dict:
    """netkeiba内部JSONAPIからオッズを取得する"""
    urls = [
        f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type={odds_type}&action=init",
        f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type={odds_type}&action=update",
    ]
    for url in urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code != 200:
                continue
            try:
                data = resp.json()
            except ValueError:
                continue
            odds = {}
            # {"data": {"odds": {"1": {"2": "15.3", ...}, ...}}}
            raw_odds = (data.get("data") or {}).get("odds", {})
            if isinstance(raw_odds, dict):
                for h1_str, inner in raw_odds.items():
                    if not isinstance(inner, dict):
                        continue
                    try:
                        h1 = int(h1_str)
                    except ValueError:
                        continue
                    for h2_str, o_val in inner.items():
                        try:
                            h2 = int(h2_str)
                            o = float(o_val)
                            if h1 != h2 and o > 1.0:
                                key = f"{min(h1,h2)}-{max(h1,h2)}"
                                odds[key] = o
                        except (ValueError, TypeError):
                            continue
            if len(odds) > 5:
                return odds
        except requests.RequestException:
            pass
    return {}


def _parse_jockey_stats(soup: BeautifulSoup) -> dict:
    """騎手成績HTMLをパースして勝率等を返す"""
    stats = {"win_rate": 0.15, "top2_rate": 0.30, "top3_rate": 0.45}
    table = soup.find("table", class_=re.compile(r'nk_tb_common', re.I))
    if not table:
        return stats
    rows = table.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        # 今年の成績行（1着・2着・3着・着外・勝率・連対率等）
        if len(cells) >= 8:
            try:
                win_rate_text = cells[5].get_text(strip=True).replace("%", "")
                top2_rate_text = cells[6].get_text(strip=True).replace("%", "")
                top3_rate_text = cells[7].get_text(strip=True).replace("%", "")
                stats["win_rate"] = float(win_rate_text) / 100
                stats["top2_rate"] = float(top2_rate_text) / 100
                stats["top3_rate"] = float(top3_rate_text) / 100
                break
            except (ValueError, IndexError):
                continue
    return stats


def _parse_horse_results(soup: BeautifulSoup, n: int) -> list[dict]:
    """馬の過去成績HTMLをパースして直近n走を返す"""
    results = []
    table = (soup.find("table", class_=re.compile(r'db_h_race_results', re.I))
             or soup.find("table", class_=re.compile(r'nk_tb_common', re.I)))
    if not table:
        all_tables = soup.find_all("table")
        if all_tables:
            table = max(all_tables, key=lambda t: len(t.find_all("tr")))
    if not table:
        return results

    rows = table.find_all("tr")[1:]  # ヘッダー除外
    for row in rows[:n]:
        cells = row.find_all("td")
        if len(cells) < 8:
            continue
        try:
            # 着順: cells[11]が定番、なければ左から12列目までで数字を探す
            rank = 99
            if len(cells) > 11:
                t = cells[11].get_text(strip=True)
                if t.isdigit() and int(t) <= 28:
                    rank = int(t)
            if rank == 99:
                for cell in cells[:15]:
                    t = cell.get_text(strip=True)
                    if t.isdigit() and 1 <= int(t) <= 28:
                        rank = int(t)
                        break

            # コース（例: "芝1600", "ダ1400"）: 固定インデックスに依存せず内容で探す
            surface, distance, condition = "芝", 0, "良"
            for cell in cells:
                t = cell.get_text(strip=True)
                m = re.match(r'(芝|ダ|ダート)(\d{3,4})', t)
                if m:
                    surface = "芝" if m.group(1) == "芝" else "ダート"
                    distance = int(m.group(2))
                    break

            # 馬場状態: "良", "稍重", "重", "不良" のいずれかを持つセルを探す
            for cell in cells:
                t = cell.get_text(strip=True)
                if t in ("不良", "重", "稍重", "良"):
                    condition = t
                    break

            results.append({
                "rank": rank,
                "surface": surface,
                "distance": distance,
                "condition": condition,
            })
        except (ValueError, IndexError):
            continue

    return results


def _parse_training_times(soup: BeautifulSoup) -> dict:
    """追い切りページをパース: 馬番ごとに最終追い切りの3F/1Fタイムを返す"""
    result = {}

    # netkeiba追い切りページのテーブルを探す
    table = (soup.find("table", class_=re.compile(r'OikiTable|oikiri', re.I))
             or soup.find("table", id=re.compile(r'oikiri', re.I)))
    if not table:
        for t in soup.find_all("table"):
            if len(t.find_all("tr")) > 5:
                table = t
                break
    if not table:
        return result

    seen = set()
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) < 5:
            continue
        try:
            horse_no = None
            for cell in cells[:3]:
                t = cell.get_text(strip=True)
                if t.isdigit() and 1 <= int(t) <= 18:
                    horse_no = int(t)
                    break
            if horse_no is None or horse_no in seen:
                continue
            seen.add(horse_no)

            track = ""
            time_vals = []
            for cell in cells:
                t = cell.get_text(strip=True)
                for tk in ["ウッド", "坂路", "ポリ", "芝", "ダート"]:
                    if tk in t:
                        track = tk
                if re.match(r'^\d{2}\.\d$', t):
                    time_vals.append(float(t))

            time_3f = next((v for v in time_vals if 30 <= v <= 50), None)
            time_1f = next((v for v in time_vals if 10 <= v <= 20), None)

            if time_3f is not None or time_1f is not None:
                result[horse_no] = {"time_3f": time_3f, "time_1f": time_1f, "track": track}
        except (ValueError, IndexError):
            continue

    return result
