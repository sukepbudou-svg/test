"""
競艇公式データダウンローダー
対象: http://www1.mbrace.or.jp/od2/
  - 番組表 (B): 出走表・選手情報
  - 競走成績 (K): レース結果・払戻金
"""

import io
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

BASE_URL = "http://www1.mbrace.or.jp/od2"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def download_file(data_type: str, date: datetime) -> Path | None:
    """
    LZHファイルをダウンロードして保存する

    Args:
        data_type: 'B'（番組表）または 'K'（競走成績）
        date: 対象日付

    Returns:
        保存したファイルパス（失敗時はNone）
    """
    prefix = data_type.lower()
    yyyymm = date.strftime("%Y%m")
    yymmdd = date.strftime("%y%m%d")
    url = f"{BASE_URL}/{data_type}/{yyyymm}/{prefix}{yymmdd}.lzh"

    save_dir = DATA_DIR / data_type
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{prefix}{yymmdd}.lzh"

    if save_path.exists():
        print(f"[SKIP] 既存ファイル: {save_path.name}")
        return save_path

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            save_path.write_bytes(response.content)
            print(f"[OK] ダウンロード完了: {save_path.name}")
            return save_path
        else:
            print(f"[SKIP] HTTP {response.status_code}: {url}")
            return None
    except requests.RequestException as e:
        print(f"[ERROR] ダウンロード失敗: {url} - {e}")
        return None


def download_range(data_type: str, start_date: datetime, end_date: datetime, interval: float = 3.0) -> list[Path]:
    """
    期間指定でLZHファイルを一括ダウンロード

    Args:
        data_type: 'B'（番組表）または 'K'（競走成績）
        start_date: 開始日
        end_date: 終了日
        interval: アクセス間隔（秒）サーバー負荷対策

    Returns:
        ダウンロードしたファイルパスのリスト
    """
    results = []
    current = start_date
    while current <= end_date:
        path = download_file(data_type, current)
        if path:
            results.append(path)
        time.sleep(interval)
        current += timedelta(days=1)
    return results


def download_today() -> tuple[Path | None, Path | None]:
    """今日の番組表と昨日の競走成績をダウンロード"""
    today = datetime.now()
    yesterday = today - timedelta(days=1)

    program = download_file("B", today)
    result = download_file("K", yesterday)
    return program, result


def extract_lzh(lzh_path: Path) -> Path | None:
    """
    LZHファイルを解凍してテキストファイルを返す

    Args:
        lzh_path: LZHファイルパス

    Returns:
        解凍したテキストファイルパス（失敗時はNone）
    """
    try:
        from lhafile import LhaFile
        archive = LhaFile(str(lzh_path))
        info = archive.infolist()[0]
        content = archive.read(info.filename)
        txt_path = lzh_path.with_suffix(".txt")
        txt_path.write_bytes(content)
        print(f"[OK] 解凍完了: {txt_path.name}")
        return txt_path
    except Exception as e:
        print(f"[ERROR] 解凍失敗: {lzh_path} - {e}")
        return None


def extract_all(data_type: str) -> list[Path]:
    """指定タイプの未解凍LZHファイルをすべて解凍"""
    lzh_dir = DATA_DIR / data_type
    if not lzh_dir.exists():
        return []

    results = []
    for lzh_path in sorted(lzh_dir.glob("*.lzh")):
        txt_path = lzh_path.with_suffix(".txt")
        if txt_path.exists():
            continue
        path = extract_lzh(lzh_path)
        if path:
            results.append(path)
    return results
