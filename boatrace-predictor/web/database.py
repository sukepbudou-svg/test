"""
PERRY AI - SQLiteデータベース操作
"""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "perry.db"


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                date TEXT NOT NULL,
                venue_name TEXT,
                race_no INTEGER,
                race_time TEXT,
                combination TEXT,
                odds TEXT,
                odds_value REAL DEFAULT 0,
                arare_score INTEGER DEFAULT 0,
                arare_reasons TEXT,
                bet_label TEXT,
                tier TEXT,
                prob TEXT,
                expected_roi TEXT,
                nigerate_str TEXT,
                boat1_risk TEXT,
                actual_combination TEXT,
                actual_payout INTEGER DEFAULT 0,
                is_hit INTEGER DEFAULT 0,
                result_recorded_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(date);
            CREATE INDEX IF NOT EXISTS idx_pred_venue_race ON predictions(date, venue_name, race_no);
        """)
        # 既存DBへのマイグレーション
        for col in ["race_time TEXT", "race_grade TEXT"]:
            try:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col}")
            except Exception:
                pass


def save_prediction(rec: dict):
    """予想1件をDBに保存（同じdateのvenue+race+comboが既存なら更新）"""
    init_db()
    date = str(rec.get("date", ""))
    venue = str(rec.get("venue_name", ""))
    race_no = int(rec.get("race_no", 0) or 0)
    combo = str(rec.get("combination", ""))

    odds_str = str(rec.get("odds", ""))
    try:
        odds_val = float(odds_str.replace("倍", "").replace("(履歴)", "").strip())
    except (ValueError, AttributeError):
        odds_val = 0.0

    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM predictions WHERE date=? AND venue_name=? AND race_no=? AND combination=?",
            (date, venue, race_no, combo)
        ).fetchone()
        if existing:
            return  # 重複は無視
        conn.execute("""
            INSERT INTO predictions
              (date, venue_name, race_no, race_time, race_grade, combination, odds, odds_value,
               arare_score, arare_reasons, bet_label, tier,
               prob, expected_roi, nigerate_str, boat1_risk)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            date, venue, race_no,
            str(rec.get("race_time", "")),
            str(rec.get("race_grade", "")),
            combo, odds_str, odds_val,
            int(rec.get("arare_score", 0) or 0),
            str(rec.get("arare_reasons", "")),
            str(rec.get("bet_label", "見送り")),
            str(rec.get("tier", "大熊")),
            str(rec.get("prob", "")),
            str(rec.get("expected_roi", "")),
            str(rec.get("nigerate_str", "")),
            str(rec.get("boat1_risk", "")),
        ))


def update_result(date: str, venue_name: str, race_no: int,
                  actual_combination: str, actual_payout: int):
    """結果をDBに反映"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, combination FROM predictions WHERE date=? AND venue_name=? AND race_no=?",
            (date, venue_name, race_no)
        ).fetchall()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for row in rows:
            is_hit = 1 if row["combination"] == actual_combination else 0
            conn.execute("""
                UPDATE predictions
                SET actual_combination=?, actual_payout=?, is_hit=?, result_recorded_at=?
                WHERE id=?
            """, (actual_combination, actual_payout, is_hit, now, row["id"]))


def get_today_predictions(date: str = None):
    """当日の予想一覧を取得（レース番号・会場順）"""
    init_db()
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM predictions
            WHERE date = ?
            ORDER BY venue_name, race_no, id
        """, (date,)).fetchall()
    return [dict(r) for r in rows]


def get_pt_stats():
    """PT帯別集計（全期間・レース単位）"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                arare_score,
                COUNT(*) as total,
                SUM(has_result) as result_count,
                SUM(is_hit_any) as hits,
                SUM(race_payout) as total_payout
            FROM (
                SELECT date, venue_name, race_no, arare_score,
                    MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result,
                    MAX(is_hit) as is_hit_any,
                    SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as race_payout
                FROM predictions
                GROUP BY date, venue_name, race_no
            )
            GROUP BY arare_score
            ORDER BY arare_score
        """).fetchall()
    return [dict(r) for r in rows]


def get_label_stats():
    """ラベル別集計（全期間・レース単位）"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                bet_label,
                COUNT(*) as total,
                SUM(has_result) as result_count,
                SUM(is_hit_any) as hits,
                SUM(race_payout) as total_payout
            FROM (
                SELECT date, venue_name, race_no,
                    MAX(bet_label) as bet_label,
                    MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result,
                    MAX(is_hit) as is_hit_any,
                    SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as race_payout
                FROM predictions
                GROUP BY date, venue_name, race_no
            )
            GROUP BY bet_label
            ORDER BY
                CASE bet_label
                    WHEN 'プチュン' THEN 1
                    WHEN '黒船熱' THEN 2
                    WHEN '見送り' THEN 3
                    ELSE 4
                END
        """).fetchall()
    return [dict(r) for r in rows]


def get_daily_summary():
    """日付別集計（最新30日・レース単位）"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                date,
                COUNT(*) as total,
                SUM(has_result) as result_count,
                SUM(is_hit_any) as hits,
                SUM(race_payout) as total_payout,
                SUM(CASE WHEN bet_label != '見送り' THEN 1 ELSE 0 END) as bet_count
            FROM (
                SELECT date, venue_name, race_no,
                    MAX(bet_label) as bet_label,
                    MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result,
                    MAX(is_hit) as is_hit_any,
                    SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as race_payout
                FROM predictions
                GROUP BY date, venue_name, race_no
            )
            GROUP BY date
            ORDER BY date DESC
            LIMIT 30
        """).fetchall()
    return [dict(r) for r in rows]


def get_venue_stats():
    """会場別集計（全期間・レース単位・参戦レースのみ）"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                venue_name,
                COUNT(*) as total,
                SUM(has_result) as result_count,
                SUM(is_hit_any) as hits,
                SUM(race_payout) as total_payout
            FROM (
                SELECT date, venue_name, race_no,
                    MAX(bet_label) as bet_label,
                    MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result,
                    MAX(is_hit) as is_hit_any,
                    SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as race_payout
                FROM predictions
                GROUP BY date, venue_name, race_no
            )
            GROUP BY venue_name
            ORDER BY total DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_grade_stats():
    """等級別集計（全期間・レース単位・参戦レースのみ）"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                COALESCE(NULLIF(race_grade,''), '不明') as race_grade,
                COUNT(*) as total,
                SUM(has_result) as result_count,
                SUM(is_hit_any) as hits,
                SUM(race_payout) as total_payout
            FROM (
                SELECT date, venue_name, race_no,
                    MAX(bet_label) as bet_label,
                    MAX(COALESCE(NULLIF(race_grade,''), '不明')) as race_grade,
                    MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result,
                    MAX(is_hit) as is_hit_any,
                    SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as race_payout
                FROM predictions
                GROUP BY date, venue_name, race_no
            )
            GROUP BY race_grade
            ORDER BY
                CASE race_grade
                    WHEN 'SG' THEN 1 WHEN 'G1' THEN 2 WHEN 'G2' THEN 3
                    WHEN 'G3' THEN 4 WHEN '一般' THEN 5 ELSE 6
                END
        """).fetchall()
    return [dict(r) for r in rows]


def get_consecutive_misses():
    """ラベル別の直近連続外れ数"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT id, bet_label, is_hit, result_recorded_at
            FROM predictions
            WHERE result_recorded_at IS NOT NULL
              AND bet_label != '見送り'
            ORDER BY id DESC
        """).fetchall()

    result = {}
    for lbl in ["プチュン", "黒船熱"]:
        streak = 0
        for r in rows:
            if r["bet_label"] != lbl:
                continue
            if r["is_hit"] == 1:
                break
            streak += 1
        result[lbl] = streak
    return result
