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
        for col in ["race_time TEXT", "race_grade TEXT", "okuma_signal_count INTEGER DEFAULT 0",
                    "bet_type TEXT DEFAULT '3連単'",
                    "strategy_version TEXT",
                    "actual_payout_3ren INTEGER DEFAULT 0"]:
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
        odds_val = float(odds_str.replace("倍", "").replace("(履歴)", "").replace("(推定)", "").strip())
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
               prob, expected_roi, nigerate_str, boat1_risk, okuma_signal_count, bet_type,
               strategy_version)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            int(rec.get("okuma_signal_count", 0) or 0),
            str(rec.get("bet_type", "3連単")),
            "2",
        ))


def update_result(date: str, venue_name: str, race_no: int,
                  actual_combination: str, actual_payout: int,
                  actual_payout_2ren: int = 0):
    """結果をDBに反映（2連単と3連単で配当を別々に保存）"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, combination FROM predictions WHERE date=? AND venue_name=? AND race_no=?",
            (date, venue_name, race_no)
        ).fetchall()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for row in rows:
            combo = row["combination"]
            parts = combo.split("-")
            if len(parts) == 2:
                # 2連単: actual_combination の先頭2艇と一致すれば的中
                actual_prefix = "-".join(actual_combination.split("-")[:2]) if actual_combination else ""
                is_hit = 1 if combo == actual_prefix else 0
                pay = actual_payout_2ren  # 2連単の配当
            else:
                is_hit = 1 if combo == actual_combination else 0
                pay = actual_payout  # 3連単の配当
            conn.execute("""
                UPDATE predictions
                SET actual_combination=?, actual_payout=?, actual_payout_3ren=?, is_hit=?, result_recorded_at=?
                WHERE id=?
            """, (actual_combination, pay, actual_payout, is_hit, now, row["id"]))


def get_recent_activity(date: str = None, limit: int = 5):
    """当日の最新更新レース一覧（会場・レース単位、更新が新しい順）"""
    init_db()
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT venue_name, race_no, arare_score,
                   MAX(bet_label) as bet_label,
                   MAX(created_at) as created_at
            FROM predictions
            WHERE date = ?
            GROUP BY venue_name, race_no
            ORDER BY MAX(created_at) DESC
            LIMIT ?
        """, (date, limit)).fetchall()
    return [dict(r) for r in rows]


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
                SUM(race_payout) as total_payout,
                SUM(CASE WHEN has_result=1 THEN combo_count ELSE 0 END) as result_combos
            FROM (
                SELECT date, venue_name, race_no, arare_score,
                    MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result,
                    MAX(is_hit) as is_hit_any,
                    SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as race_payout,
                    COUNT(*) as combo_count
                FROM predictions
                WHERE strategy_version = '2'
                GROUP BY date, venue_name, race_no
            )
            GROUP BY arare_score
            ORDER BY arare_score
        """).fetchall()
    return [dict(r) for r in rows]


def get_label_stats():
    """ラベル×賭け種別集計（全期間・レース単位）"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                bet_label,
                COALESCE(NULLIF(bet_type,''), '3連単') as bet_type,
                COUNT(*) as total,
                SUM(has_result) as result_count,
                SUM(is_hit_any) as hits,
                SUM(race_payout) as total_payout,
                SUM(CASE WHEN has_result=1 THEN combo_count ELSE 0 END) as result_combos
            FROM (
                SELECT date, venue_name, race_no,
                    MAX(bet_label) as bet_label,
                    COALESCE(NULLIF(bet_type,''), '3連単') as bet_type,
                    MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result,
                    MAX(is_hit) as is_hit_any,
                    SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as race_payout,
                    COUNT(*) as combo_count
                FROM predictions
                WHERE strategy_version = '2'
                GROUP BY date, venue_name, race_no, COALESCE(NULLIF(bet_type,''), '3連単')
            )
            GROUP BY bet_label, bet_type
            ORDER BY
                CASE bet_label
                    WHEN 'プチュン' THEN 1
                    WHEN '黒船熱' THEN 2
                    WHEN '中穴' THEN 3
                    WHEN '見送り' THEN 4
                    ELSE 5
                END,
                CASE bet_type WHEN '3連単' THEN 1 WHEN '2連単' THEN 2 ELSE 3 END
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
                SUM(CASE WHEN has_result=1 THEN combo_count ELSE 0 END) as result_combos
            FROM (
                SELECT date, venue_name, race_no,
                    MAX(bet_label) as bet_label,
                    MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result,
                    MAX(is_hit) as is_hit_any,
                    SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as race_payout,
                    COUNT(*) as combo_count
                FROM predictions
                WHERE strategy_version = '2'
                GROUP BY date, venue_name, race_no
            )
            GROUP BY date
            ORDER BY date DESC
            LIMIT 30
        """).fetchall()
    return [dict(r) for r in rows]


def get_venue_detail(venue_name: str):
    """会場別ドリルダウン: PT帯別・等級別の内訳"""
    init_db()
    with get_conn() as conn:
        pt_rows = conn.execute("""
            SELECT
                arare_score,
                MAX(bet_label) as bet_label,
                COUNT(*) as total,
                SUM(has_result) as result_count,
                SUM(is_hit_any) as hits,
                SUM(race_payout) as total_payout,
                SUM(CASE WHEN has_result=1 THEN combo_count ELSE 0 END) as result_combos
            FROM (
                SELECT date, venue_name, race_no, arare_score,
                    MAX(bet_label) as bet_label,
                    MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result,
                    MAX(is_hit) as is_hit_any,
                    SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as race_payout,
                    COUNT(*) as combo_count
                FROM predictions
                WHERE venue_name = ? AND strategy_version = '2'
                GROUP BY date, venue_name, race_no
            )
            GROUP BY arare_score
            ORDER BY arare_score
        """, (venue_name,)).fetchall()

        grade_rows = conn.execute("""
            SELECT
                COALESCE(NULLIF(race_grade,''), '不明') as race_grade,
                COUNT(*) as total,
                SUM(has_result) as result_count,
                SUM(is_hit_any) as hits,
                SUM(race_payout) as total_payout,
                SUM(CASE WHEN has_result=1 THEN combo_count ELSE 0 END) as result_combos
            FROM (
                SELECT date, venue_name, race_no,
                    MAX(bet_label) as bet_label,
                    MAX(COALESCE(NULLIF(race_grade,''), '不明')) as race_grade,
                    MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result,
                    MAX(is_hit) as is_hit_any,
                    SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as race_payout,
                    COUNT(*) as combo_count
                FROM predictions
                WHERE venue_name = ? AND strategy_version = '2'
                GROUP BY date, venue_name, race_no
            )
            GROUP BY race_grade
            ORDER BY
                CASE race_grade
                    WHEN 'SG' THEN 1 WHEN 'G1' THEN 2 WHEN 'G2' THEN 3
                    WHEN 'G3' THEN 4 WHEN '一般' THEN 5 ELSE 6
                END
        """, (venue_name,)).fetchall()

    return {
        "venue_name": venue_name,
        "pt_breakdown": [dict(r) for r in pt_rows],
        "grade_breakdown": [dict(r) for r in grade_rows],
    }


def get_venue_stats():
    """会場別集計（全期間・レース単位）"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                venue_name,
                COUNT(*) as total,
                SUM(has_result) as result_count,
                SUM(is_hit_any) as hits,
                SUM(race_payout) as total_payout,
                SUM(CASE WHEN has_result=1 THEN combo_count ELSE 0 END) as result_combos
            FROM (
                SELECT date, venue_name, race_no,
                    MAX(bet_label) as bet_label,
                    MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result,
                    MAX(is_hit) as is_hit_any,
                    SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as race_payout,
                    COUNT(*) as combo_count
                FROM predictions
                WHERE strategy_version = '2'
                GROUP BY date, venue_name, race_no
            )
            GROUP BY venue_name
            ORDER BY total DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_grade_stats():
    """等級別集計（全期間・レース単位）"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                COALESCE(NULLIF(race_grade,''), '不明') as race_grade,
                COUNT(*) as total,
                SUM(has_result) as result_count,
                SUM(is_hit_any) as hits,
                SUM(race_payout) as total_payout,
                SUM(CASE WHEN has_result=1 THEN combo_count ELSE 0 END) as result_combos
            FROM (
                SELECT date, venue_name, race_no,
                    MAX(bet_label) as bet_label,
                    MAX(COALESCE(NULLIF(race_grade,''), '不明')) as race_grade,
                    MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result,
                    MAX(is_hit) as is_hit_any,
                    SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as race_payout,
                    COUNT(*) as combo_count
                FROM predictions
                WHERE strategy_version = '2'
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


def get_payout_distribution():
    """ラベル別・実結果3連単配当分布（神熱 vs 見送り比較用）"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                bet_label,
                COUNT(*) as total_races,
                SUM(CASE WHEN actual_payout_3ren > 0 THEN 1 ELSE 0 END) as has_result,
                SUM(CASE WHEN actual_payout_3ren > 0 AND actual_payout_3ren < 5000 THEN 1 ELSE 0 END) as under_50,
                SUM(CASE WHEN actual_payout_3ren >= 5000 AND actual_payout_3ren <= 25000 THEN 1 ELSE 0 END) as range_50_250,
                SUM(CASE WHEN actual_payout_3ren > 25000 THEN 1 ELSE 0 END) as over_250,
                ROUND(AVG(CASE WHEN actual_payout_3ren > 0 THEN actual_payout_3ren END)) as avg_payout_3ren,
                MAX(actual_payout_3ren) as max_payout_3ren
            FROM (
                SELECT date, venue_name, race_no,
                    MAX(bet_label) as bet_label,
                    MAX(actual_payout_3ren) as actual_payout_3ren
                FROM predictions
                WHERE strategy_version = '2'
                GROUP BY date, venue_name, race_no
            )
            GROUP BY bet_label
            ORDER BY
                CASE bet_label WHEN '神熱' THEN 1 ELSE 2 END
        """).fetchall()
    return [dict(r) for r in rows]


def get_hero_stats():
    """ヒーロー（1着）的中率分析：ラベル別にヒーロー的中と組み合わせ的中を集計"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                bet_label,
                COUNT(*) as total_races,
                SUM(has_result) as result_count,
                SUM(hero_correct) as hero_hits,
                SUM(is_hit_any) as combo_hits
            FROM (
                SELECT date, venue_name, race_no,
                    MAX(bet_label) as bet_label,
                    MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result,
                    MAX(CASE WHEN result_recorded_at IS NOT NULL
                              AND SUBSTR(combination, 1, 1) = SUBSTR(actual_combination, 1, 1)
                              THEN 1 ELSE 0 END) as hero_correct,
                    MAX(is_hit) as is_hit_any
                FROM predictions
                WHERE strategy_version = '2'
                GROUP BY date, venue_name, race_no
            )
            GROUP BY bet_label
            ORDER BY
                CASE bet_label
                    WHEN 'プチュン' THEN 1
                    WHEN '黒船熱' THEN 2
                    WHEN '中穴' THEN 3
                    WHEN '見送り' THEN 4
                    ELSE 5
                END
        """).fetchall()
    return [dict(r) for r in rows]


def get_pt_payout_stats():
    """荒れPT帯別の実際の配当分布（荒れPTの有効性検証用）"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                arare_score,
                COUNT(*) as race_count,
                SUM(has_result) as result_count,
                ROUND(AVG(CASE WHEN has_result = 1 THEN actual_payout END)) as avg_payout,
                MAX(actual_payout) as max_payout,
                SUM(CASE WHEN has_result = 1 AND actual_payout >= 10000 THEN 1 ELSE 0 END) as over_10k,
                SUM(CASE WHEN has_result = 1 AND actual_payout >= 30000 THEN 1 ELSE 0 END) as over_30k,
                SUM(CASE WHEN has_result = 1 AND actual_payout >= 100000 THEN 1 ELSE 0 END) as over_100k
            FROM (
                SELECT date, venue_name, race_no, arare_score,
                    MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result,
                    MAX(actual_payout) as actual_payout
                FROM predictions
                WHERE strategy_version = '2'
                GROUP BY date, venue_name, race_no
            )
            GROUP BY arare_score
            ORDER BY arare_score
        """).fetchall()
    return [dict(r) for r in rows]


def get_signal_stats():
    """大穴シグナル数別集計（arare_reasonsから動的計算・全期間）"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                sig,
                COUNT(*) as race_count,
                SUM(has_result) as result_count,
                SUM(is_hit_any) as hits,
                SUM(race_payout) as total_payout,
                SUM(CASE WHEN has_result=1 THEN combo_count ELSE 0 END) as result_combos,
                SUM(CASE WHEN has_result=1 AND actual_payout >= 8000 THEN 1 ELSE 0 END) as okuma_actual_count
            FROM (
                SELECT date, venue_name, race_no,
                    (
                        CASE WHEN MAX(arare_reasons) LIKE '%M低%'       THEN 1 ELSE 0 END +
                        CASE WHEN MAX(arare_reasons) LIKE '%展示最遅%'  THEN 1 ELSE 0 END +
                        CASE WHEN MAX(arare_reasons) LIKE '%A1選手%'    THEN 1 ELSE 0 END +
                        CASE WHEN MAX(arare_reasons) LIKE '%荒れ会場%'
                               OR MAX(arare_reasons) LIKE '%江戸川%'    THEN 1 ELSE 0 END +
                        CASE WHEN MAX(arare_reasons) LIKE '%前付け%'    THEN 1 ELSE 0 END +
                        CASE WHEN MAX(arare_reasons) LIKE '%1号ST遅%'   THEN 1 ELSE 0 END
                    ) as sig,
                    MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result,
                    MAX(is_hit) as is_hit_any,
                    MAX(actual_payout) as actual_payout,
                    SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as race_payout,
                    COUNT(*) as combo_count
                FROM predictions
                WHERE strategy_version = '2'
                GROUP BY date, venue_name, race_no
            )
            GROUP BY sig
            ORDER BY sig
        """).fetchall()
    return [dict(r) for r in rows]


def get_consecutive_misses():
    """ラベル別の直近連続外れ数（後方互換）"""
    return get_all_streaks()["label"]


def _calc_streaks(rows, key_fn):
    """id降順rowsからキー別の直近連続外れ数を計算"""
    result = {}
    done = set()
    for row in rows:
        key = key_fn(row)
        if key is None or key in done:
            continue
        if key not in result:
            result[key] = 0
        if row["is_hit"] == 1:
            done.add(key)
        else:
            result[key] += 1
    return result


def get_all_streaks():
    """ラベル・賭け種・PT・会場・等級別の直近連続外れ数を一括取得（参戦レースのみ・レース単位）"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT MAX(bet_label) as bet_label,
                   COALESCE(NULLIF(bet_type,''), '3連単') as bet_type,
                   MAX(arare_score) as arare_score,
                   MAX(venue_name) as venue_name,
                   COALESCE(NULLIF(MAX(race_grade),''), '不明') as race_grade,
                   MAX(is_hit) as is_hit
            FROM predictions
            WHERE result_recorded_at IS NOT NULL
              AND bet_label != '見送り'
              AND strategy_version = '2'
            GROUP BY date, venue_name, race_no, COALESCE(NULLIF(bet_type,''), '3連単')
            ORDER BY MAX(id) DESC
        """).fetchall()
    rows = [dict(r) for r in rows]
    return {
        "label": _calc_streaks(rows, lambda r: (r["bet_label"], r["bet_type"])),
        "pt":    _calc_streaks(rows, lambda r: r["arare_score"]),
        "venue": _calc_streaks(rows, lambda r: r["venue_name"]),
        "grade": _calc_streaks(rows, lambda r: r["race_grade"]),
    }
