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
                    "actual_payout_3ren INTEGER DEFAULT 0",
                    "day_race_no INTEGER"]:
            try:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col}")
            except Exception:
                pass


def sync_race_predictions(date: str, venue_name: str, race_no: int, combinations: list):
    """このレースの最新予想に含まれない、まだ結果が付いていない古い予想行を削除する
    （オッズ変動等で選出組み合わせが再計算ごとに変わり、買い目が際限なく積み上がるのを防ぐ）"""
    if not combinations:
        return
    init_db()
    race_no = int(race_no or 0)
    with get_conn() as conn:
        placeholders = ",".join("?" for _ in combinations)
        conn.execute(f"""
            DELETE FROM predictions
            WHERE date=? AND venue_name=? AND race_no=?
              AND result_recorded_at IS NULL
              AND combination NOT IN ({placeholders})
        """, (date, venue_name, race_no, *combinations))


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
               strategy_version, day_race_no)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            str(rec.get("strategy_version", "5")),
            rec.get("day_race_no"),
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


def get_recent_activity(date: str = None, limit: int = 8):
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
                WHERE strategy_version = '3'
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
                WHERE strategy_version = '3'
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
                WHERE strategy_version = '3'
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
                WHERE venue_name = ? AND strategy_version = '3'
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
                WHERE venue_name = ? AND strategy_version = '3'
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


def get_venue_okuma_ranking():
    """会場別の万舟(実配当≥1万円)回数・確率ランキング（レース単位）
    strategy_version='5'（今の予想コードでの保存分）以降のみを対象にする。日付ではなく
    バージョンで区切ることで、実行環境とローカルPCのタイムゾーンのズレに影響されず
    「今から」を正確に区切れる。今後レースが記録されるたびに集計対象が増えていく。"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT venue_name,
                   COUNT(*) as n_races,
                   SUM(CASE WHEN actual_payout >= 10000 THEN 1 ELSE 0 END) as n_okuma
            FROM (
                SELECT date, venue_name, race_no, MAX(actual_payout) as actual_payout
                FROM predictions
                WHERE bet_type='3連単' AND result_recorded_at IS NOT NULL AND strategy_version='5'
                GROUP BY date, venue_name, race_no
            )
            GROUP BY venue_name
        """).fetchall()
    result = []
    for r in rows:
        r = dict(r)
        n_races = r["n_races"] or 0
        n_okuma = r["n_okuma"] or 0
        result.append({
            "venue_name": r["venue_name"],
            "n_races": n_races,
            "n_okuma": n_okuma,
            "okuma_rate": round(100 * n_okuma / n_races, 1) if n_races else 0.0,
        })
    result.sort(key=lambda x: (-x["okuma_rate"], -x["n_okuma"]))
    return result


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
                WHERE strategy_version = '3'
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
                WHERE strategy_version = '3'
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


def get_daily_label_stats():
    """日付×ラベル×賭け種別集計（直近30日・レース単位）"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                date,
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
                WHERE strategy_version = '3'
                GROUP BY date, venue_name, race_no, COALESCE(NULLIF(bet_type,''), '3連単')
            )
            GROUP BY date, bet_label, bet_type
            ORDER BY date DESC,
                CASE bet_label WHEN '神熱' THEN 1 ELSE 2 END,
                CASE bet_type WHEN '3連単' THEN 1 WHEN '2連単' THEN 2 ELSE 3 END
            LIMIT 200
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
                WHERE strategy_version = '3'
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
                WHERE strategy_version = '3'
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
                WHERE strategy_version = '3'
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
                WHERE strategy_version = '3'
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
              AND strategy_version = '3'
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


# ══════════════════════════════════════════════════════════════
# 新PTスコア方式（strategy_version='5'・2026-08-25〜、荒れPT満点20点の簡略版）の集計
# PTスコアの配点は同じ日のうちに何度か変わっており（2ゲート方式→満点29点→満点20点）、
# バージョンごとに点数の意味が異なる。日付や時刻ではなく strategy_version='5' で
# 厳密に絞り込むことで、タイムゾーンのズレに影響されず「最終版のコードで保存された
# 予想から」を正確に集計対象にできる（次に保存される予想から即座に反映される）。
# ══════════════════════════════════════════════════════════════

def _pt_v4_race_level_sql(select_extra: str = "") -> str:
    """strategy_version='5'のレース単位集計サブクエリ（共通部分）"""
    return f"""
        SELECT date, venue_name, race_no,
               MAX(bet_label) as label,
               MAX(arare_score) as pt,
               COUNT(*) as n_bets,
               MAX(is_hit) as race_hit,
               MAX(actual_payout) as actual_payout,
               SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as race_return,
               MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result
               {select_extra}
        FROM predictions
        WHERE strategy_version='5' AND bet_type='3連単'
        GROUP BY date, venue_name, race_no
    """


def get_pt_score_stats():
    """新PTスコア方式: PT値ごとの成績集計
    賭け数(1点=100円)・的中率・平均配当率(ROI)・万舟率(実配当≥1万円のレース比率)・現在の連続不的中数
    """
    init_db()
    with get_conn() as conn:
        bet_rows = conn.execute(f"""
            SELECT arare_score as pt,
                   COUNT(*) as n_bets,
                   SUM(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as n_resulted,
                   SUM(is_hit) as n_hits,
                   SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as total_return
            FROM predictions
            WHERE strategy_version='5' AND bet_type='3連単'
            GROUP BY arare_score
        """).fetchall()

        race_rows = conn.execute(f"""
            SELECT pt,
                   COUNT(*) as n_races,
                   SUM(has_result) as n_races_resulted,
                   SUM(CASE WHEN has_result=1 AND actual_payout >= 10000 THEN 1 ELSE 0 END) as n_okuma,
                   SUM(CASE WHEN has_result=1 THEN race_hit ELSE 0 END) as n_race_hits,
                   SUM(CASE WHEN has_result=1 THEN actual_payout ELSE 0 END) as total_actual_payout
            FROM ({_pt_v4_race_level_sql()})
            GROUP BY pt
        """).fetchall()

        streak_rows = conn.execute(f"""
            SELECT MAX(arare_score) as pt, MAX(is_hit) as is_hit
            FROM predictions
            WHERE strategy_version='5' AND bet_type='3連単' AND result_recorded_at IS NOT NULL
            GROUP BY date, venue_name, race_no
            ORDER BY MAX(id) DESC
        """).fetchall()

    streaks = _calc_streaks([dict(r) for r in streak_rows], lambda r: r["pt"])
    race_by_pt = {r["pt"]: dict(r) for r in race_rows}

    result = []
    for row in bet_rows:
        row = dict(row)
        pt = row["pt"]
        rr = race_by_pt.get(pt, {})
        n_resulted = row["n_resulted"] or 0
        n_races_resulted = rr.get("n_races_resulted", 0) or 0
        n_okuma = rr.get("n_okuma", 0) or 0
        n_race_hits = rr.get("n_race_hits", 0) or 0
        total_actual_payout = rr.get("total_actual_payout", 0) or 0
        result.append({
            "pt": pt,
            "n_bets": row["n_bets"],
            "n_resulted": n_resulted,
            "n_hits": row["n_hits"] or 0,
            "hit_rate": round(100 * (row["n_hits"] or 0) / n_resulted, 1) if n_resulted else None,
            "roi_pct": round(100 * (row["total_return"] or 0) / (n_resulted * 100), 1) if n_resulted else None,
            "avg_payout": round((row["total_return"] or 0) / row["n_hits"]) if row["n_hits"] else None,
            "avg_actual_payout": round(total_actual_payout / n_races_resulted) if n_races_resulted else None,
            "n_races": rr.get("n_races", 0) or 0,
            "n_races_resulted": n_races_resulted,
            "n_okuma": n_okuma,
            "okuma_rate": round(100 * n_okuma / n_races_resulted, 1) if n_races_resulted else None,
            "n_race_hits": n_race_hits,
            "race_hit_rate": round(100 * n_race_hits / n_races_resulted, 1) if n_races_resulted else None,
            "race_hits_per": round(n_races_resulted / n_race_hits, 1) if n_race_hits else None,
            "current_miss_streak": streaks.get(pt, 0),
        })
    result.sort(key=lambda x: -(x["pt"] if x["pt"] is not None else -1))
    return result


def get_pt_daily_entry_stats():
    """新PTスコア方式: 日付×ラベル(神熱/見送り)別の成績集計"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT date, label,
                   COUNT(*) as n_races,
                   SUM(has_result) as n_races_resulted,
                   SUM(CASE WHEN has_result=1 THEN n_bets ELSE 0 END) as n_bets_resulted,
                   SUM(CASE WHEN has_result=1 THEN race_hit ELSE 0 END) as n_hits,
                   SUM(CASE WHEN has_result=1 THEN race_return ELSE 0 END) as total_return,
                   SUM(CASE WHEN has_result=1 AND actual_payout >= 10000 THEN 1 ELSE 0 END) as n_okuma
            FROM ({_pt_v4_race_level_sql()})
            GROUP BY date, label
            ORDER BY date DESC, label
        """).fetchall()
    out = []
    for row in rows:
        row = dict(row)
        n_bets = row["n_bets_resulted"] or 0
        n_races_resulted = row["n_races_resulted"] or 0
        out.append({
            "date": row["date"],
            "label": row["label"],
            "n_races": row["n_races"],
            "n_races_resulted": n_races_resulted,
            "n_hits": row["n_hits"] or 0,
            "hit_rate": round(100 * (row["n_hits"] or 0) / n_races_resulted, 1) if n_races_resulted else None,
            "hits_per": round(n_races_resulted / row["n_hits"], 1) if row["n_hits"] else None,
            "roi_pct": round(100 * (row["total_return"] or 0) / (n_bets * 100), 1) if n_bets else None,
            "n_okuma": row["n_okuma"] or 0,
            "pnl": (row["total_return"] or 0) - n_bets * 100,
        })
    return out


def get_pt_summary():
    """新PTスコア方式: 全体サマリー（神熱/見送り/合計）"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT label,
                   COUNT(*) as n_races,
                   SUM(has_result) as n_races_resulted,
                   SUM(CASE WHEN has_result=1 THEN n_bets ELSE 0 END) as n_bets_resulted,
                   SUM(CASE WHEN has_result=1 THEN race_hit ELSE 0 END) as n_hits,
                   SUM(CASE WHEN has_result=1 THEN race_return ELSE 0 END) as total_return,
                   SUM(CASE WHEN has_result=1 AND actual_payout >= 10000 THEN 1 ELSE 0 END) as n_okuma
            FROM ({_pt_v4_race_level_sql()})
            GROUP BY label
        """).fetchall()
    out = {}
    total = {"n_races": 0, "n_races_resulted": 0, "n_bets_resulted": 0, "n_hits": 0, "total_return": 0, "n_okuma": 0}
    for row in rows:
        row = dict(row)
        for k in total:
            total[k] += row.get(k) or 0
        out[row["label"]] = row
    out["合計"] = total
    for label, row in out.items():
        n_bets = row.get("n_bets_resulted") or 0
        n_races_resulted = row.get("n_races_resulted") or 0
        row["hit_rate"] = round(100 * (row.get("n_hits") or 0) / n_races_resulted, 1) if n_races_resulted else None
        row["roi_pct"] = round(100 * (row.get("total_return") or 0) / (n_bets * 100), 1) if n_bets else None
        row["okuma_rate"] = round(100 * (row.get("n_okuma") or 0) / n_races_resulted, 1) if n_races_resulted else None
        row["pnl"] = (row.get("total_return") or 0) - n_bets * 100
    return out


def get_pt_threshold_curve():
    """新PTスコア方式: 参戦ライン候補ごとの累積成績（PT≥X の場合の的中率・ROI）
    今後の参戦ライン(PT_MIN_SCORE)調整の参考データ"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT arare_score as pt,
                   COUNT(*) as n_bets,
                   SUM(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as n_resulted,
                   SUM(is_hit) as n_hits,
                   SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as total_return
            FROM predictions
            WHERE strategy_version='5' AND bet_type='3連単'
            GROUP BY arare_score
        """).fetchall()
    rows = [dict(r) for r in rows]
    if not rows:
        return []
    max_pt = max(r["pt"] for r in rows)
    by_pt = {r["pt"]: r for r in rows}
    curve = []
    for threshold in range(max_pt, -1, -1):
        cum_bets = cum_resulted = cum_hits = cum_return = 0
        for pt in range(threshold, max_pt + 1):
            r = by_pt.get(pt)
            if not r:
                continue
            cum_bets += r["n_bets"]
            cum_resulted += r["n_resulted"] or 0
            cum_hits += r["n_hits"] or 0
            cum_return += r["total_return"] or 0
        curve.append({
            "threshold": threshold,
            "n_bets": cum_bets,
            "n_resulted": cum_resulted,
            "n_hits": cum_hits,
            "hit_rate": round(100 * cum_hits / cum_resulted, 1) if cum_resulted else None,
            "hits_per": round(cum_resulted / cum_hits, 1) if cum_hits else None,
            "roi_pct": round(100 * cum_return / (cum_resulted * 100), 1) if cum_resulted else None,
        })
    return curve


def get_pt_calibration_stats():
    """新PTスコア方式: モデルの確率推定キャリブレーション検証
    「モデルが確率X%と言った組み合わせが、実際に何%当たっているか」を帯別に集計する。
    予測確率と実際の的中率が近ければ4エージェント合成確率は信頼でき、乖離が大きい
    ほど確率推定（エージェント重み・温度スケーリング等）の見直しが必要と判断できる。
    """
    init_db()
    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT prob, is_hit
            FROM predictions
            WHERE strategy_version='5' AND bet_type='3連単'
              AND result_recorded_at IS NOT NULL
        """).fetchall()

    parsed = []
    for r in rows:
        try:
            p = float(r["prob"])
        except (TypeError, ValueError):
            continue
        parsed.append((p, r["is_hit"] or 0))

    buckets = [
        (0.00, 0.01, "0〜1%"),
        (0.01, 0.02, "1〜2%"),
        (0.02, 0.05, "2〜5%"),
        (0.05, 0.10, "5〜10%"),
        (0.10, 1.01, "10%以上"),
    ]
    result = []
    for lo, hi, label in buckets:
        bucket = [(p, h) for p, h in parsed if lo <= p < hi]
        n = len(bucket)
        if n == 0:
            result.append({"label": label, "n": 0, "avg_pred_pct": None, "n_hits": 0, "actual_hit_pct": None})
            continue
        avg_pred = sum(p for p, _ in bucket) / n
        hits = sum(h for _, h in bucket)
        result.append({
            "label": label,
            "n": n,
            "avg_pred_pct": round(avg_pred * 100, 2),
            "n_hits": hits,
            "actual_hit_pct": round(100 * hits / n, 2),
        })
    return result


def get_race_position_distribution():
    """その日の何レース目（全会場通しの順番）で的中しているかを10レース刻みで集計。
    日付をまたいで全期間累積する（strategy_version='5'以降）。
    「50〜59レース目で的中が多い＝狙い目」のような目安を作るための集計。

    day_race_noは予想保存時に本日の全番組表(schedule)から算出した値を使う
    （記録済みレース数からの逆算ではない）ため、ツールの途中再起動や
    日中からの起動でも正しい「当日何レース目か」を維持できる。
    day_race_noが未設定の古いレコード(この仕組み導入前)は対象外。"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            WITH race_level AS (
                SELECT date, venue_name, race_no,
                       MAX(day_race_no) as day_race_no,
                       MAX(is_hit) as race_hit,
                       MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result
                FROM predictions
                WHERE bet_type='3連単' AND strategy_version='5' AND day_race_no IS NOT NULL
                GROUP BY date, venue_name, race_no
            )
            SELECT (day_race_no - 1) / 10 AS bucket,
                   COUNT(*) as n_races,
                   SUM(race_hit) as n_hits
            FROM race_level
            WHERE has_result = 1
            GROUP BY bucket
            ORDER BY bucket
        """).fetchall()
    result = []
    for r in rows:
        r = dict(r)
        bucket = r["bucket"]
        n_races = r["n_races"] or 0
        n_hits = r["n_hits"] or 0
        result.append({
            "bucket": bucket,
            "label": f"{bucket * 10 + 1}〜{bucket * 10 + 10}R目",
            "n_races": n_races,
            "n_hits": n_hits,
            "hit_rate": round(100 * n_hits / n_races, 1) if n_races else None,
        })
    return result
