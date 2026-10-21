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
                    "day_race_no INTEGER",
                    "venue_day_rank INTEGER",
                    "boat1_grade TEXT"]:
            try:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col}")
            except Exception:
                pass
        # 2026-10-19: 5点目のみの選出方式変更でstrategy_versionを'8'→'9'に上げたところ、
        # 的中関連の統計が1〜4点目（無変更）の直近データまで巻き込んでリセットされてしまい、
        # ユーザーから「データが消えた」との指摘を受けて即日revert。その間に'9'で保存された
        # 行を'8'に戻し、リセット前のデータと再び地続きにする（既に'9'が無ければ無害な空振り）。
        conn.execute("UPDATE predictions SET strategy_version='8' WHERE strategy_version='9'")


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
               strategy_version, day_race_no, venue_day_rank, boat1_grade)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            str(rec.get("strategy_version", "8")),
            rec.get("day_race_no"),
            rec.get("venue_day_rank"),
            str(rec.get("boat1_grade", "")),
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
    """当日の最新予想レース一覧（会場・レース単位、予想を出した順）
    表示・並び順は予想を出した時刻（created_at）だけで決まる。結果が後から
    記録されても、そのレースが再びリストの先頭に浮上することはない
    （結果取得はログには流れない。ログに流れるのは予想のタイミングのみ）。"""
    init_db()
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT venue_name, race_no, arare_score,
                   MAX(bet_label) as bet_label,
                   MAX(boat1_grade) as boat1_grade,
                   MAX(venue_day_rank) as venue_day_rank,
                   MAX(created_at) as created_at
            FROM predictions
            WHERE date = ?
            GROUP BY venue_name, race_no
            ORDER BY created_at DESC
            LIMIT ?
        """, (date, limit)).fetchall()
    return [dict(r) for r in rows]


def get_activity_watermark(date: str = None):
    """当日の予想・結果記録のうち最も新しい更新時刻（結果記録も含む）。
    ページ全体の自動更新（SSE）が「何か変わった」を検知するためだけに使う値で、
    ログの表示内容・並び順（get_recent_activity）には一切影響しない。
    これが無いと、最終レースの予想後は新規保存が発生しないため、その結果が
    記録されてもページの自動更新（SSE）が反応しないという不具合になる。"""
    init_db()
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    with get_conn() as conn:
        row = conn.execute("""
            SELECT MAX(COALESCE(result_recorded_at, created_at)) as watermark
            FROM predictions
            WHERE date = ?
        """, (date,)).fetchone()
    return row["watermark"] if row else None


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


def get_daily_summary():
    """日付別集計（最新30日・レース単位）。的中関連の集計なので現行PT(strategy_version='8')のみ対象。"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(f"""
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
                WHERE {PT_RESET_FILTER}
                GROUP BY date, venue_name, race_no
            )
            GROUP BY date
            ORDER BY date DESC
            LIMIT 30
        """).fetchall()
    return [dict(r) for r in rows]


def get_venue_detail(venue_name: str):
    """会場別ドリルダウン: PT帯別・等級別の内訳。的中関連の集計なので現行PT(strategy_version='8')のみ対象。"""
    init_db()
    with get_conn() as conn:
        pt_rows = conn.execute(f"""
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
                WHERE venue_name = ? AND {PT_RESET_FILTER}
                GROUP BY date, venue_name, race_no
            )
            GROUP BY arare_score
            ORDER BY arare_score
        """, (venue_name,)).fetchall()

        grade_rows = conn.execute(f"""
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
                WHERE venue_name = ? AND {PT_RESET_FILTER}
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
    「今から」を正確に区切れる。今後レースが記録されるたびに集計対象が増えていく。
    レースの実際の結果だけに基づく集計（買い目の選出方式とは無関係）なので、
    2026-09-04のオッズ帯・選出方式刷新（'5'→'6'）をまたいでも積み上げ続ける
    （PT_ACCUMULATE_FILTER）。"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT venue_name,
                   COUNT(*) as n_races,
                   SUM(CASE WHEN actual_payout >= 10000 THEN 1 ELSE 0 END) as n_okuma
            FROM (
                SELECT date, venue_name, race_no, MAX(actual_payout) as actual_payout
                FROM predictions
                WHERE bet_type='3連単' AND result_recorded_at IS NOT NULL AND {PT_ACCUMULATE_FILTER}
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


def compute_today_venue_rank_map(today_venue_names) -> dict:
    """その日開催する会場だけを対象に、会場別万舟ランキング（集計中・全期間ベース）の
    順位を1位から振り直したマップ {venue_name: 当日順位} を返す（2026-10-20追加）。
    「参戦ライン」新戦略（1号艇B級 × 当日上位N会場）の検証用。ランキングに乗っていない
    （集計対象の結果がまだ無い）会場は末尾に順位を振る。auto_runner側で1日1回だけ
    呼び出し、その日の間は結果に関わらずこの順位を固定して使う
    （集計中のランキングが日中に変動しても気にしない、との方針）。"""
    ranking = get_venue_okuma_ranking()
    ranked_names = [r["venue_name"] for r in ranking]
    today_set = set(today_venue_names)
    rank_map = {}
    rank = 1
    for vn in ranked_names:
        if vn in today_set:
            rank_map[vn] = rank
            rank += 1
    for vn in today_venue_names:
        if vn not in rank_map:
            rank_map[vn] = rank
            rank += 1
    return rank_map


def get_b_class_venue_rank_curve():
    """新たな参戦ライン仮説の検証用集計（2026-10-20追加・現行PT方式とは完全に別枠）。
    「1号艇がB級（B1/B2）」のレースのみを対象に、当日の会場順位（venue_day_rank。
    会場別万舟ランキングをその日開催する会場だけで振り直した順位）が1位〜N位に
    収まる会場の累積パターンを、N=1から当日最大順位まで集計する（レース単位）。
    venue_day_rankは2026-10-20以降に予想保存時へ付与する新しい列のため、それより
    前のデータはNULLとなり自動的に対象外になる（strategy_versionでの絞り込みは不要）。
    """
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT date, venue_name, race_no,
                   MAX(venue_day_rank) as venue_day_rank,
                   COUNT(*) as n_bets,
                   MAX(is_hit) as race_hit,
                   SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as race_return,
                   MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result
            FROM predictions
            WHERE bet_type='3連単' AND venue_day_rank IS NOT NULL
              AND arare_reasons LIKE '%1号B級%'
            GROUP BY date, venue_name, race_no
        """).fetchall()
    rows = [dict(r) for r in rows]
    if not rows:
        return []

    by_rank: dict[int, list] = {}
    for r in rows:
        by_rank.setdefault(r["venue_day_rank"], []).append(r)
    max_rank = max(by_rank.keys())

    curve = []
    cum_races = cum_races_resulted = cum_race_hits = cum_bets_resulted = cum_return = 0
    for n in range(1, max_rank + 1):
        for r in by_rank.get(n, []):
            cum_races += 1
            if r["has_result"]:
                cum_races_resulted += 1
                cum_race_hits += r["race_hit"] or 0
                cum_bets_resulted += r["n_bets"]
                cum_return += r["race_return"] or 0
        curve.append({
            "top_n": n,
            "n_races": cum_races,
            "n_races_resulted": cum_races_resulted,
            "n_race_hits": cum_race_hits,
            "hit_rate": round(100 * cum_race_hits / cum_races_resulted, 1) if cum_races_resulted else None,
            "roi_pct": round(100 * cum_return / (cum_bets_resulted * 100), 1) if cum_bets_resulted else None,
        })
    return curve


def _b_class_daily_summary_sql(max_rank: int = None):
    """「1号艇B級」のレースを日別に集計する共通ロジック（2026-10-21追加）。
    max_rank指定時は当日順位がmax_rank以内の会場のレースのみに絞る。
    get_b_class_venue_rank_curve()（全期間の累積・順位別カーブ）とは別に、
    実際の掛け金・的中金額の日々の推移を記録・確認するための日別サマリー。
    venue_day_rankは2026-10-20以降のデータにしか入らないため、それより前の
    日付は自動的に対象外になる。"""
    init_db()
    rank_clause = "AND venue_day_rank <= ?" if max_rank is not None else ""
    params = [max_rank] if max_rank is not None else []
    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT date,
                   COUNT(*) as n_races,
                   SUM(has_result) as n_races_resulted,
                   SUM(race_hit) as n_race_hits,
                   SUM(CASE WHEN has_result=1 THEN n_bets ELSE 0 END) as n_bets_resulted,
                   SUM(CASE WHEN has_result=1 THEN race_return ELSE 0 END) as hit_amount
            FROM (
                SELECT date, venue_name, race_no,
                       COUNT(*) as n_bets,
                       MAX(is_hit) as race_hit,
                       SUM(CASE WHEN is_hit=1 THEN actual_payout ELSE 0 END) as race_return,
                       MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result
                FROM predictions
                WHERE bet_type='3連単' AND venue_day_rank IS NOT NULL
                  AND arare_reasons LIKE '%1号B級%'
                  {rank_clause}
                GROUP BY date, venue_name, race_no
            )
            GROUP BY date
            ORDER BY date DESC
        """, params).fetchall()

    result = []
    for r in rows:
        r = dict(r)
        n_bets_resulted = r["n_bets_resulted"] or 0
        hit_amount = r["hit_amount"] or 0
        stake = n_bets_resulted * 100
        result.append({
            "date": r["date"],
            "n_races": r["n_races"],
            "n_races_resulted": r["n_races_resulted"] or 0,
            "n_race_hits": r["n_race_hits"] or 0,
            "stake": stake,
            "hit_amount": hit_amount,
            "pnl": hit_amount - stake,
        })
    return result


def get_b_class_daily_summary_top5():
    """「1号艇B級×当日順位1〜5位」の日別サマリー（掛け金・的中金額、2026-10-21追加）。"""
    return _b_class_daily_summary_sql(max_rank=5)


def get_b_class_daily_summary_all():
    """「1号艇B級」（会場の当日順位は問わず全て）の日別サマリー
    （掛け金・的中金額、2026-10-21追加）。"""
    return _b_class_daily_summary_sql(max_rank=None)


# 1号艇の級別（A1/A2/B1・B2）とレースの実配当額の相関検証（2026-10-20追加）。
# B1とB2は「新戦略: 1号艇B級×会場当日順位」と同じ基準でまとめて1枠にする。
_GRADE_BUCKET_LABELS = {"A1": "A1", "A2": "A2", "B1": "B級(B1/B2)", "B2": "B級(B1/B2)"}


def get_boat1_b_class_hit_stats():
    """的中したレースのうち、1号艇がB級（B1/B2）だったレースが何件あったかを集計する
    （2026-10-20追加）。的中関連の集計なので現行PT(PT_RESET_FILTER=strategy_version='8')
    のみ対象。1号艇B級の判定はarare_reasonsに含まれる「1号B級」という文言で行うため
    （2026-10-16のPT方式差し戻し以降、該当すれば常に記録されている既存の判定文言）、
    2026-10-20に追加したboat1_grade列より前のデータも2026-10-16以降なら遡って
    対象になる（A1/A2の区別はできないが、B級かどうかだけなら判定できる）。"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT date, venue_name, race_no,
                   MAX(is_hit) as race_hit,
                   MAX(CASE WHEN arare_reasons LIKE '%1号B級%' THEN 1 ELSE 0 END) as is_b_class,
                   MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result
            FROM predictions
            WHERE {PT_RESET_FILTER} AND bet_type='3連単'
            GROUP BY date, venue_name, race_no
        """).fetchall()
    hits = [dict(r) for r in rows if r["has_result"] and r["race_hit"]]
    n_hits = len(hits)
    n_hits_b_class = sum(1 for r in hits if r["is_b_class"])
    return {
        "n_hits": n_hits,
        "n_hits_b_class": n_hits_b_class,
        "b_class_rate": round(100 * n_hits_b_class / n_hits, 1) if n_hits else None,
    }


def get_boat1_grade_payout_stats():
    """1号艇の級別（A1/A2/B級）によってレースの実配当額に差があるか検証する集計
    （2026-10-20追加）。的中したかどうか（自分の買い目が当たったか）とは無関係に、
    レースの実際の結果だけを見る集計（会場別万舟ランキングと同じ考え方）。
    boat1_gradeは2026-10-20以降に予想保存時へ付与する新しい列のため、それより前の
    データは空文字でこの集計からは自動的に除外される。"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT date, venue_name, race_no,
                   MAX(boat1_grade) as boat1_grade,
                   MAX(actual_payout) as actual_payout,
                   MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result
            FROM predictions
            WHERE bet_type='3連単' AND boat1_grade IS NOT NULL AND boat1_grade != ''
            GROUP BY date, venue_name, race_no
        """).fetchall()

    buckets: dict[str, list] = {"A1": [], "A2": [], "B級(B1/B2)": []}
    for r in rows:
        r = dict(r)
        if not r["has_result"]:
            continue
        label = _GRADE_BUCKET_LABELS.get(r["boat1_grade"])
        if label:
            buckets[label].append(r["actual_payout"] or 0)

    result = []
    for label in ["A1", "A2", "B級(B1/B2)"]:
        payouts = buckets[label]
        n = len(payouts)
        result.append({
            "label": label,
            "n_races": n,
            "avg_payout": round(sum(payouts) / n) if n else None,
            "okuma_rate": round(100 * sum(1 for p in payouts if p >= 10000) / n, 1) if n else None,
        })
    return result


def get_today_okuma_count(date: str = None):
    """本日、全会場通しで実際に万舟(実配当≥1万円)になったレース数をカウントする。
    的中したかどうか（自分の買い目が当たったか）とは無関係に、レースの実際の結果
    だけを見る集計。strategy_versionとも無関係（選出方式が変わっても意味は変わらない
    ため、リセット対象にしない）。"""
    init_db()
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    with get_conn() as conn:
        row = conn.execute("""
            SELECT COUNT(*) as n_races_resulted,
                   SUM(CASE WHEN actual_payout >= 10000 THEN 1 ELSE 0 END) as n_okuma
            FROM (
                SELECT date, venue_name, race_no, MAX(actual_payout) as actual_payout
                FROM predictions
                WHERE date=? AND bet_type='3連単' AND result_recorded_at IS NOT NULL
                GROUP BY date, venue_name, race_no
            )
        """, (date,)).fetchone()
    row = dict(row) if row else {}
    return {
        "n_races_resulted": row.get("n_races_resulted") or 0,
        "n_okuma": row.get("n_okuma") or 0,
    }


def get_venue_stats():
    """会場別集計（全期間・レース単位）。的中関連の集計なので現行PT(strategy_version='8')のみ対象。"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(f"""
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
                WHERE {PT_RESET_FILTER}
                GROUP BY date, venue_name, race_no
            )
            GROUP BY venue_name
            ORDER BY total DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_payout_distribution():
    """ラベル別・実結果3連単配当分布（神熱 vs 見送り比較用）。ラベル(神熱/見送り)は
    現行PT_MIN_SCOREの判定に基づくため、現行PT(strategy_version='8')のみ対象。"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(f"""
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
                WHERE {PT_RESET_FILTER}
                GROUP BY date, venue_name, race_no
            )
            GROUP BY bet_label
            ORDER BY
                CASE bet_label WHEN '神熱' THEN 1 ELSE 2 END
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
    """ラベル・賭け種・PT・会場・等級別の直近連続外れ数を一括取得（参戦レースのみ・レース単位）。
    的中関連の集計なので現行PT(strategy_version='8')のみ対象。"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT MAX(bet_label) as bet_label,
                   COALESCE(NULLIF(bet_type,''), '3連単') as bet_type,
                   MAX(arare_score) as arare_score,
                   MAX(venue_name) as venue_name,
                   COALESCE(NULLIF(MAX(race_grade),''), '不明') as race_grade,
                   MAX(is_hit) as is_hit
            FROM predictions
            WHERE result_recorded_at IS NOT NULL
              AND bet_label != '見送り'
              AND {PT_RESET_FILTER}
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
# バージョンごとに点数の意味が異なる。日付や時刻ではなく strategy_version で
# 厳密に絞り込むことで、タイムゾーンのズレに影響されず「最終版のコードで保存された
# 予想から」を正確に集計対象にできる（次に保存される予想から即座に反映される）。
#
# 2026-09-04にオッズ帯・選出方式(EV→ブレンド)を刷新し strategy_version を'5'→'6'に
# 引き上げた。この変更は「どの組み合わせを買うか」だけに影響するため、集計は2種類に
# 分かれる:
#   - 的中関連（n_bets/n_hits/hit_rate/roi_pct/avg_payout/連続不的中数等）:
#     選出方式が変わると意味が変わるため strategy_version='6'（刷新後）のみを対象に
#     ゼロから取り直す
#   - レースの実際の結果だけに基づく集計（会場別万舟率・PT毎の平均実配当額等）:
#     選出方式と無関係な事実なので strategy_version IN ('5','6') として '5' 以降を
#     通して積み上げる
# ══════════════════════════════════════════════════════════════
PT_RESET_FILTER = "strategy_version='8'"       # 的中関連（刷新後のみ・リセット）
PT_ACCUMULATE_FILTER = "strategy_version IN ('5','6','7','8')"  # 実結果関連（'5'以降を積み上げ）


def _pt_v4_race_level_sql(select_extra: str = "", version_filter: str = PT_RESET_FILTER) -> str:
    """strategy_versionで絞り込んだレース単位集計サブクエリ（共通部分）。
    version_filterはPT_RESET_FILTER（デフォルト・的中関連の集計用）または
    PT_ACCUMULATE_FILTER（実結果だけに基づく集計用）を渡す。"""
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
        WHERE {version_filter} AND bet_type='3連単'
        GROUP BY date, venue_name, race_no
    """


def get_pt_score_stats():
    """新PTスコア方式: PT値ごとの成績集計
    賭け数(1点=100円)・的中率・平均配当率(ROI)・万舟率(実配当≥1万円のレース比率)・現在の連続不的中数

    2026-09-04のオッズ帯・選出方式刷新に伴い、的中関連(n_bets/n_hits/hit_rate/roi_pct/
    avg_payout/n_race_hits/race_hits_per/current_miss_streak)はPT_RESET_FILTER
    （刷新後のみ）でゼロから集計し、レースの実際の結果だけに基づく集計
    (avg_actual_payout/n_okuma/okuma_rate)はPT_ACCUMULATE_FILTER（'5'以降を通して）
    で積み上げる。両者は分母(n_races_resulted)も別々になる点に注意。
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
            WHERE {PT_RESET_FILTER} AND bet_type='3連単'
            GROUP BY arare_score
        """).fetchall()

        race_rows_v6 = conn.execute(f"""
            SELECT pt,
                   COUNT(*) as n_races,
                   SUM(has_result) as n_races_resulted,
                   SUM(CASE WHEN has_result=1 THEN race_hit ELSE 0 END) as n_race_hits
            FROM ({_pt_v4_race_level_sql(version_filter=PT_RESET_FILTER)})
            GROUP BY pt
        """).fetchall()

        race_rows_all = conn.execute(f"""
            SELECT pt,
                   SUM(has_result) as n_races_resulted,
                   SUM(CASE WHEN has_result=1 AND actual_payout >= 10000 THEN 1 ELSE 0 END) as n_okuma,
                   SUM(CASE WHEN has_result=1 THEN actual_payout ELSE 0 END) as total_actual_payout
            FROM ({_pt_v4_race_level_sql(version_filter=PT_ACCUMULATE_FILTER)})
            GROUP BY pt
        """).fetchall()

        streak_rows = conn.execute(f"""
            SELECT MAX(arare_score) as pt, MAX(is_hit) as is_hit
            FROM predictions
            WHERE {PT_RESET_FILTER} AND bet_type='3連単' AND result_recorded_at IS NOT NULL
            GROUP BY date, venue_name, race_no
            ORDER BY MAX(id) DESC
        """).fetchall()

    streaks = _calc_streaks([dict(r) for r in streak_rows], lambda r: r["pt"])
    race_by_pt_v6 = {r["pt"]: dict(r) for r in race_rows_v6}
    race_by_pt_all = {r["pt"]: dict(r) for r in race_rows_all}

    result = []
    for row in bet_rows:
        row = dict(row)
        pt = row["pt"]
        rr6 = race_by_pt_v6.get(pt, {})
        rr_all = race_by_pt_all.get(pt, {})
        n_resulted = row["n_resulted"] or 0
        n_races_resulted = rr6.get("n_races_resulted", 0) or 0
        n_race_hits = rr6.get("n_race_hits", 0) or 0
        n_races_resulted_all = rr_all.get("n_races_resulted", 0) or 0
        n_okuma = rr_all.get("n_okuma", 0) or 0
        total_actual_payout = rr_all.get("total_actual_payout", 0) or 0
        result.append({
            "pt": pt,
            "n_bets": row["n_bets"],
            "n_resulted": n_resulted,
            "n_hits": row["n_hits"] or 0,
            "hit_rate": round(100 * (row["n_hits"] or 0) / n_resulted, 1) if n_resulted else None,
            "roi_pct": round(100 * (row["total_return"] or 0) / (n_resulted * 100), 1) if n_resulted else None,
            "avg_payout": round((row["total_return"] or 0) / row["n_hits"]) if row["n_hits"] else None,
            "avg_actual_payout": round(total_actual_payout / n_races_resulted_all) if n_races_resulted_all else None,
            "n_races": rr6.get("n_races", 0) or 0,
            "n_races_resulted": n_races_resulted,
            "n_okuma": n_okuma,
            "okuma_rate": round(100 * n_okuma / n_races_resulted_all, 1) if n_races_resulted_all else None,
            "n_race_hits": n_race_hits,
            "race_hit_rate": round(100 * n_race_hits / n_races_resulted, 1) if n_races_resulted else None,
            "race_hits_per": round(n_races_resulted / n_race_hits, 1) if n_race_hits else None,
            "current_miss_streak": streaks.get(pt, 0),
        })
    result.sort(key=lambda x: -(x["pt"] if x["pt"] is not None else -1))
    return result


def get_pt_all_time_streaks():
    """PT値ごと・全体の、日付をまたいだ全期間での現在の連続不的中数（PT_RESET_FILTER限定）
    TODAY画面のPT別集計で「当日分だけ」ではなく前日以前も込みで連続不的中を数えるために使う。
    的中関連の集計なので、オッズ帯・選出方式を刷新した2026-09-04以降のみを対象にする。"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT MAX(arare_score) as pt, MAX(is_hit) as is_hit
            FROM predictions
            WHERE {PT_RESET_FILTER} AND bet_type='3連単' AND result_recorded_at IS NOT NULL
            GROUP BY date, venue_name, race_no
            ORDER BY MAX(id) DESC
        """).fetchall()
    rows = [dict(r) for r in rows]
    by_pt = _calc_streaks(rows, lambda r: r["pt"])
    overall = _calc_streaks(rows, lambda r: True)
    return {"by_pt": by_pt, "overall": overall.get(True, 0)}


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
    """新PTスコア方式: 参戦ライン候補ごとの累積成績（PT≥X の場合のレース数・的中率・ROI）
    今後の参戦ライン(PT_MIN_SCORE)調整の参考データ。的中関連の集計なので、
    オッズ帯・選出方式を刷新した2026-09-04以降のみを対象にする。
    カウント単位はレース（1レースに複数点の買い目があっても1レースとして数える。
    回収率の計算だけは実際に賭けた金額＝買い目の点数×100円を使う）。"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT pt,
                   COUNT(*) as n_races,
                   SUM(has_result) as n_races_resulted,
                   SUM(CASE WHEN has_result=1 THEN race_hit ELSE 0 END) as n_race_hits,
                   SUM(CASE WHEN has_result=1 THEN n_bets ELSE 0 END) as n_bets_resulted,
                   SUM(CASE WHEN has_result=1 THEN race_return ELSE 0 END) as total_return
            FROM ({_pt_v4_race_level_sql()})
            GROUP BY pt
        """).fetchall()
    rows = [dict(r) for r in rows]
    if not rows:
        return []
    max_pt = max(r["pt"] for r in rows)
    by_pt = {r["pt"]: r for r in rows}
    curve = []
    for threshold in range(max_pt, -1, -1):
        cum_races = cum_races_resulted = cum_race_hits = cum_bets_resulted = cum_return = 0
        for pt in range(threshold, max_pt + 1):
            r = by_pt.get(pt)
            if not r:
                continue
            cum_races += r["n_races"]
            cum_races_resulted += r["n_races_resulted"] or 0
            cum_race_hits += r["n_race_hits"] or 0
            cum_bets_resulted += r["n_bets_resulted"] or 0
            cum_return += r["total_return"] or 0
        curve.append({
            "threshold": threshold,
            "n_races": cum_races,
            "n_races_resulted": cum_races_resulted,
            "n_race_hits": cum_race_hits,
            "hit_rate": round(100 * cum_race_hits / cum_races_resulted, 1) if cum_races_resulted else None,
            "hits_per": round(cum_races_resulted / cum_race_hits, 1) if cum_race_hits else None,
            "roi_pct": round(100 * cum_return / (cum_bets_resulted * 100), 1) if cum_bets_resulted else None,
        })
    return curve



def _classify_pt_pattern(reasons: str) -> str:
    """arare_reasons（レース単位の加点理由テキスト）の文言から、荒れPTのA〜C区分
    のうちどれが加点に寄与したかを判定する（DBに区分ごとのフラグを持たせていないため）。
    2026-10-16にPTの計算式を刷新したため区分も合わせて更新（A:1号艇の弱さ/
    B:外艇の脅威/C:環境条件）。

    2026-10-21: Cから「一般戦」を除外した。一般戦（meet_grade_num<=1、SG/G1/G2/G3
    以外の通常開催）は大多数のレースで該当する、ほぼ常にTrueになる条件のため、
    これを含めるとCが参戦ラインに乗った全レースでほぼ必ず発火してしまい、
    「Aだけ」「Bだけ」「A+Bだけ」というパターンが実質観測不可能になっていた
    （PT_MIN_SCORE到達には他の条件との組み合わせが必要なため）。除外後はCが
    本来の「風・波・荒れ会場」という環境要因だけを反映するようになり、A単独・
    B単独・A+B単独のパターンも実際に出現し得るようになる（PTスコア自体への
    一般戦の加点は変更せず、この区分判定だけの変更）。"""
    reasons = reasons or ""
    cats = []
    if "1号ST" in reasons or "1号M" in reasons or "1号B級" in reasons or "1号2率" in reasons:
        cats.append("A")
    if "展示最速" in reasons or "号A1" in reasons or "ST速" in reasons:
        cats.append("B")
    if "風速" in reasons or "波高" in reasons or "荒れ会場" in reasons or "江戸川" in reasons:
        cats.append("C")
    return "+".join(cats) if cats else "(該当なし)"


def get_pt_condition_breakdown():
    """荒れPT(A:市場ミスプライシング/B:外艇の脅威/C:1号艇の弱さ/D:環境条件)のうち、
    どの区分が実際に加点へ寄与したかの組み合わせパターン別に、レース数・的中率・
    平均実配当額を集計する（レース単位）。同じPT合計値でも、単純加算の結果どの条件から
    積み上がったかが違うレース同士を一括りにしてよいのか（例: A条件単独のPT4とB+C条件の
    PT4は同じ意味を持つか）を検証するための機能。

    2026-09-04のオッズ帯・選出方式刷新に伴い、的中関連(n_hits/hit_rate)はPT_RESET_FILTER
    （刷新後のみ）でゼロから集計し、レースの実際の結果だけに基づく集計(avg_pt/
    avg_actual_payout)はPT_ACCUMULATE_FILTER（'5'以降を通して）で積み上げる。
    """
    init_db()
    with get_conn() as conn:
        rows_v6 = conn.execute(f"""
            SELECT date, venue_name, race_no,
                   MAX(arare_reasons) as reasons,
                   MAX(is_hit) as is_hit
            FROM predictions
            WHERE {PT_RESET_FILTER} AND bet_type='3連単' AND result_recorded_at IS NOT NULL
            GROUP BY date, venue_name, race_no
        """).fetchall()
        rows_all = conn.execute(f"""
            SELECT date, venue_name, race_no,
                   MAX(arare_score) as pt,
                   MAX(arare_reasons) as reasons,
                   MAX(actual_payout) as actual_payout
            FROM predictions
            WHERE {PT_ACCUMULATE_FILTER} AND bet_type='3連単' AND result_recorded_at IS NOT NULL
            GROUP BY date, venue_name, race_no
        """).fetchall()

    hits_by_pattern = {}
    for r in rows_v6:
        r = dict(r)
        pattern = _classify_pt_pattern(r["reasons"])
        g = hits_by_pattern.setdefault(pattern, {"n_races": 0, "n_hits": 0})
        g["n_races"] += 1
        g["n_hits"] += int(r["is_hit"] or 0)

    outcome_by_pattern = {}
    for r in rows_all:
        r = dict(r)
        pattern = _classify_pt_pattern(r["reasons"])
        g = outcome_by_pattern.setdefault(pattern, {"n_races": 0, "payout_sum": 0, "pt_sum": 0})
        g["n_races"] += 1
        g["payout_sum"] += int(r["actual_payout"] or 0)
        g["pt_sum"] += int(r["pt"] or 0)

    patterns = set(hits_by_pattern) | set(outcome_by_pattern)
    result = []
    for pattern in patterns:
        h = hits_by_pattern.get(pattern, {"n_races": 0, "n_hits": 0})
        o = outcome_by_pattern.get(pattern, {"n_races": 0, "payout_sum": 0, "pt_sum": 0})
        result.append({
            "pattern": pattern,
            "n_races": h["n_races"],
            "n_hits": h["n_hits"],
            "hit_rate": round(100 * h["n_hits"] / h["n_races"], 2) if h["n_races"] else 0.0,
            "avg_pt": round(o["pt_sum"] / o["n_races"], 1) if o["n_races"] else 0.0,
            "avg_actual_payout": round(o["payout_sum"] / o["n_races"]) if o["n_races"] else 0,
        })
    result.sort(key=lambda x: -x["n_races"])
    return result


# 3連単の控除率（寺銭）。公式には約25〜27%程度とされるが、ここでは概算値として使う。
# 「モデル確率が市場の暗示確率よりどれだけ強気/弱気か」という相対比較が目的なので、
# この値の厳密さよりも一貫して同じ値を使うことの方が重要。
TAKEOUT_RATE_3REN = 0.25


def get_edge_vs_market_stats():
    """モデル確率(prob)と市場オッズが暗示する確率を比較し、「市場より強気(エッジ大)」
    なほど実際の的中率・回収率が上がっているかを検証する。
    市場の暗示確率 = (1 - 控除率) / オッズ（控除率を考慮しないと1/オッズの合計が
    120通り分で約1.33になってしまい、モデル確率と対等に比較できないため）。
    エッジ = モデル確率 / 市場の暗示確率（1倍=市場と一致、2倍=市場の2倍強気、など）。
    「歪みを検知できているか」の検証なので、買い目単位（レース単位ではない）で集計する。
    市場オッズを使った判断の検証なので、ライブオッズ取得時の買い目のみが対象になる
    （履歴オッズのみの買い目はodds_sourceの区別をDBに持たせていないため含む。
    履歴オッズの場合は市場の実際の暗示確率とズレる可能性がある点に注意）。
    的中関連の集計なので現行PT(strategy_version='8')のみ対象。
    """
    init_db()
    bins = [
        (0.0, 0.5, "0.5倍未満（市場より弱気）"),
        (0.5, 1.0, "0.5〜1.0倍"),
        (1.0, 1.5, "1.0〜1.5倍"),
        (1.5, 2.0, "1.5〜2.0倍"),
        (2.0, float("inf"), "2.0倍以上（市場より強気）"),
    ]
    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT prob, odds_value, is_hit, actual_payout, result_recorded_at
            FROM predictions
            WHERE {PT_RESET_FILTER} AND bet_type='3連単'
        """).fetchall()

    bucketed = [dict(n_bets=0, n_resulted=0, n_hits=0, total_return=0,
                     model_prob_sum=0.0, market_prob_sum=0.0) for _ in bins]
    skipped = 0
    for r in rows:
        r = dict(r)
        try:
            model_prob = float(r["prob"])
        except (TypeError, ValueError):
            skipped += 1
            continue
        odds = r["odds_value"]
        if not odds or odds <= 0 or model_prob <= 0:
            skipped += 1
            continue
        market_prob = (1 - TAKEOUT_RATE_3REN) / odds
        edge = model_prob / market_prob

        for i, (lo, hi, _label) in enumerate(bins):
            if lo <= edge < hi:
                b = bucketed[i]
                break
        else:
            continue

        b["n_bets"] += 1
        b["model_prob_sum"] += model_prob
        b["market_prob_sum"] += market_prob
        if r["result_recorded_at"] is not None:
            b["n_resulted"] += 1
            if r["is_hit"]:
                b["n_hits"] += 1
                b["total_return"] += r["actual_payout"] or 0

    result = []
    for (lo, hi, label), b in zip(bins, bucketed):
        n_bets = b["n_bets"]
        n_resulted = b["n_resulted"]
        result.append({
            "label": label,
            "n_bets": n_bets,
            "n_resulted": n_resulted,
            "n_hits": b["n_hits"],
            "hit_rate": round(100 * b["n_hits"] / n_resulted, 2) if n_resulted else None,
            "roi_pct": round(100 * b["total_return"] / (n_resulted * 100), 1) if n_resulted else None,
            "avg_model_prob_pct": round(100 * b["model_prob_sum"] / n_bets, 2) if n_bets else None,
            "avg_market_prob_pct": round(100 * b["market_prob_sum"] / n_bets, 2) if n_bets else None,
        })
    return result


def get_odds_band_stats():
    """オッズ帯（買い目のodds_value）別のレース数・的中数・的中率・回収率を集計する。
    PTではなく「実際に何倍のオッズの組み合わせを買ったか」で区切るため、
    1〜2点目(100〜150倍)・3〜4点目(151〜250倍)・5点目(251〜900倍)それぞれが
    実際にどれくらい機能しているかを直接検証できる。的中関連の集計なので、
    オッズ帯・選出方式を刷新した2026-09-04以降のみを対象にする（PT_RESET_FILTER）。
    カウント単位はレース（同じ帯に1レースで複数点の買い目があっても1レースとして
    数える。例: 1〜2点目はどちらも100〜150倍帯なので2点とも同じ帯に入る）。
    回収率の計算だけは実際に賭けた金額＝その帯での買い目の点数×100円を使う。
    """
    init_db()
    bands = [
        (100.0, 150.0, "100〜150倍（1〜2点目）"),
        (151.0, 250.0, "151〜250倍（3〜4点目）"),
        (251.0, 900.0, "251〜900倍（5点目）"),
    ]
    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT date, venue_name, race_no, odds_value, is_hit, actual_payout, result_recorded_at
            FROM predictions
            WHERE {PT_RESET_FILTER} AND bet_type='3連単'
        """).fetchall()

    # (race_key, band_index) -> {n_bets, resulted, hit, payout}
    race_band = {}
    for r in rows:
        r = dict(r)
        odds = r["odds_value"]
        band_idx = None
        for i, (lo, hi, _label) in enumerate(bands):
            if odds is not None and lo <= odds <= hi:
                band_idx = i
                break
        race_key = (r["date"], r["venue_name"], r["race_no"], band_idx)
        g = race_band.setdefault(race_key, {"n_bets": 0, "resulted": False, "hit": 0, "payout": 0})
        g["n_bets"] += 1
        if r["result_recorded_at"] is not None:
            g["resulted"] = True
            if r["is_hit"]:
                g["hit"] = 1
                g["payout"] += r["actual_payout"] or 0

    bucketed = [dict(n_races=0, n_races_resulted=0, n_race_hits=0, n_bets_resulted=0, total_return=0)
                for _ in bands]
    other = dict(n_races=0, n_races_resulted=0, n_race_hits=0, n_bets_resulted=0, total_return=0)
    for (_date, _venue, _race_no, band_idx), g in race_band.items():
        target = bucketed[band_idx] if band_idx is not None else other
        target["n_races"] += 1
        if g["resulted"]:
            target["n_races_resulted"] += 1
            target["n_race_hits"] += g["hit"]
            target["n_bets_resulted"] += g["n_bets"]
            target["total_return"] += g["payout"]

    def _to_row(label, b):
        n_races_resulted = b["n_races_resulted"]
        n_bets_resulted = b["n_bets_resulted"]
        return {
            "label": label,
            "n_races": b["n_races"],
            "n_races_resulted": n_races_resulted,
            "n_race_hits": b["n_race_hits"],
            "hit_rate": round(100 * b["n_race_hits"] / n_races_resulted, 2) if n_races_resulted else None,
            "roi_pct": round(100 * b["total_return"] / (n_bets_resulted * 100), 1) if n_bets_resulted else None,
            "avg_payout": round(b["total_return"] / b["n_race_hits"]) if b["n_race_hits"] else None,
        }

    result = [_to_row(label, b) for (lo, hi, label), b in zip(bands, bucketed)]
    if other["n_races"]:
        result.append(_to_row("その他（帯外）", other))
    return result


def get_race_position_distribution():
    """その日の何レース目（全会場通しの順番）で的中しているかを10レース刻みで集計。
    日付をまたいで全期間累積する。的中関連の集計なので、オッズ帯・選出方式を
    刷新した2026-09-04以降のみを対象にする（PT_RESET_FILTER）。
    「50〜59レース目で的中が多い＝狙い目」のような目安を作るための集計。

    day_race_noは予想保存時に本日の全番組表(schedule)から算出した値を使う
    （記録済みレース数からの逆算ではない）ため、ツールの途中再起動や
    日中からの起動でも正しい「当日何レース目か」を維持できる。
    day_race_noが未設定の古いレコード(この仕組み導入前)は対象外。

    表示する枠(バケット)の数は、的中データの有無に関わらず「これまでに記録された
    最大の当日通しレース数」まで常にフルで用意し、まだ的中データが無い枠は
    0/0(0%)として埋める（枠自体が消えたり歯抜けになったりしないようにするため）。
    枠の最大値はstrategy_versionを問わず全期間のday_race_noから決める
    （何レース目まで存在するかは選出方式と無関係な、番組表由来の構造的な値のため）。"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(f"""
            WITH race_level AS (
                SELECT date, venue_name, race_no,
                       MAX(day_race_no) as day_race_no,
                       MAX(is_hit) as race_hit,
                       MAX(CASE WHEN result_recorded_at IS NOT NULL THEN 1 ELSE 0 END) as has_result
                FROM predictions
                WHERE bet_type='3連単' AND {PT_RESET_FILTER} AND day_race_no IS NOT NULL
                GROUP BY date, venue_name, race_no
            )
            SELECT (day_race_no - 1) / 10 AS bucket,
                   COUNT(*) as n_races,
                   SUM(race_hit) as n_hits
            FROM race_level
            WHERE has_result = 1
            GROUP BY bucket
        """).fetchall()
        max_row = conn.execute("""
            SELECT MAX(day_race_no) as max_day_race_no
            FROM predictions
            WHERE day_race_no IS NOT NULL
        """).fetchone()

    data_by_bucket = {dict(r)["bucket"]: dict(r) for r in rows}
    max_day_race_no = dict(max_row).get("max_day_race_no") or 0
    max_bucket = (max_day_race_no - 1) // 10 if max_day_race_no > 0 else -1

    result = []
    for bucket in range(0, max_bucket + 1):
        r = data_by_bucket.get(bucket, {})
        n_races = r.get("n_races") or 0
        n_hits = r.get("n_hits") or 0
        result.append({
            "bucket": bucket,
            "label": f"{bucket * 10 + 1}〜{bucket * 10 + 10}R目",
            "n_races": n_races,
            "n_hits": n_hits,
            "hit_rate": round(100 * n_hits / n_races, 1) if n_races else 0.0,
        })
    return result
