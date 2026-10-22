"""
PERRY AI - Flask Webアプリ
localhost:5001 でブラウザから予想を確認・集計を閲覧する
"""

import json
import time
from datetime import datetime
from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from web.database import (
    get_today_predictions, get_race_preview, get_today_venue_ranking,
    get_daily_summary, get_consecutive_misses, init_db, update_result,
    get_venue_stats, get_venue_detail, get_all_streaks,
    get_recent_activity, get_activity_watermark,
    get_payout_distribution,
    get_pt_score_stats, get_pt_daily_entry_stats, get_pt_summary, get_pt_threshold_curve,
    get_grade_tier_stats, get_grade_tier_all_time_streaks,
    get_venue_okuma_ranking, get_race_position_distribution,
    get_today_okuma_count,
    get_pt_condition_breakdown, get_odds_band_stats, get_edge_vs_market_stats,
    get_b_class_venue_rank_curve, get_boat1_grade_payout_stats, get_boat1_b_class_hit_stats,
    get_b_class_daily_summary_top3, get_b_class_daily_summary_top5, get_b_class_daily_summary_all,
    get_b_class_daily_hit_detail_top3, get_b_class_daily_hit_detail_top5, get_b_class_daily_hit_detail_all,
)
from src.model.predictor import PT_MIN_SCORE


# TODAY画面「本日の進行状況」の1号艇級別集計（2026-10-21、PT別集計を置き換え）。
# B1B2_TOP5・B1B2_TOP3はいずれもB1B2_ALLの内数（当日順位1〜5位・1〜3位の会場だけに
# 絞った別系列）であり、本日合計の算出では二重計上しないようA1+A2+B1B2_ALLのみを
# 合算する。B1B2_TOP3は2026-10-22追加（colorの"magenta"は2026-10-22新設の配色）。
_GRADE_TIER_META = [
    {"tier": "A1", "label": "A1", "color": "skip"},
    {"tier": "A2", "label": "A2", "color": "skip"},
    {"tier": "B1B2_ALL", "label": "B1B2(全順位)", "color": "shinnetsu"},
    {"tier": "B1B2_TOP5", "label": "B1B2(本日1〜5位)", "color": "pink"},
    {"tier": "B1B2_TOP3", "label": "B1B2(本日1〜3位)", "color": "magenta"},
]


def _sum_races(races):
    """レースのリストから(件数, 的中数, 掛け金, 払戻)を集計する共通ロジック"""
    n_races = len(races)
    n_hits = 0
    cost = 0
    ret = 0
    for race in races:
        race_hit = False
        for b in race["bets"]:
            cost += 100
            if b.get("is_hit"):
                race_hit = True
                ret += b.get("actual_payout") or 0
        if race_hit:
            n_hits += 1
    return n_races, n_hits, cost, ret


def _build_grade_counts(race_list, all_time_streaks=None, all_time_grade_stats=None):
    """TODAY画面の1号艇級別集計を計算する（2026-10-21、PT別集計(_build_pt_counts)を
    置き換え）。A1・A2・B1B2(全順位)・B1B2(本日順位1〜5位)の4行+本日合計を返す
    （1点=100円換算・的中控除後の収支は当日分、連続不的中数・何回に1回は前日以前も
    込みの全期間分）。
    Args:
        all_time_streaks: get_grade_tier_all_time_streaks()の戻り値
        all_time_grade_stats: get_grade_tier_stats()の戻り値（「何回に1回」用）
    Returns: (grade_counts: list[dict], grade_total: dict)
    """
    all_time_streaks = all_time_streaks or {"by_tier": {}, "overall": 0}
    by_tier_streak = all_time_streaks.get("by_tier", {})
    all_time_grade_stats = all_time_grade_stats or {}

    a1_races = [r for r in race_list if r.get("boat1_grade") == "A1"]
    a2_races = [r for r in race_list if r.get("boat1_grade") == "A2"]
    b_all_races = [r for r in race_list if r.get("boat1_grade") in ("B1", "B2")]
    b_top5_races = [
        r for r in b_all_races
        if r.get("venue_day_rank") is not None and r["venue_day_rank"] <= 5
    ]
    b_top3_races = [
        r for r in b_all_races
        if r.get("venue_day_rank") is not None and r["venue_day_rank"] <= 3
    ]
    tier_races = {
        "A1": a1_races, "A2": a2_races,
        "B1B2_ALL": b_all_races, "B1B2_TOP5": b_top5_races, "B1B2_TOP3": b_top3_races,
    }

    grade_counts = []
    for meta in _GRADE_TIER_META:
        tier = meta["tier"]
        n_races, n_hits, cost, ret = _sum_races(tier_races[tier])
        alltime = all_time_grade_stats.get(tier, {})
        grade_counts.append({
            "tier": tier, "label": meta["label"], "color": meta["color"],
            "races": n_races, "hits": n_hits,
            "cost": cost, "return": ret, "pnl": ret - cost,
            "miss_streak": by_tier_streak.get(tier, 0),
            "hits_per": alltime.get("hits_per"),
            "race_objs": tier_races[tier],
        })

    total_races, total_hits, total_cost, total_return = _sum_races(a1_races + a2_races + b_all_races)
    grade_total = {
        "races": total_races, "hits": total_hits,
        "cost": total_cost, "return": total_return, "pnl": total_return - total_cost,
        "miss_streak": all_time_streaks.get("overall", 0),
        "hits_per": all_time_grade_stats.get("OVERALL", {}).get("hits_per"),
    }
    return grade_counts, grade_total


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")

    init_db()

    @app.route("/")
    def index():
        date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        predictions = get_today_predictions(date)
        # レース単位でまとめる
        races = {}
        for p in predictions:
            key = (p["venue_name"], p["race_no"])
            if key not in races:
                races[key] = {
                    "venue_name": p["venue_name"],
                    "race_no": p["race_no"],
                    "race_time": p.get("race_time", ""),
                    "arare_score": p["arare_score"],
                    "arare_reasons": p["arare_reasons"],
                    "nigerate_str": p["nigerate_str"],
                    "boat1_risk": p["boat1_risk"],
                    "day_race_no": p.get("day_race_no"),
                    "boat1_grade": p.get("boat1_grade"),
                    "venue_day_rank": p.get("venue_day_rank"),
                    "bets": [],
                }
            races[key]["bets"].append(p)
        race_list = sorted(races.values(), key=lambda x: (x["venue_name"], x["race_no"]))
        # 会場ごとにグループ化
        venues = {}
        for race in race_list:
            vn = race["venue_name"]
            if vn not in venues:
                venues[vn] = []
            venues[vn].append(race)
        venue_list = [{"venue_name": k, "races": v} for k, v in venues.items()]
        venue_okuma_ranking = get_venue_okuma_ranking()
        grade_counts, grade_total = _build_grade_counts(
            race_list, get_grade_tier_all_time_streaks(), get_grade_tier_stats()
        )
        race_position_distribution = get_race_position_distribution()
        today_okuma = get_today_okuma_count(date)
        # 「現在何R目か」は記録済みレース数(race_list|length)ではなく、本日の全番組表
        # から算出したday_race_noの最大値を使う。途中再起動・日中からの起動でも
        # 正しい「当日何レース目か」を維持できる。day_race_no未設定の日は件数で代替。
        day_race_nos = [r["day_race_no"] for r in race_list if r.get("day_race_no") is not None]
        today_progress = max(day_race_nos) if day_race_nos else len(race_list)
        # 朝一プレビュー（2026-10-21追加）: 発走10分前の実際の予想を待たずに、
        # 番組表から先読みした1号艇B級レース一覧（auto_runner起動時にrace_previewへ
        # 保存される）。当日順位1〜5位かどうかも合わせて渡す。
        race_preview = get_race_preview(date)
        preview_b_class = [r for r in race_preview if r.get("boat1_grade") in ("B1", "B2")]
        preview_b_top5 = [
            r for r in preview_b_class
            if r.get("venue_day_rank") is not None and r["venue_day_rank"] <= 5
        ]
        preview_b_top3 = [
            r for r in preview_b_class
            if r.get("venue_day_rank") is not None and r["venue_day_rank"] <= 3
        ]
        # 本日の会場 当日順位ランキング（2026-10-22追加）。クリックするとその会場の
        # B級レース一覧(preview_b_class)を開く。
        today_venue_ranking = get_today_venue_ranking(date)
        return render_template("index.html", date=date, race_list=race_list, venue_list=venue_list,
                               venue_okuma_ranking=venue_okuma_ranking,
                               grade_counts=grade_counts, grade_total=grade_total,
                               race_position_distribution=race_position_distribution,
                               today_progress=today_progress, today_okuma=today_okuma,
                               preview_b_class=preview_b_class, preview_b_top5=preview_b_top5,
                               preview_b_top3=preview_b_top3, today_venue_ranking=today_venue_ranking)

    @app.route("/stats")
    def stats():
        daily = get_daily_summary()
        streaks = get_consecutive_misses()

        # 収支累計（日付別）
        cumulative = []
        running = 0
        for d in reversed(daily):
            bet_cost = d["result_combos"] * 100
            running += d["total_payout"] - bet_cost
            cumulative.append({"date": d["date"], "pnl": running})

        venue_stats = get_venue_stats()
        all_streaks = get_all_streaks()
        payout_dist = get_payout_distribution()

        # 新PTスコア方式（的中関連はstrategy_version='6'、実結果関連は'5'以降を集計）
        pt_score_stats = get_pt_score_stats()
        pt_daily_entry = get_pt_daily_entry_stats()
        pt_summary = get_pt_summary()
        pt_threshold_curve = get_pt_threshold_curve()
        pt_condition_breakdown = get_pt_condition_breakdown()
        odds_band_stats = get_odds_band_stats()
        edge_vs_market_stats = get_edge_vs_market_stats()
        b_class_rank_curve = get_b_class_venue_rank_curve()
        boat1_grade_payout_stats = get_boat1_grade_payout_stats()
        boat1_b_class_hit_stats = get_boat1_b_class_hit_stats()
        b_class_daily_top3 = get_b_class_daily_summary_top3()
        b_class_daily_top5 = get_b_class_daily_summary_top5()
        b_class_daily_all = get_b_class_daily_summary_all()
        b_class_daily_hits_top3 = get_b_class_daily_hit_detail_top3()
        b_class_daily_hits_top5 = get_b_class_daily_hit_detail_top5()
        b_class_daily_hits_all = get_b_class_daily_hit_detail_all()

        return render_template("stats.html",
                               daily=daily,
                               streaks=streaks,
                               cumulative=cumulative,
                               venue_stats=venue_stats,
                               all_streaks=all_streaks,
                               payout_dist=payout_dist,
                               pt_score_stats=pt_score_stats,
                               pt_daily_entry=pt_daily_entry,
                               pt_summary=pt_summary,
                               pt_threshold_curve=pt_threshold_curve,
                               pt_condition_breakdown=pt_condition_breakdown,
                               odds_band_stats=odds_band_stats,
                               edge_vs_market_stats=edge_vs_market_stats,
                               b_class_rank_curve=b_class_rank_curve,
                               boat1_grade_payout_stats=boat1_grade_payout_stats,
                               boat1_b_class_hit_stats=boat1_b_class_hit_stats,
                               b_class_daily_top3=b_class_daily_top3,
                               b_class_daily_top5=b_class_daily_top5,
                               b_class_daily_all=b_class_daily_all,
                               b_class_daily_hits_top3=b_class_daily_hits_top3,
                               b_class_daily_hits_top5=b_class_daily_hits_top5,
                               b_class_daily_hits_all=b_class_daily_hits_all,
                               pt_min_score=PT_MIN_SCORE)

    @app.route("/api/today")
    def api_today():
        date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        return jsonify(get_today_predictions(date))

    @app.route("/api/recent_activity")
    def api_recent_activity():
        date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        return jsonify({
            "rows": get_recent_activity(date),
            "watermark": get_activity_watermark(date),
        })

    @app.route("/api/activity_stream")
    def api_activity_stream():
        date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        def generate():
            last_wm = None
            while True:
                # ログの表示内容(get_recent_activity)は予想のタイミングだけで決まるが、
                # ページ全体を更新すべきタイミングの検知には結果記録も含める必要がある
                # （最終レースの結果が記録されてもページの自動更新が反応しなくなるため）。
                # そのため、この2つは別々の値（watermark）で管理する。
                wm = get_activity_watermark(date)
                if wm != last_wm:
                    last_wm = wm
                    payload = {"rows": get_recent_activity(date), "watermark": wm}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                time.sleep(3)
        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/venue_detail/<venue_name>")
    def api_venue_detail(venue_name):
        return jsonify(get_venue_detail(venue_name))

    @app.route("/api/update_result", methods=["POST"])
    def api_update_result():
        data = request.json or {}
        update_result(
            data.get("date", ""),
            data.get("venue_name", ""),
            int(data.get("race_no", 0)),
            data.get("actual_combination", ""),
            int(data.get("actual_payout", 0)),
            int(data.get("actual_payout_2ren", 0)),
        )
        return jsonify({"ok": True})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
