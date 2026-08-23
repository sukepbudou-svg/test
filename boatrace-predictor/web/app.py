"""
PERRY AI - Flask Webアプリ
localhost:5001 でブラウザから予想を確認・集計を閲覧する
"""

import json
import time
from datetime import datetime
from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from web.database import (
    get_today_predictions, get_pt_stats, get_label_stats,
    get_daily_summary, get_consecutive_misses, init_db, update_result,
    get_venue_stats, get_grade_stats, get_venue_detail, get_all_streaks,
    get_recent_activity, get_hero_stats, get_pt_payout_stats, get_signal_stats,
    get_payout_distribution, get_daily_label_stats,
)


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
        return render_template("index.html", date=date, race_list=race_list, venue_list=venue_list)

    @app.route("/stats")
    def stats():
        pt_stats = get_pt_stats()
        label_stats = get_label_stats()
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
        grade_stats = get_grade_stats()
        all_streaks = get_all_streaks()
        hero_stats = get_hero_stats()
        pt_payout_stats = get_pt_payout_stats()
        signal_stats = get_signal_stats()
        payout_dist = get_payout_distribution()
        daily_label = get_daily_label_stats()
        return render_template("stats.html",
                               pt_stats=pt_stats,
                               label_stats=label_stats,
                               daily=daily,
                               streaks=streaks,
                               cumulative=cumulative,
                               venue_stats=venue_stats,
                               grade_stats=grade_stats,
                               all_streaks=all_streaks,
                               hero_stats=hero_stats,
                               pt_payout_stats=pt_payout_stats,
                               signal_stats=signal_stats,
                               payout_dist=payout_dist,
                               daily_label=daily_label)

    @app.route("/api/today")
    def api_today():
        date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        return jsonify(get_today_predictions(date))

    @app.route("/api/pt_stats")
    def api_pt_stats():
        return jsonify(get_pt_stats())

    @app.route("/api/recent_activity")
    def api_recent_activity():
        date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        return jsonify(get_recent_activity(date))

    @app.route("/api/activity_stream")
    def api_activity_stream():
        date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        def generate():
            last_ts = None
            while True:
                rows = get_recent_activity(date)
                new_ts = rows[0]["created_at"] if rows else None
                if new_ts != last_ts:
                    last_ts = new_ts
                    yield f"data: {json.dumps(rows, ensure_ascii=False)}\n\n"
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
    app.run(host="0.0.0.0", port=5001, debug=False)
