from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

VENUES = [
    "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖",
    "蒲郡", "常滑", "津", "三国", "びわこ", "住之江",
    "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山",
    "下関", "若松", "芦屋", "福岡", "唐津", "大村"
]

WIND_OPTIONS = [
    "向かい風1m", "向かい風2m", "向かい風3m", "向かい風4m", "向かい風5m以上",
    "追い風1m", "追い風2m", "追い風3m", "追い風4m", "追い風5m以上",
    "左横風1m", "左横風2m", "左横風3m以上",
    "右横風1m", "号横風2m", "右横風3m以上",
]


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def predict(boats, kimari=None):
    """
    boats: list of 6 dicts (course, exhibit_time, avg_st, tilt, win1_rate, win2_rate ...)
    kimari: optional dict from 決まり手/逃げ data:
      nige_2nd: list[5] → 逃がし2着率 for courses 2-6
      nige_odds: dict {course_str: pct} → 出目確率 e.g. {"2":17.8,"3":15.6,...}
      sashi:  list[6] → 差し率 per course index 0-5
      makuri: list[6] → 捲り率
      makuri_sashi: list[6] → 捲り差し率
    """
    course_to_boat = {}
    for i, b in enumerate(boats):
        c = int(safe_float(b.get("course", i + 1), i + 1))
        course_to_boat[c] = i

    def boat_num(course_num):
        idx = course_to_boat.get(course_num)
        return idx + 1 if idx is not None else course_num

    def base_score(boat_index):
        b = boats[boat_index]
        et   = safe_float(b.get("exhibit_time"), 6.9)
        st   = safe_float(b.get("avg_st"), 0.18)
        tilt = safe_float(b.get("tilt"), 0)
        w1   = safe_float(b.get("win1_rate"), 0)
        w2   = safe_float(b.get("win2_rate"), 0)
        base = et - (0.18 - st) * 1.5 - tilt * 0.05
        win_bonus = (w1 / 100.0) * 0.3 + (w2 / 100.0) * 0.1
        return base - win_bonus

    scores = {c: base_score(idx) for c, idx in course_to_boat.items()}

    # 逃がし2着率からスコアを上書き（大きいほど良い）
    nige_2nd = {}
    if kimari and kimari.get("nige_2nd"):
        for idx, rate in enumerate(kimari["nige_2nd"]):
            course_num = idx + 2  # course 2-6
            if course_num in course_to_boat:
                nige_2nd[course_num] = safe_float(rate, 0)

    # 出目確率
    nige_odds = {}
    if kimari and kimari.get("nige_odds"):
        for k, v in kimari["nige_odds"].items():
            nige_odds[int(k)] = safe_float(v, 0)

    def fmt(a, b, c):
        return f"{boat_num(a)}-{boat_num(b)}-{boat_num(c)}"

    results = []

    # === 1コース逃げ ===
    if 1 in course_to_boat:
        others = [c for c in [2, 3, 4, 5, 6] if c in course_to_boat]
        if nige_2nd:
            # 逃がし2着率でソート（高いほど良い）
            second_cands = sorted(others, key=lambda c: -nige_2nd.get(c, 0))
        else:
            second_cands = sorted(others, key=lambda c: scores.get(c, 99))
        if len(second_cands) >= 2:
            # 出目確率があれば上位2特を本命・対抗に使用
            if nige_odds:
                top2 = sorted(others, key=lambda c: -nige_odds.get(c, 0))[:2]
                rest = [c for c in second_cands if c not in top2]
                third_cands = sorted(rest or [c for c in others if c != top2[0]],
                                     key=lambda c: -nige_2nd.get(c, 0) if nige_2nd else scores.get(c, 99))
                if len(top2) >= 2:
                    t3 = third_cands[0] if third_cands else top2[1]
                    results.append(("逃げ", fmt(1, top2[0], top2[1])))
                    results.append(("逃げ", fmt(1, top2[0], t3)))
                    results.append(("逃げ", fmt(1, top2[1], top2[0])))
            else:
                results.append(("逃げ", fmt(1, second_cands[0], second_cands[1])))
                results.append(("逃げ", fmt(1, second_cands[1], second_cands[0])))

    # === 4コースまくり ===
    if 4 in course_to_boat:
        rest56 = sorted([c for c in [5, 6] if c in course_to_boat], key=lambda c: scores.get(c, 99))
        if rest56:
            third_cands = sorted([c for c in course_to_boat if c not in [4, rest56[0]]],
                                  key=lambda c: scores.get(c, 99))
            if third_cands:
                results.append(("まくり", fmt(4, rest56[0], third_cands[0])))
        if 1 in course_to_boat:
            third_cands2 = sorted([c for c in course_to_boat if c not in [4, 1]],
                                   key=lambda c: scores.get(c, 99))
            if third_cands2:
                results.append(("まくり差し", fmt(4, 1, third_cands2[0])))

    # === 5コースまくり差し → 2着は1号艇 ===
    if 5 in course_to_boat and 1 in course_to_boat:
        third_cands = sorted([c for c in course_to_boat if c not in [5, 1]],
                             key=lambda c: scores.get(c, 99))
        if third_cands:
            results.append(("まくり差し", fmt(5, 1, third_cands[0])))

    # === 6コースまくり差し → 2着は1号艇 ===
    if 6 in course_to_boat and 1 in course_to_boat:
        third_cands = sorted([c for c in course_to_boat if c not in [6, 1]],
                             key=lambda c: scores.get(c, 99))
        if third_cands:
            results.append(("まくり差し", fmt(6, 1, third_cands[0])))

    if not results:
        return {"honmei": "データ不足", "taikou": "-", "ana": "-", "patterns": []}

    return {
        "honmei": results[0][1],
        "taikou": results[1][1] if len(results) > 1 else "-",
        "ana":    results[2][1] if len(results) > 2 else "-",
        "patterns": [(p, c) for p, c in results]
    }


@app.route("/")
def index():
    return render_template("index.html", venues=VENUES, wind_options=WIND_OPTIONS)


@app.route("/predict", methods=["POST"])
def do_predict():
    data = request.get_json()
    boats = data.get("boats", [])
    kimari = data.get("kimari", None)
    result = predict(boats, kimari=kimari)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
