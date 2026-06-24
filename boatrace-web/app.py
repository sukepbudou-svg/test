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
    "右横風1m", "右横風2m", "右横風3m以上",
]


def predict(boats):
    course_to_boat = {}
    for i, b in enumerate(boats):
        c = int(b.get("course", i + 1))
        course_to_boat[c] = i

    def boat_num(course_num):
        idx = course_to_boat.get(course_num)
        return idx + 1 if idx is not None else course_num

    def score(boat_index):
        b = boats[boat_index]
        try:
            et = float(b.get("exhibit_time") or 6.9)
        except (ValueError, TypeError):
            et = 6.9
        try:
            st = float(b.get("avg_st") or 0.18)
        except (ValueError, TypeError):
            st = 0.18
        try:
            tilt = float(b.get("tilt") or 0)
        except (ValueError, TypeError):
            tilt = 0
        try:
            win1 = float(b.get("win1_rate") or 0)
        except (ValueError, TypeError):
            win1 = 0
        try:
            win2 = float(b.get("win2_rate") or 0)
        except (ValueError, TypeError):
            win2 = 0

        # スコア: 小さいほど良い。展示タイム・平均STは物理的能力、勝率は実績を反映
        base = et - (0.18 - st) * 1.5 - tilt * 0.05
        # 勝率ボーナス: 1着率・2連対率が高いとスコアを下げる（良い方向）
        win_bonus = (win1 / 100.0) * 0.3 + (win2 / 100.0) * 0.1
        return base - win_bonus

    scores = {c: score(idx) for c, idx in course_to_boat.items()}

    def fmt(a, b, c):
        return f"{boat_num(a)}-{boat_num(b)}-{boat_num(c)}"

    results = []

    # 1コース逃げ
    if 1 in course_to_boat:
        inner = [c for c in [2, 3, 4] if c in course_to_boat]
        outer = [c for c in [5, 6] if c in course_to_boat]
        second_cands = sorted(inner + outer, key=lambda c: scores.get(c, 99))
        if len(second_cands) >= 2:
            results.append(("逃げ", fmt(1, second_cands[0], second_cands[1])))
            results.append(("逃げ", fmt(1, second_cands[1], second_cands[0])))

    # 4コースまくり
    if 4 in course_to_boat:
        rest = sorted([c for c in [5, 6] if c in course_to_boat], key=lambda c: scores.get(c, 99))
        if rest:
            third_cands = sorted([c for c in course_to_boat if c not in [4, rest[0]]], key=lambda c: scores.get(c, 99))
            if third_cands:
                results.append(("まくり", fmt(4, rest[0], third_cands[0])))
        if 1 in course_to_boat:
            third_cands2 = sorted([c for c in course_to_boat if c not in [4, 1]], key=lambda c: scores.get(c, 99))
            if third_cands2:
                results.append(("まくり差し", fmt(4, 1, third_cands2[0])))

    # 5コースまくり差し → 2着は1号艇
    if 5 in course_to_boat and 1 in course_to_boat:
        third_cands = sorted([c for c in course_to_boat if c not in [5, 1]], key=lambda c: scores.get(c, 99))
        if third_cands:
            results.append(("まくり差し", fmt(5, 1, third_cands[0])))

    # 6コースまくり差し → 2着は1号艇
    if 6 in course_to_boat and 1 in course_to_boat:
        third_cands = sorted([c for c in course_to_boat if c not in [6, 1]], key=lambda c: scores.get(c, 99))
        if third_cands:
            results.append(("まくり差し", fmt(6, 1, third_cands[0])))

    if not results:
        return {"honmei": "データ不足", "taikou": "-", "ana": "-", "patterns": []}

    return {
        "honmei": results[0][1] if len(results) > 0 else "-",
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
    result = predict(boats)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
