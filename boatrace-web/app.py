from flask import Flask, render_template, request, jsonify
import itertools

app = Flask(__name__)

VENUES = [
    "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖",
    "蒲郡", "常滑", "津", "三国", "びわこ", "住之江",
    "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山",
    "下関", "若松", "芦屋", "福岡", "唐津", "大村"
]

WIND_OPTIONS = [
    "向かい風1m", "向かい風2m", "向かい風3m", "向かい風4m",
    "向かい風5m", "向かい風6m以上",
    "追い風1m", "追い風2m", "追い風3m", "追い風4m",
    "追い風5m", "追い風6m以上",
    "横風（左）強", "横風（右）強",
    "無風", "その他"
]

# 会場プロファイル
# in_rate: 1コース逃げのしやすさ補正（高いほどイン有利）
# upset:   荒れやすさ補正（高いほど外枠を加点）
# wind:    風の影響度（高いほど風向きの影響が大きい）
VENUE_PROFILES = {
    "桐生":  {"in_rate": 0.5,  "upset": 0.3, "wind": 0.4},
    "戸田":  {"in_rate": 0.3,  "upset": 0.5, "wind": 0.3},
    "江戸川":{"in_rate": -0.5, "upset": 1.5, "wind": 1.5},  # 最も荒れやすく風影響大
    "平和島":{"in_rate": 0.0,  "upset": 0.8, "wind": 0.5},
    "多摩川":{"in_rate": 0.5,  "upset": 0.3, "wind": 0.2},
    "浜名湖":{"in_rate": 0.0,  "upset": 0.8, "wind": 1.2},  # 風影響大
    "蒲郡":  {"in_rate": 0.5,  "upset": 0.3, "wind": 0.3},
    "常滑":  {"in_rate": 0.8,  "upset": 0.2, "wind": 0.4},
    "津":    {"in_rate": 0.3,  "upset": 0.4, "wind": 0.5},
    "三国":  {"in_rate": -0.3, "upset": 1.0, "wind": 1.0},  # 荒れやすい・風影響
    "びわこ":{"in_rate": 0.0,  "upset": 0.6, "wind": 0.8},
    "住之江":{"in_rate": 1.2,  "upset": 0.1, "wind": 0.1},  # 最もイン有利・堅い
    "尼崎":  {"in_rate": 0.8,  "upset": 0.2, "wind": 0.2},
    "鳴門":  {"in_rate": 0.3,  "upset": 0.5, "wind": 0.8},
    "丸亀":  {"in_rate": 0.3,  "upset": 0.5, "wind": 0.7},
    "児島":  {"in_rate": 0.5,  "upset": 0.4, "wind": 0.5},
    "宮島":  {"in_rate": 0.3,  "upset": 0.6, "wind": 0.6},
    "徳山":  {"in_rate": 0.8,  "upset": 0.2, "wind": 0.3},
    "下関":  {"in_rate": 0.5,  "upset": 0.4, "wind": 0.5},
    "若松":  {"in_rate": 0.3,  "upset": 0.6, "wind": 0.7},
    "芦屋":  {"in_rate": 0.8,  "upset": 0.2, "wind": 0.3},
    "福岡":  {"in_rate": 0.5,  "upset": 0.4, "wind": 0.4},
    "唐津":  {"in_rate": 0.3,  "upset": 0.5, "wind": 0.6},
    "大村":  {"in_rate": 1.5,  "upset": 0.0, "wind": 0.1},  # 最もイン勝率高い
}

# 風向きの向かい風強度（強いほど外枠有利）
WIND_UPSET_BONUS = {
    "向かい風1m": 0.0, "向かい風2m": 0.1, "向かい風3m": 0.3,
    "向かい風4m": 0.5, "向かい風5m": 0.8, "向かい風6m以上": 1.2,
    "追い風1m": 0.0,  "追い風2m": 0.0,  "追い風3m": 0.1,
    "追い風4m": 0.2,  "追い風5m": 0.3,  "追い風6m以上": 0.5,
    "横風（左）強": 0.6, "横風（右）強": 0.6,
    "無風": 0.0, "その他": 0.0,
}

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def predict(boats, kimari=None, venue=None, wind=None):
    # 会場プロファイル取得
    vp = VENUE_PROFILES.get(venue, {"in_rate": 0.0, "upset": 0.5, "wind": 0.5})
    wind_bonus = WIND_UPSET_BONUS.get(wind, 0.0)
    # 風影響度 = 会場の風感度 × 風の強さ
    wind_effect = vp["wind"] * wind_bonus

    scores = []
    for b in boats:
        course = int(b.get('course', b.get('boat_number', 1)))
        et = safe_float(b.get('exhibit_time', 0))
        avg_st = safe_float(b.get('avg_st', 0))
        tilt = safe_float(b.get('tilt', 0))
        is_f = b.get('is_f', False)
        win1 = safe_float(b.get('win1_rate', 0))
        win2 = safe_float(b.get('win2_rate', 0))
        motor_win1 = safe_float(b.get('motor_win1', 0))
        motor_contrib = safe_float(b.get('motor_contrib', 0))

        # Base score: lower exhibit time = better, lower avg ST = better
        base = 0.0
        if et > 0:
            base -= (et - 6.7) * 10
        if avg_st > 0:
            base -= (avg_st - 0.15) * 8
        # Tilt penalty
        base -= tilt * 2
        # F持ちペナルティ
        if is_f:
            base -= 1.5

        # Player bonus
        player_bonus = win1 * 0.3 + win2 * 0.1

        # Motor bonus
        motor_bonus = motor_win1 * 0.15 + motor_contrib * 0.05

        score = base + player_bonus + motor_bonus

        # コース補正（基本）
        course_base = {1: 3.0, 2: 0.5, 3: 0.2, 4: 0.0, 5: -0.2, 6: -0.5}
        score += course_base.get(course, 0.0)

        # 会場イン有利補正: 1コースをさらに加点
        if course == 1:
            score += vp["in_rate"]

        # 荒れやすい場・向かい風: 外枠（4〜6コース）を加点
        upset_effect = vp["upset"] + wind_effect
        if course == 4:
            score += upset_effect * 0.4
        elif course == 5:
            score += upset_effect * 0.5
        elif course == 6:
            score += upset_effect * 0.3

        scores.append({'boat': b, 'score': score, 'course': course})

    scores.sort(key=lambda x: x['score'], reverse=True)

    # Determine predicted patterns
    results = []
    top = scores[0]
    c = top['course']

    # Apply kimari data if available
    nige_2nd_rates = {}
    if kimari and 'sim' in kimari:
        sim = kimari['sim']
        # sim: {1: {nige_2nd_rates: {2: 30.0, 3: 20.0, ...}}}
        if c in sim and 'nige_2nd_rates' in sim[c]:
            nige_2nd_rates = sim[c]['nige_2nd_rates']

    def get_boat_by_course(course_num):
        for s in scores:
            if s['course'] == course_num:
                return s
        return None

    def score_order_except(exclude_courses):
        return [s for s in scores if s['course'] not in exclude_courses]

    boat1 = get_boat_by_course(1)

    # Course 1: 逃げ
    if c == 1:
        first = top['boat']['boat_number']
        if nige_2nd_rates:
            sorted_2nd = sorted(nige_2nd_rates.items(), key=lambda x: x[1], reverse=True)
            seconds = [get_boat_by_course(int(k)) for k, v in sorted_2nd[:3] if get_boat_by_course(int(k))]
        else:
            seconds = score_order_except([1])[:3]
        thirds = score_order_except([1])[:4]
        for s2 in seconds[:2]:
            for s3 in thirds[:3]:
                if s3['course'] != s2['course']:
                    results.append({
                        'type': '本命',
                        'combo': f"{first}-{s2['boat']['boat_number']}-{s3['boat']['boat_number']}"
                    })
                    if len(results) >= 3:
                        break
            if len(results) >= 3:
                break

    # Course 4: まくり or まくり差し → 2着は1号艇
    elif c == 4:
        first = top['boat']['boat_number']
        second = boat1['boat']['boat_number'] if boat1 else scores[1]['boat']['boat_number']
        thirds = score_order_except([c, 1])[:3]
        for s3 in thirds:
            results.append({
                'type': '本命' if len(results) == 0 else '対抗',
                'combo': f"{first}-{second}-{s3['boat']['boat_number']}"
            })

    # Course 5: まくり差し → 2着は1号艇
    elif c == 5:
        first = top['boat']['boat_number']
        second = boat1['boat']['boat_number'] if boat1 else scores[1]['boat']['boat_number']
        thirds = score_order_except([c, 1])[:3]
        for s3 in thirds:
            results.append({
                'type': '本命' if len(results) == 0 else '対抗',
                'combo': f"{first}-{second}-{s3['boat']['boat_number']}"
            })

    # Course 6: まくり差し → 2着は1号艇
    elif c == 6:
        first = top['boat']['boat_number']
        second = boat1['boat']['boat_number'] if boat1 else scores[1]['boat']['boat_number']
        thirds = score_order_except([c, 1])[:3]
        for s3 in thirds:
            results.append({
                'type': '本命' if len(results) == 0 else '対抗',
                'combo': f"{first}-{second}-{s3['boat']['boat_number']}"
            })

    # Other courses: score-based
    else:
        for i, s1 in enumerate(scores[:2]):
            for s2 in scores[:4]:
                if s2['course'] == s1['course']:
                    continue
                for s3 in scores[:5]:
                    if s3['course'] in [s1['course'], s2['course']]:
                        continue
                    results.append({
                        'type': ['本命', '対抗', '穴'][min(i, 2)],
                        'combo': f"{s1['boat']['boat_number']}-{s2['boat']['boat_number']}-{s3['boat']['boat_number']}"
                    })
                    if len(results) >= 6:
                        break
                if len(results) >= 6:
                    break
            if len(results) >= 6:
                break

    # Fill穴 if < 6 results
    if len(results) < 6:
        for s1 in scores[:3]:
            for s2 in scores[:4]:
                if s2['course'] == s1['course']:
                    continue
                for s3 in scores[:5]:
                    if s3['course'] in [s1['course'], s2['course']]:
                        continue
                    combo = f"{s1['boat']['boat_number']}-{s2['boat']['boat_number']}-{s3['boat']['boat_number']}"
                    if not any(r['combo'] == combo for r in results):
                        results.append({'type': '穴', 'combo': combo})
                    if len(results) >= 6:
                        break
                if len(results) >= 6:
                    break
            if len(results) >= 6:
                break

    return {
        'predictions': results[:6],
        'score_order': [{'course': s['course'], 'boat_number': s['boat']['boat_number'], 'score': round(s['score'], 2)} for s in scores]
    }


@app.route('/')
def index():
    return render_template('index.html', venues=VENUES, wind_options=WIND_OPTIONS)


@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    boats = data.get('boats', [])
    kimari = data.get('kimari', None)
    venue = data.get('venue', None)
    wind = data.get('wind', None)
    result = predict(boats, kimari, venue=venue, wind=wind)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
