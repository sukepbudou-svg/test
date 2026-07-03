from flask import Flask, render_template, request, jsonify
import itertools
import sqlite3
import json
import os
import re
import urllib.request
from datetime import datetime

app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), 'history.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_date TEXT NOT NULL,
            venue TEXT NOT NULL,
            race_no INTEGER NOT NULL,
            predictions TEXT NOT NULL,
            result_1st INTEGER,
            result_2nd INTEGER,
            result_3rd INTEGER,
            payout INTEGER,
            purchase INTEGER,
            is_hit INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            nige_rate REAL,
            wind TEXT
        )
    ''')
    conn.commit()
    # 既存DBへのマイグレーション
    try:
        conn.execute('ALTER TABLE records ADD COLUMN nige_rate REAL')
        conn.commit()
    except Exception:
        pass
    try:
        conn.execute('ALTER TABLE records ADD COLUMN wind TEXT')
        conn.commit()
    except Exception:
        pass
    conn.close()

init_db()

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
    "横風（左）1m", "横風（左）2m", "横風（左）3m以上",
    "横風（右）1m", "横風（右）2m", "横風（右）3m以上",
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

# 会場×コース別1着率（直近3ヶ月実績、単位：%）
VENUE_COURSE_RATES = {
    "桐生":  [49.9, 14.4, 11.5, 14.1,  9.0,  2.2],
    "戸田":  [39.6, 17.9, 19.4, 14.6,  7.2,  2.3],
    "江戸川":[48.8, 14.7, 16.6, 12.6,  5.8,  2.8],
    "平和島":[49.1, 15.0, 16.5, 11.0,  7.4,  2.3],
    "多摩川":[54.0, 14.9, 15.1,  9.4,  5.6,  1.6],
    "浜名湖":[55.2, 13.4, 14.2, 10.6,  5.5,  2.1],
    "蒲郡":  [56.4, 12.3, 10.4, 13.7,  5.9,  2.1],
    "常滑":  [55.7, 13.8, 10.7, 11.1,  6.6,  2.5],
    "津":    [58.1, 13.8, 12.4, 10.0,  5.7,  0.7],
    "三国":  [49.1, 17.9, 15.0,  9.9,  7.2,  2.3],
    "びわこ":[52.5, 13.9, 16.3,  9.4,  7.0,  1.6],
    "住之江":[61.8, 11.6, 11.0, 10.4,  4.5,  1.4],
    "尼崎":  [61.4, 13.5, 10.9,  8.5,  5.0,  1.4],
    "鳴門":  [47.2, 15.2, 17.0, 10.0,  8.2,  3.2],
    "丸亀":  [59.1, 13.5, 11.6,  8.0,  6.7,  2.0],
    "児島":  [57.8, 11.0, 13.2, 10.6,  5.9,  2.2],
    "宮島":  [58.1, 12.5, 12.6, 10.7,  5.6,  1.7],
    "徳山":  [65.0, 10.7, 11.1,  7.9,  4.5,  1.2],
    "下関":  [64.5, 10.0, 11.6,  7.9,  5.4,  1.3],
    "若松":  [59.3, 15.7, 11.3,  8.2,  4.1,  1.9],
    "芦屋":  [61.8,  9.9, 10.0, 10.1,  7.3,  2.3],
    "福岡":  [62.0, 14.0, 14.8,  6.4,  2.5,  1.0],
    "唐津":  [57.9, 15.1, 11.4,  9.3,  5.7,  1.6],
    "大村":  [59.0, 13.0, 10.7, 10.5,  6.7,  1.3],
}

# 全会場平均（会場未選択時のフォールバック用）
DEFAULT_COURSE_RATES = [54.6, 13.9, 13.8, 10.8,  6.2,  2.0]

# 風向きの向かい風強度（強いほど外枠有利）
WIND_UPSET_BONUS = {
    "向かい風1m": 0.0, "向かい風2m": 0.1, "向かい風3m": 0.3,
    "向かい風4m": 0.5, "向かい風5m": 0.8, "向かい風6m以上": 1.2,
    "追い風1m": 0.0,  "追い風2m": 0.0,  "追い風3m": 0.1,
    "追い風4m": 0.2,  "追い風5m": 0.3,  "追い風6m以上": 0.5,
    "横風（左）1m": 0.2, "横風（左）2m": 0.4, "横風（左）3m以上": 0.7,
    "横風（右）1m": 0.2, "横風（右）2m": 0.4, "横風（右）3m以上": 0.7,
    "無風": 0.0, "その他": 0.0,
}

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def predict(boats, kimari=None, venue=None, wind=None, nige_rate=None, force_arekote=False):
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
        lap = safe_float(b.get('lap', 0))
        avg_lap = safe_float(b.get('avg_lap', 0))
        straight = safe_float(b.get('straight', 0))
        avg_straight = safe_float(b.get('avg_straight', 0))
        mawariashi = safe_float(b.get('mawariashi', 0))
        avg_mawari = safe_float(b.get('avg_mawari', 0))
        player_class = b.get('player_class', '')

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

        # 級別ST修正能力補正（展示STが悪い時ほど級の差が出る）
        exhibit_st = safe_float(b.get('exhibit_st', 0))
        st_penalty = max(0, (exhibit_st - 0.15)) if exhibit_st > 0 else 0
        class_bonus = {'A1': 0.8, 'A2': 0.4, 'B1': 0.0, 'B2': -0.5}.get(player_class, 0.0)
        base += class_bonus * (1 + st_penalty * 10)

        # 周回タイム: 当日×0.6 + 平均×0.4 (小さいほど速い=良い)
        if lap > 0 and avg_lap > 0:
            combined_lap = lap * 0.6 + avg_lap * 0.4
            base += (36.0 - combined_lap) * 0.5
        elif lap > 0:
            base += (36.0 - lap) * 0.5

        # 直線タイム: 当日×0.6 + 平均×0.4 (小さいほど速い=良い)
        if straight > 0 and avg_straight > 0:
            combined_str = straight * 0.6 + avg_straight * 0.4
            base += (6.5 - combined_str) * 3.0
        elif straight > 0:
            base += (6.5 - straight) * 3.0

        # まわり足: 当日×0.6 + 平均×0.4 (小さいほど速い=良い)
        if mawariashi > 0 and avg_mawari > 0:
            combined_maw = mawariashi * 0.6 + avg_mawari * 0.4
            base += (5.0 - combined_maw) * 0.5
        elif mawariashi > 0:
            base += (5.0 - mawariashi) * 0.5

        # Player bonus
        player_bonus = win1 * 0.3 + win2 * 0.1

        # Motor bonus
        motor_bonus = motor_win1 * 0.15 + motor_contrib * 0.05

        score = base + player_bonus + motor_bonus

        # 会場×コース別1着率をベーススコアに使用（%→スケール変換）
        course_rates = VENUE_COURSE_RATES.get(venue, DEFAULT_COURSE_RATES)
        # 1着率をそのままスコアに（50%=+5, 10%=+1 程度のスケール）
        score += course_rates[course - 1] * 0.1

        # 荒れやすい場・向かい風: 外枠（4〜6コース）を加点
        upset_effect = vp["upset"] + wind_effect
        if course == 4:
            score += upset_effect * 0.4
        elif course == 5:
            score += upset_effect * 0.5
        elif course == 6:
            score += upset_effect * 0.3

        scores.append({'boat': b, 'score': score, 'course': course, 'exhibit_st': exhibit_st})

    # ===== 展開補正: コース間のST比較で差し/まくりリスクを反映 =====
    # exhibit_stが入力されている艇だけ対象（0は未入力とみなす）
    st_by_course = {s['course']: s['exhibit_st'] for s in scores if s['exhibit_st'] > 0}

    if len(st_by_course) >= 2:
        def get_st(c): return st_by_course.get(c, 0.20)  # 未入力は遅め扱い

        st1 = get_st(1); st2 = get_st(2); st3 = get_st(3)
        st4 = get_st(4); st5 = get_st(5); st6 = get_st(6)

        for s in scores:
            c = s['course']
            adj = 0.0

            if c == 1:
                # 2コースとの差: 2が速いほどインの逃げリスク上昇
                sashi_threat = max(0, (st1 - st2) - 0.03)  # 0.03以上の差で差しリスク
                adj -= sashi_threat * 15
                # 3〜4コースがまくってきてもインは関係薄い
                # 2コースより内側の艇がいない場合は安全
                if st1 < st2 - 0.02:  # 1コースが明確に速い → 逃げ安全
                    adj += 0.5

            elif c == 2:
                # 1コースより速い → 差し期待値UP
                if st2 < st1 - 0.02:
                    adj += 0.6
                # 3コースが2より速い → 2コースは外から被される
                outer_threat = max(0, (st2 - st3) - 0.02)
                adj -= outer_threat * 10

            elif c == 3:
                # 2コース以内より速い → まくり期待値UP
                inner_avg = (st1 + st2) / 2
                if st3 < inner_avg - 0.03:
                    adj += 0.7
                # 4コースより遅い → 外から被される
                outer_threat = max(0, (st3 - st4) - 0.02)
                adj -= outer_threat * 8

            elif c == 4:
                # 3コース以内より速い → まくり期待値UP
                inner_avg = (st1 + st2 + st3) / 3
                if st4 < inner_avg - 0.04:
                    adj += 0.8
                # 5コースより遅い → 被される
                outer_threat = max(0, (st4 - st5) - 0.02)
                adj -= outer_threat * 6

            elif c == 5:
                inner_avg = (st2 + st3 + st4) / 3
                if st5 < inner_avg - 0.04:
                    adj += 0.6

            elif c == 6:
                inner_avg = (st3 + st4 + st5) / 3
                if st6 < inner_avg - 0.05:
                    adj += 0.5

            # 風の影響: 向かい風は外枠のまくりをさらに後押し
            if wind_effect > 0 and c >= 3:
                adj += wind_effect * 0.2

            s['score'] += adj

    scores.sort(key=lambda x: x['score'], reverse=True)

    def get_boat_by_course(course_num):
        for s in scores:
            if s['course'] == course_num:
                return s
        return None

    # 決まり手データから2着重み付けを取得
    nige_2nd_rates = {}
    top_course = scores[0]['course']
    if kimari and 'sim' in kimari:
        sim = kimari['sim']
        if top_course in sim and 'nige_2nd_rates' in sim[top_course]:
            nige_2nd_rates = sim[top_course]['nige_2nd_rates']

    # スコア順に全6艇を並べた候補リストを生成（最大12候補）
    candidates = []
    seen = set()
    for s1 in scores[:4]:
        bn1 = s1['boat']['boat_number']
        # 2着候補: 決まり手データがあれば優先、なければスコア順
        if nige_2nd_rates and s1['course'] == top_course:
            sorted_2nd_courses = [int(k) for k, _ in sorted(nige_2nd_rates.items(), key=lambda x: x[1], reverse=True)]
            seconds = [get_boat_by_course(c) for c in sorted_2nd_courses if get_boat_by_course(c)]
            seconds += [s for s in scores if s['boat']['boat_number'] != bn1 and s not in seconds]
        else:
            seconds = [s for s in scores if s['boat']['boat_number'] != bn1]
        for s2 in seconds[:4]:
            bn2 = s2['boat']['boat_number']
            for s3 in scores:
                bn3 = s3['boat']['boat_number']
                if bn3 in (bn1, bn2):
                    continue
                combo = f"{bn1}-{bn2}-{bn3}"
                if combo not in seen:
                    seen.add(combo)
                    # 複合スコア = 1着スコア×3 + 2着スコア×2 + 3着スコア
                    combined = s1['score'] * 3 + s2['score'] * 2 + s3['score']
                    candidates.append({'combo': combo, 'combined': combined, 'type': None})
                if len(candidates) >= 60:
                    break
            if len(candidates) >= 60:
                break
        if len(candidates) >= 60:
            break

    candidates.sort(key=lambda x: x['combined'], reverse=True)

    # ===== 本命2点: スコア1位の艇が1着の上位2コンボ =====
    honmei = []
    top_bn = scores[0]['boat']['boat_number']
    for c in candidates:
        if c['combo'].startswith(f"{top_bn}-"):
            honmei.append(c)
        if len(honmei) >= 2:
            break

    # ===== 対抗3点: 3つの異なるシナリオをカバー =====
    taikou = []
    used_combos = {c['combo'] for c in honmei}

    # --- 対抗1点目: スコア2位の艇が1着（展開逆転シナリオ）---
    if len(scores) >= 2:
        second_bn = scores[1]['boat']['boat_number']
        for c in candidates:
            if c['combo'].startswith(f"{second_bn}-") and c['combo'] not in used_combos:
                taikou.append(c)
                used_combos.add(c['combo'])
                break

    # --- 対抗2点目: 1位1着 + STが最速の艇が2着（展示ST展開シナリオ）---
    st_entries = [(s['course'], s['exhibit_st'], s['boat']['boat_number'])
                  for s in scores if s.get('exhibit_st', 0) > 0
                  and s['boat']['boat_number'] != top_bn]
    if st_entries:
        best_st_bn = min(st_entries, key=lambda x: x[1])[2]  # STが小さい=速い
        for c in candidates:
            parts = c['combo'].split('-')
            if parts[0] == str(top_bn) and parts[1] == str(best_st_bn) and c['combo'] not in used_combos:
                taikou.append(c)
                used_combos.add(c['combo'])
                break

    # --- 対抗3点目: 残りスコア上位コンボから本命と被らないもの ---
    for c in candidates:
        if c['combo'] not in used_combos:
            taikou.append(c)
            used_combos.add(c['combo'])
            break

    # 不足分はスコア順で補完
    for c in candidates:
        if len(honmei) >= 2 and len(taikou) >= 3:
            break
        if c['combo'] not in used_combos:
            if len(honmei) < 2:
                honmei.append(c)
            elif len(taikou) < 3:
                taikou.append(c)
            used_combos.add(c['combo'])

    results = []
    for c in honmei[:2]:
        results.append({'combo': c['combo'], 'type': '本命', 'combined': round(c['combined'], 2)})
    for c in taikou[:3]:
        results.append({'combo': c['combo'], 'type': '対抗', 'combined': round(c['combined'], 2)})

    chaos = calc_chaos(scores, boats, vp, wind_effect, nige_rate)

    # 荒れモード: 強制フラグのときのみ（自動切り替えは廃止）
    if force_arekote:
        arekote = predict_arekote(scores, candidates, wind, wind_effect)
        return {
            'predictions': results,
            'candidates': [{'combo': c['combo'], 'combined': round(c['combined'], 2)} for c in candidates[:60]],
            'score_order': [{'course': s['course'], 'boat_number': s['boat']['boat_number'], 'score': round(s['score'], 2)} for s in scores],
            'chaos': chaos,
            'arekote_mode': True,
            'arekote_predictions': arekote,
        }

    return {
        'predictions': results,
        'candidates': [{'combo': c['combo'], 'combined': round(c['combined'], 2)} for c in candidates[:60]],
        'score_order': [{'course': s['course'], 'boat_number': s['boat']['boat_number'], 'score': round(s['score'], 2)} for s in scores],
        'chaos': chaos,
    }


def predict_arekote(scores, candidates, wind, wind_effect):
    """
    荒れモード予想（イン逃げ率 < 50% 時）
    万舟×2 + 裏熊×3（展開重視、1号艇除外）+ 裏熊×1（差し/まくり判断で1号艇2or3着）
    """
    # 万舟×2: 既存候補リストの最後尾（低スコア＝高倍率）上位2点
    # candidatesはスコア降順なので末尾から取る（ただし候補が少ない場合は後半から）
    manzoku = []
    seen_manzoku = set()
    for c in reversed(candidates):
        if c['combo'] not in seen_manzoku:
            manzoku.append({'combo': c['combo'], 'type': '万舟'})
            seen_manzoku.add(c['combo'])
        if len(manzoku) >= 2:
            break

    # 展開スコア算出（1号艇除外）
    # 追い風/横風: 3-4コース加点、向かい風: 2号艇加点
    wind_str = wind or ''
    is_mukai = '向かい風' in wind_str
    is_oi_or_yoko = any(w in wind_str for w in ['追い風', '横風'])

    outer_boats = [s for s in scores if s['course'] != 1]
    tenkai_scores = []
    for s in outer_boats:
        ts = s['score']
        c = s['course']
        if is_mukai and c == 2:
            ts += wind_effect * 1.5
        if is_oi_or_yoko and c in (3, 4):
            ts += wind_effect * 1.2
        tenkai_scores.append({'course': c, 'boat_number': s['boat']['boat_number'], 'ts': ts})

    tenkai_scores.sort(key=lambda x: x['ts'], reverse=True)
    top_tenkai = tenkai_scores[:4]

    # 裏熊×3: 展開スコア上位4艇の3連単コンボ（スコア上位3点）
    import itertools
    ura_combos = []
    seen_ura = set()
    top_bns = [t['boat_number'] for t in top_tenkai]
    for perm in itertools.permutations(top_bns, 3):
        combo_str = f"{perm[0]}-{perm[1]}-{perm[2]}"
        if combo_str in seen_ura:
            continue
        seen_ura.add(combo_str)
        # スコア = 1着TS×3 + 2着TS×2 + 3着TS
        ts_map = {t['boat_number']: t['ts'] for t in top_tenkai}
        combined = ts_map[perm[0]] * 3 + ts_map[perm[1]] * 2 + ts_map[perm[2]]
        ura_combos.append({'combo': combo_str, 'combined': combined})

    ura_combos.sort(key=lambda x: x['combined'], reverse=True)
    ura_3 = [{'combo': u['combo'], 'type': '裏熊'} for u in ura_combos[:3]]

    # 裏熊×1: 差し/まくり判断で1号艇を2着or3着に配置
    # まくり要因: 追い風/横風 + 3-4コースが上位スコア → 1号艇3着
    # 差し要因: 向かい風 + 2コース上位 → 1号艇2着
    outer_top3_courses = [s['course'] for s in outer_boats[:3]]
    makuri_score = 0
    sashi_score = 0
    if is_oi_or_yoko:
        makuri_score += wind_effect
    if is_mukai:
        sashi_score += wind_effect
    if 3 in outer_top3_courses or 4 in outer_top3_courses:
        makuri_score += 1.0
    if 2 in outer_top3_courses:
        sashi_score += 1.0

    # 1号艇の配置決定
    boat1_bn = None
    for s in scores:
        if s['course'] == 1:
            boat1_bn = s['boat']['boat_number']
            break
    if boat1_bn is None:
        boat1_bn = 1

    # 最も展開スコアが高い非1号艇を1着に
    best_first = top_tenkai[0]['boat_number'] if top_tenkai else 2
    # 2着候補（1着・1号艇以外）
    seconds = [t['boat_number'] for t in top_tenkai if t['boat_number'] not in (best_first, boat1_bn)]

    if makuri_score >= sashi_score:
        # まくり: 1号艇3着
        second_bn = seconds[0] if seconds else (top_tenkai[1]['boat_number'] if len(top_tenkai) > 1 else 2)
        ura_1_combo = f"{best_first}-{second_bn}-{boat1_bn}"
    else:
        # 差し: 1号艇2着
        third_bn = seconds[0] if seconds else (top_tenkai[1]['boat_number'] if len(top_tenkai) > 1 else 3)
        ura_1_combo = f"{best_first}-{boat1_bn}-{third_bn}"

    ura_1 = [{'combo': ura_1_combo, 'type': '裏熊'}]

    return manzoku + ura_3 + ura_1


def calc_chaos(scores, boats, vp, wind_effect, nige_rate=None):
    reasons = []
    chaos = 0.0

    # イン逃げ率
    if nige_rate is not None:
        if nige_rate < 40:
            chaos += 3.0
            reasons.append(f"イン逃げ率が極めて低い({nige_rate:.1f}%)")
        elif nige_rate < 50:
            chaos += 1.5
            reasons.append(f"イン逃げ率が低い({nige_rate:.1f}%)")

    # スコア差：1位と2位の差が小さいほど荒れやすい
    if len(scores) >= 2:
        score_gap = scores[0]['score'] - scores[1]['score']
        if score_gap < 1.0:
            chaos += 2.5
            reasons.append("上位艇のスコア差が僅差")
        elif score_gap < 2.0:
            chaos += 1.0

    # 4〜6コースが上位スコアにいる
    outer_top = [s for s in scores[:3] if s['course'] >= 4]
    if len(outer_top) >= 2:
        chaos += 2.0
        reasons.append("外枠艇が上位スコアに複数")
    elif len(outer_top) == 1:
        chaos += 1.0
        reasons.append(f"{outer_top[0]['course']}コースが上位スコア")

    # 1コースのF持ち
    for b in boats:
        course = int(b.get('course', b.get('boat_number', 1)))
        if course == 1 and b.get('is_f', False):
            chaos += 2.5
            reasons.append("1コースにF持ち艇")
            break

    # 会場の荒れやすさ
    upset = vp.get('upset', 0.5)
    if upset >= 1.2:
        chaos += 2.0
        reasons.append("荒れやすい会場")
    elif upset >= 0.8:
        chaos += 1.0

    # 風の影響
    if wind_effect >= 0.8:
        chaos += 1.5
        reasons.append("風の影響が強い")
    elif wind_effect >= 0.4:
        chaos += 0.5

    # 最大10点でクランプ
    chaos = min(chaos, 10.0)
    level = 'low'
    label = '低め'
    if chaos >= 7:
        level = 'high'; label = '高い⚠️'
    elif chaos >= 4:
        level = 'mid'; label = 'やや高め'

    return {
        'score': round(chaos, 1),
        'level': level,
        'label': label,
        'reasons': reasons
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
    nige_rate = data.get('nige_rate', None)
    force_arekote = data.get('force_arekote', False)
    result = predict(boats, kimari, venue=venue, wind=wind, nige_rate=nige_rate, force_arekote=force_arekote)
    return jsonify(result)


@app.route('/save_record', methods=['POST'])
def save_record():
    data = request.get_json()
    conn = get_db()
    conn.execute('''
        INSERT INTO records (race_date, venue, race_no, predictions, created_at, nige_rate, wind)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        data.get('race_date', ''),
        data.get('venue', ''),
        data.get('race_no', 0),
        json.dumps(data.get('predictions', []), ensure_ascii=False),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        data.get('nige_rate'),
        data.get('wind')
    ))
    conn.commit()
    record_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
    conn.close()
    return jsonify({'success': True, 'id': record_id})


VENUE_CODES = {
    "桐生": "01", "戸田": "02", "江戸川": "03", "平和島": "04",
    "多摩川": "05", "浜名湖": "06", "蒲郡": "07", "常滑": "08",
    "津": "09", "三国": "10", "びわこ": "11", "住之江": "12",
    "尼崎": "13", "鳴門": "14", "丸亀": "15", "児島": "16",
    "宮島": "17", "徳山": "18", "下関": "19", "若松": "20",
    "芦屋": "21", "福岡": "22", "唐津": "23", "大村": "24",
}

@app.route('/debug_html', methods=['POST'])
def debug_html():
    """boatrace.jp から生HTML を取得して重要部分だけ返すデバッグ用エンドポイント"""
    data = request.get_json()
    venue = data.get('venue', '')
    race_no = data.get('race_no', 1)
    race_date = data.get('race_date', '')
    jcd = VENUE_CODES.get(venue)
    if not jcd:
        return jsonify({'error': '会場コード不明'})
    hd = race_date.replace('-', '')
    url = 'https://www.boatrace.jp/owpc/pc/race/raceresult?rno=' + str(race_no) + '&jcd=' + jcd + '&hd=' + hd
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'ja,en;q=0.5',
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return jsonify({'error': str(e), 'url': url})

    # boatColorが登場する前後50文字のスニペット
    snippets = []
    for m in re.finditer(r'boatColor', html):
        start = max(0, m.start() - 20)
        end = min(len(html), m.end() + 80)
        snippets.append(html[start:end].replace('\n', ' ').replace('\r', ''))
        if len(snippets) >= 10:
            break

    # HTMLをファイルに保存（直接確認用）
    save_path = os.path.join(os.path.dirname(__file__), 'debug_result.html')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html)

    # 3連単セクション（2000文字）
    trio_idx = html.find('3連単')
    trio_snippet = html[trio_idx:trio_idx+2000].replace('\n', ' ') if trio_idx >= 0 else '(3連単 not found)'

    return jsonify({
        'url': url,
        'html_length': len(html),
        'boatcolor_snippets': snippets,
        'trio_section': trio_snippet,
        'saved_to': save_path,
    })


@app.route('/fetch_result', methods=['POST'])
def fetch_result():
    data = request.get_json()
    venue = data.get('venue', '')
    race_no = data.get('race_no', 1)
    race_date = data.get('race_date', '')  # YYYY-MM-DD

    jcd = VENUE_CODES.get(venue)
    if not jcd:
        return jsonify({'success': False, 'error': '会場コードが見つかりません'})

    hd = race_date.replace('-', '')
    url = 'https://www.boatrace.jp/owpc/pc/race/raceresult?rno=' + str(race_no) + '&jcd=' + jcd + '&hd=' + hd

    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ja,en;q=0.5',
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return jsonify({'success': False, 'error': '取得失敗: ' + str(e), 'url': url})

    boats_found = []
    payout = 0

    # --- 着順取得 ---
    # 実際のHTML: <td class="is-fs14 is-fBold is-boatColor1">1</td>
    # boatColor の数字（色番号）ではなく td 内テキストが艇番
    m_boats = re.findall(r'boatColor\d+">\s*([1-6])\s*</td>', html)
    if len(m_boats) >= 3:
        boats_found = [int(x) for x in m_boats[:3]]

    # フォールバック: クラス末尾の " が省略されているケース
    if len(boats_found) < 3:
        m_boats2 = re.findall(r'boatColor\d+[^>]*>\s*([1-6])\s*</td>', html)
        if len(m_boats2) >= 3:
            boats_found = [int(x) for x in m_boats2[:3]]

    if len(boats_found) < 3:
        # 結果未確定か判断するヒントをHTMLから探す
        if 'boatColor' not in html:
            msg = 'レース結果がまだ公開されていません。終了後1〜3分待ってから再試行してください。'
        else:
            msg = '着順データが見つかりません。結果が反映されていないか、特殊なレース（中止・欠場等）の可能性があります。手動で入力してください。'
        return jsonify({'success': False, 'error': msg, 'url': url})

    r1, r2, r3 = boats_found[0], boats_found[1], boats_found[2]

    # --- 払戻金取得 ---
    # 実際のHTML構造: 3連単セクションにnumberSet1_number spanが並び、
    # その後の td に払戻金額が入っている
    # 例: <span class="numberSet1_number is-type1">1</span>...<span is-type6>6</span>...<span is-type2>2</span>
    # 金額は別の td に "2,340" 形式で入る

    # 実際の形式: <span class="is-payout1">&yen;420</span>
    # &yen; (¥記号のHTMLエンティティ) の後に金額が入っている
    m_payout = re.search(r'is-payout1[^>]*>\s*(?:&yen;|¥|￥)\s*([\d,]+)', html)
    if m_payout:
        try:
            payout = int(m_payout.group(1).replace(',', ''))
        except Exception:
            pass

    # フォールバック: &yen; を含む数字を3連単セクションから取得
    if payout == 0:
        for keyword in ['3連単', '三連単']:
            idx = html.find(keyword)
            if idx >= 0:
                area = html[idx:idx+2000]
                for a in re.findall(r'(?:&yen;|¥|￥)\s*([\d,]+)', area):
                    try:
                        val = int(a.replace(',', ''))
                        if val >= 100:
                            payout = val
                            break
                    except Exception:
                        pass
                if payout:
                    break

    # 最終フォールバック: is-payout / is-pay クラスから取得
    if payout == 0:
        for pat in [r'is-payout\d*[^>]*>\s*(?:&yen;|¥|￥)?\s*([\d,]+)', r'is-pay[^>]*>\s*(?:&yen;|¥|￥)?\s*([\d,]+)']:
            pm = re.search(pat, html)
            if pm:
                try:
                    val = int(pm.group(1).replace(',', ''))
                    if val >= 100:
                        payout = val
                        break
                except Exception:
                    pass

    return jsonify({
        'success': True,
        'r1': r1, 'r2': r2, 'r3': r3,
        'payout': payout,
        'payout_debug': payout_debug if payout == 0 else '',
        'url': url
    })


@app.route('/update_result', methods=['POST'])
def update_result():
    data = request.get_json()
    record_id = data.get('id')
    result_1st = data.get('result_1st')
    result_2nd = data.get('result_2nd')
    result_3rd = data.get('result_3rd')
    payout = data.get('payout', 0)
    purchase = data.get('purchase', 100)

    conn = get_db()
    row = conn.execute('SELECT predictions FROM records WHERE id=?', (record_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({'success': False, 'error': 'Not found'})

    predictions = json.loads(row['predictions'])
    result_combo = f"{result_1st}-{result_2nd}-{result_3rd}"
    is_hit = 1 if any(p['combo'] == result_combo for p in predictions) else 0

    conn.execute('''
        UPDATE records SET result_1st=?, result_2nd=?, result_3rd=?,
        payout=?, purchase=?, is_hit=? WHERE id=?
    ''', (result_1st, result_2nd, result_3rd, payout, purchase, is_hit, record_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'is_hit': is_hit})


@app.route('/get_records')
def get_records():
    period = request.args.get('period', 'all')
    conn = get_db()
    if period == 'week':
        rows = conn.execute("SELECT * FROM records WHERE race_date >= date('now', '-7 days') ORDER BY id DESC").fetchall()
    elif period == 'month':
        rows = conn.execute("SELECT * FROM records WHERE race_date >= date('now', '-30 days') ORDER BY id DESC").fetchall()
    else:
        rows = conn.execute("SELECT * FROM records ORDER BY id DESC").fetchall()
    conn.close()

    records = []
    for r in rows:
        records.append({
            'id': r['id'],
            'race_date': r['race_date'],
            'venue': r['venue'],
            'race_no': r['race_no'],
            'predictions': json.loads(r['predictions']),
            'result_1st': r['result_1st'],
            'result_2nd': r['result_2nd'],
            'result_3rd': r['result_3rd'],
            'payout': r['payout'],
            'purchase': r['purchase'],
            'is_hit': r['is_hit'],
            'created_at': r['created_at'],
            'nige_rate': r['nige_rate'],
            'wind': r['wind'],
        })
    return jsonify(records)


@app.route('/stats_summary', methods=['POST'])
def stats_summary():
    """会場×イン逃げ率帯の過去データをSQLiteで集計して返す"""
    data = request.get_json()
    venue = data.get('venue', '')
    nige_rate = data.get('nige_rate', None)

    conn = get_db()

    # 会場フィルタ
    if not venue:
        conn.close()
        return jsonify({'error': '会場未指定'})

    rows = conn.execute(
        "SELECT predictions, is_hit, payout, purchase, nige_rate, wind, result_1st, result_2nd, result_3rd, race_date FROM records WHERE venue=? AND purchase IS NOT NULL",
        (venue,)
    ).fetchall()
    conn.close()

    # イン逃げ率帯フィルタ（5%単位切り捨てを下限にする）
    if nige_rate is not None:
        import math
        nige_min = math.floor(nige_rate / 5) * 5
        filtered = [r for r in rows if r['nige_rate'] is not None and r['nige_rate'] >= nige_min]
    else:
        nige_min = None
        filtered = list(rows)

    total = len(filtered)
    if total == 0:
        return jsonify({'total': 0, 'venue': venue, 'nige_rate': nige_rate, 'nige_min': nige_min})

    # 集計
    hits = sum(1 for r in filtered if r['is_hit'])
    total_purchase = sum(r['purchase'] or 0 for r in filtered)
    total_payout = sum((r['payout'] or 0) for r in filtered if r['is_hit'])
    hit_rate = round(hits / total * 100, 1)
    recovery = round(total_payout / total_purchase * 100, 1) if total_purchase > 0 else 0

    # 予想タイプ別集計（レース単位：そのレースでそのタイプが1点でも当たったか）
    type_stats = {}
    for r in filtered:
        try:
            preds = json.loads(r['predictions'])
        except Exception:
            continue
        has_result = r['result_1st'] and r['result_2nd'] and r['result_3rd']
        result_combo = f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}" if has_result else None
        # このレースに含まれるタイプを収集
        types_in_race = {}
        for p in preds:
            t = p.get('type', '')
            if not t:
                continue
            if t not in types_in_race:
                types_in_race[t] = False
            if result_combo and p.get('combo') == result_combo:
                types_in_race[t] = True
        # レース単位でカウント
        for t, hit in types_in_race.items():
            if t not in type_stats:
                type_stats[t] = {'count': 0, 'hit': 0}
            type_stats[t]['count'] += 1
            if hit:
                type_stats[t]['hit'] += 1

    type_hit_rates = {}
    for t, s in type_stats.items():
        if s['count'] > 0:
            type_hit_rates[t] = {
                'count': s['count'],
                'hit': s['hit'],
                'rate': round(s['hit'] / s['count'] * 100, 1)
            }

    # 風向き別集計（is_hitのレースのみカウント）
    wind_stats = {}
    for r in filtered:
        w = r['wind'] if 'wind' in r.keys() else None
        if not w:
            w = '無風'
        if w not in wind_stats:
            wind_stats[w] = {'count': 0, 'hit': 0}
        wind_stats[w]['count'] += 1
        if r['is_hit']:
            wind_stats[w]['hit'] += 1

    # 風向き表示順（追い風→向かい風→横風→無風）
    WIND_ORDER = [
        '追い風1m', '追い風2m', '追い風3m以上',
        '向かい風1m', '向かい風2m', '向かい風3m以上',
        '横風（左）1m', '横風（左）2m', '横風（左）3m以上',
        '横風（右）1m', '横風（右）2m', '横風（右）3m以上',
        '無風', 'その他',
    ]
    wind_hit_rates = []
    seen_winds = set()
    for w in WIND_ORDER:
        if w in wind_stats:
            s = wind_stats[w]
            wind_hit_rates.append({
                'wind': w,
                'count': s['count'],
                'hit': s['hit'],
                'rate': round(s['hit'] / s['count'] * 100, 1) if s['count'] > 0 else 0,
            })
            seen_winds.add(w)
    # WIND_ORDERにないものも末尾に追加
    for w, s in wind_stats.items():
        if w not in seen_winds:
            wind_hit_rates.append({
                'wind': w,
                'count': s['count'],
                'hit': s['hit'],
                'rate': round(s['hit'] / s['count'] * 100, 1) if s['count'] > 0 else 0,
            })

    # 平均収支（1レースあたり）
    avg_profit = round((total_payout - total_purchase) / total, 0) if total > 0 else 0

    # 的中した時の平均払戻
    hit_records = [r for r in filtered if r['is_hit'] and r['payout']]
    avg_payout_on_hit = round(sum(r['payout'] for r in hit_records) / len(hit_records), 0) if hit_records else 0

    # 直近5戦の結果（新しい順）
    recent5 = []
    sorted_filtered = sorted(filtered, key=lambda r: r['race_date'] if 'race_date' in r.keys() else '', reverse=True)
    for r in sorted_filtered[:5]:
        recent5.append('hit' if r['is_hit'] else 'miss')

    # 最後に的中した日付
    last_hit_date = None
    for r in sorted_filtered:
        if r['is_hit']:
            last_hit_date = r['race_date'] if 'race_date' in r.keys() else None
            break

    # 1コース以外が1着になった率（荒れ率）
    upset_count = 0
    result_total = 0
    for r in filtered:
        if r['result_1st']:
            result_total += 1
            if str(r['result_1st']) != '1':
                upset_count += 1
    upset_rate = round(upset_count / result_total * 100, 1) if result_total > 0 else None

    # 推奨度判定
    if total < 5:
        level = 'unknown'
    elif recovery >= 100 and hit_rate >= 25:
        level = 'hot'
    elif recovery >= 90 and hit_rate >= 20:
        level = 'watch'
    else:
        level = 'pass'

    return jsonify({
        'venue': venue,
        'nige_rate': nige_rate,
        'nige_min': nige_min,
        'total': total,
        'hits': hits,
        'hit_rate': hit_rate,
        'recovery': recovery,
        'avg_profit': avg_profit,
        'avg_payout_on_hit': avg_payout_on_hit,
        'recent5': recent5,
        'last_hit_date': last_hit_date,
        'upset_rate': upset_rate,
        'type_hit_rates': type_hit_rates,
        'wind_hit_rates': wind_hit_rates,
        'level': level,
    })


@app.route('/payout_stats', methods=['GET'])
def payout_stats():
    """種別（本命/対抗/中穴/万舟）ごとの的中時平均配当を集計"""
    conn = get_db()
    rows = conn.execute(
        "SELECT predictions, result_1st, result_2nd, result_3rd, payout FROM records WHERE payout IS NOT NULL AND result_1st IS NOT NULL"
    ).fetchall()
    conn.close()

    TYPES = ['本命', '対抗', '中穴', '万舟']
    stats = {t: {'hits': 0, 'total': 0, 'payouts': []} for t in TYPES}

    for row in rows:
        try:
            preds = json.loads(row['predictions'])
        except Exception:
            continue
        result = f"{row['result_1st']}-{row['result_2nd']}-{row['result_3rd']}"
        # レース単位で集計（種別ごとに1レース=1カウント）
        types_in_race = {}
        for p in preds:
            t = p.get('type')
            if t not in TYPES:
                continue
            if t not in types_in_race:
                types_in_race[t] = False
            if p.get('combo') == result:
                types_in_race[t] = True
        for t, is_hit in types_in_race.items():
            stats[t]['total'] += 1
            if is_hit:
                stats[t]['hits'] += 1
                if row['payout'] and row['payout'] > 0:
                    stats[t]['payouts'].append(row['payout'])

    result_data = {}
    for t in TYPES:
        s = stats[t]
        avg_pay = round(sum(s['payouts']) / len(s['payouts'])) if s['payouts'] else None
        hit_rate = round(s['hits'] / s['total'] * 100, 1) if s['total'] > 0 else 0
        result_data[t] = {
            'hits': s['hits'],
            'total': s['total'],
            'hit_rate': hit_rate,
            'avg_payout': avg_pay,
        }

    return jsonify(result_data)


@app.route('/history_stats', methods=['POST'])
def history_stats():
    """絞り込み条件をサーバーで処理して集計結果を返す"""
    data = request.get_json()
    period     = data.get('period', 'all')
    venues     = data.get('venues', [])
    nige_min   = data.get('nige_min', 0)
    nige_max   = data.get('nige_max', 100)
    manzoku    = data.get('manzoku', False)

    conn = get_db()
    if period == 'week':
        rows = conn.execute("SELECT * FROM records WHERE race_date >= date('now', '-7 days') ORDER BY id DESC").fetchall()
    elif period == 'month':
        rows = conn.execute("SELECT * FROM records WHERE race_date >= date('now', '-30 days') ORDER BY id DESC").fetchall()
    else:
        rows = conn.execute("SELECT * FROM records ORDER BY id DESC").fetchall()
    conn.close()

    # フィルター適用
    filtered = []
    for r in rows:
        if venues and r['venue'] not in venues:
            continue
        if nige_min > 0 or nige_max < 100:
            if r['nige_rate'] is None:
                continue
            if r['nige_rate'] < nige_min or r['nige_rate'] > nige_max:
                continue
        if manzoku and (not r['payout'] or r['payout'] < 10000):
            continue
        filtered.append(r)

    total = len(filtered)
    with_result = [r for r in filtered if r['result_1st']]
    hits = [r for r in with_result if r['is_hit']]
    hit_rate = round(len(hits) / len(with_result) * 100, 1) if with_result else 0
    total_payout = sum(r['payout'] or 0 for r in hits)
    total_purchase = sum(r['purchase'] or 0 for r in with_result if r['purchase'])
    recovery = round(total_payout / total_purchase * 100, 1) if total_purchase > 0 else 0
    profit = total_payout - total_purchase

    # 種別別集計
    TYPES = ['本命', '対抗', '中穴', '万舟']
    TYPE_BY_INDEX = ['本命','本命','対抗','対抗','対抗','万舟','万舟']
    type_stats = {t: {'hits': 0, 'total': 0, 'payouts': []} for t in TYPES}
    for r in with_result:
        try:
            preds = json.loads(r['predictions'])
        except Exception:
            continue
        result_combo = f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}"
        types_in_race = {}
        for idx, p in enumerate(preds):
            t = p.get('type') or (TYPE_BY_INDEX[idx] if idx < len(TYPE_BY_INDEX) else '')
            if t not in TYPES:
                continue
            if t not in types_in_race:
                types_in_race[t] = False
            if p.get('combo') == result_combo:
                types_in_race[t] = True
        for t, is_hit in types_in_race.items():
            type_stats[t]['total'] += 1
            if is_hit:
                type_stats[t]['hits'] += 1
                if r['payout'] and r['payout'] > 0:
                    type_stats[t]['payouts'].append(r['payout'])

    pattern = {}
    for t in TYPES:
        s = type_stats[t]
        avg_pay = round(sum(s['payouts']) / len(s['payouts'])) if s['payouts'] else None
        hr = round(s['hits'] / s['total'] * 100, 1) if s['total'] > 0 else 0
        breakeven = round(avg_pay * s['hits'] / s['total']) if avg_pay and s['total'] > 0 else None
        pattern[t] = {'hits': s['hits'], 'total': s['total'], 'hit_rate': hr,
                      'avg_payout': avg_pay, 'breakeven': breakeven}

    # 本命の外れ方内訳（1着は合っていたか？）
    miss_breakdown = {'hit': 0, 'first_second': 0, 'first_only': 0, 'first_wrong': 0, 'total': 0}
    for r in with_result:
        try:
            preds = json.loads(r['predictions'])
        except Exception:
            continue
        honmei_combos = []
        for idx, p in enumerate(preds):
            t = p.get('type') or (TYPE_BY_INDEX[idx] if idx < len(TYPE_BY_INDEX) else '')
            if t == '本命' and p.get('combo'):
                honmei_combos.append(p['combo'])
        if not honmei_combos:
            continue
        miss_breakdown['total'] += 1
        result_combo = f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}"
        if result_combo in honmei_combos:
            miss_breakdown['hit'] += 1
            continue
        first_match = any(c.split('-')[0] == str(r['result_1st']) for c in honmei_combos)
        second_match = any(c.split('-')[0] == str(r['result_1st']) and c.split('-')[1] == str(r['result_2nd']) for c in honmei_combos)
        if second_match:
            miss_breakdown['first_second'] += 1
        elif first_match:
            miss_breakdown['first_only'] += 1
        else:
            miss_breakdown['first_wrong'] += 1

    # 会場別集計
    venue_stats = {}
    for r in with_result:
        v = r['venue']
        if v not in venue_stats:
            venue_stats[v] = {'hits': 0, 'total': 0, 'payout': 0, 'purchase': 0}
        venue_stats[v]['total'] += 1
        if r['is_hit']:
            venue_stats[v]['hits'] += 1
            venue_stats[v]['payout'] += r['payout'] or 0
        venue_stats[v]['purchase'] += r['purchase'] or 0

    venue_list = []
    for v, s in venue_stats.items():
        hr = round(s['hits'] / s['total'] * 100, 1) if s['total'] > 0 else 0
        rec = round(s['payout'] / s['purchase'] * 100, 1) if s['purchase'] > 0 else 0
        venue_list.append({'venue': v, 'hits': s['hits'], 'total': s['total'],
                           'hit_rate': hr, 'recovery': rec})
    venue_list.sort(key=lambda x: x['hit_rate'], reverse=True)

    # 収支推移（日別）
    daily = {}
    for r in sorted(with_result, key=lambda x: x['race_date']):
        d = r['race_date']
        if d not in daily:
            daily[d] = {'payout': 0, 'purchase': 0}
        daily[d]['payout'] += r['payout'] or 0
        daily[d]['purchase'] += r['purchase'] or 0
    profit_series = []
    cumulative = 0
    for d, s in sorted(daily.items()):
        cumulative += s['payout'] - s['purchase']
        profit_series.append({'date': d, 'cumulative': cumulative})

    return jsonify({
        'total': total,
        'with_result': len(with_result),
        'hits': len(hits),
        'hit_rate': hit_rate,
        'recovery': recovery,
        'profit': profit,
        'total_payout': total_payout,
        'total_purchase': total_purchase,
        'pattern': pattern,
        'miss_breakdown': miss_breakdown,
        'venue_list': venue_list,
        'profit_series': profit_series,
    })


@app.route('/delete_record', methods=['POST'])
def delete_record():
    data = request.get_json()
    conn = get_db()
    conn.execute('DELETE FROM records WHERE id=?', (data.get('id'),))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/history')
def history():
    return render_template('history.html')


if __name__ == '__main__':
    app.run(debug=True, port=5001)
